#include <unordered_map>
#include <memory>
#include <atomic>
#include <mutex>
#include <chrono>
#include <random>
#include <sstream>
#include "whisper-server.h"
#include "client-session.h"
#include "whisper.h"

namespace {
    ServerParams params;
    std::unordered_map<std::string, std::unique_ptr<ClientSession>> clients;
    std::mutex clients_mtx;
    std::thread processor_thread;
    std::atomic<bool> running{true};
    std::mutex g_ctx_mtx;
    whisper_context* g_ctx = nullptr;
    constexpr int CHUNK_SIZE = 3 * 16000;
}

std::string generate_uuid_v4() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    ss << std::hex;

    for (int i = 0; i < 8; i++) ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 4; i++) ss << dis(gen);
    ss << "-4";  // v4
    for (int i = 0; i < 3; i++) ss << dis(gen);
    ss << "-";
    ss << dis2(gen);
    for (int i = 0; i < 3; i++) ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 12; i++) ss << dis(gen);

    return ss.str();
}

void handleMessage(std::shared_ptr<ix::ConnectionState> state,
                  ix::WebSocket& ws,
                  const ix::WebSocketMessagePtr& msg) {
    const std::string client_id = state->getId();

    if (msg->type == ix::WebSocketMessageType::Open) {
        printf("[%s] new client\n", client_id.c_str());
        std::lock_guard<std::mutex> lock(clients_mtx);
        clients[client_id] = std::make_unique<ClientSession>();
        // UUID v4
        clients[client_id]->buffToBackend.connection_id = generate_uuid_v4();
        ws.sendText("CONNECTION_ID:" + clients[client_id]->buffToBackend.connection_id);

        clients[client_id]->connection = &ws;
    }
    else if (msg->type == ix::WebSocketMessageType::Close) {
        printf("[%s] delete client\n", client_id.c_str());
        clients[client_id]->buffToBackend.flush();
        std::lock_guard<std::mutex> lock(clients_mtx);
        if (clients.count(client_id)) {
            clients[client_id]->active = false;
            clients.erase(client_id);
        }
    }
    else if (msg->type == ix::WebSocketMessageType::Message && msg->binary) {
        std::lock_guard<std::mutex> lock(clients_mtx);
        if (!clients.count(client_id)) return;

        auto& session = *clients[client_id];
        const auto& data = msg->str;
        #ifdef CONVERT_FROM_PCM_16
            const int16_t* pcm16 = reinterpret_cast<const int16_t*>(data.data());
            size_t n_samples = data.size() / sizeof(int16_t);

            std::lock_guard<std::mutex> session_lock(session.mtx);
            for (size_t i = 0; i < n_samples; i++) {
                session.pcm_buffer.push_back(pcm16[i] / 32768.0f);
            }
        #else
            const int32_t* pcm32 = reinterpret_cast<const int32_t*>(data.data());
            //also we may use memcpy ))
            size_t n_samples = data.size() / sizeof(int32_t);

            std::lock_guard<std::mutex> session_lock(session.mtx);
            for (size_t i = 0; i < n_samples; i++) {
                session.pcm_buffer.push_back(pcm32[i]);
            }
        #endif
    }
}

void processChunk(std::vector<float> &chunk, const std::string &id, ClientSession *session) {
    std::lock_guard<std::mutex> ctx_lock(g_ctx_mtx);
    whisper_full_params wparams = whisper_full_default_params(
        params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH
                            : WHISPER_SAMPLING_GREEDY);

    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate       = params.translate;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;
    wparams.beam_search.beam_size = params.beam_size;
    wparams.audio_ctx        = params.audio_ctx;
    wparams.tdrz_enable      = params.tinydiarize;

    if (whisper_full(g_ctx, wparams, chunk.data(), chunk.size()) == 0) {
        const char* text = whisper_full_get_segment_text(g_ctx, 0);
        printf("[%s] %s\n", id.c_str(), text);
        session->connection->sendText(text);
        session->buffToBackend.add_message(text);
    }
    whisper_reset_timings(g_ctx);
}

void process() {
    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::lock_guard<std::mutex> lock(clients_mtx);
        for (auto& [id, session] : clients) {
            std::lock_guard<std::mutex> session_lock(session->mtx);
            if (session->pcm_buffer.size() < CHUNK_SIZE) continue;

            std::vector<float> chunk(
                session->pcm_buffer.begin(),
                session->pcm_buffer.begin() + CHUNK_SIZE
            );
            session->pcm_buffer.erase(
                session->pcm_buffer.begin(),
                session->pcm_buffer.begin() + CHUNK_SIZE
            );

            processChunk(chunk, id, session.get());
        }
    }
}

WhisperServer::WhisperServer(const ServerParams& _params) : server(params.port, params.host) {
    params = _params;

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    g_ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    server.setTLSOptions({});
    server.setOnClientMessageCallback([this](auto&&... args) {
        handleMessage(args...);
    });

    processor_thread = std::thread([this] { process(); });
}

WhisperServer::~WhisperServer() {
    running = false;
    server.stop();
    if (processor_thread.joinable()) processor_thread.join();
    std::lock_guard<std::mutex> lock(clients_mtx);
    for (auto& [id, session] : clients) {
        session->buffToBackend.flush();
    }
    whisper_free(g_ctx);
}

void WhisperServer::run() {
    server.listenAndStart();
    while (running) std::this_thread::sleep_for(std::chrono::seconds(1));
}
