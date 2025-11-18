#include "common.h"
#include "whisper.h"

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/signal_set.hpp> 
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>
#include <atomic>
#include <fstream>
#include <iomanip> 

// Define namespaces
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = boost::beast::websocket; 
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

// Define constants
constexpr int INPUT_SAMPLE_RATE = 16000; // Expected sample rate from client
// --- RE-ADD THESE CONSTANTS ---
constexpr int MAX_BUFFER_SECONDS = 45; // Max audio buffer size (seconds)
constexpr int MAX_AUDIO_SAMPLES = INPUT_SAMPLE_RATE * MAX_BUFFER_SECONDS;

// Command-line parameters struct
struct whisper_params {
    int32_t n_threads = std::max(1, (int32_t) std::thread::hardware_concurrency() / 2);
    int32_t port = 8080;
    std::string model_path = "models/ggml-base.en.bin";
    std::string language = "en";
    bool translate = false;
    bool use_gpu = true;
    bool print_timestamps = false; 

    // --- REVERTED --- (Removed keep_ms)
    int32_t step_ms = 400;   // How often to process audio
    int32_t max_tokens = 0;   
    int32_t beam_size = -1;   
    float temperature = 0.0f; 
    bool no_fallback = false;
};

// Forward declarations
void whisper_print_usage(int argc, char ** argv, const whisper_params & params);
bool whisper_params_parse(int argc, char ** argv, whisper_params & params);
void fail(beast::error_code ec, char const* what, bool is_error = true);

// Per-session state
class session : public std::enable_shared_from_this<session> {
    websocket::stream<beast::tcp_stream> ws_;
    beast::flat_buffer buffer_;        // Buffer for incoming WebSocket frames
    whisper_context* ctx_ = nullptr;     // Whisper context FOR THIS SESSION ONLY
    whisper_params app_params_;        // Copy of app params
    std::atomic<bool> processing_{false}; // Ensure only one inference runs at a time

    // --- REVERTED ---
    // Back to your original (accurate) buffering logic
    std::vector<float> pcmf32_new_;       // Buffer for newly received audio samples
    std::vector<float> pcmf32_processed_; // Buffer passed to whisper_full (grows)
    int n_samples_step_;                  // Samples per processing step

public:
    // Takes ownership of the socket and app params
    explicit session(tcp::socket&& socket, const whisper_params& app_params)
        : ws_(std::move(socket)), app_params_(app_params)
    {
        std::cerr << "[" << std::this_thread::get_id() << "] Session constructor entered." << std::endl;

        // --- REVERTED ---
        n_samples_step_ = (1e-3 * app_params_.step_ms) * INPUT_SAMPLE_RATE;
        std::cerr << "[" << std::this_thread::get_id() << "] step_ms: " << app_params_.step_ms << " (" << n_samples_step_ << " samples)" << std::endl;

        // --- Initialize Whisper Context (Per Session) ---
        struct whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = app_params_.use_gpu; 

        ctx_ = whisper_init_from_file_with_params(app_params_.model_path.c_str(), cparams);

        if (!ctx_) {
            std::cerr << "[" << std::this_thread::get_id() << "] Error: Failed to initialize whisper context from model: " << app_params_.model_path << std::endl;
        } else {
            std::cerr << "[" << std::this_thread::get_id() << "] Session context initialized successfully." << std::endl;

            // --- KEPT THE GPU WARM-UP ---
            std::cerr << "[" << std::this_thread::get_id() << "] Warming up GPU..." << std::endl;
            auto t_start = std::chrono::high_resolution_clock::now();
            
            std::vector<float> pcmf32_warmup(INPUT_SAMPLE_RATE, 0.0f); // 1 sec silence
            whisper_full_params wparams_warmup = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            wparams_warmup.print_progress = false;
            wparams_warmup.print_realtime = false;
            wparams_warmup.language = app_params_.language.c_str(); 
            wparams_warmup.n_threads = app_params_.n_threads;

            whisper_full(ctx_, wparams_warmup, pcmf32_warmup.data(), pcmf32_warmup.size());
            
            auto t_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration_ms = t_end - t_start;
            std::cerr << "[" << std::this_thread::get_id() << "] GPU warm-up completed in " << duration_ms.count() << " ms." << std::endl;
            // --- END WARM-UP ---
        }
    }

    ~session() {
        if (ctx_) {
            whisper_free(ctx_);
            ctx_ = nullptr;
            std::cerr << "[" << std::this_thread::get_id() << "] Session context freed." << std::endl;
        } else {
            std::cerr << "[" << std::this_thread::get_id() << "] Session ending (context was null or already freed)." << std::endl;
        }
    }

    // Start the session
    void run() {
        if (!ctx_) {
            std::cerr << "[" << std::this_thread::get_id() << "] Session::run() aborted: context initialization failed." << std::endl;
            beast::error_code ec;
            ws_.close(websocket::close_code::internal_error, ec);
            return;
        }
         std::cerr << "[" << std::this_thread::get_id() << "] Session::run() called." << std::endl;
        net::post(ws_.get_executor(),
                    beast::bind_front_handler(&session::on_run, shared_from_this()));
    }

    void on_run() {
         std::cerr << "[" << std::this_thread::get_id() << "] Session::on_run() entered. Setting options and starting accept." << std::endl;
        ws_.set_option(websocket::stream_base::timeout::suggested(beast::role_type::server));
        ws_.async_accept(
            beast::bind_front_handler(&session::on_accept, shared_from_this()));
    }

    void on_accept(beast::error_code ec) {
         std::cerr << "[" << std::this_thread::get_id() << "] Session::on_accept() entered. Error code: " << ec.message() << std::endl;
        if (ec)
            return fail(ec, "accept");
        do_read();
    }

    void do_read() {
         std::cerr << "[" << std::this_thread::get_id() << "] Session::do_read() called. Initiating async_read..." << std::endl;
        ws_.async_read(
            buffer_,
            beast::bind_front_handler(&session::on_read, shared_from_this()));
    }

    // --- REVERTED ---
    void on_read(beast::error_code ec, std::size_t bytes_transferred) {
         std::cerr << "[" << std::this_thread::get_id() << "] Session::on_read() entered. Error code: " << ec.message() << ", Bytes: " << bytes_transferred << std::endl;
        boost::ignore_unused(bytes_transferred);

        // Handle closure or errors
        if (ec == websocket::error::closed || ec == net::error::eof) {
            std::cerr << "[" << std::this_thread::get_id() << "] WebSocket closed by client." << std::endl;
            if (!pcmf32_new_.empty() || !pcmf32_processed_.empty()) {
                std::cerr << "[" << std::this_thread::get_id() << "] Processing remaining audio before closing..." << std::endl;
                // Move remaining new audio to processed buffer
                pcmf32_processed_.insert(pcmf32_processed_.end(), pcmf32_new_.begin(), pcmf32_new_.end());
                pcmf32_new_.clear();
                process_audio(true); // Mark as final
            }
            return; // Stop reading
        }
        if (ec) return fail(ec, "read");

        bool is_final_chunk = false;

        if (ws_.got_binary()) {
            if (bytes_transferred == 0) {
                 std::cerr << "[" << std::this_thread::get_id() << "] Received empty binary frame (end of stream signal)." << std::endl;
                is_final_chunk = true;
            } else {
                auto data = buffer_.data();
                const int16_t* pcm16 = static_cast<const int16_t*>(data.data());
                std::size_t n_samples = data.size() / sizeof(int16_t);

                pcmf32_new_.reserve(pcmf32_new_.size() + n_samples);
                for (size_t i = 0; i < n_samples; ++i) {
                    pcmf32_new_.push_back(static_cast<float>(pcm16[i]) / 32768.0f);
                }
                 std::cerr << "[" << std::this_thread::get_id() << "] Added " << n_samples << " samples to pcmf32_new_. New total: " << pcmf32_new_.size() << std::endl;
            }

            // --- REVERTED --- Trigger processing logic from your original code
            bool expect_processing = false;
            if (!processing_.load() && (!pcmf32_new_.empty() || !pcmf32_processed_.empty()) && (pcmf32_new_.size() >= n_samples_step_ || is_final_chunk)) {
                expect_processing = true;
                processing_.store(true); // Mark as processing

                // Move new audio to the buffer that whisper_full uses
                pcmf32_processed_.insert(pcmf32_processed_.end(), pcmf32_new_.begin(), pcmf32_new_.end());
                pcmf32_new_.clear();

                // Trim pcmf32_processed_ if it exceeds max size (relative to Whisper's 30s window)
                if (pcmf32_processed_.size() > MAX_AUDIO_SAMPLES) {
                    int excess_samples = pcmf32_processed_.size() - MAX_AUDIO_SAMPLES;
                    std::cerr << "[" << std::this_thread::get_id() << "] Trimming " << excess_samples << " old samples from processed buffer." << std::endl;
                    pcmf32_processed_.erase(pcmf32_processed_.begin(), pcmf32_processed_.begin() + excess_samples);
                }

                // Process asynchronously
                auto self = shared_from_this();
                net::post(ws_.get_executor(), [self, is_final_chunk]() {
                    self->process_audio(is_final_chunk);
                    self->processing_.store(false); // Mark processing finished
                });

            } else if (processing_.load()) {
                 std::cerr << "[" << std::this_thread::get_id() << "] Skipping process trigger: Already processing." << std::endl;
            } else {
                 std::cerr << "[" << std::this_thread::get_id() << "] Skipping process trigger: Not enough new data (" << pcmf32_new_.size() << "/" << n_samples_step_ << ") and not final chunk." << std::endl;
            }


        } else if (ws_.got_text()) {
             std::string received_text = beast::buffers_to_string(buffer_.data());
             std::cerr << "[" << std::this_thread::get_id() << "] Warning: Received text frame, content: " << received_text << ". Expected binary PCM audio." << std::endl;
        }

        buffer_.consume(buffer_.size()); // Clear the read buffer

        if (!is_final_chunk && ws_.is_open()) {
             do_read();
        } else if (is_final_chunk) {
             std::cerr << "[" << std::this_thread::get_id() << "] End of stream signal received, stopping read loop." << std::endl;
        }
    }

    // --- REVERTED ---
    void process_audio(bool is_final) {
        if (!ctx_) {
            std::cerr << "[" << std::this_thread::get_id() << "] Error: Whisper context is null during processing." << std::endl;
            return;
        }
        if (pcmf32_processed_.empty()) {
            std::cerr << "[" << std::this_thread::get_id() << "] Warning: process_audio called with empty buffer (non-final)." << std::endl;
            return;
        }

        // --- Prepare Whisper Parameters (Full buffer + new params) ---
        whisper_full_params wparams = whisper_full_default_params(
            app_params_.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY
        );

        wparams.print_progress = false;
        wparams.print_special = false;
        wparams.print_realtime = false;
        wparams.print_timestamps = app_params_.print_timestamps;
        wparams.translate = app_params_.translate;
        wparams.language = app_params_.language.c_str();
        wparams.n_threads = app_params_.n_threads;
        wparams.audio_ctx = 0; 
        wparams.max_tokens = app_params_.max_tokens;
        
        wparams.temperature = app_params_.temperature;
        wparams.temperature_inc = app_params_.no_fallback ? 0.0f : wparams.temperature_inc;
        wparams.beam_search.beam_size = app_params_.beam_size;
        
        std::cerr << "[" << std::this_thread::get_id() << "] Processing " << pcmf32_processed_.size() << " total audio samples... (is_final=" << is_final << ")" << std::endl;

        // --- Run Inference ---
        auto t_start = std::chrono::high_resolution_clock::now();
        int ret = whisper_full(ctx_, wparams, pcmf32_processed_.data(), pcmf32_processed_.size());
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms = t_end - t_start;

        if (ret != 0) {
            fprintf(stderr, "[%s] %s: whisper_full failed, return code = %d\n", std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())).c_str(), __func__, ret);
            pcmf32_processed_.clear(); // Clear buffer on error
            json error_json; error_json["error"] = "Whisper processing failed"; error_json["code"] = ret;
            send_json(error_json);
            return;
        }
         std::cerr << "[" << std::this_thread::get_id() << "] whisper_full completed successfully in " << duration_ms.count() << " ms." << std::endl;

        // --- Extract Results ---
        const int n_segments = whisper_full_n_segments(ctx_);
        std::string current_transcription = "";
        json response_json;
         std::cerr << "[" << std::this_thread::get_id() << "] Found " << n_segments << " segments." << std::endl;

        for (int i = 0; i < n_segments; ++i) {
            const char* text = whisper_full_get_segment_text(ctx_, i);
            if (text) {
                current_transcription += text;
                 std::cerr << "[" << std::this_thread::get_id() << "] Segment " << i << ": " << text << std::endl;
            } else {
                 std::cerr << "[" << std::this_thread::get_id() << "] Warning: Got null text for segment " << i << std::endl;
            }
        }
        
        // --- Send Result Back ---
        if (!current_transcription.empty() || is_final) {
            response_json["text"] = current_transcription;
            response_json["is_final"] = is_final;
            send_json(response_json);
        } else {
             std::cerr << "[" << std::this_thread::get_id() << "] No transcription text generated for this interval (non-final)." << std::endl;
        }
        // (We do not clear pcmf32_processed_ here)
    }

    // Helper to send JSON response asynchronously
    void send_json(const json& j) {
        if (!ws_.is_open()) {
             std::cerr << "[" << std::this_thread::get_id() << "] Send JSON failed: WebSocket is not open." << std::endl;
            return;
        }
        std::string response_string = j.dump();
         std::cerr << "[" << std::this_thread::get_id() << "] Sending JSON: " << response_string << std::endl;
        ws_.text(true); // Ensure text mode

        auto self = shared_from_this();
        net::post(ws_.get_executor(), [self, response_string]() {
            if (!self->ws_.is_open()) { std::cerr << "[" << std::this_thread::get_id() << "] Write aborted: WebSocket is not open." << std::endl; return; }
             std::cerr << "[" << std::this_thread::get_id() << "] Initiating async_write..." << std::endl;
            self->ws_.async_write(
                net::buffer(response_string),
                [self](beast::error_code ec, std::size_t bytes_transferred) {
                     std::cerr << "[" << std::this_thread::get_id() << "] async_write completed. Error code: " << ec.message() << ", Bytes: " << bytes_transferred << std::endl;
                    if (ec && ec != net::error::operation_aborted && ec != websocket::error::closed) {
                        fail(ec, "write");
                    }
                });
        });
    }

}; // End class session

//------------------------------------------------------------------------------
// (Listener class is unchanged)
// ...
class listener : public std::enable_shared_from_this<listener> {
    net::io_context& ioc_;
    tcp::acceptor acceptor_;
    whisper_params app_params_; 

public:
    listener(net::io_context& ioc, tcp::endpoint endpoint, const whisper_params& app_params)
        : ioc_(ioc), acceptor_(ioc), app_params_(app_params) {
        beast::error_code ec;
        acceptor_.open(endpoint.protocol(), ec); if (ec) { fail(ec, "open"); return; }
        acceptor_.set_option(net::socket_base::reuse_address(true), ec); if (ec) { fail(ec, "set_option"); return; }
        acceptor_.bind(endpoint, ec); if (ec) { fail(ec, "bind"); return; }
        acceptor_.listen(net::socket_base::max_listen_connections, ec); if (ec) { fail(ec, "listen"); return; }
         std::cerr << "[" << std::this_thread::get_id() << "] Listener initialized successfully." << std::endl;
    }

    void run() {
         std::cerr << "[" << std::this_thread::get_id() << "] Listener::run() called. Starting accept loop." << std::endl;
        do_accept();
    }

    void stop() {
        beast::error_code ec;
        acceptor_.close(ec);
        if (ec) {
             fail(ec, "acceptor close");
        }
         std::cerr << "[" << std::this_thread::get_id() << "] Listener stopped." << std::endl;
    }


private:
    void do_accept() {
        if (!acceptor_.is_open()) {
             std::cerr << "[" << std::this_thread::get_id() << "] Acceptor closed, stopping accept loop in do_accept." << std::endl;
            return;
        }
         std::cerr << "[" << std::this_thread::get_id() << "] Listener::do_accept() called. Waiting for connection..." << std::endl;
        acceptor_.async_accept(
            net::make_strand(ioc_), 
            beast::bind_front_handler(&listener::on_accept, shared_from_this()));
    }

    void on_accept(beast::error_code ec, tcp::socket socket) {
         std::cerr << "[" << std::this_thread::get_id() << "] Listener::on_accept() entered. Error code: " << ec.message() << std::endl;

        if (ec == net::error::operation_aborted) {
              std::cerr << "[" << std::this_thread::get_id() << "] Accept operation aborted (likely server stopping)." << std::endl;
            return; 
        }
        if (ec) {
            fail(ec, "accept");
            if (acceptor_.is_open()) do_accept();
            return;
        }

        try {
              std::cerr << "[" << std::this_thread::get_id() << "] Connection accepted from: " << socket.remote_endpoint() << std::endl;
            std::make_shared<session>(std::move(socket), app_params_)->run();
        } catch (const std::exception& e) {
              std::cerr << "[" << std::this_thread::get_id() << "] Exception during session creation/run: " << e.what() << std::endl;
              beast::error_code close_ec; socket.shutdown(tcp::socket::shutdown_both, close_ec); socket.close(close_ec);
        } catch (...) {
              std::cerr << "[" << std::this_thread::get_id() << "] Unknown exception during session creation/run." << std::endl;
              beast::error_code close_ec; socket.shutdown(tcp::socket::shutdown_both, close_ec); socket.close(close_ec);
        }

        if (acceptor_.is_open()) {
              do_accept();
        } else {
              std::cerr << "[" << std::this_thread::get_id() << "] Acceptor is closed, stopping accept loop." << std::endl;
        }
    }
}; // End class listener
//------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // Load all compiled backends (e.g., CUDA)
    ggml_backend_load_all();

    whisper_params params;
    if (!whisper_params_parse(argc, argv, params)) { return 1; }

{ std::ifstream model__file(params.model_path); if (!model__file.good()) {
        fprintf(stderr, "Error: Cannot open model file '%s'. Make sure it exists and is accessible.\n", params.model_path.c_str());
        return 2;
    } }
     fprintf(stderr, "Using model: %s\n", params.model_path.c_str());

    try {
        auto const address = net::ip::make_address("0.0.0.0");
        auto const port = static_cast<unsigned short>(params.port);
        auto const threads = std::max<int>(1, params.n_threads);

        net::io_context ioc{threads};

        net::signal_set signals(ioc, SIGINT, SIGTERM);
        signals.async_wait([&](beast::error_code const&, int signal_number){
            std::cerr << "\nReceived signal " << signal_number << ". Shutting down..." << std::endl;
            ioc.stop();
        });

        auto main_listener = std::make_shared<listener>(ioc, tcp::endpoint{address, port}, params);
        main_listener->run();

        fprintf(stderr, "Whisper WebSocket server listening on %s:%d\n", address.to_string().c_str(), port);
        fprintf(stderr, "Using %d I/O threads.\n", threads);
        fprintf(stderr, "Press Ctrl+C to stop.\n");
        fprintf(stderr, "Starting Boost Asio io_context run loop...\n");

        std::vector<std::thread> v;
        v.reserve(threads); 
        for (auto i = 0; i < threads; ++i)
            v.emplace_back([&ioc] {
                std::cerr << "Starting io_context thread ID: " << std::this_thread::get_id() << std::endl;
                try {
                     ioc.run(); 
                } catch (const std::exception& e) {
                     std::cerr << "Exception in io_context thread: " << e.what() << std::endl;
                } catch (...) {
                     std::cerr << "Unknown exception in io_context thread." << std::endl;
                }
                std::cerr << "io_context thread ID: " << std::this_thread::get_id() << " finished." << std::endl;
               });

        for(auto& t : v) {
            if (t.joinable()) {
                 t.join();
            }
        }

        main_listener->stop();

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error in main: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    fprintf(stderr, "Server shut down cleanly.\n");
    return EXIT_SUCCESS;
}

// --- REVERTED --- Updated parsing function
bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
     for (int i = 1; i < argc; i++) {
         std::string arg = argv[i];
         if (arg == "-h" || arg == "--help")      { whisper_print_usage(argc, argv, params); exit(0); }
         else if (arg == "-t" || arg == "--threads")   { params.n_threads = std::stoi(argv[++i]); }
         else if (arg == "--port")               { params.port      = std::stoi(argv[++i]); }
         else if (arg == "-m" || arg == "--model")     { params.model_path = argv[++i]; }
         else if (arg == "-l" || arg == "--language")  { params.language  = argv[++i]; }
         else if (arg == "-tr"|| arg == "--translate") { params.translate = true; }
         else if (arg == "-ng" || arg == "--no-gpu")   { params.use_gpu = false; }
         else if (arg == "--step")               { params.step_ms     = std::stoi(argv[++i]); }
         // (Removed --keep)
         else if (arg == "--temp")               { params.temperature = std::stof(argv[++i]); }
         else if (arg == "-bs" || arg == "--beam-size") { params.beam_size = std::stoi(argv[++i]); }
         else if (arg == "-mt" || arg == "--max-tokens"){ params.max_tokens = std::stoi(argv[++i]); }
         else if (arg == "-nf" || arg == "--no-fallback"){ params.no_fallback = true; }
         else if (arg == "-ts" || arg == "--timestamps"){ params.print_timestamps = true; }
         else { fprintf(stderr, "error: unknown argument: %s\n", arg.c_str()); whisper_print_usage(argc, argv, params); exit(1); }
     }
     // Validate params
     if (params.n_threads <= 0) { fprintf(stderr, "error: number of threads must be positive (%d)\n", params.n_threads); return false; }
     if (params.port <= 0 || params.port > 65535) { fprintf(stderr, "error: port number must be between 1 and 65535 (%d)\n", params.port); return false; }
     if (params.step_ms <= 0) { fprintf(stderr, "error: step_ms must be positive (%d)\n", params.step_ms); return false; }

     std::ifstream f(params.model_path.c_str());
     if (!f.good()) { fprintf(stderr, "error: model file does not exist or cannot be read: %s\n", params.model_path.c_str()); return false; }

     return true;
}

// --- REVERTED --- Updated usage function
void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
      fprintf(stderr, "\nusage: %s [options]\n\n", argv[0]);
      fprintf(stderr, "options:\n");
      fprintf(stderr, "  -h,       --help          show this help message and exit\n");
      fprintf(stderr, "            --port PORT     [%d] port to listen on\n", params.port);
      fprintf(stderr, "  -t N,     --threads N     [%d] number of threads for I/O and computation\n", params.n_threads);
      fprintf(stderr, "  -m FNAME, --model FNAME   [%s] path to the Whisper GGML model file\n", params.model_path.c_str());
      fprintf(stderr, "  -l LANG,  --language LANG [%s] spoken language ('auto' to detect, 'en', 'es', etc.)\n", params.language.c_str());
      fprintf(stderr, "  -tr,      --translate     [%s] translate result to english\n", params.translate ? "true" : "false");
      fprintf(stderr, "  -ng,      --no-gpu        [%s] disable GPU inference (use CPU only)\n", !params.use_gpu ? "true" : "false");
      fprintf(stderr, "  -ts,      --timestamps    [%s] print timestamps in output (for JSON)\n", params.print_timestamps ? "true" : "false");
      fprintf(stderr, "            --step N        [%d] audio step size in milliseconds (how often to process)\n", params.step_ms);
      // (Removed --keep)
      fprintf(stderr, "            --temp N        [%.1f] transcription temperature (0.0=greedy)\n", params.temperature);
      fprintf(stderr, "  -bs N,    --beam-size N   [%d] beam size for beam search (-1=use default/greedy)\n", params.beam_size);
      fprintf(stderr, "  -mt N,    --max-tokens N  [%d] maximum number of tokens per audio chunk (0=no limit)\n", params.max_tokens);
      fprintf(stderr, "  -nf,      --no-fallback   [%s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
      fprintf(stderr, "\n");
}

// --- Global fail function (unchanged) ---
void fail(beast::error_code ec, char const* what, bool is_error /*= true*/) {
    if (ec == net::error::operation_aborted || ec == websocket::error::closed || ec == net::error::eof) {
        std::cerr << "[" << std::this_thread::get_id() << "] Info: Operation stopped or socket closed normally (" << what << "). Code: " << ec.value() << std::endl;
        return;
    }
    if (is_error) {
         std::cerr << "[" << std::this_thread::get_id() << "] Error encountered in '" << what << "': " << ec.message() << " (Code: " << ec.value() << ")" << "\n";
    } else {
         std::cerr << "[" << std::this_thread::get_id() << "] Warning encountered in '" << what << "': " << ec.message() << " (Code: " << ec.value() << ")" << "\n";
    }
}
