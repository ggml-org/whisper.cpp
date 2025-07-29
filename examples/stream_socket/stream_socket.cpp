#include "common-whisper.h"
#include "whisper.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <iomanip>
#include <sstream>

// Very small helper: write all bytes, handling short writes
static bool write_all(int fd, const void * data, size_t len) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    while (len > 0) {
        ssize_t w = ::write(fd, p, len);
        if (w <= 0) return false;
        p += w; len -= w;
    }
    return true;
}

// ---------- Ring-buffer for inter-thread transfer ---------------------------------
class pcm_ring_buffer {
public:
    void push(const float * data, size_t n) {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_buf.insert(m_buf.end(), data, data + n);
        m_cv.notify_all();
    }

    // blocking pop of up to n samples – returns 0 if finished and nothing left
    size_t pop(size_t n, std::vector<float> & out) {
        std::unique_lock<std::mutex> lock(m_mtx);
        m_cv.wait(lock, [&]{ return m_finished || m_buf.size() >= n; });
        size_t n_pop = std::min(n, m_buf.size());
        if (n_pop == 0) return 0;
        out.assign(m_buf.begin(), m_buf.begin() + n_pop);
        m_buf.erase(m_buf.begin(), m_buf.begin() + n_pop);
        return n_pop;
    }

    void pop_all(std::vector<float> & out) {
        std::lock_guard<std::mutex> lock(m_mtx);
        out.assign(m_buf.begin(), m_buf.end());
        m_buf.clear();
    }

    void mark_finished() {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_finished = true;
        m_cv.notify_all();
    }

    bool finished() const {
        std::lock_guard<std::mutex> lock(m_mtx);
        return m_finished && m_buf.empty();
    }

    // Drop the first n samples (no-op if fewer are present)
    void drop(size_t n) {
        std::lock_guard<std::mutex> lock(m_mtx);
        if (n >= m_buf.size()) {
            m_buf.clear();
        } else {
            m_buf.erase(m_buf.begin(), m_buf.begin() + n);
        }
    }

    // Current buffered duration in milliseconds
    size_t duration_ms() const {
        std::lock_guard<std::mutex> lock(m_mtx);
        return (m_buf.size() * 1000) / WHISPER_SAMPLE_RATE;
    }

private:
    std::vector<float>       m_buf;
    bool                     m_finished = false;
    mutable std::mutex       m_mtx;
    std::condition_variable  m_cv;
};

// Forward declaration of abort callback
static bool whisper_should_abort(void * user_data);

// ---------- Audio reader thread ----------------------------------------------------

void reader_thread(int client_fd, pcm_ring_buffer & rb, std::atomic<bool> * abort_flag) {
    constexpr size_t BUF_SZ = 4096;
    std::vector<int16_t> buf(BUF_SZ / sizeof(int16_t));

    while (true) {
        ssize_t r = ::read(client_fd, buf.data(), BUF_SZ);
        if (r <= 0) break; // EOF or error → finish

        size_t n_samples = r / sizeof(int16_t);
        std::vector<float> f32(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            f32[i] = static_cast<float>(buf[i]) / 32768.0f;
        }
        rb.push(f32.data(), f32.size());
    }

    // Signal main thread to cancel any ongoing whisper_full() calls
    if (abort_flag) {
        abort_flag->store(true);
    }

    rb.mark_finished();
}

// ---------- Helper: concatenate all segments --------------------------------------
static std::string collect_segments(struct whisper_context * ctx) {
    std::string out;
    const int n = whisper_full_n_segments(ctx);
    for (int i = 0; i < n; ++i) {
        const char * txt = whisper_full_get_segment_text(ctx, i);
        if (i) out += " ";
        out += txt;
    }
    return out;
}

// ---------- Helper: timestamped logging ---------------------------------------
static void log_ts(const std::string & msg) {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto now_c = system_clock::to_time_t(now);
    auto ms   = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm_now;
#if defined(_MSC_VER)
    localtime_s(&tm_now, &now_c);
#else
    localtime_r(&now_c, &tm_now);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count()
        << " " << msg;
    std::cerr << oss.str() << std::endl;
}

// -----------------------------------------------------------------------------
// Global config (tuned via CLI in main)
static int32_t g_step_ms   = 500;   // emit partials every 0.5 s
static int32_t g_length_ms = 15000; // 10-s rolling window fed to Whisper
static int32_t g_keep_ms   = 200;   // overlap between windows

// When true, suppress incremental partial transcriptions and only run a final
// full-context whisper pass after the audio stream ends (set via --no-stream).
static bool    g_no_stream = false;

// Adaptive-scheduler & safety-net ---------------------------------------------
static const int32_t MIN_STEP_MS   = 400;   // lower bound for real-time feel
static const int32_t MAX_STEP_MS   = 2000;  // upper bound – keeps latency bounded
static const float   EWMA_ALPHA    = 0.30f; // smoothing for running average
static const float   SAFETY_FACTOR = 1.10f; // 10 % head-room between passes

// Ring-buffer hard cap (discard oldest audio when exceeded)
static const int32_t RING_CAP_MS   = 20000; // never queue more than 20 s

// ---------- Main per-connection handler -------------------------------------------

void process_connection(int client_fd, struct whisper_context * ctx) {
    // whisper params (could expose CLI later)
    // Abort flag shared between reader_thread and whisper abort callback
    std::atomic<bool> abort_requested(false);

    int32_t step_ms   = g_step_ms; // mutable for adaptive scheduling
    const int32_t length_ms = g_length_ms;
    const int32_t keep_ms   = g_keep_ms;

    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t beam_size = -1;

    // Adaptive scheduler state
    float   avg_ms   = (float)step_ms; // initialise EWMA

    const int n_samples_len  = length_ms * WHISPER_SAMPLE_RATE / 1000;
    const int n_samples_keep = keep_ms   * WHISPER_SAMPLE_RATE / 1000;

    pcm_ring_buffer rb;
    std::thread reader(reader_thread, client_fd, std::ref(rb), &abort_requested);

    std::vector<float> pcmf32_old;
    // Capture the *entire* audio stream so we can run a final full-context
    // transcription once the user stops speaking.  This guarantees that the
    // final output covers the *whole* utterance (even >60 s) instead of only
    // whatever fit into the rolling 10-s window used for partial updates.
    std::vector<float> pcmf32_all;
    // Accumulated transcript across all iterations
    std::string transcript_accum;

    auto merge_into_accum = [&](const std::string & part){
        // Append part to transcript_accum, removing any overlap prefix
        if (transcript_accum.empty()) {
            transcript_accum = part;
            return;
        }
        // Find the maximum overlap between end of accum and beginning of part
        size_t max_overlap = std::min(transcript_accum.size(), part.size());
        size_t overlap = 0;
        for (size_t len = max_overlap; len > 0; --len) {
            if (transcript_accum.compare(transcript_accum.size() - len, len, part, 0, len) == 0) {
                overlap = len;
                break;
            }
        }
        transcript_accum += part.substr(overlap);
    };

    auto send_json = [&](const std::string & type, const std::string & text){
        std::string line = std::string("{\"type\":\"") + type + "\",\"text\":\"" + text + "\"}\n";
        write_all(client_fd, line.data(), line.size());
    };

    log_ts("Mic started / connection opened");
    // processing loop
    while (true) {
        // Compute step size and pop corresponding samples from the ring buffer
        const int n_samples_step = step_ms * WHISPER_SAMPLE_RATE / 1000;

        std::vector<float> pcmf32_new;
        size_t popped = rb.pop(n_samples_step, pcmf32_new);
        if (popped == 0) {
            if (rb.finished()) break;
            continue;
        }

        // If the audio stream has finished, skip further partial inference and jump to final pass.
        if (rb.finished()) {
            // Still append the last batch of samples to the cumulative buffer so the final pass sees them.
            pcmf32_all.insert(pcmf32_all.end(), pcmf32_new.begin(), pcmf32_new.end());
            break;
        }

        const int n_samples_new  = pcmf32_new.size();
        const int n_samples_take = std::min((int)pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

        std::vector<float> pcmf32_cur(n_samples_new + n_samples_take);
        if (n_samples_take) std::copy(pcmf32_old.end()-n_samples_take, pcmf32_old.end(), pcmf32_cur.begin());
        std::copy(pcmf32_new.begin(), pcmf32_new.end(), pcmf32_cur.begin() + n_samples_take);
        pcmf32_old = pcmf32_cur;

        // ------------------------------------------------------------------
        // Append *new* samples (no overlap) to the cumulative buffer so we
        // have the raw audio for a high-fidelity final pass later.
        // ------------------------------------------------------------------
        pcmf32_all.insert(pcmf32_all.end(), pcmf32_new.begin(), pcmf32_new.end());

        if (!g_no_stream) {
            whisper_full_params wparams = whisper_full_default_params(beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
            wparams.print_progress   = false;
            wparams.print_realtime   = false;
            wparams.print_timestamps = false;
            wparams.max_tokens       = 0;
            wparams.n_threads        = n_threads;
            wparams.beam_search.beam_size = beam_size;

            // Set abort callback so we can cancel this inference if recording stops
            wparams.abort_callback = whisper_should_abort;
            wparams.abort_callback_user_data = &abort_requested;
            abort_requested.store(false); // reset for this inference

            auto t_start = std::chrono::steady_clock::now();
            whisper_full(ctx, wparams, pcmf32_cur.data(), pcmf32_cur.size());
            auto t_end   = std::chrono::steady_clock::now();
            auto dur_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
            std::string part = collect_segments(ctx);
            merge_into_accum(part);
            send_json("partial", part);
            log_ts(std::string("[PART] transcription time: ") + std::to_string(dur_ms) + " ms");

            // ------------------------------------------------------------------
            // Adaptive step: update EWMA and derive next step_ms within bounds
            // ------------------------------------------------------------------
            avg_ms = (1.0f - EWMA_ALPHA) * avg_ms + EWMA_ALPHA * (float)dur_ms;

            // Manual clamp (C++11 compatible) – avoids std::clamp dependency
            int32_t new_step = int(avg_ms * SAFETY_FACTOR);
            if (new_step < MIN_STEP_MS) new_step = MIN_STEP_MS;
            if (new_step > MAX_STEP_MS) new_step = MAX_STEP_MS;
            step_ms = new_step;
        }

        // ------------------------------------------------------------------
        // Ring-buffer cap – discard oldest audio if backlog exceeds threshold
        // ------------------------------------------------------------------
        while (rb.duration_ms() > RING_CAP_MS) {
            rb.drop(n_samples_step);
        }
        }

    // flush leftovers
    {
        std::vector<float> pcmf32_new;
        rb.pop_all(pcmf32_new);

        if (!pcmf32_new.empty()) {
            // Just add the leftover samples to the cumulative buffer for the final pass.
            // Skip the redundant whisper processing since the final pass will handle everything.
            pcmf32_all.insert(pcmf32_all.end(), pcmf32_new.begin(), pcmf32_new.end());
        }
    }

    // ------------------------------------------------------------------
    // FINAL PASS – transcribe the *full* audio for maximum accuracy
    // ------------------------------------------------------------------

    std::string final_transcript;
    if (!pcmf32_all.empty()) {
        abort_requested.store(false); // ensure final pass is not aborted
        auto t_start = std::chrono::steady_clock::now();
        whisper_full_params wparams_final = whisper_full_default_params(beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
        wparams_final.print_progress   = false;
        wparams_final.print_realtime   = false;
        wparams_final.print_timestamps = false;
        wparams_final.max_tokens       = 0;
        wparams_final.n_threads        = n_threads;
        wparams_final.beam_search.beam_size = beam_size;

        // No abort callback for final pass (fully process)
        wparams_final.abort_callback = nullptr;
        wparams_final.abort_callback_user_data = nullptr;

        if (whisper_full(ctx, wparams_final, pcmf32_all.data(), pcmf32_all.size()) != 0) {
            std::cerr << "whisper_full() failed on full-audio pass" << std::endl;
        }
        auto t_end   = std::chrono::steady_clock::now();
        auto dur_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        final_transcript = collect_segments(ctx);
        log_ts(std::string("[FINAL] transcription time: ") + std::to_string(dur_ms) + " ms");
    } else {
        // Fallback to whatever we accumulated during streaming (should not happen)
        final_transcript = transcript_accum;
    }

    send_json("final", final_transcript);

    log_ts("Mic ended / connection closed");

    reader.join();
    ::shutdown(client_fd, SHUT_RDWR);
    ::close(client_fd);
}

// ---------- ggml abort callback -----------------------------------------------------
static bool whisper_should_abort(void * user_data) {
    const std::atomic<bool> * flag = static_cast<const std::atomic<bool> *>(user_data);
    return flag && flag->load();
}

// ---------- Main ------------------------------------------------------------------

int main(int argc, char ** argv) {
    const char * sock_path = "/tmp/whisper_stream.sock";
    if (argc > 1 && std::strcmp(argv[1], "--socket") == 0 && argc > 2) {
        sock_path = argv[2];
    }

    // Optional tuning parameters
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--step") == 0) {
            g_step_ms = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--length") == 0) {
            g_length_ms = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--keep") == 0) {
            g_keep_ms = std::atoi(argv[i + 1]);
        }
    }

    // Independent scan for flag-style options (no additional parameter)
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-stream") == 0) {
            g_no_stream = true;
        }
    }

    ::unlink(sock_path); // remove previous

    int srv_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv_fd < 0) {
        perror("socket");
        return 1;
    }

    struct sockaddr_un addr; std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path)-1);

    if (::bind(srv_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }

    if (::listen(srv_fd, 4) < 0) {
        perror("listen");
        return 1;
    }

    std::cerr << "[whisper-socket] listening on " << sock_path << std::endl;

    ggml_backend_load_all();

    // ---------------------------------------------------------------------
    // Model path – can be passed via --model <file>, otherwise env WHISPER_MODEL or default path
    const char * model_path = getenv("WHISPER_MODEL");
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0) {
            model_path = argv[i + 1];
            break;
        }
    }
    if (!model_path) {
        model_path = "models/ggml-base.en.bin"; // relative to repo root
    }

    std::cerr << "[whisper-socket] loading model " << model_path << " …" << std::endl;

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu  = true;
    cparams.flash_attn = true;
    struct whisper_context * ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        std::cerr << "failed to load model" << std::endl;
        return 2;
    }

    while (true) {
        int client_fd = ::accept(srv_fd, nullptr, nullptr);
        if (client_fd < 0) {
            perror("accept");
            continue;
        }
        std::cerr << "[whisper-socket] client connected" << std::endl;
        process_connection(client_fd, ctx);
        std::cerr << "[whisper-socket] client done" << std::endl;
    }

    whisper_free(ctx);
    return 0;
}
