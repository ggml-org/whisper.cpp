// Resumable / asynchronous streaming transcription example.
//
// This demonstrates the resumable whisper API:
//   - whisper_append_audio_with_state()       feed PCM incrementally
//   - whisper_full_resumable_with_state(.., false)  decode complete 30s windows
//   - whisper_full_resumable_with_state(.., true)   flush the trailing window
//
// Unlike examples/stream (which repeatedly calls whisper_full() on a sliding
// window and therefore re-decodes overlapping audio, producing output that
// changes between iterations), this decodes each window exactly once, resumes
// the seek position and the rolling text context from the state, and never
// revises already-emitted segments. The result is consistent with a single
// batch run.
//
// The design that matters for a real application:
//   - ONE producer thread captures audio and pushes PCM into a queue.
//   - ONE worker thread owns a dedicated whisper_state and runs inference,
//     decoupled from capture so transcription can run at full quality while
//     recording continues.
//   - The model weights (whisper_context) are shared read-only; each worker
//     would use its own whisper_state.
//
// Here the "producer" reads a WAV file and (optionally) paces it in real time
// to simulate a live source. In your application, replace the producer with
// your microphone / network audio source and push 16 kHz mono f32 PCM.

#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// -----------------------------------------------------------------------------
// A minimal thread-safe PCM queue (single producer, single consumer).
// -----------------------------------------------------------------------------
class pcm_queue {
public:
    void push(const float * data, size_t n) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            buf_.insert(buf_.end(), data, data + n);
        }
        cv_.notify_one();
    }

    // Signal that no more audio will arrive.
    void close() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    // Block until at least one sample is available or the queue is closed.
    // Drains everything currently buffered into `out`. Returns false only when
    // the queue is closed AND drained (i.e. the stream has ended).
    bool pop_all(std::vector<float> & out) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&] { return !buf_.empty() || closed_; });
        out.assign(buf_.begin(), buf_.end());
        buf_.clear();
        return !out.empty() || !closed_;
    }

private:
    std::mutex                mtx_;
    std::condition_variable   cv_;
    std::deque<float>         buf_;
    bool                      closed_ = false;
};

// -----------------------------------------------------------------------------
struct cli_params {
    std::string model    = "models/ggml-base.en.bin";
    std::string fname;
    std::string language = "en";
    int   n_threads      = std::min(4, (int) std::thread::hardware_concurrency());
    int   chunk_ms       = 1000;  // how much audio the producer emits per push
    bool  realtime       = false; // pace the producer to wall-clock time
    bool  window_norm    = false; // use the per-window (live) mel normalization
    float half_life      = 3.0f;  // release half-life in seconds (window norm)
    bool  translate      = false;
};

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m <model.bin> -f <audio.wav> [options]\n"
        "  -m, --model PATH       model path (default: models/ggml-base.en.bin)\n"
        "  -f, --file PATH        input WAV (16 kHz mono); required\n"
        "  -l, --language LANG    spoken language or 'auto' (default: en)\n"
        "  -t, --threads N        inference threads (default: %d)\n"
        "      --chunk-ms N       producer chunk size in ms (default: 1000)\n"
        "      --realtime         pace the producer to real time (simulate live)\n"
        "      --window-norm      per-window mel normalization (live AGC) instead of global\n"
        "      --half-life S      release half-life in seconds for --window-norm (default: 3.0)\n"
        "      --translate        translate to English\n",
        argv0, std::min(4, (int) std::thread::hardware_concurrency()));
}

static bool parse_args(int argc, char ** argv, cli_params & p) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(1); }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")    p.model    = next("model");
        else if (a == "-f" || a == "--file")     p.fname    = next("file");
        else if (a == "-l" || a == "--language") p.language = next("language");
        else if (a == "-t" || a == "--threads")  p.n_threads = std::stoi(next("threads"));
        else if (a == "--chunk-ms")              p.chunk_ms = std::stoi(next("chunk-ms"));
        else if (a == "--realtime")              p.realtime = true;
        else if (a == "--window-norm")           p.window_norm = true;
        else if (a == "--half-life")             p.half_life = std::stof(next("half-life"));
        else if (a == "--translate")             p.translate = true;
        else if (a == "-h" || a == "--help")     { print_usage(argv[0]); exit(0); }
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return false; }
    }
    if (p.fname.empty()) { print_usage(argv[0]); return false; }
    return true;
}

// Print every segment that was produced since `already_printed`, return the new total.
static int print_new_segments(whisper_state * state, int already_printed) {
    const int n = whisper_full_n_segments_from_state(state);
    for (int i = already_printed; i < n; i++) {
        const int64_t t0 = whisper_full_get_segment_t0_from_state(state, i);
        const int64_t t1 = whisper_full_get_segment_t1_from_state(state, i);
        const char * text = whisper_full_get_segment_text_from_state(state, i);
        printf("[%s --> %s]%s\n", to_timestamp(t0).c_str(), to_timestamp(t1).c_str(), text);
    }
    fflush(stdout);
    return n;
}

int main(int argc, char ** argv) {
    cli_params p;
    if (!parse_args(argc, argv, p)) return 1;

    // load the audio up front (the producer thread streams it out below)
    std::vector<float> pcm;
    std::vector<std::vector<float>> pcms;
    if (!read_audio_data(p.fname, pcm, pcms, /*stereo=*/false)) {
        fprintf(stderr, "error: failed to read audio '%s'\n", p.fname.c_str());
        return 1;
    }

    // shared, read-only context (model weights)
    whisper_context_params cparams = whisper_context_default_params();
    whisper_context * ctx = whisper_init_from_file_with_params_no_state(p.model.c_str(), cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to load model '%s'\n", p.model.c_str());
        return 1;
    }

    // dedicated inference state for the worker (one per concurrent stream)
    whisper_state * state = whisper_init_state(ctx);
    if (!state) {
        fprintf(stderr, "error: failed to init state\n");
        whisper_free(ctx);
        return 1;
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress   = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    wparams.translate        = p.translate;
    wparams.language         = p.language.c_str();
    wparams.n_threads        = p.n_threads;
    wparams.no_context       = false; // carry rolling context across windows
    if (p.window_norm) {
        wparams.mel_norm_mode = WHISPER_MEL_NORM_WINDOW;
        wparams.mel_norm_half_life = p.half_life;
    } else {
        wparams.mel_norm_mode = WHISPER_MEL_NORM_GLOBAL;
    }

    whisper_resumable_reset_with_state(ctx, state);

    pcm_queue queue;
    std::atomic<bool> failed{false};

    // ---- worker thread: append audio + decode complete windows as they arrive ----
    std::thread worker([&] {
        int printed = 0;
        std::vector<float> chunk;
        while (queue.pop_all(chunk)) {
            if (chunk.empty()) continue; // woke up but nothing buffered yet
            if (whisper_append_audio_with_state(ctx, state, chunk.data(), (int) chunk.size()) != 0) {
                failed = true; return;
            }
            const int ret = whisper_full_resumable_with_state(ctx, state, wparams, /*finalize=*/false);
            if (ret < 0) { failed = true; return; }
            printed = print_new_segments(state, printed);
        }
        // stream ended: flush the trailing (< 30s) window
        const int ret = whisper_full_resumable_with_state(ctx, state, wparams, /*finalize=*/true);
        if (ret < 0) { failed = true; return; }
        print_new_segments(state, printed);
    });

    // ---- producer: stream the file out in chunks (this is where a mic would feed in) ----
    const int chunk_n = (p.chunk_ms * WHISPER_SAMPLE_RATE) / 1000;
    for (size_t off = 0; off < pcm.size(); off += chunk_n) {
        const size_t n = std::min((size_t) chunk_n, pcm.size() - off);
        queue.push(pcm.data() + off, n);
        if (p.realtime) {
            std::this_thread::sleep_for(std::chrono::milliseconds((1000 * (int) n) / WHISPER_SAMPLE_RATE));
        }
    }
    queue.close();

    worker.join();

    whisper_free_state(state);
    whisper_free(ctx);

    return failed ? 2 : 0;
}
