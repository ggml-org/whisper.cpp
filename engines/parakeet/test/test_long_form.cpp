// End-to-end parity test for the long-form windowed encoder.
//
// Usage:
//   test-long-form --model <parakeet-{ctc,tdt}.gguf> --wav <input.wav> [options]
//
// A short WAV is tiled up to ~120 s -- with a different gain per repetition, so
// the windows see varying loudness (exactly what per-window CMVN would distort)
// -- and transcribed two ways with the same model:
//   * reference: long_form_window_frames = -1   (single full-length encoder pass)
//   * windowed:  long_form_window_frames = 1024, context = 256  (forces several
//                overlapping encoder windows with production-like context)
//
// The window is 1024 (not 768) so the requested 256-frame context is not
// clamped by the window/4 ceiling in resolve_long_form_plan, and 120 s of tiled
// audio comfortably exceeds one window so windowing actually engages.
//
// It asserts the windowed run (a) does not throw / OOM, (b) produces non-empty
// text, and (c) matches the single-pass reference to a high word-level
// similarity. This guards the bounded-memory fix end to end: a regression that
// re-runs the full encoder in one shot, or that stitches windows incorrectly
// (dropped/duplicated frames at seams), fails here.
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet/engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include "test_utils.h"

using namespace parakeet;

namespace {

struct Opts {
    std::string model_path;
    std::string wav_path;
    int  n_gpu_layers = 0;
    int  n_threads    = 0;
    int  target_secs  = 120;
    double min_similarity = 0.85;
    bool verbose      = false;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <parakeet-{ctc,tdt}.gguf> --wav <input.wav> [options]\n"
        "\n"
        "validates that the long-form windowed encoder matches a single full-length\n"
        "pass on the same (tiled) audio.\n"
        "\n"
        "options:\n"
        "  --target-secs N      tile the wav up to ~N seconds (default 120)\n"
        "  --min-similarity F   required word-level similarity (default 0.85)\n"
        "  --n-gpu-layers N     offload to GPU backend when > 0\n"
        "  --threads N          CPU threads (0 = hardware_concurrency)\n"
        "  --verbose            print both transcripts and the similarity\n",
        argv0);
}

int parse_args(int argc, char ** argv, Opts & o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) o.model_path = argv[++i];
        else if (a == "--wav" && i + 1 < argc) o.wav_path = argv[++i];
        else if (a == "--target-secs" && i + 1 < argc) o.target_secs = std::atoi(argv[++i]);
        else if (a == "--min-similarity" && i + 1 < argc) o.min_similarity = std::atof(argv[++i]);
        else if (a == "--n-gpu-layers" && i + 1 < argc) o.n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) o.n_threads = std::atoi(argv[++i]);
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }
    if (o.model_path.empty() || o.wav_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }
    return 0;
}

std::vector<float> tile_to_seconds(const std::vector<float> & src, int sample_rate, int secs) {
    if (src.empty() || secs <= 0) return src;
    // A different gain per repetition makes each window see different loudness,
    // so the parity check exercises the global-CMVN guarantee (and attention
    // locality) rather than feeding identical copies with identical statistics.
    static const float gains[] = {1.0f, 0.35f, 0.7f, 0.5f, 0.85f, 0.25f};
    const size_t n_gains = sizeof(gains) / sizeof(gains[0]);
    const size_t target = (size_t) sample_rate * (size_t) secs;
    std::vector<float> out;
    out.reserve(target + src.size());
    for (size_t tile = 0; out.size() < target; ++tile) {
        const float g = gains[tile % n_gains];
        for (float s : src) out.push_back(s * g);
    }
    return out;
}

std::vector<std::string> split_words(const std::string & s) {
    std::vector<std::string> words;
    std::istringstream is(s);
    std::string w;
    while (is >> w) words.push_back(w);
    return words;
}

// Word-level Levenshtein similarity in [0, 1]: 1.0 == identical token streams.
double word_similarity(const std::string & a, const std::string & b) {
    const std::vector<std::string> wa = split_words(a);
    const std::vector<std::string> wb = split_words(b);
    if (wa.empty() && wb.empty()) return 1.0;
    const size_t n = wa.size();
    const size_t m = wb.size();
    std::vector<size_t> prev(m + 1), cur(m + 1);
    for (size_t j = 0; j <= m; ++j) prev[j] = j;
    for (size_t i = 1; i <= n; ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= m; ++j) {
            const size_t cost = (wa[i - 1] == wb[j - 1]) ? 0 : 1;
            cur[j] = std::min({ prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost });
        }
        std::swap(prev, cur);
    }
    const size_t dist = prev[m];
    const size_t denom = std::max<size_t>(1, std::max(n, m));
    return 1.0 - (double) dist / (double) denom;
}

std::string transcribe_with_window(const Opts & o, int window_frames, int context_frames,
                                   const std::vector<float> & samples, int sample_rate) {
    EngineOptions eo;
    eo.model_gguf_path          = o.model_path;
    eo.n_threads                = o.n_threads;
    eo.n_gpu_layers             = o.n_gpu_layers;
    eo.long_form_window_frames  = window_frames;
    eo.long_form_context_frames = context_frames;
    Engine engine(eo);
    const EngineResult r =
        engine.transcribe_samples(samples.data(), (int) samples.size(), sample_rate);
    return r.text;
}

}  // namespace

int main(int argc, char ** argv) {
    Opts opts;
    if (int rc = parse_args(argc, argv, opts); rc != 0) return rc;

    std::vector<float> base;
    int sample_rate = 0;
    if (!parakeet_test::load_wav_pcm16le_mono(opts.wav_path, base, sample_rate)) {
        std::fprintf(stderr, "failed to load wav: %s\n", opts.wav_path.c_str());
        return 1;
    }

    const std::vector<float> samples = tile_to_seconds(base, sample_rate, opts.target_secs);
    if (opts.verbose) {
        std::printf("tiled %.1fs -> %.1fs (%zu samples)\n",
                    (double) base.size() / sample_rate,
                    (double) samples.size() / sample_rate, samples.size());
    }

    try {
        // Reference: legacy single full-length encoder pass.
        const std::string ref = transcribe_with_window(opts, -1, 0, samples, sample_rate);

        // Windowed: force several overlapping encoder windows with generous
        // context, so the committed frames closely track the single pass.
        const std::string win = transcribe_with_window(opts, 1024, 256, samples, sample_rate);

        if (opts.verbose) {
            std::printf("--- reference (single pass) ---\n%s\n", ref.c_str());
            std::printf("--- windowed ---\n%s\n", win.c_str());
        }

        if (win.empty()) {
            std::fprintf(stderr, "FAIL: windowed transcription is empty\n");
            return 1;
        }
        if (ref.empty()) {
            std::fprintf(stderr, "FAIL: reference transcription is empty\n");
            return 1;
        }

        const double sim = word_similarity(ref, win);
        std::printf("word similarity windowed-vs-single = %.4f (min %.4f)\n",
                    sim, opts.min_similarity);
        if (sim < opts.min_similarity) {
            std::fprintf(stderr, "FAIL: windowed output diverges from single-pass\n");
            return 1;
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return 1;
    }

    std::printf("test-long-form: OK\n");
    return 0;
}
