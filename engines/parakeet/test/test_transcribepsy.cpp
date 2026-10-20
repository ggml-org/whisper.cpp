// End-to-end checks for TranscribePsy, the 254M FastConformer-TDT distilled
// from Parakeet-TDT-0.6B-v3 (https://huggingface.co/qvac/TranscribePsy-250M-EU-sub-gguf),
// on the engine API:
//   1. the published GGUF loads and transcribes jfk.wav to the expected words;
//   2. transcribe_mel(compute_mel_wav(...)) returns the same tokens as transcribe();
//   3. compute_mel_wav on a worker thread, concurrent with a transcription on
//      the main thread, changes neither result;
//   4. a 150 s tiling transcribed with the 60 s / 25 % windows recommended on
//      phones matches the single pass to a high word similarity;
//   5. EngineResult's stage times account for total_ms;
//   6. transcribe_mel rejects a PrecomputedMel whose buffer does not hold the
//      declared frames x bins (empty, truncated, oversized, zero frames, wrong
//      bin count, absurd frame count) and still accepts an intact one;
//   7. a pause-free English recording of about three minutes (jfk.wav and
//      sample-16k.wav concatenated six times) transcribes to its known words
//      within one percent word error, both with the engine's default options
//      and with the 60 s / 25 % windows, on whichever backend the test runs
//      (ggml CPU, a GPU through --n-gpu-layers, or the Core ML sidecars of
//      the encoder-less GGUF), so every backend is held to the same accuracy;
//      on the ggml paths the default options take the model's validated
//      windows and transcribe exactly like the explicit request.
//
// With --long-wav it additionally transcribes that recording once and prints
// the stage times and the transcript, for measuring a long input on whichever
// path the loaded model takes (ggml windows, or Core ML segments with a TDT
// decoder sidecar); nothing is asserted about it beyond a non-empty result.
//
// Usage:
//   test-transcribepsy --model <transcribepsy.gguf> --wav <jfk.wav> --wav2 <other.wav>
//                      --wav-sample <sample-16k.wav> [--n-gpu-layers N]
//                      [--long-wav <long.wav>] [--threads N] [--min-similarity F] [--verbose]
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet/engine.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <future>
#include <limits>
#include <string>
#include <vector>

#include "test_utils.h"

using namespace parakeet;
using parakeet_test::tile_to_seconds;
using parakeet_test::word_similarity;

namespace {

constexpr const char * kExpectedJfkWords =
    "and so my fellow americans ask not what your country can do for you "
    "ask what you can do for your country";
constexpr const char * kExpectedSampleWords =
    "alice was beginning to get very tired of sitting by her sister on the bank and of "
    "having nothing to do once or twice she had peeped into the book her sister was "
    "reading but it had no pictures or conversations in it and what is the use of a book "
    "thought alice without pictures or conversations";
constexpr int    kTiledSeconds       = 150;
constexpr int    kLongEnglishTiles   = 6;     // (jfk + sample-16k) x 6 = 187 s without pauses
constexpr double kMinLongAccuracy    = 0.99;  // at most one word error per hundred
constexpr int    kPhoneWindowFrames  = 750;  // 60 s at the 80 ms encoder stride
constexpr int    kPhoneContextFrames = 94;   // 25 % overlap: 12.5 % each side
constexpr double kDefaultMinSimilarity = 0.85;
constexpr double kTimingSlackMs      = 1.0;

struct Opts {
    std::string model_path;
    std::string wav_path;
    std::string wav2_path;
    std::string wav_sample_path;
    std::string long_wav_path;
    int    n_threads      = 0;
    int    n_gpu_layers   = 0;
    double min_similarity = kDefaultMinSimilarity;
    bool   verbose        = false;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <transcribepsy.gguf> --wav <jfk.wav> --wav2 <other.wav> [options]\n"
        "\n"
        "options:\n"
        "  --wav-sample PATH    sample-16k.wav, tiled with jfk.wav into the long English recording\n"
        "  --n-gpu-layers N     offload the encoder to the GPU backend of this build\n"
        "  --long-wav PATH      also transcribe this recording once and print its stage times\n"
        "  --threads N          CPU threads (0 = hardware_concurrency)\n"
        "  --min-similarity F   required windowed-vs-single word similarity (default %.2f)\n"
        "  --verbose            print the transcripts and timings\n",
        argv0, kDefaultMinSimilarity);
}

int parse_args(int argc, char ** argv, Opts & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) o.model_path = argv[++i];
        else if (a == "--wav" && i + 1 < argc) o.wav_path = argv[++i];
        else if (a == "--wav2" && i + 1 < argc) o.wav2_path = argv[++i];
        else if (a == "--wav-sample" && i + 1 < argc) o.wav_sample_path = argv[++i];
        else if (a == "--n-gpu-layers" && i + 1 < argc) o.n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--long-wav" && i + 1 < argc) o.long_wav_path = argv[++i];
        else if (a == "--threads" && i + 1 < argc) o.n_threads = std::atoi(argv[++i]);
        else if (a == "--min-similarity" && i + 1 < argc) o.min_similarity = std::atof(argv[++i]);
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }
    if (o.model_path.empty() || o.wav_path.empty() || o.wav2_path.empty() ||
        o.wav_sample_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }
    return 0;
}

// Lower-cased letters and digits separated by single spaces, so punctuation
// and casing, which differ legitimately between decoders, do not count.
std::string normalized_words(const std::string & text) {
    std::string out;
    bool pending_space = false;
    for (unsigned char c : text) {
        if (std::isalnum(c)) {
            if (pending_space && !out.empty()) out.push_back(' ');
            pending_space = false;
            out.push_back((char) std::tolower(c));
        } else {
            pending_space = true;
        }
    }
    return out;
}

bool same_tokens(const EngineResult & a, const EngineResult & b) {
    return a.token_ids == b.token_ids && a.text == b.text;
}

int check_stage_times(const EngineResult & r);

void print_stage_times(const char * label, const EngineResult & r) {
    std::printf("%s: audio %.1f s, mel %.1f ms, encoder %.1f ms, decoder %.1f ms, "
                "segment plan %.1f ms, assemble %.1f ms, total %.1f ms, encoder frames %d, "
                "encoder on Core ML %d\n",
                label, (double) r.audio_samples / r.sample_rate, r.preprocess_ms, r.encoder_ms,
                r.decode_ms, r.segment_plan_ms, r.assemble_ms, r.total_ms, r.encoder_frames,
                (int) r.encoder_used_coreml);
}

int check_expected_transcript(Engine & engine, const Opts & o, EngineResult & jfk) {
    jfk = engine.transcribe(o.wav_path);
    if (o.verbose) {
        std::printf("jfk: %s\n", jfk.text.c_str());
        print_stage_times("jfk", jfk);
    }
    if (normalized_words(jfk.text) != kExpectedJfkWords) {
        std::fprintf(stderr, "FAIL: unexpected transcript: %s\n", jfk.text.c_str());
        return 1;
    }
    return 0;
}

int check_precomputed_mel_parity(Engine & engine, const Opts & o, const EngineResult & jfk) {
    const EngineResult via_mel = engine.transcribe_mel(engine.compute_mel_wav(o.wav_path));
    if (!same_tokens(jfk, via_mel)) {
        std::fprintf(stderr, "FAIL: transcribe_mel differs from transcribe\n  %s\n  %s\n",
                     jfk.text.c_str(), via_mel.text.c_str());
        return 1;
    }
    if (via_mel.mel_frames != jfk.mel_frames || via_mel.audio_samples != jfk.audio_samples) {
        std::fprintf(stderr, "FAIL: transcribe_mel reports a different shape\n");
        return 1;
    }
    return 0;
}

int check_concurrent_mel(Engine & engine, const Opts & o, const EngineResult & jfk) {
    const EngineResult sequential = engine.transcribe(o.wav2_path);

    std::future<PrecomputedMel> next = std::async(std::launch::async,
        [&] { return engine.compute_mel_wav(o.wav2_path); });
    const EngineResult jfk_again = engine.transcribe(o.wav_path);
    const EngineResult overlapped = engine.transcribe_mel(next.get());

    if (!same_tokens(jfk, jfk_again)) {
        std::fprintf(stderr, "FAIL: a concurrent compute_mel_wav changed the transcription\n");
        return 1;
    }
    if (!same_tokens(sequential, overlapped)) {
        std::fprintf(stderr, "FAIL: a mel computed on a worker thread decodes differently\n  %s\n  %s\n",
                     sequential.text.c_str(), overlapped.text.c_str());
        return 1;
    }
    if (o.verbose) std::printf("wav2: %s\n", overlapped.text.c_str());
    return 0;
}

int expect_mel_rejected(Engine & engine, PrecomputedMel mel, const char * what) {
    try {
        engine.transcribe_mel(std::move(mel));
    } catch (const std::exception &) {
        return 0;
    }
    std::fprintf(stderr, "FAIL: transcribe_mel accepted %s\n", what);
    return 1;
}

// A PrecomputedMel is a public struct: every inconsistency between its
// declared shape and its buffer must be refused before the encoder reads it.
int check_mel_rejections(Engine & engine, const Opts & o, const EngineResult & jfk) {
    const PrecomputedMel good = engine.compute_mel_wav(o.wav_path);

    PrecomputedMel empty_buffer = good;
    empty_buffer.mel.clear();
    PrecomputedMel truncated = good;
    truncated.mel.pop_back();
    PrecomputedMel oversized = good;
    oversized.mel.push_back(0.0f);
    PrecomputedMel no_frames = good;
    no_frames.n_frames = 0;
    PrecomputedMel wrong_bins = good;
    wrong_bins.n_mels += 1;
    PrecomputedMel absurd_frames = good;
    absurd_frames.n_frames = std::numeric_limits<int>::max();

    int rc = 0;
    rc |= expect_mel_rejected(engine, empty_buffer, "an empty buffer with positive n_frames");
    rc |= expect_mel_rejected(engine, truncated, "a buffer one value short");
    rc |= expect_mel_rejected(engine, oversized, "a buffer one value long");
    rc |= expect_mel_rejected(engine, no_frames, "n_frames == 0");
    rc |= expect_mel_rejected(engine, wrong_bins, "a bin count the model does not have");
    rc |= expect_mel_rejected(engine, absurd_frames, "an INT_MAX frame count");
    if (rc != 0) return rc;

    const EngineResult intact = engine.transcribe_mel(good);
    if (!same_tokens(jfk, intact)) {
        std::fprintf(stderr, "FAIL: an intact PrecomputedMel no longer transcribes like transcribe()\n");
        return 1;
    }
    return 0;
}

EngineResult transcribe_tiled(const Opts & o, int window_frames, int context_frames,
                              const std::vector<float> & samples, int sample_rate) {
    EngineOptions eo;
    eo.model_gguf_path          = o.model_path;
    eo.n_threads                = o.n_threads;
    eo.n_gpu_layers             = o.n_gpu_layers;
    eo.long_form_window_frames  = window_frames;
    eo.long_form_context_frames = context_frames;
    Engine engine(eo);
    return engine.transcribe_samples(samples.data(), (int) samples.size(), sample_rate);
}

std::vector<float> concat_tiles(const std::vector<float> & a, const std::vector<float> & b, int tiles) {
    std::vector<float> out;
    out.reserve((a.size() + b.size()) * (size_t) tiles);
    for (int i = 0; i < tiles; ++i) {
        out.insert(out.end(), a.begin(), a.end());
        out.insert(out.end(), b.begin(), b.end());
    }
    return out;
}

std::string expected_long_english_words(int tiles) {
    std::string out;
    for (int i = 0; i < tiles; ++i) {
        if (!out.empty()) out.push_back(' ');
        out += kExpectedJfkWords;
        out.push_back(' ');
        out += kExpectedSampleWords;
    }
    return out;
}

int check_long_accuracy(const char * label, const std::string & text, const std::string & expected) {
    const double sim = word_similarity(normalized_words(text), expected);
    std::printf("long English, %s: word similarity to the expected words = %.4f (min %.4f)\n",
                label, sim, kMinLongAccuracy);
    if (sim < kMinLongAccuracy) {
        std::fprintf(stderr, "FAIL: %s loses accuracy on the long English recording:\n%s\n",
                     label, text.c_str());
        return 1;
    }
    return 0;
}

// TranscribePsy's GGUF is recognised by the loader, so the default options
// plan its validated 60 s / 25 % windows on the ggml paths and transcribe
// exactly like the explicit request; the Core ML segments path is its own.
int check_defaults_take_validated_windows(const EngineResult & defaults, const EngineResult & windowed) {
    if (defaults.encoder_used_coreml) return 0;
    if (!same_tokens(defaults, windowed)) {
        std::fprintf(stderr, "FAIL: the default options did not take TranscribePsy's validated windows\n");
        return 1;
    }
    return 0;
}

// The same recording must transcribe to the same words on every backend: the
// default options (one pass, or the Core ML segments of the encoder-less GGUF)
// and the 60 s / 25 % windows are each held to kMinLongAccuracy against the
// clips' known words, so a backend that drops words at its seams fails here.
int check_long_english_accuracy(const Opts & o) {
    std::vector<float> jfk, sample;
    int jfk_rate = 0, sample_rate = 0;
    if (!parakeet_test::load_wav_pcm16le_mono(o.wav_path, jfk, jfk_rate) ||
        !parakeet_test::load_wav_pcm16le_mono(o.wav_sample_path, sample, sample_rate) ||
        jfk_rate != sample_rate) {
        std::fprintf(stderr, "failed to load %s and %s at one sample rate\n",
                     o.wav_path.c_str(), o.wav_sample_path.c_str());
        return 1;
    }
    const std::vector<float> samples  = concat_tiles(jfk, sample, kLongEnglishTiles);
    const std::string        expected = expected_long_english_words(kLongEnglishTiles);

    const EngineResult defaults = transcribe_tiled(o, 0, 0, samples, sample_rate);
    const EngineResult windowed = transcribe_tiled(o, kPhoneWindowFrames, kPhoneContextFrames,
                                                   samples, sample_rate);
    if (o.verbose) {
        std::printf("--- long English, default options ---\n%s\n--- long English, 60 s / 25 %% windows ---\n%s\n",
                    defaults.text.c_str(), windowed.text.c_str());
    }
    int rc = 0;
    rc |= check_long_accuracy("default options", defaults.text, expected);
    rc |= check_long_accuracy("60 s / 25 % windows", windowed.text, expected);
    rc |= check_defaults_take_validated_windows(defaults, windowed);
    return rc;
}

int check_phone_windows(const Opts & o) {
    std::vector<float> base;
    int sample_rate = 0;
    if (!parakeet_test::load_wav_pcm16le_mono(o.wav_path, base, sample_rate)) {
        std::fprintf(stderr, "failed to load wav: %s\n", o.wav_path.c_str());
        return 1;
    }
    const std::vector<float> samples = tile_to_seconds(base, sample_rate, kTiledSeconds);

    const std::string single   = transcribe_tiled(o, -1, 0, samples, sample_rate).text;
    const std::string windowed = transcribe_tiled(o, kPhoneWindowFrames, kPhoneContextFrames,
                                                  samples, sample_rate).text;
    if (o.verbose) {
        std::printf("--- single pass ---\n%s\n--- 60 s / 25 %% windows ---\n%s\n",
                    single.c_str(), windowed.c_str());
    }
    if (single.empty() || windowed.empty()) {
        std::fprintf(stderr, "FAIL: empty long-form transcription\n");
        return 1;
    }
    const double sim = word_similarity(single, windowed);
    std::printf("word similarity 60 s / 25 %% windows vs single pass = %.4f (min %.4f)\n",
                sim, o.min_similarity);
    if (sim < o.min_similarity) {
        std::fprintf(stderr, "FAIL: windowed output diverges from the single pass\n");
        return 1;
    }
    return 0;
}

int transcribe_long_recording(Engine & engine, const Opts & o) {
    if (o.long_wav_path.empty()) return 0;
    const EngineResult r = engine.transcribe(o.long_wav_path);
    print_stage_times("long", r);
    std::printf("long: %s\n", r.text.c_str());
    if (r.text.empty()) {
        std::fprintf(stderr, "FAIL: empty transcript for %s\n", o.long_wav_path.c_str());
        return 1;
    }
    return check_stage_times(r);
}

int check_stage_times(const EngineResult & r) {
    const double stages = r.preprocess_ms + r.encoder_ms + r.decode_ms +
                          r.segment_plan_ms + r.assemble_ms;
    if (r.total_ms + kTimingSlackMs < stages) {
        std::fprintf(stderr, "FAIL: stage times (%.2f ms) exceed total_ms (%.2f ms)\n",
                     stages, r.total_ms);
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    Opts opts;
    if (int rc = parse_args(argc, argv, opts); rc != 0) return rc;

    try {
        EngineOptions eo;
        eo.model_gguf_path = opts.model_path;
        eo.n_threads       = opts.n_threads;
        eo.n_gpu_layers    = opts.n_gpu_layers;
        Engine engine(eo);

        EngineResult jfk;
        if (check_expected_transcript(engine, opts, jfk) != 0) return 1;
        if (check_stage_times(jfk) != 0) return 1;
        if (check_precomputed_mel_parity(engine, opts, jfk) != 0) return 1;
        if (check_concurrent_mel(engine, opts, jfk) != 0) return 1;
        if (check_mel_rejections(engine, opts, jfk) != 0) return 1;
        if (check_phone_windows(opts) != 0) return 1;
        if (check_long_english_accuracy(opts) != 0) return 1;
        if (transcribe_long_recording(engine, opts) != 0) return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return 1;
    }

    std::printf("test-transcribepsy: OK\n");
    return 0;
}
