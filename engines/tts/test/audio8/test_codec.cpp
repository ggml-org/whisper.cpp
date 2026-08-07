// Audio8 codec parity against the PyTorch fixtures, both directions.
//
// The reference dump captures the boundary between every stage, so this walks
// the same boundaries. Decoding: the two quantizer banks, the windowed post
// transformer, the upsampled latent and finally the waveform. Encoding: the
// convolutional stack, the downsampled latent and the codes the residual
// quantizer picks, which have to match exactly -- a single different code is a
// different voice.
//
// The fixtures are torch [channels, length]; the engine holds the same values
// channels-inner, so the comparison transposes as it reads.

#include "audio8/internal.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace tts_cpp::audio8::detail;

namespace {

// The engine and the reference differ only in float reassociation. The latent
// stages carry activations of order 10 and land within 3e-5 of the reference;
// the waveform is bounded by tanh and lands within 2e-6. Both bars leave an
// order of magnitude for thread-count changes to reorder a reduction.
constexpr double LATENT_TOLERANCE = 5e-4;
constexpr double WAVEFORM_TOLERANCE = 5e-5;

struct fixture {
    std::string dir;

    npy_array load(const std::string & name) const {
        return npy_load(dir + "/codec_" + name + ".npy");
    }
};

std::vector<int32_t> to_i32(const npy_array & array) {
    std::vector<int32_t> out(array.n_elements());
    if (array.dtype == "<i8") {
        const int64_t * source = reinterpret_cast<const int64_t *>(array.data.data());
        for (size_t index = 0; index < out.size(); ++index) {
            out[index] = static_cast<int32_t>(source[index]);
        }
        return out;
    }
    const int32_t * source = reinterpret_cast<const int32_t *>(array.data.data());
    out.assign(source, source + out.size());
    return out;
}

const float * as_f32(const npy_array & array) {
    return reinterpret_cast<const float *>(array.data.data());
}

double megabytes(ggml_gallocr_t allocr) {
    return ggml_gallocr_get_buffer_size(allocr, 0) / (1024.0 * 1024.0);
}

// The blocked half should stay flat as the reference gets longer while the
// whole-sequence half grows with it, so the two are worth seeing apart.
void report_scratch(const char * tag, const codec_model & model) {
    std::printf("%s scratch %.0f MB: %.0f whole-sequence, %.0f per block\n", tag,
                megabytes(model.allocr) + megabytes(model.block_allocr),
                megabytes(model.allocr), megabytes(model.block_allocr));
}

bool report(const char * tag, const compare_stats & stats, double tolerance) {
    print_compare(tag, stats);
    if (!std::isfinite(stats.max_abs_err)) {
        std::fprintf(stderr, "%s: FAIL non-finite values\n", tag);
        return false;
    }
    if (stats.max_abs_err > tolerance) {
        std::fprintf(stderr, "%s: FAIL max|delta| %.3e > %.1e\n", tag, stats.max_abs_err,
                     tolerance);
        return false;
    }
    return true;
}

bool check_flat(const char * tag, const std::vector<float> & got, const npy_array & want,
                double tolerance) {
    if (got.size() != want.n_elements()) {
        std::fprintf(stderr, "%s: FAIL %zu values, reference has %zu\n", tag, got.size(),
                     want.n_elements());
        return false;
    }
    return report(tag, compare_f32(got.data(), as_f32(want), got.size()), tolerance);
}

// The reference array is [channels, length]; `got` holds the same values with
// channels inner-most.
bool check_transposed(const char * tag, const std::vector<float> & got,
                      const npy_array & want, double tolerance) {
    if (got.size() != want.n_elements()) {
        std::fprintf(stderr, "%s: FAIL %zu values, reference has %zu\n", tag, got.size(),
                     want.n_elements());
        return false;
    }
    const size_t channels = want.shape[0];
    const size_t length = want.shape[1];
    std::vector<float> expected(got.size());
    for (size_t channel = 0; channel < channels; ++channel) {
        for (size_t step = 0; step < length; ++step) {
            expected[step * channels + channel] =
                as_f32(want)[channel * length + step];
        }
    }
    return report(tag, compare_f32(got.data(), expected.data(), got.size()), tolerance);
}

bool check_stages(const decode_taps & taps, const fixture & data) {
    bool ok = check_transposed("semantic", taps.semantic, data.load("semantic"),
                               LATENT_TOLERANCE);
    ok &= check_transposed("residual", taps.residual, data.load("residual"),
                           LATENT_TOLERANCE);
    ok &= check_transposed("post", taps.post, data.load("post"), LATENT_TOLERANCE);
    ok &= check_transposed("latent", taps.latent, data.load("latent"), LATENT_TOLERANCE);
    return ok;
}

// Both directions run their sample-rate stack in blocks, and both stacks are
// causal, so each block carries the history its convolutions reach back over
// and a narrower block cannot change a single output. Re-running at a block
// size the fixtures actually have to split over is what proves the carried
// context is long enough; the reference comparisons above all fit in one
// block and would not notice.
constexpr int NARROW_SYNTHESIS_FRAMES = 4;
constexpr int NARROW_ANALYSIS_COLUMNS = 16;

bool check_blocked_decode(codec_model & model, const std::vector<int32_t> & codes,
                          int n_frames, int n_threads, const std::vector<float> & whole) {
    const int restore = model.synthesis_block_frames;
    model.synthesis_block_frames = NARROW_SYNTHESIS_FRAMES;
    std::vector<float> blocked;
    std::string error;
    const bool ran =
        decode_codes(model, codes.data(), n_frames, n_threads, nullptr, blocked, &error);
    model.synthesis_block_frames = restore;
    if (!ran) {
        std::fprintf(stderr, "blocked decode: %s\n", error.c_str());
        return false;
    }
    if (blocked.size() != whole.size()) {
        std::fprintf(stderr, "blocked: FAIL %zu samples against %zu\n", blocked.size(),
                     whole.size());
        return false;
    }
    return report("blocked", compare_f32(blocked.data(), whole.data(), whole.size()),
                  WAVEFORM_TOLERANCE);
}

bool check_blocked_encode(codec_model & model, const npy_array & audio, int n_threads,
                          const std::vector<int32_t> & whole) {
    const int restore = model.analysis_block_columns;
    model.analysis_block_columns = NARROW_ANALYSIS_COLUMNS;
    std::vector<int32_t> blocked;
    int n_frames = 0;
    std::string error;
    const bool ran = encode_audio(model, as_f32(audio),
                                  static_cast<int>(audio.n_elements()), n_threads, nullptr,
                                  blocked, n_frames, &error);
    model.analysis_block_columns = restore;
    if (!ran) {
        std::fprintf(stderr, "blocked encode: %s\n", error.c_str());
        return false;
    }
    const size_t mismatches = blocked == whole ? 0 : 1;
    std::fprintf(stderr, "  [blocked] n=%zu  mismatches=%zu\n", whole.size(), mismatches);
    return mismatches == 0;
}

// Cancellation is only observable between blocks, so both directions run at
// the narrow block size and stop after the first one. Checking how much came
// back is what separates a real stop from a pass that ran to the end and
// reported the cancel afterwards.
constexpr int BLOCKS_BEFORE_CANCEL = 1;

cancel_hook cancel_after(int blocks) {
    auto seen = std::make_shared<int>(0);
    return [seen, blocks] { return (*seen)++ >= blocks; };
}

bool report_cancel(const char * tag, bool ran, const std::string & error, size_t produced,
                   size_t expected) {
    if (ran) {
        std::fprintf(stderr, "%s: FAIL ran to completion despite the cancel\n", tag);
        return false;
    }
    if (error != CANCELLED) {
        std::fprintf(stderr, "%s: FAIL stopped with '%s', not the cancellation\n", tag,
                     error.c_str());
        return false;
    }
    if (produced != expected) {
        std::fprintf(stderr, "%s: FAIL kept %zu values, expected the %zu of one block\n",
                     tag, produced, expected);
        return false;
    }
    std::fprintf(stderr, "  [%s] stopped after %d block, %zu values\n", tag,
                 BLOCKS_BEFORE_CANCEL, produced);
    return true;
}

bool check_cancelled_decode(codec_model & model, const std::vector<int32_t> & codes,
                            int n_frames, int n_threads) {
    const int restore = model.synthesis_block_frames;
    model.synthesis_block_frames = NARROW_SYNTHESIS_FRAMES;
    std::vector<float> pcm;
    std::string error;
    const bool ran = decode_codes(model, codes.data(), n_frames, n_threads,
                                  cancel_after(BLOCKS_BEFORE_CANCEL), pcm, &error);
    model.synthesis_block_frames = restore;
    const size_t one_block = static_cast<size_t>(BLOCKS_BEFORE_CANCEL) *
                             NARROW_SYNTHESIS_FRAMES * model.hp.frame_size;
    return report_cancel("cancelled decode", ran, error, pcm.size(), one_block);
}

// The encoder's blocks feed the analysis graph rather than the caller, so a
// cancel leaves nothing behind at all.
bool check_cancelled_encode(codec_model & model, const npy_array & audio, int n_threads) {
    const int restore = model.analysis_block_columns;
    model.analysis_block_columns = NARROW_ANALYSIS_COLUMNS;
    std::vector<int32_t> codes;
    int n_frames = 0;
    std::string error;
    const bool ran =
        encode_audio(model, as_f32(audio), static_cast<int>(audio.n_elements()), n_threads,
                     cancel_after(BLOCKS_BEFORE_CANCEL), codes, n_frames, &error);
    model.analysis_block_columns = restore;
    return report_cancel("cancelled encode", ran, error, codes.size(), 0);
}

bool run_decode(codec_model & model, const fixture & data, int n_threads) {
    const npy_array codes = data.load("codes");
    const std::vector<int32_t> values = to_i32(codes);
    const int n_frames = static_cast<int>(codes.shape[1]);

    std::vector<float> pcm;
    decode_taps taps;
    std::string error;
    if (!decode_codes(model, values.data(), n_frames, n_threads, nullptr, pcm, &error,
                      &taps)) {
        std::fprintf(stderr, "decode: %s\n", error.c_str());
        return false;
    }
    std::printf("decoded %d frames into %zu samples\n", n_frames, pcm.size());
    report_scratch("decode", model);

    bool ok = check_stages(taps, data);
    ok &= check_flat("waveform", pcm, data.load("wav"), WAVEFORM_TOLERANCE);
    ok &= check_blocked_decode(model, values, n_frames, n_threads, pcm);
    ok &= check_cancelled_decode(model, values, n_frames, n_threads);
    return ok;
}

bool check_codes(const std::vector<int32_t> & got, const npy_array & want) {
    const std::vector<int32_t> expected = to_i32(want);
    const size_t books = want.shape[0];
    const size_t frames = want.shape[1];
    size_t mismatches = 0;
    for (size_t book = 0; book < books; ++book) {
        for (size_t frame = 0; frame < frames; ++frame) {
            const size_t at = book * frames + frame;
            if (got[at] == expected[at]) continue;
            if (mismatches < 8) {
                std::fprintf(stderr, "  codebook %zu frame %zu: %d != %d\n", book, frame,
                             got[at], expected[at]);
            }
            ++mismatches;
        }
    }
    std::fprintf(stderr, "  [codes] n=%zu  mismatches=%zu\n", books * frames, mismatches);
    return mismatches == 0;
}

bool run_encode(codec_model & model, const fixture & data, int n_threads) {
    const npy_array audio = data.load("audio");
    const npy_array want = data.load("enc_codes");

    std::vector<int32_t> codes;
    int n_frames = 0;
    encode_taps taps;
    std::string error;
    if (!encode_audio(model, as_f32(audio), static_cast<int>(audio.n_elements()),
                      n_threads, nullptr, codes, n_frames, &error, &taps)) {
        std::fprintf(stderr, "encode: %s\n", error.c_str());
        return false;
    }
    std::printf("encoded %zu samples into %d frames\n", audio.n_elements(), n_frames);
    report_scratch("encode", model);
    if (n_frames != static_cast<int>(want.shape[1])) {
        std::fprintf(stderr, "encode: FAIL %d frames, reference has %zu\n", n_frames,
                     want.shape[1]);
        return false;
    }

    bool ok = check_transposed("encoded", taps.encoded, data.load("encoded"),
                               LATENT_TOLERANCE);
    ok &= check_transposed("downsampled", taps.downsampled, data.load("downsampled"),
                           LATENT_TOLERANCE);
    ok &= check_codes(codes, want);
    ok &= check_blocked_encode(model, audio, n_threads, codes);
    ok &= check_cancelled_encode(model, audio, n_threads);
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <codec-decoder.gguf> <codec-encoder.gguf> <ref-dir> "
                     "[threads]\n",
                     argv[0]);
        return 1;
    }
    const fixture data{argv[3]};
    const int n_threads = argc > 4 ? std::atoi(argv[4]) : 4;

    codec_model decoder;
    std::string error;
    if (!load_codec(argv[1], decoder, &error)) {
        std::fprintf(stderr, "load decoder: %s\n", error.c_str());
        return 1;
    }
    const bool decoded = run_decode(decoder, data, n_threads);
    free_codec(decoder);

    codec_model encoder;
    if (!load_codec(argv[2], encoder, &error)) {
        std::fprintf(stderr, "load encoder: %s\n", error.c_str());
        return 1;
    }
    const bool encoded = run_encode(encoder, data, n_threads);
    free_codec(encoder);

    const bool ok = decoded && encoded;
    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
