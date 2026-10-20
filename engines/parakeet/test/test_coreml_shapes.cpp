// Unit test for the Core ML input-shape resolution (coreml_resolve_input_dims /
// coreml_match_trailing_dims). Pure integer logic -- no model, ggml, or Apple
// frameworks -- so it runs on every platform and locks the wrapper's variable-length
// behaviour: a fixed-shape model at its trace length is honoured verbatim, a flexible
// export is rebuilt at the requested mel length in the model's declared orientation,
// and an unknown/non-concrete declared shape falls back to features-major.

#include "coreml/parakeet_coreml_shape.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

std::string to_str(const std::vector<int64_t> & v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        s += std::to_string(v[i]);
        if (i + 1 < v.size()) s += ",";
    }
    return s + "]";
}

void expect(const std::string & name,
            const std::vector<int64_t> & declared,
            int64_t n_mel_frames, int64_t n_mels,
            const std::vector<int64_t> & want_dims, bool want_transpose) {
    const parakeet::CoremlInputDims got =
        parakeet::coreml_resolve_input_dims(declared, n_mel_frames, n_mels);
    const bool ok = got.dims == want_dims && got.transpose == want_transpose;
    std::printf("[%s] %-34s -> dims=%s transpose=%d\n",
                ok ? "ok  " : "FAIL", name.c_str(), to_str(got.dims).c_str(), got.transpose);
    if (!ok) {
        std::printf("       expected: dims=%s transpose=%d\n", to_str(want_dims).c_str(), want_transpose);
        ++g_failures;
    }
}

void expect_capacity(const std::string & name,
                     const std::vector<int64_t> & dims,
                     int64_t required_rows, int64_t cols,
                     bool want_matched, bool want_transpose) {
    const parakeet::CoremlTrailingMatch got =
        parakeet::coreml_match_trailing_capacity(dims, required_rows, cols);
    const bool ok = got.matched == want_matched &&
                    got.transpose == want_transpose;
    std::printf("[%s] %-34s -> matched=%d transpose=%d\n",
                ok ? "ok  " : "FAIL", name.c_str(), got.matched, got.transpose);
    if (!ok) {
        std::printf("       expected: matched=%d transpose=%d\n",
                    want_matched, want_transpose);
        ++g_failures;
    }
}

}  // namespace

void expect_frames(const std::string & name, int64_t mel_frames, int64_t want) {
    const int64_t got = parakeet::coreml_offline_encoder_frames(mel_frames);
    const bool ok = got == want;
    std::printf("[%s] %-34s -> %lld encoder frames\n", ok ? "ok  " : "FAIL", name.c_str(), (long long) got);
    if (!ok) {
        std::printf("       expected: %lld\n", (long long) want);
        ++g_failures;
    }
}

void expect_valid(const std::string & name, int64_t n_mel_frames, int64_t n_mel_valid,
                  int64_t n_enc_frames, int64_t want_mel, int64_t want_enc) {
    const parakeet::CoremlAneValidity got =
        parakeet::coreml_ane_valid_lengths(n_mel_frames, n_mel_valid, n_enc_frames);
    const bool ok = got.mel_frames == want_mel && got.enc_frames == want_enc;
    std::printf("[%s] %-34s -> mel=%lld enc=%lld\n", ok ? "ok  " : "FAIL", name.c_str(),
                (long long) got.mel_frames, (long long) got.enc_frames);
    if (!ok) {
        std::printf("       expected: mel=%lld enc=%lld\n", (long long) want_mel, (long long) want_enc);
        ++g_failures;
    }
}

// The fixed-window export's masks must follow the frames that hold audio: the
// mel front-end's zeroed trailing padding frame is padding for the masks too.
void check_ane_valid_lengths() {
    expect_frames("1500 mel -> 188 frames", 1500, 188);
    expect_frames("1101 mel -> 138 frames", 1101, 138);
    expect_frames("1 mel -> 1 frame",       1,    1);
    expect_frames("0 mel -> 0 frames",      0,    0);

    expect_valid("whole slab valid",          1500, 1500, 188, 1500, 188);
    expect_valid("valid unknown (0)",         1500, 0,    188, 1500, 188);
    expect_valid("valid beyond the slab",     1500, 2000, 188, 1500, 188);
    expect_valid("one zeroed trailing frame", 1101, 1100, 138, 1100, 138);
    expect_valid("short audio in the window", 1500, 1200, 188, 1200, 150);
    expect_valid("toy slab 37 valid 36",      37,   36,   5,   36,   5);
    expect_valid("enc capped by the caller",  1500, 1500, 100, 1500, 100);
}

int main() {
    const int64_t N_MELS = 128;

    check_ane_valid_lengths();

    // Fixed model already sized to the input -> honoured verbatim (no rebuild).
    expect("fixed features-major exact", {1, N_MELS, 1101}, 1101, N_MELS, {1, N_MELS, 1101}, true);
    expect("fixed time-major exact",     {1, 1101, N_MELS}, 1101, N_MELS, {1, 1101, N_MELS}, false);
    expect("2d features-major exact",    {N_MELS, 1101},    1101, N_MELS, {N_MELS, 1101},    true);

    // Flexible model (declared shape reports one length): rebuild at the requested
    // length, preserving the declared orientation and leading dims.
    expect("flex features-major rebuild", {1, N_MELS, 1101}, 2014, N_MELS, {1, N_MELS, 2014}, true);
    expect("flex time-major rebuild",     {1, 1101, N_MELS}, 2014, N_MELS, {1, 2014, N_MELS}, false);
    expect("flex 2d rebuild shorter",     {N_MELS, 1101},    326,  N_MELS, {N_MELS, 326},     true);

    // No usable concrete shape -> NeMo-natural features-major default.
    expect("empty declared default",   {},              900, N_MELS, {1, N_MELS, 900}, true);
    expect("non-concrete (dyn) default", {-1, N_MELS},   900, N_MELS, {1, N_MELS, 900}, true);
    expect("rank-1 default",           {N_MELS},         900, N_MELS, {1, N_MELS, 900}, true);
    // Neither trailing axis is the feature dim -> default rather than a bad rebuild.
    expect("no feature axis default",  {1, 7, 9},        900, N_MELS, {1, N_MELS, 900}, true);

    // Fixed output backing may have more time rows than the current (unpadded)
    // utterance needs. Both time-major and features-major layouts are accepted,
    // but a short time axis, wrong feature width, or rank below two is rejected.
    expect_capacity("capacity time-major exact", {1, 188, 1024}, 188, 1024, true, false);
    expect_capacity("capacity time-major padded", {1, 188, 1024}, 100, 1024, true, false);
    expect_capacity("capacity features-major", {1, 1024, 188}, 100, 1024, true, true);
    expect_capacity("capacity too short", {1, 99, 1024}, 100, 1024, false, false);
    expect_capacity("capacity wrong width", {1, 188, 512}, 100, 1024, false, false);
    expect_capacity("capacity rank one", {1024}, 100, 1024, false, false);

    if (g_failures == 0) {
        std::printf("[coreml-shapes] PASS\n");
        return 0;
    }
    std::printf("[coreml-shapes] FAIL: %d case(s)\n", g_failures);
    return 1;
}
