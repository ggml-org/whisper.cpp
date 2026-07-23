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

}  // namespace

int main() {
    const int64_t N_MELS = 128;

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

    if (g_failures == 0) {
        std::printf("[coreml-shapes] PASS\n");
        return 0;
    }
    std::printf("[coreml-shapes] FAIL: %d case(s)\n", g_failures);
    return 1;
}
