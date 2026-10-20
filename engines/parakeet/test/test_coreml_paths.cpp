// Unit test for the Core ML sidecar path derivation (coreml_encoder_sidecar_path).
// Pure string logic -- no model, ggml, or Apple frameworks -- so it runs on every
// platform and locks the `<model>-encoder.mlmodelc` naming. The quant tag is
// stripped (dot- or dash-separated) so one exported encoder serves every decoder
// quantisation of a model, and a dot inside a directory name is not mistaken for
// an extension.

#include "parakeet_coreml_path.h"

#include <cstdio>
#include <string>

namespace {

int g_failures = 0;

void expect(const std::string & gguf, const std::string & expected) {
    const std::string got = parakeet::coreml_encoder_sidecar_path(gguf);
    const bool ok = got == expected;
    std::printf("[%s] %-46s -> %s\n", ok ? "ok  " : "FAIL", gguf.c_str(), got.c_str());
    if (!ok) {
        std::printf("       expected: %s\n", expected.c_str());
        ++g_failures;
    }
}

void expect_bypass(const std::string & gguf, const std::string & expected) {
    const std::string got = parakeet::coreml_bypass_encoder_sidecar_path(gguf);
    const bool ok = got == expected;
    std::printf("[%s] %-46s -> %s\n", ok ? "ok  " : "FAIL", gguf.c_str(), got.c_str());
    if (!ok) {
        std::printf("       expected: %s\n", expected.c_str());
        ++g_failures;
    }
}

void expect_decoder(const std::string & gguf, const std::string & expected) {
    const std::string got = parakeet::coreml_decoder_sidecar_path(gguf);
    const bool ok = got == expected;
    std::printf("[%s] %-46s -> %s\n", ok ? "ok  " : "FAIL", gguf.c_str(), got.c_str());
    if (!ok) {
        std::printf("       expected: %s\n", expected.c_str());
        ++g_failures;
    }
}

}  // namespace

int main() {
    expect("models/parakeet-tdt-0.6b-v3.f16.gguf",  "models/parakeet-tdt-0.6b-v3-encoder.mlmodelc");
    expect("models/parakeet-tdt-0.6b-v3.bf16.gguf", "models/parakeet-tdt-0.6b-v3-encoder.mlmodelc");
    expect("models/parakeet-tdt-0.6b-v3.q8_0.gguf", "models/parakeet-tdt-0.6b-v3-encoder.mlmodelc");
    expect("models/parakeet-unified-en-0.6b.q8_0.gguf",
           "models/parakeet-unified-en-0.6b-encoder.mlmodelc");
    expect("models/parakeet-unified-en-0.6b.f16.gguf",
           "models/parakeet-unified-en-0.6b-encoder.mlmodelc");
    expect("parakeet-ctc-0.6b.q8_0.gguf",           "parakeet-ctc-0.6b-encoder.mlmodelc");
    expect("/opt/models/parakeet-tdt-0.6b-v3.gguf", "/opt/models/parakeet-tdt-0.6b-v3-encoder.mlmodelc");
    expect("model-q8_0.gguf",                       "model-encoder.mlmodelc");
    expect("model.f32.gguf",                        "model-encoder.mlmodelc");
    expect("model.q5_0.gguf",                       "model-encoder.mlmodelc");
    expect("model.q4_0.gguf",                       "model-encoder.mlmodelc");
    expect("/x.y/model.gguf",                       "/x.y/model-encoder.mlmodelc");
    expect("models/diar_streaming_sortformer_4spk-v2.1.f16.gguf",
           "models/diar_streaming_sortformer_4spk-v2.1-encoder.mlmodelc");
    expect_bypass("models/diar_streaming_sortformer_4spk-v2.1.q8_0.gguf",
                  "models/diar_streaming_sortformer_4spk-v2.1-encoder-bypass-pre-encode.mlmodelc");

    // The TranscribePsy Core ML distribution: both sidecars hang off the same stem.
    expect("models/transcribepsy-250m-eu-sub-coreml.gguf",
           "models/transcribepsy-250m-eu-sub-coreml-encoder.mlmodelc");
    expect_decoder("models/transcribepsy-250m-eu-sub-coreml.gguf",
                   "models/transcribepsy-250m-eu-sub-coreml-decoder.mlmodelc");
    expect_decoder("models/transcribepsy-250m-eu-sub.q8_0.gguf",
                   "models/transcribepsy-250m-eu-sub-decoder.mlmodelc");
    expect_decoder("/opt/models/parakeet-tdt-0.6b-v3.gguf",
                   "/opt/models/parakeet-tdt-0.6b-v3-decoder.mlmodelc");

    if (g_failures == 0) {
        std::printf("[coreml-paths] PASS\n");
        return 0;
    }
    std::printf("[coreml-paths] FAIL: %d case(s)\n", g_failures);
    return 1;
}
