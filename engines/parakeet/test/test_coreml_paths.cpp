// Unit test for the Core ML sidecar path derivation. Pure string logic -- no
// model, ggml, or Apple frameworks -- so it runs on every platform and locks the
// `<model>-encoder.mlmodelc` and `<model>-diarizer-<frames>.mlmodelc` naming. The
// quant tag is stripped (dot- or dash-separated) so one exported sidecar serves
// every quantisation of a model, and a dot inside a directory name is not
// mistaken for an extension. Also covers the diarizer capacity ladder and the
// fit projection of the bundles a workload loads.

#include "parakeet_coreml_path.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

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

void expect_diarizer(const std::string & gguf, int capacity, const std::string & expected) {
    const std::string got = parakeet::coreml_diarizer_sidecar_path(gguf, capacity);
    const bool ok = got == expected;
    std::printf("[%s] %-46s -> %s\n", ok ? "ok  " : "FAIL", gguf.c_str(), got.c_str());
    if (!ok) {
        std::printf("       expected: %s\n", expected.c_str());
        ++g_failures;
    }
}

void expect_capacity(const std::string & gguf, const std::string & candidate, int expected) {
    const int got = parakeet::coreml_diarizer_sidecar_capacity(gguf, candidate);
    const bool ok = got == expected;
    std::printf("[%s] %-46s -> %d\n", ok ? "ok  " : "FAIL", candidate.c_str(), got);
    if (!ok) {
        std::printf("       expected: %d\n", expected);
        ++g_failures;
    }
}

constexpr uint64_t kMiB = 1024ull * 1024;
constexpr uint64_t kRuntimeMiB = 160;
const std::vector<int> kLadder = {358, 745};
const std::vector<parakeet::CoremlDiarizerBundle> kBundles = {{358, 100 * kMiB}, {745, 120 * kMiB}};

void expect_ladder(int frames, int expected) {
    const int got = parakeet::coreml_diarizer_capacity_for(kLadder, frames);
    const bool ok = got == expected;
    std::printf("[%s] ladder for %-35d -> %d\n", ok ? "ok  " : "FAIL", frames, got);
    if (!ok) {
        std::printf("       expected: %d\n", expected);
        ++g_failures;
    }
}

void expect_resident(const char * label, const std::vector<parakeet::CoremlFrameRange> & ranges,
                     uint64_t expected_mib) {
    const uint64_t got = parakeet::coreml_diarizer_resident_bytes(kBundles, ranges);
    const bool ok = got == expected_mib * kMiB;
    std::printf("[%s] resident %-37s -> %llu MiB\n", ok ? "ok  " : "FAIL", label,
                (unsigned long long) (got / kMiB));
    if (!ok) {
        std::printf("       expected: %llu MiB\n", (unsigned long long) expected_mib);
        ++g_failures;
    }
}

void expect_resident_cases() {
    expect_resident("no calls", {}, 0);
    expect_resident("offline in the small bundle", {{341, 341}}, 2 * 100 + kRuntimeMiB);
    expect_resident("offline and live share a bundle", {{341, 341}, {1, 201}}, 2 * 100 + kRuntimeMiB);
    expect_resident("offline in the large bundle", {{400, 400}}, 2 * 120 + kRuntimeMiB);
    expect_resident("long-form blocks up to 700", {{1, 700}}, 2 * 220 + kRuntimeMiB);
    expect_resident("long-form blocks above every bundle", {{1, 900}}, 2 * 220 + kRuntimeMiB);
    expect_resident("offline above every bundle", {{900, 900}}, 0);
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
    expect("models/Nemotron-3-Diarization.converted.q4_k.gguf",
           "models/Nemotron-3-Diarization.converted-encoder.mlmodelc");
    expect("model.nvfp4.gguf",                      "model-encoder.mlmodelc");
    expect_diarizer("models/Nemotron-3-Diarization.q8_0.gguf", 358,
                    "models/Nemotron-3-Diarization-diarizer-358.mlmodelc");
    expect_capacity("models/Nemotron-3-Diarization.q6_k.gguf",
                    "models/Nemotron-3-Diarization-diarizer-745.mlmodelc", 745);
    expect_capacity("models/Nemotron-3-Diarization.q8_0.gguf",
                    "models/Nemotron-3-Diarization-diarizer-.mlmodelc", 0);
    expect_capacity("models/Nemotron-3-Diarization.q8_0.gguf",
                    "models/Nemotron-3-Diarization-diarizer-35x.mlmodelc", 0);
    expect_capacity("models/Nemotron-3-Diarization.q8_0.gguf",
                    "models/Nemotron-3-Diarization-diarizer-358.mlpackage", 0);
    expect_capacity("models/Nemotron-3-Diarization.q8_0.gguf",
                    "other/Nemotron-3-Diarization-diarizer-358.mlmodelc", 0);
    expect_capacity("models/Nemotron-3-Diarization.q8_0.gguf",
                    "models/Nemotron-3-Diarization-diarizer-99999999999.mlmodelc", 0);
    expect_capacity("Nemotron-3-Diarization.gguf",
                    "Nemotron-3-Diarization-diarizer-1125.mlmodelc", 1125);
    expect_ladder(0, 0);
    expect_ladder(1, 358);
    expect_ladder(358, 358);
    expect_ladder(359, 745);
    expect_ladder(746, 0);
    expect_resident_cases();

    if (g_failures == 0) {
        std::printf("[coreml-paths] PASS\n");
        return 0;
    }
    std::printf("[coreml-paths] FAIL: %d case(s)\n", g_failures);
    return 1;
}
