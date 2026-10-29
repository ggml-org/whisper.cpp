#include "gguf_fixtures.h"

#include "tts-cpp/moss/engine.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

using namespace moss_fixtures;

namespace {

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

tts_cpp::moss::EngineOptions fixture_options(const std::filesystem::path & backbone,
                                             const std::filesystem::path & decoder) {
    tts_cpp::moss::EngineOptions options;
    options.backbone_path = backbone.string();
    options.decoder_path = decoder.string();
    options.n_threads = 1;
    options.context = 4096;
    options.max_new_tokens = 3800;
    options.text_temperature = 0.0f;
    options.audio_temperature = 0.0f;
    return options;
}

void test_concurrent_cancel(tts_cpp::moss::Engine & engine) {
    std::thread canceller([&engine]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        engine.cancel();
    });
    const tts_cpp::moss::SynthesisResult result = engine.synthesize("hi");
    canceller.join();
    check(result.cancelled, "concurrent cancel() is reported in the result");
    check(result.pcm.empty(), "a cancelled synthesis returns no PCM");
}

void test_engine_recovers_after_cancel(tts_cpp::moss::Engine & engine) {
    const tts_cpp::moss::SynthesisResult result = engine.synthesize("hi");
    check(!result.cancelled, "the next request is not cancelled");
    check(!result.pcm.empty(), "the next request produces PCM");
}

} // namespace

int main() {
    const auto backbone = write_generating_backbone("cancel-backbone");
    const auto decoder = write_decoder("cancel-decoder", N_VQ, 3 * D_MODEL,
            [](gguf_context *) {});
    int rc = 0;
    try {
        tts_cpp::moss::Engine engine(fixture_options(backbone, decoder));
        test_concurrent_cancel(engine);
        test_engine_recovers_after_cancel(engine);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        rc = 1;
    }
    std::filesystem::remove(backbone);
    std::filesystem::remove(decoder);
    if (rc != 0) {
        return rc;
    }
    if (failures == 0) {
        std::printf("moss cancellation: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss cancellation: %d failures\n", failures);
    return 1;
}
