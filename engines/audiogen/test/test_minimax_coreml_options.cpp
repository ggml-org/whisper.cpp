#include "audiogen-cpp/minimax/engine.h"
#include "minimax/coreml-options.h"
#include "minimax_tiny_model.h"

#ifdef AUDIOGEN_USE_MINIMAX_COREML
#include "minimax/ane-validation.h"
#endif

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
using tts_cpp::minimax::Engine;
using tts_cpp::minimax::EngineOptions;
int failures = 0;

void check(bool condition, const char * expression) {
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL %s\n", expression);
    }
}
#define CHECK(condition) check((condition), #condition)

template <typename Function> void rejects(Function function, const std::string & expected) {
    try {
        function();
        CHECK(false);
    } catch (const std::exception & error) {
        CHECK(std::string(error.what()).find(expected) != std::string::npos);
    }
}

void test_explicit_pair() {
    EngineOptions options;
    tts_cpp::minimax::detail::validate_coreml_options(options);
    CHECK(!tts_cpp::minimax::detail::coreml_requested(options));
    options.coreml_profile_dir = "profile";
    rejects([&] { Engine::create(options); }, "requires both");
    options.coreml_profile_dir.clear();
    options.coreml_compute_units = "CPU_AND_NE";
    rejects([&] { Engine::create(options); }, "requires both");
    options.coreml_profile_dir = "profile";
    options.coreml_compute_units = "GPU_ONLY";
    rejects([&] { Engine::create(options); }, "coreml_compute_units must be");
    options.coreml_compute_units = "CPU_ONLY";
    options.coreml_profile_dir = std::string("profile\0other", 13);
    rejects([&] { Engine::create(options); }, "null character");
}

void test_allowed_units() {
    for (const char * units : {"CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"}) {
        EngineOptions options;
        options.coreml_profile_dir = "profile";
        options.coreml_compute_units = units;
        tts_cpp::minimax::detail::validate_coreml_options(options);
#ifndef AUDIOGEN_USE_MINIMAX_COREML
        rejects([&] { Engine::create(options); }, "support was not built");
#endif
    }
}

#ifdef AUDIOGEN_USE_MINIMAX_COREML
void test_manifest_failures() {
    if (!mm3_ane_validation::available())
        return;
    minimax_tiny::Pair pair("coreml-options");
    EngineOptions options;
    options.lm_model_path = pair.lm.string();
    options.synth_model_path = pair.synth.string();
    options.coreml_profile_dir = (pair.directory / "profile").string();
    options.coreml_compute_units = "CPU_ONLY";
    std::filesystem::create_directory(options.coreml_profile_dir);
    rejects([&] { Engine::create(options); }, "manifest");
    const auto manifest = std::filesystem::path(options.coreml_profile_dir) / "manifest.json";
    for (const char * invalid : {"[]", "{", "{\"format\":true}", "{\"format\":2}", "{\"format\":1,\"context\":0}",
                                 "{\"format\":1,\"context\":10240,\"depth_context\":0}",
                                 "{\"format\":1,\"context\":10240,\"depth_context\":32,\"condition_"
                                 "frames\":0}",
                                 "{\"format\":1,\"context\":10240,\"depth_context\":32,\"condition_"
                                 "frames\":-1}"}) {
        std::ofstream(manifest) << invalid;
        rejects([&] { Engine::create(options); }, "Core ML");
    }
}
#endif
} // namespace

int main() {
    test_explicit_pair();
    test_allowed_units();
#ifdef AUDIOGEN_USE_MINIMAX_COREML
    test_manifest_failures();
#endif
    std::printf("MiniMax Core ML options: %d failures\n", failures);
    return failures ? 1 : 0;
}
