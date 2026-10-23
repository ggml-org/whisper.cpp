#include "audiogen-cpp/minimax/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char * expression) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL %s\n", expression);
}

#define CHECK(condition) check((condition), #condition)

struct StageLog {
    std::vector<std::string> names;

    bool record(const std::string & stage, int64_t, int64_t) {
        names.push_back(stage);
        return true;
    }

    bool contains(const char * stage) const {
        return std::find(names.begin(), names.end(), stage) != names.end();
    }
};

tts_cpp::minimax::EngineOptions make_engine_options(const char * models_dir) {
    tts_cpp::minimax::EngineOptions options;
    options.model_dir = models_dir;
    options.n_threads = 4;
    if (const char * profile = std::getenv("AUDIOGEN_TEST_MINIMAX_COREML_PROFILE")) {
        options.coreml_profile_dir = profile;
        const char * units = std::getenv("AUDIOGEN_TEST_MINIMAX_COREML_COMPUTE_UNITS");
        options.coreml_compute_units = units ? units : "";
    }
    // An explicit device wins over MM3_DEVICE, keeping the deterministic part
    // of the suite on the CPU even when the environment requests a GPU.
    options.device = "cpu";
    if (const char * backends_dir = std::getenv("AUDIOGEN_TEST_BACKENDS_DIR")) {
        options.backends_dir = backends_dir;
    }
    return options;
}

tts_cpp::minimax::GenerateParams make_generate_params() {
    tts_cpp::minimax::GenerateParams params;
    params.caption = "A short warm piano note.";
    params.lyrics = "[Instrumental]";
    params.max_frames = 1;
    params.seed = 7;
    params.inference_steps = 1;
    params.cfg_scale = 1.7f;
    return params;
}

bool all_samples_are_finite(const std::vector<float> & pcm) {
    return std::all_of(pcm.begin(), pcm.end(), [](float sample) {
        return std::isfinite(sample);
    });
}

void verify_result(tts_cpp::minimax::Engine & engine,
                   const tts_cpp::minimax::GenerateResult & result) {
    CHECK(!result.pcm.empty());
    CHECK(result.sample_rate == engine.sample_rate());
    CHECK(result.sample_rate > 0);
    CHECK(result.channels == 2);
    CHECK(result.pcm.size() % static_cast<size_t>(result.channels) == 0);
    CHECK(result.emitted_frames == 1);
    CHECK(result.total_ms > 0.0);
    CHECK(all_samples_are_finite(result.pcm));
}

void verify_recursive_generation(tts_cpp::minimax::Engine & engine) {
    StageLog stages;
    bool attempted = false;
    bool rejected = false;
    std::string rejection;
    const auto outer = engine.generate(
        make_generate_params(),
        [&engine, &stages, &attempted, &rejected, &rejection](
            const std::string & stage, int64_t current, int64_t total) {
            stages.record(stage, current, total);
            if (!attempted) {
                attempted = true;
                try {
                    engine.generate(make_generate_params());
                } catch (const std::logic_error & error) {
                    rejected = true;
                    rejection = error.what();
                }
            }
            return true;
        });
    CHECK(attempted);
    CHECK(rejected);
    CHECK(rejection == "minimax engine: recursive generate() is not allowed");
    verify_result(engine, outer);
    CHECK(stages.contains("ar"));
    CHECK(stages.contains("cond"));
    CHECK(stages.contains("flow"));
    CHECK(stages.contains("vocode"));
    CHECK(stages.contains("stitch"));
    CHECK(stages.contains("done"));
    const auto subsequent = engine.generate(make_generate_params());
    verify_result(engine, subsequent);
}

void verify_cancellation(tts_cpp::minimax::Engine & engine) {
    bool callback_called = false;
    const auto result = engine.generate(
        make_generate_params(),
        [&engine, &callback_called](const std::string &, int64_t, int64_t) {
            callback_called = true;
            engine.cancel();
            return false;
        });
    CHECK(callback_called);
    CHECK(result.pcm.empty());
}

// Deterministic regression for the startup race: a cancellation requested
// before generate() is entered must cancel that run immediately (no progress
// callback needed), be consumed by it, and leave the next run unaffected.
void verify_pre_armed_cancellation(tts_cpp::minimax::Engine & engine) {
    engine.cancel();
    bool progressed = false;
    const auto cancelled = engine.generate(
        make_generate_params(),
        [&progressed](const std::string &, int64_t, int64_t) {
            progressed = true;
            return true;
        });
    CHECK(cancelled.pcm.empty());
    CHECK(!progressed);
    const auto subsequent = engine.generate(make_generate_params());
    verify_result(engine, subsequent);
}

// When the environment requests a non-CPU device, run one generation with the
// engine deferring to MM3_DEVICE. device=gpu on a machine without a usable GPU
// must fail engine creation (only device=auto may fall back to the CPU).
void verify_environment_device_request(const char * models_dir) {
    const char * requested = std::getenv("MM3_DEVICE");
    if (!requested || !*requested || std::strcmp(requested, "cpu") == 0) {
        return;
    }
    auto options = make_engine_options(models_dir);
    options.device.clear();
    try {
        std::unique_ptr<tts_cpp::minimax::Engine> engine = tts_cpp::minimax::Engine::create(options);
        CHECK(!engine->backend_name().empty());
        std::fprintf(stderr, "[test-minimax-integration] MM3_DEVICE=%s ran on %s\n", requested,
                     engine->backend_name().c_str());
        verify_result(*engine, engine->generate(make_generate_params()));
        verify_cancellation(*engine);
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        CHECK(message.find("no usable GPU") != std::string::npos);
    }
}

int run_integration(const char * models_dir) {
    try {
        {
            std::unique_ptr<tts_cpp::minimax::Engine> engine =
                tts_cpp::minimax::Engine::create(make_engine_options(models_dir));
            CHECK(engine->backend_name() == "CPU");
            verify_recursive_generation(*engine);
            verify_cancellation(*engine);
            verify_pre_armed_cancellation(*engine);
        }
        verify_environment_device_request(models_dir);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL integration exception: %s\n", error.what());
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}

}

int main() {
    const char * models_dir = std::getenv("AUDIOGEN_TEST_MINIMAX_MODELS_DIR");
    if (!models_dir || !*models_dir) {
        std::fprintf(
            stderr,
            "[test-minimax-integration] skipped: AUDIOGEN_TEST_MINIMAX_MODELS_DIR is unset\n");
        return 77;
    }
    return run_integration(models_dir);
}
