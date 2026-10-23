#include "tts-cpp/audio8/engine.h"
#include "voice_features.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace tts_cpp::audio8;
constexpr int FRAME_LIMIT = 48;
constexpr int THREADS = 4;
constexpr int OUTPUT_RATE = 24000;
constexpr const char * TEXT =
    "When we build a voice assistant, the first sound should arrive quickly, even when the answer is long. "
    "The listener can begin following the explanation while the remaining sentences are still being generated.";
constexpr const char * CALLBACK_ERROR = "test callback error";

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

SynthesisResult collect(Engine & engine) {
    std::vector<float> delivered;
    size_t chunks = 0;
    const int expected_rate = engine.sample_rate();
    const auto result = engine.synthesize_stream(TEXT, [&](const float * pcm, size_t count, int rate) {
        require(count > 0 && pcm, "empty PCM callback");
        require(rate == expected_rate, "callback sample rate mismatch");
        delivered.insert(delivered.end(), pcm, pcm + count);
        ++chunks;
        return true;
    });
    require(chunks > 0, "no PCM callbacks");
    require(delivered == result.pcm, "callback PCM differs from result");
    require(result.frames == FRAME_LIMIT, "test prompt ended before the frame limit");
    return result;
}

void expect_failure(Engine & engine, const AudioCallback & callback, const std::string & message) {
    bool failed = false;
    try {
        engine.synthesize_stream(TEXT, callback);
    } catch (const std::runtime_error & error) {
        require(std::string(error.what()).find(message) != std::string::npos,
                "wrong error: " + std::string(error.what()));
        failed = true;
    }
    require(failed, "streaming should have failed");
}

void check_cancellation(Engine & engine) {
    size_t calls = 0;
    expect_failure(engine, [&](const float *, size_t, int) { ++calls; return false; }, "cancelled");
    require(calls == 1, "cancellation delivered additional chunks");
}

void check_exception(Engine & engine) {
    size_t calls = 0;
    expect_failure(engine, [&](const float *, size_t, int) -> bool {
        ++calls;
        throw std::runtime_error(CALLBACK_ERROR);
    }, CALLBACK_ERROR);
    require(calls == 1, "callback exception delivered additional chunks");
}

EngineOptions options(char ** argv) {
    EngineOptions opts;
    opts.lm_gguf_path = argv[1];
    opts.codec_decoder_gguf_path = argv[2];
    opts.max_frames = FRAME_LIMIT;
    opts.n_threads = THREADS;
    opts.backend = "cpu";
    opts.greedy = true;
    return opts;
}

void check_resampling(const EngineOptions & native_opts, const SynthesisResult & native) {
    auto opts = native_opts;
    opts.output_sample_rate = OUTPUT_RATE;
    Engine engine(opts);
    const auto result = collect(engine);
    require(result.codes == native.codes, "resampling changed generated codes");
    require(result.pcm == resample_for_output(native.pcm, native.sample_rate, OUTPUT_RATE),
            "streaming resampler differs from whole-waveform resampling");
    require(result.timings.resample_ms > 0.0, "streaming resampling time was not recorded");
}

void check_engine(char ** argv) {
    const auto opts = options(argv);
    Engine engine(opts);
    expect_failure(engine, {}, "callback must not be empty");
    const auto native = collect(engine);
    require(native.timings.resample_ms == 0.0, "native output counted resampling work");
    check_cancellation(engine);
    check_exception(engine);
    require(collect(engine).pcm == native.pcm, "engine did not recover after callback failure");
    check_resampling(opts, native);
}

}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "expected LM and decoder GGUF paths");
        check_engine(argv);
        std::puts("[audio8-streaming] PASS");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[audio8-streaming] FAIL: %s\n", error.what());
        return 1;
    }
}
