// Test double for the measurement consumer only. This supplies the installed
// public Engine interface; it does not test Chatterbox inference or HTP math.
#include <tts-cpp/chatterbox/engine.h>

#include <chrono>
#include <limits>
#include <stdexcept>
#include <thread>

namespace tts_cpp::chatterbox {
struct Engine::Impl {
    EngineOptions opts;
    explicit Impl(const EngineOptions & o) : opts(o) {}
};

Engine::Engine(const EngineOptions & opts) : pimpl_(std::make_unique<Impl>(opts)) {
    if (opts.t3_gguf_path == "throw-constructor") throw std::runtime_error("injected constructor failure");
}
Engine::~Engine() = default;

SynthesisResult Engine::synthesize(const std::string & text) { return synthesize(text, {}); }

SynthesisResult Engine::synthesize(const std::string & text, const StreamCallback & cb) {
    if (text == "throw-synthesis") throw std::runtime_error("injected synthesis failure");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    SynthesisResult result;
    result.sample_rate = 24000;
    result.pcm = {-1.25f, -0.5f, 0.25f, 1.25f, 0.5f, 0.0f, -0.25f, 0.75f};
    result.audio_samples = static_cast<int>(result.pcm.size());
    result.t3_tokens = 2;
    result.t3_ms = 2;
    result.s3gen_ms = 3;
    if (text == "empty") result.pcm.clear();
    if (text == "non-finite") result.pcm[2] = std::numeric_limits<float>::quiet_NaN();
    if (text == "bad-sample-count") ++result.audio_samples;
    if (text == "bad-stage-time") result.t3_ms = std::numeric_limits<double>::infinity();
    if (cb && pimpl_->opts.stream_chunk_tokens > 0) {
        if (text == "missing-callback") return result;
        if (text == "null-callback") cb(nullptr, 2, 0, false);
        cb(result.pcm.data(), 4, text == "bad-index" ? 1 : 0, text == "early-last");
        cb(result.pcm.data() + 4, 4, 1, text != "missing-last");
        if (text == "mismatched-pcm") result.pcm[0] = 0;
    }
    return result;
}

std::string Engine::backend_name() const {
    if (pimpl_->opts.t3_gguf_path == "wrong-primary") return "CPU";
    if (pimpl_->opts.t3_gguf_path == "unknown-primary") return "(unknown)";
    return pimpl_->opts.backend == "hexagon" ? "HTP0" : pimpl_->opts.backend == "opencl" ? "OpenCL" : "CPU";
}
BackendDevice Engine::backend_device() const {
    return backend_name() == "CPU" ? BackendDevice::CPU : BackendDevice::GPU;
}
} // namespace tts_cpp::chatterbox
