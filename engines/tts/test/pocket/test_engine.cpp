#include "tts-cpp/pocket/engine.h"
#include "test_workload.h"
#include "test_backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <thread>

using namespace tts_cpp::pocket;
template<class F> void rejects(const char * name, F fn) {
    bool threw = false; try { fn(); } catch (const std::exception &) { threw = true; }
    if (!threw) throw std::runtime_error(std::string("accepted ")+name);
}
static void compare(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || a.size() != b.size()) throw std::runtime_error("PCM length mismatch");
    double error = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("non-finite PCM");
        error = std::max(error, std::abs(double(a[i])-b[i]));
    }
    if (error > 0.0001) throw std::runtime_error("stream/batch/reuse differ");
}
static void check_eos_tail(EngineOptions opts) {
    // Force EOS on the first frame. A 100-frame tail used to outlive the
    // token-only budget for this short prompt and throw after streaming PCM.
    opts.eos_threshold = -1e30f;
    opts.frames_after_eos = 100;
    opts.context = context_with_tail(opts, "Hi.");
    Engine engine(opts);
    size_t samples = 0;
    const auto result = engine.synthesize_stream("Hi.", [&](const float *, size_t n, int) {
        samples += n; return true;
    });
    if (result.cancelled || result.generated_frames != opts.frames_after_eos || samples != size_t(opts.frames_after_eos)*tts_cpp::pocket::detail::Mimi::frame_samples())
        throw std::runtime_error("explicit EOS tail did not complete");
    --opts.context;
    Engine too_small(opts);
    int callbacks = 0;
    rejects("context without full EOS tail", [&] {
        too_small.synthesize_stream("Hi.", [&](const float *, size_t, int) { ++callbacks; return true; });
    });
    if (callbacks) throw std::runtime_error("context rejected after delivering PCM");
}
static void check_parallel_instances(EngineOptions opts) {
    opts.eos_threshold = -1e30f; opts.frames_after_eos = 2;
    std::vector<float> first, second;
    std::exception_ptr first_error, second_error;
    auto run = [&](std::vector<float> & pcm, std::exception_ptr & error) {
        try { Engine instance(opts); pcm = instance.synthesize("Hi.").pcm; }
        catch (...) { error = std::current_exception(); }
    };
    // Include construction and destruction: OpenCL handles share backend
    // lifecycle state as well as graph scratch and mutable kernel arguments.
    std::thread a(run, std::ref(first), std::ref(first_error));
    std::thread b(run, std::ref(second), std::ref(second_error));
    a.join(); b.join();
    if (first_error) std::rethrow_exception(first_error);
    if (second_error) std::rethrow_exception(second_error);
    compare(first, second);
}
int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) { std::fprintf(stderr, "usage: test-pocket-engine model-directory flow-lm.gguf [--gpu]\n"); return 2; }
    try {
        const std::string dir = argv[1];
        EngineOptions opts; opts.flow_lm_path = argv[2]; opts.mimi_path = dir+"/mimi.gguf";
        opts.frontend_path = dir+"/frontend.json"; opts.voice_path = dir+"/voice.gguf";
        opts.use_gpu = argc == 4 && std::string(argv[3]) == "--gpu";
        opts.backend = argc == 4 && !opts.use_gpu ? argv[3] : "";
        check_eos_tail(opts);
        check_parallel_instances(opts);
        Engine engine(opts);
        const auto backend = std::string(engine.backend_name());
        if (backend != engine.flow_lm_backend_name() || backend != engine.mimi_backend_name())
            throw std::runtime_error("stage backend reporting differs");
        if (opts.backend == "opencl" && backend.find("OpenCL") == std::string::npos)
            throw std::runtime_error("OpenCL test ran on another backend");
        if (opts.backend != "opencl") check_pocket_backend(engine.backend_name(), opts.use_gpu);
        std::printf("FlowLM: %s; Mimi: %s\n", engine.flow_lm_backend_name(), engine.mimi_backend_name());
        const std::string text = "Hello! We can generate speech with Fabric.";
        const auto batch = engine.synthesize(text);
        if (batch.cancelled || batch.sample_rate != 24000 || batch.generated_frames <= 0 || batch.first_audio_ms <= 0) throw std::runtime_error("invalid synthesis result");
        std::vector<float> streamed; int callbacks = 0;
        const auto stream = engine.synthesize_stream(text, [&](const float * p, size_t n, int rate) {
            if (rate != 24000 || !n || n > size_t(tts_cpp::pocket::detail::Mimi::max_decode_frames())*tts_cpp::pocket::detail::Mimi::frame_samples()) throw std::runtime_error("invalid callback chunk");
            streamed.insert(streamed.end(), p, p+n);
            if (++callbacks == 1) {
                rejects("reentrant synthesis", [&] { engine.synthesize("Nested request."); });
                bool rejected = false;
                std::thread other([&] { try { engine.synthesize("Concurrent request."); } catch (const std::exception &) { rejected = true; } });
                other.join(); if (!rejected) throw std::runtime_error("concurrent call accepted");
            }
            return true;
        });
        if (stream.cancelled || !stream.pcm.empty() || callbacks < 2) throw std::runtime_error("invalid stream result");
        compare(batch.pcm, streamed);
        callbacks = 0;
        auto cancelled = engine.synthesize_stream(text, [&](const float *, size_t, int) { ++callbacks; return false; });
        if (!cancelled.cancelled || callbacks != 1) throw std::runtime_error("callback cancellation failed");
        compare(engine.synthesize(text).pcm, batch.pcm);
        cancelled = engine.synthesize_stream(text, [&](const float *, size_t, int) {
            std::thread stop([&] { engine.cancel(); }); stop.join(); return true;
        });
        if (!cancelled.cancelled) throw std::runtime_error("cross-thread cancellation failed");
        rejects("callback exception", [&] { engine.synthesize_stream(text, [](const float *, size_t, int) -> bool { throw std::runtime_error("callback failure"); }); });
        rejects("empty text", [&] { engine.synthesize("  \n"); });
        rejects("invalid UTF-8", [&] { engine.synthesize(std::string(1, char(0xff))); });
        compare(engine.synthesize(text).pcm, batch.pcm);
        for (int rate : {8000, 44100}) {
            opts.output_sample_rate = rate; Engine resampled(opts);
            if (resampled.sample_rate() != rate) throw std::runtime_error("wrong advertised rate");
            const auto output = resampled.synthesize(text); std::vector<float> chunks;
            resampled.synthesize_stream(text, [&](const float * p, size_t n, int sr) {
                if (sr != rate) throw std::runtime_error("wrong callback rate");
                chunks.insert(chunks.end(), p, p+n); return true;
            });
            compare(output.pcm, chunks);
            if (output.sample_rate != rate || std::abs(double(output.pcm.size())/rate-double(batch.pcm.size())/24000) > 1.0/rate)
                throw std::runtime_error("resampling changed duration");
        }
        opts.output_sample_rate = 0;
        rejects("invalid sample rate", [&] { Engine invalid(opts); });
        opts.output_sample_rate = 24000;
        opts.eos_threshold = 1e30f;
        Engine no_eos(opts);
        rejects("producer EOS failure", [&] { no_eos.synthesize("Hello."); });
        rejects("producer EOS failure reuse", [&] { no_eos.synthesize("Hello."); });
        opts.eos_threshold = -4; opts.steps = 0;
        rejects("invalid steps", [&] { Engine invalid(opts); });
        std::puts("Pocket Engine synthesis, streaming, cancellation, reentry, concurrency and error recovery passed"); return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
