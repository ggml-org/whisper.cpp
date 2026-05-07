#pragma once

// Persistent Supertonic engine.
//
// Loads the Supertonic GGUF once, validates the requested voice / language /
// step count, and keeps the model resident so subsequent calls to
// `synthesize()` only pay the per-call preprocess + duration + text-encoder +
// vector-estimator + vocoder cost - not the GGUF tensor load.
//
// Usage:
//
//     using tts_cpp::supertonic::Engine;
//     using tts_cpp::supertonic::EngineOptions;
//
//     EngineOptions opts;
//     opts.model_gguf_path = "models/supertonic.gguf";
//     opts.n_gpu_layers    = 0;                      // CPU only today
//
//     Engine engine(opts);
//     for (const auto & line : lines) {
//         auto result = engine.synthesize(line);
//         write_wav(result.pcm, result.sample_rate);
//     }
//
// Threading model:
//   - synthesize() on the same Engine instance is NOT safe to call
//     concurrently - the per-stage thread_local caches and the seeded
//     RNG are per-instance shared state.
//   - synthesize() on different Engine instances from different
//     threads is safe.  The supertonic generation_id (set per Engine
//     ctor) keys the stage-internal caches so two Engines don't collide.
//   - cancel() is safe from any thread.
//
// Implemented in src/supertonic_engine.cpp on top of the library-internal
// helpers in src/supertonic_internal.h.

#include "tts-cpp/backend.h"
#include "tts-cpp/export.h"

#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::supertonic {

struct EngineOptions {
    // Required.
    std::string model_gguf_path;

    // Empty / zero values use the defaults stored in the GGUF metadata.
    std::string voice;
    std::string language = "en";
    int   steps    = 0;
    float speed    = 0.0f;
    int   seed     = 42;
    int   n_threads     = 0;
    int   n_gpu_layers  = 0;

    // Optional path to a .npy file containing the initial noise tensor of
    // shape [1, latent_channels, latent_len] (float32).  When provided,
    // latent_len is taken from the npy file (overriding the duration-
    // predicted length) and the seeded RNG is bypassed.  Useful for
    // byte-exact reproduction of an ONNX/PyTorch reference run.
    std::string noise_npy_path;
};

struct SynthesisResult {
    std::vector<float> pcm;
    int   sample_rate = 44100;
    float duration_s  = 0.0f;
};

// Persistent engine.  Loads the GGUF once at construction; subsequent
// synthesize() calls reuse the resident model.
class TTS_CPP_API Engine {
public:
    // Loads the Supertonic GGUF, initialises the backend, validates
    // opts.voice / opts.language up front.  Throws std::runtime_error
    // on any hard failure (GGUF not found, GGUF malformed, unsupported
    // voice).
    explicit Engine(const EngineOptions & opts);

    // Frees the backend + all ggml contexts.
    ~Engine();

    Engine(const Engine &)            = delete;
    Engine & operator=(const Engine &) = delete;

    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    // Synthesize `text` into PCM (44.1 kHz mono float32 by default;
    // see SynthesisResult::sample_rate).  Throws std::runtime_error
    // on failure.  Empty `text` is rejected.
    //
    // Not safe to call concurrently on the same Engine instance.
    SynthesisResult synthesize(const std::string & text);

    // Best-effort cancel of an in-flight synthesize() call on another
    // thread.  Setting the flag is all this does; actual termination
    // happens at the next cancellation check inside the vector-
    // estimator loop (one step is the worst-case cancel latency).
    void cancel();

    // Return the options the engine was constructed with (convenience
    // for callers that want to introspect the resolved n_gpu_layers /
    // n_threads after defaults are applied).
    const EngineOptions & options() const;

    // Return the registered name of the backend the engine actually
    // resolved to during construction (e.g. "CPU", "Metal").  Returns
    // "(unknown)" when the backend is unset.
    std::string backend_name() const;

    // Resolved compute device.  CPU when the build has no GPU backend
    // compiled in, when no GPU was requested (n_gpu_layers <= 0), or
    // when the requested GPU backend refused to initialise.  GPU
    // otherwise.  Stable for the lifetime of the Engine.
    BackendDevice backend_device() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Convenience one-shot wrapper around Engine.  Equivalent to:
//   Engine e(opts); return e.synthesize(text);
// Use the Engine class directly for any host that synthesizes more
// than once - this wrapper pays the full GGUF load + free per call.
TTS_CPP_API SynthesisResult synthesize(const EngineOptions & opts, const std::string & text);

} // namespace tts_cpp::supertonic
