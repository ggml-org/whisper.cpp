#pragma once

// Audio8 TTS engine: a DualAR semantic language model over a DAC-style neural
// codec, at 44.1 kHz.
//
// A 24-layer "slow" transformer reads the ChatML prompt and emits one semantic
// token per 2048 audio samples; a 4-layer "fast" head expands each of those
// into the nine remaining codebook values for the same frame; the codec turns
// the ten codebooks into a waveform.
//
// Voice cloning runs entirely in this process. Hand synthesize() a reference
// waveform and its transcript and the codec's encoder turns the audio into
// codes, which are prepended to the prompt as the speaker's own history. The
// encoder lives in a separate GGUF, so a text-only build can leave it out.
//
// Validated against the reference implementation on CPU, Metal, and Vulkan, each
// compared to the same F32 reference rather than to the other.

#include "tts-cpp/backend.h"
#include "tts-cpp/export.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp {
namespace audio8 {

struct EngineOptions {
    // Required: the DualAR language model and the codec's synthesis half.
    std::string lm_gguf_path;
    std::string codec_decoder_gguf_path;

    // Required only for cloning: the codec's analysis half, which turns a
    // reference waveform into codes.
    std::string codec_encoder_gguf_path;

    int n_threads = 0;   // 0 => min(hardware_concurrency, 4)

    // Any positive value puts the whole model on a validated GPU backend, 0
    // keeps it on CPU. There is no partial offload: this selects the backend the
    // model runs on rather than splitting layers across two. Falls back to CPU
    // when no validated GPU is present.
    int n_gpu_layers = 0;

    // Explicit backend selection. Accepted values: "auto" / "" (default,
    // tier-based GPU selection honouring n_gpu_layers), "cpu", or a specific
    // backend name such as "hexagon" (Snapdragon HTP0) or an exact ggml
    // device name (e.g. "HTP0"). Explicit requests fail construction when
    // the named backend is unavailable; "auto" never fails on that path and
    // falls back to CPU. Hexagon is experimental and currently requires a
    // build that ships the matching ggml Hexagon backend and DSP library
    // next to the binary.
    std::string backend = "auto";

    // Sampling. The reference filters candidates by top_k and top_p on the raw
    // logits and only then applies the temperature, and this follows it.
    // greedy ignores all three and takes the argmax, which is what the parity
    // fixtures use.
    bool greedy = false;
    int seed = 42;
    float temperature = 0.7f;
    int top_k = 50;
    float top_p = 0.9f;

    // Cap on generated frames, each 2048 samples (~46 ms). 0 => 512, the
    // reference default, which is about 24 s. The prompt and the frames it
    // generates share whatever context the language model GGUF declares, so a
    // long prompt lowers the ceiling.
    int max_frames = 0;

    // Rate to deliver the result at; 0 keeps the codec's own 44.1 kHz.
    int output_sample_rate = 0;

    // Directory for dynamically-loaded ggml backends (GGML_BACKEND_DL builds);
    // empty keeps the default search behaviour.
    std::string backends_dir;

    // Print the per-stage timing breakdown of every synthesize() to stderr.
    // The same numbers are always available in SynthesisResult::timings.
    bool verbose = false;
};

// Wall time of each synthesize() stage, in milliseconds. The stages are
// disjoint, so they sum to total_ms up to the bookkeeping between them.
struct StageTimings {
    double voice_encode_ms = 0.0;  // codec encoder over the reference, cloning only
    double prompt_ms       = 0.0;  // tokenisation and prompt assembly
    double prefill_ms      = 0.0;  // one slow-AR pass over the whole prompt
    double sample_ms       = 0.0;  // picking the semantic token of each frame
    double fast_decode_ms  = 0.0;  // per-frame fast-AR passes and their sampling
    double slow_decode_ms  = 0.0;  // per-frame slow-AR steps after the prefill
    double codec_latent_ms = 0.0;  // quantiser banks, post transformer
    // Upsampling and the sample-rate stack. On the Core ML sidecar most windows
    // run during generation, so this is only the part left after the last frame.
    double codec_synth_ms  = 0.0;
    double resample_ms     = 0.0;  // only when output_sample_rate differs
    double total_ms        = 0.0;
};

// A voice to clone: mono float32 at sample_rate, plus what is said in it. The
// transcript matters -- the model conditions on it as the turn the reference
// audio answers, and a wrong one degrades the clone. Anything that is not
// 44.1 kHz is resampled on the way in.
struct VoicePrompt {
    std::vector<float> pcm;
    int sample_rate = 44100;
    std::string transcript;

    bool empty() const { return pcm.empty(); }
};

struct SynthesisResult {
    std::vector<float> pcm;   // mono float32
    int sample_rate = 44100;
    float duration_s = 0.0f;
    int frames = 0;           // codec frames the language model emitted
    StageTimings timings;

    // The discrete trajectory behind the waveform: num_codebooks values per
    // frame, frame-major, so codes[f * num_codebooks + b] is codebook b of frame
    // f. Two runs that agree here differ only in the codec's arithmetic, which
    // makes this the sharpest way to compare backends or quantisation tiers.
    std::vector<int> codes;

    // Where this synthesis's codec synthesis stack actually ran: "ggml" (on
    // backend_name()) or the Core ML sidecar's compute label ("coreml-all",
    // "coreml-gpu", ...). Unlike Engine::codec_on_coreml(), this is per call:
    // a loaded sidecar that cannot serve a call falls back to ggml here.
    std::string codec_synthesis_backend = "ggml";
};

using AudioCallback = std::function<bool(const float *, size_t, int)>;

class TTS_CPP_API Engine {
public:
    // Throws std::runtime_error on any hard failure, a set of GGUFs that does
    // not come from one checkpoint among them.
    explicit Engine(const EngineOptions & opts);
    ~Engine();

    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;
    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    // Speaks `text` in the model's own voice. Not thread-safe per instance.
    SynthesisResult synthesize(const std::string & text);

    // Speaks `text` in the voice of `voice`. Throws when the engine was built
    // without codec_encoder_gguf_path.
    SynthesisResult synthesize(const std::string & text, const VoicePrompt & voice);

    SynthesisResult synthesize_stream(const std::string & text, const AudioCallback & callback);
    SynthesisResult synthesize_stream(const std::string & text, const VoicePrompt & voice,
                                      const AudioCallback & callback);

    // Best-effort cancel of an in-flight synthesize() on another thread; takes
    // effect at the next language model step or codec block, and leaves
    // synthesize() throwing rather than returning a partial waveform.
    void cancel();

    const EngineOptions & options() const;
    int sample_rate() const;
    std::string backend_name() const;
    BackendDevice backend_device() const;

    // True while an Apple Core ML sidecar for the codec's synthesis stack is
    // attached (a TTS_CPP_COREML build with a compiled model next to the decoder
    // GGUF). A sidecar whose first use fails is retired and later calls go
    // straight to ggml, after which this returns false.
    // SynthesisResult::codec_synthesis_backend reports where each call actually
    // ran. The language model and the codec's post transformer always run on
    // the backend backend_name() reports.
    bool codec_on_coreml() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Reads a WAV into a VoicePrompt: any bit depth dr_wav handles, downmixed to
// mono, left at the file's own rate for synthesize() to resample. Throws
// std::runtime_error when the file cannot be read.
TTS_CPP_API VoicePrompt load_voice_prompt(const std::string & wav_path,
                                          const std::string & transcript);

// One-shot convenience wrapper (pays a full model load per call).
TTS_CPP_API SynthesisResult synthesize(const EngineOptions & opts, const std::string & text);

}  // namespace audio8
}  // namespace tts_cpp
