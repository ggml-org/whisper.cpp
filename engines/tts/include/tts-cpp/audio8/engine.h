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
// CPU-validated against the reference implementation; the backend plumbing
// mirrors the other engines so GPU backends can be enabled later.

#include "tts-cpp/backend.h"
#include "tts-cpp/export.h"

#include <cstddef>
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

    // GPU layers to offload; 0 = CPU. Reserved: the engine is CPU-only today
    // and any positive value is ignored with a warning.
    int n_gpu_layers = 0;

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
};

// Persistent engine. Loads the GGUFs once at construction; synthesize() reuses
// the resident models. Codes for the most recent voice prompt are cached, so
// consecutive calls with the same reference skip the codec encoder.
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

    // Best-effort cancel of an in-flight synthesize() on another thread; takes
    // effect at the next language model step or codec block, and leaves
    // synthesize() throwing rather than returning a partial waveform.
    void cancel();

    const EngineOptions & options() const;
    int sample_rate() const;
    std::string backend_name() const;
    BackendDevice backend_device() const;

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
