#pragma once

// Persistent CosyVoice3 engine (Fun-CosyVoice3-0.5B / 1.5B).
//
// Native C++/ggml, CPU implementation of the Fun-CosyVoice3 back-half, under the
// same public API as the Chatterbox / Supertonic engines so the JS addon +
// @qvac/sdk flow works end-to-end.  synthesize() returns real 24 kHz speech.
//
// Pipeline (CPU path):
//   text --(Qwen2 BPE tokenizer)--> Qwen2.5 LM --> speech tokens [0, 6561)
//        --> DiT conditional-flow-matching (Euler ODE) --> mel
//        --> CausalHiFT vocoder --> 24 kHz mono PCM
//   Zero-shot voice cloning adds: reference audio --> S3 tokenizer (prompt
//   tokens) + CAM++ (192-d speaker embedding).  Iteration 1 uses a baked voice
//   (voice.gguf); the native S3 tokenizer + CAM++ path is a follow-up.  Instruct
//   mode (EngineOptions::instruct_text) selects a Chinese dialect / emotion /
//   speed / volume, mirroring CosyVoice3 inference_instruct2.
//
// CosyVoice3 ships as a small set of GGUFs (llm / flow / hift / s3tok /
// campplus / voices).  Point `model_dir` at the folder that holds them, or
// set the per-component paths explicitly.
//
// Usage:
//
//     using tts_cpp::cosyvoice::Engine;
//     using tts_cpp::cosyvoice::EngineOptions;
//
//     EngineOptions opts;
//     opts.model_dir     = "models/cosyvoice3-0.5b";
//     opts.n_gpu_layers  = 0;                 // 0 = CPU (iteration 1 is CPU-only)
//
//     Engine engine(opts);
//     auto result = engine.synthesize("Hello world.");
//     write_wav(result.pcm, result.sample_rate);   // 24 kHz mono float32
//
// Threading model (mirrors the sibling engines):
//   - synthesize() on the same Engine instance is NOT safe to call
//     concurrently.
//   - synthesize() on different Engine instances from different threads is
//     safe.
//   - cancel() is safe from any thread.
//
// Implemented in src/cosyvoice_engine.cpp.

#include "tts-cpp/backend.h"
#include "tts-cpp/export.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::cosyvoice {

struct EngineOptions {
    // Directory holding the standard CosyVoice3 GGUFs
    // (cosyvoice3-{llm,flow,hift,s3tok,campplus,voices}-*.gguf).  Either set
    // this, or set the per-component paths below explicitly (explicit paths
    // win over the ones discovered under model_dir).
    std::string model_dir;

    std::string llm_gguf_path;       // Qwen2.5 LM (text -> speech tokens)
    std::string flow_gguf_path;      // DiT conditional-flow-matching (tokens -> mel)
    std::string hift_gguf_path;      // CausalHiFT vocoder (mel -> 24 kHz PCM)
    std::string s3tok_gguf_path;     // supervised S3 speech tokenizer (zero-shot prompt)
    std::string campplus_gguf_path;  // CAM++ speaker encoder (zero-shot)

    // Qwen2 byte-level BPE text frontend (vocab.json + merges.txt) and the
    // baked default-voice GGUF (voice/prompt_stok, prompt_token, prompt_feat,
    // embedding — see scripts/bake-cosyvoice3-voice.py).  Resolved under
    // model_dir when left empty (vocab.json / merges.txt / voice.gguf).
    std::string vocab_path;
    std::string merges_path;
    std::string voice_gguf_path;

    // ---- Zero-shot voice cloning (optional) ----
    // Reserved: cloning from arbitrary reference audio needs the native S3
    // tokenizer + CAM++ port (s3tok/campplus GGUFs above), which is a follow-up.
    // If set, iteration 1 logs a warning and falls back to the baked voice.
    // reference_audio = a few seconds of clean speech; prompt_text = its transcript.
    std::string reference_audio;
    std::string prompt_text;

    // Instruct mode (CosyVoice3 instruct2): a natural-language instruction that
    // controls dialect / accent / emotion / speed / volume, e.g.
    // "请用广东话表达。" (speak in Cantonese) or "speak slowly and cheerfully".
    // When non-empty the LM is conditioned on this instruction and drops its
    // prompt speech tokens; the baked/selected voice still supplies the timbre.
    // Empty (default) = zero-shot (prompt_text / baked-voice transcript path).
    std::string instruct_text;
    // Reserved: named-voice selection from a multi-voice voices.gguf is not yet
    // wired.  Use voice_gguf_path to point at a single baked voice.gguf.
    std::string voice;

    // Language hint.  Reserved: the text-normalization frontend (wetext) is not
    // yet integrated, so this is currently accepted but not acted on.
    std::string language = "en";

    int seed         = 42;
    int n_threads    = 0;   // 0 = leave the backend's default thread pool.
    int n_gpu_layers = 0;   // 0 = CPU; >0 selects the GPU path (OpenCL/Adreno only).

    // Argmax speech-token decode instead of RAS nucleus sampling.  Sampling is
    // chaotic in the logits -- a 1-ulp difference re-rolls the whole token
    // trajectory -- so greedy is what makes a CPU-vs-GPU comparison meaningful.
    // Off by default: sampling is the better-sounding production path.
    bool greedy      = false;

    // Pin the LM speech-token trajectory instead of decoding one.  Sampling is
    // chaotic in the logits, so two backends will not agree on tokens even at
    // the same seed; pinning them is what makes the flow + vocoder stages
    // comparable across backends, and what makes two benchmark legs do provably
    // equal work.  Empty (the default) = decode normally.
    std::vector<int> force_speech_tokens;

    // Reserved: output resampling is not yet implemented.  CosyVoice3 output is
    // always the native 24 kHz (reported on SynthesisResult::sample_rate),
    // regardless of this value.
    int output_sample_rate = 0;

    // Reserved: the DiT flow currently runs a fixed 10 Euler steps (the
    // parity-validated schedule); this override is not yet honored.
    int cfm_steps = 0;

    // ---------------- Streaming synthesis ----------------------------
    //
    // When `stream_chunk_tokens > 0` AND a non-empty callback is passed to
    // synthesize(), the callback is driven progressively with 24 kHz PCM in
    // chunks.  The returned SynthesisResult.pcm still holds the full concatenated
    // audio (the callback is an *addition*, not a replacement), so
    // `result.pcm == concat(callback chunks)` holds.  Batch path when
    // stream_chunk_tokens == 0 OR the callback is empty.
    //
    // NOTE (iteration 1): the audio is computed in full first, then the callback
    // is driven over it in chunks.  The native token-by-token token2wav loop that
    // would make early chunks *arrive* early (true low first-audio latency), and
    // stream_left_context_tokens (per-chunk cost bounding), are reserved and not
    // yet realized.
    //
    //   stream_chunk_tokens        Speech tokens per emitted chunk (0 = batch).
    //   stream_first_chunk_tokens  Size of the first emitted chunk (0 = same as
    //                              stream_chunk_tokens).
    //   stream_left_context_tokens Reserved (see note); not yet used.
    int stream_chunk_tokens        = 0;
    int stream_first_chunk_tokens  = 0;
    int stream_left_context_tokens = 0;

    // Directory to scan for dynamically-loaded ggml backend plugins.  Consulted
    // only when n_gpu_layers > 0.  Process-global and first-Engine-wins.
    std::string backends_dir;

    // Writable directory for the OpenCL program-binary cache (Android).  Empty
    // leaves ggml's default, which on Android resolves under HOME and is inert.
    std::string opencl_cache_dir;
};

// Per-chunk PCM callback for streaming synthesis.  Receives `samples`
// consecutive float32 mono samples at SynthesisResult::sample_rate (24 kHz by
// default).  The buffer is owned by the engine and must not be retained past
// the callback; copy out if you need the data.
//   `chunk_index`  0-based; increments by exactly 1 per chunk.
//   `is_last`      true only on the final chunk.
// Throwing from this callback aborts synthesis (the exception propagates out
// of synthesize()).
using StreamCallback = std::function<void(
    const float * pcm, std::size_t samples, int chunk_index, bool is_last)>;

// Native sample rate (Hz) the CosyVoice3 CausalHiFT vocoder emits. Kept in
// lock-step with the internal vocoder constant by a static_assert in
// cosyvoice_engine.cpp, so this public value can't silently drift from it.
constexpr int kNativeSampleRate = 24000;

// Per-stage wall-clock accounting for one synthesize() call, in milliseconds.
// Measured after a backend sync, so an async GPU submission cannot read as
// speed.  The counters exist so two runs can be PROVEN to have done the same
// work before their times are compared -- a run that decoded a different number
// of tokens is void, not faster.
struct StageTimings {
    double lm_prefill_ms    = 0;
    double lm_decode_ms     = 0;
    double flow_frontend_ms = 0;
    double dit_euler_ms     = 0;
    double hift_f0_ms       = 0;
    double hift_source_ms   = 0;
    double hift_stft_ms     = 0;
    double hift_decode_ms   = 0;
    double total_ms         = 0;

    int n_decode_steps  = 0;
    int n_speech_tokens = 0;
    int tm              = 0;   // flow frames in (prompt + generated)
    int mel_len         = 0;   // mel frames out (after the prompt trim)
};

struct SynthesisResult {
    std::vector<float> pcm;
    int   sample_rate = kNativeSampleRate;
    float duration_s  = 0.0f;

    StageTimings     timings;        // per-stage wall clock for this call
    std::vector<int> speech_tokens;  // the LM trajectory this call actually used
};

// Persistent engine.  Loads the model once at construction; subsequent
// synthesize() calls reuse the resident model.
class TTS_CPP_API Engine {
public:
    // Loads the model and prepares the resident engine.  Throws
    // std::runtime_error on a hard failure (e.g. a required GGUF missing under
    // model_dir).  Loading happens once here; synthesize() reuses it.
    explicit Engine(const EngineOptions & opts);
    ~Engine();

    Engine(const Engine &)             = delete;
    Engine & operator=(const Engine &) = delete;
    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    // Synthesize `text` into PCM (24 kHz mono float32 by default).  Throws
    // std::runtime_error on failure.  Empty `text` is rejected.
    // Not safe to call concurrently on the same Engine instance.
    SynthesisResult synthesize(const std::string & text);

    // Streaming variant — see EngineOptions streaming block.  Falls through
    // to the batch path when streaming is disabled.
    SynthesisResult synthesize(const std::string & text,
                               const StreamCallback & on_chunk);

    // Best-effort cancel of an in-flight synthesize() on another thread.
    void cancel();

    const EngineOptions & options() const;

    // Registered name of the resolved backend ("CPU", "Metal", ...).
    std::string backend_name() const;

    // Resolved compute device (CPU — iteration 1 is CPU-only).
    BackendDevice backend_device() const;

    // True when a GPU device was present but unusable (fell back to CPU).
    bool gpu_unsupported() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Convenience one-shot wrapper.  Equivalent to:
//   Engine e(opts); return e.synthesize(text);
TTS_CPP_API SynthesisResult synthesize(const EngineOptions & opts, const std::string & text);

} // namespace tts_cpp::cosyvoice
