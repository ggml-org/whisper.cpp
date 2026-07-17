#pragma once

// Public ACE-Step music-generation engine API (QVAC-21921).
//
// End-to-end text-to-music: a text prompt (+ optional lyrics) in, stereo
// 48 kHz audio out. This is the facade the @qvac/audiogen-ggml native addon
// links against (mirrors how tts_cpp::supertonic::Engine / chatterbox back the
// @qvac/tts-ggml addon), so the addon never shells out to a binary and
// compiles for every platform tts-cpp supports.
//
// Pipeline stages (all ggml graphs on the ggml-speech fork, CPU-first):
//     text-encoder (Qwen3-Embedding)  -> prompt embeddings
//     LM (acestep-lm, Qwen3 causal)   -> lyrics + acoustic codes
//     DiT (diffusion transformer)     -> 64-channel acoustic latent
//     VAE (AutoencoderOobleck)        -> 48 kHz stereo PCM   [see vae.h]
//
// Port status (QVAC-21921):
//   [x] custom ggml ops: ggml_col2im_1d, ggml_snake (CPU) in ggml-speech
//   [x] VAE stage (tts_cpp::acestep::Vae) — decode/encode validated on CPU
//   [x] DiT stage (dit_ggml) — load + forward + Euler flow-matching sampler
//   [x] LM stage (lm_ggml + bpe_tokenizer + lm_pipeline) — Phase-2 codes
//   [x] FSQ detokenizer (detok_ggml) — codes -> DiT context latents
//   [x] text-encoder (textenc_ggml) + cond-encoder (cond_ggml)
//   [x] Engine::generate() end-to-end: text -> LM -> detok -> textenc/cond ->
//       DiT -> VAE -> stereo 48 kHz (native, no acestep.cpp binaries).
//   [x] LM Phase-2 CFG (multi-set KV in lm_ggml) + upstream sampling defaults.
//   [x] LM Phase-1 CoT/metadata auto-gen + metadata FSM (metadata_fsm.h).
//   [x] is_turbo auto-detect -> steps/shift (turbo 8/3.0, base/sft 50/1.0).
//   [x] Parity vs acestep.cpp: synth bit-close (corr 0.98-0.99 on same codes);
//       LM greedy matches upstream argmax (divergence = CPU-F32 vs Metal-F16).
// Deferred: DiT CFG/APG (guidance>1, base/sft only), DCW wavelet correction.

#include "tts-cpp/export.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

// GGUF weights for each stage. Either point at a directory holding the four
// GGUFs (models_dir) and let the engine auto-classify by architecture, or set
// explicit per-stage paths (explicit wins over the directory scan).
struct EngineOptions {
    std::string models_dir;

    std::string text_enc_model_path;  // Qwen3-Embedding-*.gguf
    std::string lm_model_path;        // acestep-5Hz-lm-*.gguf
    std::string dit_model_path;       // acestep-v15-*.gguf
    std::string vae_model_path;       // vae-*.gguf

    int  n_threads     = 0;   // 0 = hardware concurrency
    int  n_gpu_layers  = 0;   // 0 = CPU-only (the QVAC-21921 target)
    bool verbose       = false;

    // VAE tiling for bounded memory on long tracks (see vae.h decode path).
    int vae_chunk   = 1024;
    int vae_overlap = 64;
};

struct GenerateParams {
    std::string caption;                 // required text prompt
    std::string lyrics = "[Instrumental]";
    float       duration = 20.0f;        // target seconds (drives LM code count)
    int         inference_steps = 0;     // 0 = auto (turbo: 8, base/sft: 50)
    float       shift = 0.0f;            // 0 = auto (turbo: 3.0, base/sft: 1.0)
    std::string vocal_language;          // optional hint, e.g. "en"
    int         bpm = 0;                 // optional; 0 => N/A (LM/DiT infer)
    std::string keyscale;                // optional, e.g. "C major"
    std::string timesignature;           // optional, e.g. "4/4"
    int         seed = -1;               // <0 = random
    // LM sampling (Phase-2 audio codes). Defaults mirror acestep.cpp.
    float       lm_temperature = 0.85f;
    float       lm_top_p       = 0.9f;
    int         lm_top_k       = 0;      // 0 = disabled (top_p only)
    float       lm_cfg_scale   = 2.0f;   // classifier-free guidance for codes
    bool        lm_phase1      = true;   // auto-fill missing metadata (FSM CoT)

    // Pre-supplied FSQ audio codes (LM output). When non-empty, the LM stage is
    // skipped and these codes are used directly (parity / caching / editing).
    std::vector<int> audio_codes;
};

// LM-enriched metadata surfaced alongside the audio (the same fields
// acestep.cpp writes into request0.json).
struct GenerateMetadata {
    std::string caption;         // enriched caption produced by the LM
    std::string lyrics;
    std::string keyscale;
    std::string vocal_language;
    int         bpm = 0;
    int         timesignature = 0;
    long long   seed = 0;
    int         n_codes = 0;
};

struct GenerateResult {
    std::vector<float> pcm;          // interleaved stereo, [t*2 + ch]
    int                sample_rate = 48000;
    int                channels    = 2;
    GenerateMetadata   metadata;
};

// Optional progress callback: stage name ("lm"|"dit"|"vae"), current step,
// total steps (total <= 0 when unknown). Return false to request cancellation.
using ProgressFn = std::function<bool(const std::string & stage, int step, int total)>;

class TTS_CPP_API Engine {
public:
    // Load all stages. Throws std::runtime_error on failure (missing GGUF,
    // wrong architecture, alloc failure) or if a stage is not yet ported.
    static std::unique_ptr<Engine> create(const EngineOptions & opts);

    ~Engine();
    Engine(const Engine &)             = delete;
    Engine & operator=(const Engine &) = delete;

    // Generate music from a text prompt. Empty pcm on cancellation.
    GenerateResult generate(const GenerateParams & params, const ProgressFn & progress = {}) const;

    // Cooperative cancel for an in-flight generate() on another thread.
    void cancel() const;

    int         sample_rate() const;  // 48000
    std::string backend_name() const;

private:
    Engine();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::acestep
