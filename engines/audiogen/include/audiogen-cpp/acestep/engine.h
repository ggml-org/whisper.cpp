#pragma once

// Public ACE-Step music-generation engine API.
//
// End-to-end text-to-music: a text prompt (+ optional lyrics) in, stereo
// 48 kHz audio out. This is the facade the @qvac/audiogen-ggml native addon
// links against (mirrors how tts_cpp::supertonic::Engine / chatterbox back the
// @qvac/tts-ggml addon), so the addon never shells out to a binary and
// compiles for every platform tts-cpp supports.
//
// Pipeline stages (all ggml graphs on the ggml-speech fork):
//     LM (acestep-lm, Qwen3 causal)   -> metadata + acoustic codes
//     FSQ detokenizer                 -> DiT context latents
//     text-encoder (Qwen3-Embedding)  -> prompt embeddings
//     condition encoder               -> cross-attention states
//     DiT (diffusion transformer)     -> 64-channel acoustic latent
//     VAE (AutoencoderOobleck)        -> 48 kHz stereo PCM   [see vae.h]
//
// With a GPU selected, DiT, VAE and the encoders use it by default. The LM and
// detokenizer use the validated Metal/OpenCL and Vulkan/Metal/OpenCL allowlists,
// respectively, with CPU fallback for unmeasured backends.
//
// Port status:
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
//   [x] Informal parity vs acestep.cpp: synth correlation measured at 0.98-0.99
//       on same codes; no reproducible result artifact is committed.
//   [x] DiT sampler Haar DCW "double" correction (official ACE-Step defaults).
//   [x] Simple Mode: LM inspire pass expands a short query into the full
//       request (caption, lyrics, metadata) ahead of synthesis.
// Deferred: DiT CFG/APG (guidance>1, base/sft only).

#include "audiogen-cpp/export.h"
#include "audiogen-cpp/gpu_fallback.h"

#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace tts_cpp::acestep {

// GGUF weights for each stage. Either point at a directory holding the four
// GGUFs (models_dir) and let the engine classify by filename substring, or set
// explicit per-stage paths (explicit wins over the directory scan). Scanned
// filenames must contain one of the documented stems: embedding/text-enc/textenc,
// -lm/lm-/_lm/ace-lm/5hz-lm, turbo/dit/v15/sft, or vae.
struct EngineOptions {
    std::string models_dir;

    std::string text_enc_model_path;  // Qwen3-Embedding-*.gguf
    std::string lm_model_path;        // acestep-5Hz-lm-*.gguf
    std::string dit_model_path;       // acestep-v15-*.gguf
    std::string vae_model_path;       // vae-*.gguf

    int  n_threads     = 0;   // 0 = hardware concurrency
    int  n_gpu_layers  = 0;   // 0 = CPU-only (CPU target)
    bool verbose       = false;
    // Directory holding the dlopen'd ggml backend modules the addon staged next
    // to its `.bare` (the per-arch `<bare-target>/<module>` subdir). Required on
    // arm64 (Android + Linux), where the ggml-speech port ships the CPU backend
    // as per-microarch MODULE .so files (GGML_BACKEND_DL) rather than static
    // archives; the engine calls `ggml_backend_load_all_from_path()` on it before
    // acquiring any backend from the registry. Empty -> rely on ggml's built-in
    // search path (static-linked desktop / Apple builds need nothing here).
    std::string backends_dir;
    // "auto" (or empty) keeps the n_gpu_layers request and runtime GPU tiering.
    // "cpu", "opencl", "webgpu" and "hexagon" (the HTP0 NPU session) select a
    // backend explicitly, as does an exact ggml device name ("HTP0", "Vulkan0", ...).
    // An explicit request overrides n_gpu_layers and makes create() throw when
    // the device is missing or fails to initialise. Hexagon is never chosen
    // automatically. The stage-placement allowlist still applies on top.
    std::string backend = "auto";
    // "auto" keeps the stage-placement policy for the LM. Any other value uses
    // the same names as `backend` and puts the LM on that device, which may
    // differ from the primary one (the Hexagon NPU for the diffusion stages
    // with the Adreno GPU for the LM, say); create() throws if it is missing.
    std::string lm_backend = "auto";
    // When non-empty, generate() writes one .bin per pipeline stage into this
    // directory (3x int32 header [ndim, d0, d1] then float32 payload). Used to
    // localise a backend divergence to the stage that introduces it; the
    // directory must already exist. Empty = no dumping and no overhead.
    std::string dump_stages_dir;
    // NOTE: VAE windowed decode probes the active backend's allocation cap and
    // adapts its window for bounded memory on long tracks. It is intentionally
    // not exposed as an API option; ACESTEP_VAE_WIN_CORE is diagnostic only.
    // VAE encode remains a full-graph operation.
};

inline constexpr char AUDIO_EDIT_DEFAULT_LYRICS[] = "[Instrumental]";
inline constexpr char AUDIO_EDIT_REPAINT_STAGE[] = "repaint";
inline constexpr char AUDIO_EDIT_FLOW_STAGE[] = "flow-edit";
inline constexpr float AUDIO_EDIT_MIN_RATIO = 0.0f;
inline constexpr float AUDIO_EDIT_MAX_RATIO = 1.0f;
inline constexpr float REPAINT_SOURCE_END_SECONDS = -1.0f;
inline constexpr float REPAINT_DEFAULT_STRENGTH = 0.5f;
inline constexpr int FLOW_EDIT_DEFAULT_AVERAGES = 1;
inline constexpr float FLOW_EDIT_NO_CFG_SCALE = 1.0f;

enum class RepaintMode {
    Conservative,
    Balanced,
    Aggressive,
};

struct RepaintParams {
    float       start_seconds = AUDIO_EDIT_MIN_RATIO;
    float       end_seconds   = REPAINT_SOURCE_END_SECONDS;
    RepaintMode mode          = RepaintMode::Balanced;
    float       strength      = REPAINT_DEFAULT_STRENGTH;
    std::string caption;
    std::string lyrics;
};

struct FlowEditParams {
    std::string source_caption;
    std::string source_lyrics = AUDIO_EDIT_DEFAULT_LYRICS;
    std::string target_caption;
    std::string target_lyrics = AUDIO_EDIT_DEFAULT_LYRICS;
    float       n_min         = AUDIO_EDIT_MIN_RATIO;
    float       n_max         = AUDIO_EDIT_MAX_RATIO;
    int         n_avg         = FLOW_EDIT_DEFAULT_AVERAGES;
    float diffusion_guidance_scale = FLOW_EDIT_NO_CFG_SCALE;
    bool  dcw_enabled              = false;
    bool  use_adg                  = false;
    bool  use_heun                 = false;
};

using AudioEditParams = std::variant<RepaintParams, FlowEditParams>;

struct GenerateParams {
    std::string caption;                 // required text prompt
    std::string lyrics = AUDIO_EDIT_DEFAULT_LYRICS;
    float       duration = 20.0f;        // target seconds (drives LM code count)
    int         inference_steps = 0;     // 0 = auto (turbo: 8, base/sft: 50)
    float       shift = 0.0f;            // 0 = auto (turbo: 3.0, base/sft: 1.0)
    float       guidance_scale = 0.0f;   // 0 = auto (turbo: 1.0, base/sft: 7.0); >1 runs CFG via APG
    std::string vocal_language;          // optional hint, e.g. "en"
    int         bpm = 0;                 // optional; 0 => N/A (LM/DiT infer)
    std::string keyscale;                // optional, e.g. "C major"
    std::string timesignature;           // optional, e.g. "4/4"
    bool        augment_caption_with_metadata = false;
    long long   seed = -1;               // <0 = random (uint32 range: torch/philox parity)
    // LM sampling (Phase-2 audio codes). Defaults mirror acestep.cpp.
    float       lm_temperature = 0.85f;
    float       lm_top_p       = 0.9f;
    int         lm_top_k       = 0;      // 0 = disabled (top_p only)
    float       lm_cfg_scale   = 2.0f;   // classifier-free guidance for codes
    bool        lm_phase1      = true;   // auto-fill missing metadata (FSM CoT)
    // Percentile loudness normalization on the generation output PCM (the
    // acestep.cpp export behavior): the 99.999th-percentile sample scales to
    // 1.0 and the tail above it hard-clips, maximizing perceived loudness.
    // Disable to get the raw VAE output. Audio-edit outputs and lego stems are
    // never normalized: repaint preserves untouched source regions bit-for-bit
    // and a stem keeps its mix gain relative to its source.
    bool        normalize_loudness = true;
    // Simple Mode: treat `caption` as a short natural-language query and let
    // the LM inspire pass compose the full request before synthesis — detailed
    // caption, lyrics, and any metadata left unset (bpm, key/scale, time
    // signature, duration <= 0, vocal language). Set fields are kept. Requires
    // text2music with no pre-supplied audio_codes; `lyrics` must be empty (the
    // LM writes them) or "[Instrumental]" (forwarded as the instrumental hint).
    // NOTE: `lyrics` DEFAULTS to "[Instrumental]" — assign an empty string
    // explicitly for LM-written vocals, or every request stays instrumental.
    bool        simple_mode    = false;
    // Query Rewriting: the LM FORMAT pass rewrites `caption` into a detailed
    // musical description and regenerates `lyrics` preserving their content,
    // filling any unset metadata, before synthesis. Unlike simple_mode, the
    // full request (caption AND lyrics) is the input — use simple_mode for a
    // bare query. Requires text2music with no pre-supplied audio_codes and
    // lm_phase1; mutually exclusive with simple_mode. Faithful rewriting
    // needs the 1.7B LM; the 0.6B drifts genre/voice/language.
    bool        rewrite_query = false;
    // Synchronized lyric timestamps: after synthesis, one extra DiT forward at
    // the final timestep captures the lyric cross-attention heads and DTW
    // aligns each lyric line with the audio. The LRC text and its alignment
    // score land in GenerateResult::metadata. Requires lyrics (with Simple
    // Mode the LM-written lyrics are used) and is unavailable on the audio
    // edit path.
    bool        generate_lrc = false;
    // Teacher-forced LM quality scoring of the generated audio codes against
    // the resolved request (Simple Mode scores what the LM composed). Fills
    // GenerateMetadata::quality_score / quality_report at the cost of extra
    // LM forwards after code generation. Requires the LM code path, so it is
    // rejected for cover / lego tasks and on the audio edit path.
    bool        compute_quality_score = false;
    // Official sampler-side Haar DCW "double" correction. Applied on turbo
    // DiTs only: the official preset disables DCW for base/sft models.
    bool        dcw_enabled     = true;
    float       dcw_scaler      = 0.05f;  // low band coefficient: t * scaler
    float       dcw_high_scaler = 0.02f;  // high band coefficient: (1-t) * scaler

    // Optional timbre reference: normalized interleaved stereo PCM at 48 kHz.
    // The VAE encoder converts it to 25 Hz features consumed by the existing
    // condition encoder. Empty preserves the canonical silence reference.
    // For cover / cover-nofsq, empty falls back to source_audio (acestep.cpp
    // recommendation: pass the same buffer as --src-audio and --ref-audio).
    std::vector<float> reference_audio;

    // Optional source / cover audio: same layout as reference_audio. Required
    // when task_type is "cover" or "cover-nofsq". Encoded by the VAE into the
    // DiT context so generation follows the source structure.
    std::vector<float> source_audio;

    // Task discriminator (mirrors acestep.cpp AceRequest::task_type).
    // Supported today: "text2music" | "cover-nofsq" | "lego".
    // "cover" (FSQ roundtrip) is accepted at the API but not implemented yet.
    // "lego" generates a new instrument layer that follows source_audio and
    // returns only that layer; it requires a base DiT (turbo and sft are
    // rejected).
    std::string task_type = "text2music";

    // Lego target layer. Required when task_type is "lego"; one of:
    // vocals|backing_vocals|drums|bass|guitar|keyboard|percussion|strings|
    // synth|fx|brass|woodwinds.
    std::string track;

    // Fraction of DiT steps that keep the source context (0..1). Default 1.0
    // keeps source context for every step. Values < 1.0 switch the DiT to a
    // silence context and the text2music instruction at step
    // floor(steps * strength), so the tail of the run generates freely.
    float audio_cover_strength = 1.0f;

    // Blend initial DiT noise toward clean source latents (0..1). 0 = pure
    // Philox noise; 1 = nearly the source latent. Matches acestep.cpp.
    float cover_noise_strength = 0.0f;

    // Pre-supplied FSQ audio codes (LM output). When non-empty, the LM stage is
    // skipped and these codes are used directly (parity / caching / editing).
    // Ignored for cover / cover-nofsq (those skip the LM entirely).
    std::vector<int> audio_codes;

    std::vector<AudioEditParams> edit_plan;
};

// LM-enriched metadata surfaced alongside the audio (the same fields
// acestep.cpp writes into request0.json).
struct GenerateMetadata {
    std::string caption;         // enriched caption produced by the LM
    std::string lyrics;
    std::string keyscale;
    std::string vocal_language;
    int         bpm = 0;
    // atoi-style numeric prefix: "4/4" and "4foo" -> 4; no prefix -> 0.
    int         timesignature = 0;
    long long   seed = 0;
    int         n_codes = 0;
    // Filled when GenerateParams::generate_lrc is set: LRC-formatted lyric
    // timestamps and the alignment confidence score in [0, 1].
    std::string lrc;
    double      lyrics_score = 0.0;
    // Populated only when GenerateParams::compute_quality_score was set:
    // weighted global quality in [0, 1] (caption/lyrics PMI + metadata
    // recall) and its human-readable per-condition breakdown.
    double      quality_score = 0.0;
    std::string quality_report;
};

struct GenerateResult {
    std::vector<float> pcm;          // interleaved stereo, [t*2 + ch]
    int                sample_rate = 48000;
    int                channels    = 2;
    GenerateMetadata   metadata;
};

// Reverse pipeline (audio understanding): audio in, musical description out.
struct UnderstandParams {
    // Required: normalized interleaved stereo PCM at 48 kHz, [t*2 + ch].
    std::vector<float> audio;
    // Optional language hint (e.g. "es"); forced through the metadata FSM and
    // echoed to the result instead of the LM's guess.
    std::string vocal_language;
    // LM sampling. Defaults mirror the generation LM.
    float     lm_temperature = 0.85f;
    float     lm_top_p       = 0.9f;
    int       lm_top_k       = 0;   // 0 = disabled (top_p only)
    long long seed           = -1;  // <0 = random
};

// What the listener heard. Lyrics are intentionally NOT reported: the LM's
// transcription hallucinates on real songs, so the field is unsupported.
struct UnderstandResult {
    std::string      caption;         // descriptive caption of the audio
    int              bpm = 0;
    float            duration = 0.0f; // LM estimate in seconds (codes fix the true length)
    std::string      keyscale;
    std::string      timesignature;
    std::string      vocal_language;
    std::vector<int> audio_codes;     // recovered FSQ codes; reusable as GenerateParams::audio_codes
    long long        seed = 0;
};

// Optional progress callback: stage name
// ("reference"|"source"|"lm"|"score"|"tok"|"understand"|"dit"|"vae"),
// current step, total steps (total <= 0 when unknown). Return false to
// request cancellation.
using ProgressFn = std::function<bool(const std::string & stage, int step, int total)>;

class AUDIOGEN_API Engine {
public:
    // Validate model paths, GGUF metadata, and tokenizers. Stage weights load
    // lazily inside generate() and are released after use by default. A supported
    // truthy ACESTEP_KEEP_STAGES value eagerly loads and keeps every stage.
    // Throws std::runtime_error on missing/invalid models or allocation failure.
    static std::unique_ptr<Engine> create(const EngineOptions & opts);

    ~Engine();
    Engine(const Engine &)             = delete;
    Engine & operator=(const Engine &) = delete;

    // Generate music from a text prompt. Empty pcm on cancellation.
    GenerateResult generate(const GenerateParams & params, const ProgressFn & progress = {}) const;

    // Describe audio: VAE-encode it, FSQ-tokenize the latents, and let the LM
    // report metadata, caption, and the recovered codes. Stages report as
    // "source" (VAE encode), "tok", and "understand" (LM decode, unknown
    // total). Empty caption and codes on cancellation. Throws
    // std::runtime_error on invalid input or a failed stage.
    UnderstandResult understand(const UnderstandParams & params, const ProgressFn & progress = {}) const;

    // Cooperative cancel for an in-flight generate() or understand() on
    // another thread.
    void cancel() const;

    int         sample_rate() const;  // 48000
    std::string backend_name() const;

    // Why a run asked for a GPU and got the CPU. `not_requested` when
    // n_gpu_layers <= 0, `none` when a GPU backend was acquired.
    GpuFallbackReason gpu_fallback_reason() const;

private:
    Engine();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::acestep
