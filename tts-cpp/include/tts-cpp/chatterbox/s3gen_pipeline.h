#pragma once

// Back half of the Chatterbox pipeline: S3Gen encoder → 2-step meanflow CFM →
// HiFT vocoder. Takes T3-generated speech tokens + reference voice conditioning
// and writes a 24 kHz WAV.
//
// Implementation in src/chatterbox_tts.cpp.

#include "tts-cpp/export.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp::chatterbox {

inline constexpr int kS3GenSilenceToken    = 4299;
inline constexpr int kS3GenLookaheadTokens = 3;

} // namespace tts_cpp::chatterbox

struct s3gen_synthesize_opts {
    std::string s3gen_gguf_path;  // required: chatterbox-s3gen.gguf

    // Where to write the 24 kHz wav.  Required unless `pcm_out` is set, in
    // which case an empty string means "do not write any file" (streaming
    // drivers set this to avoid littering the filesystem with per-chunk
    // wavs they'd just read back in-memory).
    std::string out_wav_path;

    // Optional: if non-null, the pipeline writes the 24 kHz float32 mono PCM
    // samples here in addition to (or instead of) `out_wav_path`.  Lets
    // callers consume the audio without a temp-file round-trip.
    //
    // Semantics: clear-then-fill.  s3gen_synthesize_to_wav assigns the
    // produced PCM to *pcm_out by replacement, dropping any prior
    // contents.  Callers that want to accumulate across multiple calls
    // (e.g. one chunk per call from the streaming driver) should use a
    // fresh local std::vector<float> per call and concatenate themselves.
    std::vector<float> * pcm_out = nullptr;

    // If empty, use the built-in voice embedded in the GGUF
    // (s3gen/builtin/{embedding,prompt_token,prompt_feat}).
    // Otherwise load embedding.npy / prompt_token.npy / prompt_feat.npy from
    // this directory.
    std::string ref_dir;

    // Optional: if non-empty, override the prompt_feat tensor (S3Gen reference
    // mel spectrogram) with these values instead of loading it from
    // ref_dir/prompt_feat.npy or from s3gen/builtin. Layout is row-major
    // (T_mel, 80). Used by --reference-audio in main.cpp to inject a mel
    // computed natively in C++ from a reference wav.
    //
    // Owning storage; copied into / out of by callers that want full
    // value semantics (e.g. tts-cli, where the override is built once
    // and survives the s3gen_synthesize_to_wav call).
    std::vector<float> prompt_feat_override;
    int prompt_feat_rows_override = 0;

    // Optional: if non-empty, override the 192-d speaker `embedding` that's
    // produced by CAMPPlus.  Same motivation as prompt_feat_override: lets
    // main.cpp replace Python's embedding.npy with a C++ CAMPPlus output
    // when --reference-audio is given.
    std::vector<float> embedding_override;

    // Optional: if non-empty, override the S3Gen-side reference speech
    // tokens (`prompt_token`).  Populated from --reference-audio via
    // S3TokenizerV2 in main.cpp (Phase 2e).
    std::vector<int32_t> prompt_token_override;

    // Non-owning views over the same data.  Streaming hosts that hold
    // these tensors on a long-lived owner (e.g. chatterbox::Engine,
    // which bakes the voice profile once at construction) point these
    // at their internal storage to avoid the per-chunk MB-sized
    // value-copy that the *_override vectors would otherwise force.
    //
    // Lifetime: the caller owns the underlying data; the views must
    // stay valid for the duration of the s3gen_synthesize_to_wav call.
    // When set (non-null + non-zero size), each view takes precedence
    // over the matching *_override vector above; mixing the two for the
    // same field is undefined (don't).
    const float *   prompt_feat_view_data       = nullptr;
    size_t          prompt_feat_view_size       = 0;
    int             prompt_feat_view_rows       = 0;
    const float *   embedding_view_data         = nullptr;
    size_t          embedding_view_size         = 0;
    const int32_t * prompt_token_view_data      = nullptr;
    size_t          prompt_token_view_size      = 0;

    int  seed      = 42;
    int  n_threads = 0;          // 0 = hardware_concurrency
    int  sr        = 24000;
    bool debug     = false;      // validation mode; requires ref_dir
    bool verbose   = false;      // print per-stage wall times to stderr

    // When > 0, try to run S3Gen + HiFT on a GPU backend (CUDA / Metal / Vulkan
    // depending on what the build enables).  Falls back to CPU if the backend
    // cannot be initialised.  The actual layer count is not yet used for split
    // offload; any positive value enables the GPU path.
    int  n_gpu_layers = 0;

    // ---------------- streaming support (PROGRESS.md B1) ----------------
    //
    // Controls for chunked / streaming synthesis.  Defaults preserve the
    // original batch behaviour, so non-streaming callers can ignore them.
    //
    //   finalize                    mirrors Python's flow.inference(finalize).
    //                               When false, drop the last
    //                               `pre_lookahead_len * token_mel_ratio = 6`
    //                               mel frames from CFM output — they'll be
    //                               re-emitted on the next chunk with more
    //                               right-context.
    //
    //   append_lookahead_silence    whether to auto-pad `speech_tokens` with
    //                               3 S3GEN_SIL tokens before running the
    //                               encoder.  Streaming callers handle the
    //                               silence at the full-sequence level (once
    //                               at the end of the last chunk), so they
    //                               set this to false; batch callers leave
    //                               it true.
    //
    //   skip_mel_frames             offset into the "beyond-prompt" CFM mel
    //                               output.  Streaming caller sets this to
    //                               `mels_emitted_so_far` so each chunk
    //                               returns only the *new* mel frames it
    //                               contributes.  Defaults to 0.
    bool finalize                  = true;
    bool append_lookahead_silence  = true;
    int  skip_mel_frames           = 0;

    // Debug hook: if non-empty, dump the post-CFM mel (shape (T_mel_effective, 80)
    // as float32) to this path.  Used by the streaming validation harness
    // to compare each chunk's C++ mel against Python's chunk_{k}_mels_new.npy.
    std::string dump_mel_path;

    // Full CFM-initial-noise override.  When non-empty, the pipeline uses
    // these values verbatim instead of drawing from std::mt19937(seed).
    // Expected layout: row-major (80, T_mu) — the same shape the C++ z buffer
    // already uses internally, matching Python flow_matching's torch.randn_like(mu)
    // squeezed to (80, T_mu).
    //
    // Lets the streaming validation harness run a C++ chunk with the EXACT
    // noise Python used for the same chunk, getting bit-exact parity instead
    // of the rel~0.25 gap that comes from torch.randn vs std::mt19937
    // divergence.
    std::vector<float> cfm_z0_override;

    // ---------------- HiFT streaming (PROGRESS.md B1 phase 3) ----------------
    //
    // Chatterbox's HiFT vocoder (`HiFTGenerator.inference`) supports
    // cross-chunk continuity via a small "source cache": the last N samples of
    // the previous chunk's SineGen output (post `m_source` tanh), which get
    // pasted over the first N samples of the current chunk's source so F0
    // phase is continuous across the seam.  The first chunk additionally
    // applies a raised-cosine `trim_fade` to mask HiFT's resnet cold start.
    //
    //   hift_cache_source          When non-empty, overwrite the leading
    //                              `hift_cache_source.size()` samples of the
    //                              post-SineGen source with these values.
    //                              Callers pass the tail of the previous
    //                              chunk's source here.  Python uses a
    //                              480-sample (1 mel hop = 20 ms) overlap.
    //
    //   apply_trim_fade            When true, multiply the first
    //                              2 * (sr / 50) = 960 samples of the output
    //                              wav by a raised-cosine fade-in (first half
    //                              zero, second half 0→1).  Batch callers set
    //                              this to true to mask reference-audio
    //                              bleed-through; streaming callers set it
    //                              only on chunk 0.  Defaults to true.
    //
    //   hift_source_tail_out       Output slot: the pipeline writes the last
    //                              `source_tail_samples` values of the chunk's
    //                              post-SineGen source here so the caller can
    //                              feed them as `hift_cache_source` on the
    //                              next chunk.
    //
    //   source_tail_samples        Size of the tail slice to export.  Must
    //                              match `hift_cache_source.size()` on the
    //                              next call.  Defaults to 480 (1 mel hop).
    std::vector<float>   hift_cache_source;
    bool                 apply_trim_fade       = true;
    std::vector<float> * hift_source_tail_out  = nullptr;
    int                  source_tail_samples   = 480;

    // Number of Euler steps for the CFM meanflow sampler.  Python defaults
    // to 2 for meanflow; setting this to 1 halves CFM cost at the price of
    // some extra high-frequency noise.  0 → use the default (2).
    int                  cfm_steps             = 0;

    // Experimental OpenCL/mobile latency option: run CFM flash attention with
    // F32 Q and F16 K/V.  This may trade a small amount of quality for speed.
    bool                 cfm_f16_kv_attn       = false;

    // Optional cooperative-cancellation flag.  When non-null and set to
    // true mid-call, s3gen_synthesize_to_wav() bails out with a non-zero
    // exit code at the next safe checkpoint instead of running the full
    // pipeline through.  Cooperatively checked between CFM steps and
    // between HiFT upsample stages; in-flight ggml_backend_graph_compute
    // calls cannot be preempted, so cancel latency is bounded by the
    // longest single graph submission (one CFM step on multilingual,
    // ~80 ms on M3 Ultra Metal; longer on CPU).
    //
    // Lifetime: the caller owns the atomic; s3gen_synthesize_to_wav only
    // reads from it.  Engine::cancel() wires this to its internal
    // cancel_flag so a single Engine::cancel() reaches both the T3
    // decode loop and the S3Gen + HiFT path.
    const std::atomic<bool> * cancel_flag       = nullptr;
};

// Runs encoder + CFM + HiFT on the given T3 speech tokens and writes a WAV.
// Returns 0 on success, non-zero on error.
TTS_CPP_API int s3gen_synthesize_to_wav(
    const std::vector<int32_t> & speech_tokens,
    const s3gen_synthesize_opts & opts);

// Eagerly load the S3Gen GGUF into the internal model cache so the first
// call to s3gen_synthesize_to_wav skips the ~700 ms tensor-load cost.
// Useful for streaming pipelines: run this on a worker thread (or right
// after the T3 GGUF load) while T3 is still inferring, then the first
// streamed chunk is available as soon as T3 emits its first N tokens.
// Returns 0 on success.
//
// Lifetime contract: s3gen_preload / s3gen_unload form a refcounted
// pair.  Each successful s3gen_preload bumps an internal refcount;
// s3gen_unload decrements it, and the actual cache release runs only
// when the count reaches zero.  Multi-Engine hosts that share an S3Gen
// GGUF (e.g. two chatterbox::Engine instances loading the same path,
// or one Engine + a host that calls s3gen_preload directly) MUST call
// s3gen_preload + s3gen_unload in matched pairs so that one teardown
// doesn't clobber the cache an unrelated component still expects to
// hold.
//
// Direct callers of s3gen_synthesize_to_wav (the back-half pipeline)
// do NOT bump the refcount: the cache populates on first synth and is
// reused, but unbalanced unload from a parallel Engine teardown will
// force a re-load on the direct caller's next synth (~700 ms latency
// spike, not a crash).  Bracket direct synthesize_to_wav usage with
// an explicit s3gen_preload / s3gen_unload pair when running alongside
// chatterbox::Engine.
TTS_CPP_API int s3gen_preload(const std::string & s3gen_gguf_path, int n_gpu_layers);

// Release one reference to the internal S3Gen cache (weights + backend
// + allocator).  Decrements the refcount established by s3gen_preload;
// when the count reaches zero, frees the cached model_ctx, gallocator,
// and backend.  Idempotent: an unload-without-matching-preload clamps
// at zero and runs the release anyway (preserves the legacy
// "preload-once / unload-once-no-matter-what" pattern).
//
// Long-running processes that cycle through models, as well as wrappers
// that need deterministic teardown before the host backend is destroyed
// (e.g. the tts-cpp Bare addon), should call this before tearing down
// their own ggml backend.  Otherwise the cache is freed at process exit
// via static destructors, after which the ggml-metal global device may
// have already been finalised - tripping its resource-leak assertion.
TTS_CPP_API void s3gen_unload();
