#pragma once

#include <cstdint>
#include <array>
#include <cmath>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "backend_util.h"
#include "sched_dispatch.h"

namespace tts_cpp::supertonic::detail {

// round 4 — multi-dtype K/V flash-attention dispatch.
//
// Generalises the round-1 `use_f16_attn` boolean (F16 vs F32
// only) into a four-valued enum so operators can opt into BF16
// K/V (Vulkan coopmat2 — better quality than F16 at identical
// bandwidth, no underflow on small attention scores) or Q8_0 K/V
// (Vulkan + half the K/V upload bandwidth) when their adapter
// advertises the corresponding capability.
//
// Sentinel `autoselect` is used only on `EngineOptions::kv_attn_type`
// (= -1) and as a "not yet resolved" marker; the resolver
// always returns a concrete dispatch dtype (f32/f16/bf16/q8_0).
//
// Underlying-type-pinned int so the value can be cast cleanly
// to/from `EngineOptions::kv_attn_type` (also int, default -1).
//
// Declared up here (above `supertonic_model`) so the model can
// carry a `kv_attn_dtype` field without a forward declaration.
enum class kv_attn_dtype : int {
    autoselect = -1,
    f32        = 0,
    f16        = 1,
    bf16       = 2,
    q8_0       = 3,
};

struct supertonic_hparams {
    std::string arch = "supertonic2";
    std::string ftype = "f32";
    int sample_rate = 44100;
    int base_chunk_size = 512;
    int ttl_chunk_compress_factor = 6;
    int latent_dim = 24;
    int latent_channels = 144;
    int default_steps = 5;
    float default_speed = 1.05f;
    // Vector-estimator text cross-attention head count.  The only internal
    // topology dim that differs across families (v1/v2: 4 heads, v3: 8 heads;
    // head_dim stays 64).  Read from GGUF `supertonic.vector_text_attn_heads`,
    // defaulting to 4 for bundles converted before the key existed.
    int vector_text_attn_heads = 4;
    // Per-block dwconv dilations of the text-encoder ConvNeXt stack.  v1/v2 use
    // dilation 1 everywhere; Supertonic 3 dilates it (e.g. {1,1,2,2,4,4}).  Read
    // from GGUF `supertonic.text_convnext_dilations`.  Empty => every block uses
    // dilation 1 (the v1/v2 behaviour, and the fallback for older bundles whose
    // converter predates the key).
    std::vector<int> text_convnext_dilations;
    // Classifier-free-guidance scales for the vector estimator.  v3 bakes a
    // batch-2 (conditional + unconditional) field whose velocity is
    // `cfg_cond_scale*v_cond - cfg_uncond_scale*v_uncond` (e.g. 4*cond-3*uncond).
    // v1/v2 (and pre-key bundles) default to (1, 0) = single conditional pass.
    float cfg_cond_scale = 1.0f;
    float cfg_uncond_scale = 0.0f;
    bool cfg_enabled() const { return cfg_uncond_scale != 0.0f; }
    std::string language_wrap_mode = "open_close";
    std::string default_voice = "F1";

    // Dilation for text-encoder ConvNeXt block `block`, defaulting to 1 when the
    // bundle does not pin a per-block schedule (v1/v2, or pre-key conversions).
    int text_convnext_dilation(size_t block) const {
        return block < text_convnext_dilations.size() ? text_convnext_dilations[block] : 1;
    }
};

struct supertonic_voice_style {
    std::string name;
    ggml_tensor * ttl = nullptr; // (256, 50, 1) in ggml axis order for JSON (1, 50, 256)
    ggml_tensor * dp  = nullptr; // (16, 8, 1) in ggml axis order for JSON (1, 8, 16)
};

// round 7 — voice ttl/dp host cache.
//
// `Engine::Impl::synthesize()` historically downloaded the per-
// voice style tensors (`ttl`, `dp`) on EVERY call:
//
//     std::vector<float> style_ttl = read_tensor_f32(vit->second.ttl);
//     std::vector<float> style_dp  = read_tensor_f32(vit->second.dp);
//
// On Vulkan / OpenCL backends each `read_tensor_f32` is a
// synchronous GPU→host download.  The voice tensors are part of
// the load-time GGUF state and never mutate after load, so
// caching them per-engine keyed by voice name eliminates two sync
// points per `synthesize()` call after the first per-voice.
//
// This helper is intentionally extracted from `Engine::Impl` so
// the lookup-or-load semantics are testable on CPU without
// instantiating a full Engine.  See
// `test-supertonic-voice-host-cache` for the contract.
//
// Reference-stability contract: the returned `entry` reference is
// stable across subsequent `get_or_load` calls for OTHER voices
// (`std::unordered_map`'s reference-stability guarantee — element
// references survive `insert` even when the table rehashes; only
// iterators are invalidated).  Callers may hold the reference
// across the next `get_or_load` on the same instance, BUT must
// NOT call `clear()` or `erase()` on the cache while holding the
// reference.  The Engine::Impl call site captures `e.ttl.data()`
// / `e.dp.data()` and forwards them to the synthesis pipeline,
// which expects them to stay valid for the duration of the
// call — `clear()` is currently only reachable on Engine
// destruction (post-synthesis).
//
// THREAD-SAFETY (PR #18 review): voice_host_cache is NOT
// internally synchronised.  Concurrent invocations of any
// non-const method (`get_or_load`, `clear`) from multiple
// threads on the SAME instance is UB (standard `unordered_map`
// rules: writes need exclusive access).  The Engine's
// documented threading model is single-threaded synthesis per
// Engine instance; concurrent synthesis requires one Engine per
// thread (each Engine carries its own voice_host_cache), which
// is also what the iOS load/unload race fix (36a2c56) enforces
// for the s3gen preload path.  If a future refactor lifts that
// constraint (e.g. a thread-pool dispatch over a single
// Engine), the call site MUST add an external mutex around
// `voice_host_cache::get_or_load` + the downstream `.data()`
// capture, OR switch this cache to a `std::shared_mutex`-
// guarded internal lock.  Marked deliberately as caller's
// responsibility today because the single-threaded model also
// keeps the cache hot-path zero-cost (no atomic / lock-acquire
// per call) — the cache exists to eliminate per-call GPU
// downloads, and giving back any of that saving to internal
// locking would be premature.
struct voice_host_cache {
    struct entry {
        std::vector<float> ttl;
        std::vector<float> dp;
    };

    // Returns a stable reference to the cached entry for
    // `voice_name`.  On cache miss, calls `read_tensor_f32` on
    // `ttl_tensor` and `dp_tensor`, stores the result, and
    // returns the new entry.  On cache hit, returns the existing
    // entry without touching the GGML tensors at all (the host
    // vectors are reused as-is — `ttl_tensor` / `dp_tensor` may
    // legally be null on a cache hit).
    //
    // Throws std::runtime_error if the entry is missing AND
    // either tensor pointer is null (loud-failure for an Impl
    // bug; never expected to fire on the production path because
    // Impl validates `voices.find()` before calling).
    const entry & get_or_load(const std::string & voice_name,
                              ggml_tensor * ttl_tensor,
                              ggml_tensor * dp_tensor);

    // Drops every cached entry.  Currently only reachable on
    // Engine destruction; included for forward-compat with hot-
    // swap scenarios where the underlying backend is replaced
    // while the engine handle is reused.
    void clear();

    // Diagnostic — number of entries currently cached.  Used by
    // the test to assert lookup-vs-load semantics (size doesn't
    // grow on a cache hit).
    size_t size() const;

private:
    std::unordered_map<std::string, entry> by_name_;
};

// round 10 — pointer-compare upload-skip tracker.
//
// Background: per-step uploads of `text_emb` to the front-block
// cache and to the 3 group-graph caches happen 5 times per synth
// (once per denoise step), but `text_emb` is a host
// `std::vector<float>` allocated ONCE in
// `Engine::Impl::synthesize()` (and once per bench run) — so the
// SAME pointer flows through 4 caches × 5 steps = 20 uploads /
// synth, of which 16 are redundant re-uploads of identical data.
//
// The F4 pattern (already in `vector_res_style_qkv_cache` for
// `style_v_in` / `kctx_in`) skips redundant uploads via pointer
// comparison: if the host vector pointer is the same as the last
// successful upload's pointer, skip.  This struct generalises
// that pattern.
//
// CROSS-SYNTH HAZARD: `text_emb` lives on the
// `Engine::Impl::synthesize()` stack (or the bench loop's stack)
// — destructed at end of call.  Modern heap allocators
// (jemalloc / tcmalloc / glibc) very often return the SAME
// address for an immediately-following same-size allocation
// (size-class reuse, locality optimisation), so synth N+1 may
// have `text_emb.data() == synth_N.text_emb.data()` despite
// holding completely different data.  A naive pointer-compare
// upload-skip would silently send stale text-encoder embeddings
// to the next synth.
//
// MITIGATION: caller MUST invoke `reset()` at every synth
// boundary (i.e., when `current_step == 0`).  The first step of
// every synth always uploads (cold-miss), populating the
// tracker; steps 1..N-1 hit the pointer-compare and skip.
// Across synths, the reset invalidates the cached pointer so
// the next synth's upload always fires regardless of pointer
// match.
//
// Reset is also required after a cache rebuild (the underlying
// GPU buffer is reallocated and any cached upload-skip state is
// stale).  In tree, cache rebuilds happen via `cache = {}`
// which zero-initialises the tracker fields and effectively
// resets it without an explicit `reset()` call.
struct upload_skip_tracker {
    const void * last_uploaded = nullptr;

    // True iff `current` differs from the last recorded pointer
    // (i.e., we MUST upload).  False iff we can skip.  After
    // the consumer's upload call returns, they MUST call
    // `mark_uploaded(current)` to update the cached pointer
    // (else the next call re-uploads).
    bool needs_upload(const void * current) const {
        return current != last_uploaded;
    }

    // Records a successful upload.  Call AFTER the upload
    // completes (so a failed upload doesn't pin the pointer —
    // the next call would correctly re-attempt).
    void mark_uploaded(const void * current) {
        last_uploaded = current;
    }

    // Drops the cached pointer.  Caller invokes at synth
    // boundary (current_step == 0) AND on cache rebuild (cache
    // = {} also achieves this via zero-init of last_uploaded).
    void reset() {
        last_uploaded = nullptr;
    }
};

// round 7 — Vulkan env-var passthrough.
//
// Applies a map of `GGML_VK_*` env-var overrides via
// `set_env_if_unset` so the `init_supertonic_backend()` path
// picks them up at backend construction time.  `set_env_if_unset`
// semantics: an operator-set env var (already present in the
// environment when this is called) WINS over the EngineOptions
// override.  Lets a debugging operator force-disable a setting
// from the shell without recompiling, while still letting an
// EngineOptions configuration set the same knob in production.
//
// Throws std::runtime_error on a key that doesn't start with
// `GGML_VK_` (loud-failure for operator-config typos like
// `GMML_VK_PREFER_HOST_MEMORY`).  ALL-OR-NOTHING: validation
// happens BEFORE any env var is touched, so a partial-success
// can't leave the env in a half-applied state.
//
// Pass an empty map for a no-op (the default
// `EngineOptions::vulkan_env_overrides` value).
//
// Must be called BEFORE `init_supertonic_backend()` runs; called
// from `Engine::Impl` ctor and from `supertonic-bench` main right
// before `load_supertonic_gguf()`.
void apply_vulkan_env_overrides(const std::map<std::string, std::string> & overrides);

struct supertonic_vocoder_convnext_weights {
    ggml_tensor * dw_w = nullptr;
    ggml_tensor * dw_b = nullptr;
    ggml_tensor * norm_g = nullptr;
    ggml_tensor * norm_b = nullptr;
    ggml_tensor * pw1_w = nullptr;
    ggml_tensor * pw1_b = nullptr;
    ggml_tensor * pw2_w = nullptr;
    ggml_tensor * pw2_b = nullptr;
    ggml_tensor * gamma = nullptr;
};

struct supertonic_vocoder_weights {
    ggml_tensor * normalizer_scale = nullptr;
    ggml_tensor * latent_mean = nullptr;
    ggml_tensor * latent_std = nullptr;
    ggml_tensor * embed_w = nullptr;
    ggml_tensor * embed_b = nullptr;
    std::array<supertonic_vocoder_convnext_weights, 10> convnext{};
    ggml_tensor * final_norm_g = nullptr;
    ggml_tensor * final_norm_b = nullptr;
    ggml_tensor * final_norm_running_mean = nullptr;
    ggml_tensor * final_norm_running_var = nullptr;
    ggml_tensor * head1_w = nullptr;
    ggml_tensor * head1_b = nullptr;
    ggml_tensor * head_prelu = nullptr;
    ggml_tensor * head2_w = nullptr;

    // Audit finding F2 — pre-baked vocoder BN scale + shift.
    //
    //   bn_scale_pre[c] = final_norm_g[c] / sqrt(final_norm_var[c] + 1e-5)
    //   bn_shift_pre[c] = final_norm_b[c] - final_norm_mean[c] * bn_scale_pre[c]
    //
    // Both are constants for the model lifetime; pre-computing once
    // at `load_supertonic_gguf()` time and uploading into a small
    // dedicated backend buffer avoids the per-synth pattern of:
    //
    //   - 4 × `ggml_backend_tensor_get` (final_norm_g/b/mean/var, 512 floats each)
    //   - host-side 512-element scale/shift compute
    //   - 2 × `ggml_backend_tensor_set` (bn_scale_in/bn_shift_in graph inputs)
    //
    // The vocoder graph cache references these tensors directly
    // (no `ggml_set_input` markers needed — they're weights, not
    // graph inputs).  See AUDIT_SUPERTONIC_OPENCL.md F2 + PLAN
    // Phase 2F.
    ggml_tensor * bn_scale_pre = nullptr;
    ggml_tensor * bn_shift_pre = nullptr;
};

struct supertonic_trace_tensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct supertonic_model {
    supertonic_hparams hparams;
    supertonic_vocoder_weights vocoder;

    uint64_t generation_id = 0;
    int n_threads = 0;
    ggml_backend_t backend = nullptr;
    // Scheduler bundle [backend, CPU-last] (sched_dispatch.h) so ops the GPU
    // backend can't run (notably GGML_OP_CUSTOM CPU kernels in the vector
    // estimator / vocoder) auto-route to CPU instead of being silently
    // skipped on a single backend; degenerate [backend]-only sched when the
    // primary is itself CPU (cpu_backend stays null).  Created lazily by
    // supertonic_sched_alloc on the first sched-needing dispatch, not at
    // load.  mutable: the dispatch helpers take const supertonic_model &
    // (same precedent as chatterbox).  supertonic_model is never
    // moved/copied, so the non-movable sched_fallback is a plain member.
    mutable ::tts_cpp::detail::sched_fallback sched_fb;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;

    // True when the resolved compute backend is the GGML CPU backend; the
    // BLAS-backed `ggml_custom_4d` fast paths in the vocoder / vector
    // estimator depend on the backend's CPU-side scheduler invoking the
    // op callbacks and the tensor data pointers being host-addressable.
    // On any non-CPU backend (CUDA / Metal / Vulkan / OpenCL) the runtime
    // must take the pure-GGML fallback path instead — that's what the
    // supertonic_op_dispatch_scope below toggles inside the graph-build
    // helpers.  Set once in load_supertonic_gguf() right after
    // init_supertonic_backend() resolves the device and is stable for
    // the lifetime of the model.  See `OpenCL bring-up` section in
    // PROGRESS_SUPERTONIC.md for the rationale.
    bool backend_is_cpu = true;
    // True when GPU was requested but a present GPU device was unusable (e.g. an
    // Android GPU outside the allowlist) and we fell back to CPU; distinct from a GPU-less host.
    bool gpu_unsupported = false;
    // / Vulkan bring-up: True when the resolved backend is
    // ggml-vulkan (`ggml_backend_is_vk`).  Mirrors `backend_is_cpu` in
    // intent — informational + dispatch-key.  Set once in
    // load_supertonic_gguf() right after the backend is resolved.
    // Stable for the model lifetime.  Used by supertonic_bench /
    // engine.cpp for the human-readable backend description (so the
    // bench log shows "Vulkan (device 0: NVIDIA RTX 5090)" instead
    // of just "Vulkan") and by the dispatch helpers below to pick
    // between the OpenCL-conservative `leaky_relu_portable_ggml`
    // decomposition and the native `ggml_leaky_relu` op.  See the
    // PROGRESS_SUPERTONIC.md "Vulkan bring-up" section for the
    // rationale + supported-op matrix.
    bool backend_is_vk = false;
    // backend supports `GGML_OP_LEAKY_RELU` natively.
    // Resolved at load time via `ggml_backend_supports_op` against
    // a synthetic LEAKY_RELU node.  Three reasons we don't piggy-
    // back on `backend_is_cpu`:
    //   1. CPU obviously supports it (builtin); we want the same flag
    //      to ride the CPU path through the helper without a special
    //      case.
    //   2. Vulkan / Metal / CUDA support it natively (verified against
    //      ggml-vulkan.cpp:`pipeline_leaky_relu_f32`,
    //      ggml-metal:`kernel_leaky_relu_f32`,
    //      ggml-cuda:`leaky_relu`).
    //   3. Plain upstream ggml-opencl does NOT support it; chatterbox
    //      ships a patch that adds the kernel (see chatterbox
    //      PROGRESS.md "What was missing"), but that patch may or may
    //      not be applied at the consumer's vendored ggml.
    // The dynamic `ggml_backend_supports_op` query handles all four
    // cases without a hard-coded backend table.  When the query
    // returns `false`, `leaky_relu_portable_ggml` decomposes into
    // RELU + SCALE + ADD (universally supported, slightly more
    // dispatches).  When it returns `true`, the helper emits the
    // single fused builtin — fewer dispatches, lower scheduler
    // overhead on the GPU command-buffer side.  Default `true`
    // matches the historical CPU-only path.
    bool use_native_leaky_relu = true;
    // True when the resolved backend implements the Supertonic fused custom
    // ops (GGML_OP_SUPERTONIC_DEPTHWISE_1D / LAYER_NORM_CHANNEL / EDGE_PAD_1D /
    // BIAS_GELU / PW2_RESIDUAL).  Those are implemented in ggml-cpu and
    // ggml-metal only — NOT ggml-vulkan / ggml-opencl.  Resolved at load time
    // via `ggml_backend_supports_op` against a synthetic depthwise_1d node
    // (the 5 ops ship as one overlay set, so one probe is representative).
    // The graph-build helpers (depthwise_same_ggml / layer_norm_ggml /
    // edge_clamp_pad_1d / bias_gelu_ggml / pw2_residual_ggml in both the text
    // encoder and vector estimator, plus vector_convnext_ggml_ct) gate the
    // fused fast path on this; when false they emit the pure-GGML decomposition
    // (im2col+mul_mat / ggml_norm / concat+repeat / add+gelu / matmul+scale+add)
    // which every backend executes.  Without this gate the fused ops are
    // silently skipped on Vulkan/OpenCL single-backend graphs → garbage output.
    // Default false = safe (pure-GGML) until the load-time probe runs.
    bool backend_supports_fused_supertonic_ops = false;
    // ARM Mali/Valhall Vulkan miscomputes a GEMM mul_mat whose output dim < ~48; set via
    // device-identity (not supports_op: driver claims support). st_mul_mat pads to 64; harmless elsewhere.
    bool mulmat_needs_pad = false;
    // When true, the per-step vector-estimator attention graphs materialise
    // K/V into contiguous F16 before calling ggml_flash_attn_ext so OpenCL
    // (and other backends carrying the mixed-precision kernel) dispatch
    // the `flash_attn_f32_f16` path instead of the F32-only one — large
    // win on Adreno (see chatterbox PROGRESS.md OpenCL log).  Defaults to
    // false on CPU (the cblas attention path is already efficient there);
    // engine.cpp auto-enables it when the resolved backend is non-CPU,
    // matching chatterbox's --cfm-f16-kv-attn behaviour.  On Vulkan the
    // F16 K/V path goes through `kernel_flash_attn_*` shaders that
    // accept any HSK / HSV that's a multiple of 8 (see
    // ggml-vulkan.cpp `GGML_OP_FLASH_ATTN_EXT` supports_op gate);
    // Supertonic's head_dim=64 satisfies that constraint by
    // construction.
    bool use_f16_attn = false;

    // Phase 2A — load-time F16 materialization for the hot
    // matmul / pointwise-conv weights identified by
    // `should_materialise_f16_weight`.  Halves the GPU read
    // bandwidth into those ops on non-CPU backends.  Captured on
    // the model state at load time so the graph builders can fall
    // back through `repeat_like(model.vocoder.bn_scale_pre, …)`-
    // style casts when a tensor's storage type changed.  Auto-
    // enables on GPU backends, off on CPU (mirrors `use_f16_attn`).
    // Override via `EngineOptions::f16_weights` / `--f16-weights`.
    bool use_f16_weights = false;

    // The compute precision the model was loaded with — set by
    // `load_supertonic_gguf`.  Lets graph builders dispatch precision-
    // specific code paths (e.g. asymmetric q8_0 load on Metal).
    // Orthogonal to `use_f16_weights` above (that's a per-op runtime
    // selector for the OpenCL hot-weight materialisation; this is the
    // global storage-type selector).
    int precision_id = 0; // supertonic_precision::F32

    // round 6 — count of tensors that the curated allow-
    // list would have promoted to F16 but the user-supplied
    // `f16_weights_deny_list` excluded.  Surfaced in bench output
    // so operators can confirm their deny-list took effect.  Zero
    // for the default empty deny-list path (zero behaviour change).
    int f16_weights_excluded_count = 0;

    // round 4 — resolved K/V flash-attention dispatch
    // dtype.  Default `f32` (no surprise dispatch on a default-
    // constructed model).  `load_supertonic_gguf` resolves the
    // policy from `EngineOptions::kv_attn_type` + the round-2/3
    // backend probes via `resolve_kv_attn_type` and sets this.
    // The `supertonic_op_dispatch_scope` mirrors it onto the
    // thread-local accessor read by the vector-estimator
    // dispatch site.
    //
    // Forward-compat note: when `kv_attn_type != f32`, the
    // legacy `use_f16_attn` boolean above is ALSO updated to
    // `(kv_attn_type == f16)` so any code path still keying on
    // the boolean (text-encoder / duration / vocoder) sees the
    // historically-correct value.  The vector estimator (the
    // only consumer that gains from the multi-dtype dispatch)
    // reads `kv_attn_type` directly.
    kv_attn_dtype kv_attn_type = kv_attn_dtype::f32;

    std::map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, ggml_tensor *> source_tensors;
    std::unordered_map<std::string, supertonic_voice_style> voices;

    // Pre-transposed copies of matmul weights, materialized at load time
    // to eliminate the per-call `cont(transpose(w))` dispatch that
    // `dense_matmul_time_ggml` issues on every graph compute.  Keyed by
    // the ORIGINAL weight tensor pointer (i.e. the value in
    // `source_tensors[<MatMul_*>]`); the mapped value is the transposed
    // f32 copy with `ne = [IC, OC]` and lives in `ctx_w_extra` /
    // `buffer_w_extra`.  Lookup via `try_pretransposed_weight(model, w)`.
    ggml_context * ctx_w_extra = nullptr;
    ggml_backend_buffer_t buffer_w_extra = nullptr;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> pretransposed_weights;

    std::vector<int32_t> unicode_indexer;
    std::vector<std::string> languages;
    std::string tts_json;

    // ----- OpenCL optimization caches (audit F1 / F9) -----
    //
    // F1: cached copy of the vector-estimator RoPE θ tensor (the
    // `vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta`
    // entry).  All four group attention sites in the production GGML
    // path read from the same source tensor; caching once at load
    // saves 4 × N_STEPS GPU→host downloads per synth on a non-CPU
    // backend.  Empty if the GGUF doesn't carry the theta tensor.
    // Populated unconditionally at load time so call sites can use
    // it without a fallback.
    std::vector<float> vector_rope_theta;

    // F9: per-(current_step, total_steps) cache of
    // `time_embedding(model, …)` outputs.  The vector denoising
    // schedule fires at most `total_steps` distinct (current, total)
    // pairs per synth; cache hit rate is ≥(steps − 1) / steps once
    // warm.  `mutable` because the cache populates lazily on
    // const-method paths; thread-unsafe by design (matches the rest
    // of supertonic_model: one engine per thread).  Key is
    // `(current << 32) | total`.
    mutable std::unordered_map<uint64_t, std::array<float, 64>> time_emb_cache;

    // ----- Audit follow-up #2 caches (F13 / F16) -----
    //
    // F13: text-encoder LN weight host-side cache.  The text-encoder
    // GGML production path runs four relpos + LN + FFN + LN
    // iterations followed by a final speech-prompted LN; the LN
    // step on each iteration calls the scalar `layer_norm_channel`
    // which used to download γ + β from the backend on every call
    // (~18 GPU→host downloads / synth on a non-CPU backend).
    // Populated at `load_supertonic_gguf` time from
    // `text_encoder:...attn_encoder.norm_layers_{1,2}.{0..3}.norm.{weight,bias}`
    // plus the final `speech_prompted_text_encoder.norm.norm.*`.
    // Keyed by the source-tensor name so the call-site rewrite
    // becomes `auto & v = model.text_encoder_ln_weights[name]`.
    // Empty entries fall back to `read_f32(model, name)` so a GGUF
    // missing one of the rostered names degrades gracefully.
    std::unordered_map<std::string, std::vector<float>> text_encoder_ln_weights;

    // F16: speech-prompted attention `tanh_k` host-side cache.
    // Indexed by attention layer (0 or 1).  Source tensors:
    //   speech_tanh_k_cache[0] ←
    //     "text_encoder:/speech_prompted_text_encoder/attention1/tanh/Tanh_output_0"
    //   speech_tanh_k_cache[1] ←
    //     "text_encoder:/speech_prompted_text_encoder/attention2/tanh/Tanh_output_0"
    // Each ≈ 50 × 256 = 51.2 KiB; saves 2 sync points + ~100 KiB
    // of redundant traffic per synth.
    std::array<std::vector<float>, 2> speech_tanh_k_cache;

    // ----- Audit follow-up #3 cache (F17) -----
    //
    // F17: generic lazy host-side cache for any source weight that
    // a scalar-CPU continuation needs.  The duration stage's
    // post-graph scalar attention (relpos K/V embeddings, conv_o,
    // 4 LN pairs, 2 FFN's conv_{1,2} pairs, proj_out weight) — and
    // any future stage that uses `cached_read_f32` — populates
    // this on first touch.  Keyed by the source-tensor name; value
    // is the F32 byte payload sized to `ggml_nelements(src)`.
    //
    // Memory cost: bounded by the union of stages' scalar-
    // continuation weight footprints.  Empirically ~3-5 MB on a
    // Supertonic-2 GGUF, vs. the savings of ~30 GPU→host syncs per
    // duration synth (+ ~15 from the text-encoder LN cache (F13)
    // and the speech tanh_k cache (F16) already shipped).
    //
    // `mutable` because the cache populates lazily on const-method
    // paths; thread-unsafe by design (one engine per thread).
    mutable std::unordered_map<std::string, std::vector<float>> scalar_weight_cache;
};

// `f16_weights`:
//   -1 → auto (on when the resolved backend is non-CPU, off on CPU).
//    0 → force off (every hot weight stays at its GGUF storage type).
//    1 → force on  (every hot weight matching
//        `should_materialise_f16_weight` is allocated as F16,
//        regardless of backend).
// See Phase 2A in `aiDocs/PLAN_SUPERTONIC_OPENCL.md` for the
// roster + auto-policy rationale.
//
// `precision` (separate concern): selects the storage type for
// matmul weights at GGUF load time.  Mirrors the public
// `tts_cpp::supertonic::Precision` enum.  F32 is the historical
// default; Q8_0 / F16 trigger asymmetric loads on Metal.
enum class supertonic_precision {
    F32 = 0,
    F16 = 1,
    Q8_0 = 2,
};

// `vulkan_device`:
//   ≥ 0 → adapter index passed to `ggml_backend_vk_init(idx)`.
//        Range-checked against `ggml_backend_vk_get_device_count()`;
//        an out-of-range index is a hard error (no silent CPU
//        fallback — that would mask CLI typos / wrong-machine
//        config).  Default 0 (the historical hard-coded value).
//   < 0 → reserved for future "auto-pick best device" behaviour;
//        treated as 0 today.
// Has no effect when the build wasn't compiled with `GGML_VULKAN`
// or when `n_gpu_layers <= 0`.
// round 6 — `f16_weights_deny_list`:
//   Extra deny-list (substring patterns) for the F16-weights
//   materialization predicate.  Layered ON TOP of the curated
//   allow-list in `should_materialise_f16_weight()`.  Empty
//   default → zero behaviour change for every existing call site.
//   See `EngineOptions::f16_weights_deny_list` for the full
//   contract + use cases.
bool load_supertonic_gguf(const std::string & path,
                          supertonic_model & model,
                          int n_gpu_layers = 0,
                          bool verbose = false,
                          int f16_weights = -1,
                          supertonic_precision precision = supertonic_precision::F32,
                          int vulkan_device = 0,
                          const std::vector<std::string> & f16_weights_deny_list = {});
void free_supertonic_model(supertonic_model & model);
void supertonic_set_n_threads(supertonic_model & model, int n_threads);
void supertonic_graph_compute(const supertonic_model & model, ggml_cgraph * graph);

// Per-TU thread-local cache release helpers — called from
// `free_supertonic_model` BEFORE `ggml_backend_free`, so the
// gallocators inside each per-stage thread_local graph cache
// hit their normal `ggml_gallocr_free` path against a live
// backend instead of the dead-backend skip in
// `supertonic_safe_gallocr_free` (the skip is correct — it
// avoids a crash in the backend dylib finaliser — but each
// skipped free leaks the gallocr's internal hash tables /
// per-leaf records / per-buffer-type allocators, on the order
// of several KB per cache).
//
// Multi-thread caveat: each helper only releases caches on
// the CALLING thread.  Other threads that have populated their
// own thread_local caches (multi-threaded synth hosts that
// share an Engine across threads) fall back to the lazy
// release-on-next-cache-miss path with the skip-and-leak
// semantics.  This matches the documented Engine threading
// contract (one Engine per thread; concurrent `synthesize()`
// on the same Engine is not safe) so the SDK pattern is
// fully covered.  See `register_supertonic_alive` for the
// per-engine generation_id machinery the dead-backend skip
// hangs off of.
void release_vector_estimator_thread_local_caches();
void release_text_encoder_thread_local_caches();
void release_vocoder_thread_local_caches();
void release_duration_thread_local_caches();

// True when the model's compute backend supports the per-stage CPU fast paths
// (the `ggml_custom_4d` callbacks in conv1d_f32 / depthwise_same_ggml /
// layer_norm_ggml etc.).  ggml custom ops are CPU-only by design; on Metal /
// CUDA / Vulkan the helpers must fall through to their stock-ggml-op paths.
// Mirrors the `!ggml_backend_is_cpu(backend)` idiom Chatterbox uses to gate
// its Metal-only batched-CFG path.
inline bool model_prefers_cpu_kernels(const supertonic_model & model) {
    // `ggml_backend_is_cpu` lives in the CPU backend shared library, which is
    // unlinkable under GGML_BACKEND_DL. Route through the registry-based shim.
    return model.backend == nullptr || ::tts_cpp::detail::backend_is_cpu(model.backend);
}

// scheduler-based alloc + compute (Option A), used by stages
// migrated off the per-graph ggml_gallocr.  Pairing contract at each call
// site:
//   supertonic_sched_alloc(model, gf);            // reset + allocate via sched
//   ggml_backend_tensor_set(input_leaf, ...);     // inputs now have memory
//   supertonic_sched_compute(model, gf);          // run (routes customs -> CPU)
// GRAPH LIFETIME CONTRACT: a graph fed to supertonic_sched_alloc is
// SINGLE-USE — sched alloc rewires node->src[] into sched-owned copy
// tensors that the NEXT sched pass frees, and tensor buffer/data bindings
// survive sched_reset, so re-feeding the same graph computes garbage
// deterministically (or crashes on a multi-backend sched).  Every dual-path
// run_* site therefore REBUILDS its graph before each sched pass (the sched
// route leaves cache.allocr null, so the build early-return never reuses),
// and *_gpu cross-graph tensor handles must only ever come from a
// DIRECT-run producer (sched-owned slabs are reset/re-planned by the
// consumer's own sched pass).  See ggml-backend.h ("single-use in terms of
// allocation" / sched_reset docs) and
// docs/supertonic-sched-graph-reuse-investigation.md.
void supertonic_sched_alloc(const supertonic_model & model, ggml_cgraph * graph);
void supertonic_sched_compute(const supertonic_model & model, ggml_cgraph * graph);

// Dispatch gate shared by every dual-path stage: supports_op walk over the
// graph + the TTS_CPP_FORCE_SCHED escape hatch (safe: every dual-path site
// rebuilds its graph before a sched pass — see the contract note above —
// so the forced path is bit-identical to direct;
// test-supertonic-sched-equivalence enforces that on CPU and Metal).
bool supertonic_use_sched(const supertonic_model & model, const ggml_cgraph * graph);

ggml_tensor * require_tensor(const supertonic_model & model, const std::string & name);
ggml_tensor * require_source_tensor(const supertonic_model & model, const std::string & source_name);
ggml_tensor * try_source_tensor(const supertonic_model & model, const std::string & source_name);

// Look up a pre-transposed copy of a matmul weight.  Returns nullptr if no
// pre-transposed copy was materialized for `w` at load time (e.g. CPU backend
// — pre-transposition is a Metal-perf-only optimization).  When non-null, the
// returned tensor has `ne = [IC, OC]` (the swapped layout of `w`), is f32 and
// contiguous in `model.buffer_w_extra`.  Callers should reshape it as the
// conv1d kernel `[K=1, IC, OC]` directly and skip the cont(transpose(w)).
ggml_tensor * try_pretransposed_weight(const supertonic_model & model, const ggml_tensor * w);

// `supported_languages`, when non-null and non-empty, is the authoritative set
// of accepted language codes (the model's `supertonic.languages` GGUF array).
// When null/empty the function falls back to the legacy built-in 5-language
// allowlist so direct callers/tests without a model keep working.
std::string supertonic_preprocess_text(const std::string & text,
                                       const std::string & language,
                                       const std::string & language_wrap_mode,
                                       bool is_continuation = false,
                                       const std::vector<std::string> * supported_languages = nullptr);
bool supertonic_text_to_ids(const supertonic_model & model,
                            const std::string & text,
                            const std::string & language,
                            std::vector<int32_t> & ids,
                            std::string * normalized_text = nullptr,
                            std::string * error = nullptr,
                            bool is_continuation = false);

bool supertonic_vocoder_forward_cpu(const supertonic_model & model,
                                    const float * latent,
                                    int latent_len,
                                    std::vector<float> & wav_out,
                                    std::string * error = nullptr);

bool supertonic_vocoder_forward_ggml(const supertonic_model & model,
                                     const float * latent,
                                     int latent_len,
                                     std::vector<float> & wav_out,
                                     std::string * error = nullptr);

bool supertonic_vocoder_trace_scalar(const supertonic_model & model,
                                     const float * latent,
                                     int latent_len,
                                     std::vector<supertonic_trace_tensor> & trace_out,
                                     std::string * error = nullptr);

bool supertonic_vocoder_trace_ggml(const supertonic_model & model,
                                   const float * latent,
                                   int latent_len,
                                   std::vector<supertonic_trace_tensor> & trace_out,
                                   std::string * error = nullptr);

bool supertonic_duration_forward_cpu(const supertonic_model & model,
                                     const int64_t * text_ids,
                                     int text_len,
                                     const float * style_dp,
                                     float & duration_out,
                                     std::string * error = nullptr);

bool supertonic_duration_forward_ggml(const supertonic_model & model,
                                      const int64_t * text_ids,
                                      int text_len,
                                      const float * style_dp,
                                      float & duration_out,
                                      std::string * error = nullptr);

bool supertonic_duration_trace_ggml(const supertonic_model & model,
                                    const int64_t * text_ids,
                                    int text_len,
                                    std::vector<supertonic_trace_tensor> & scalar_trace,
                                    std::vector<supertonic_trace_tensor> & ggml_trace,
                                    std::string * error = nullptr,
                                    bool include_scalar_trace = true,
                                    bool include_ggml_trace = true,
                                    std::vector<float> * sentence_proj_out = nullptr);

bool supertonic_text_encoder_forward_cpu(const supertonic_model & model,
                                         const int64_t * text_ids,
                                         int text_len,
                                         const float * style_ttl,
                                         std::vector<float> & text_emb_out,
                                         std::string * error = nullptr);

bool supertonic_text_encoder_forward_ggml(const supertonic_model & model,
                                          const int64_t * text_ids,
                                          int text_len,
                                          const float * style_ttl,
                                          std::vector<float> & text_emb_out,
                                          std::string * error = nullptr);

// round 12 #6 — text-encoder speech-prompted-attention
// GPU bridge.
//
// Master's Metal-port branch (PR #15) shipped a fully-built
// `speech_prompted_merged_cache` graph in
// `supertonic_text_encoder.cpp` — one ggml graph that does QKV
// projection + head-split + flash-attn + out-proj end-to-end on
// the GPU.  The graph builder
// (`build_speech_prompted_merged_cache`) was present + reviewed
// at the implementation level but the run path was never wired
// in.  So the production text-encoder path stayed on the pre-
// Phase-A4 two-cache pattern with host-side Q/V download →
// pack → re-upload between the QKV cache and the flash-attn
// cache (5 sync points × 2 layers per synth).
//
// Round 12 adds `run_speech_prompted_merged_cache` and switches
// the dispatch in `speech_prompted_attention_ggml` to use it on
// non-CPU backends.  CPU stays on the legacy two-cache path
// because that path leans on the host BLAS fast path for the
// QKV matmuls and downstream scalar code keeps the host-side
// head-split as a free-ish memcpy.  Saves 10 sync points /
// synth on Vulkan / OpenCL / Metal.
//
// Struct + helpers exposed via the header so a CPU-only unit
// test can SFINAE-pin the field contract + free-default
// destructor without dragging the whole text-encoder TU into
// the test binary.
struct speech_prompted_merged_cache {
    const supertonic_model * model = nullptr;
    uint64_t generation_id = 0;
    int idx = -1;
    int L = 0;
    int Lctx = 0;
    std::string out_w_source;
    std::string out_b_source;
    std::vector<uint8_t> buf;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor * x_in = nullptr;       // ne=[L, C], channel-major-flat memory
    ggml_tensor * style_in = nullptr;   // ne=[Lctx, C], same memory layout
    ggml_tensor * out = nullptr;        // ne=[L, C] result, channel-major-flat
};

void free_speech_prompted_merged_cache(speech_prompted_merged_cache & cache);

void build_speech_prompted_merged_cache(speech_prompted_merged_cache & cache,
                                        const supertonic_model & m,
                                        int idx,
                                        int L,
                                        int Lctx,
                                        const std::string & q_w_source,
                                        const std::string & v_w_source,
                                        const std::string & out_w_source,
                                        const std::string & out_b_source,
                                        const std::string & tanh_k_source,
                                        const std::string & q_b_source,
                                        const std::string & v_b_source);

// Round 12: run the merged graph once with the given host-side
// `x_lc` / `style_ttl` inputs.  Caller MUST have ensured the
// cache is built (`build_speech_prompted_merged_cache`) AND keyed
// against the current `(model, idx, L, Lctx)`.  This is the
// drop-in replacement for the legacy two-cache path inside
// `speech_prompted_attention_ggml` — same input / output
// conventions (`x_lc`, `out_lc` are time-major-flat `[t*C + c]`).
//
// `style_ttl` is also time-major-flat (`style_ttl[t*C + c]`),
// matching the layout `speech_prompted_attention_ggml`'s caller
// in `supertonic_text_encoder_forward_ggml` passes.
void run_speech_prompted_merged_cache(speech_prompted_merged_cache & cache,
                                       const supertonic_model & m,
                                       const std::vector<float> & x_lc,
                                       int L,
                                       const float * style_ttl,
                                       std::vector<float> & out_lc);

bool supertonic_text_encoder_trace_ggml(const supertonic_model & model,
                                        const int64_t * text_ids,
                                        int text_len,
                                        std::vector<supertonic_trace_tensor> & scalar_trace,
                                        std::vector<supertonic_trace_tensor> & ggml_trace,
                                        std::string * error = nullptr);

bool supertonic_vector_step_cpu(const supertonic_model & model,
                                const float * noisy_latent,
                                int latent_len,
                                const float * text_emb,
                                int text_len,
                                const float * style_ttl,
                                const float * latent_mask,
                                int current_step,
                                int total_steps,
                                std::vector<float> & next_latent_out,
                                std::string * error = nullptr);

bool supertonic_vector_step_ggml(const supertonic_model & model,
                                 const float * noisy_latent,
                                 int latent_len,
                                 const float * text_emb,
                                 int text_len,
                                 const float * style_ttl,
                                 const float * latent_mask,
                                 int current_step,
                                 int total_steps,
                                 std::vector<float> & next_latent_out,
                                 std::string * error = nullptr);

// Audit finding F9 — `time_embedding(model, current, total)` is a
// pure function over (current_step, total_steps) whose output (64
// floats) is reused once per group inside the vector estimator.
// `cached_time_embedding` populates `model.time_emb_cache` on first
// touch and returns a stored reference on every subsequent call
// with the same key.  Steady-state per-synth recomputation cost
// drops from `total_steps` invocations to zero after the first
// synth.  See PLAN_SUPERTONIC_OPENCL.md Phase 2F.
std::array<float, 64> cached_time_embedding(const supertonic_model & model,
                                            int current_step,
                                            int total_steps);

// Phase 2A — hot-weight predicate for F16 materialization.
//
// Returns `true` when `source_name` (the
// `<stage>:<onnx-or-pytorch-path>` source key in
// `model.source_tensors`) names one of the bandwidth-bound matmul /
// pointwise-conv weights identified by the audit, and the load-time
// hook should allocate it as `GGML_TYPE_F16` instead of `F32` when
// `model.use_f16_weights` is on.  Pure function over the string; no
// model state needed.  Documented in test_supertonic_f16_weights.cpp
// with explicit positive + negative + edge-case rosters.
//
// Conservative roster:
//   - vector_estimator attention W_query/W_key/W_value/W_out matmul
//     weights (only those whose source name matches `onnx::MatMul_NNNN`
//     where NNNN ∈ {3101..3110, 3116..3119, 3146..3155, 3161..3164,
//                   3191..3200, 3206..3209, 3236..3245, 3251..3254}).
//   - vector_estimator pwconv1/pwconv2 inside every convnext block,
//     including `last_convnext`.
//   - vocoder convnext pwconv1/pwconv2 + `head.layer1.net.weight`.
//   - text-encoder linear weights `text_encoder:onnx::MatMul_*` and
//     the per-layer FFN conv1/conv2 weights (`conv_1.weight`,
//     `conv_2.weight`).
//
// Cold-weights list (predicate must return `false`):
//   biases, per-channel γ/β, embedding tables, depthwise conv
//   kernels, RoPE θ, BN scale/shift, normalizer scalars,
//   pre-transposed `__T` companions, and anything else not on the
//   audit's hot list.  See test_supertonic_f16_weights.cpp.
bool should_materialise_f16_weight(const std::string & source_name);

// round 6 — 2-arg overload that layers a user-
// overridable substring deny-list on top of the curated allow-
// list above.  Returns `false` when ANY non-empty substring in
// `extra_deny_substrings` is found inside `source_name`; otherwise
// forwards to the 1-arg version.
//
// Contract:
//   - Empty deny-list (default for every existing call site)
//     behaves identically to the 1-arg version — zero behaviour
//     change for the default path.
//   - The deny-list is a DENY list, not an allow list: it can
//     only flip `true → false`, never `false → true`.  A pattern
//     that matches a cold weight is a no-op (cold + deny = cold).
//   - Empty strings inside the deny-list are SKIPPED, not treated
//     as universal matches (defensive against config typos that
//     would otherwise silently disable F16 weights entirely).
//   - Substring matching, not regex (matches the curated
//     predicate's audit-friendly style; no regex compile cost,
//     no invalid-pattern error surface).
//
// Use cases:
//   - Researcher A/B testing a specific tensor pattern without
//     recompiling.
//   - Operator force-keeping a tensor as F32 if they observe
//     drift on their hardware.
//   - Safety net for new tensor patterns added in future GGUFs
//     that the curated allow-list inadvertently scoops in.
//
// Plumbed through `EngineOptions::f16_weights_deny_list` →
// `load_supertonic_gguf(..., f16_weights_deny_list)` → the
// per-tensor allocation loop in `load_supertonic_gguf`.
bool should_materialise_f16_weight(const std::string & source_name,
                                   const std::vector<std::string> & extra_deny_substrings);

// Phase 2D — machine-readable per-island timing emitter.
//
// Three-function API:
//   - `supertonic_profile_csv_enabled()` — true when either the
//     env var `SUPERTONIC_PROFILE_CSV=PATH.csv` is set OR a
//     subsequent `_set_path(PATH)` has installed a path.
//   - `supertonic_profile_csv_record(stage, island, step, wall_ms)`
//     — appends one row to the CSV.  No-op when disabled.
//   - `supertonic_profile_csv_flush()` — flushes buffered writes
//     to disk.  Called from each per-stage profile hook after the
//     synth completes, plus at process exit via atexit.
//   - `supertonic_profile_csv_set_path(PATH | nullptr)` — test-only
//     hook to override the env var without touching `setenv`.
//     Passing `nullptr` closes the active file + disables the
//     emitter; passing a new path reopens (header is written
//     only when the file is empty, so re-open appends).
//
// Thread-safety: single-threaded by design.  Recording from
// multiple threads at once is undefined; callers serialise via the
// usual single-engine-per-thread convention.  See
// `test_supertonic_profile_csv.cpp` for the schema contract.
bool supertonic_profile_csv_enabled();
void supertonic_profile_csv_record(const char * stage, const char * island,
                                   int step, double wall_ms);
void supertonic_profile_csv_flush();
void supertonic_profile_csv_set_path(const char * path);

// Phase A1+A2 (Metal): run ALL `total_steps` CFM denoising steps inside
// ONE ggml_cgraph, dispatched with a single ggml_backend_graph_compute
// call.  On non-CPU backends this replaces the engine's per-step loop
// entirely (latent stays in GPU memory step-to-step, no host round-trip).
// On CPU it falls back to a per-step loop over `supertonic_vector_step_ggml`
// so the cblas fastpaths still apply.  Override the GPU path with
// SUPERTONIC_DISABLE_LOOP_GRAPH=1 to A/B against the per-step path.
bool supertonic_vector_loop_ggml(const supertonic_model & model,
                                  const float * initial_noisy_latent,
                                  int latent_len,
                                  const float * text_emb,
                                  int text_len,
                                  const float * style_ttl,
                                  const float * latent_mask,
                                  int total_steps,
                                  std::vector<float> & final_latent_out,
                                  std::string * error = nullptr);

bool supertonic_vector_trace_proj_ggml(const supertonic_model & model,
                                       const float * noisy_latent,
                                       const float * text_emb,
                                       int text_len,
                                       const float * style_ttl,
                                       const float * latent_mask,
                                       int latent_len,
                                       int current_step,
                                       int total_steps,
                                       std::vector<supertonic_trace_tensor> & scalar_trace,
                                       std::vector<supertonic_trace_tensor> & ggml_trace,
                                       std::string * error = nullptr,
                                       bool include_scalar_trace = true,
                                       bool include_ggml_trace = true,
                                       std::vector<float> * next_latent_tc_out = nullptr,
                                       // CFG (Supertonic 3): when both provided, the
                                       // production ggml path uses these channel-major
                                       // [50,256] style layouts instead of deriving the
                                       // conditional ones from `style_ttl` /
                                       // `/Expand_output_0`.  Lets the caller run an
                                       // unconditional pass with the learned null tokens.
                                       const std::vector<float> * style_v_raw_override = nullptr,
                                       const std::vector<float> * kctx_raw_override = nullptr);

// Process-wide alive registry: each loaded supertonic_model registers
// its generation_id with this set on success and unregisters at the
// start of free_supertonic_model.  The thread_local graph caches in
// supertonic_vocoder.cpp / supertonic_text_encoder.cpp /
// supertonic_vector_estimator.cpp / supertonic_duration.cpp own
// ggml_gallocr_t handles allocated against a specific model's
// ggml_backend_t; on a cache miss the existing teardown code calls
// ggml_gallocr_free(cache.allocr).  When the model that backed the
// cache has already been destroyed, that free path asserts inside
// the GPU-backend dylib finaliser.  The is_supertonic_alive() check
// at every free_*_cache() site lets the teardown SKIP the gallocr_free
// call for a generation that's no longer alive.
//
// IMPORTANT: the skip path leaks the gallocr's full internal state —
// not just the ~80-byte struct previously documented, but its
// node→buffer hash tables, per-leaf records, and per-buffer-type
// allocators (~83 KB per skipped gallocr at our graph sizes).
// Across the ~30 per-stage caches a single Supertonic synth
// populates, that's tens of MB leaked per Engine cycle.
//
// The primary fix for the SDK pattern (one Engine per thread,
// loaded + run + destroyed on the same thread) is the
// `release_*_thread_local_caches` machinery declared at the top of
// this header: free_supertonic_model invokes those BEFORE the
// backend is freed, so every cache on the destruction thread gets
// its normal ggml_gallocr_free path against a live backend.  The
// skip-and-leak fallback below remains for the harder multi-thread
// pattern (Engine on thread A populates caches, then thread B
// destroys it without first releasing thread A's caches — not safe
// per the Engine contract, but the skip keeps us from crashing).
//
// Thread-safe: backed by a single std::mutex.  Lookup is on the
// hot per-call path inside free_*_cache(), but the lock is held only
// for an unordered_set::find() so contention is minimal.
void register_supertonic_alive(uint64_t generation_id);
void unregister_supertonic_alive(uint64_t generation_id);
bool is_supertonic_alive(uint64_t generation_id);

// Helper consumed by every per-stage free_*_cache().  Skips the
// ggml_gallocr_free call when the allocr's backend has already been
// torn down (model.generation_id no longer in the alive registry).
inline void supertonic_safe_gallocr_free(ggml_gallocr_t & allocr, uint64_t generation_id) {
    if (allocr && is_supertonic_alive(generation_id)) {
        ggml_gallocr_free(allocr);
    }
    allocr = nullptr;
}

// ---------------------------------------------------------------------
// Portable LeakyReLU(x, α) = (1-α)·relu(x) + α·x.
//
// `ggml_leaky_relu` (GGML_OP_LEAKY_RELU) is a CPU builtin and is also
// present on the QVAC `ggml-speech` vcpkg port via the chatterbox
// `ggml-opencl-chatterbox-ops.patch`, but baseline upstream
// `ggml-opencl` and several other GPU backends still reject the op at
// graph-execute time.  Routing through this helper keeps every
// Supertonic graph executable on every backend:
//
//   - On CPU we keep the single fused builtin (cheaper, single op
//     callback per row instead of three).
//   - On GPU we decompose into `RELU + SCALE + ADD`, all universally
//     supported (see `ggml_opencl_supports_op()`).
//
// Defined inline in the header so every TU that includes this header
// gets the same lowering, and so the dispatch test can call it
// directly without depending on which TU happens to instantiate it.
// The thread-local `supertonic_use_cpu_custom_ops()` flag flips
// behaviour; the inline body is a thin wrapper, so neither branch
// retains hidden state.
//
// Bit-exact equivalence between the two lowerings is checked in
// `test/test_supertonic_portable_ops.cpp` on a CPU backend.
inline ggml_tensor * leaky_relu_portable_ggml(ggml_context * ctx, ggml_tensor * x, float alpha);

// ---------------------------------------------------------------------
// Op-dispatch policy for the GGML graph builders.
//
// The Supertonic vocoder + vector estimator carry several
// `ggml_custom_4d` fast paths whose op callbacks invoke CBLAS / direct
// pointer loads against the tensor `data` field.  Those paths are
// only valid on the GGML CPU backend (the only backend that exposes
// host-addressable tensor data inside an op callback and schedules
// custom ops at all — every other backend rejects GGML_OP_CUSTOM
// outright).  When the resolved compute backend is non-CPU
// (CUDA / Metal / Vulkan / OpenCL) those sites must take the
// pure-GGML fallback path so the graph stays GPU-executable.
//
// Threading the decision through every graph-build helper would
// touch dozens of file-static functions across three TUs.  Instead,
// each public forward entry point (e.g. supertonic_vocoder_forward_ggml,
// supertonic_vector_step_ggml) instantiates a
// `supertonic_op_dispatch_scope` on entry, which sets a thread_local
// flag mirroring `model.backend_is_cpu`.  Graph-build helpers query
// it via `supertonic_use_cpu_custom_ops()` at the cblas-vs-fallback
// branch.  RAII teardown guarantees the flag is cleared even on
// exception paths, so a CPU-only second engine in the same thread
// still sees the default `true` after a GPU engine's forward returns.
bool supertonic_use_cpu_custom_ops();
bool supertonic_use_f16_attn();
// Thread-local mirror of `supertonic_model::backend_supports_fused_supertonic_ops`,
// set by `supertonic_op_dispatch_scope` for the duration of each public
// `*_forward_ggml` / `*_trace_ggml` entry (same RAII pattern as
// `supertonic_use_cpu_custom_ops` / `supertonic_use_native_leaky_relu`).
// Graph-build helpers consult it to choose the fused custom op vs the
// pure-GGML decomposition.  Defaults to `false` (pure-GGML) when no scope
// is active, so a helper called outside a scope never emits a backend-
// unsupported fused op.
bool supertonic_use_fused_supertonic_ops();

// Thread-local mirror of `supertonic_model::mulmat_needs_pad`, set by the dispatch scope.
// Defaults to false outside any scope, so st_mul_mat emits a plain ggml_mul_mat.
bool supertonic_mulmat_needs_pad();

// Drop-in for ggml_mul_mat: when mulmat_needs_pad, zero-pad a GEMM output dim < 64 up to 64,
// then slice back the [M,N] block (exact). No-op on healthy backends, mat-vec, or non-F32 operands.
inline ggml_tensor * st_mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    if (!supertonic_mulmat_needs_pad()) return ggml_mul_mat(ctx, a, b);
    const int64_t M = a->ne[1];
    const int64_t N = b->ne[1];
    // GEMM-path predicate (the only path that miscomputes).  Anything that would
    // dispatch as mat-vec is correct on the driver and must not be perturbed.
    // The N<=8 carve-out mirrors ggml-vulkan's mat-vec dispatch (mul_mat_vec_max_cols,
    // currently 8: mul_mat_vec when dst->ne[1]==1 || (dst->ne[1]<=8 && src1 batch==1)).
    // Revisit if that upstream threshold changes.
    const bool is_gemm = (N != 1) && !(N <= 8 && b->ne[2] * b->ne[3] == 1);
    if (!is_gemm) return ggml_mul_mat(ctx, a, b);
    const bool pad_a = (M < 64 && a->type == GGML_TYPE_F32);
    const bool pad_b = (N < 64 && b->type == GGML_TYPE_F32);
    if (!pad_a && !pad_b) return ggml_mul_mat(ctx, a, b);  // nothing paddable (e.g. non-F32 src)
    ggml_tensor * ap = a;
    ggml_tensor * bp = b;
    if (pad_a) {
        // Materialise strided/permuted views first so ggml_pad sees a simple
        // contiguous layout. Only on the pad path, so healthy backends pay nothing.
        if (!ggml_is_contiguous(ap)) ap = ggml_cont(ctx, ap);
        ap = ggml_pad(ctx, ap, 0, (int) (64 - M), 0, 0);
    }
    if (pad_b) {
        if (!ggml_is_contiguous(bp)) bp = ggml_cont(ctx, bp);
        bp = ggml_pad(ctx, bp, 0, (int) (64 - N), 0, 0);
    }
    ggml_tensor * r = ggml_mul_mat(ctx, ap, bp);             // [Mpad, Npad, ne2, ne3], contiguous F32
    ggml_tensor * v = ggml_view_4d(ctx, r, M, N, r->ne[2], r->ne[3],
                                   r->nb[1], r->nb[2], r->nb[3], 0);  // real top-left sub-block
    return ggml_cont(ctx, v);  // repack tight: downstream consumers require contiguity
}

// round 4 — thread-local accessor for the currently-
// active K/V dispatch dtype, mirroring `supertonic_use_f16_attn`'s
// pattern.  Returns `kv_attn_dtype::f32` when no
// `supertonic_op_dispatch_scope` is active (matches the model's
// default-constructed value, so a graph builder called outside a
// scope never accidentally takes the F16 / BF16 / Q8_0 path).
//
// The dispatch-scope ctor populates this from
// `model.kv_attn_type`; the dtor restores the previous value
// (RAII teardown, exception-safe).
kv_attn_dtype supertonic_kv_attn_type();

// round 4 — pure-logic resolver for the multi-dtype
// K/V dispatch policy.  Maps the EngineOptions int + the
// resolved-backend probes into the concrete `kv_attn_dtype` to
// dispatch.
//
// Behaviour matrix:
//
//   | requested | legacy_use_f16_attn | resolved                       |
//   |-----------|---------------------|--------------------------------|
//   | -1 (auto) | true                | f16 if supports_f16 else f32   |
//   | -1 (auto) | false               | f32                            |
//   |  0 (f32 force) | any            | f32                            |
//   |  1 (f16 force) | any            | f16 if supports_f16 else f32   |
//   |  2 (bf16 force)| any            | bf16 if supports_bf16 else f32 |
//   |  3 (q8_0 force)| any            | q8_0 if supports_q8_0 else f32 |
//   | < -1 or > 3    | any            | throws std::runtime_error      |
//
// Fall-through to `f32` (instead of throw) on probe-rejected
// explicit requests is intentional: probes are advisory, and an
// operator setting `--kv-attn-type bf16` once in their production
// config should work on both NVIDIA Ampere+ (BF16 effective) and
// Intel ARC (no coopmat2 → silent F32 fallback) without crashing.
// Loud-failure stays for actual config errors (out-of-range int).
//
// PR #18 reviewer (Omar) follow-up — the "silent" part of that
// fallback was hiding an operator surprise.  Optional
// `out_was_downgraded` pointer is set to `true` IFF the operator
// explicitly requested f16 / bf16 / q8_0 AND the corresponding
// backend probe returned false AND the resolver therefore
// returned `f32` instead.  The CLI-facing call sites (Engine
// ctor + supertonic-bench) consult this flag and emit a
// `fprintf(stderr, "warning: ...")` so the operator knows their
// `--kv-attn-type bf16` config silently degraded.  Auto (`-1`)
// + missing probe is NOT a downgrade (the operator didn't ask
// for a specific dtype, so the auto-policy is doing its job) —
// the flag stays false on that path.
//
// Pass `nullptr` (the default) to ignore the downgrade signal
// — the pure-logic unit tests use this so test runs don't spam
// stderr with warnings.
//
// Pure logic, no Vulkan symbols touched here — same split
// pattern as `resolve_vulkan_device_index` from round 3.
kv_attn_dtype resolve_kv_attn_type(int requested,
                                   bool legacy_use_f16_attn,
                                   bool backend_supports_f16,
                                   bool backend_supports_bf16,
                                   bool backend_supports_q8_0,
                                   bool * out_was_downgraded = nullptr);
// true when the resolved backend supports
// `GGML_OP_LEAKY_RELU` natively.  Mirrored from
// `supertonic_model::use_native_leaky_relu` by
// `supertonic_op_dispatch_scope` for the duration of each public
// `*_forward_ggml` / `*_trace_ggml` entry.  Consulted by
// `leaky_relu_portable_ggml` to skip the RELU+SCALE+ADD
// decomposition when the backend has the fused op available.
bool supertonic_use_native_leaky_relu();

// load-time backend-capability probes used by the
// engine + bench auto-policy for `use_f16_attn`.  Returns `true`
// when the resolved backend would accept a Supertonic-shaped
// `ggml_flash_attn_ext(Q=F32, K/V=F16)` graph node — the auto-
// enable policy gates on this so a backend that doesn't ship the
// mixed-precision kernel doesn't crash at first synth call.
// Manual override via `EngineOptions::f16_attn=1` still forces
// dispatch (useful for benchmarking with a debug-shim backend).
//
// follow-up — both probes are now memoised
// process-wide by `ggml_backend_t` handle, so the engine + bench
// + load_supertonic_gguf trio doesn't re-run the same probe two
// or three times per backend.  Defined out of line in
// supertonic_gguf.cpp.
bool supertonic_backend_supports_f16_kv_flash_attn(ggml_backend_t backend);

// follow-up — load-time backend-capability probe used by
// the engine + bench + `load_supertonic_gguf` auto-policy for
// `use_f16_weights`.  Symmetric to the F16-K/V flash-attn probe:
// returns `true` when the resolved backend would accept the hot
// `mul_mat(F16 weight, F32 activation) → F32` graph node Supertonic
// dispatches every step (vector-estimator W_query, vocoder head
// linear, text-encoder linears, etc.).  The auto-enable policy
// gates on this so a partial-port backend that ships F16 storage
// but rejects F16 mul_mat for the hot shape keeps the F32 path
// — slower but guaranteed not to crash at first synth call.
// Manual override via `EngineOptions::f16_weights=1` still forces
// materialisation.
bool supertonic_backend_supports_f16_mul_mat(ggml_backend_t backend);

// follow-up — load-time backend-capability probe for
// the Q8_0 K/V `FLASH_ATTN_EXT` variant.  Forward-compat: returns
// `true` when the backend would accept a Supertonic-shaped
// `ggml_flash_attn_ext(Q=F32, K/V=Q8_0)` graph node.  Vulkan's
// `supports_op` advertises Q8_0 K/V in both scalar and coopmat2
// paths (`ggml-vulkan.cpp:GGML_OP_FLASH_ATTN_EXT`), which would
// halve the per-step K/V upload bandwidth on memory-bandwidth-
// bound mobile GPUs in exchange for a small (~0.5 %) drift on the
// attention output.  This PR adds the probe + caches the result;
// the live dispatch site is not yet wired through Q8_0 because the
// drift hasn't been measured against the F16 K/V parity harness on
// a real Vulkan adapter.  See PROGRESS_SUPERTONIC.md "Deferred
// work" for the follow-up.
bool supertonic_backend_supports_q8_0_kv_flash_attn(ggml_backend_t backend);

// round 3 — load-time backend-capability probe for the
// BF16 K/V `FLASH_ATTN_EXT` variant.  Forward-compat: returns
// `true` when the backend would accept a Supertonic-shaped
// `ggml_flash_attn_ext(Q=F32, K/V=BF16)` graph node.  Vulkan
// advertises BF16 K/V in the coopmat2 path only
// (`ggml-vulkan.cpp:GGML_OP_FLASH_ATTN_EXT`); BF16 has the same
// 2-byte per-element footprint as F16 (so identical upload
// bandwidth) but the wider 8-bit exponent range avoids the
// occasional small-score underflow that drives F16's tolerance
// widening on the parity harness.  Live dispatch site isn't yet
// wired (a follow-up gates `--kv-attn-type bf16` on this probe);
// caching it here primes the cache for that work.
bool supertonic_backend_supports_bf16_kv_flash_attn(ggml_backend_t backend);

// round 3 — backend capability probe for Vulkan's
// `ggml_backend_vk_host_buffer_type()`.  Returns `true` iff the
// backend is Vulkan AND the host-pinned buffer type is non-null.
// Forward-compat — primes the capability cache for a follow-up
// per-engine input-scratchpad refactor that skips ggml-vulkan's
// internal staging-buffer hop on per-step uploads (text-emb,
// time-step encoding, style embedding) by allocating those
// tensors in the host-pinned buffer type instead of the default
// device-local buffer.
bool supertonic_backend_supports_pinned_host_buffer(ggml_backend_t backend);

// round 12 #5 — pinned-host-buffer input allocator.
//
// Round 3 shipped the capability probe; round 12 lands the actual
// per-engine input-scratchpad refactor.  Callers create a small
// `ggml_context` (with `no_alloc=true`) containing ONLY the hot
// per-step input tensors (front-block `x_in` / `mask_in` /
// `t_emb_in`, group-cache `x_in` / `temb_in`, etc.) and pass it
// here.  On Vulkan (where `ggml_backend_vk_host_buffer_type()`
// returns non-null) the helper allocates a buffer from the
// host-pinned buft and binds every tensor in `input_ctx` to it
// — `ggml_backend_tensor_set` then writes from the host's heap
// directly into BAR-mapped GPU memory without an intermediate
// staging-buffer copy.
//
// Return contract:
//   - `nullptr` if `model.backend == nullptr`, `input_ctx == nullptr`,
//     or the backend doesn't expose `ggml_backend_vk_host_buffer_type()`.
//     Caller falls back to letting `ggml_gallocr_alloc_graph`
//     handle the input tensors via the default buffer type —
//     correct, just one staging-buffer hop per upload.
//   - Otherwise the returned `ggml_backend_buffer_t` is OWNED by
//     the caller.  Free at cache destruction with
//     `ggml_backend_buffer_free(buf)`.
//
// On Vulkan adapters that expose a host-coherent BAR-mapped pool
// (every modern discrete + every UMA iGPU), this skips one
// memcpy per `ggml_backend_tensor_set` on the bound tensors.
// Per synth at the 4 attention-feeding caches × 3 small per-step
// inputs × 5 denoise steps ≈ 60 staging-hops saved.  Each hop
// is ~5–15 us on the dev rig; aggregate ~0.3–1 ms / synth.
//
// CPU-only test (`test_supertonic_pinned_host_buffer.cpp`) pins
// the symbol + the conservative `nullptr` return contract on
// CPU backend + null-input safety in error paths.  End-to-end
// behaviour validated by Vulkan synth + bench on real hardware.
ggml_backend_buffer_t try_alloc_inputs_in_pinned_host_buffer(
    const supertonic_model & model,
    ggml_context * input_ctx);

// round 13 #1 — input-scratchpad allocator that
// consolidates the round-12 boilerplate.
//
// Round 12 #5 inlined the "try pinned-host first, fall back to
// default backend buffer, throw if both fail" idiom at 4 cache
// sites.  Round 13 extends the pattern to 5+ additional cache
// sites (vector_loop_one_graph, vocoder, style residual + QKV,
// merged speech-prompted, ...) — a 5x boilerplate copy is
// error-prone (the failure-cleanup ordering is subtle:
// `ggml_free(input_ctx)` BEFORE nulling the input-tensor
// pointers leaves dangling pointers in the cache struct that a
// subsequent free path will dereference).
//
// Contract:
//   - Tries `try_alloc_inputs_in_pinned_host_buffer(model, ctx)`
//     first.  Returns its buffer on success.
//   - On failure (CPU / non-Vulkan / probe miss), falls back to
//     `ggml_backend_alloc_ctx_tensors(ctx, model.backend)`.
//     Returns that buffer on success.
//   - On BOTH failing (system resource exhaustion, dead backend,
//     etc.), throws `std::runtime_error` with a message that
//     includes `cache_name` so operators can attribute the
//     failure to a specific cache.
//   - Defensive throws on `model.backend == nullptr`,
//     `input_ctx == nullptr`, `cache_name == nullptr` — these
//     are caller-bug guards in error-handler paths.
//
// Caller owns the returned buffer.  Standard teardown order
// remains: gallocr → main ctx → input_buf → input_ctx (reversed
// would dangle pointers in the cache struct).
//
// CPU-only test (`test_supertonic_input_scratchpad.cpp`) pins
// the symbol + CPU-fallback contract + null-argument throws.
// End-to-end Vulkan validation lives in the cache-build paths
// that consume the helper (round 13 #1 wiring at
// `vector_loop_one_graph_cache`, `vocoder_graph_cache`, etc.).
ggml_backend_buffer_t alloc_input_scratchpad_or_throw(
    const supertonic_model & model,
    ggml_context * input_ctx,
    const char * cache_name);

// round 3 — multi-device Vulkan auto-pick policy.
//
// `init_supertonic_backend` calls `ggml_backend_vk_get_device_count()`
// + `ggml_backend_vk_get_device_memory()` per device to build the
// `free_vram_per_device` list, then dispatches into this pure-
// logic helper to pick the device index.  Splitting the policy
// from the Vulkan-only plumbing means the policy is testable on
// CPU with synthetic inputs (see test_supertonic_vulkan_device_select.cpp).
//
// Behaviour matrix:
//
//   | requested | dev_count | result                                  |
//   |-----------|-----------|-----------------------------------------|
//   | -1        | 0         | throws (no device to pick)              |
//   | N>=0      | 0         | throws (no device to pick)              |
//   | -1        | 1         | 0  (only choice)                        |
//   | -1        | N>1       | argmax(free_vram); ties → lower index   |
//   | N>=0      | dev_count | N if N<dev_count, else throws           |
//   | N<-1      | any       | throws (negative != -1 reserved)        |
//
// Throws `std::runtime_error` on invalid input; the caller surfaces
// the message verbatim (same pattern as the existing
// `--vulkan-device N out of range` error in `init_supertonic_backend`).
//
// Tie-breaking on equal free VRAM picks the lower index so
// identical-spec multi-GPU machines (lab racks of A100s, e.g.)
// produce stable per-run device assignment instead of depending
// on driver enumeration order.  Operators who need a different
// policy can `--vulkan-device N` explicitly.
//
// round 12 — `is_uma_per_device` (optional 3rd arg)
// biases the auto-pick against UMA / iGPU devices when a
// discrete device is also present.  Background: on hybrid
// machines (NVIDIA RTX 5090 discrete + AMD RADV iGPU, or
// similar), `ggml_backend_vk_get_device_memory()` reports the
// iGPU's free pool as system RAM (often 120+ GB) because UMA
// shares the host RAM with the CPU.  The round-3 argmax then
// picks the iGPU, silently dropping ~40× realtime on synth
// throughput vs. the discrete card.
//
// New policy (when `is_uma_per_device.size() == free_vram_per_device.size()`):
//
//   1. If at least one device has `is_uma_per_device[i] == false`,
//      run argmax(free_vram) over the DISCRETE subset only.
//   2. Otherwise (all UMA) fall back to argmax over all devices.
//   3. Explicit `requested >= 0` passthrough is UMA-agnostic
//      (operator-pinned index always wins).
//
// `is_uma_per_device` is OPTIONAL — empty list (default) means
// "no UMA flags available, use round-3 policy".  Mismatched-
// length non-empty lists throw (caller bug guard).
//
// Caller wiring lives in `init_supertonic_backend`: query
// `ggml_backend_vk_get_device_type()` per device, set the bool
// to `true` for `IntegratedGpu` / `Cpu` / `Other` types.  Pure
// logic, no Vulkan symbols touched here — same split pattern
// as the round-3 free-VRAM list.
int resolve_vulkan_device_index(int requested,
                                const std::vector<size_t> & free_vram_per_device,
                                const std::vector<bool> & is_uma_per_device = {});

// follow-up — test seams for the capability cache.
// `supertonic_clear_capability_cache` drops every cached entry so
// the regression test in `test_supertonic_capability_cache.cpp`
// can verify the cache short-circuits on a hit (the cold-cache
// call bumps `supertonic_capability_probe_call_count`; subsequent
// cached calls don't until the cache is cleared).
//
// Not part of the supported public API — exported only for the
// in-process test harness.  Keeping the declaration in this
// internal header (which production callers don't include) is
// the cheapest way to avoid the symbol leaking into the public
// surface while still letting the unit test reach it.
void supertonic_clear_capability_cache();
uint64_t supertonic_capability_probe_call_count();

struct supertonic_op_dispatch_scope {
    bool prev_use_cpu_custom_ops;
    bool prev_use_f16_attn;
    bool prev_use_native_leaky_relu;
    bool prev_use_fused_supertonic_ops;
    // saved `mulmat_needs_pad` flag for RAII teardown.
    bool prev_mulmat_needs_pad;
    // round 4 — saved K/V dispatch dtype for RAII
    // teardown.  Restored on scope destruction so a follow-on
    // engine on the same thread sees the default value, not the
    // previous engine's dispatch dtype (matters for nested
    // synthesis flows where two engines share a worker thread).
    kv_attn_dtype prev_kv_attn_type;
    explicit supertonic_op_dispatch_scope(const supertonic_model & model);
    ~supertonic_op_dispatch_scope();
    supertonic_op_dispatch_scope(const supertonic_op_dispatch_scope &)             = delete;
    supertonic_op_dispatch_scope & operator=(const supertonic_op_dispatch_scope &) = delete;
};

// ---------------------------------------------------------------------
// Audit finding F20 (partial / Phase 2H) — RoPE rotation in-graph
// with host-precomputed cos/sin tables.
//
// Replaces the per-attention-site `apply_rope(theta, q, L, H, D)`
// host loop with a GPU-native rotation that reuses cos/sin tables
// uploaded once per (L, θ).  Eliminates the CPU rotation step
// (~50 µs × 40 sites/synth ≈ 2 ms) and is the prerequisite for a
// follow-up that wires Q/K directly from the QKV graph into the
// attention graph (cuts the host round-trip on Q and K outright).
//
// Formula it matches (exactly mirrors the scalar `apply_rope` in
// `supertonic_vector_estimator.cpp`):
//
//     angle = (t / L) * theta[d]            ← `t/L`, not absolute t
//     cs = cos(angle), sn = sin(angle)
//     for d in [0, half):
//         x[t, h, d]      := x[t, h, d]*cs       - x[t, h, half+d]*sn
//         x[t, h, half+d] := x[t, h, half+d]*cs  + x[t, h, d]*sn
//
// Tensor contract:
//   - `x`         : F32, ne=[head_dim, n_heads, L].  Memory layout
//                   matches the scalar reference's
//                   `data[t*H*D + h*D + d]`.
//   - `cos_table` : F32, ne=[half, L]. cos_table[t*half + d] = cos((t/L)*θ[d]).
//   - `sin_table` : F32, ne=[half, L]. Analogous.
//   - returns     : F32, ne=[head_dim, n_heads, L].  Rotated x.
//
// Op-set used:
//   `ggml_view_3d`, `ggml_reshape_3d`, `ggml_repeat`, `ggml_mul`,
//   `ggml_sub`, `ggml_add`, `ggml_concat`.
// All universally supported (incl. baseline upstream OpenCL —
// see `ggml_opencl_supports_op()`), so the helper doesn't require
// the chatterbox-patched `ggml_sin` / `ggml_cos` / `ggml_rope`.
//
// Parity-tested in `test_supertonic_rope_in_graph.cpp` against
// the scalar `apply_rope` for the two hot vector-estimator shapes
// + a zero-θ identity check.  Tolerance `1e-4` absolute.
inline ggml_tensor * apply_rope_in_graph(ggml_context * ctx,
                                         ggml_tensor * x,
                                         ggml_tensor * cos_table,
                                         ggml_tensor * sin_table) {
    // Shape contracts (asserted at caller via test harness; here
    // we only deref the fields).
    const int64_t head_dim = x->ne[0];
    const int64_t n_heads  = x->ne[1];
    const int64_t L        = x->ne[2];
    const int64_t half     = head_dim / 2;

    // Split x along axis 0 into lower and upper halves.  Both
    // halves share x's strides (`nb[0..2]`); the upper half just
    // adds a half-byte offset.  Memory underneath is unchanged;
    // these are views, not copies.
    ggml_tensor * x_lower = ggml_view_3d(
        ctx, x, half, n_heads, L,
        /*nb1=*/x->nb[1], /*nb2=*/x->nb[2],
        /*offset=*/0);
    ggml_tensor * x_upper = ggml_view_3d(
        ctx, x, half, n_heads, L,
        /*nb1=*/x->nb[1], /*nb2=*/x->nb[2],
        /*offset=*/(size_t) half * x->nb[0]);

    // Broadcast cos/sin over n_heads: cos has ne=[half, L]; we
    // need [half, n_heads, L] to align with x_lower/x_upper.
    // `ggml_reshape_3d(c, half, 1, L)` gives ne=[half, 1, L] (a
    // shape-changing zero-cost view of the same memory); then
    // `ggml_repeat(c_3d, x_lower)` broadcasts axis 1 from 1 to
    // n_heads.  ggml_can_repeat accepts the (..., 1, ...) → (...,
    // N, ...) broadcast pattern unconditionally.
    ggml_tensor * cos_3d = ggml_reshape_3d(ctx, cos_table, half, 1, L);
    ggml_tensor * sin_3d = ggml_reshape_3d(ctx, sin_table, half, 1, L);
    ggml_tensor * cos_b  = ggml_repeat(ctx, cos_3d, x_lower);
    ggml_tensor * sin_b  = ggml_repeat(ctx, sin_3d, x_lower);

    // Rotation: standard 2×2 cos/-sin / sin/cos block applied
    // pointwise.  ggml_concat dim=0 stitches the lower + upper
    // halves back into a [head_dim, n_heads, L] tensor with the
    // same memory layout x came in with.
    ggml_tensor * new_lower = ggml_sub(ctx,
        ggml_mul(ctx, x_lower, cos_b),
        ggml_mul(ctx, x_upper, sin_b));
    ggml_tensor * new_upper = ggml_add(ctx,
        ggml_mul(ctx, x_upper, cos_b),
        ggml_mul(ctx, x_lower, sin_b));
    return ggml_concat(ctx, new_lower, new_upper, /*dim=*/0);
}

// Host-side helper: precompute the (cos, sin) tables consumed by
// `apply_rope_in_graph` for a given (L, θ) pair.  Output layout
// matches the GGML tensor's natural row-major upload: element
// (t, d) at `out[t*half + d]`.  Callers cache by L on
// `supertonic_model::rope_cos_sin_cache` and upload once per cold
// miss.  Pure function over (theta, L, half); no model state.
inline void make_rope_cos_sin_tables(const float * theta,
                                     int L,
                                     int half,
                                     std::vector<float> & cos_out,
                                     std::vector<float> & sin_out) {
    cos_out.resize((size_t) L * half);
    sin_out.resize((size_t) L * half);
    for (int t = 0; t < L; ++t) {
        const float t_frac = (float) t / (float) L;
        for (int d = 0; d < half; ++d) {
            const float angle = t_frac * theta[d];
            cos_out[(size_t) t * half + d] = std::cos(angle);
            sin_out[(size_t) t * half + d] = std::sin(angle);
        }
    }
}

// ---------------------------------------------------------------------
// Audit finding F23 (F20 integration / Phase 2H follow-through) —
// packed-QK RoPE adapter for the Q/K-producing graphs.
//
// `apply_rope_in_graph` operates on a tensor with `ne=[head_dim,
// n_heads, L]` — the natural layout the scalar `apply_rope`
// reference indexes into (`data[t*H*D + h*D + d]`).  Every actual
// call site in the vector estimator produces Q/K via
// `dense_matmul_time_ggml`, whose output is a 2D tensor with
// `ne=[L, HD]` — axis 0 = L (time, fastest along natural strides
// `nb=[elem, L*elem]`) and axis 1 = HD = n_heads * head_dim
// (packed channels h*D+d, slowest).  In flat memory the element
// (t, c) sits at byte offset `(t + c*L)*elem` — i.e. **channel-
// major-flat** (`data[t + c*L]`), which is the bit-exact transpose
// of the time-major-flat layout the scalar `apply_rope` reference
// indexes through (`data[t*H*D + h*D + d]`).
//
// same-shape matmul on every backend: confirmed by
// inspection of the CPU custom-op fast path (`ggml_custom_4d(F32,
// x->ne[0] /* = L */, w->ne[0] /* = OC */, …)` → `[L, OC]`) and
// the `conv1d_f32(K=1)` fallback (`ggml_reshape_3d(result,
// im2col->ne[1] /* = L */, kernel->ne[2] /* = OC */, …)` → also
// `[L, OC]`).  Both code paths produce the same ne contract — so
// this helper's adapter has to bridge the **matmul-output**
// channel-major-flat layout onto `apply_rope_in_graph`'s natural-
// strides `[D, H, L]` contract.
//
// History note: the original (PR #16 follow-up #5) version of
// this helper assumed `q->ne[0] = HD` and `q->ne[1] = L` — i.e.,
// the transpose of what the matmul actually produces.  That
// older contract crashed at the defensive assertion below on
// every real synth (the moment a GGUF carrying `vector_rope_theta`
// enabled the in-graph rotation path).  The CPU unit test that
// landed alongside `apply_rope_to_packed_qk` hand-built Q under
// the `[HD, L]` assumption, so the failure mode was invisible to
// CI.  GPU backends (Metal / CUDA / Vulkan / OpenCL) silently
// dispatched a transposed view through the rotation, masking the
// shape problem until a CPU `--n-gpu-layers 0` synth hit the
// assert. `test_supertonic_rope_packed_qk.cpp`
// now reproduces the **production** matmul layout and pins both
// the input and output shape contracts.
//
// Pipeline (production layout):
//   - Step 1: `ggml_cont(ggml_transpose(q))` — view-swap axes
//     0/1 (zero-cost stride flip) then materialise to natural
//     strides.  Result has ne=[HD, L] with **time-major-flat**
//     memory layout (`data[c + t*HD]`).  This is the SAME layout
//     `q_tc_in` (`ggml_new_tensor_2d(A, L)` in
//     `vector_text_attention_cache`) expects for the
//     `ggml_backend_tensor_copy` device→device blit at the GPU-
//     bridge dispatch site.
//   - Step 2: Re-view the packed tensor as `[head_dim, n_heads,
//     L]` via the zero-cost stride trick `nb[0]=elem,
//     nb[1]=D*elem, nb[2]=HD*elem` — element (d, h, l) lands at
//     offset `d + h*D + l*HD` (elem units), identical to the
//     post-transpose layout's element (col=h*D+d, row=l) at
//     `col + row*HD`.
//   - Step 3: Materialise a contiguous `[D, H, L]` copy so the
//     downstream `ggml_concat` inside `apply_rope_in_graph` sees
//     monotonically-increasing strides.
//   - Step 4: `apply_rope_in_graph(ctx, x_dhl, cos, sin)`.
//   - Step 5: Reshape the rotated `[D, H, L]` result back to
//     `[HD, L]` — same memory, different ne labels.  Bytes are
//     in time-major-flat layout `data[c + t*HD]`, byte-for-byte
//     identical to scalar `apply_rope`'s output and to what
//     `q_tc_in` expects.
//
// Call-site impact for the bytes-out contract:
//   - GPU bridge (`run_text_attention_cache_gpu`): unchanged.
//     `ggml_backend_tensor_copy(q_rope, q_tc_in)` already passes
//     `ggml_nbytes(src) == ggml_nbytes(dst)` (same nelements)
//     and now also matches the destination's memory layout
//     bit-for-bit.
//   - Legacy host bridge: `tensor_to_time_channel(q_rope)` was
//     designed for the (incorrectly-shaped) old contract and
//     would now read the transpose-of-the-transpose if called
//     unchanged.  Use `tensor_raw_f32(q_rope)` instead — the
//     bytes are already time-major-flat (matches scalar
//     `apply_rope`'s output buffer contract), and uploading
//     them via `ggml_backend_tensor_set` to `q_tc_in` lands the
//     same bytes the GPU-bridge `ggml_backend_tensor_copy`
//     would.  The four production call sites in
//     `supertonic_vector_estimator.cpp` are updated in lock-step
//     with this helper.
//   - Trace mode: the `PUSH_GGML_TRACE` entries push a
//     `std::vector<float>` shaped as `{L, HD}` (i.e., flat
//     `out[t*HD + c]` — scalar `apply_rope`'s native indexing).
//     `tensor_raw_f32(q_rope)` returns exactly that layout, so
//     trace parity vs. the scalar harness is preserved without
//     any further re-pack.
//
// Cost vs. the pre-fix (broken) helper:
//   - Adds one `ggml_cont` per site (the head-of-pipeline
//     transpose).  On CPU it is a single memcpy of `L * HD * 4`
//     bytes; on GPU backends (Vulkan one ~256-thread shader
//     dispatch, Metal / OpenCL equivalents) it is one shader
//     dispatch per cache build.  The cache is built ONCE and
//     reused across all 5 denoise steps, so the cost is fully
//     amortised.
//   - Eliminates 40 CPU rotations / synth (~50 µs each ≈ 2 ms
//     wall-time on the default 5-step × 4-RoPE-site schedule).
//   - Net (Vulkan branch only): the original rounds-8/9 GPU-
//     bridge wins are preserved AND now actually run end-to-end
//     without crashing.
//
// Universally-supported ops only: `ggml_transpose`, `ggml_cont`,
// `ggml_view_3d`, `ggml_reshape_2d` + everything
// `apply_rope_in_graph` uses.  Green on baseline upstream OpenCL.
//
// Parity-tested in `test_supertonic_rope_packed_qk.cpp` against
// the scalar `apply_rope` on the two hot vector-estimator shapes
// (`q_len=20 × H=4 × D=64`, `kv_len=32 × H=4 × D=64`), a
// degenerate `L=1` trip-wire, and an explicit output-shape
// contract check that pins `ne[0]=HD, ne[1]=L`.  Tolerance
// `1e-4` absolute.
inline ggml_tensor * apply_rope_to_packed_qk(ggml_context * ctx,
                                              ggml_tensor * q,
                                              ggml_tensor * cos_table,
                                              ggml_tensor * sin_table,
                                              int n_heads,
                                              int head_dim) {
    // Step 1 — transpose `ne=[L, HD]` (matmul-output contract,
    // channel-major-flat memory) into `ne=[HD, L]` with natural
    // time-major-flat memory.  `ggml_transpose` is a view-only
    // axis swap (nb[0] ↔ nb[1]); `ggml_cont` materialises the
    // natural strides `nb=[elem, HD*elem]`.  This is the SAME
    // memory layout the downstream `q_tc_in` consumes — the
    // helper's output then plumbs unchanged into both the GPU-
    // bridge `ggml_backend_tensor_copy` and the legacy host-
    // bridge `tensor_raw_f32` paths.
    ggml_tensor * q_packed = ggml_cont(ctx, ggml_transpose(ctx, q));

    const int64_t L  = q_packed->ne[1];
    const int64_t HD = q_packed->ne[0];
    (void) HD; // assertion-only; compiler may drop in NDEBUG.
    GGML_ASSERT(HD == (int64_t) n_heads * head_dim);

    // Step 2 — re-view the `[HD, L]` packed tensor as `[D, H, L]`
    // via the zero-cost stride trick.  q_packed has natural
    // strides nb=[elem, HD*elem]; the view nb=[elem, D*elem,
    // HD*elem] gives element (d, h, l) at offset `d + h*D + l*HD`
    // (elem units) — bit-identical to (col=h*D+d, row=l) at
    // `col + row*HD` in the original packed layout.
    ggml_tensor * q_dhl_view = ggml_view_3d(ctx, q_packed,
        head_dim, n_heads, L,
        /*nb1=*/(size_t) head_dim * sizeof(float),
        /*nb2=*/(size_t) n_heads * head_dim * sizeof(float),
        /*offset=*/0);
    // Step 3 — materialise a contiguous [D, H, L] copy so the
    // downstream `ggml_concat` / `ggml_repeat` ops in
    // `apply_rope_in_graph` see natural strides
    // (`nb=[elem, D*elem, D*H*elem]`).  The view above is legal
    // but non-natural (`nb[1]<nb[2]` with a `D*elem`/`H*D*elem`
    // ratio that some backends' op implementations refuse).
    ggml_tensor * q_dhl = ggml_cont(ctx, q_dhl_view);
    ggml_tensor * q_rot = apply_rope_in_graph(ctx, q_dhl, cos_table, sin_table);
    // Step 4 — reshape back to the packed `[HD, L]` shape; same
    // memory, different ne labels.  Bytes are in time-major-flat
    // layout `data[c + t*HD]`.
    return ggml_reshape_2d(ctx, q_rot, (int64_t) n_heads * head_dim, L);
}

// ---------------------------------------------------------------------
// Audit finding F7 / Phase 2J — fused ConvNeXt block builder for
// the Supertonic vocoder.
//
// `convnext_block_ggml` (in supertonic_vocoder.cpp) used to compose
// the per-block residual chain as:
//
//   x [T0, C] ── depthwise_conv1d_causal_ggml ──▶ dw [T0, C]
//             ──▶ layer_norm_channel_ggml ──▶ ln [T0, C]
//                  (permute → cont [C,T0] → norm → mul → add →
//                   permute → cont [T0,C])     ← 2 conts each call
//             ──▶ conv1d_causal_ggml (pw1, K=1)
//                  (pad-noop → im2col [C,T0] → mul_mat → reshape)
//             ──▶ gelu
//             ──▶ conv1d_causal_ggml (pw2, K=1) (im2col again)
//             ──▶ mul γ  ──▶ add residual
//
// That chain costs per-block:
//   - 2 `ggml_cont` copies (LN front + LN back).
//   - 2 `ggml_im2col` copies (pw1 + pw2; K=1 reduces im2col to a
//     pure layout-shuffle copy).
// = 4 [T0=420, C=512] copies / block ≈ 3.36 MiB / block.
// × 10 ConvNeXt blocks = ~33.6 MiB redundant memory traffic
// per vocoder pass on a discrete GPU.
//
// The fused builder cuts this in half by:
//   1. Keeping the LN result in `[C, T0]` (channel-major) memory —
//      no back-permute / back-cont after `ggml_norm + mul + add`.
//   2. Lowering pw1 / pw2 to direct `ggml_mul_mat(w_2d, x_perm)`
//      against that `[C, T0]` LN output.  No `im2col` needed for
//      `K=1` — the same mathematical operation as the existing
//      `conv1d_causal_ggml` path with identical summation order.
//   3. Re-permuting once at the very end so the block output is
//      `[T0, C]` for the next block (and the existing trace /
//      readback plumbing keeps working unchanged).
//
// Net per block:
//   - Conts: 2 → 2 (LN front + final back-permute).  Same count.
//   - im2col copies: 2 → 0.  **Saves 2 [T0, C] copies per block.**
//   = 1.68 MiB / block × 10 blocks = ~16.8 MiB redundant traffic
//   eliminated per vocoder pass.  Matches the audit's F7 cost
//   estimate (the redundant 2× permute+cont copy traffic the
//   audit measured was the pair the LN front/back conts cause —
//   the im2col copies were missed by the audit but show the same
//   pattern, so the same fix removes both).
//
// Shape contract (mirrors the in-tree
// `supertonic_vocoder_convnext_weights`):
//   - `residual`    : F32, ne=[T0, C].  Block input + residual
//                     summed at the end.
//   - `dw_out`      : F32, ne=[T0, C].  Output of the upstream
//                     depthwise conv (kept outside this helper so
//                     the depthwise op stays in supertonic_vocoder.cpp).
//   - `ln_g`, `ln_b`: F32, ne=[C].  Layer-norm gamma + beta.
//   - `pw1_w`       : F32, ne=[K=1, IC=C, OC=hidden].
//   - `pw1_b`       : F32, ne=[hidden].  Nullable.
//   - `pw2_w`       : F32, ne=[K=1, IC=hidden, OC=C].
//   - `pw2_b`       : F32, ne=[C].  Nullable.
//   - `block_gamma` : F32, ne=[C].  Per-channel scaling.
//   - returns       : F32, ne=[T0, C].  Block output.
//
// Op-set used: `ggml_permute`, `ggml_cont`, `ggml_norm`,
// `ggml_reshape_2d`, `ggml_repeat`, `ggml_mul`, `ggml_add`,
// `ggml_mul_mat`, `ggml_gelu_erf`.  All universally supported
// (incl. baseline upstream OpenCL — no new ops introduced beyond
// the existing convnext block's surface).
//
// Parity-tested in `test_supertonic_convnext_block_fused.cpp`
// against a scalar reference of the per-block math on three
// shapes (tiny K=3/dilation=1, K=7/dilation=2, scale-up
// K=7/dilation=4).  Tolerance 1e-4 absolute on tiny shapes,
// 5e-4 on the scale-up (mul_mat sum-order parity).
inline ggml_tensor * convnext_block_fused_ggml(
        ggml_context * ctx,
        ggml_tensor *  residual,
        ggml_tensor *  dw_out,
        ggml_tensor *  ln_g,
        ggml_tensor *  ln_b,
        ggml_tensor *  pw1_w,
        ggml_tensor *  pw1_b,
        ggml_tensor *  pw2_w,
        ggml_tensor *  pw2_b,
        ggml_tensor *  block_gamma,
        float          eps = 1e-6f) {
    const int64_t C      = dw_out->ne[1];
    const int64_t hidden = pw1_w->ne[2];

    // Layer-norm — permute → cont → norm → γ·x + β.  Result stays
    // in `[C, T0]` (channel-major) so the next two pointwise convs
    // can consume it directly as a mul_mat right-hand side without
    // any im2col / re-permute overhead.
    ggml_tensor * y = ggml_cont(ctx, ggml_permute(ctx, dw_out, 1, 0, 2, 3));
    y = ggml_norm(ctx, y, eps);
    {
        // `repeat_like(v[C], y[C, T0]) → reshape(v, C, 1) + repeat`.
        // Reproduced inline so the helper stays header-only and
        // doesn't reach into the vocoder's anonymous-namespace
        // `repeat_like` wrapper.
        ggml_tensor * ln_g_2d = ggml_reshape_2d(ctx, ln_g, C, 1);
        ggml_tensor * ln_b_2d = ggml_reshape_2d(ctx, ln_b, C, 1);
        y = ggml_mul(ctx, y, ggml_repeat(ctx, ln_g_2d, y));
        y = ggml_add(ctx, y, ggml_repeat(ctx, ln_b_2d, y));
    }

    // pw1 — K=1 pointwise conv via `ggml_mul_mat`.
    //
    // pw1_w has ne=[1, IC=C, OC=hidden]; reshape to [IC, OC].
    // mul_mat(A=[K=IC, n=OC], B=[K=IC, m=T0]) → ne=[OC=hidden, T0]
    // with C[oc, t] = Σ_ic w_2d[ic, oc] * y[ic, t] — identical
    // arithmetic to the existing `conv1d_causal_ggml` path's
    // `mul_mat(im2col_reshape, w_reshape)` for `K=1`.
    ggml_tensor * pw1_w_2d = ggml_reshape_2d(
        ctx, pw1_w, pw1_w->ne[0] * pw1_w->ne[1], pw1_w->ne[2]);
    ggml_tensor * pw1_out = st_mul_mat(ctx, pw1_w_2d, y);
    if (pw1_b) {
        ggml_tensor * pw1_b_2d = ggml_reshape_2d(ctx, pw1_b, hidden, 1);
        pw1_out = ggml_add(ctx, pw1_out, ggml_repeat(ctx, pw1_b_2d, pw1_out));
    }

    // GELU is element-wise; the `[hidden, T0]` layout flows through
    // verbatim.
    ggml_tensor * gelu_out = ggml_gelu_erf(ctx, pw1_out);

    // pw2 — symmetric to pw1.  Output is `[C, T0]`.
    ggml_tensor * pw2_w_2d = ggml_reshape_2d(
        ctx, pw2_w, pw2_w->ne[0] * pw2_w->ne[1], pw2_w->ne[2]);
    ggml_tensor * pw2_out = st_mul_mat(ctx, pw2_w_2d, gelu_out);
    if (pw2_b) {
        ggml_tensor * pw2_b_2d = ggml_reshape_2d(ctx, pw2_b, C, 1);
        pw2_out = ggml_add(ctx, pw2_out, ggml_repeat(ctx, pw2_b_2d, pw2_out));
    }

    // Block-level γ scaling applied per-channel (broadcast over T0)
    // BEFORE the back-permute — gamma is a per-channel constant so
    // the multiplication commutes with the layout flip and we save
    // one ggml_repeat over [T0, C] vs. doing it after.
    {
        ggml_tensor * g_2d = ggml_reshape_2d(ctx, block_gamma, C, 1);
        pw2_out = ggml_mul(ctx, pw2_out, ggml_repeat(ctx, g_2d, pw2_out));
    }

    // Back to `[T0, C]` for the residual add and the next block.
    // This is the second (and last) ggml_cont in the helper — the
    // back-half of the F7 cost / savings pair.
    ggml_tensor * pw2_back = ggml_cont(
        ctx, ggml_permute(ctx, pw2_out, 1, 0, 2, 3));
    return ggml_add(ctx, residual, pw2_back);
}

// ---------------------------------------------------------------------
// Audit finding F12 / Phase 2L — in-graph time/channel transpose
// to kill the per-call `pack_time_channel_for_ggml` CPU loops.
//
// Background
// ----------
// The vector / text / duration estimator graph caches today hold
// their primary activation input as `ne=[L, C]` (axis 0 = L = time
// in GGML semantic).  GGML stores that as channel-major memory
// (`buf[c*L + t]`), but every caller hands the data in CPU-native
// time-major form (`x[t*C + c]`).  Callers paper over the
// mismatch by running `pack_time_channel_for_ggml(x_tc, L, C)` on
// the host — an `O(L * C)` loop with strided stores — and then
// uploading the packed buffer.  Audit F12: this is dozens of
// small CPU transposes per synth that also serialise the GPU
// dispatch.
//
// The fix (audit's recommended Option 2): keep the cache's upload
// tensor in `ne=[C, L]` (axis 0 = C = channels), so the caller
// can `ggml_backend_tensor_set` the CPU-native buffer byte-for-
// byte without any host pack, and have the graph itself emit
// `ggml_cont(ctx, ggml_transpose(ctx, x_tc_in))` to recover the
// `[L, C]` view downstream ops already consume.
//
// Why bit-exact
// -------------
// `ggml_transpose` is a strides-only view (zero arithmetic);
// `ggml_cont` is a memory rearrangement that materialises the
// natural-stride layout of `ne=[L, C]` — element (l, c) lands at
// byte `(l + c*L) * sizeof(float)`.  The host pack
// `pack_time_channel_for_ggml` writes `out[c*L + t] = x[t*C + c]`,
// i.e. the SAME byte at offset `(c*L + t) * sizeof(float)` carries
// the SAME float value.  See
// `test/test_supertonic_in_graph_transpose.cpp` for the bit-exact
// parity assertion.
//
// Shape contract:
//   - `x_tc_in` : F32, ne=[C, L].  Uploaded raw from CPU-native
//                 `x[t*C + c]` buffer (no pack).
//   - returns   : F32, ne=[L, C], naturally strided
//                 (`nb=[4, L*4]`).
//
// Op-set used: `ggml_transpose` + `ggml_cont`.  Both universally
// supported (incl. baseline upstream OpenCL).  No new ops.
inline ggml_tensor * transpose_time_channel_ggml(ggml_context * ctx,
                                                 ggml_tensor *  x_tc_in) {
    // `ggml_transpose` swaps axes 0 and 1 by reordering strides
    // (zero cost — same memory, new view).  `ggml_cont` then
    // materialises the natural-stride [L, C] layout that
    // downstream graph builders treat as the canonical
    // time-major input.  Byte-for-byte identical to
    // `pack_time_channel_for_ggml` writes.
    return ggml_cont(ctx, ggml_transpose(ctx, x_tc_in));
}

// Inline definition of the forward-declared portable leaky-relu helper
// above.  Must come after `supertonic_use_cpu_custom_ops()` and
// `supertonic_use_native_leaky_relu()` are declared so the dispatcher
// resolves at every call site.
//
// Two-stage dispatch:
//  1. CPU custom-op fast path — keeps the fused `ggml_leaky_relu`
//     builtin (one op + one `to_t` worker pass) on the CPU backend.
//  2. Backend-aware fast path — if the resolved GPU backend reports
//     it implements `GGML_OP_LEAKY_RELU` natively (Vulkan / Metal /
//     CUDA, plus chatterbox-patched OpenCL), emit the same single
//     fused builtin.  This collapses to one shader dispatch per
//     vocoder leaky-relu site instead of three (relu + scale + add)
//     and keeps the GPU command buffer ~33 % shorter on the vocoder
//     post-conv chain.
//  3. Otherwise, decompose into `(1-α)·relu(x) + α·x` — three
//     universally-supported ops.  The historical OpenCL bring-up
//     path (no chatterbox patch) lands here; correctness is bit-
//     identical to a fused builtin for the F32 path Supertonic uses.
//
// The `use_native_leaky_relu` query is set at backend init time by
// `ggml_backend_supports_op` against a synthetic LEAKY_RELU node, so
// the helper gets the right answer for every backend without a
// per-backend table.  See `supertonic_internal.h::supertonic_model::
// use_native_leaky_relu` for the rationale.
inline ggml_tensor * leaky_relu_portable_ggml(ggml_context * ctx, ggml_tensor * x, float alpha) {
    if (supertonic_use_cpu_custom_ops() || supertonic_use_native_leaky_relu()) {
        return ggml_leaky_relu(ctx, x, alpha, /*inplace=*/false);
    }
    // Conservative GPU fallback (op not advertised by the backend):
    // (1 - α)·relu(x) + α·x.  Three universally-supported ops.
    ggml_tensor * pos    = ggml_scale(ctx, ggml_relu(ctx, x), 1.0f - alpha);
    ggml_tensor * scaled = ggml_scale(ctx, x, alpha);
    return ggml_add(ctx, pos, scaled);
}

} // namespace tts_cpp::supertonic::detail
