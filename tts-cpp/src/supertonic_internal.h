#pragma once

#include <cstdint>
#include <array>
#include <cmath>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace tts_cpp::supertonic::detail {

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
    std::string language_wrap_mode = "open_close";
    std::string default_voice = "F1";
};

struct supertonic_voice_style {
    std::string name;
    ggml_tensor * ttl = nullptr; // (256, 50, 1) in ggml axis order for JSON (1, 50, 256)
    ggml_tensor * dp  = nullptr; // (16, 8, 1) in ggml axis order for JSON (1, 8, 16)
};

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
    // When true, the per-step vector-estimator attention graphs materialise
    // K/V into contiguous F16 before calling ggml_flash_attn_ext so OpenCL
    // (and other backends carrying the mixed-precision kernel) dispatch
    // the `flash_attn_f32_f16` path instead of the F32-only one — large
    // win on Adreno (see chatterbox PROGRESS.md OpenCL log).  Defaults to
    // false on CPU (the cblas attention path is already efficient there);
    // engine.cpp auto-enables it when the resolved backend is non-CPU,
    // matching chatterbox's --cfm-f16-kv-attn behaviour.
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

    std::map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, ggml_tensor *> source_tensors;
    std::unordered_map<std::string, supertonic_voice_style> voices;

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
bool load_supertonic_gguf(const std::string & path,
                          supertonic_model & model,
                          int n_gpu_layers = 0,
                          bool verbose = false,
                          int f16_weights = -1);
void free_supertonic_model(supertonic_model & model);
void supertonic_set_n_threads(supertonic_model & model, int n_threads);
void supertonic_graph_compute(const supertonic_model & model, ggml_cgraph * graph);

ggml_tensor * require_tensor(const supertonic_model & model, const std::string & name);
ggml_tensor * require_source_tensor(const supertonic_model & model, const std::string & source_name);

std::string supertonic_preprocess_text(const std::string & text,
                                       const std::string & language,
                                       const std::string & language_wrap_mode);
bool supertonic_text_to_ids(const supertonic_model & model,
                            const std::string & text,
                            const std::string & language,
                            std::vector<int32_t> & ids,
                            std::string * normalized_text = nullptr,
                            std::string * error = nullptr);

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
                                       std::vector<float> * next_latent_tc_out = nullptr);

// Process-wide alive registry: each loaded supertonic_model registers
// its generation_id with this set on success and unregisters at the
// start of free_supertonic_model.  The thread_local graph caches in
// supertonic_vocoder.cpp / supertonic_text_encoder.cpp /
// supertonic_vector_estimator.cpp own ggml_gallocr_t handles allocated
// against a specific model's ggml_backend_t; on a cache miss the
// existing teardown code calls ggml_gallocr_free(cache.allocr).  When
// the model that backed the cache has already been destroyed, that
// free path asserts inside the GPU-backend dylib finaliser.  The
// is_supertonic_alive() check at every free_*_cache() site lets the
// teardown skip the gallocr_free call for a generation that's no
// longer alive (the underlying GPU buffers were freed when the
// model's backend was freed; we only leak the gallocr bookkeeping
// struct itself, ~80 bytes per cache type).
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

struct supertonic_op_dispatch_scope {
    bool prev_use_cpu_custom_ops;
    bool prev_use_f16_attn;
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
// `dense_matmul_time_ggml`, whose output is a 2D packed tensor
// with `ne=[H*D, L]` (channel-major along axis 0, time along
// axis 1).  `apply_rope_to_packed_qk` adapts between the two
// layouts so the graph builders can bake the rotation in-place
// without reshaping the rest of the QKV plumbing:
//
//   - Re-views the packed tensor as `[head_dim, n_heads, L]` via
//     a zero-cost stride trick (`nb[0]=elem, nb[1]=D*4, nb[2]=H*D*4`)
//     — the memory pattern `data[t*H*D + h*D + d]` is preserved
//     bit-exactly.
//   - Materialises a contiguous copy (`ggml_cont`) so the
//     downstream `ggml_concat` inside `apply_rope_in_graph` sees
//     monotonically-increasing strides.
//   - Calls `apply_rope_in_graph(ctx, x_dhl, cos, sin)`.
//   - Reshapes the rotated `[D, H, L]` result back to `[H*D, L]`
//     so call sites can keep their existing `ggml_set_output` +
//     `tensor_to_time_channel` plumbing unchanged.
//
// Cost vs. the current host-side `apply_rope`:
//   - Eliminates 40 CPU rotations / synth (~50 µs each ≈ 2 ms
//     wall-time on the default 5-step × 4-RoPE-site schedule).
//   - Trades for one extra `ggml_cont` per site (small kernel
//     on GPU; ~0).  No new ops beyond `view + cont + reshape +
//     ggml_concat` chain already proven by the inner helper.
//
// Universally-supported ops only: `ggml_view_3d`, `ggml_cont`,
// `ggml_reshape_2d` + everything `apply_rope_in_graph` uses.
// Green on baseline upstream OpenCL.
//
// Parity-tested in `test_supertonic_rope_packed_qk.cpp` against
// the scalar `apply_rope` on the two hot vector-estimator shapes
// (`q_len=20 × H=4 × D=64`, `kv_len=32 × H=4 × D=64`) and a
// degenerate `L=1` trip-wire.  Tolerance `1e-4` absolute.
inline ggml_tensor * apply_rope_to_packed_qk(ggml_context * ctx,
                                              ggml_tensor * q,
                                              ggml_tensor * cos_table,
                                              ggml_tensor * sin_table,
                                              int n_heads,
                                              int head_dim) {
    const int64_t L  = q->ne[1];
    const int64_t HD = q->ne[0];
    (void) HD; // assertion-only; compiler may drop in NDEBUG.
    GGML_ASSERT(HD == (int64_t) n_heads * head_dim);
    // q has natural strides for ne=[HD, L]: nb[0]=elem_size,
    // nb[1]=HD*elem_size.  A view with ne=[D, H, L] sharing the
    // same memory needs nb=[elem, D*elem, HD*elem]; element
    // (d, h, l) lands at offset `d + h*D + l*HD` in 4-byte units
    // — identical memory pattern to the original packed layout's
    // element (col=h*D+d, row=l) at `col + row*HD` = `h*D + d + l*HD`.
    ggml_tensor * q_dhl_view = ggml_view_3d(ctx, q,
        head_dim, n_heads, L,
        /*nb1=*/(size_t) head_dim * sizeof(float),
        /*nb2=*/(size_t) n_heads * head_dim * sizeof(float),
        /*offset=*/0);
    // Materialise a contiguous [D, H, L] copy so the downstream
    // concat / repeat ops in `apply_rope_in_graph` see natural
    // strides (`nb=[elem, D*elem, D*H*elem]`).  The view above is
    // legal but non-natural (nb[1]<nb[2] with a `D*elem`/`H*D*elem`
    // ratio that some backends' op implementations refuse).
    ggml_tensor * q_dhl = ggml_cont(ctx, q_dhl_view);
    ggml_tensor * q_rot = apply_rope_in_graph(ctx, q_dhl, cos_table, sin_table);
    // Reshape back to the packed [HD, L] shape; same memory,
    // different ne labels — downstream consumers (flash-attn cache
    // upload buffers, trace download) see the original layout.
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
    ggml_tensor * pw1_out = ggml_mul_mat(ctx, pw1_w_2d, y);
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
    ggml_tensor * pw2_out = ggml_mul_mat(ctx, pw2_w_2d, gelu_out);
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
// above.  Must come after `supertonic_use_cpu_custom_ops()` is
// declared so the dispatcher resolves at every call site.
inline ggml_tensor * leaky_relu_portable_ggml(ggml_context * ctx, ggml_tensor * x, float alpha) {
    if (supertonic_use_cpu_custom_ops()) {
        return ggml_leaky_relu(ctx, x, alpha, /*inplace=*/false);
    }
    // GPU lowering: (1 - α)·relu(x) + α·x.  Three universally-supported
    // ops, no GGML_OP_LEAKY_RELU dependency.
    ggml_tensor * pos    = ggml_scale(ctx, ggml_relu(ctx, x), 1.0f - alpha);
    ggml_tensor * scaled = ggml_scale(ctx, x, alpha);
    return ggml_add(ctx, pos, scaled);
}

} // namespace tts_cpp::supertonic::detail
