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
