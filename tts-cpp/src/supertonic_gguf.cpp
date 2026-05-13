#include "supertonic_internal.h"

#include "ggml-cpu.h"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <thread>

namespace tts_cpp::supertonic::detail {
namespace {

int64_t require_key(const gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) throw std::runtime_error(std::string("missing GGUF key: ") + key);
    return id;
}

uint32_t get_u32(const gguf_context * ctx, const char * key) {
    return gguf_get_val_u32(ctx, require_key(ctx, key));
}

float get_f32(const gguf_context * ctx, const char * key) {
    return gguf_get_val_f32(ctx, require_key(ctx, key));
}

bool get_bool_u32(const gguf_context * ctx, const char * key, bool fallback = false) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    return gguf_get_val_u32(ctx, id) != 0;
}

std::string get_string(const gguf_context * ctx, const char * key, const std::string & fallback = {}) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    return gguf_get_val_str(ctx, id);
}

std::vector<std::string> get_string_array(const gguf_context * ctx, const char * key) {
    int64_t id = require_key(ctx, key);
    size_t n = gguf_get_arr_n(ctx, id);
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(ctx, id, i));
    }
    return out;
}

ggml_tensor * get_tensor_or_null(const supertonic_model & model, const std::string & name) {
    auto it = model.tensors.find(name);
    return it == model.tensors.end() ? nullptr : it->second;
}

// Compute the storage type for a model tensor given the source type from
// the GGUF and the engine's compute-precision selector.  Non-matmul tensors
// (biases, norms, embeddings — stored as f32 in the GGUF) are unaffected;
// only quantized matmul weights actually change destination type.
//
// Truth table:
//   precision \ src_type      | F32  | F16  | Q8_0
//   --------------------------+------+------+------
//   F32 (default)             | F32  | F32  | F32
//   F16  (Phase B1)           | F32  | F16  | F16
//   Q8_0 (Phase A3)           | F32  | F32  | Q8_0   <-- key win: Metal keeps q8_0
//
// F32 row preserves the historical behaviour exactly.
// Predicate: is `tensor_name` a true matmul weight that lands in a
// `ggml_mul_mat(weight, activation)` call (weight as src0) where Metal
// can dispatch `kernel_mul_mm_q8_0_f32` directly?
//
// Today this is only the vector_estimator's per-step matmul weights —
// those go through `dense_matmul_time_wt_pretransposed_ggml` (the
// B2-partial helper) which uses the pretransposed weight as src0 and
// dispatches the optimised q8_0 mat-mat kernel.
//
// Other GGUF q8_0 sources (text_encoder, duration, speech-prompted
// attention) still flow through `dense_matmul_time_ggml`, which does
// `ggml_cont(ggml_transpose(w))` at compute time — and Metal has no
// CONT kernel for q8_0, so we'd crash.  Phase A3 follow-up: extend
// the pretranspose-aware helper to those sites and broaden this
// predicate.
bool is_supertonic_matmul_weight_name(const std::string & name) {
    return name.find("vector_estimator:onnx::MatMul_") != std::string::npos;
}

ggml_type target_supertonic_storage_type(const std::string & name,
                                         enum ggml_type src_type,
                                         supertonic_precision precision,
                                         bool backend_is_cpu) {
    // Only quantized matmul-weight tensors are subject to the precision
    // selector.  Everything else (biases, norms, scales, the unicode
    // indexer i32 lookup, etc.) is passed through unchanged so we don't
    // attempt a dequant on types that don't have a to_float trait.
    const bool is_quantized_weight =
        (src_type == GGML_TYPE_Q8_0) || (src_type == GGML_TYPE_F16);
    if (!is_quantized_weight) return src_type;

    switch (precision) {
        case supertonic_precision::F32:  return GGML_TYPE_F32;
        case supertonic_precision::F16:
            // Asymmetric like q8_0: on CPU dequant everything to f32 (AMX
            // cblas takes f32).  On non-CPU keep f16 ONLY for true matmul-
            // weight tensors that flow through dense_matmul_time_pretransposed_*
            // — these dispatch ggml-metal's `kernel_mul_mm_f16_f32` directly.
            // Other quantized GGUF tensors (relpos embeddings, conv1d
            // kernels, per-channel scales used in plain ggml_mul) flow into
            // ggml_metal_op_bin which asserts f32 on both srcs, so we dequant
            // them at load.
            if (!backend_is_cpu && is_supertonic_matmul_weight_name(name)) {
                return GGML_TYPE_F16;
            }
            return GGML_TYPE_F32;
        case supertonic_precision::Q8_0:
            // Asymmetric: on CPU, ALWAYS dequant to f32 so cblas/AMX takes
            // the weights (q8_0 path on CPU is NEON-only and loses the AMX
            // advantage; not worth the parity drift).  On non-CPU backends,
            // keep q8_0 ONLY for true matmul-weight tensors that flow
            // through `dense_matmul_time_wt_pretransposed_ggml`'s
            // weight-as-src0 ordering — other quantized GGUF tensors
            // (relpos embeddings, conv1d kernels) use op patterns that
            // Metal lacks q8_0 kernels for.
            if (!backend_is_cpu &&
                src_type == GGML_TYPE_Q8_0 &&
                is_supertonic_matmul_weight_name(name)) {
                return GGML_TYPE_Q8_0;
            }
            return GGML_TYPE_F32;
    }
    return GGML_TYPE_F32;
}

bool needs_supertonic_tensor_conversion(enum ggml_type src_type,
                                        enum ggml_type dst_type) {
    return src_type != dst_type;
}

bool should_expand_supertonic_tensor(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0;
}

std::vector<float> expand_supertonic_tensor_to_f32(const ggml_tensor * src) {
    const int64_t n = ggml_nelements(src);
    std::vector<float> out((size_t) n);
    const void * data = ggml_get_data(src);
    // Use the public ggml_get_type_traits() API instead of the
    // internal ggml-quants.h helpers.  ggml-quants.h lives under
    // ggml/src/ and isn't shipped by the ggml-speech vcpkg port,
    // so direct includes break system-ggml builds (the integrated
    // tts-cpp port path).  The type-traits to_float function pointer
    // is the public dequantization entry-point and covers F16, Q8_0
    // and every other ggml type uniformly.
    const ggml_type_traits * tr = ggml_get_type_traits(src->type);
    if (!tr || !tr->to_float) {
        throw std::runtime_error(std::string("unsupported Supertonic tensor expansion type ") +
                                 ggml_type_name(src->type));
    }
    tr->to_float(data, out.data(), n);
    return out;
}

// Convert a GGUF tensor's data into `out_buf`, which the caller has sized
// to `ggml_row_size(dst_type, n_elems) * (n_rows ...)` — i.e. ggml_nbytes
// for the destination tensor shape.  Supports any pair the ggml type
// traits cover: F32 ↔ F16 ↔ Q8_0.  Always converts via f32 as the pivot
// because that's the only API surface ggml exports publicly.
void convert_supertonic_tensor_data(const ggml_tensor * src,
                                    enum ggml_type dst_type,
                                    std::vector<uint8_t> & out_buf) {
    const int64_t n = ggml_nelements(src);
    const void * src_data = ggml_get_data(src);

    if (src->type == dst_type) {
        // No conversion needed — caller should ideally have skipped this path
        // and uploaded the raw GGUF bytes, but handle it for completeness.
        const size_t bytes = ggml_nbytes(src);
        out_buf.resize(bytes);
        std::memcpy(out_buf.data(), src_data, bytes);
        return;
    }

    // Pivot through f32 using the public ggml_get_type_traits() API.
    // `ggml_get_type_traits_cpu()->from_float` is also public for the
    // reverse direction (f32 → quantized).
    std::vector<float> f32_pivot((size_t) n);
    const ggml_type_traits * src_tr = ggml_get_type_traits(src->type);
    if (!src_tr || !src_tr->to_float) {
        throw std::runtime_error(std::string("Supertonic load: missing to_float for ") +
                                 ggml_type_name(src->type));
    }
    src_tr->to_float(src_data, f32_pivot.data(), n);

    if (dst_type == GGML_TYPE_F32) {
        out_buf.resize(f32_pivot.size() * sizeof(float));
        std::memcpy(out_buf.data(), f32_pivot.data(), out_buf.size());
        return;
    }

    const size_t dst_bytes = ggml_row_size(dst_type, n);
    out_buf.resize(dst_bytes);

    const ggml_type_traits_cpu * dst_tr = ggml_get_type_traits_cpu(dst_type);
    if (!dst_tr || !dst_tr->from_float) {
        throw std::runtime_error(std::string("Supertonic load: missing from_float for ") +
                                 ggml_type_name(dst_type));
    }
    dst_tr->from_float(f32_pivot.data(), out_buf.data(), n);
}

ggml_backend_t init_supertonic_backend(int n_gpu_layers, bool verbose, int vulkan_device = 0) {
#ifdef GGML_USE_CUDA
    if (n_gpu_layers > 0) {
        ggml_backend_t b = ggml_backend_cuda_init(0);
        if (b) { if (verbose) fprintf(stderr, "supertonic: using CUDA backend\n"); return b; }
    }
#endif
#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        ggml_backend_t b = ggml_backend_metal_init();
        if (b) { if (verbose) fprintf(stderr, "supertonic: using Metal backend\n"); return b; }
    }
#endif
#ifdef GGML_USE_VULKAN
    if (n_gpu_layers > 0) {
        // QVAC-18605 round 3 — Vulkan device selection, robust init
        // with multi-device auto-pick.
        //
        // Range-check the requested index against
        // `ggml_backend_vk_get_device_count()` so an out-of-range
        // value (CLI typo / wrong-machine config) fails loud here
        // rather than silently falling through to CPU and hiding
        // the perf cliff under a "Vulkan was on, why is it slow?"
        // mystery.  `vulkan_device == -1` triggers auto-pick: walk
        // every visible adapter, query `ggml_backend_vk_get_device_memory`
        // to read the free VRAM, and dispatch into the pure-logic
        // `resolve_vulkan_device_index` helper which picks
        // `argmax(free_vram)` (ties → lower index).  Negative values
        // other than -1 are reserved for future policies and throw.
        const int dev_count = ggml_backend_vk_get_device_count();
        if (dev_count <= 0) {
            // No Vulkan adapter visible — try the next backend in the
            // priority list (OpenCL below, then CPU).  This branch
            // matters on machines that ship libvulkan + the loader
            // but no working ICD (e.g. headless CI without llvmpipe).
            if (verbose) {
                fprintf(stderr, "supertonic: GGML_USE_VULKAN=1 but ggml_backend_vk_get_device_count()=0; falling through\n");
            }
        } else {
            std::vector<size_t> free_vram_per_device;
            free_vram_per_device.reserve((size_t) dev_count);
            for (int i = 0; i < dev_count; ++i) {
                size_t free = 0, total = 0;
                ggml_backend_vk_get_device_memory(i, &free, &total);
                free_vram_per_device.push_back(free);
                if (verbose && vulkan_device == -1) {
                    char desc[256] = {0};
                    ggml_backend_vk_get_device_description(i, desc, sizeof(desc) - 1);
                    fprintf(stderr,
                            "supertonic: vulkan device %d: %s — free %.0f MB / total %.0f MB\n",
                            i,
                            desc[0] ? desc : "unknown",
                            (double) free  / (1024.0 * 1024.0),
                            (double) total / (1024.0 * 1024.0));
                }
            }
            // Throws on invalid input; let it propagate so the CLI
            // surfaces the message verbatim.
            const int idx = resolve_vulkan_device_index(vulkan_device, free_vram_per_device);
            ggml_backend_t b = ggml_backend_vk_init((size_t) idx);
            if (b) {
                if (verbose) {
                    char desc[256] = {0};
                    ggml_backend_vk_get_device_description(idx, desc, sizeof(desc) - 1);
                    if (vulkan_device == -1) {
                        fprintf(stderr,
                                "supertonic: auto-picked Vulkan device %d (%s) — most free VRAM of %d adapter(s)\n",
                                idx, desc[0] ? desc : "unknown", dev_count);
                    } else {
                        fprintf(stderr, "supertonic: using Vulkan backend (device %d: %s)\n",
                                idx, desc[0] ? desc : "unknown");
                    }
                }
                return b;
            }
            if (verbose) {
                fprintf(stderr, "supertonic: ggml_backend_vk_init(%d) failed; falling through\n", idx);
            }
        }
    }
#else
    (void) vulkan_device;
#endif
#ifdef GGML_USE_OPENCL
    if (n_gpu_layers > 0) {
        ggml_backend_reg_t reg = ggml_backend_opencl_reg();
        if (reg && ggml_backend_reg_dev_count(reg) > 0) {
            ggml_backend_t b = ggml_backend_opencl_init();
            if (b) { if (verbose) fprintf(stderr, "supertonic: using OpenCL backend\n"); return b; }
        }
    }
#endif
    ggml_backend_t b = ggml_backend_cpu_init();
    if (!b) throw std::runtime_error("ggml_backend_cpu_init failed");
    if (verbose) fprintf(stderr, "supertonic: using CPU backend\n");
    return b;
}

// QVAC-18605 — backend capability probe for `GGML_OP_LEAKY_RELU`.
//
// Builds a throwaway 1-element F32 tensor + a LEAKY_RELU node (no
// alloc, no compute) inside a tiny `ggml_init` scratch context, then
// asks the backend whether it would accept the op.  The synthetic
// node is the same shape Supertonic actually emits (axis-0 contig F32),
// so a `true` answer guarantees the real graphs in the vocoder will
// dispatch the fused builtin.
//
// Why dynamic instead of a hard-coded backend table?  The set of
// backends shipping `LEAKY_RELU` shifts with chatterbox-ggml patch
// state (OpenCL gets it via a vendored patch but plain upstream
// doesn't).  The dynamic probe keeps the right answer when the patch
// is added or removed without touching this TU.
//
// Costs nothing on the hot path — runs once per `load_supertonic_gguf`
// call.
bool backend_supports_native_leaky_relu(ggml_backend_t backend) {
    if (!backend) return false;
    ggml_init_params probe_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * probe_ctx = ggml_init(probe_params);
    if (!probe_ctx) return false;
    bool ok = false;
    try {
        ggml_tensor * x  = ggml_new_tensor_1d(probe_ctx, GGML_TYPE_F32, 16);
        ggml_tensor * op = ggml_leaky_relu(probe_ctx, x, 0.1f, /*inplace=*/false);
        ok = (op != nullptr) && ggml_backend_supports_op(backend, op);
    } catch (...) {
        ok = false;
    }
    ggml_free(probe_ctx);
    return ok;
}

// QVAC-18605 — runtime check: backend is `ggml-vulkan`.
//
// Wraps `ggml_backend_is_vk` behind a `#ifdef GGML_USE_VULKAN` guard so
// the flag-population code in `load_supertonic_gguf` works on both
// Vulkan-enabled and Vulkan-disabled builds without `#ifdef` clutter
// at every consumer site.  Returns `false` on Vulkan-disabled builds
// so the dispatch helpers behave as if the backend were not Vulkan
// (which is correct — the backend can't be Vulkan if Vulkan isn't in
// the build).
bool backend_is_vulkan(ggml_backend_t backend) {
#ifdef GGML_USE_VULKAN
    return backend && ggml_backend_is_vk(backend);
#else
    (void) backend;
    return false;
#endif
}

// QVAC-18605 — internal-named alias for the public probe symbol.
// The anon-namespace function name keeps the local TU references
// short; the public-symbol forwarder below resolves the
// `supertonic_backend_supports_f16_kv_flash_attn` declaration in
// `supertonic_internal.h`.
//
// QVAC-18605 — backend capability probe for F16-K/V `FLASH_ATTN_EXT`.
//
// The OpenCL bring-up's auto-enable policy (`!backend_is_cpu`) blindly
// turns on F16 K/V dispatch on any non-CPU backend.  That works for
// OpenCL (the chatterbox patch unconditionally accepts the op) and
// for Vulkan when the head dim is a multiple of 8 (Supertonic's
// head_dim=64 satisfies that), but a future backend / driver / shape
// combo could reject the op at graph time — and a graph-build failure
// at the first synth call is much harder to triage than a load-time
// auto-disable + a clear log line.
//
// The probe builds a synthetic `ggml_flash_attn_ext` node with the
// shape Supertonic actually emits — Q=[head_dim, q_len, n_heads] F32,
// K/V=[head_dim, kv_len, n_heads] F16, no mask — matching the live
// call site in `build_text_attention_cache` (supertonic_vector_estimator.cpp).
// q_len is set to a multiple of n_heads (= 16) so the live `q_len=70`
// (not divisible by 4) doesn't tickle a probe-only `ggml_can_mul_mat`
// rejection; the GPU dispatch supports both the divisible and non-
// divisible cases at runtime, so probe-shape divisibility is purely
// a probe-API concern.
//
// On a `false` answer the auto-policy refuses to enable F16 attention
// (the F32 path stays correct, just slower).  Manual override via
// `--f16-attn 1` still forces the F16 path for benchmarking; this
// probe only gates the *auto* policy.
//
// Cost: one ggml_init + ~6 tensor allocations + one supports_op call
// at load time.  Zero hot-path cost — and the result is now memoised
// per `ggml_backend_t` handle by `cached_backend_supports_*` below so
// the engine + bench + load_supertonic_gguf trio doesn't re-run the
// probe three times for the same backend.
bool backend_supports_f16_kv_flash_attn_uncached(ggml_backend_t backend) {
    if (!backend) return false;
    ggml_init_params probe_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * probe_ctx = ggml_init(probe_params);
    if (!probe_ctx) return false;
    bool ok = false;
    try {
        constexpr int head_dim = 64;
        constexpr int n_heads  = 4;
        // q_len chosen as `n_heads * 4` so `ggml_can_mul_mat(k, q)`'s
        // probe-only `q.ne[2] % k.ne[2] == 0` constraint is satisfied
        // (n_heads % n_heads = 0 is the live-call invariant; here we
        // use a Q with ne[2] = n_heads, ne[1] = q_len, so the same
        // shape contract holds).
        constexpr int q_len    = 16;
        constexpr int kv_len   = 16;
        // Live shape from `build_text_attention_cache`:
        //   q_in: [head_dim, q_len, n_heads]  (F32)
        //   k_in: [head_dim, kv_len, n_heads] (F16 after `ggml_cpy`)
        //   v_in: [head_dim, kv_len, n_heads] (F16 after `ggml_cpy`)
        ggml_tensor * q  = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_F32, head_dim, q_len, n_heads);
        ggml_tensor * k  = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_F16, head_dim, kv_len, n_heads);
        ggml_tensor * v  = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_F16, head_dim, kv_len, n_heads);
        ggml_tensor * op = ggml_flash_attn_ext(probe_ctx, q, k, v, nullptr,
                                               1.0f / (float) head_dim, 0.0f, 0.0f);
        ok = (op != nullptr) && ggml_backend_supports_op(backend, op);
    } catch (...) {
        ok = false;
    }
    ggml_free(probe_ctx);
    return ok;
}

// QVAC-18605 follow-up — backend capability probe for the Q8_0
// K/V `FLASH_ATTN_EXT` variant.
//
// Vulkan's `GGML_OP_FLASH_ATTN_EXT` `supports_op` advertises Q8_0
// (and Q4_0) K/V types in the scalar and coopmat2 paths
// (`ggml-vulkan.cpp:15257`).  Switching K/V from F16 to Q8_0
// halves the upload bandwidth into the per-step attention cache
// (50 KB → 25 KB per K and V on Supertonic's hot shape),
// equivalently ~1 MB / synth on the default 5-step × 4-site
// schedule, in exchange for a small (~0.5 %) relative-error drift
// vs F16 K/V on the attention output.  Worth the trade on memory-
// bandwidth-bound mobile GPUs (Adreno, Mali) once measured on a
// real device.
//
// This PR adds the probe + caches the result, but does NOT yet
// wire `model.use_q8_kv_attn` into the live dispatch site — Q8_0
// K/V drift hasn't been measured against the existing F16 K/V
// parity harness on a real Vulkan adapter.  The probe primes the
// capability cache so a follow-up patch can flip the dispatch
// behind a `--kv-attn-type q8_0` opt-in without re-running the
// `supports_op` query.  Tracked in PROGRESS_SUPERTONIC.md
// "Deferred work".
bool backend_supports_q8_0_kv_flash_attn_uncached(ggml_backend_t backend) {
    if (!backend) return false;
    ggml_init_params probe_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * probe_ctx = ggml_init(probe_params);
    if (!probe_ctx) return false;
    bool ok = false;
    try {
        // Same shape as the F16-K/V probe; only K/V dtype differs.
        // Q8_0 is a 32-element-per-block quantisation, so kv_len
        // must be a multiple of 32 to satisfy the live
        // `ggml_can_repeat` / row-stride invariants the GPU
        // dispatch requires.  The live call site has kv_len = 50;
        // we pick 32 here as the smallest multiple-of-Q8_0-block
        // that exercises the same `supports_op` switch.
        constexpr int head_dim = 64;
        constexpr int n_heads  = 4;
        constexpr int q_len    = 16;
        constexpr int kv_len   = 32;
        ggml_tensor * q  = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_F32,  head_dim, q_len,  n_heads);
        ggml_tensor * k  = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_Q8_0, head_dim, kv_len, n_heads);
        ggml_tensor * v  = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_Q8_0, head_dim, kv_len, n_heads);
        ggml_tensor * op = ggml_flash_attn_ext(probe_ctx, q, k, v, nullptr,
                                               1.0f / (float) head_dim, 0.0f, 0.0f);
        ok = (op != nullptr) && ggml_backend_supports_op(backend, op);
    } catch (...) {
        ok = false;
    }
    ggml_free(probe_ctx);
    return ok;
}

// QVAC-18605 round 3 — backend capability probe for Vulkan's
// `ggml_backend_vk_host_buffer_type()`.
//
// Vulkan exposes a host-visible, device-coherent buffer type
// that lets the CPU fill an input tensor without going through
// ggml-vulkan's internal staging buffer.  Wiring the actual
// upload path through that buffer is a per-engine refactor
// (input scratchpad allocator separate from the model gallocr);
// this round only adds the probe so the capability cache is
// primed for that follow-up.  The bench output surfaces the
// flag so operators can confirm the host-buffer-type path is
// available on their adapter before flipping the (future)
// `--vulkan-pinned-uploads` opt-in.
//
// Probe is trivial: succeeds iff the backend is Vulkan AND
// `ggml_backend_vk_host_buffer_type()` returns non-null.  On a
// Vulkan-disabled build the entire branch compiles out to
// `return false`.
bool backend_supports_pinned_host_buffer_uncached(ggml_backend_t backend) {
    if (!backend) return false;
#ifdef GGML_USE_VULKAN
    if (!ggml_backend_is_vk(backend)) return false;
    return ggml_backend_vk_host_buffer_type() != nullptr;
#else
    return false;
#endif
}

// QVAC-18605 round 3 — backend capability probe for the BF16 K/V
// `FLASH_ATTN_EXT` variant.
//
// Vulkan's `GGML_OP_FLASH_ATTN_EXT` `supports_op` advertises
// BF16 K/V via the coopmat2-only path
// (`ggml-vulkan.cpp:GGML_OP_FLASH_ATTN_EXT` case branch around
// line 15257).  BF16 has the same per-element size as F16 (2
// bytes), so the upload bandwidth is identical, but BF16's
// wider exponent range (8 bits vs. F16's 5) avoids the
// occasional underflow on small attention scores that drives
// F16's ~0.2 % tolerance widening on the parity harness.
// On hardware with `cooperative_matrix2` (NVIDIA Ampere+, AMD
// RDNA3+) BF16 K/V is also faster than F16 K/V because the
// coopmat2 BF16 multiply-accumulate ops are dispatched at
// hardware-tensor-core throughput.
//
// Like the Q8_0 K/V probe, this round adds the probe + caches
// the result as a forward-compat capability; the live dispatch
// site isn't yet wired (a follow-up will gate `--kv-attn-type
// bf16` on the probe so the dispatch flips when the cache says
// the hardware accepts the op).
//
// Probe shape mirrors the F16-K/V probe with the K/V dtype set
// to `GGML_TYPE_BF16` — same `kv_len = 16` (BF16 row stride is
// `head_dim * 2` bytes, identical to F16).
bool backend_supports_bf16_kv_flash_attn_uncached(ggml_backend_t backend) {
    if (!backend) return false;
    ggml_init_params probe_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * probe_ctx = ggml_init(probe_params);
    if (!probe_ctx) return false;
    bool ok = false;
    try {
        constexpr int head_dim = 64;
        constexpr int n_heads  = 4;
        constexpr int q_len    = 16;
        constexpr int kv_len   = 16;
        ggml_tensor * q  = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_F32,  head_dim, q_len,  n_heads);
        ggml_tensor * k  = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_BF16, head_dim, kv_len, n_heads);
        ggml_tensor * v  = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_BF16, head_dim, kv_len, n_heads);
        ggml_tensor * op = ggml_flash_attn_ext(probe_ctx, q, k, v, nullptr,
                                               1.0f / (float) head_dim, 0.0f, 0.0f);
        ok = (op != nullptr) && ggml_backend_supports_op(backend, op);
    } catch (...) {
        ok = false;
    }
    ggml_free(probe_ctx);
    return ok;
}

// QVAC-18605 follow-up — backend capability probe for the hot
// F16-weight `mul_mat` shape Supertonic dispatches every step.
//
// Mirror of `backend_supports_f16_kv_flash_attn_uncached`: the
// `use_f16_weights` auto-policy used to flip on `!backend_is_cpu`
// blindly, with no check that the resolved backend would accept the
// resulting `mul_mat(F16 weight, F32 activation) → F32` graph node
// for the shapes the audit identified as hot.  Every shipping GPU
// backend (CUDA / Metal / Vulkan / OpenCL) does support this combo,
// but a future debug-shim / partial-port backend that wires up
// `mul_mat` for F32-only would crash at first synth call when
// `f16_weights` was auto-enabled — exactly the failure mode the
// F16-K/V probe was added to prevent.
//
// Probe shape mirrors the vector-estimator attention W_query
// matmul (`[head_dim*n_heads = 256, in_dim = 256]` weight, F16
// storage; `[256, q_len = 16]` activation, F32; output F32),
// which is the most common F16-weight matmul site in the
// production graph (32 such matmuls per synth, 5-step schedule).
//
// Cost: one ggml_init + 3 tensor allocations + one supports_op
// call at load time.  Zero hot-path cost — memoised per
// `ggml_backend_t` by `cached_backend_supports_*` below.
bool backend_supports_f16_mul_mat_uncached(ggml_backend_t backend) {
    if (!backend) return false;
    ggml_init_params probe_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * probe_ctx = ggml_init(probe_params);
    if (!probe_ctx) return false;
    bool ok = false;
    try {
        // Live shape from the vector-estimator attention W_query /
        // W_key / W_value matmul site.
        constexpr int head_dim = 64;
        constexpr int n_heads  = 4;
        constexpr int width    = head_dim * n_heads;  // 256
        constexpr int q_len    = 16;
        ggml_tensor * w  = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_F16, width, width);
        ggml_tensor * x  = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_F32, width, q_len);
        ggml_tensor * op = ggml_mul_mat(probe_ctx, w, x);
        ok = (op != nullptr) && ggml_backend_supports_op(backend, op);
    } catch (...) {
        ok = false;
    }
    ggml_free(probe_ctx);
    return ok;
}

// QVAC-18605 follow-up — process-wide capability-probe cache.
//
// Three sites probe the same `ggml_backend_t` for the same op
// support boolean: `load_supertonic_gguf` (LEAKY_RELU at backend
// resolution time), `Engine::Engine` and `supertonic_bench`'s
// `main` (F16-K/V flash-attn at auto-policy time).  Engine + bench
// life-cycles also call `load_supertonic_gguf` themselves, so the
// uncached probe set fires on average 2–3 times per backend per
// process.  On a CPU backend each probe costs ~1 µs (ggml_init +
// supports_op walks a small switch).  On Vulkan, `supports_op`
// inspects the device's pipeline state and may force coopmat
// shader specialisation lookup — measured ~50–200 µs on Adreno /
// llvmpipe / RADV in microbenchmarks.  Negligible per-probe but
// visible in cold-start traces, and the cache eliminates 100 % of
// the redundancy.
//
// Cache shape: `unordered_map<ggml_backend_t, probe_results>`.
// Key is the backend handle (stable for the backend's lifetime;
// recycled keys after a backend is freed are technically possible
// but the per-handle entry cost is ~24 bytes, so we don't bother
// invalidating on free).  Test seam: `supertonic_clear_capability_cache`
// drops every entry — used by the unit test to verify the cache
// is hit on the second call.
//
// Thread-safety: guarded by a single std::mutex.  Hot path is
// load-time only, never the per-synth path, so contention is
// negligible.
struct backend_capabilities {
    bool native_leaky_relu;
    bool f16_kv_flash_attn;
    bool f16_mul_mat;
    // QVAC-18605 follow-up — Q8_0 K/V flash-attn support.  Probed
    // here as a forward-compat capability; the dispatch isn't yet
    // wired (see `backend_supports_q8_0_kv_flash_attn_uncached`'s
    // docstring + PROGRESS_SUPERTONIC.md "Deferred work").
    bool q8_0_kv_flash_attn;
    // QVAC-18605 round 3 — BF16 K/V flash-attn support.  Probed
    // here as a forward-compat capability; the dispatch isn't yet
    // wired (see `backend_supports_bf16_kv_flash_attn_uncached`'s
    // docstring + PROGRESS_SUPERTONIC.md "Deferred work").  BF16
    // K/V is the wider-exponent alternative to F16 K/V — mostly
    // useful on Vulkan with cooperative_matrix2 support.
    bool bf16_kv_flash_attn;
    // QVAC-18605 round 3 — pinned-host-buffer-type availability.
    // True iff the backend is Vulkan AND
    // `ggml_backend_vk_host_buffer_type()` returns non-null.
    // Forward-compat — primes the cache for a future per-engine
    // input-scratchpad refactor that uses the host-pinned buffer
    // to skip ggml-vulkan's internal staging-buffer hop on the
    // per-step uploads.
    bool pinned_host_buffer;
};

inline std::mutex & capability_cache_mu() {
    static std::mutex m;
    return m;
}
inline std::unordered_map<ggml_backend_t, backend_capabilities> & capability_cache() {
    static std::unordered_map<ggml_backend_t, backend_capabilities> c;
    return c;
}
// Probe-call counter for the regression test in
// test_supertonic_capability_cache.cpp: each cached_backend_supports_*
// helper bumps the counter only when it actually invokes the
// uncached probe (i.e. on a cold cache).  The test asserts that
// the counter advances by exactly one across N consecutive
// cached_backend_supports_native_leaky_relu(b) calls on the same
// backend.
std::atomic<uint64_t> & capability_probe_call_counter() {
    static std::atomic<uint64_t> n{0};
    return n;
}

const backend_capabilities & cached_backend_capabilities(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lk(capability_cache_mu());
    auto & c = capability_cache();
    auto it = c.find(backend);
    if (it != c.end()) return it->second;
    capability_probe_call_counter().fetch_add(1, std::memory_order_relaxed);
    backend_capabilities caps;
    caps.native_leaky_relu   = backend_supports_native_leaky_relu(backend);
    caps.f16_kv_flash_attn   = backend_supports_f16_kv_flash_attn_uncached(backend);
    caps.f16_mul_mat         = backend_supports_f16_mul_mat_uncached(backend);
    caps.q8_0_kv_flash_attn  = backend_supports_q8_0_kv_flash_attn_uncached(backend);
    caps.bf16_kv_flash_attn  = backend_supports_bf16_kv_flash_attn_uncached(backend);
    caps.pinned_host_buffer  = backend_supports_pinned_host_buffer_uncached(backend);
    return c.emplace(backend, caps).first->second;
}

// Backwards-compatible name kept for the in-tree callers that already
// reference it; routes through the cache.
bool backend_supports_f16_kv_flash_attn(ggml_backend_t backend) {
    return cached_backend_capabilities(backend).f16_kv_flash_attn;
}

void set_env_if_unset(const char * name, const char * value) {
    if (std::getenv(name) != nullptr) return;
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

// QVAC-18605 round 7 — pure-logic key-validator for the
// `apply_vulkan_env_overrides` ALL-OR-NOTHING contract.  Returns
// `true` (with `out_bad_key` populated) on the first key that
// doesn't start with `GGML_VK_`, `false` on success.  Split out
// so the public helper validates the entire map BEFORE touching
// any env var.
//
// Out-param + bool return (instead of returning `std::string`
// with empty-as-success) because an empty-string KEY is itself
// invalid input — a pure-string return would conflate "no bad
// key found" with "the bad key was the empty string".
bool find_invalid_vulkan_env_key(const std::map<std::string, std::string> & overrides,
                                 std::string & out_bad_key) {
    static const std::string prefix = "GGML_VK_";
    for (const auto & kv : overrides) {
        const std::string & key = kv.first;
        if (key.size() <= prefix.size() ||
            key.compare(0, prefix.size(), prefix) != 0) {
            out_bad_key = key;
            return true;
        }
    }
    return false;
}

void configure_supertonic_blas_threads_once() {
#if defined(TTS_CPP_USE_ACCELERATE)
    static bool configured = false;
    if (configured) return;
    configured = true;
    // The Supertonic CPU graphs already parallelize across GGML tasks. Letting
    // Accelerate spawn a second worker pool for every small pointwise matmul
    // hurts vector scaling on 3-4 thread runs.
    set_env_if_unset("VECLIB_MAXIMUM_THREADS", "1");
#elif defined(TTS_CPP_USE_CBLAS)
    static bool configured = false;
    if (configured) return;
    configured = true;
    set_env_if_unset("OPENBLAS_NUM_THREADS", "1");
    set_env_if_unset("MKL_NUM_THREADS", "1");
    set_env_if_unset("BLIS_NUM_THREADS", "1");
#endif
}

void print_supertonic_setup_hint() {
    fprintf(stderr,
            "Supertonic GGUFs are generated locally and intentionally ignored by git.\n"
            "Create the multilingual Supertonic 2 GGUF with:\n"
            "  bash scripts/setup-supertonic2.sh\n"
            "or create the English-only Supertonic GGUF with:\n"
            "  bash scripts/setup-supertonic2.sh --arch supertonic\n");
}

uint64_t next_supertonic_generation_id() {
    static std::atomic<uint64_t> next_id{1};
    return next_id.fetch_add(1, std::memory_order_relaxed);
}

// Process-wide alive-set keyed on generation_id.  See
// supertonic_internal.h for the rationale; contract is local to the
// register_supertonic_alive / unregister_supertonic_alive /
// is_supertonic_alive triple defined further down at the
// detail-namespace scope (so the symbols match the header
// declarations and aren't accidentally hidden in this TU's anon
// namespace).
inline std::mutex & supertonic_alive_mu() {
    static std::mutex m;
    return m;
}
inline std::unordered_set<uint64_t> & supertonic_alive_ids() {
    static std::unordered_set<uint64_t> s;
    return s;
}

} // namespace

void register_supertonic_alive(uint64_t generation_id) {
    std::lock_guard<std::mutex> lk(supertonic_alive_mu());
    supertonic_alive_ids().insert(generation_id);
}

void unregister_supertonic_alive(uint64_t generation_id) {
    std::lock_guard<std::mutex> lk(supertonic_alive_mu());
    supertonic_alive_ids().erase(generation_id);
}

bool is_supertonic_alive(uint64_t generation_id) {
    if (generation_id == 0) return false;
    std::lock_guard<std::mutex> lk(supertonic_alive_mu());
    return supertonic_alive_ids().find(generation_id) != supertonic_alive_ids().end();
}

// QVAC-18605 — public forwarder for the F16-K/V flash-attn probe.
// Lets engine.cpp / supertonic_bench.cpp gate the auto-policy on
// the resolved backend's actual capability instead of the
// historical "any non-CPU backend" heuristic — saves a graph-build
// crash on backends that ship `flash_attn_ext` but reject the
// F16 K/V variant for the Supertonic shape.  See the inline probe
// `backend_supports_f16_kv_flash_attn_uncached` in this TU for
// the rationale.  Routes through `cached_backend_capabilities`
// (process-wide cache keyed by `ggml_backend_t`) so engine + bench
// + load trio doesn't re-run the probe three times for the same
// backend.
bool supertonic_backend_supports_f16_kv_flash_attn(ggml_backend_t backend) {
    return cached_backend_capabilities(backend).f16_kv_flash_attn;
}

// QVAC-18605 follow-up — public forwarder for the F16-weight
// `mul_mat` probe.  Symmetric to the F16-K/V probe above; gates
// the `use_f16_weights` auto-policy in engine.cpp + bench so a
// backend that ships F16 storage but rejects F16 mul_mat for the
// hot vector-estimator attention shape doesn't crash at first
// synth call.  Cached.
bool supertonic_backend_supports_f16_mul_mat(ggml_backend_t backend) {
    return cached_backend_capabilities(backend).f16_mul_mat;
}

// QVAC-18605 follow-up — public forwarder for the Q8_0 K/V
// flash-attn probe.  Forward-compat — primes the capability
// cache for a future `--kv-attn-type q8_0` opt-in (cuts K/V
// upload bandwidth ~2× on memory-bandwidth-bound mobile GPUs)
// without forcing the live dispatch through Q8_0 today.  See
// `backend_supports_q8_0_kv_flash_attn_uncached` for the
// rationale + the deferred-work entry in PROGRESS_SUPERTONIC.md.
bool supertonic_backend_supports_q8_0_kv_flash_attn(ggml_backend_t backend) {
    return cached_backend_capabilities(backend).q8_0_kv_flash_attn;
}

// QVAC-18605 round 3 — public forwarder for the BF16 K/V flash-
// attn probe.  Forward-compat — primes the capability cache for
// a future `--kv-attn-type bf16` opt-in (BF16's wider exponent
// range avoids the F16 underflow on small attention scores
// without paying a 2× bandwidth cost).  Mostly useful on Vulkan
// devices that advertise `cooperative_matrix2` (NVIDIA Ampere+,
// AMD RDNA3+).  See `backend_supports_bf16_kv_flash_attn_uncached`
// for the rationale + the deferred-work entry in
// PROGRESS_SUPERTONIC.md.
bool supertonic_backend_supports_bf16_kv_flash_attn(ggml_backend_t backend) {
    return cached_backend_capabilities(backend).bf16_kv_flash_attn;
}

// QVAC-18605 round 3 — public forwarder for the pinned-host-
// buffer-type probe.  Symmetric to the BF16 / Q8_0 K/V
// forwarders above; primes the capability cache with whether
// `ggml_backend_vk_host_buffer_type()` is callable on this
// backend so a future per-engine input-scratchpad refactor can
// gate the host-pinned upload path on the cached answer
// (avoids re-querying the Vulkan backend per synth step).
bool supertonic_backend_supports_pinned_host_buffer(ggml_backend_t backend) {
    return cached_backend_capabilities(backend).pinned_host_buffer;
}

// QVAC-18605 round 3 — multi-device Vulkan auto-pick policy.
//
// Pure logic — no Vulkan symbols touched here.  The Vulkan-only
// wrapper (`init_supertonic_backend`'s `#ifdef GGML_USE_VULKAN`
// branch) calls `ggml_backend_vk_get_device_memory()` per device
// to build the `free_vram_per_device` list, then dispatches into
// this helper.  Splitting the policy from the plumbing means the
// behaviour matrix is testable on CPU with synthetic inputs (see
// test_supertonic_vulkan_device_select.cpp).
//
// See the docstring on the declaration in supertonic_internal.h
// for the behaviour matrix.
int resolve_vulkan_device_index(int requested,
                                const std::vector<size_t> & free_vram_per_device) {
    const int dev_count = (int) free_vram_per_device.size();
    if (dev_count <= 0) {
        throw std::runtime_error(
            "supertonic: cannot resolve --vulkan-device against an empty "
            "device list (no Vulkan adapter visible)");
    }
    // Reserved-future negative value — fail loud instead of
    // silently treating as 0 (would mask a CLI typo).
    if (requested < -1) {
        throw std::runtime_error(
            "supertonic: --vulkan-device " + std::to_string(requested) +
            " is reserved (only -1 means auto-pick)");
    }
    // Auto-pick: argmax(free VRAM); ties → lower index.  std::max_element
    // returns the first iterator that compares equal under `<` so the
    // tie-breaking rule is implicit in the std::less<> default.
    if (requested == -1) {
        const auto it = std::max_element(free_vram_per_device.begin(),
                                         free_vram_per_device.end());
        return (int) std::distance(free_vram_per_device.begin(), it);
    }
    // Explicit index — range-check.
    if (requested >= dev_count) {
        throw std::runtime_error(
            "supertonic: --vulkan-device " + std::to_string(requested) +
            " out of range (visible adapters: " +
            std::to_string(dev_count) + ")");
    }
    return requested;
}

// Test seam — drops every cached entry so the regression test in
// `test_supertonic_capability_cache.cpp` can verify the cache is
// hit on the second call (the cold-cache call bumps the probe
// counter; subsequent calls don't until the cache is cleared).
// Not part of the supported public API; the symbol is exported
// only for the in-process test harness and not declared in the
// `supertonic_internal.h` header for external consumers.
void supertonic_clear_capability_cache() {
    std::lock_guard<std::mutex> lk(capability_cache_mu());
    capability_cache().clear();
}

// Test seam — exposes the cold-cache probe call counter so the
// regression test can assert the cache short-circuits the
// uncached path on a hit.  Returns the counter's *current* value,
// which the caller compares before / after `cached_backend_*`
// calls to verify zero increments on a hot cache.
uint64_t supertonic_capability_probe_call_count() {
    return capability_probe_call_counter().load(std::memory_order_relaxed);
}

// QVAC-18605 round 7 — Vulkan env-var passthrough.
//
// ALL-OR-NOTHING: validate every key starts with `GGML_VK_`
// BEFORE touching the environment.  An operator-config typo like
// `GMML_VK_PREFER_HOST_MEMORY` throws cleanly without leaving the
// env in a half-applied state where the good entries took effect
// but the bad one didn't.  Empty map is a no-op (regression-
// guarded by `test_empty_map_is_noop`).
//
// `set_env_if_unset` semantics: an operator-set env var (already
// present in the environment when this is called) WINS over the
// EngineOptions override.  Lets a debugging operator force-disable
// a setting from the shell without recompiling, while still
// letting the production EngineOptions configuration set the same
// knob in the absence of a shell override.
void apply_vulkan_env_overrides(const std::map<std::string, std::string> & overrides) {
    if (overrides.empty()) return;
    std::string bad;
    if (find_invalid_vulkan_env_key(overrides, bad)) {
        throw std::runtime_error(
            "supertonic: invalid Vulkan env-var override key '" + bad +
            "' — keys must start with 'GGML_VK_' (operator-config typo guard)");
    }
    for (const auto & kv : overrides) {
        set_env_if_unset(kv.first.c_str(), kv.second.c_str());
    }
}

// QVAC-18605 round 7 — voice ttl/dp host cache.
//
// Implementation matches the contract documented on the struct
// declaration in supertonic_internal.h.  Inlines the
// `read_tensor_f32` body (defined in supertonic_engine.cpp, not
// linkable from here) — three lines, zero abstraction cost.
const voice_host_cache::entry &
voice_host_cache::get_or_load(const std::string & voice_name,
                              ggml_tensor * ttl_tensor,
                              ggml_tensor * dp_tensor) {
    auto it = by_name_.find(voice_name);
    if (it != by_name_.end()) {
        // Cache HIT: return the existing entry without touching
        // the GGML tensors.  Caller may legally pass nullptr for
        // ttl/dp on a hit (see test_second_load_hits_cache).
        return it->second;
    }
    if (!ttl_tensor || !dp_tensor) {
        throw std::runtime_error(
            "voice_host_cache: cache miss for voice '" + voice_name +
            "' but ttl/dp tensor is null (Engine::Impl bug — voices.find() should "
            "have validated the voice before this call)");
    }
    entry e;
    e.ttl.resize((size_t) ggml_nelements(ttl_tensor));
    ggml_backend_tensor_get(ttl_tensor, e.ttl.data(), 0, ggml_nbytes(ttl_tensor));
    e.dp.resize((size_t) ggml_nelements(dp_tensor));
    ggml_backend_tensor_get(dp_tensor, e.dp.data(), 0, ggml_nbytes(dp_tensor));
    auto inserted = by_name_.emplace(voice_name, std::move(e));
    return inserted.first->second;
}

void voice_host_cache::clear() {
    by_name_.clear();
}

size_t voice_host_cache::size() const {
    return by_name_.size();
}

// Phase 2A — hot-weight predicate.
//
// Returns true for source names that should be materialised as
// F16 on a non-CPU backend when `model.use_f16_weights` is set.
// See the docstring on `should_materialise_f16_weight` in
// supertonic_internal.h for the full roster + test references.
//
// Implementation rules:
//   - String matching uses explicit suffix / contains checks; no
//     regex (the predicate runs once per GGUF tensor at load time,
//     not on the hot path, but we still want it cheap + audit-
//     friendly).
//   - Pre-transposed `__T` companions are excluded (the original
//     gets materialised; the companion lives separately).
//   - Bias / norm-weight / γ tensors are excluded by suffix.
//   - Embedding tables and small fixed-shape per-channel vectors
//     are excluded by name fragment.
bool should_materialise_f16_weight(const std::string & source_name) {
    if (source_name.empty()) return false;

    auto ends_with = [&](const std::string & suffix) {
        return source_name.size() >= suffix.size() &&
               std::equal(suffix.rbegin(), suffix.rend(), source_name.rbegin());
    };
    auto contains = [&](const std::string & frag) {
        return source_name.find(frag) != std::string::npos;
    };

    // Bias / scale / shift / γ — always cold.  Catches both
    // `*.bias` and bias-like `linear.bias` substrings the audit
    // explicitly negative-tested against.
    if (ends_with(".bias"))                  return false;
    if (contains(".linear.bias"))            return false;
    if (contains(".norm.norm.weight"))       return false;
    if (contains(".norm.norm.bias"))         return false;
    if (ends_with(".gamma"))                 return false;
    if (contains(".char_embedder.weight"))   return false;
    if (contains(".emb_rel_k"))              return false;
    if (contains(".emb_rel_v"))              return false;
    if (contains("normalizer.scale"))        return false;
    if (contains("PRelu_"))                  return false;
    if (contains(".dwconv."))                return false;
    if (contains(".attn.theta"))             return false;
    // Pre-transposed companions (F6) are stored separately; the
    // original goes through this predicate normally.  The `__T`
    // suffix tags them.
    if (ends_with("__T"))                    return false;
    // Negative trap (test_supertonic_f16_weights.cpp covers this):
    // a bias-like suffix could otherwise sneak through if it has
    // a digit suffix that happens to match `_NNNN` below.
    if (contains("MatMul_") && ends_with("_bias")) return false;

    // Positive list:
    //
    //  - vector_estimator attention matmuls: `onnx::MatMul_NNNN`
    //    where NNNN is the per-group / per-attention-site ID.
    //    Cover-all by the `onnx::MatMul_` substring inside the
    //    `vector_estimator:` namespace.
    //  - vector_estimator convnext pwconv1/2: anything ending in
    //    `.pwconv1.weight` or `.pwconv2.weight`.
    //  - vocoder convnext pwconv1/2 + head linear: same suffix
    //    convention.
    //  - text-encoder linears: `text_encoder:onnx::MatMul_` and
    //    the FFN `conv_1.weight` / `conv_2.weight`.
    const bool ve  = source_name.rfind("vector_estimator:", 0) == 0;
    const bool voc = source_name.rfind("vocoder:", 0) == 0;
    const bool tex = source_name.rfind("text_encoder:", 0) == 0;
    if (!ve && !voc && !tex) return false;

    if (contains("onnx::MatMul_")) {
        // Reject `onnx::MatMul_` followed by an empty / non-digit
        // tail (audit test edge case: `"vector_estimator:onnx::MatMul_"`).
        const size_t pos = source_name.find("onnx::MatMul_");
        if (pos != std::string::npos) {
            const std::string tail = source_name.substr(pos + 13);
            if (tail.empty()) return false;
            // First char of tail must be a digit; otherwise it's
            // a name like `MatMul_bias_3101` which is a manufactured
            // negative.  See predicate-negatives test.
            if (!(tail[0] >= '0' && tail[0] <= '9')) return false;
        }
        return true;
    }
    if (ends_with(".pwconv1.weight")) return true;
    if (ends_with(".pwconv2.weight")) return true;
    if (ends_with(".head.layer1.net.weight")) return true;
    if (ends_with(".head.layer2.weight"))     return true;
    if (contains(".conv_1.weight")) return true;
    if (contains(".conv_2.weight")) return true;

    return false;
}

// QVAC-18605 round 6 — 2-arg overload.
//
// Two-stage decision:
//
//   1. If any non-empty entry in `extra_deny_substrings` is a
//      substring of `source_name`, return `false` immediately.
//      Operator-supplied deny patterns short-circuit the curated
//      allow-list (they're meant to FORCE F32 even for tensors
//      the curated path would have promoted).
//
//   2. Otherwise, forward to the 1-arg version (curated allow-
//      list).
//
// Empty deny-list → behaviour identical to the 1-arg version
// (zero behaviour change for every existing call site that
// passes the default empty list).
//
// Empty strings inside the deny-list are SKIPPED on purpose:
// substring `""` would otherwise match every name and silently
// disable F16 weights for the entire model, which is almost
// certainly an operator typo (e.g. trailing comma in a config
// file producing an empty entry).  Surfacing the typo via a
// loud warning would be nicer, but `should_materialise_f16_weight`
// is a pure predicate with no logging hook; the defensive skip
// keeps the predicate honest while a higher-layer config
// validator can warn separately if desired.
bool should_materialise_f16_weight(const std::string & source_name,
                                   const std::vector<std::string> & extra_deny_substrings) {
    if (source_name.empty()) return false;
    for (const std::string & pattern : extra_deny_substrings) {
        if (pattern.empty()) continue;  // defensive skip
        if (source_name.find(pattern) != std::string::npos) {
            return false;
        }
    }
    return should_materialise_f16_weight(source_name);
}

// Thread-local dispatch flags consulted by the GGML graph builders to
// pick between the CBLAS-backed `ggml_custom_4d` fast paths (CPU only)
// and the portable pure-GGML fallbacks (any backend).  See the
// supertonic_op_dispatch_scope comment in supertonic_internal.h.
//
// QVAC-18605 — `g_supertonic_use_native_leaky_relu` carries the
// resolved-backend's `LEAKY_RELU` capability into the
// `leaky_relu_portable_ggml` helper.  Defaults to `true` so the
// historical CPU-only path keeps using the fused builtin even when no
// scope is active (matches `g_supertonic_use_cpu_custom_ops`'s default
// rationale).
namespace {
thread_local bool g_supertonic_use_cpu_custom_ops    = true;
thread_local bool g_supertonic_use_f16_attn          = false;
thread_local bool g_supertonic_use_native_leaky_relu = true;
// QVAC-18605 round 4 — current K/V flash-attn dispatch dtype.
// Defaults to f32 so a graph builder called outside any
// `supertonic_op_dispatch_scope` doesn't accidentally take the
// F16/BF16/Q8_0 path (matches the model's default value).
thread_local kv_attn_dtype g_supertonic_kv_attn_type  = kv_attn_dtype::f32;
}

bool supertonic_use_cpu_custom_ops() {
    return g_supertonic_use_cpu_custom_ops;
}

bool supertonic_use_f16_attn() {
    return g_supertonic_use_f16_attn;
}

bool supertonic_use_native_leaky_relu() {
    return g_supertonic_use_native_leaky_relu;
}

kv_attn_dtype supertonic_kv_attn_type() {
    return g_supertonic_kv_attn_type;
}

supertonic_op_dispatch_scope::supertonic_op_dispatch_scope(const supertonic_model & model)
    : prev_use_cpu_custom_ops(g_supertonic_use_cpu_custom_ops),
      prev_use_f16_attn(g_supertonic_use_f16_attn),
      prev_use_native_leaky_relu(g_supertonic_use_native_leaky_relu),
      prev_kv_attn_type(g_supertonic_kv_attn_type) {
    g_supertonic_use_cpu_custom_ops    = model.backend_is_cpu;
    g_supertonic_use_f16_attn          = model.use_f16_attn;
    g_supertonic_use_native_leaky_relu = model.use_native_leaky_relu;
    g_supertonic_kv_attn_type          = model.kv_attn_type;
}

supertonic_op_dispatch_scope::~supertonic_op_dispatch_scope() {
    g_supertonic_use_cpu_custom_ops    = prev_use_cpu_custom_ops;
    g_supertonic_use_f16_attn          = prev_use_f16_attn;
    g_supertonic_use_native_leaky_relu = prev_use_native_leaky_relu;
    g_supertonic_kv_attn_type          = prev_kv_attn_type;
}

// QVAC-18605 round 4 — pure-logic resolver for the multi-dtype
// K/V dispatch policy.  Implementation matches the behaviour
// matrix documented on the declaration in supertonic_internal.h.
//
// Out-of-range inputs throw to surface CLI typos loudly; probe-
// rejected explicit requests fall back to f32 silently (same
// "advisory probes" pattern as the round-1 use_f16_attn auto-
// policy fallback).
kv_attn_dtype resolve_kv_attn_type(int requested,
                                   bool legacy_use_f16_attn,
                                   bool backend_supports_f16,
                                   bool backend_supports_bf16,
                                   bool backend_supports_q8_0) {
    if (requested < -1 || requested > 3) {
        throw std::runtime_error(
            "supertonic: --kv-attn-type " + std::to_string(requested) +
            " out of range (valid: -1=auto, 0=f32, 1=f16, 2=bf16, 3=q8_0)");
    }
    switch (requested) {
        case -1:  // auto
            if (legacy_use_f16_attn && backend_supports_f16) return kv_attn_dtype::f16;
            return kv_attn_dtype::f32;
        case 0:   // f32 forced
            return kv_attn_dtype::f32;
        case 1:   // f16 forced (probe-gated fallback)
            return backend_supports_f16  ? kv_attn_dtype::f16  : kv_attn_dtype::f32;
        case 2:   // bf16 forced (probe-gated fallback)
            return backend_supports_bf16 ? kv_attn_dtype::bf16 : kv_attn_dtype::f32;
        case 3:   // q8_0 forced (probe-gated fallback)
            return backend_supports_q8_0 ? kv_attn_dtype::q8_0 : kv_attn_dtype::f32;
        default:
            // Unreachable — the range check above covers every
            // valid request.  Defensive throw in case the switch
            // is extended without updating the range check.
            throw std::runtime_error("supertonic: resolve_kv_attn_type unreachable");
    }
}

// ---------------------------------------------------------------------
// Phase 2D — `SUPERTONIC_PROFILE_CSV` machine-readable timing emitter.
//
// Implementation lives here (in `supertonic_gguf.cpp`) rather than a
// dedicated TU because:
//   - the supertonic library already pulls this file in unconditionally
//     (load_supertonic_gguf is the public entry point).
//   - the file-local state (FILE *, mutex, env-probe latch) doesn't
//     need to be shared across TUs.
//
// Storage model:
//   - One `FILE *` opened at "first record after path set" time.
//   - A mutex guards record / flush / set_path so the emitter is
//     safe to call from any thread (the rest of the engine is
//     single-threaded per model, but tests may spawn helpers).
//   - The env var `SUPERTONIC_PROFILE_CSV` is probed lazily on the
//     first `record` / `enabled` call after process start; tests
//     override via `set_path(PATH)` which bypasses the env probe.
//
// Schema (matches the contract in
// `test_supertonic_profile_csv.cpp`):
//
//   stage,island,step,wall_ms,unix_us
//
// The header row is written once, lazily, the first time we open
// a new file that's empty.  Re-opening the same path appends, so
// long-running bench harnesses can record many synths without
// stomping their header / data.
namespace {

struct profile_csv_state {
    std::mutex   mu;
    std::FILE *  fp = nullptr;
    std::string  path;
    bool         env_checked = false;
};

profile_csv_state & profile_csv() {
    static profile_csv_state s;
    return s;
}

void profile_csv_close_locked(profile_csv_state & s) {
    if (s.fp) {
        std::fclose(s.fp);
        s.fp = nullptr;
    }
    s.path.clear();
}

void profile_csv_open_locked(profile_csv_state & s, const std::string & path) {
    // Append mode so multiple sessions can share one CSV.
    // We only write the header when the file is empty (fresh).
    bool need_header = false;
    {
        std::FILE * probe = std::fopen(path.c_str(), "rb");
        if (probe) {
            std::fseek(probe, 0, SEEK_END);
            const long sz = std::ftell(probe);
            need_header = (sz == 0);
            std::fclose(probe);
        } else {
            need_header = true;
        }
    }
    s.fp = std::fopen(path.c_str(), "ab");
    if (!s.fp) return; // open failure → emitter stays disabled
    s.path = path;
    if (need_header) {
        std::fprintf(s.fp, "stage,island,step,wall_ms,unix_us\n");
        std::fflush(s.fp);
    }
}

void profile_csv_atexit_flush() {
    // Best-effort flush + close on normal process exit; if the
    // bench harness segfaults we lose buffered rows but that's
    // the same trade-off any FILE *-based logger makes.
    profile_csv_state & s = profile_csv();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.fp) {
        std::fflush(s.fp);
        std::fclose(s.fp);
        s.fp = nullptr;
    }
}

void profile_csv_probe_env_locked(profile_csv_state & s) {
    if (s.env_checked) return;
    s.env_checked = true;
    const char * env = std::getenv("SUPERTONIC_PROFILE_CSV");
    if (env && *env) {
        profile_csv_open_locked(s, env);
        // Register an atexit hook the first time we open via the
        // env var.  Tests that flip the path via `_set_path` get
        // the flush via their explicit teardown call instead;
        // they don't need an atexit because the unit harness
        // explicitly cleans up.
        std::atexit(profile_csv_atexit_flush);
    }
}

} // namespace

bool supertonic_profile_csv_enabled() {
    profile_csv_state & s = profile_csv();
    std::lock_guard<std::mutex> lk(s.mu);
    profile_csv_probe_env_locked(s);
    return s.fp != nullptr;
}

void supertonic_profile_csv_record(const char * stage, const char * island,
                                   int step, double wall_ms) {
    profile_csv_state & s = profile_csv();
    std::lock_guard<std::mutex> lk(s.mu);
    profile_csv_probe_env_locked(s);
    if (!s.fp) return;
    // Wall clock in microseconds-since-epoch so the CSV is sortable
    // across separate bench harness invocations.  `steady_clock`
    // would be cheaper but isn't comparable across processes; the
    // CSV is post-analysed not perf-critical.
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const long long unix_us =
        std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    std::fprintf(s.fp, "%s,%s,%d,%.3f,%lld\n",
                 stage ? stage : "",
                 island ? island : "",
                 step,
                 wall_ms,
                 unix_us);
}

void supertonic_profile_csv_flush() {
    profile_csv_state & s = profile_csv();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.fp) std::fflush(s.fp);
}

void supertonic_profile_csv_set_path(const char * path) {
    profile_csv_state & s = profile_csv();
    std::lock_guard<std::mutex> lk(s.mu);
    profile_csv_close_locked(s);
    // Latch the env probe even when the caller passes nullptr so
    // that a subsequent enabled()/record() call doesn't accidentally
    // re-pick-up the env var after the test asked us to disable.
    s.env_checked = true;
    if (path && *path) {
        profile_csv_open_locked(s, path);
    }
}

ggml_tensor * require_tensor(const supertonic_model & model, const std::string & name) {
    ggml_tensor * t = get_tensor_or_null(model, name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

ggml_tensor * require_source_tensor(const supertonic_model & model, const std::string & source_name) {
    auto it = model.source_tensors.find(source_name);
    if (it == model.source_tensors.end() || !it->second) {
        throw std::runtime_error("missing source tensor: " + source_name);
    }
    return it->second;
}

ggml_tensor * try_source_tensor(const supertonic_model & model, const std::string & source_name) {
    auto it = model.source_tensors.find(source_name);
    if (it == model.source_tensors.end()) return nullptr;
    return it->second;
}

ggml_tensor * try_pretransposed_weight(const supertonic_model & model, const ggml_tensor * w) {
    if (!w) return nullptr;
    auto it = model.pretransposed_weights.find(w);
    if (it == model.pretransposed_weights.end()) return nullptr;
    return it->second;
}

void supertonic_set_n_threads(supertonic_model & model, int n_threads) {
    configure_supertonic_blas_threads_once();
    if (n_threads <= 0) {
        const int hw = (int) std::thread::hardware_concurrency();
        n_threads = std::min(std::max(1, hw), 4);
    }
    model.n_threads = std::max(1, n_threads);
}

void supertonic_graph_compute(const supertonic_model & model, ggml_cgraph * graph) {
    if (ggml_backend_is_cpu(model.backend) && model.n_threads > 0) {
        ggml_backend_cpu_set_n_threads(model.backend, model.n_threads);
    }
    static const bool count_dispatches = std::getenv("SUPERTONIC_COUNT_DISPATCHES") != nullptr;
    static const bool dump_op_histogram = std::getenv("SUPERTONIC_DUMP_OP_HISTOGRAM") != nullptr;
    if (dump_op_histogram) {
        static thread_local int hist_call = 0;
        ++hist_call;
        const int n = ggml_graph_n_nodes(graph);
        std::map<std::string, int> hist;
        for (int i = 0; i < n; ++i) {
            ggml_tensor * t = ggml_graph_node(graph, i);
            hist[ggml_op_name(t->op)] += 1;
        }
        fprintf(stderr, "=== supertonic_graph_compute #%d op histogram (n_nodes=%d) ===\n", hist_call, n);
        std::vector<std::pair<int, std::string>> sorted;
        for (auto & kv : hist) sorted.emplace_back(kv.second, kv.first);
        std::sort(sorted.rbegin(), sorted.rend());
        for (auto & p : sorted) {
            fprintf(stderr, "  %4d  %s\n", p.first, p.second.c_str());
        }
    }
    if (count_dispatches) {
        static thread_local int n_calls = 0;
        static thread_local double total_us = 0.0;
        ++n_calls;
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(model.backend, graph);
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        total_us += us;
        fprintf(stderr, "supertonic_graph_compute #%d nodes=%d  wall=%.1fus  cumul=%.2fms\n",
                n_calls, ggml_graph_n_nodes(graph), us, total_us / 1000.0);
        return;
    }
    ggml_backend_graph_compute(model.backend, graph);
}

static void bind_vocoder_weights(supertonic_model & model) {
    auto & v = model.vocoder;
    v.normalizer_scale = require_source_tensor(model, "vocoder:tts.ttl.normalizer.scale");
    v.latent_mean = require_source_tensor(model, "vocoder:tts.ae.latent_mean");
    v.latent_std = require_source_tensor(model, "vocoder:tts.ae.latent_std");
    v.embed_w = require_source_tensor(model, "vocoder:onnx::Conv_1440");
    v.embed_b = require_source_tensor(model, "vocoder:onnx::Conv_1441");
    for (int i = 0; i < 10; ++i) {
        const std::string p = "vocoder:tts.ae.decoder.convnext." + std::to_string(i);
        auto & c = v.convnext[(size_t) i];
        c.dw_w = require_source_tensor(model, p + ".dwconv.net.weight");
        c.dw_b = require_source_tensor(model, p + ".dwconv.net.bias");
        c.norm_g = require_source_tensor(model, p + ".norm.norm.weight");
        c.norm_b = require_source_tensor(model, p + ".norm.norm.bias");
        c.pw1_w = require_source_tensor(model, p + ".pwconv1.weight");
        c.pw1_b = require_source_tensor(model, p + ".pwconv1.bias");
        c.pw2_w = require_source_tensor(model, p + ".pwconv2.weight");
        c.pw2_b = require_source_tensor(model, p + ".pwconv2.bias");
        c.gamma = require_source_tensor(model, p + ".gamma");
    }
    v.final_norm_g = require_source_tensor(model, "vocoder:tts.ae.decoder.final_norm.norm.weight");
    v.final_norm_b = require_source_tensor(model, "vocoder:tts.ae.decoder.final_norm.norm.bias");
    v.final_norm_running_mean = require_source_tensor(model, "vocoder:tts.ae.decoder.final_norm.norm.running_mean");
    v.final_norm_running_var = require_source_tensor(model, "vocoder:tts.ae.decoder.final_norm.norm.running_var");
    v.head1_w = require_source_tensor(model, "vocoder:tts.ae.decoder.head.layer1.net.weight");
    v.head1_b = require_source_tensor(model, "vocoder:tts.ae.decoder.head.layer1.net.bias");
    v.head_prelu = require_source_tensor(model, "vocoder:onnx::PRelu_1505");
    v.head2_w = require_source_tensor(model, "vocoder:tts.ae.decoder.head.layer2.weight");
}

bool load_supertonic_gguf(const std::string & path,
                          supertonic_model & model,
                          int n_gpu_layers,
                          bool verbose,
                          int f16_weights,
                          supertonic_precision precision,
                          int vulkan_device,
                          const std::vector<std::string> & f16_weights_deny_list) {
    model.generation_id = next_supertonic_generation_id();
    model.precision_id = static_cast<int>(precision);
    // The load path supports F32 / F16 / Q8_0 destination types.
    // - F32: fully wired.
    // - Q8_0: storage on Metal only for `:onnx::MatMul_*` weights (the
    //   optimised `kernel_mul_mm_q8_0_f32` dispatches via the swapped-
    //   args `dense_matmul_time_wt_pretransposed_ggml` helper).  Other
    //   tensors expand to f32.  On CPU everything expands to f32 so
    //   cblas/AMX keeps the lead.
    // - F16: same asymmetric scheme as Q8_0 — `:onnx::MatMul_*` weights
    //   stay f16 on Metal (dispatches `kernel_mul_mm_f16_f32`), other
    //   GGUF-f16 tensors (relpos embeddings, per-channel scales used in
    //   plain `ggml_mul`) expand to f32 so they don't trip `ggml_metal_op_bin`'s
    //   f32-only assertion.  Pretranspose pass covers f16 alongside f32/q8_0.
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gp);
    if (!gguf_ctx) {
        fprintf(stderr, "load_supertonic_gguf: failed to open '%s'\n", path.c_str());
        print_supertonic_setup_hint();
        return false;
    }

    try {
        std::string arch = get_string(gguf_ctx, "supertonic.arch");
        if (arch != "supertonic2" && arch != "supertonic") {
            throw std::runtime_error("unexpected supertonic.arch: " + arch);
        }

        model.hparams.arch = arch;
        model.hparams.ftype = get_string(gguf_ctx, "supertonic.ftype", "f32");
        model.hparams.sample_rate = (int) get_u32(gguf_ctx, "supertonic.sample_rate");
        model.hparams.base_chunk_size = (int) get_u32(gguf_ctx, "supertonic.base_chunk_size");
        model.hparams.ttl_chunk_compress_factor =
            (int) get_u32(gguf_ctx, "supertonic.ttl_chunk_compress_factor");
        model.hparams.latent_dim = (int) get_u32(gguf_ctx, "supertonic.latent_dim");
        model.hparams.latent_channels = (int) get_u32(gguf_ctx, "supertonic.latent_channels");
        model.hparams.default_steps = (int) get_u32(gguf_ctx, "supertonic.default_steps");
        model.hparams.default_speed = get_f32(gguf_ctx, "supertonic.default_speed");
        model.hparams.language_wrap_mode = get_string(gguf_ctx, "supertonic.language_wrap_mode");
        if (model.hparams.language_wrap_mode.empty()) {
            bool language_wrap = get_bool_u32(gguf_ctx, "supertonic.language_wrap", arch != "supertonic");
            model.hparams.language_wrap_mode = language_wrap ? (arch == "supertonic2" ? "open_close" : "prefix") : "none";
        }
        model.hparams.default_voice = get_string(gguf_ctx, "supertonic.default_voice", "F1");
        model.languages = get_string_array(gguf_ctx, "supertonic.languages");
        model.tts_json = get_string(gguf_ctx, "supertonic.tts_json");

        model.backend = init_supertonic_backend(n_gpu_layers, verbose, vulkan_device);
        // The graph builders below dispatch between CBLAS-backed
        // `ggml_custom_4d` fast paths (CPU only) and pure-GGML fallbacks
        // (any backend) based on this flag.  Stable for the model's
        // lifetime; see the supertonic_op_dispatch_scope comment in
        // supertonic_internal.h for the threading contract.
        model.backend_is_cpu = ggml_backend_is_cpu(model.backend);
        // QVAC-18605 — Vulkan-specific dispatch capture.
        //
        // `backend_is_vk` is informational (the bench / engine show it
        // in the human-readable backend description), but it also
        // documents WHICH non-CPU backend the model resolved to —
        // useful when triaging "why is leaky_relu slow on this run?"
        // against the audit's expected fast-path matrix.
        model.backend_is_vk = backend_is_vulkan(model.backend);
        // Probe the backend's `LEAKY_RELU` capability so the
        // `leaky_relu_portable_ggml` helper can route to the fused
        // builtin on backends that have it (Vulkan / Metal / CUDA /
        // CPU; OpenCL only with chatterbox patch) and to the
        // RELU+SCALE+ADD decomposition otherwise.  Probe runs once
        // per backend (memoised by `cached_backend_capabilities`)
        // — zero hot-path cost.
        model.use_native_leaky_relu = cached_backend_capabilities(model.backend).native_leaky_relu;
        if (verbose) {
            fprintf(stderr, "supertonic: backend_is_cpu=%s backend_is_vk=%s use_native_leaky_relu=%s\n",
                    model.backend_is_cpu ? "true" : "false",
                    model.backend_is_vk ? "true" : "false",
                    model.use_native_leaky_relu ? "true" : "false");
        }

        // Phase 2A — auto/force policy for F16 weight materialization.
        // Auto-enable on non-CPU backends; never auto-enable on CPU
        // (the CBLAS custom-op fast paths require F32 storage).
        //
        // QVAC-18605 follow-up — the auto policy is now backend-
        // capability-gated.  Symmetric to the F16-K/V flash-attn
        // probe: a backend that ships F16 storage but rejects the
        // hot `mul_mat(F16, F32)` shape Supertonic dispatches every
        // step would crash at first synth call when this flipped on
        // blindly.  The probe (`backend_supports_f16_mul_mat_uncached`
        // → `cached_backend_capabilities`) tries the live shape
        // (W=[256, 256] F16, X=[256, 16] F32) at backend resolution
        // time; on a `false` answer the auto policy refuses to
        // materialise F16 weights — slower but correct.  Manual
        // override via `--f16-weights 1` still forces dispatch
        // (useful for debug-shim backends and forward-compat tests).
        if (f16_weights < 0) {
            model.use_f16_weights = !model.backend_is_cpu &&
                                    cached_backend_capabilities(model.backend).f16_mul_mat;
        } else {
            model.use_f16_weights = (f16_weights != 0);
        }
        if (verbose) {
            fprintf(stderr, "supertonic: use_f16_weights=%s\n",
                    model.use_f16_weights ? "true" : "false");
            // Round 6 — log the user-supplied deny-list (if any) so
            // operators can confirm their config got plumbed through.
            // Empty list (the default) is silent — same baseline as
            // the round-3 log output.
            if (model.use_f16_weights && !f16_weights_deny_list.empty()) {
                fprintf(stderr,
                        "supertonic: f16_weights_deny_list (%zu pattern%s):\n",
                        f16_weights_deny_list.size(),
                        f16_weights_deny_list.size() == 1 ? "" : "s");
                for (const auto & p : f16_weights_deny_list) {
                    fprintf(stderr, "  - \"%s\"%s\n", p.c_str(),
                            p.empty() ? " (empty — skipped at predicate time)" : "");
                }
            }
        }

        // Phase 2A pre-step: build a (tensor_name → source_name)
        // lookup BEFORE the alloc loop so we can apply the hot-
        // weight predicate at allocation time (and pick F16 vs F32
        // storage accordingly).  Same metadata arrays as the
        // post-alloc source_tensors map further below; reading them
        // twice is cheap.
        std::unordered_map<std::string, std::string> tensor_to_source_for_alloc;
        if (model.use_f16_weights) {
            int64_t id_tn = gguf_find_key(gguf_ctx, "supertonic.tensor_names");
            int64_t id_sn = gguf_find_key(gguf_ctx, "supertonic.source_names");
            if (id_tn >= 0 && id_sn >= 0) {
                const size_t n_tn = gguf_get_arr_n(gguf_ctx, id_tn);
                const size_t n_sn = gguf_get_arr_n(gguf_ctx, id_sn);
                if (n_tn == n_sn) {
                    for (size_t i = 0; i < n_tn; ++i) {
                        tensor_to_source_for_alloc[gguf_get_arr_str(gguf_ctx, id_tn, i)] =
                            gguf_get_arr_str(gguf_ctx, id_sn, i);
                    }
                }
            }
        }

        const int64_t num_tensors = gguf_get_n_tensors(gguf_ctx);
        // Reserve a small surplus of tensor-overhead slots for the
        // audit-driven pre-baked tensors that load_supertonic_gguf
        // appends to `model.ctx_w` below: F2 vocoder bn_scale_pre +
        // bn_shift_pre, plus F6's pre-transposed companions for the
        // five hot t_proj weights.  A surplus of 16 covers the
        // current roster + headroom for follow-up audit phases.
        constexpr int64_t kPrebakedTensorSurplus = 16;
        ggml_init_params params = {
            /*.mem_size=*/ ggml_tensor_overhead() * (size_t)(num_tensors + kPrebakedTensorSurplus),
            /*.mem_buffer=*/ nullptr,
            /*.no_alloc=*/ true,
        };
        model.ctx_w = ggml_init(params);
        if (!model.ctx_w) throw std::runtime_error("ggml_init failed");

        std::unordered_map<std::string, std::vector<float>>     expanded_f32_tensors;
        // Phase 2A: tensors materialised as F16 land their host-side
        // F16 payload here.  `ggml_fp16_t` is a 16-bit half-float;
        // we use `uint16_t` storage to avoid a public-header dep on
        // ggml's f16 typedef.
        std::unordered_map<std::string, std::vector<uint16_t>>   f16_materialised_tensors;
        // Tensors that need a Metal-specific type conversion (e.g.
        // f32 → q8_0 for `--precision q8_0`) keep their converted
        // bytes here, held alive until the backend upload loop runs.
        std::unordered_map<std::string, std::vector<uint8_t>>    converted_tensors;

        // Ensure the source-alias map is populated even when the
        // Phase 2A `use_f16_weights` path didn't already build it —
        // the precision-driven decision below also needs it to
        // recognise `:onnx::MatMul_` sources for Metal asymmetric load.
        if (tensor_to_source_for_alloc.empty()) {
            int64_t id_tn = gguf_find_key(gguf_ctx, "supertonic.tensor_names");
            int64_t id_sn = gguf_find_key(gguf_ctx, "supertonic.source_names");
            if (id_tn >= 0 && id_sn >= 0) {
                const size_t n_tn = gguf_get_arr_n(gguf_ctx, id_tn);
                const size_t n_sn = gguf_get_arr_n(gguf_ctx, id_sn);
                if (n_tn == n_sn) {
                    for (size_t i = 0; i < n_tn; ++i) {
                        tensor_to_source_for_alloc[gguf_get_arr_str(gguf_ctx, id_tn, i)] =
                            gguf_get_arr_str(gguf_ctx, id_sn, i);
                    }
                }
            }
        }

        // Decide per-tensor destination type:
        //  1. F32 sources on the F16-weights hot-path roster +
        //     `use_f16_weights` on → materialise as F16 (Phase 2A).
        //  2. Else fall through to the precision-driven path:
        //     `target_supertonic_storage_type` returns F32 / F16 / Q8_0
        //     depending on `--precision` and whether the source name is
        //     a `:onnx::MatMul_` weight on a non-CPU backend.
        //  3. Anything else preserves the source type via dup.
        for (int64_t i = 0; i < num_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
            if (!src) throw std::runtime_error(std::string("missing tmp tensor: ") + name);

            // Phase 2A predicate check.  Only fires when
            // `use_f16_weights` was on and the source resolved to
            // a hot-roster name AND its current GGML type is
            // either F32 or one of the expand-to-F32 types
            // (otherwise the source already carries narrower
            // precision than F16 and we don't widen).
            //
            // QVAC-18605 round 6 — the 2-arg overload layers the
            // user-supplied `f16_weights_deny_list` substring
            // patterns on top of the curated allow-list.  Empty
            // deny-list (the default) → identical behaviour to
            // the round-1/2/3 path.  When the deny-list flips a
            // would-be-hot tensor back to F32 we bump
            // `model.f16_weights_excluded_count` so bench output
            // can confirm the user's deny-list took effect.
            //
            // Master's Phase 2A keys the decision off the source
            // name resolved from `tensor_to_source_for_alloc`
            // (falling back to the dst `name` when absent); round
            // 6 narrows that to require the map lookup to succeed
            // so the deny-list operates on a known-stable source
            // identifier.  Net: a tensor that previously went F16
            // via the dst-name fallback now stays at its native
            // precision-path type — the curated allow-list isn't
            // expected to hit on dst names so this is a no-op in
            // practice.
            // Resolve a stable "decision name" up-front.  Used both
            // by the round-6 deny-list check below and by master's
            // precision-driven `target_supertonic_storage_type`
            // dispatch.  Falls back to the dst tensor `name` when
            // the source-map lookup misses (matches master's Phase
            // 2A behaviour pre-rebase).
            auto src_it = tensor_to_source_for_alloc.find(name);
            const std::string decision_name =
                (src_it != tensor_to_source_for_alloc.end())
                    ? src_it->second
                    : std::string(name);

            bool f16_materialise = false;
            if (model.use_f16_weights &&
                src_it != tensor_to_source_for_alloc.end() &&
                (src->type == GGML_TYPE_F32 ||
                 should_expand_supertonic_tensor(src->type))) {
                const bool curated_hot = should_materialise_f16_weight(decision_name);
                const bool denied      = curated_hot &&
                    !should_materialise_f16_weight(decision_name, f16_weights_deny_list);
                if (denied) {
                    ++model.f16_weights_excluded_count;
                } else if (curated_hot) {
                    f16_materialise = true;
                }
            }

            ggml_type dst_type;
            if (f16_materialise) {
                dst_type = GGML_TYPE_F16;
            } else {
                // Precision-driven path (ours): F32 / F16 / Q8_0 per
                // the `--precision` flag.  Returns src->type unchanged
                // for tensors that don't need conversion.
                dst_type = target_supertonic_storage_type(
                    decision_name, src->type, precision,
                    /*backend_is_cpu=*/ ggml_backend_is_cpu(model.backend));
            }

            ggml_tensor * dst = (dst_type == src->type)
                ? ggml_dup_tensor(model.ctx_w, src)
                : ggml_new_tensor(model.ctx_w, dst_type, ggml_n_dims(src), src->ne);
            ggml_set_name(dst, name);
            model.tensors[name] = dst;

            if (f16_materialise) {
                // Phase 2A F16 materialise path.
                std::vector<float> src_f32;
                if (should_expand_supertonic_tensor(src->type)) {
                    src_f32 = expand_supertonic_tensor_to_f32(src);
                } else {
                    const int64_t n = ggml_nelements(src);
                    src_f32.resize((size_t) n);
                    std::memcpy(src_f32.data(), ggml_get_data(src), (size_t) n * sizeof(float));
                }
                std::vector<uint16_t> & f16 = f16_materialised_tensors[name];
                f16.resize(src_f32.size());
                ggml_fp32_to_fp16_row(src_f32.data(),
                                      reinterpret_cast<ggml_fp16_t *>(f16.data()),
                                      (int64_t) src_f32.size());
            } else if (needs_supertonic_tensor_conversion(src->type, dst_type)) {
                // Precision-driven conversion (ours).  Covers f32 → q8_0,
                // q8_0 → f32, f16 → f32 etc.  Buffered here, uploaded later.
                convert_supertonic_tensor_data(src, dst_type, converted_tensors[name]);
            } else if (should_expand_supertonic_tensor(src->type)) {
                // Legacy fallback: f16/q8_0 src with f32 dst that
                // didn't go through the conversion helper above.
                expanded_f32_tensors[name] = expand_supertonic_tensor_to_f32(src);
            }
        }

        // Audit finding F2 — declare the pre-baked vocoder BN
        // tensors BEFORE `ggml_backend_alloc_ctx_tensors` so they
        // get a slot in the same backend buffer as the rest of the
        // model weights.  Data is uploaded after the source-tensor
        // upload loop further down; see the F2 hook after
        // `bind_vocoder_weights`.
        model.vocoder.bn_scale_pre = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, 512);
        ggml_set_name(model.vocoder.bn_scale_pre, "vocoder/bn_scale_pre");
        model.vocoder.bn_shift_pre = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, 512);
        ggml_set_name(model.vocoder.bn_shift_pre, "vocoder/bn_shift_pre");

        // Audit finding F6 — declare the pre-transposed companion
        // tensors for the four t_proj matmul weights.  Each one has
        // shape [512, 64] in the GGUF (matches the Supertonic-2
        // architecture's time-embedding projection); the transposed
        // form is [64, 512], i.e. axes 0/1 swapped.  Data uploaded
        // after `bind_vocoder_weights` in the F6 post-bind hook.
        // The roster matches AUDIT_SUPERTONIC_OPENCL.md F6 + the
        // test in test_supertonic_load_caches.cpp.
        //
        // Phase 2A interaction: the F6 hook only supports F32
        // sources (the host-side transpose loop assumes 4-byte
        // strides).  When F16 weights are on, the same matmul
        // weights have already been materialised as F16, so we
        // skip F6's allocation + upload entirely; call sites in
        // `supertonic_vector_estimator.cpp` fall back to the
        // legacy in-graph `ggml_cont(ggml_transpose(W))` path.
        ggml_tensor * pretrans_t_proj[4] = {nullptr, nullptr, nullptr, nullptr};
        static const char * const kF6PretransNames[4] = {
            "vector_estimator:onnx::MatMul_3095__T",
            "vector_estimator:onnx::MatMul_3140__T",
            "vector_estimator:onnx::MatMul_3185__T",
            "vector_estimator:onnx::MatMul_3230__T",
        };
        const bool f6_active = !model.use_f16_weights;
        if (f6_active) {
            for (int i = 0; i < 4; ++i) {
                pretrans_t_proj[i] = ggml_new_tensor_2d(model.ctx_w, GGML_TYPE_F32, 64, 512);
                ggml_set_name(pretrans_t_proj[i], kF6PretransNames[i]);
            }
        }

        model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
        if (!model.buffer_w) throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");

        for (ggml_tensor * cur = ggml_get_first_tensor(model.ctx_w);
             cur;
             cur = ggml_get_next_tensor(model.ctx_w, cur)) {
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
            if (!src) {
                // Pre-baked tensor (F2 / F6 / future audit phases):
                // declared in model.ctx_w earlier in this function but
                // doesn't have a GGUF source row — data is uploaded by
                // the dedicated post-bind hook further down.  Skip
                // here so we don't deref a null `src`.
                continue;
            }
            // Phase 2A: F16-materialised tensors take precedence over
            // the precision-converted / F32-expanded paths (they may
            // have been promoted from either F32 or F16/Q8_0 sources).
            auto f16_mat = f16_materialised_tensors.find(ggml_get_name(cur));
            if (f16_mat != f16_materialised_tensors.end()) {
                ggml_backend_tensor_set(cur, f16_mat->second.data(), 0,
                                        f16_mat->second.size() * sizeof(uint16_t));
                continue;
            }
            // Precision-driven conversion (`--precision q8_0`/f16 etc.) —
            // bytes are already in dst-type representation.
            auto converted = converted_tensors.find(ggml_get_name(cur));
            if (converted != converted_tensors.end()) {
                ggml_backend_tensor_set(cur, converted->second.data(), 0,
                                        converted->second.size());
            } else if (auto expanded = expanded_f32_tensors.find(ggml_get_name(cur));
                       expanded != expanded_f32_tensors.end()) {
                // Legacy f16/q8_0 → f32 expansion (used when the
                // conversion helper didn't run).
                ggml_backend_tensor_set(cur, expanded->second.data(), 0,
                                        expanded->second.size() * sizeof(float));
            } else {
                ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
            }
        }

        {
            ggml_tensor * unicode = require_tensor(model, "supertonic/unicode_indexer");
            model.unicode_indexer.resize((size_t) ggml_nelements(unicode));
            ggml_backend_tensor_get(unicode, model.unicode_indexer.data(), 0, ggml_nbytes(unicode));
        }

        // Populate the model's source_tensors lookup from the
        // GGUF's `supertonic.tensor_names` / `supertonic.source_names`
        // pair (the `tensor_to_source_for_alloc` map above only carries
        // the same data for the pre-alloc decision; we re-read here so
        // we don't have to widen its scope).
        {
            std::vector<std::string> tensor_names = get_string_array(gguf_ctx, "supertonic.tensor_names");
            std::vector<std::string> source_names = get_string_array(gguf_ctx, "supertonic.source_names");
            if (tensor_names.size() != source_names.size()) {
                throw std::runtime_error("supertonic.tensor_names / source_names length mismatch");
            }
            for (size_t i = 0; i < tensor_names.size(); ++i) {
                ggml_tensor * t = require_tensor(model, tensor_names[i]);
                model.source_tensors[source_names[i]] = t;
            }
        }

        for (const std::string & voice_name : get_string_array(gguf_ctx, "supertonic.voice_names")) {
            supertonic_voice_style voice;
            voice.name = voice_name;
            voice.ttl = require_tensor(model, "supertonic/voices/" + voice_name + "/ttl");
            voice.dp  = require_tensor(model, "supertonic/voices/" + voice_name + "/dp");
            model.voices[voice_name] = voice;
        }

        bind_vocoder_weights(model);

        // Audit finding F1 — cache the vector-estimator RoPE θ
        // tensor on the host once at load time.  All four group
        // attention sites in `supertonic_vector_step_ggml`'s
        // production GGML path read from the same source tensor;
        // caching here avoids 4 × N_STEPS GPU→host downloads per
        // synth on a non-CPU backend.  Tensor is small (64 floats
        // typical), so the host-side copy cost is negligible
        // compared with the sync-point savings.  See
        // AUDIT_SUPERTONIC_OPENCL.md F1 + PLAN Phase 2F.
        //
        // The source tensor is mandatory for any production
        // Supertonic GGUF (all four group attention sites depend
        // on it); fail-fast at load time so the call-site
        // assumption "model.vector_rope_theta.data() is non-null"
        // can stay assertion-free.  Matches the previous behaviour
        // where the same tensor was looked up via
        // `read_f32(model, "...theta")` on the hot path and would
        // throw `runtime_error("missing source tensor: ...")`.
        {
            ggml_tensor * theta_src = require_source_tensor(model,
                "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
            model.vector_rope_theta.resize((size_t) ggml_nelements(theta_src));
            ggml_backend_tensor_get(theta_src,
                                    model.vector_rope_theta.data(),
                                    0, ggml_nbytes(theta_src));
        }

        // Audit finding F2 — compute the vocoder BN scale / shift
        // pre-bake.  Downloads the four final_norm.* tensors that
        // were just uploaded a few lines above (so this is a single
        // round-trip at load time, not per-synth), folds them into
        // the BN-fused form, and uploads to bn_scale_pre /
        // bn_shift_pre which the vocoder graph cache references
        // directly as weights.  Every subsequent synth call skips
        // the 4 reads + CPU compute + 2 uploads that the old path
        // did.  See AUDIT_SUPERTONIC_OPENCL.md F2.
        {
            auto download = [](ggml_tensor * t, std::vector<float> & out) {
                out.resize((size_t) ggml_nelements(t));
                ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
            };
            std::vector<float> gamma, beta, mean, var;
            download(model.vocoder.final_norm_g, gamma);
            download(model.vocoder.final_norm_b, beta);
            download(model.vocoder.final_norm_running_mean, mean);
            download(model.vocoder.final_norm_running_var,  var);
            if (gamma.size() != 512 || beta.size() != 512 ||
                mean.size() != 512  || var.size()  != 512) {
                throw std::runtime_error(
                    "vocoder final_norm.* size mismatch (expected 512 each)");
            }
            std::vector<float> bn_scale_pre(512), bn_shift_pre(512);
            for (int c = 0; c < 512; ++c) {
                bn_scale_pre[c] = gamma[c] / std::sqrt(var[c] + 1e-5f);
                bn_shift_pre[c] = beta[c] - mean[c] * bn_scale_pre[c];
            }
            ggml_backend_tensor_set(model.vocoder.bn_scale_pre,
                                    bn_scale_pre.data(), 0, 512 * sizeof(float));
            ggml_backend_tensor_set(model.vocoder.bn_shift_pre,
                                    bn_shift_pre.data(), 0, 512 * sizeof(float));
        }

        // Audit finding F6 — populate the pre-transposed t_proj
        // companions from the source tensors.  Gated on
        // `f6_active`; see the declaration block above for the
        // Phase 2A interaction note.
        if (f6_active) {
            static const char * const kF6Sources[4] = {
                "vector_estimator:onnx::MatMul_3095",
                "vector_estimator:onnx::MatMul_3140",
                "vector_estimator:onnx::MatMul_3185",
                "vector_estimator:onnx::MatMul_3230",
            };
            for (int i = 0; i < 4; ++i) {
                if (!pretrans_t_proj[i]) continue;
                auto it = model.source_tensors.find(kF6Sources[i]);
                if (it == model.source_tensors.end() || !it->second) continue;
                ggml_tensor * orig = it->second;
                // Defensive: only pre-transpose the F32 [512, 64]
                // shape the audit roster targets.  Any other layout
                // means the GGUF doesn't fit the assumed
                // architecture (or has already been quantized below
                // F32, in which case the call-site rewrite would
                // need a different lowering anyway).
                if (orig->type != GGML_TYPE_F32 ||
                    orig->ne[0] != 512 || orig->ne[1] != 64 ||
                    orig->ne[2] != 1   || orig->ne[3] != 1) {
                    continue;
                }
                std::vector<float> src((size_t) ggml_nelements(orig));
                ggml_backend_tensor_get(orig, src.data(), 0, ggml_nbytes(orig));
                std::vector<float> dst((size_t) 64 * 512);
                // Transpose: dst[i, j] = src[j, i] where source ne=
                // [512, 64].  Memory: src[j * 512 + i],
                // dst[i * 64 + j].
                for (int j = 0; j < 64; ++j) {
                    for (int ii = 0; ii < 512; ++ii) {
                        dst[(size_t) ii * 64 + j] = src[(size_t) j * 512 + ii];
                    }
                }
                ggml_backend_tensor_set(pretrans_t_proj[i], dst.data(), 0, dst.size() * sizeof(float));
                model.source_tensors[std::string(kF6Sources[i]) + "__T"] = pretrans_t_proj[i];
            }
        }

        // Audit follow-up #2 — F13 + F16.
        //
        // F13: pre-download the text-encoder layer-norm weights
        // that the GPU production path's scalar `layer_norm_channel`
        // continuation consumes on every synth.  Roster covers the
        // four `attn_encoder.norm_layers_{1,2}.{0..3}` pairs plus
        // the trailing `speech_prompted_text_encoder.norm.norm.*`
        // pair — 18 entries total — saving ~18 GPU→host syncs per
        // synth on a non-CPU backend.  See
        // `AUDIT_SUPERTONIC_OPENCL.md` § F13 (audit follow-up #2).
        {
            auto cache_if_present = [&](const std::string & name) {
                auto it = model.source_tensors.find(name);
                if (it == model.source_tensors.end() || !it->second) return;
                std::vector<float> & dst = model.text_encoder_ln_weights[name];
                dst.resize((size_t) ggml_nelements(it->second));
                ggml_backend_tensor_get(it->second, dst.data(), 0, ggml_nbytes(it->second));
            };
            static const char * const kLnStems[] = {
                "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1.0",
                "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1.1",
                "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1.2",
                "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1.3",
                "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2.0",
                "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2.1",
                "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2.2",
                "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2.3",
                "text_encoder:tts.ttl.speech_prompted_text_encoder.norm",
            };
            for (const char * stem : kLnStems) {
                cache_if_present(std::string(stem) + ".norm.weight");
                cache_if_present(std::string(stem) + ".norm.bias");
            }
        }

        // F16: pre-download the two `tanh_k` tensors consumed by
        // the speech-prompted attention's CPU-side packing loop.
        // Each is ~50 × 256 floats; the per-synth pattern of "open
        // a fresh ggml graph + read tanh_k + pack q/k/v + run
        // flash attention + tear graph down" still requires the
        // host-side tanh_k bytes for the pack loop, but those
        // bytes don't need a fresh download on every synth.
        {
            static const char * const kTanhKSources[2] = {
                "text_encoder:/speech_prompted_text_encoder/attention1/tanh/Tanh_output_0",
                "text_encoder:/speech_prompted_text_encoder/attention2/tanh/Tanh_output_0",
            };
            for (int i = 0; i < 2; ++i) {
                auto it = model.source_tensors.find(kTanhKSources[i]);
                if (it == model.source_tensors.end() || !it->second) continue;
                model.speech_tanh_k_cache[i].resize((size_t) ggml_nelements(it->second));
                ggml_backend_tensor_get(it->second,
                                        model.speech_tanh_k_cache[i].data(),
                                        0, ggml_nbytes(it->second));
            }
        }

        // Materialize pre-transposed copies of matmul weights to drop the
        // runtime `cont(transpose(w))` dispatch that `dense_matmul_time_ggml`
        // emits on every graph compute (~32 sites × 5 CFM steps per synth).
        // CPU's `cblas_sgemm` already handles the transpose via its `Trans`
        // flag, so this is a Metal-perf-only optimization — skip the extra
        // memory + load-time cost on CPU.  Override via
        // `SUPERTONIC_DISABLE_WEIGHT_PRETRANSPOSE=1` to debug the unpacked
        // path.
        //
        // Coexists with the F6 pre-transposed t_proj pass above: that one
        // handles 4 specific `[512, 64]` `t_proj` weights and registers
        // them under the `__T` suffix; this one handles every other
        // `:onnx::MatMul_` weight under the `:T` suffix.  No collisions.
        static const bool disable_pretranspose =
            std::getenv("SUPERTONIC_DISABLE_WEIGHT_PRETRANSPOSE") != nullptr;
        if (!disable_pretranspose && model.backend &&
            !ggml_backend_is_cpu(model.backend)) {
            std::vector<std::pair<std::string, ggml_tensor *>> to_pretranspose;
            for (const auto & [src_name, t] : model.source_tensors) {
                if (!t) continue;
                if (src_name.find(":onnx::MatMul_") == std::string::npos) continue;
                if (ggml_n_dims(t) != 2) continue;
                // Pretranspose f32 weights (default precision) AND q8_0 / f16
                // weights (asymmetric load modes).  For q8_0 / f16 we
                // dequant→transpose→requantize through f32; the round-trip
                // introduces tiny rounding within the type's existing noise
                // tolerance.  This is what unlocks A3 step 2
                // (kernel_mul_mm_q8_0_f32 / kernel_mul_mm_f16_f32 dispatches
                // when both (a) the pretransposed weight is available as
                // src0 and (b) the new dense_matmul_time_wt_pretransposed_ggml
                // swaps the mul_mat args so the weight is src0).
                if (t->type != GGML_TYPE_F32 &&
                    t->type != GGML_TYPE_F16  &&
                    t->type != GGML_TYPE_Q8_0) continue;
                to_pretranspose.push_back({src_name, t});
            }
            if (!to_pretranspose.empty()) {
                ggml_init_params extra_params = {
                    /*.mem_size=*/ ggml_tensor_overhead() * to_pretranspose.size(),
                    /*.mem_buffer=*/ nullptr,
                    /*.no_alloc=*/ true,
                };
                model.ctx_w_extra = ggml_init(extra_params);
                if (!model.ctx_w_extra) {
                    throw std::runtime_error("ggml_init ctx_w_extra failed");
                }
                std::vector<std::pair<ggml_tensor *, ggml_tensor *>> orig_to_pre;
                orig_to_pre.reserve(to_pretranspose.size());
                for (const auto & [src_name, t] : to_pretranspose) {
                    // Pre tensor has same type as orig (f32 stays f32,
                    // q8_0 stays q8_0); only the shape swaps.
                    ggml_tensor * tt = ggml_new_tensor_2d(model.ctx_w_extra,
                        t->type, t->ne[1], t->ne[0]);
                    const std::string tt_name = std::string(ggml_get_name(t)) + ":T";
                    ggml_set_name(tt, tt_name.c_str());
                    model.source_tensors[src_name + ":T"] = tt;
                    orig_to_pre.push_back({t, tt});
                }
                model.buffer_w_extra =
                    ggml_backend_alloc_ctx_tensors(model.ctx_w_extra, model.backend);
                if (!model.buffer_w_extra) {
                    throw std::runtime_error(
                        "ggml_backend_alloc_ctx_tensors ctx_w_extra failed");
                }
                // Upload the transposed data.  For f32 weights this is a
                // straight host-side reorder.  For q8_0 weights we dequant
                // to f32, transpose in f32, then requantize via from_float
                // into the pretransposed q8_0 tensor.  Both directions go
                // through the public ggml type-traits APIs.
                for (const auto & [orig, pre] : orig_to_pre) {
                    const int OC = (int) orig->ne[0];
                    const int IC = (int) orig->ne[1];
                    const size_t n = (size_t) OC * IC;

                    // Step 1: download `orig` data, dequantize to f32 if needed.
                    std::vector<float> host_orig_f32(n);
                    if (orig->type == GGML_TYPE_F32) {
                        ggml_backend_tensor_get(orig, host_orig_f32.data(), 0,
                                                n * sizeof(float));
                    } else {
                        std::vector<uint8_t> raw(ggml_nbytes(orig));
                        ggml_backend_tensor_get(orig, raw.data(), 0, raw.size());
                        const ggml_type_traits * tr = ggml_get_type_traits(orig->type);
                        if (!tr || !tr->to_float) {
                            throw std::runtime_error(
                                std::string("pretranspose: missing to_float for ") +
                                ggml_type_name(orig->type));
                        }
                        tr->to_float(raw.data(), host_orig_f32.data(), (int64_t) n);
                    }

                    // Step 2: transpose in f32.
                    std::vector<float> host_pre_f32(n);
                    for (int oc = 0; oc < OC; ++oc) {
                        for (int ic = 0; ic < IC; ++ic) {
                            host_pre_f32[(size_t) ic + (size_t) oc * IC] =
                                host_orig_f32[(size_t) oc + (size_t) ic * OC];
                        }
                    }

                    // Step 3: upload (requantizing if needed).
                    if (pre->type == GGML_TYPE_F32) {
                        ggml_backend_tensor_set(pre, host_pre_f32.data(), 0,
                                                n * sizeof(float));
                    } else {
                        const size_t dst_bytes = ggml_row_size(pre->type, n);
                        std::vector<uint8_t> raw(dst_bytes);
                        const ggml_type_traits_cpu * dtr =
                            ggml_get_type_traits_cpu(pre->type);
                        if (!dtr || !dtr->from_float) {
                            throw std::runtime_error(
                                std::string("pretranspose: missing from_float for ") +
                                ggml_type_name(pre->type));
                        }
                        dtr->from_float(host_pre_f32.data(), raw.data(), (int64_t) n);
                        ggml_backend_tensor_set(pre, raw.data(), 0, raw.size());
                    }
                    model.pretransposed_weights[orig] = pre;
                }
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "load_supertonic_gguf: %s\n", e.what());
        gguf_free(gguf_ctx);
        if (tmp_ctx) ggml_free(tmp_ctx);
        free_supertonic_model(model);
        return false;
    }

    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
    // Mark this model alive only after all the load steps succeeded.
    // The per-stage thread_local graph caches consult is_supertonic_alive()
    // before calling ggml_gallocr_free() to skip the free path against a
    // backend that's already been torn down.
    register_supertonic_alive(model.generation_id);
    return true;
}

void free_supertonic_model(supertonic_model & model) {
    // Unregister BEFORE freeing the backend so any concurrent / subsequent
    // free_*_cache() call on a stale thread_local cache sees the
    // generation as no-longer-alive and skips ggml_gallocr_free against
    // the soon-to-be-dead backend.
    if (model.generation_id != 0) {
        unregister_supertonic_alive(model.generation_id);
    }
    if (model.buffer_w_extra) {
        ggml_backend_buffer_free(model.buffer_w_extra);
        model.buffer_w_extra = nullptr;
    }
    if (model.buffer_w) {
        ggml_backend_buffer_free(model.buffer_w);
        model.buffer_w = nullptr;
    }
    if (model.backend) {
        ggml_backend_free(model.backend);
        model.backend = nullptr;
    }
    if (model.ctx_w_extra) {
        ggml_free(model.ctx_w_extra);
        model.ctx_w_extra = nullptr;
    }
    if (model.ctx_w) {
        ggml_free(model.ctx_w);
        model.ctx_w = nullptr;
    }
    model.pretransposed_weights.clear();
    model.tensors.clear();
    model.source_tensors.clear();
    model.vocoder = {};
    model.voices.clear();
    model.unicode_indexer.clear();
    model.languages.clear();
    model.tts_json.clear();
    // Reset the OpenCL optimization caches (audit F1 / F9 + F13 /
    // F16) added to supertonic_model.  The vector-estimator RoPE θ
    // cache is a bare std::vector so its clear() is sufficient; the
    // time embedding cache map is mutable so we clear it explicitly
    // here even though dtor would handle it on the next load reuse.
    model.vector_rope_theta.clear();
    model.time_emb_cache.clear();
    model.text_encoder_ln_weights.clear();
    for (auto & v : model.speech_tanh_k_cache) v.clear();
    model.scalar_weight_cache.clear();
    model.generation_id = 0;
}

} // namespace tts_cpp::supertonic::detail
