// CosyVoice3 native pipeline — shared library core.  See cosyvoice_pipeline.h.
//
// The graph code here is lifted verbatim (numerically) from the validated
// parity CLIs; only the I/O boundary changed (in-memory args/returns instead
// of npy files).  Keep the ggml op sequences byte-for-byte identical to the
// CLIs — the bring-up bugs found during validation were all subtle op
// ordering / layout issues.

#include "cosyvoice_pipeline.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "gguf_stream.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// GGUF loader / accessors
// ===========================================================================
// Weights the pipeline gathers rows from with a raw host pointer
// (build_lm_input and the per-token embedding lookup); never graph operands.
// ggml-opencl's buffer_get_base() returns a small sentinel rather than a
// mappable pointer, so a device-resident ggml_get_data() on these is a segfault
// -- and lm/embed_tokens/weight is ~545 MB the GPU would never read.  Keeping
// them in a host buffer is both the correctness fix and the memory win.
static bool cosyvoice_host_resident(const char * name) {
    return std::strcmp(name, "lm/embed_tokens/weight")     == 0 ||
           std::strcmp(name, "lm/speech_embedding/weight") == 0;
}

// Raw row pointer into a weight the CPU indexes directly.  Throws rather than
// returning a device pointer, so a table that stops being host-resident (a
// rename, a new lookup) fails loudly here instead of segfaulting inside the
// gather.  Keep this and cosyvoice_host_resident in step.
static const float * host_rows(ggml_tensor * t, const char * who) {
    if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
        throw std::runtime_error(std::string(who) + ": " + ggml_get_name(t) +
                                 " must be host-resident (see cosyvoice_host_resident)");
    }
    return (const float *) ggml_get_data(t);
}

// Arena for a graph of at most `n` tensors: n tensor structs + the cgraph
// object (node/leaf arrays + hash set), plus 1 MiB of slack.
//
// mem_buffer is nullptr on purpose: ggml then owns the block via
// ggml_aligned_malloc, which does NOT zero it.  The previous
// `std::vector<uint8_t> buf(N)` value-initialised its whole range -- up to
// 2 GiB of memset, and the matching RSS, per call -- purely to hand ggml
// scratch it immediately overwrites.
//
// Under-sizing is loud, not silent: ggml_new_object logs "not enough space in
// the context's memory pool" and the following GGML_ASSERT aborts.  Pass the
// SAME n to ggml_new_graph_custom so the graph's own allocation is covered.
static inline ggml_init_params cosy_arena(size_t n) {
    return { ggml_tensor_overhead() * n + ggml_graph_overhead_custom(n, false) + (size_t)(1u << 20),
             nullptr, /*no_alloc=*/true };
}

// Graph node budgets.  Measured on a 3.8 s utterance (LM depth 24, DiT depth 22,
// conv_groups 16): LM 1155, DiT 2323, flow frontend 31, HiFT f0 51, HiFT decode
// 1219, STFT 53.  These counts are STRUCTURAL -- they follow depth/groups and
// never the input length (reflect_pad_1d loops over the pad width, the HiFT
// loops run over the 3 fixed upsample stages) -- so a hyper-parameter-derived
// budget stays valid for any text.  Adding ops to a stage means re-checking its
// budget by hand: nothing here is enforced by a test yet.
static inline size_t cosy_lm_nodes(const qwen_hp & hp) {
    return (size_t)hp.depth * 64 + 256;                    // measured ~48/layer
}
static inline size_t cosy_dit_nodes(const dit_hp & hp) {
    // ~100 nodes/layer, plus conv_pos_embed which emits per (group, batch).
    return (size_t)hp.depth * 128 + (size_t)hp.conv_groups * 96 + 512;
}
static constexpr size_t kCosyFlowFrontendNodes = 256;
static constexpr size_t kCosyHiftF0Nodes       = 256;
static constexpr size_t kCosyHiftDecodeNodes   = 2048;
static constexpr size_t kCosyStftNodes         = 256;


// ---- optional per-stage instrumentation -----------------------------------
// Synchronises the backend before reading the clock: on an async backend the
// submit call returns immediately, so an unsynchronised timer measures enqueue
// latency, not compute.
using cosy_clk = std::chrono::steady_clock;
static inline double cosy_ms_since(cosy_clk::time_point t0, ggml_backend_t b) {
    if (b) ggml_backend_synchronize(b);
    return std::chrono::duration<double, std::milli>(cosy_clk::now() - t0).count();
}

// ---- dual-path graph dispatch ---------------------------------------------
// Which path a stage takes is DERIVED from the backend's supports_op walk, not
// hardcoded: a ggml sync that lands a missing kernel upgrades that stage to the
// direct path by itself, and one that drops support falls back to the scheduler
// instead of aborting inside graph_compute.  That last part is load-bearing --
// ggml-opencl has no per-op fallback, an unsupported op there is a GGML_ASSERT.
//
// Direct path: caller-owned gallocr, and the graph MAY be replayed.
// Sched path : the sched owns allocation and graphs are SINGLE-USE, so a caller
//              that replays a graph must rebuild it before every pass.
static bool cosy_dispatch_prepare(model_ctx & m, ggml_cgraph * gf, ggml_gallocr_t allocr,
                                  size_t graph_size, bool & use_sched, const char * caller) {
    namespace det = ::tts_cpp::detail;
    use_sched = det::sched_force_enabled() || !det::graph_fully_supported(m.backend, gf);
    if (!use_sched) {
        if (!ggml_gallocr_reserve(allocr, gf) || !ggml_gallocr_alloc_graph(allocr, gf)) {
            fprintf(stderr, "%s: graph allocation failed\n", caller);
            return false;
        }
        return true;
    }
    if (det::graph_has_unsupported_preallocated_op(m.backend, gf)) {
        fprintf(stderr, "%s: an op writing a pre-allocated buffer is runnable by neither "
                        "the primary backend nor the CPU; cannot enter the scheduler\n", caller);
        return false;
    }
    // ggml_backend_sched_alloc_graph asserts hash_set.size >= n_nodes+n_leafs.
    // n_leafs has no public accessor, but ggml_new_graph_custom(c, graph_size)
    // caps BOTH arrays at graph_size, so 2*graph_size is the safe bound.
    const size_t need = 2 * graph_size;
    if (m.sched_fb && m.sched_graph_size < need) {
        det::sched_fallback_free(*m.sched_fb);   // too small for this stage -- rebuild
        m.sched_fb.reset();
    }
    if (!m.sched_fb) { m.sched_fb.reset(new det::sched_fallback()); m.sched_graph_size = need; }
    const std::vector<ggml_backend_buffer_t> wbufs = { m.buffer_w };
    if (!det::sched_fallback_ensure(*m.sched_fb, m.backend, m.sched_graph_size, wbufs)) return false;
    return det::sched_fallback_alloc(*m.sched_fb, gf);
}

static bool cosy_dispatch_compute(model_ctx & m, ggml_cgraph * gf, bool use_sched,
                                  const char * caller) {
    namespace det = ::tts_cpp::detail;
    const ggml_status st = use_sched
        ? det::sched_fallback_compute(*m.sched_fb, m.backend, gf, m.n_threads)
        : det::direct_compute(m.backend, gf, m.n_threads);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: compute failed (status %d)\n", caller, (int)st);
        return false;
    }
    return true;
}

// ggml's CPU quant kernels require GGUF general.alignment (default 32); a tensor
// whose mapped pointer is under-aligned falls back to an allocated copy.
static constexpr uintptr_t kCosyvoiceTensorAlignment = 32;

model_ctx cosyvoice_load_gguf(const std::string & path, ggml_backend_t backend) {
    // Heap-allocated, matching chatterbox's model_ctx: a stack model_ctx corrupts
    // its own std::map during load on the win32 MSVC-lib + clang-addon link.
    auto mp = std::make_unique<model_ctx>();
    model_ctx & m = *mp;

    // Map the GGUF so CPU/host weights can be backed in place; fall back to a
    // resident copy when mapping fails.
    const bool have_map = tts_cpp::cosyvoice::mapped_file_open(m.mapped, path, "cosyvoice");

    ggml_context * tmp_ctx = nullptr;
    // Shapes only; bytes come from the mapping or the bounds-checked stream
    // reader below, never from tmp_ctx.
    gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ &tmp_ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        tts_cpp::cosyvoice::mapped_file_close(m.mapped);
        throw std::runtime_error("gguf_init_from_file failed: " + path);
    }
    if (backend) {
        m.backend = backend;              // borrowed; the caller frees it
    } else {
        // NOT ggml_backend_cpu_init(): that symbol lives in libggml-cpu, which
        // is not linked under GGML_BACKEND_DL=ON (the Android/OpenCL build).
        m.backend = ::tts_cpp::detail::init_cpu_backend();
        m.owns_backend = true;
        if (!m.backend) {
            gguf_free(g); ggml_free(tmp_ctx);
            tts_cpp::cosyvoice::mapped_file_close(m.mapped);
            throw std::runtime_error("cosyvoice: failed to init a CPU backend for " + path);
        }
    }
    // Split only when the backend is not the CPU: on CPU everything is already
    // host-resident, so the single-context layout stays exactly as before.
    const bool split_host = !::tts_cpp::detail::backend_is_cpu(m.backend);
    const int64_t n_tensors = gguf_get_n_tensors(g);
    int64_t n_host = 0;
    if (split_host) {
        for (int64_t i = 0; i < n_tensors; ++i) {
            if (cosyvoice_host_resident(gguf_get_tensor_name(g, i))) n_host++;
        }
    }
    const int64_t n_dev = n_tensors - n_host;

    ggml_init_params p = { ggml_tensor_overhead() * (size_t)(n_dev + 1), nullptr, true };
    m.ctx_w = ggml_init(p);
    if (n_host > 0) {
        ggml_init_params ph = { ggml_tensor_overhead() * (size_t)(n_host + 1), nullptr, true };
        m.ctx_h = ggml_init(ph);
    }

    if (have_map) {
        m.map_buf = ggml_backend_cpu_buffer_from_ptr((void *) m.mapped.data, m.mapped.size);
    }
    // mapping gates pointer map-in-place; have_map still uploads from the mapping
    // (not tmp_ctx) if only the wrapper buffer failed.
    const bool mapping    = have_map && m.map_buf != nullptr;
    const size_t data_off = have_map ? gguf_get_data_offset(g) : 0;

    // Point host tensors (all on CPU; the host-resident tables on GPU) at the
    // mapping when in-bounds + aligned; false leaves the tensor for alloc+upload.
    auto map_in_place = [&](ggml_tensor * dst, int64_t gi, bool host_ctx) -> bool {
        if (!mapping) return false;
        const bool wants_host = (!split_host) || host_ctx;
        if (!wants_host) return false;
        const size_t nb  = ggml_nbytes(dst);
        const size_t off = data_off + (size_t) gguf_get_tensor_offset(g, gi);
        if (off > m.mapped.size || nb > m.mapped.size - off) return false;
        const uint8_t * src = m.mapped.data + off;
        if (((uintptr_t) src % kCosyvoiceTensorAlignment) != 0) return false;
        dst->data   = (void *) src;
        dst->buffer = m.map_buf;
        return true;
    };

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        const bool host_ctx = split_host && cosyvoice_host_resident(name);
        ggml_context * into = host_ctx ? m.ctx_h : m.ctx_w;
        ggml_tensor * dst = ggml_dup_tensor(into, src);
        ggml_set_name(dst, name);
        map_in_place(dst, i, host_ctx);
        m.tensors[name] = dst;
    }

    // alloc_ctx_tensors skips tensors already backed by the mapping; on the
    // pure-CPU mapped path nothing is allocated (buffer_w stays null).
    auto has_unmapped = [](ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data == nullptr) return true;
        }
        return false;
    };
    if (!mapping || has_unmapped(m.ctx_w)) {
        m.buffer_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, m.backend);
    }
    if (m.ctx_h && (!mapping || has_unmapped(m.ctx_h))) {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu_dev) {
            gguf_free(g); ggml_free(tmp_ctx);
            tts_cpp::cosyvoice::mapped_file_close(m.mapped);
            throw std::runtime_error("cosyvoice: no CPU device for host-resident weights");
        }
        m.buffer_h = ggml_backend_alloc_ctx_tensors_from_buft(m.ctx_h, ggml_backend_dev_buffer_type(cpu_dev));
    }

    // Stream the unmapped tensors with the bounds-checked reader (a truncated
    // GGUF fails cleanly here instead of reading past the mapping).
    {
        ::tts_cpp::detail::gguf_stream_reader rd(g, path);
        if (!rd.ok()) {
            gguf_free(g); ggml_free(tmp_ctx);
            tts_cpp::cosyvoice::mapped_file_close(m.mapped);
            throw std::runtime_error("cosyvoice: failed to reopen GGUF for streaming: " + path);
        }
        for (ggml_context * ctx : { m.ctx_w, m.ctx_h }) {
            if (!ctx) continue;
            for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
                if (mapping && cur->buffer == m.map_buf) continue;
                if (!rd.to_backend(ggml_get_name(cur), cur)) {
                    gguf_free(g); ggml_free(tmp_ctx);
                    tts_cpp::cosyvoice::mapped_file_close(m.mapped);
                    throw std::runtime_error(std::string("cosyvoice: failed to stream tensor ") +
                                             ggml_get_name(cur));
                }
            }
        }
    }
    // Capture scalar KV metadata (architecture sizes + special-token ids the
    // converter wrote) so downstream graphs read them instead of hardcoding.
    for (int64_t i = 0, nkv = gguf_get_n_kv(g); i < nkv; ++i) {
        const char * key = gguf_get_key(g, i);
        switch (gguf_get_kv_type(g, i)) {
            case GGUF_TYPE_UINT8:   m.kv_i[key] = gguf_get_val_u8(g, i);  break;
            case GGUF_TYPE_INT8:    m.kv_i[key] = gguf_get_val_i8(g, i);  break;
            case GGUF_TYPE_UINT16:  m.kv_i[key] = gguf_get_val_u16(g, i); break;
            case GGUF_TYPE_INT16:   m.kv_i[key] = gguf_get_val_i16(g, i); break;
            case GGUF_TYPE_UINT32:  m.kv_i[key] = gguf_get_val_u32(g, i); break;
            case GGUF_TYPE_INT32:   m.kv_i[key] = gguf_get_val_i32(g, i); break;
            case GGUF_TYPE_UINT64:  m.kv_i[key] = (int64_t)gguf_get_val_u64(g, i); break;
            case GGUF_TYPE_INT64:   m.kv_i[key] = gguf_get_val_i64(g, i); break;
            case GGUF_TYPE_FLOAT32: m.kv_f[key] = gguf_get_val_f32(g, i); break;
            default: break;  // strings handled separately; arrays/bools ignored
        }
    }
    gguf_free(g);
    ggml_free(tmp_ctx);
    return std::move(m);
}

void cosyvoice_free(model_ctx & m) {
    // Order is load-bearing: the sched holds refs into `backend`, so it must go
    // first (sched_dispatch.h ordering contract).  The backend itself is freed
    // only when this model_ctx owns it -- a shared engine backend outlives all
    // four model_ctx and is freed by its creator afterwards.
    if (m.sched_fb) { ::tts_cpp::detail::sched_fallback_free(*m.sched_fb); m.sched_fb.reset(); }
    if (m.buffer_h) ggml_backend_buffer_free(m.buffer_h);
    if (m.ctx_h)    ggml_free(m.ctx_h);
    if (m.buffer_w) ggml_backend_buffer_free(m.buffer_w);
    if (m.ctx_w)    ggml_free(m.ctx_w);
    // Free the wrapper (not the mapping) after ctx_w, then munmap.
    if (m.map_buf)  ggml_backend_buffer_free(m.map_buf);
    tts_cpp::cosyvoice::mapped_file_close(m.mapped);
    if (m.owns_backend && m.backend) ggml_backend_free(m.backend);
    m.buffer_h = nullptr; m.ctx_h = nullptr;
    m.buffer_w = nullptr; m.ctx_w = nullptr;
    m.map_buf = nullptr;
    m.backend = nullptr;  m.owns_backend = false;
    m.tensors.clear();
}

ggml_tensor * cosyvoice_get(const model_ctx & m, const std::string & name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}

std::string cosyvoice_gguf_meta_str(const std::string & path, const std::string & key,
                                    const std::string & fallback) {
    gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return fallback;
    std::string out = fallback;
    int64_t kid = gguf_find_key(g, key.c_str());
    if (kid >= 0 && gguf_get_kv_type(g, kid) == GGUF_TYPE_STRING) {
        out = gguf_get_val_str(g, kid);
    }
    gguf_free(g);
    return out;
}

int64_t cosyvoice_meta_i(const model_ctx & m, const std::string & key, int64_t fallback) {
    auto it = m.kv_i.find(key);
    return it != m.kv_i.end() ? it->second : fallback;
}

float cosyvoice_meta_f(const model_ctx & m, const std::string & key, float fallback) {
    auto it = m.kv_f.find(key);
    return it != m.kv_f.end() ? it->second : fallback;
}

qwen_hp cosyvoice_qwen_hp(const model_ctx & m) {
    qwen_hp hp;  // struct defaults are the per-field fallback
    hp.depth    = (int)cosyvoice_meta_i(m, "cosyvoice3.llm.depth",    hp.depth);
    hp.hidden   = (int)cosyvoice_meta_i(m, "cosyvoice3.llm.hidden",   hp.hidden);
    hp.n_head   = (int)cosyvoice_meta_i(m, "cosyvoice3.llm.n_head",   hp.n_head);
    hp.n_kv     = (int)cosyvoice_meta_i(m, "cosyvoice3.llm.n_kv",     hp.n_kv);
    hp.head_dim = (int)cosyvoice_meta_i(m, "cosyvoice3.llm.head_dim", hp.head_dim);
    hp.inter    = (int)cosyvoice_meta_i(m, "cosyvoice3.llm.inter",    hp.inter);
    hp.theta    = cosyvoice_meta_f(m, "cosyvoice3.llm.rope_theta", hp.theta);
    hp.eps      = cosyvoice_meta_f(m, "cosyvoice3.llm.rms_eps",    hp.eps);
    return hp;
}

// short local aliases matching the CLI call sites
static inline ggml_tensor * T(const model_ctx & m, const std::string & n) { return cosyvoice_get(m, n); }
static inline ggml_tensor * G(const model_ctx & m, const std::string & n) { return cosyvoice_get(m, n); }

// ===========================================================================
// Shared graph helpers
// ===========================================================================
static ggml_tensor * linear(ggml_context * c, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(c, w, x);
    if (b) y = ggml_add(c, y, ggml_reshape_3d(c, b, b->ne[0], 1, 1));
    return y;
}
static ggml_tensor * ln_noaffine(ggml_context * c, ggml_tensor * x) { return ggml_norm(c, x, 1e-6f); }
static ggml_tensor * adaln_modulate(ggml_context * c, ggml_tensor * x_ln,
                                    ggml_tensor * scale, ggml_tensor * shift) {
    ggml_tensor * s = ggml_reshape_3d(c, scale, scale->ne[0], 1, scale->ne[1]);
    ggml_tensor * h = ggml_reshape_3d(c, shift, shift->ne[0], 1, shift->ne[1]);
    return ggml_add(c, ggml_add(c, x_ln, ggml_mul(c, x_ln, s)), h);
}
static ggml_tensor * silu(ggml_context * c, ggml_tensor * x) { return ggml_silu(c, x); }
// x * tanh(softplus(x)) -- the definition PyTorch's F.mish uses.  ggml's
// SOFTPLUS is logf(1+expf(x)) below the x>20 cutoff, so this matches the old
// explicit exp/log form exactly while dropping GGML_OP_LOG (which ggml-opencl
// does not implement) and the host-allocated `one` operand.
static ggml_tensor * mish(ggml_context * c, ggml_tensor * x) {
    return ggml_mul(c, x, ggml_tanh(c, ggml_softplus(c, x)));
}

// Grouped conv1d over time. x: [Nlen, Cin, B] (ne0=time), weight [K, Cin/groups, Cout].
//
// Emitted per (group, batch) with a 2-D signal and the im2col matrix as operand
// A -- the same operand order conv1d_f32 uses.  Both details are load-bearing on
// GPU backends that fuse IM2COL into the GEMM: the fusion is refused when the
// signal is batched or when the im2col is src[1].  Unfused, materialising that
// matrix costs an order of magnitude more than the convolution's own arithmetic
// (measured on Adreno: 77 s of a 143 s synthesis for ~2% of the FLOPs).
// Arithmetically identical to the batched form -- same dot products, same order.
static ggml_tensor * conv1d_grouped(ggml_context * c, ggml_tensor * w, ggml_tensor * x, int groups) {
    const int Cout  = (int)w->ne[2];
    const int Cin   = (int)x->ne[1];
    const int B     = (int)x->ne[2];
    const int cin_g = Cin / groups, cout_g = Cout / groups;
    ggml_tensor * out = nullptr;
    for (int b = 0; b < B; ++b) {
        ggml_tensor * acc = nullptr;
        for (int g = 0; g < groups; ++g) {
            ggml_tensor * xg = ggml_view_2d(c, x, x->ne[0], cin_g, x->nb[1],
                                            (size_t)b * x->nb[2] + (size_t)g * cin_g * x->nb[1]);
            xg = ggml_cont(c, xg);                       // [Nlen, cin_g] -- 2-D signal
            ggml_tensor * wg = ggml_view_3d(c, w, w->ne[0], w->ne[1], cout_g,
                                            w->nb[1], w->nb[2], (size_t)g * cout_g * w->nb[2]);
            wg = ggml_cont(c, wg);
            ggml_tensor * im = ggml_im2col(c, wg, xg, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
            ggml_tensor * yg = ggml_mul_mat(c,
                ggml_reshape_2d(c, im, im->ne[0], im->ne[1]),           // A = im2col [K*cin_g, OW]
                ggml_reshape_2d(c, wg, wg->ne[0] * wg->ne[1], wg->ne[2]));
            acc = acc ? ggml_concat(c, acc, yg, 1) : yg;                // -> [OW, Cout]
        }
        out = out ? ggml_concat(c, out, acc, 2) : acc;                  // -> [OW, Cout, B]
    }
    return out;
}

// Plain conv1d over time: input [Nlen, Cin, B] (ne0=time), weight [K, Cin, Cout].
// im2col FIRST, kernel SECOND (conv1d operand order matters here).
static ggml_tensor * conv1d_f32(ggml_context * c, ggml_tensor * w, ggml_tensor * x,
                                int stride, int padding, int dilation) {
    ggml_tensor * im = ggml_im2col(c, w, x, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * r = ggml_mul_mat(c,
        ggml_reshape_2d(c, im, im->ne[0], im->ne[2] * im->ne[1]),
        ggml_reshape_2d(c, w, w->ne[0] * w->ne[1], w->ne[2]));
    return ggml_reshape_3d(c, r, im->ne[1], w->ne[2], im->ne[2]);
}

static ggml_tensor * rmsnorm(ggml_context * c, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

// ===========================================================================
// Stage 5 — Qwen2.5 speech LM
// ===========================================================================
static std::string lb(int i, const std::string & s) { return "lm/blk/" + std::to_string(i) + "/" + s; }

ggml_tensor * build_qwen(ggml_context * c, const model_ctx & m, const qwen_hp & hp,
                         ggml_tensor * x, ggml_tensor * pos, ggml_tensor * mask, int L) {
    const int HD = hp.head_dim, NH = hp.n_head, NKV = hp.n_kv, G_ = NH / NKV;
    const float scale = 1.0f / std::sqrt((float)HD);
    for (int i = 0; i < hp.depth; ++i) {
        ggml_tensor * h = rmsnorm(c, x, G(m, lb(i, "in_ln/weight")), hp.eps);
        ggml_tensor * q = ggml_add(c, ggml_mul_mat(c, G(m, lb(i, "q_proj/weight")), h), G(m, lb(i, "q_proj/bias")));
        ggml_tensor * k = ggml_add(c, ggml_mul_mat(c, G(m, lb(i, "k_proj/weight")), h), G(m, lb(i, "k_proj/bias")));
        ggml_tensor * v = ggml_add(c, ggml_mul_mat(c, G(m, lb(i, "v_proj/weight")), h), G(m, lb(i, "v_proj/bias")));
        q = ggml_reshape_3d(c, q, HD, NH, L);
        k = ggml_reshape_3d(c, k, HD, NKV, L);
        v = ggml_reshape_3d(c, v, HD, NKV, L);
        q = ggml_rope_ext(c, q, pos, nullptr, HD, GGML_ROPE_TYPE_NEOX, 0, hp.theta, 1.0f, 0, 1, 0, 0);
        k = ggml_rope_ext(c, k, pos, nullptr, HD, GGML_ROPE_TYPE_NEOX, 0, hp.theta, 1.0f, 0, 1, 0, 0);
        ggml_tensor * qh = ggml_reshape_4d(c, ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3)), HD, L, G_, NKV);
        ggml_tensor * kh = ggml_reshape_4d(c, ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3)), HD, L, 1, NKV);
        ggml_tensor * vh = ggml_reshape_4d(c, ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3)), HD, L, 1, NKV);
        ggml_tensor * sc = ggml_mul_mat(c, kh, qh);
        sc = ggml_soft_max_ext(c, sc, mask, scale, 0.0f);
        ggml_tensor * vt = ggml_cont(c, ggml_permute(c, vh, 1, 0, 2, 3));
        ggml_tensor * o = ggml_mul_mat(c, vt, sc);
        o = ggml_cont(c, ggml_permute(c, o, 0, 3, 1, 2));
        o = ggml_reshape_3d(c, o, HD, NH, L);
        o = ggml_reshape_2d(c, o, static_cast<int64_t>(HD) * NH, L);
        o = ggml_mul_mat(c, G(m, lb(i, "o_proj/weight")), o);
        x = ggml_add(c, x, o);
        ggml_tensor * hn = rmsnorm(c, x, G(m, lb(i, "post_ln/weight")), hp.eps);
        ggml_tensor * gate = ggml_silu(c, ggml_mul_mat(c, G(m, lb(i, "gate/weight")), hn));
        ggml_tensor * up   = ggml_mul_mat(c, G(m, lb(i, "up/weight")), hn);
        ggml_tensor * down = ggml_mul_mat(c, G(m, lb(i, "down/weight")), ggml_mul(c, gate, up));
        x = ggml_add(c, x, down);
    }
    x = rmsnorm(c, x, G(m, "lm/norm/weight"), hp.eps);
    ggml_set_name(x, "hidden"); ggml_set_output(x);
    return ggml_mul_mat(c, G(m, "lm/llm_decoder/weight"), x);
}

// Per-layer KV cache holding POST-rope K / (unroped) V, resident in a backend
// buffer sized once to the max sequence length (rope is position-deterministic,
// so caching after rope is correct — same as llama.cpp).  Each step appends its
// new Lq columns in-graph (ggml_cpy into a column-offset view) and reads the
// past as a view of the first P columns, so a step moves O(Lq) data instead of
// round-tripping the whole O(P) cache host<->backend every step (which was
// O(L^2) over a full decode).  Layout per layer: [HD, NKV, max_P] (ne2=time).
struct qwen_kvcache {
    ggml_backend_t          backend = nullptr;
    ggml_context *          ctx     = nullptr;
    ggml_backend_buffer_t   buf     = nullptr;
    std::vector<ggml_tensor*> k, v;   // per layer, resident [HD, NKV, max_P]
    int P = 0, max_P = 0;
    void init(model_ctx & m, const qwen_hp & hp, int max_tokens) {
        backend = m.backend; max_P = max_tokens; P = 0;
        const int HD = hp.head_dim, NKV = hp.n_kv, depth = hp.depth;
        ggml_init_params p = { ggml_tensor_overhead() * (size_t)(2 * depth) + 64, nullptr, /*no_alloc=*/true };
        ctx = ggml_init(p);
        k.assign(depth, nullptr); v.assign(depth, nullptr);
        for (int i = 0; i < depth; ++i) {
            k[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, NKV, max_P);
            v[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, NKV, max_P);
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }
    void free() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr; ctx = nullptr; P = 0; k.clear(); v.clear();
    }
};

// One prefill/decode step with a KV cache.  x_new: [D, Lq], Lq tokens starting
// at absolute position cache.P.  Only K/V for the Lq new tokens are computed;
// the past K/V come from the cache (concatenated in), so a decode step is O(L)
// not O(L^2).  Updates the cache in place; returns the LAST-position logits [VS].
static std::vector<float> qwen_step_kv(model_ctx & m, const qwen_hp & hp,
        const float * x_new, int Lq, int D, int VS,
        qwen_kvcache & cache, ggml_gallocr_t al) {
    const int HD = hp.head_dim, NH = hp.n_head, NKV = hp.n_kv, G_ = NH / NKV;
    const float scale = 1.0f / std::sqrt((float)HD);
    const int P = cache.P, Lk = P + Lq;

    const size_t nmax = cosy_lm_nodes(hp);
    ggml_init_params gp = cosy_arena(nmax);
    ggml_context * c = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(c, nmax, false);

    ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, Lq); ggml_set_name(x,"x"); ggml_set_input(x);
    ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, Lq); ggml_set_name(pos,"pos"); ggml_set_input(pos);
    ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, Lk, Lq); ggml_set_name(mask,"mask"); ggml_set_input(mask);

    // In-graph appends of this step's post-rope K/V into the resident cache.
    std::vector<ggml_tensor*> cpy_k(hp.depth, nullptr), cpy_v(hp.depth, nullptr);

    ggml_tensor * xx = x;
    for (int i = 0; i < hp.depth; ++i) {
        ggml_tensor * h = rmsnorm(c, xx, G(m, lb(i,"in_ln/weight")), hp.eps);
        ggml_tensor * q = ggml_add(c, ggml_mul_mat(c, G(m,lb(i,"q_proj/weight")), h), G(m,lb(i,"q_proj/bias")));
        ggml_tensor * k = ggml_add(c, ggml_mul_mat(c, G(m,lb(i,"k_proj/weight")), h), G(m,lb(i,"k_proj/bias")));
        ggml_tensor * v = ggml_add(c, ggml_mul_mat(c, G(m,lb(i,"v_proj/weight")), h), G(m,lb(i,"v_proj/bias")));
        q = ggml_reshape_3d(c, q, HD, NH, Lq);
        k = ggml_reshape_3d(c, k, HD, NKV, Lq);
        v = ggml_reshape_3d(c, v, HD, NKV, Lq);
        q = ggml_rope_ext(c, q, pos, nullptr, HD, GGML_ROPE_TYPE_NEOX, 0, hp.theta, 1.0f,0,1,0,0);
        k = ggml_rope_ext(c, k, pos, nullptr, HD, GGML_ROPE_TYPE_NEOX, 0, hp.theta, 1.0f,0,1,0,0);
        // Past K/V come from the resident cache's first P columns (written by
        // prior steps); attention runs over past ++ this step's new K/V.
        ggml_tensor * past_k = (P > 0) ? ggml_view_3d(c, cache.k[i], HD, NKV, P,
                                             cache.k[i]->nb[1], cache.k[i]->nb[2], 0) : nullptr;
        ggml_tensor * past_v = (P > 0) ? ggml_view_3d(c, cache.v[i], HD, NKV, P,
                                             cache.v[i]->nb[1], cache.v[i]->nb[2], 0) : nullptr;
        ggml_tensor * kc = (P > 0) ? ggml_concat(c, past_k, k, 2) : k;   // [HD,NKV,Lk]
        ggml_tensor * vc = (P > 0) ? ggml_concat(c, past_v, v, 2) : v;
        // Append this step's new K/V into the resident cache at columns [P,Lk)
        // for future steps.  Disjoint from past_k/past_v (which read [0,P)), so
        // there is no read-after-write hazard within this graph.
        ggml_tensor * dst_k = ggml_view_3d(c, cache.k[i], HD, NKV, Lq,
                                  cache.k[i]->nb[1], cache.k[i]->nb[2], (size_t)P * cache.k[i]->nb[2]);
        ggml_tensor * dst_v = ggml_view_3d(c, cache.v[i], HD, NKV, Lq,
                                  cache.v[i]->nb[1], cache.v[i]->nb[2], (size_t)P * cache.v[i]->nb[2]);
        cpy_k[i] = ggml_cpy(c, k, dst_k);
        cpy_v[i] = ggml_cpy(c, v, dst_v);
        ggml_tensor * qh = ggml_reshape_4d(c, ggml_cont(c, ggml_permute(c, q, 0,2,1,3)), HD, Lq, G_, NKV);
        ggml_tensor * kh = ggml_reshape_4d(c, ggml_cont(c, ggml_permute(c, kc, 0,2,1,3)), HD, Lk, 1, NKV);
        ggml_tensor * vh = ggml_reshape_4d(c, ggml_cont(c, ggml_permute(c, vc, 0,2,1,3)), HD, Lk, 1, NKV);
        ggml_tensor * sc = ggml_mul_mat(c, kh, qh);          // [Lk, Lq, G, NKV]
        sc = ggml_soft_max_ext(c, sc, mask, scale, 0.0f);
        ggml_tensor * vt = ggml_cont(c, ggml_permute(c, vh, 1,0,2,3)); // [Lk, HD, 1, NKV]
        ggml_tensor * o = ggml_mul_mat(c, vt, sc);           // [HD, Lq, G, NKV]
        o = ggml_cont(c, ggml_permute(c, o, 0,3,1,2));       // [HD, G, NKV, Lq]
        o = ggml_reshape_2d(c, o, static_cast<int64_t>(HD) * NH, Lq);
        o = ggml_mul_mat(c, G(m, lb(i,"o_proj/weight")), o);
        xx = ggml_add(c, xx, o);
        ggml_tensor * hn = rmsnorm(c, xx, G(m, lb(i,"post_ln/weight")), hp.eps);
        ggml_tensor * gate = ggml_silu(c, ggml_mul_mat(c, G(m,lb(i,"gate/weight")), hn));
        ggml_tensor * up   = ggml_mul_mat(c, G(m,lb(i,"up/weight")), hn);
        ggml_tensor * down = ggml_mul_mat(c, G(m,lb(i,"down/weight")), ggml_mul(c, gate, up));
        xx = ggml_add(c, xx, down);
    }
    xx = rmsnorm(c, xx, G(m, "lm/norm/weight"), hp.eps);
    ggml_tensor * logits = ggml_mul_mat(c, G(m, "lm/llm_decoder/weight"), xx); // [VS, Lq]
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    for (int i = 0; i < hp.depth; ++i) { ggml_build_forward_expand(gf, cpy_k[i]); ggml_build_forward_expand(gf, cpy_v[i]); }

    // `al` is owned by the caller and reused across the whole decode: creating
    // and destroying an allocator per token means a backend buffer alloc/free
    // per token, which on a GPU is a driver round-trip plus heap churn.
    // ggml_gallocr_reserve only re-allocates when a chunk grows, and the
    // prefill graph (Lq = L0) is the largest, so the buffer settles on step 0.
    bool use_sched = false;
    if (!cosy_dispatch_prepare(m, gf, al, nmax, use_sched, "cosyvoice_lm")) {
        ggml_free(c);
        throw std::runtime_error("cosyvoice: LM graph dispatch failed");
    }

    ggml_backend_tensor_set(x, x_new, 0, (size_t)Lq * D * 4);
    { std::vector<int32_t> pv_(Lq); for (int i = 0; i < Lq; ++i) pv_[i] = P + i; ggml_backend_tensor_set(pos, pv_.data(), 0, pv_.size() * 4); }
    { std::vector<float> mk((size_t)Lk * Lq); for (int j = 0; j < Lq; ++j) for (int kk = 0; kk < Lk; ++kk) mk[(size_t)j * Lk + kk] = (kk <= P + j) ? 0.f : -INFINITY;
      ggml_backend_tensor_set(mask, mk.data(), 0, mk.size() * 4); }
    // Appends this step's K/V into the resident cache.
    if (!cosy_dispatch_compute(m, gf, use_sched, "cosyvoice_lm")) {
        ggml_free(c);
        throw std::runtime_error("cosyvoice: LM compute failed");
    }

    std::vector<float> out(VS);
    ggml_backend_tensor_get(logits, out.data(), (size_t)(Lq - 1) * VS * 4, (size_t)VS * 4);
    cache.P = Lk;
    ggml_free(c);
    return out;
}

static void log_softmax_inplace(std::vector<float> & v) {
    double mx = -1e30; for (float f : v) mx = std::max(mx, (double)f);
    double se = 0; for (float f : v) se += std::exp(f - mx); double lse = mx + std::log(se);
    for (float & f : v) f = (float)(f - lse);
}

// RAS sampling (nucleus top-p/top-k + repetition-aware fallback). `logp` = log-softmax.
static int ras_sample(const std::vector<float> & logp, const std::vector<int> & decoded,
                      std::mt19937 & rng, int speech_token_size, bool greedy) {
    int VS = (int)logp.size();
    if (greedy) { int a = 0; for (int i = 1; i < VS; ++i) if (logp[i] > logp[a]) a = i; return a; }
    std::vector<float> prob(VS); for (int i = 0; i < VS; ++i) prob[i] = std::exp(logp[i]);
    std::vector<int> order(VS); for (int i = 0; i < VS; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return prob[a] > prob[b]; });
    std::vector<int> idx; std::vector<double> pr; double cum = 0;
    for (int i = 0; i < VS; ++i) { if (cum < 0.8 && (int)idx.size() < 25) { cum += prob[order[i]]; pr.push_back(prob[order[i]]); idx.push_back(order[i]); } else break; }
    auto multinomial = [&](const std::vector<double> & w, const std::vector<int> & ids) -> int {
        double s = 0; for (double x : w) s += x; std::uniform_real_distribution<double> U(0, s);
        double r = U(rng), acc = 0; for (size_t i = 0; i < w.size(); ++i) { acc += w[i]; if (r <= acc) return ids[i]; } return ids.back(); };
    int top = multinomial(pr, idx);
    int win = 10; int rep = 0; for (int i = std::max(0, (int)decoded.size() - win); i < (int)decoded.size(); ++i) if (decoded[i] == top) rep++;
    if (rep >= 1) {
        std::vector<double> p2(prob.begin(), prob.end()); p2[top] = 0; std::vector<int> all(VS); for (int i = 0; i < VS; ++i) all[i] = i;
        top = multinomial(p2, all);
    }
    (void)speech_token_size;
    return top;
}

// Build lm_input = [sos_emb, embed_tokens(text_ids), task_id_emb, speech_embedding(prompt_stok)]
// as a flat [D, L] (each position's D-vec contiguous).
static std::vector<float> build_lm_input(model_ctx & m, const std::vector<int> & text_ids,
                                         const std::vector<int> & prompt_stok, int & L_out, int & D_out) {
    ggml_tensor * et = G(m, "lm/embed_tokens/weight");
    ggml_tensor * se = G(m, "lm/speech_embedding/weight");
    int D = (int)et->ne[0];
    const float * etd = host_rows(et, "build_lm_input");
    const float * sed = host_rows(se, "build_lm_input");
    const int SOS  = (int)cosyvoice_meta_i(m, "cosyvoice3.llm.sos",     6561);
    const int TASK = (int)cosyvoice_meta_i(m, "cosyvoice3.llm.task_id", 6563);
    std::vector<float> seq;
    auto push_row = [&](const float * base, int row) { seq.insert(seq.end(), base + (size_t)row * D, base + (size_t)(row + 1) * D); };
    push_row(sed, SOS);
    for (int t : text_ids) push_row(etd, t);
    push_row(sed, TASK);
    for (int t : prompt_stok) push_row(sed, t);
    L_out = (int)(seq.size() / D); D_out = D;
    return seq;
}

std::vector<int> cosyvoice_llm_generate(model_ctx & m, const qwen_hp & hp,
                                        const std::vector<int> & text_ids,
                                        const std::vector<int> & prompt_stok,
                                        int max_steps, bool greedy, int seed, int min_len,
                                        cosyvoice_timings * tmg) {
    // VS (LM output size) comes straight from the speech head's weight so the
    // graph can't disagree with the tensor; STS (speech_token_size, the EOS
    // threshold) from KV with the historical value as fallback.
    const int VS  = (int)G(m, "lm/llm_decoder/weight")->ne[1];
    const int STS = (int)cosyvoice_meta_i(m, "cosyvoice3.llm.speech_token_size", 6561);
    int L0 = 0, D = 0;
    std::vector<float> seq = build_lm_input(m, text_ids, prompt_stok, L0, D);
    ggml_tensor * se = G(m, "lm/speech_embedding/weight");
    const float * spk_emb = host_rows(se, "cosyvoice_llm_generate");
    int SE_D = (int)se->ne[0];
    std::vector<int> tokens;
    std::mt19937 rng(seed);
    // Cache holds the L0 prefill positions plus up to max_steps decode positions.
    qwen_kvcache cache; cache.init(m, hp, L0 + max_steps + 1);
    // One allocator for the entire decode (see the note in qwen_step_kv).
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!al) { cache.free(); throw std::runtime_error("cosyvoice: gallocr alloc failed (LM)"); }

    // Prefill the prompt (sos + text + task + prompt speech tokens) in one pass;
    // its last-position logits give the step-0 distribution.
    auto t_pre = cosy_clk::now();
    std::vector<float> logits = qwen_step_kv(m, hp, seq.data(), L0, D, VS, cache, al);
    if (tmg) tmg->lm_prefill_ms += cosy_ms_since(t_pre, m.backend);
    for (int step = 0; step < max_steps; ++step) {
        log_softmax_inplace(logits);
        if (step < min_len) for (int t = STS; t < VS; ++t) logits[t] = -1e30f;
        int tok = ras_sample(logits, tokens, rng, STS, greedy);
        if (tok >= STS) break;
        tokens.push_back(tok);
        // Decode one step: feed this token's speech_embedding at the next position.
        std::vector<float> x1(spk_emb + (size_t)tok * SE_D, spk_emb + (size_t)(tok + 1) * SE_D);
        auto t_dec = cosy_clk::now();
        logits = qwen_step_kv(m, hp, x1.data(), 1, D, VS, cache, al);
        if (tmg) { tmg->lm_decode_ms += cosy_ms_since(t_dec, m.backend); tmg->n_decode_steps++; }
    }
    if (tmg) tmg->n_speech_tokens = (int) tokens.size();
    ggml_gallocr_free(al);
    cache.free();
    return tokens;
}

// ===========================================================================
// Stage 4 — DiT flow
// ===========================================================================
static const char * P = "flow/";
static std::string bp(int i, const std::string & s) { return std::string(P) + "blk/" + std::to_string(i) + "/" + s; }

ggml_tensor * build_dit(ggml_context * c, const model_ctx & m, const dit_hp & hp,
                        ggml_tensor * x, ggml_tensor * mu, ggml_tensor * cond,
                        ggml_tensor * spks, ggml_tensor * time_sin, ggml_tensor * pos,
                        int N, int B) {
    ggml_tensor * t = linear(c, T(m, std::string(P) + "time_embed/time_mlp/0/weight"),
                                T(m, std::string(P) + "time_embed/time_mlp/0/bias"), time_sin);
    t = silu(c, t);
    t = linear(c, T(m, std::string(P) + "time_embed/time_mlp/2/weight"),
                  T(m, std::string(P) + "time_embed/time_mlp/2/bias"), t);

    ggml_tensor * spks_rep = ggml_repeat(c, ggml_reshape_3d(c, spks, spks->ne[0], 1, B),
                                         ggml_new_tensor_3d(c, GGML_TYPE_F32, spks->ne[0], N, B));
    ggml_tensor * cat = ggml_concat(c, ggml_concat(c, ggml_concat(c, x, cond, 0), mu, 0), spks_rep, 0);
    ggml_tensor * h = linear(c, T(m, std::string(P) + "input_embed/proj/weight"),
                                T(m, std::string(P) + "input_embed/proj/bias"), cat);
    {
        ggml_tensor * xt = ggml_cont(c, ggml_permute(c, h, 1, 0, 2, 3));
        for (int layer = 0; layer < 2; ++layer) {
            std::string pfx = std::string(P) + "input_embed/conv_pos_embed/conv" + std::to_string(layer + 1) + "/0/";
            xt = ggml_pad_ext(c, xt, hp.conv_k - 1, 0, 0, 0, 0, 0, 0, 0);
            xt = conv1d_grouped(c, T(m, pfx + "weight"), xt, hp.conv_groups);
            ggml_tensor * b = T(m, pfx + "bias");
            xt = ggml_add(c, xt, ggml_reshape_3d(c, b, 1, b->ne[0], 1));
            xt = mish(c, xt);
        }
        ggml_tensor * cpos = ggml_cont(c, ggml_permute(c, xt, 1, 0, 2, 3));
        h = ggml_add(c, h, cpos);
    }

    const float attn_scale = 1.0f / std::sqrt((float)hp.dim_head);
    for (int i = 0; i < hp.depth; ++i) {
        ggml_tensor * emb = linear(c, T(m, bp(i, "attn_norm/linear/weight")),
                                      T(m, bp(i, "attn_norm/linear/bias")), silu(c, t));
        auto chunk = [&](int idx) {
            return ggml_cont(c, ggml_view_2d(c, emb, hp.dim, B, emb->nb[1], (size_t)idx * hp.dim * emb->nb[0]));
        };
        ggml_tensor * shift_msa = chunk(0), * scale_msa = chunk(1), * gate_msa = chunk(2);
        ggml_tensor * shift_mlp = chunk(3), * scale_mlp = chunk(4), * gate_mlp = chunk(5);

        ggml_tensor * norm = adaln_modulate(c, ln_noaffine(c, h), scale_msa, shift_msa);

        ggml_tensor * q = linear(c, T(m, bp(i, "attn/to_q/weight")), T(m, bp(i, "attn/to_q/bias")), norm);
        ggml_tensor * k = linear(c, T(m, bp(i, "attn/to_k/weight")), T(m, bp(i, "attn/to_k/bias")), norm);
        ggml_tensor * v = linear(c, T(m, bp(i, "attn/to_v/weight")), T(m, bp(i, "attn/to_v/bias")), norm);
        auto rope_head0 = [&](ggml_tensor * z) {
            ggml_tensor * h0 = ggml_cont(c, ggml_view_3d(c, z, hp.dim_head, N, B, z->nb[1], z->nb[2], 0));
            h0 = ggml_reshape_4d(c, h0, hp.dim_head, 1, N, B);
            h0 = ggml_rope_ext(c, h0, pos, nullptr, hp.dim_head, GGML_ROPE_TYPE_NORMAL, 0,
                               10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            h0 = ggml_reshape_3d(c, h0, hp.dim_head, N, B);
            ggml_tensor * rest = ggml_cont(c, ggml_view_3d(c, z, hp.dim - hp.dim_head, N, B,
                                          z->nb[1], z->nb[2], (size_t)hp.dim_head * z->nb[0]));
            return ggml_concat(c, h0, rest, 0);
        };
        q = rope_head0(q);
        k = rope_head0(k);
        q = ggml_reshape_4d(c, q, hp.dim_head, hp.heads, N, B);
        k = ggml_reshape_4d(c, k, hp.dim_head, hp.heads, N, B);
        v = ggml_reshape_4d(c, v, hp.dim_head, hp.heads, N, B);
        q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
        k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
        v = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));
        ggml_tensor * scores = ggml_mul_mat(c, k, q);
        scores = ggml_soft_max_ext(c, scores, nullptr, attn_scale, 0.0f);
        ggml_tensor * vt = ggml_cont(c, ggml_permute(c, v, 1, 0, 2, 3));
        ggml_tensor * o = ggml_mul_mat(c, vt, scores);
        o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));
        o = ggml_reshape_3d(c, o, hp.dim, N, B);
        o = linear(c, T(m, bp(i, "attn/to_out/0/weight")), T(m, bp(i, "attn/to_out/0/bias")), o);

        h = ggml_add(c, h, ggml_mul(c, o, ggml_reshape_3d(c, gate_msa, hp.dim, 1, B)));

        ggml_tensor * fn = adaln_modulate(c, ln_noaffine(c, h), scale_mlp, shift_mlp);
        ggml_tensor * ff = linear(c, T(m, bp(i, "ff/ff/0/0/weight")), T(m, bp(i, "ff/ff/0/0/bias")), fn);
        ff = ggml_gelu(c, ff);
        ff = linear(c, T(m, bp(i, "ff/ff/2/weight")), T(m, bp(i, "ff/ff/2/bias")), ff);
        h = ggml_add(c, h, ggml_mul(c, ff, ggml_reshape_3d(c, gate_mlp, hp.dim, 1, B)));
    }

    ggml_tensor * embf = linear(c, T(m, std::string(P) + "norm_out/linear/weight"),
                                   T(m, std::string(P) + "norm_out/linear/bias"), silu(c, t));
    ggml_tensor * scale_f = ggml_cont(c, ggml_view_2d(c, embf, hp.dim, B, embf->nb[1], 0));
    ggml_tensor * shift_f = ggml_cont(c, ggml_view_2d(c, embf, hp.dim, B, embf->nb[1], (size_t)hp.dim * embf->nb[0]));
    h = adaln_modulate(c, ln_noaffine(c, h), scale_f, shift_f);
    h = linear(c, T(m, std::string(P) + "proj_out/weight"), T(m, std::string(P) + "proj_out/bias"), h);
    ggml_set_name(h, "dit_out"); ggml_set_output(h);
    return h;
}

std::vector<float> sinus_time_emb(const std::vector<float> & t, int dim) {
    int half = dim / 2, B = (int)t.size();
    std::vector<float> out((size_t)dim * B, 0.0f);
    double logk = std::log(10000.0) / (half - 1);
    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < half; ++j) {
            double freq = std::exp(j * -logk);
            double a = 1000.0 * (double)t[b] * freq;
            out[(size_t)b * dim + j]        = (float)std::sin(a);
            out[(size_t)b * dim + half + j] = (float)std::cos(a);
        }
    }
    return out;
}

// Flow front-end (DiT graph A): upsampled token features mu and the affine-
// projected speaker vector.  tokids = prompt++speech token ids; on return
// mu_host is [MEL,TM] channel-major-flattened (mel-fastest) and spks_host[MEL].
static void build_flow_frontend(model_ctx & m, const std::vector<int32_t> & tokids,
                                const std::vector<float> & embedding,
                                int T_tok, int TM, int SPK,
                                std::vector<float> & mu_host, std::vector<float> & spks_host) {
    const int MEL = 80;
    mu_host.assign((size_t)MEL * TM, 0.f);
    spks_host.assign(MEL, 0.f);
    ggml_init_params gp = cosy_arena(kCosyFlowFrontendNodes);
    ggml_context * c = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(c, kCosyFlowFrontendNodes, false);

    ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, T_tok); ggml_set_name(ids, "ids"); ggml_set_input(ids);
    ggml_tensor * emb1d = ggml_new_tensor_1d(c, GGML_TYPE_F32, SPK); ggml_set_name(emb1d, "emb"); ggml_set_input(emb1d);

    ggml_tensor * e = ggml_get_rows(c, T(m, "flow/input_embedding/weight"), ids);
    ggml_tensor * x = ggml_cont(c, ggml_permute(c, e, 1, 0, 2, 3));
    ggml_tensor * res = x;
    ggml_tensor * xp = ggml_pad_ext(c, x, 0, 3, 0, 0, 0, 0, 0, 0);
    ggml_tensor * h = conv1d_f32(c, T(m, "flow/pre_lookahead_layer/conv1/weight"), xp, 1, 0, 1);
    h = ggml_add(c, h, ggml_reshape_2d(c, T(m, "flow/pre_lookahead_layer/conv1/bias"), 1, 1024));
    h = ggml_leaky_relu(c, h, 0.01f, false);
    ggml_tensor * hpad = ggml_pad_ext(c, h, 2, 0, 0, 0, 0, 0, 0, 0);
    h = conv1d_f32(c, T(m, "flow/pre_lookahead_layer/conv2/weight"), hpad, 1, 0, 1);
    h = ggml_add(c, h, ggml_reshape_2d(c, T(m, "flow/pre_lookahead_layer/conv2/bias"), 1, MEL));
    h = ggml_add(c, h, res);
    ggml_tensor * h3 = ggml_reshape_3d(c, h, 1, T_tok, MEL);
    ggml_tensor * h2 = ggml_repeat(c, h3, ggml_new_tensor_3d(c, GGML_TYPE_F32, 2, T_tok, MEL));
    ggml_tensor * up = ggml_reshape_2d(c, ggml_cont(c, h2), TM, MEL);
    ggml_tensor * mu = ggml_cont(c, ggml_permute(c, up, 1, 0, 2, 3));
    ggml_set_name(mu, "mu"); ggml_set_output(mu);

    ggml_tensor * n = ggml_rms_norm(c, emb1d, 1e-12f);
    n = ggml_scale(c, n, 1.0f / std::sqrt((float)SPK));
    ggml_tensor * sp = ggml_mul_mat(c, T(m, "flow/spk_embed_affine_layer/weight"), n);
    sp = ggml_add(c, sp, T(m, "flow/spk_embed_affine_layer/bias"));
    ggml_set_name(sp, "spks"); ggml_set_output(sp);

    ggml_build_forward_expand(gf, mu);
    ggml_build_forward_expand(gf, sp);
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    bool use_sched = false;
    if (!al || !cosy_dispatch_prepare(m, gf, al, kCosyFlowFrontendNodes, use_sched, "cosyvoice_flow_frontend")) {
        if (al) ggml_gallocr_free(al);
        ggml_free(c);
        throw std::runtime_error("cosyvoice: flow front-end graph dispatch failed");
    }
    ggml_backend_tensor_set(ids, tokids.data(), 0, tokids.size() * 4);
    ggml_backend_tensor_set(emb1d, embedding.data(), 0, (size_t)SPK * 4);
    if (!cosy_dispatch_compute(m, gf, use_sched, "cosyvoice_flow_frontend")) {
        ggml_gallocr_free(al); ggml_free(c);
        throw std::runtime_error("cosyvoice: flow front-end compute failed");
    }
    ggml_backend_tensor_get(mu, mu_host.data(), 0, mu_host.size() * 4);
    ggml_backend_tensor_get(sp, spks_host.data(), 0, spks_host.size() * 4);
    ggml_gallocr_free(al); ggml_free(c);
}

// CausalConditionalCFM: 10-step Euler integration of the DiT estimator with
// classifier-free guidance (batch-2: conditional + unconditional).  Integrates
// x_host [MEL,TM] channel-major-flattened in place over the cosine time schedule.
static void run_euler_steps(model_ctx & m, const dit_hp & hp,
                            const std::vector<float> & mu_host,
                            const std::vector<float> & cond_host,
                            const std::vector<float> & spks_host, int TM,
                            std::vector<float> & x_host) {
    const int MEL = 80;
    int B = 2, N = TM;
    const size_t nmax = cosy_dit_nodes(hp);
    ggml_context * c = nullptr;
    ggml_cgraph  * gf = nullptr;
    ggml_tensor  * x = nullptr, * mu = nullptr, * cnd = nullptr;
    ggml_tensor  * spks = nullptr, * tsin = nullptr, * pos = nullptr, * out = nullptr;
    // The graph is shape-invariant across the 10 Euler steps, so the direct path
    // builds it once and replays it.  The sched path cannot: sched graphs are
    // single-use for allocation, and replaying one computes garbage.  Rebuilding
    // is cheap (a few thousand tensor structs into a ~2 MiB arena), so the sched
    // branch just rebuilds per step rather than giving up graph reuse anywhere.
    auto build = [&]() {
        if (c) ggml_free(c);
        c  = ggml_init(cosy_arena(nmax));
        gf = ggml_new_graph_custom(c, nmax, false);
        x    = ggml_new_tensor_3d(c, GGML_TYPE_F32, MEL, N, B); ggml_set_name(x, "x");       ggml_set_input(x);
        mu   = ggml_new_tensor_3d(c, GGML_TYPE_F32, MEL, N, B); ggml_set_name(mu, "mu");     ggml_set_input(mu);
        cnd  = ggml_new_tensor_3d(c, GGML_TYPE_F32, MEL, N, B); ggml_set_name(cnd, "cond");  ggml_set_input(cnd);
        spks = ggml_new_tensor_2d(c, GGML_TYPE_F32, MEL, B);    ggml_set_name(spks, "spks"); ggml_set_input(spks);
        tsin = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, B);    ggml_set_name(tsin, "tsin"); ggml_set_input(tsin);
        pos  = ggml_new_tensor_1d(c, GGML_TYPE_I32, N);         ggml_set_name(pos, "pos");   ggml_set_input(pos);
        out  = build_dit(c, m, hp, x, mu, cnd, spks, tsin, pos, N, B);
        ggml_build_forward_expand(gf, out);
    };
    build();
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    bool use_sched = false;
    if (!al || !cosy_dispatch_prepare(m, gf, al, nmax, use_sched, "cosyvoice_dit")) {
        if (al) ggml_gallocr_free(al);
        ggml_free(c);
        throw std::runtime_error("cosyvoice: DiT graph dispatch failed");
    }

    std::vector<float> mu_in((size_t)MEL * N * B, 0.f);  memcpy(mu_in.data(), mu_host.data(), (size_t)MEL * N * 4);
    std::vector<float> cnd_in((size_t)MEL * N * B, 0.f); memcpy(cnd_in.data(), cond_host.data(), (size_t)MEL * N * 4);
    std::vector<float> spks_in((size_t)MEL * B, 0.f);    memcpy(spks_in.data(), spks_host.data(), (size_t)MEL * 4);
    std::vector<int32_t> pos_in(N); for (int i = 0; i < N; ++i) pos_in[i] = i;

    std::vector<float> tspan(11);
    for (int i = 0; i <= 10; ++i) { float u = (float)i / 10.0f; tspan[i] = 1.0f - std::cos(u * 0.5f * (float)M_PI); }
    const float cfg = 0.7f;
    std::vector<float> dphi((size_t)MEL * N * B);
    std::vector<float> xin((size_t)MEL * N * B);
    for (int step = 0; step < 10; ++step) {
        if (use_sched && step > 0) {
            build();   // fresh graph for every sched pass (single-use allocation)
            bool again = false;
            if (!cosy_dispatch_prepare(m, gf, al, nmax, again, "cosyvoice_dit")) {
                ggml_gallocr_free(al); ggml_free(c);
                throw std::runtime_error("cosyvoice: DiT graph dispatch failed");
            }
        }
        float t = tspan[step], dt = tspan[step + 1] - tspan[step];
        memcpy(xin.data(), x_host.data(), (size_t)MEL * N * 4);
        memcpy(xin.data() + (size_t)MEL * N, x_host.data(), (size_t)MEL * N * 4);
        ggml_backend_tensor_set(x, xin.data(), 0, xin.size() * 4);
        std::vector<float> tv = { t, t };
        std::vector<float> se = sinus_time_emb(tv, 256);
        ggml_backend_tensor_set(tsin, se.data(), 0, se.size() * 4);
        ggml_backend_tensor_set(mu,   mu_in.data(),   0, mu_in.size() * 4);
        ggml_backend_tensor_set(cnd,  cnd_in.data(),  0, cnd_in.size() * 4);
        ggml_backend_tensor_set(spks, spks_in.data(), 0, spks_in.size() * 4);
        ggml_backend_tensor_set(pos,  pos_in.data(),  0, pos_in.size() * 4);
        if (!cosy_dispatch_compute(m, gf, use_sched, "cosyvoice_dit")) {
            ggml_gallocr_free(al); ggml_free(c);
            throw std::runtime_error("cosyvoice: DiT compute failed");
        }
        ggml_backend_tensor_get(out, dphi.data(), 0, dphi.size() * 4);
        for (size_t i = 0; i < (size_t)MEL * N; ++i) {
            float dc = dphi[i], du = dphi[i + (size_t)MEL * N];
            float d = (1.0f + cfg) * dc - cfg * du;
            x_host[i] += dt * d;
        }
    }
    ggml_gallocr_free(al); ggml_free(c);
}

// Drop the prompt frames: mel = x_host[:, mel_len1:] laid out [MEL, mel_len2]
// channel-major (mel[ch*T + t]), the layout the HiFT stage consumes.
static std::vector<float> trim_prompt_mel(const std::vector<float> & x_host, int mel_len1, int mel_len2) {
    const int MEL = 80;
    std::vector<float> mel((size_t)MEL * mel_len2);
    for (int ch = 0; ch < MEL; ++ch) for (int t = 0; t < mel_len2; ++t)
        mel[(size_t)ch * mel_len2 + t] = x_host[(size_t)ch + (size_t)(t + mel_len1) * MEL];
    return mel;
}

std::vector<float> cosyvoice_flow_run(model_ctx & m,
                                      const std::vector<int> & prompt_token,
                                      const std::vector<int> & speech_tokens,
                                      const std::vector<float> & prompt_feat, int mel_len1,
                                      const std::vector<float> & embedding, int & out_mel_len,
                                      cosyvoice_timings * tmg) {
    dit_hp hp;
    const int MEL = 80;
    int T_ptok = (int)prompt_token.size(), T_stok = (int)speech_tokens.size();
    int T_tok = T_ptok + T_stok;
    int TM = T_tok * 2;                       // token_mel_ratio = 2
    int mel_len2 = TM - mel_len1;
    int SPK = (int)embedding.size();

    std::vector<int32_t> tokids(T_tok);
    for (int i = 0; i < T_ptok; ++i) tokids[i] = prompt_token[i];
    for (int i = 0; i < T_stok; ++i) tokids[T_ptok + i] = speech_tokens[i];

    // Front-end graph A: tokens + embedding -> mu[80,TM], spks[80].
    std::vector<float> mu_host, spks_host;
    auto t_fe = cosy_clk::now();
    build_flow_frontend(m, tokids, embedding, T_tok, TM, SPK, mu_host, spks_host);
    if (tmg) { tmg->flow_frontend_ms += cosy_ms_since(t_fe, m.backend); tmg->tm = TM; }

    // cond[80,T]: first mel_len1 columns = prompt_feat, rest 0
    std::vector<float> cond_host((size_t)MEL * TM, 0.f);
    memcpy(cond_host.data(), prompt_feat.data(), (size_t)mel_len1 * MEL * 4);

    // noise z[80,T] from baked rand_noise (ne=[15000,80] => rn[a,b]=rand_noise[ch=b,time=a])
    ggml_tensor * rnt = T(m, "flow/rand_noise");
    std::vector<float> rn_host(ggml_nelements(rnt));
    ggml_backend_tensor_get(rnt, rn_host.data(), 0, rn_host.size() * 4);
    int RNW = (int)rnt->ne[0];
    // The baked rand_noise is [RNW=15000, 80]; the gather below reads time index
    // t in [0,TM). PyTorch caps the CFM at the same 15000 frames, so refuse
    // (rather than over-read the buffer) when the requested mel length exceeds
    // what the baked noise covers — ~5 min of speech at token_mel_ratio=2.
    if (TM > RNW) {
        throw std::runtime_error(
            "cosyvoice_flow_run: requested mel length " + std::to_string(TM) +
            " exceeds baked rand_noise frames " + std::to_string(RNW) +
            " (input too long; max ~" + std::to_string(RNW / 2) + " tokens)");
    }
    std::vector<float> x_host((size_t)MEL * TM);
    for (int ch = 0; ch < MEL; ++ch) for (int t = 0; t < TM; ++t) x_host[(size_t)ch + (size_t)t * MEL] = rn_host[(size_t)t + (size_t)ch * RNW];

    // DiT graph B: 10 Euler steps (integrates x_host in place).
    auto t_dit = cosy_clk::now();
    run_euler_steps(m, hp, mu_host, cond_host, spks_host, TM, x_host);
    if (tmg) tmg->dit_euler_ms += cosy_ms_since(t_dit, m.backend);

    // trim prompt part -> [80, mel_len2] channel-major
    std::vector<float> mel = trim_prompt_mel(x_host, mel_len1, mel_len2);
    out_mel_len = mel_len2;
    if (tmg) tmg->mel_len = mel_len2;
    return mel;
}

// ===========================================================================
// Stage 3 — CausalHiFT vocoder
// ===========================================================================
static ggml_tensor * snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * inv_alpha) {
    ggml_tensor * a  = ggml_reshape_2d(ctx, alpha,     1, alpha->ne[0]);
    ggml_tensor * ia = ggml_reshape_2d(ctx, inv_alpha, 1, inv_alpha->ne[0]);
    ggml_tensor * ax = ggml_mul(ctx, x, a);
    ggml_tensor * s  = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    return ggml_add(ctx, x, ggml_mul(ctx, s2, ia));
}
static std::vector<float> invert_alpha_cpu(const model_ctx & m, const std::string & name) {
    ggml_tensor * t = T(m, name);
    std::vector<float> a(ggml_nelements(t));
    ggml_backend_tensor_get(t, a.data(), 0, ggml_nbytes(t));
    std::vector<float> inv(a.size());
    for (size_t i = 0; i < a.size(); ++i) inv[i] = 1.0f / (a[i] + 1e-9f);
    return inv;
}
static ggml_tensor * reflect_pad_1d(ggml_context * ctx, ggml_tensor * x, int p_left, int p_right) {
    ggml_tensor * y = x;
    for (int i = 0; i < p_left; ++i) {
        int src_idx = p_left - i;
        ggml_tensor * s = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], (size_t)src_idx * x->nb[0]);
        s = ggml_cont(ctx, s);
        y = ggml_concat(ctx, s, y, 0);
    }
    int L_orig = (int)x->ne[0];
    for (int i = 0; i < p_right; ++i) {
        int src_idx = L_orig - 2 - i;
        ggml_tensor * s = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], (size_t)src_idx * x->nb[0]);
        s = ggml_cont(ctx, s);
        y = ggml_concat(ctx, y, s, 0);
    }
    return y;
}
static std::vector<float> build_hann_window(int n, bool periodic = true) {
    std::vector<float> w(n);
    double N = periodic ? (double)n : (double)(n - 1);
    const double two_pi = 2.0 * M_PI;
    for (int i = 0; i < n; ++i) w[i] = (float)(0.5 * (1.0 - std::cos(two_pi * (double)i / N)));
    return w;
}
static std::vector<float> build_stft_kernel(int n_fft, const std::vector<float> & window) {
    int F = n_fft / 2 + 1;
    std::vector<float> K((size_t)n_fft * 1 * (2 * F), 0.0f);
    const double two_pi = 2.0 * M_PI;
    for (int f = 0; f < F; ++f) for (int n = 0; n < n_fft; ++n) {
        double th = two_pi * f * n / n_fft; float w = window[n];
        K[n + f       * n_fft] = (float)(std::cos(th) * w);
        K[n + (F + f) * n_fft] = (float)(-std::sin(th) * w);
    }
    return K;
}
static std::vector<float> build_istft_kernel(int n_fft, const std::vector<float> & window) {
    int F = n_fft / 2 + 1;
    std::vector<float> K((size_t)n_fft * 1 * (2 * F), 0.0f);
    const double two_pi = 2.0 * M_PI;
    const double inv_N = 1.0 / (double)n_fft;
    for (int f = 0; f < F; ++f) {
        double coef_re = (f == 0 || f == n_fft / 2) ? 1.0 : 2.0;
        double coef_im = (f == 0 || f == n_fft / 2) ? 0.0 : 2.0;
        for (int n = 0; n < n_fft; ++n) {
            double th = two_pi * f * n / n_fft; float w = window[n];
            K[n + f       * n_fft] = (float)(coef_re * std::cos(th) * w * inv_N);
            K[n + (F + f) * n_fft] = (float)(-coef_im * std::sin(th) * w * inv_N);
        }
    }
    return K;
}
static std::vector<float> build_window_sum(int T_stft, int n_fft, int hop, const std::vector<float> & window) {
    int L = (T_stft - 1) * hop + n_fft;
    std::vector<float> ws(L, 0.0f);
    for (int t = 0; t < T_stft; ++t) { int base = t * hop; for (int n = 0; n < n_fft; ++n) ws[base + n] += window[n] * window[n]; }
    return ws;
}
static std::vector<double> linterp(const std::vector<double> & src, int out_size) {
    int in_size = (int)src.size();
    std::vector<double> out(out_size);
    double scale = (double)in_size / (double)out_size;
    for (int i = 0; i < out_size; ++i) {
        double pos = (i + 0.5) * scale - 0.5;
        if (pos < 0) pos = 0;
        if (pos > in_size - 1) pos = in_size - 1;
        int i0 = (int)std::floor(pos);
        int i1 = std::min(i0 + 1, in_size - 1);
        double frac = pos - i0;
        out[i] = src[i0] * (1.0 - frac) + src[i1] * frac;
    }
    return out;
}
static std::vector<float> sinegen2_source(const std::vector<float> & f0_wav,
                                          int sampling_rate, int harmonic_num,
                                          float sine_amp, float noise_std,
                                          float voiced_threshold, int upsample_scale,
                                          const std::vector<float> & l_linear_w,
                                          float l_linear_b, uint32_t seed) {
    int Tn = (int)f0_wav.size();
    int H = harmonic_num + 1;
    int T_down = Tn / upsample_scale;
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<std::vector<float>> sines(H, std::vector<float>(Tn, 0.0f));
    for (int h = 0; h < H; ++h) {
        std::vector<double> rad(Tn);
        double mult = (double)(h + 1) / (double)sampling_rate;
        for (int t = 0; t < Tn; ++t) { double v = (double)f0_wav[t] * mult; rad[t] = v - std::floor(v); }
        std::vector<double> rad_down = linterp(rad, T_down);
        std::vector<double> phase_down(T_down);
        double acc = 0.0;
        for (int i = 0; i < T_down; ++i) { acc += rad_down[i]; phase_down[i] = acc * 2.0 * M_PI * upsample_scale; }
        // CosyVoice3 SineGen2 is CAUSAL: the phase is upsampled with mode='nearest'
        // (staircase), NOT linear.  The vocoder was trained on this exact
        // excitation; linear upsampling puts the source off-distribution and the
        // network renders it with a metallic artifact (wrong harmonic phase).
        for (int t = 0; t < Tn; ++t) {
            int i0 = t / upsample_scale;
            if (i0 >= T_down) i0 = T_down - 1;
            sines[h][t] = (float)std::sin(phase_down[i0]);
        }
    }
    std::vector<float> source(Tn, 0.0f);
    for (int t = 0; t < Tn; ++t) {
        bool voiced = f0_wav[t] > voiced_threshold;
        float uv = voiced ? 1.0f : 0.0f;
        float noise_amp = uv * noise_std + (1.0f - uv) * sine_amp / 3.0f;
        float s = l_linear_b;
        for (int h = 0; h < H; ++h) { float sw = sines[h][t] * sine_amp * uv + noise_amp * gauss(rng); s += l_linear_w[h] * sw; }
        source[t] = std::tanh(s);
    }
    return source;
}

static std::vector<float> run_f0_predictor(model_ctx & m, const std::vector<float> & mel, int T_mel) {
    ggml_init_params gp = cosy_arena(kCosyHiftF0Nodes);
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kCosyHiftF0Nodes, false);
    ggml_tensor * mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, 80);
    ggml_set_name(mel_in, "mel_in"); ggml_set_input(mel_in);
    ggml_tensor * x = mel_in;
    for (int i = 0; i < 5; ++i) {
        std::string pfx = "hift/f0_predictor/condnet/" + std::to_string(i * 2);
        ggml_tensor * w = T(m, pfx + "/weight");
        ggml_tensor * b = T(m, pfx + "/bias");
        int C_out = (int)w->ne[2];
        int K = (int)w->ne[0];
        int pl = (i == 0) ? 0 : (K - 1);
        int pr = (i == 0) ? (K - 1) : 0;
        ggml_tensor * xp = ggml_pad_ext(ctx, x, pl, pr, 0, 0, 0, 0, 0, 0);
        x = conv1d_f32(ctx, w, xp, 1, 0, 1);
        x = ggml_add(ctx, x, ggml_reshape_2d(ctx, b, 1, C_out));
        x = ggml_unary(ctx, x, GGML_UNARY_OP_ELU);
    }
    ggml_tensor * xp = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * cw = T(m, "hift/f0_predictor/classifier/weight");
    ggml_tensor * cb = T(m, "hift/f0_predictor/classifier/bias");
    ggml_tensor * y = ggml_mul_mat(ctx, cw, xp);
    y = ggml_add(ctx, y, cb);
    y = ggml_abs(ctx, y);
    y = ggml_reshape_1d(ctx, y, T_mel);
    ggml_set_name(y, "out"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    bool use_sched = false;
    if (!allocr || !cosy_dispatch_prepare(m, gf, allocr, kCosyHiftF0Nodes, use_sched, "cosyvoice_hift_f0")) {
        if (allocr) ggml_gallocr_free(allocr);
        ggml_free(ctx);
        throw std::runtime_error("cosyvoice: HiFT f0 graph dispatch failed");
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel_in"), mel.data(), 0, mel.size() * sizeof(float));
    if (!cosy_dispatch_compute(m, gf, use_sched, "cosyvoice_hift_f0")) {
        ggml_gallocr_free(allocr); ggml_free(ctx);
        throw std::runtime_error("cosyvoice: HiFT f0 compute failed");
    }
    std::vector<float> f0(T_mel);
    ggml_backend_tensor_get(y, f0.data(), 0, ggml_nbytes(y));
    ggml_gallocr_free(allocr); ggml_free(ctx);
    return f0;
}

static std::vector<float> run_hift_decode(model_ctx & m,
                                          const std::vector<float> & mel, int T_mel,
                                          const std::vector<float> & s_stft, int T_stft) {
    const int MEL = 80;
    const int NFFT2 = 18;
    const int BASE_CH = 512;
    const int n_fft = 16;
    const int hop = 4;
    const int F = n_fft / 2 + 1;
    std::vector<int> ups_rates = {8, 5, 3};
    std::vector<int> ups_ksizes = {16, 11, 7};
    std::vector<int> ups_ch = {256, 128, 64};
    std::vector<int> rb_ksizes = {3, 7, 11};
    std::vector<std::vector<int>> rb_dilations = {{1,3,5},{1,3,5},{1,3,5}};
    std::vector<int> src_rb_ksizes = {7, 7, 11};
    std::vector<std::vector<int>> src_rb_dilations = {{1,3,5},{1,3,5},{1,3,5}};

    ggml_init_params gp = cosy_arena(kCosyHiftDecodeNodes);
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kCosyHiftDecodeNodes, false);

    ggml_tensor * mel_in    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, MEL);    ggml_set_name(mel_in, "mel_in"); ggml_set_input(mel_in);
    ggml_tensor * s_stft_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_stft, NFFT2); ggml_set_name(s_stft_in, "s_stft_in"); ggml_set_input(s_stft_in);

    struct inv_entry { std::string gn; std::vector<float> data; };
    std::vector<inv_entry> inv_alphas;
    auto mk_inv = [&](const std::string & name_pref, int C) {
        std::string gn = "inv_" + name_pref;
        std::vector<float> inv = invert_alpha_cpu(m, name_pref);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        ggml_set_name(t, gn.c_str()); ggml_set_input(t);
        inv_alphas.push_back({gn, std::move(inv)});
        return t;
    };
    auto load_rb = [&](const std::string & prefix, int C) {
        struct rb_data { ggml_tensor *a1, *c1w, *c1b, *a2, *c2w, *c2b, *ia1, *ia2; };
        std::vector<rb_data> p(3);
        for (int i = 0; i < 3; ++i) {
            p[i].a1 = T(m, prefix + "/activations1/" + std::to_string(i) + "/alpha");
            p[i].c1w = T(m, prefix + "/convs1/" + std::to_string(i) + "/weight");
            p[i].c1b = T(m, prefix + "/convs1/" + std::to_string(i) + "/bias");
            p[i].a2 = T(m, prefix + "/activations2/" + std::to_string(i) + "/alpha");
            p[i].c2w = T(m, prefix + "/convs2/" + std::to_string(i) + "/weight");
            p[i].c2b = T(m, prefix + "/convs2/" + std::to_string(i) + "/bias");
            p[i].ia1 = mk_inv(prefix + "/activations1/" + std::to_string(i) + "/alpha", C);
            p[i].ia2 = mk_inv(prefix + "/activations2/" + std::to_string(i) + "/alpha", C);
        }
        return p;
    };
    auto rb_forward = [&](auto & rb, ggml_tensor * x, int C, const std::vector<int> & dils, int k_sz) {
        for (int i = 0; i < 3; ++i) {
            auto & p = rb[i];
            int dilation = dils[i];
            int pad1 = dilation * (k_sz - 1);
            int pad2 = (k_sz - 1);
            ggml_tensor * xt = snake(ctx, x, p.a1, p.ia1);
            xt = ggml_pad_ext(ctx, xt, pad1, 0, 0, 0, 0, 0, 0, 0);
            xt = conv1d_f32(ctx, p.c1w, xt, 1, 0, dilation);
            xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, p.c1b, 1, C));
            xt = snake(ctx, xt, p.a2, p.ia2);
            xt = ggml_pad_ext(ctx, xt, pad2, 0, 0, 0, 0, 0, 0, 0);
            xt = conv1d_f32(ctx, p.c2w, xt, 1, 0, 1);
            xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, p.c2b, 1, C));
            x = ggml_add(ctx, x, xt);
        }
        return x;
    };

    ggml_tensor * cpw = T(m, "hift/conv_pre/weight");
    ggml_tensor * cpb = T(m, "hift/conv_pre/bias");
    ggml_tensor * x = ggml_pad_ext(ctx, mel_in, 0, 4, 0, 0, 0, 0, 0, 0);
    x = conv1d_f32(ctx, cpw, x, 1, 0, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, cpb, 1, BASE_CH));

    for (int i = 0; i < 3; ++i) {
        x = ggml_leaky_relu(ctx, x, 0.1f, false);
        ggml_tensor * uw = T(m, "hift/ups/" + std::to_string(i) + "/weight");
        ggml_tensor * ub = T(m, "hift/ups/" + std::to_string(i) + "/bias");
        int64_t T_up = x->ne[0] * ups_rates[i];
        x = ggml_interpolate(ctx, x, T_up, x->ne[1], x->ne[2], x->ne[3], GGML_SCALE_MODE_NEAREST);
        x = ggml_pad_ext(ctx, x, ups_ksizes[i] - 1, 0, 0, 0, 0, 0, 0, 0);
        x = conv1d_f32(ctx, uw, x, 1, 0, 1);
        x = ggml_add(ctx, x, ggml_reshape_2d(ctx, ub, 1, ups_ch[i]));
        // CausalHiFTGenerator.decode: ReflectionPad1d((1,0)) at the LAST upsample,
        // AFTER the upsample conv and BEFORE the source fusion.  Omitting this
        // 1-sample left-reflection shifts every iSTFT frame's phase by one sample
        // -> a metallic artifact that frame-level log-mel correlation can't see
        // (which is why the 0.989 log-mel parity gate missed it).
        if (i == 2) {
            x = reflect_pad_1d(ctx, x, 1, 0);
        }
        ggml_tensor * sw = T(m, "hift/source_downs/" + std::to_string(i) + "/weight");
        ggml_tensor * sb = T(m, "hift/source_downs/" + std::to_string(i) + "/bias");
        int sd_stride = (i == 0) ? 15 : (i == 1) ? 3 : 1;
        int sd_pad    = sd_stride - 1;
        int sd_oc     = (int)sw->ne[2];
        ggml_tensor * sin_pad = ggml_pad_ext(ctx, s_stft_in, sd_pad, 0, 0, 0, 0, 0, 0, 0);
        ggml_tensor * si = conv1d_f32(ctx, sw, sin_pad, sd_stride, 0, 1);
        si = ggml_add(ctx, si, ggml_reshape_2d(ctx, sb, 1, sd_oc));
        auto srb = load_rb("hift/source_resblocks/" + std::to_string(i), ups_ch[i]);
        si = rb_forward(srb, si, ups_ch[i], src_rb_dilations[i], src_rb_ksizes[i]);
        if (si->ne[0] != x->ne[0]) {
            si = ggml_cont(ctx, ggml_view_3d(ctx, si, x->ne[0], si->ne[1], si->ne[2], si->nb[1], si->nb[2], 0));
        }
        x = ggml_add(ctx, x, si);
        ggml_tensor * xs = nullptr;
        for (int j = 0; j < 3; ++j) {
            auto rb = load_rb("hift/resblocks/" + std::to_string(i * 3 + j), ups_ch[i]);
            ggml_tensor * rb_out = rb_forward(rb, x, ups_ch[i], rb_dilations[j], rb_ksizes[j]);
            xs = (xs == nullptr) ? rb_out : ggml_add(ctx, xs, rb_out);
        }
        x = ggml_scale(ctx, xs, 1.0f / 3.0f);
    }

    x = ggml_leaky_relu(ctx, x, 0.01f, false);
    ggml_tensor * cp2w = T(m, "hift/conv_post/weight");
    ggml_tensor * cp2b = T(m, "hift/conv_post/bias");
    x = ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0);
    x = conv1d_f32(ctx, cp2w, x, 1, 0, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, cp2b, 1, NFFT2));

    int T_out = (int)x->ne[0];
    size_t col_stride = x->nb[1];
    ggml_tensor * mag_log = ggml_cont(ctx, ggml_view_2d(ctx, x, T_out, F, col_stride, 0));
    mag_log = ggml_clamp(ctx, mag_log, -1e6f, 1e2f);
    ggml_tensor * mag = ggml_exp(ctx, mag_log);
    ggml_tensor * ph_in = ggml_cont(ctx, ggml_view_2d(ctx, x, T_out, F, col_stride, (size_t)F * col_stride));
    ggml_tensor * ph = ggml_sin(ctx, ph_in);
    ggml_tensor * real = ggml_mul(ctx, mag, ggml_cos(ctx, ph));
    ggml_tensor * imag = ggml_mul(ctx, mag, ggml_sin(ctx, ph));
    ggml_tensor * spec = ggml_concat(ctx, real, imag, 1);

    auto window = build_hann_window(n_fft, true);
    auto istft_kernel = build_istft_kernel(n_fft, window);
    auto w_sum = build_window_sum(T_out, n_fft, hop, window);

    ggml_tensor * istft_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_fft, 1, 2 * F);
    ggml_set_name(istft_k, "istft_k"); ggml_set_input(istft_k);
    ggml_tensor * ws_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int)w_sum.size(), 1);
    ggml_set_name(ws_in, "w_sum"); ggml_set_input(ws_in);

    ggml_tensor * y = ggml_conv_transpose_1d(ctx, istft_k, spec, hop, 0, 1);
    y = ggml_div(ctx, y, ws_in);
    int pad_amt = n_fft / 2;
    int L_wav = (int)w_sum.size() - n_fft;
    ggml_tensor * y_trim = ggml_cont(ctx, ggml_view_2d(ctx, y, L_wav, y->ne[1], y->nb[1], (size_t)pad_amt * y->nb[0]));
    y_trim = ggml_clamp(ctx, y_trim, -0.99f, 0.99f);
    ggml_set_name(y_trim, "wav"); ggml_set_output(y_trim);
    ggml_build_forward_expand(gf, y_trim);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    // The iSTFT is a CONV_TRANSPOSE_1D, which ggml-opencl does not implement, so
    // on OpenCL this graph takes the sched path and that one node runs on CPU.
    // Built and computed once per utterance -- no graph-reuse conflict.
    bool use_sched = false;
    if (!allocr || !cosy_dispatch_prepare(m, gf, allocr, kCosyHiftDecodeNodes, use_sched, "cosyvoice_hift_decode")) {
        if (allocr) ggml_gallocr_free(allocr);
        ggml_free(ctx);
        throw std::runtime_error("cosyvoice: HiFT decode graph dispatch failed");
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel_in"), mel.data(), 0, mel.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "s_stft_in"), s_stft.data(), 0, s_stft.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "istft_k"), istft_kernel.data(), 0, istft_kernel.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w_sum"), w_sum.data(), 0, w_sum.size() * sizeof(float));
    for (auto & ia : inv_alphas) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, ia.gn.c_str()), ia.data.data(), 0, ia.data.size() * sizeof(float));
    }
    if (!cosy_dispatch_compute(m, gf, use_sched, "cosyvoice_hift_decode")) {
        ggml_gallocr_free(allocr); ggml_free(ctx);
        throw std::runtime_error("cosyvoice: HiFT decode compute failed");
    }
    std::vector<float> wav(ggml_nelements(y_trim));
    ggml_backend_tensor_get(y_trim, wav.data(), 0, ggml_nbytes(y_trim));
    ggml_gallocr_free(allocr); ggml_free(ctx);
    return wav;
}

static std::vector<float> run_stft(model_ctx & m, const std::vector<float> & src) {
    const int n_fft = 16;
    const int hop = 4;
    const int F = n_fft / 2 + 1;
    int T_src = (int)src.size();
    auto window = build_hann_window(n_fft, true);
    auto kernel = build_stft_kernel(n_fft, window);
    ggml_init_params gp = cosy_arena(kCosyStftNodes);
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kCosyStftNodes, false);
    ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_src, 1);
    ggml_set_name(s, "s"); ggml_set_input(s);
    ggml_tensor * s_padded = reflect_pad_1d(ctx, s, n_fft / 2, n_fft / 2);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_fft, 1, 2 * F);
    ggml_set_name(k, "k"); ggml_set_input(k);
    ggml_tensor * spec = conv1d_f32(ctx, k, s_padded, hop, 0, 1);
    ggml_set_name(spec, "spec"); ggml_set_output(spec);
    ggml_build_forward_expand(gf, spec);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    bool use_sched = false;
    if (!allocr || !cosy_dispatch_prepare(m, gf, allocr, kCosyStftNodes, use_sched, "cosyvoice_stft")) {
        if (allocr) ggml_gallocr_free(allocr);
        ggml_free(ctx);
        throw std::runtime_error("cosyvoice: STFT graph dispatch failed");
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "s"), src.data(), 0, src.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "k"), kernel.data(), 0, kernel.size() * sizeof(float));
    if (!cosy_dispatch_compute(m, gf, use_sched, "cosyvoice_stft")) {
        ggml_gallocr_free(allocr); ggml_free(ctx);
        throw std::runtime_error("cosyvoice: STFT compute failed");
    }
    std::vector<float> out(ggml_nelements(spec));
    ggml_backend_tensor_get(spec, out.data(), 0, ggml_nbytes(spec));
    ggml_gallocr_free(allocr); ggml_free(ctx);
    return out;
}

std::vector<float> cosyvoice_hift_synth(model_ctx & m,
                                        const std::vector<float> & mel, int mel_len, int seed,
                                        cosyvoice_timings * tmg) {
    const int T_mel = mel_len;
    const int sampling_rate = kCosyvoiceNativeSampleRate;
    auto t_f0 = cosy_clk::now();
    auto f0 = run_f0_predictor(m, mel, T_mel);
    if (tmg) tmg->hift_f0_ms += cosy_ms_since(t_f0, m.backend);
    int upsample = 8 * 5 * 3 * 4;  // prod(upsample_rates) * hop_len = 480
    int T_wav = T_mel * upsample;
    std::vector<float> f0_up(T_wav);
    for (int i = 0; i < T_mel; ++i) for (int j = 0; j < upsample; ++j) f0_up[i * upsample + j] = f0[i];
    std::vector<float> l_linear_w(9);
    ggml_tensor * llw = T(m, "hift/m_source/l_linear/weight");
    ggml_tensor * llb = T(m, "hift/m_source/l_linear/bias");
    ggml_backend_tensor_get(llw, l_linear_w.data(), 0, 9 * sizeof(float));
    float l_linear_b;
    ggml_backend_tensor_get(llb, &l_linear_b, 0, sizeof(float));
    int harmonic_num = 8;
    float sine_amp = 0.1f, noise_std = 0.003f, voiced_threshold = 10.0f;
    auto t_src = cosy_clk::now();
    auto src = sinegen2_source(f0_up, sampling_rate, harmonic_num,
                               sine_amp, noise_std, voiced_threshold, 480,
                               l_linear_w, l_linear_b, (uint32_t)seed);
    if (tmg) tmg->hift_source_ms += cosy_ms_since(t_src, nullptr);   // pure CPU
    auto t_stft = cosy_clk::now();
    auto s_stft = run_stft(m, src);
    if (tmg) tmg->hift_stft_ms += cosy_ms_since(t_stft, m.backend);
    int T_stft = (int)(s_stft.size() / 18);
    auto t_dec = cosy_clk::now();
    auto wav = run_hift_decode(m, mel, T_mel, s_stft, T_stft);
    if (tmg) tmg->hift_decode_ms += cosy_ms_since(t_dec, m.backend);
    return wav;
}
