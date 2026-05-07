// chatterbox_tts: end-to-end text -> wav synthesis using T3-generated speech
// tokens and a pre-computed speaker/prompt reference dump.
//
// Pipeline (all in C++/ggml):
//   flow_input_tokens = concat(prompt_token, speech_tokens_padded)
//   x = input_embedding(flow_input_tokens)               (T, D=512)
//   mu = encoder(x) + upsample2x + encoder_proj          (2T, 80)
//   spks = affine(F.normalize(embedding))                (80,)
//   conds[:mel_len1] = prompt_feat, rest = 0             (2T, 80)
//   z = randn(80, 2T)
//   for t,r in [(0, 0.5), (0.5, 1.0)]:
//       dxdt = estimator(z, mu, t_emb, spks, conds)
//       z = z + (r-t) * dxdt
//   mel = z[:, mel_len1:]                                (80, 2T - mel_len1)
//   f0 = f0_predictor(mel)
//   source = sinegen(upsample(f0)) -> tanh(linear)       (T_wav,)
//   s_stft = stft(source)
//   wav = hift_decode(mel, s_stft) + istft
//
// Usage:
//   chatterbox_tts --s3gen-gguf MODEL.gguf --ref-dir DIR \
//                  --tokens-file TOKENS.txt --out OUT.wav

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "npy.h"
#include "chatterbox_tts_test_hooks.h"

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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Global thread count (set in main; used to configure CPU backend in each graph run)
static int g_n_threads = 1;

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void compute(ggml_backend_t backend, ggml_cgraph * gf) {
    if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, g_n_threads);
    ggml_backend_graph_compute(backend, gf);
}
struct scoped_timer {
    const char * label;
    double t0;
    scoped_timer(const char * l) : label(l), t0(now_ms()) {}
    ~scoped_timer() { fprintf(stderr, "  [%-16s] %.1f ms\n", label, now_ms() - t0); }
};
#define TIMED(label) scoped_timer _st_##__LINE__(label)

// ============================================================================
// GGUF loader + helpers
// ============================================================================

struct model_ctx {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Variant metadata read from GGUF once at load time.
    //   meanflow=true  : Turbo (2-step Euler, time_embed_mixer, noised_mels overlay,
    //                    no CFG on the CFM side)
    //   meanflow=false : Multilingual (10-step Euler with cosine t_schedule,
    //                    classifier-free-guidance via cfg_rate)
    bool  meanflow    = true;
    int   n_timesteps = 2;
    float cfg_rate    = 0.0f;
};

static ggml_backend_t s3gen_init_backend(int n_gpu_layers, bool verbose) {
#ifdef GGML_USE_CUDA
    if (n_gpu_layers > 0) {
        auto * b = ggml_backend_cuda_init(0);
        if (b) { if (verbose) fprintf(stderr, "s3gen: using CUDA backend\n"); return b; }
    }
#endif
#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        auto * b = ggml_backend_metal_init();
        if (b) { if (verbose) fprintf(stderr, "s3gen: using Metal backend\n"); return b; }
    }
#endif
#ifdef GGML_USE_VULKAN
    if (n_gpu_layers > 0) {
        auto * b = ggml_backend_vk_init(0);
        if (b) {
            if (verbose) {
                char desc[256] = {0};
                ggml_backend_vk_get_device_description(0, desc, sizeof(desc));
                fprintf(stderr, "s3gen: using Vulkan backend (device 0: %s)\n", desc);
            }
            return b;
        }
    }
#endif
#if defined(GGML_USE_OPENCL)
    if (n_gpu_layers > 0) {
        ggml_backend_reg_t ocl_reg = ggml_backend_opencl_reg();
        if (ocl_reg && ggml_backend_reg_dev_count(ocl_reg) > 0) {
            auto * b = ggml_backend_opencl_init();
            if (b) {
                if (verbose) {
                    fprintf(stderr, "s3gen: using OpenCL backend\n");
                }
                return b;
            }
        } else if (verbose && ocl_reg) {
            if (ggml_backend_reg_dev_count(ocl_reg) == 0) {
                fprintf(stderr, "s3gen: no OpenCL device; using CPU\n");
            } else {
                fprintf(stderr, "s3gen: OpenCL init failed; using CPU\n");
            }
        }
    }
#endif
    auto * b = ggml_backend_cpu_init();
    if (!b) throw std::runtime_error("ggml_backend_cpu_init() failed");
    if (verbose) fprintf(stderr, "s3gen: using CPU backend\n");
    return b;
}

// Process-wide cache of the loaded S3Gen GGUF so repeated calls (streaming
// or server mode) pay the ~700 ms tensor-load cost only once.  Keyed on
// (path, n_gpu_layers) — switching backend invalidates the cache.
//
// Thread-safety: simple mutex around load/lookup.  The returned pointer
// stays alive for the lifetime of the process (we never evict), which
// matches the streaming CLI's single-voice use case.  A proper LRU would
// belong in a server front-end.
static model_ctx load_s3gen_gguf(const std::string & path, int n_gpu_layers, bool verbose);

// Tears down every per-synth host-side cache before ggml_backend_free
// runs: the CFM estimator graph cache, the encoder / HiFT / F0 graph
// caches, and the scaffolding caches (pos_emb, inv_alpha, hann_window,
// istft_kernel, window_sum).  Defined later, alongside the cache
// structs themselves.  Forward-declared here so
// s3gen_model_cache_release / cache-miss / s3gen_unload can all call it
// without moving the struct definitions earlier in the file.
static void s3gen_release_synth_caches();

namespace {
struct s3gen_cache_entry { std::string path; int gpu = 0; std::unique_ptr<model_ctx> m; };
static std::mutex                            g_s3gen_cache_mu;
static std::unique_ptr<s3gen_cache_entry>    g_s3gen_cache_entry;
static double                                g_s3gen_cache_last_load_ms = 0.0;

// Refcount over the cache so multi-Engine hosts that share an S3Gen
// GGUF (e.g. two chatterbox::Engine instances backing two
// ChatterboxModels in the same Bare addon, both ::std::make_shared<>()d
// per load()) don't have one Engine's destructor clobber the other
// Engine's cached weights.  Bumped by s3gen_preload(), decremented
// by s3gen_unload(); the actual cache release runs only when the
// count reaches zero.  Independent of the mutex below so the unload
// fast-path doesn't take it.
static std::atomic<int>                      g_s3gen_cache_refcount{0};
}  // namespace

// Forward declaration: clear all per-synth caches.  The persistent
// graph caches (cfm_estimator + time_mlp scaffolding) and the CPU
// weight mirrors are tied to the model's backend, so they must be
// torn down BEFORE ggml_backend_free or the gallocators / backend
// buffers freed there would be released against a dead device.
//
// Defined further down (after cfm_estimator_cache is in scope).
static void s3gen_release_synth_caches();

// Release any cached model_ctx (frees its backend buffer, ggml context and
// backend).  Must run before the ggml-metal / ggml-cuda / ggml-vulkan dylib
// tears down its static device list; otherwise their static destructors hit
// a "rsets->data count != 0" assert (residency sets still referenced by an
// orphan backend buffer).  We register it with atexit() on first cache
// insertion so it runs before process-exit dylib finalisers.
static void s3gen_model_cache_release() {
    // Tear down the per-synth caches first so any gallocrs they hold
    // (cfm_estimator_cache::allocr) are freed against the still-alive
    // backend, then drop the model.  Reverse order would crash on
    // Vulkan/Metal/CUDA where ggml_gallocr_free against a freed
    // backend asserts.
    s3gen_release_synth_caches();

    std::lock_guard<std::mutex> lk(g_s3gen_cache_mu);
    // Tear down every persistent host-side cache BEFORE freeing the
    // backend.  The graph caches own ggml_gallocr_t handles that hold
    // Vulkan (or Metal/CUDA) buffers allocated against the soon-to-be-
    // freed backend; gallocr_free against
    // a dangling vk_device asserts inside ggml-vulkan.  Same constraint as
    // the existing thread_local time_mlp_cache documents.
    s3gen_release_synth_caches();
    if (!g_s3gen_cache_entry) return;
    model_ctx * m = g_s3gen_cache_entry->m.get();
    if (m) {
        if (m->buffer_w) { ggml_backend_buffer_free(m->buffer_w); m->buffer_w = nullptr; }
        if (m->ctx_w)    { ggml_free(m->ctx_w);                   m->ctx_w    = nullptr; }
        if (m->backend)  { ggml_backend_free(m->backend);         m->backend  = nullptr; }
        m->tensors.clear();
    }
    g_s3gen_cache_entry.reset();
}

static model_ctx * s3gen_model_cache_get(const std::string & path, int n_gpu_layers, bool verbose) {
    {
        std::lock_guard<std::mutex> lk(g_s3gen_cache_mu);
        if (g_s3gen_cache_entry &&
            g_s3gen_cache_entry->path == path &&
            g_s3gen_cache_entry->gpu  == n_gpu_layers) {
            if (verbose) {
                fprintf(stderr, "  %zu tensors (cached — skip GGUF load)\n",
                        g_s3gen_cache_entry->m->tensors.size());
            }
            g_s3gen_cache_last_load_ms = 0.0;
            return g_s3gen_cache_entry->m.get();
        }
    }
    // Backend swap (different path or n_gpu_layers).  Tear down every
    // persistent cache against the OLD backend before freeing it, then
    // drop the s3gen_cache_entry.  Same reasoning as
    // s3gen_model_cache_release.
    if (g_s3gen_cache_entry) {
        s3gen_release_synth_caches();
    }
    if (verbose) fprintf(stderr, "Loading %s\n", path.c_str());
    double t0 = now_ms();
    auto m = std::make_unique<model_ctx>(load_s3gen_gguf(path, n_gpu_layers, verbose));
    g_s3gen_cache_last_load_ms = now_ms() - t0;
    if (verbose) fprintf(stderr, "  %zu tensors loaded (%.1f ms)\n", m->tensors.size(), g_s3gen_cache_last_load_ms);
    g_s3gen_cache_entry = std::make_unique<s3gen_cache_entry>(
        s3gen_cache_entry{path, n_gpu_layers, std::move(m)});

    // Register the release on first insertion.  atexit() handlers run in
    // LIFO and execute before any static destructors in DSOs loaded *after*
    // this point — which on macOS / Linux is how we avoid Metal's device
    // teardown assert on process exit.
    static bool registered = false;
    if (!registered) {
        std::atexit(s3gen_model_cache_release);
        registered = true;
    }
    return g_s3gen_cache_entry->m.get();
}

static double s3gen_model_cache_last_load_ms() { return g_s3gen_cache_last_load_ms; }

static model_ctx load_s3gen_gguf(const std::string & path, int n_gpu_layers, bool verbose) {
    model_ctx m;
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) throw std::runtime_error("gguf_init_from_file failed: " + path);
    m.backend = s3gen_init_backend(n_gpu_layers, verbose);
    int64_t n_tensors = gguf_get_n_tensors(g);
    ggml_init_params p = { ggml_tensor_overhead() * (size_t)n_tensors, nullptr, true };
    m.ctx_w = ggml_init(p);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        ggml_tensor * dst = ggml_dup_tensor(m.ctx_w, src);
        ggml_set_name(dst, name);
        m.tensors[name] = dst;
    }
    m.buffer_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, m.backend);
    for (ggml_tensor * cur = ggml_get_first_tensor(m.ctx_w); cur; cur = ggml_get_next_tensor(m.ctx_w, cur)) {
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
    }

    {
        int64_t k_mf = gguf_find_key(g, "s3gen.meanflow");
        int64_t k_ts = gguf_find_key(g, "s3gen.n_timesteps");
        int64_t k_cf = gguf_find_key(g, "s3gen.cfg_rate");
        m.meanflow    = (k_mf >= 0) ? gguf_get_val_bool(g, k_mf) : true;
        m.n_timesteps = (k_ts >= 0) ? (int) gguf_get_val_u32(g, k_ts) : (m.meanflow ? 2 : 10);
        m.cfg_rate    = (k_cf >= 0) ? gguf_get_val_f32(g, k_cf) : (m.meanflow ? 0.0f : 0.7f);
        if (k_mf < 0 && k_ts < 0 && k_cf < 0) {
            // Pre-§3.19 GGUFs lack the variant keys.  Defaults match the
            // historical Turbo behaviour, so legacy chatterbox-s3gen.gguf
            // files continue to work unchanged.  Print a one-time warning
            // because the same defaults applied to a *multilingual* S3Gen
            // GGUF produced by an older converter would silently run the
            // wrong CFM solver and emit garbage.  Re-converting picks up
            // the proper keys.
            fprintf(stderr, "warning: s3gen GGUF lacks variant keys "
                            "(s3gen.meanflow / n_timesteps / cfg_rate); "
                            "assuming Turbo (meanflow, 2 steps).  If this is "
                            "an MTL S3Gen, re-run "
                            "scripts/convert-s3gen-to-gguf.py --variant mtl.\n");
        }
        if (verbose) {
            fprintf(stderr, "  s3gen variant: %s (n_timesteps=%d, cfg_rate=%.2f)\n",
                    m.meanflow ? "meanflow" : "standard CFM + CFG",
                    m.n_timesteps, m.cfg_rate);
        }
    }

    gguf_free(g);
    ggml_free(tmp_ctx);
    return m;
}

static ggml_tensor * find_tensor(const model_ctx & m, const std::string & name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}

// F32 conv1d via im2col + mul_mat.  kernel ne=[K, IC, OC]
static ggml_tensor * conv1d_f32(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * input,
                                int stride, int padding, int dilation) {
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    return ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
}

// Batch-aware F32 conv1d.  Input: (L, IC, B) or (L, IC).  Kernel: (K, IC, OC).
// Output: (L_out, OC, B).  B=1 degenerates to the old (L_out, OC, 1) layout.
//
// ggml_mul_mat broadcasts its FIRST operand over the SECOND's ne[2..3]
// (the docs state: "A: [ne03, ne02, n, k]", "B: [ne03*x, ne02*y, m, k]"),
// and ggml_can_mul_mat rejects the opposite broadcast direction.  When the
// im2col output carries a batch dim and the kernel does not, we therefore
// put the kernel first and permute the result back to (L_out, OC, B) so
// downstream bias-add + LayerNorm layouts stay the same as the batch=1
// path.
static ggml_tensor * conv1d_f32_b(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * input,
                                  int stride, int padding, int dilation) {
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    // im2col ne=[K*IC, L_out, B]
    ggml_tensor * k_flat = ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]);
    // mul_mat(A=k_flat[k=K*IC, n=OC], B=im2col[k=K*IC, m=L_out, ne02=B])
    //   → result ne=[OC, L_out, B]
    ggml_tensor * prod = ggml_mul_mat(ctx, k_flat, im2col);
    // Permute (OC, L_out, B) → (L_out, OC, B).
    return ggml_cont(ctx, ggml_permute(ctx, prod, 1, 0, 2, 3));
}

// Drops the trailing ggml_cont.  The only caller is run_hift_decode's
// upsample loop, where the result is immediately consumed by
// ggml_add(x, ggml_reshape_2d(bias)) — a strided-tolerant pattern.
// The view's nb[1]/nb[2] are the original out's strides (which span the
// pre-trim length), so element-wise add iterates with the proper byte
// offsets.  After add, x is a fresh contiguous tensor again, so the
// downstream ggml_view_3d / ggml_concat / rb_fwd → conv1d_f32 chain sees
// contig input.  Saves 3 dispatches per HiFT decode (1 per ups stage).
static ggml_tensor * conv_transpose_1d_f32(ggml_context * ctx, ggml_tensor * kernel,
                                           ggml_tensor * input, int stride, int padding) {
    ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, input, stride, 0, 1);
    if (padding == 0) return out;
    int64_t L_new = out->ne[0] - 2 * padding;
    return ggml_view_3d(ctx, out, L_new, out->ne[1], out->ne[2],
                        out->nb[1], out->nb[2], (size_t)padding * out->nb[0]);
}

// Metal backend currently has no PAD / PAD_EXT dispatcher entry, so emulate
// front/back zero padding on dim 0 via concat(scale(view, 0), x) /
// concat(x, scale(view, 0)).  The scale(..., 0) trick produces a defined
// zero tensor (as opposed to allocating an uninitialised one and hoping).
static ggml_tensor * zero_pad_dim0(ggml_context * ctx, ggml_tensor * x, int p_front, int p_back) {
    if (p_front <= 0 && p_back <= 0) return x;
    ggml_tensor * y = x;
    if (p_front > 0) {
        GGML_ASSERT(p_front <= (int)x->ne[0]);
        ggml_tensor * head = ggml_view_4d(ctx, x, p_front, x->ne[1], x->ne[2], x->ne[3],
                                           x->nb[1], x->nb[2], x->nb[3], 0);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, head), 0.0f);
        y = ggml_concat(ctx, z, y, 0);
    }
    if (p_back > 0) {
        GGML_ASSERT(p_back <= (int)x->ne[0]);
        ggml_tensor * tail = ggml_view_4d(ctx, x, p_back, x->ne[1], x->ne[2], x->ne[3],
                                           x->nb[1], x->nb[2], x->nb[3],
                                           (size_t)(x->ne[0] - p_back) * x->nb[0]);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, tail), 0.0f);
        y = ggml_concat(ctx, y, z, 0);
    }
    return y;
}

static ggml_tensor * ggml_mish_fn(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * sp = ggml_unary(ctx, x, GGML_UNARY_OP_SOFTPLUS);
    ggml_tensor * th = ggml_unary(ctx, sp, GGML_UNARY_OP_TANH);
    return ggml_mul(ctx, x, th);
}

static ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * w, ggml_tensor * b, float eps = 1e-5f) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w);
    return ggml_add(ctx, y, b);
}

// LayerNorm on channel axis where x ne=[T, C].
static ggml_tensor * layer_norm_on_channel(ggml_context * ctx, ggml_tensor * x,
                                           ggml_tensor * w, ggml_tensor * b, float eps = 1e-5f) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    xt = ggml_norm(ctx, xt, eps);
    xt = ggml_mul(ctx, xt, w);
    xt = ggml_add(ctx, xt, b);
    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
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

// ============================================================================
// CPU-side persistent caches (multilingual TTS optimisation)
// ============================================================================
//
// Caches that amortise per-synth host-side overhead across calls:
//
//   (a) compute_time_mlp graph submissions (10× / synth on multilingual)
//   (b) cfm_estimator_cache so the CFM estimator graph isn't rebuilt
//       every synth
//   (c) CPU mirrors of the 13–28 MB flow/input_embedding + the speaker
//       affine matrices, instead of paying ggml_backend_tensor_get
//       per synth
//   (d) S3Gen Conformer encoder graph + gallocator (~700 nodes;
//       ~3-5 ms saved per synth)
//   (e) HiFT decoder graph (~3000 nodes across 3 upsample stages ×
//       9 ResBlocks; ~10-30 ms saved per synth)
//   (f) F0 predictor graph (~25 nodes; <1 ms saved per synth)
//   (g) compute_pos_emb result (T trig ops, fired twice per encoder run)
//   (h) build_hann_window / build_istft_kernel scaffolding for HiFT
//       (~1.85M F32 mults + cos/sin in build_istft_kernel alone)
//   (i) build_window_sum scaffolding (T_stft × n_fft F32 ops)
//   (j) invert_alpha_cpu fired ~72× per HiFT call (12 ResBlocks × 6
//       alpha tensors; each does a tensor_get + per-element reciprocal)
//
// Every cache is process-wide, keyed by the shape parameters that
// drive graph topology (so streaming chunks of varying length still
// produce correct output — the cache rebuilds when its key
// diverges).  Cleanup happens in s3gen_release_synth_caches before
// ggml_backend_free, so the gallocators in the graph caches release
// against a still-valid backend.

// Generic per-stage graph cache (encoder / HiFT / F0 predictor).  Owns
// the ggml_context, graph, and gallocator.  `key` encodes the shape
// parameters that drive graph topology (e.g. T for the encoder,
// pack(T_mel, T_stft) for HiFT) — a build is reused iff the requested
// `key` matches the cached one.  -1 means "no graph built".
struct graph_cache {
    int64_t                key = -1;
    ggml_context *         ctx = nullptr;
    ggml_cgraph *          gf  = nullptr;
    ggml_gallocr_t         allocr = nullptr;
    std::vector<uint8_t>   buf;

    void destroy() {
        if (allocr) { ggml_gallocr_free(allocr); allocr = nullptr; }
        if (ctx)    { ggml_free(ctx);            ctx    = nullptr; }
        gf  = nullptr;
        key = -1;
        // Keep `buf` reservation; reusing it avoids a multi-MB malloc
        // on the next rebuild.
    }
};

// Pack (T_mel, T_stft) into a single int64_t key for the HiFT graph
// cache.  Both dimensions are positive int32 in practice; combining
// them this way gives a unique key with no collision.
static int64_t pack_hift_key(int T_mel, int T_stft) {
    return ((int64_t) T_mel << 32) | (uint32_t) T_stft;
}

// CFM estimator graph cache (struct definition; the global instance
// lives in the cache-state block below alongside the other graph
// caches).  Cache key is (T, b2): a graph built for batch=1
// (cfm_estimator_forward) cannot be reused for the batch=2 path
// (cfm_estimator_forward_b2) since the input tensor layouts differ
// (ne[2] = 1 vs 2).  Today `use_b2` is constant per
// `s3gen_synthesize_to_wav` invocation so the key disambiguation is
// belt-and-braces — but a future change that switches modes
// mid-utterance (e.g. CFG warm-up where step 0 is single-pass and
// steps 1+ are batched) would silently reuse a wrong-shape graph and
// crash inside the allocator.
struct cfm_estimator_cache {
    int  T  = -1;
    bool b2 = false;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    std::vector<uint8_t> buf;
    ~cfm_estimator_cache() {
        if (allocr) ggml_gallocr_free(allocr);
        if (ctx) ggml_free(ctx);
    }
    // Explicit reset usable from s3gen_release_synth_caches() — the
    // global instance never goes out of scope, so the destructor alone
    // wouldn't run before ggml_backend_free in the normal teardown
    // ordering.  Idempotent.
    void destroy() {
        if (allocr) { ggml_gallocr_free(allocr); allocr = nullptr; }
        if (ctx)    { ggml_free(ctx);            ctx    = nullptr; }
        gf  = nullptr;
        T   = -1;
        b2  = false;
        // Keep `buf` allocated — it's just a heap arena, no backend
        // resource bound to it.  Reusing it avoids a 64 MB malloc on
        // the next synth.
    }
};

// Bit-cast cache key for floats — avoids ambiguous std::hash<float>
// behaviour on -0.0/+0.0 and NaN bit patterns.  Tested by
// test_cpu_caches.cpp::test_cache_keys.
static uint32_t g_float_bits(float t_val) {
    uint32_t bits;
    std::memcpy(&bits, &t_val, sizeof(bits));
    return bits;
}
static uint64_t g_float_pair_bits(float t_val, float r_val) {
    return ((uint64_t) g_float_bits(t_val) << 32) | (uint64_t) g_float_bits(r_val);
}

namespace {
// Single mutex around every cache.  Held only across cache-state
// mutations (insert / clear / size queries), not across the heavy
// compute itself.
static std::mutex                                                        g_synth_caches_mu;

// Result caches (per-shape memoised compute).
static std::unordered_map<uint32_t, std::vector<float>>                  g_time_mlp_results;
static std::unordered_map<uint64_t, std::vector<float>>                  g_time_emb_results;
static std::unordered_map<const ggml_tensor *, std::vector<float>>       g_weight_cpu_mirror;
static cfm_estimator_cache                                               g_cfm_estimator_cache;

// Round 2 graph caches.
static graph_cache                                                       g_encoder_graph_cache;
static graph_cache                                                       g_hift_graph_cache;
static graph_cache                                                       g_f0_graph_cache;
// Parallel metadata for HiFT: the (graph-input-name, model-tensor-ptr)
// pairs for every alpha tensor referenced by the cached HiFT graph.
// Used on cache hits to refresh each alpha-input slot with the data
// from g_inv_alpha_results without rebuilding the graph.
static std::vector<std::pair<std::string, const ggml_tensor *>>          g_hift_inv_alpha_entries;

// Round 2 result caches (pure-compute scaffolding).
static std::unordered_map<int64_t, std::vector<float>>                   g_pos_emb_results;
static std::unordered_map<const ggml_tensor *, std::vector<float>>       g_inv_alpha_results;
static std::unordered_map<int, std::vector<float>>                       g_hann_window_cache;
static std::unordered_map<int, std::vector<float>>                       g_istft_kernel_cache;
static std::unordered_map<int64_t, std::vector<float>>                   g_window_sum_cache;

// Round 5 (PROGRESS.md §3.36): STFT graph + analysis-kernel caches.
// `run_stft` runs once per synth as part of the HiFT path (between
// SineGen and the HiFT decoder).  Both the graph and the analysis
// kernel were rebuilt every synth in the un-optimised path; caching
// them eliminates a 4 MB context buffer + ggml_init + graph build +
// gallocator alloc cycle per synth, plus the small hann × trig
// build inside `build_stft_kernel`.
//
// Keying:
//   * g_stft_graph_cache.key = T_src (= T_mel × 480 in chatterbox).
//     Streaming chunks of varying length still produce correct output
//     — the cache rebuilds when its key diverges.
//   * g_stft_kernel_cache key = n_fft (int).  Constant 16 in the
//     chatterbox HiFT path; tiny per-build cost (~144 floats) but
//     pure waste across synths.
static graph_cache                                                       g_stft_graph_cache;
static std::unordered_map<int, std::vector<float>>                       g_stft_kernel_cache;
}  // namespace

// Cached F32 mirror of a model tensor.  Returns a pointer into the
// cache; valid until s3gen_unload().  Caller must NOT free.
//
// First call: ggml_backend_tensor_get into a freshly allocated
// std::vector<float>.  Subsequent calls: hit-cache and return the
// existing pointer.
//
// Requires the source tensor to be F32; chatterbox's bandwidth-heavy
// per-synth weights (input_embedding, spk_embed_affine/{w,b}) all
// live as F32, so a templated variant for F16/Q8_0 isn't needed here.
static const float * cached_cpu_weights_f32(const ggml_tensor * t) {
    if (!t) return nullptr;
    {
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        auto it = g_weight_cpu_mirror.find(t);
        if (it != g_weight_cpu_mirror.end()) {
            return it->second.data();
        }
    }
    // Read outside the lock (the get is ~ms-scale on a GPU backend).
    std::vector<float> staged(ggml_nelements(t));
    ggml_backend_tensor_get(t, staged.data(), 0, ggml_nbytes(t));

    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    auto [it, inserted] = g_weight_cpu_mirror.try_emplace(t, std::move(staged));
    return it->second.data();
}

// Tear down every per-synth cache.  Safe to call multiple times; safe
// before/after s3gen_model_cache_release.  Mutex held just long
// enough to flip the data structures — if a synth is mid-flight on
// another thread it must finish before this returns (gallocr_free on
// a graph that's about to be reused is undefined).
static void s3gen_release_synth_caches() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    g_cfm_estimator_cache.destroy();
    g_encoder_graph_cache.destroy();
    g_hift_graph_cache.destroy();
    g_f0_graph_cache.destroy();
    g_stft_graph_cache.destroy();
    g_hift_inv_alpha_entries.clear();
    g_time_mlp_results.clear();
    g_time_emb_results.clear();
    g_weight_cpu_mirror.clear();
    g_pos_emb_results.clear();
    g_inv_alpha_results.clear();
    g_hann_window_cache.clear();
    g_istft_kernel_cache.clear();
    g_window_sum_cache.clear();
    g_stft_kernel_cache.clear();
}

// ============================================================================
// Encoder (Conformer) — produces mu for CFM
// ============================================================================

struct conformer_w {
    ggml_tensor *norm_mha_w, *norm_mha_b, *norm_ff_w, *norm_ff_b;
    ggml_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    ggml_tensor *pos_w, *pos_bias_u, *pos_bias_v;
    ggml_tensor *ff1_w, *ff1_b, *ff2_w, *ff2_b;
};

static conformer_w load_conformer(const model_ctx & m, const std::string & pfx) {
    conformer_w w;
    w.norm_mha_w = find_tensor(m, pfx + "/norm_mha/w");
    w.norm_mha_b = find_tensor(m, pfx + "/norm_mha/b");
    w.norm_ff_w  = find_tensor(m, pfx + "/norm_ff/w");
    w.norm_ff_b  = find_tensor(m, pfx + "/norm_ff/b");
    w.q_w = find_tensor(m, pfx + "/attn/q/w"); w.q_b = find_tensor(m, pfx + "/attn/q/b");
    w.k_w = find_tensor(m, pfx + "/attn/k/w"); w.k_b = find_tensor(m, pfx + "/attn/k/b");
    w.v_w = find_tensor(m, pfx + "/attn/v/w"); w.v_b = find_tensor(m, pfx + "/attn/v/b");
    w.o_w = find_tensor(m, pfx + "/attn/o/w"); w.o_b = find_tensor(m, pfx + "/attn/o/b");
    w.pos_w = find_tensor(m, pfx + "/attn/pos/w");
    w.pos_bias_u = find_tensor(m, pfx + "/attn/pos_bias_u");
    w.pos_bias_v = find_tensor(m, pfx + "/attn/pos_bias_v");
    w.ff1_w = find_tensor(m, pfx + "/ff/w1/w"); w.ff1_b = find_tensor(m, pfx + "/ff/w1/b");
    w.ff2_w = find_tensor(m, pfx + "/ff/w2/w"); w.ff2_b = find_tensor(m, pfx + "/ff/w2/b");
    return w;
}

static ggml_tensor * conformer_block(ggml_context * ctx, const conformer_w & w,
                                     ggml_tensor * x, ggml_tensor * pos_emb,
                                     int D, int T, int H, int HD, float eps = 1e-12f) {
    ggml_tensor * residual = x;
    ggml_tensor * xn = ggml_norm(ctx, x, eps);
    xn = ggml_add(ctx, ggml_mul(ctx, xn, w.norm_mha_w), w.norm_mha_b);

    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, w.q_w, xn), w.q_b);
    ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, w.k_w, xn), w.k_b);
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, w.v_w, xn), w.v_b);
    ggml_tensor * p = ggml_mul_mat(ctx, w.pos_w, pos_emb);

    q = ggml_reshape_3d(ctx, q, HD, H, T);
    k = ggml_reshape_3d(ctx, k, HD, H, T);
    v = ggml_reshape_3d(ctx, v, HD, H, T);
    p = ggml_reshape_3d(ctx, p, HD, H, pos_emb->ne[1]);

    ggml_tensor * q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor * p_perm = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));

    ggml_tensor * u_bias = ggml_reshape_3d(ctx, w.pos_bias_u, HD, 1, H);
    ggml_tensor * v_bias = ggml_reshape_3d(ctx, w.pos_bias_v, HD, 1, H);
    ggml_tensor * q_plus_u = ggml_add(ctx, q_perm, u_bias);
    ggml_tensor * q_plus_v = ggml_add(ctx, q_perm, v_bias);

    ggml_tensor * ac = ggml_mul_mat(ctx, k_perm, q_plus_u);
    ggml_tensor * bd = ggml_mul_mat(ctx, p_perm, q_plus_v);
    ggml_tensor * bd_padded = zero_pad_dim0(ctx, bd, 1, 0);
    ggml_tensor * bd_viewed = ggml_reshape_3d(ctx, bd_padded, T, 2*T, H);
    ggml_tensor * bd_sliced = ggml_view_3d(ctx, bd_viewed, T, 2*T - 1, H,
                                           bd_viewed->nb[1], bd_viewed->nb[2], bd_viewed->nb[1]);
    ggml_tensor * bd_reshaped = ggml_reshape_3d(ctx, ggml_cont(ctx, bd_sliced), 2*T - 1, T, H);
    ggml_tensor * bd_final = ggml_view_3d(ctx, bd_reshaped, T, T, H,
                                          bd_reshaped->nb[1], bd_reshaped->nb[2], 0);
    bd_final = ggml_cont(ctx, bd_final);

    // Rel-pos Conformer MHA is kept on the classic ggml_soft_max +
    // separate V mat-mul path rather than ggml_flash_attn_ext because
    // the f16 cast of the relative-position bias `bd_final` (which
    // flash_attn_ext requires for its mask argument — ggml.c:5320
    // GGML_ASSERT(mask->type == GGML_TYPE_F16)) drifts the softmax
    // output by ~1e-4 per block, which compounds through the
    // 10-step CFM estimator downstream and fails the WAV quality
    // gate (cos 0.998647 vs required > 0.9998, md5 differs vs the
    // §3.22 reference 79002f09bc48dda95ec0c2cfc2b895bd). Measured
    // speed upside was −13 ms S3Gen / −1.8 % total on M3 Ultra with
    // Metal, Q4_0, Spanish prompt, seed 42 — real but not worth
    // trading against the audio quality threshold. See PROGRESS
    // §3.25 for the full negative-finding writeup. Same pattern
    // works on parakeet.cpp (see §15.8 there) because parakeet's
    // downstream is a joint argmax over tokens, which is invariant
    // to sub-bit-15 precision drift in attention scores.
    ggml_tensor * scores = ggml_add(ctx, ac, bd_final);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float)HD));
    ggml_tensor * attn = ggml_soft_max(ctx, scores);

    ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_permute(ctx, v_perm, 1, 0, 2, 3));
    ggml_tensor * attn_v = ggml_mul_mat(ctx, v_for_mm, attn);
    ggml_tensor * merged = ggml_cont(ctx, ggml_permute(ctx, attn_v, 0, 2, 1, 3));
    ggml_tensor * flat = ggml_reshape_2d(ctx, merged, HD * H, T);

    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, w.o_w, flat), w.o_b);
    x = ggml_add(ctx, residual, attn_out);

    residual = x;
    xn = ggml_norm(ctx, x, eps);
    xn = ggml_add(ctx, ggml_mul(ctx, xn, w.norm_ff_w), w.norm_ff_b);
    ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff1_w, xn), w.ff1_b);
    ff = ggml_silu(ctx, ff);
    ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff2_w, ff), w.ff2_b);
    return ggml_add(ctx, residual, ff);
}

// The `graph_cache` struct, `pack_hift_key`, and the cache-state
// globals (g_encoder_graph_cache, g_hift_graph_cache,
// g_f0_graph_cache, g_hift_inv_alpha_entries, g_pos_emb_results,
// g_inv_alpha_results, g_hann_window_cache, g_istft_kernel_cache,
// g_window_sum_cache) all live in the CPU-side cache block earlier
// in this file — declared above run_encoder so its definition can
// use them, and torn down in s3gen_release_synth_caches() against
// the still-live backend.

// Scaffolding-helper forward declarations (definitions live later, alongside
// the cfm_estimator_cache + cached_cpu_weights_f32 helpers, where the
// underlying build_* functions are visible).  Declared up here so the
// graph-build sites that consume them (run_encoder, run_f0_predictor,
// run_hift_decode) compile.
static const std::vector<float> & cached_pos_emb(int T, int D);
static const std::vector<float> & cached_inv_alpha(const model_ctx & m,
                                                   const std::string & name);
static const std::vector<float> & cached_hann_window(int n_fft);
static const std::vector<float> & cached_istft_kernel(int n_fft);
static const std::vector<float> & cached_window_sum(int T_stft, int n_fft, int hop);

static void compute_pos_emb(std::vector<float> & pe, int T, int D) {
    int L = 2 * T - 1;
    pe.assign(L * D, 0.0f);
    const float log10000 = std::log(10000.0f);
    std::vector<float> div_term(D / 2);
    for (int i = 0; i < D / 2; ++i) div_term[i] = std::exp(-((float)(2*i) * log10000 / (float)D));
    std::vector<std::vector<float>> pos_pe(T, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> neg_pe(T, std::vector<float>(D, 0.0f));
    for (int i = 0; i < T; ++i) {
        for (int k = 0; k < D / 2; ++k) {
            pos_pe[i][2*k]     = std::sin((float)i * div_term[k]);
            pos_pe[i][2*k + 1] = std::cos((float)i * div_term[k]);
            neg_pe[i][2*k]     = std::sin(-(float)i * div_term[k]);
            neg_pe[i][2*k + 1] = std::cos(-(float)i * div_term[k]);
        }
    }
    for (int t = 0; t < T; ++t) {
        int src = T - 1 - t;
        for (int d = 0; d < D; ++d) pe[t*D + d] = pos_pe[src][d];
    }
    for (int t = 1; t < T; ++t) {
        for (int d = 0; d < D; ++d) pe[(T - 1 + t)*D + d] = neg_pe[t][d];
    }
}

// Cached wrapper around compute_pos_emb.  Keyed by pack(T, D); for
// chatterbox D is constant=512 and T is determined by the encoder
// input length.  Streaming chunks at the same T after the first
// synth pay zero compute_pos_emb work.
static const std::vector<float> & cached_pos_emb(int T, int D) {
    const int64_t key = ((int64_t) T << 32) | (uint32_t) D;
    {
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        auto it = g_pos_emb_results.find(key);
        if (it != g_pos_emb_results.end()) return it->second;
    }
    std::vector<float> pe;
    compute_pos_emb(pe, T, D);
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    auto [it, inserted] = g_pos_emb_results.try_emplace(key, std::move(pe));
    return it->second;
}

// Run the full S3Gen encoder: input (T, D=512) -> mu (2T, 80)
// Graph + gallocator cached process-wide via g_encoder_graph_cache
// (keyed on T = encoder input length).  Same-shape calls (e.g. batch
// synthesis of constant-length prompts, or streaming chunks at a
// stable T) skip the rebuild + gallocr_reserve.  pos_emb
// vectors are cached separately by cached_pos_emb (keyed on (T, D));
// re-used across every same-T synth.
static std::vector<float> run_encoder(const model_ctx & m, const std::vector<float> & input_embed, int T, int D = 512) {
    const int H = 8, HEAD_DIM = 64;
    const int T2 = 2 * T;

    graph_cache & cache = g_encoder_graph_cache;
    const bool build_graph = (cache.key != (int64_t) T) || (cache.ctx == nullptr);
    if (build_graph) {
        if (cache.allocr) { ggml_gallocr_free(cache.allocr); cache.allocr = nullptr; }
        if (cache.ctx)    { ggml_free(cache.ctx);            cache.ctx    = nullptr; }
        cache.buf.resize(64 * 1024 * 1024);
        ggml_init_params gp = { cache.buf.size(), cache.buf.data(), true };
        cache.ctx = ggml_init(gp);
        cache.gf  = ggml_new_graph_custom(cache.ctx, 32768, false);
        cache.key = (int64_t) T;
    }
    ggml_context * ctx = cache.ctx;
    ggml_cgraph * gf = cache.gf;

    if (build_graph) {

    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
    ggml_set_name(x_in, "x_in"); ggml_set_input(x_in);
    ggml_tensor * pos1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 2*T - 1);
    ggml_set_name(pos1, "pos1"); ggml_set_input(pos1);
    ggml_tensor * pos2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 2*T2 - 1);
    ggml_set_name(pos2, "pos2"); ggml_set_input(pos2);

    // encoder_embed: Linear + LayerNorm + scale
    ggml_tensor * elw = find_tensor(m, "flow/encoder/embed/linear/w");
    ggml_tensor * elb = find_tensor(m, "flow/encoder/embed/linear/b");
    ggml_tensor * enw = find_tensor(m, "flow/encoder/embed/norm/w");
    ggml_tensor * enb = find_tensor(m, "flow/encoder/embed/norm/b");
    ggml_tensor * x = ggml_add(ctx, ggml_mul_mat(ctx, elw, x_in), elb);
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_add(ctx, ggml_mul(ctx, x, enw), enb);
    x = ggml_scale(ctx, x, std::sqrt((float)D));

    // pre_lookahead: 2 convs + LeakyReLU + residual
    ggml_tensor * residual = x;
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * pw1 = find_tensor(m, "flow/encoder/pre_lookahead/conv1/w");
    ggml_tensor * pb1 = find_tensor(m, "flow/encoder/pre_lookahead/conv1/b");
    ggml_tensor * pw2 = find_tensor(m, "flow/encoder/pre_lookahead/conv2/w");
    ggml_tensor * pb2 = find_tensor(m, "flow/encoder/pre_lookahead/conv2/b");
    xt = zero_pad_dim0(ctx, xt, 0, 3);
    xt = conv1d_f32(ctx, pw1, xt, 1, 0, 1);
    xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, pb1, 1, D));
    xt = ggml_leaky_relu(ctx, xt, 0.01f, false);
    xt = zero_pad_dim0(ctx, xt, 2, 0);
    xt = conv1d_f32(ctx, pw2, xt, 1, 0, 1);
    xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, pb2, 1, D));
    xt = ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
    x = ggml_add(ctx, xt, residual);

    // 6 conformer blocks at length T
    for (int i = 0; i < 6; ++i) {
        auto w = load_conformer(m, "flow/encoder/block" + std::to_string(i));
        x = conformer_block(ctx, w, x, pos1, D, T, H, HEAD_DIM);
    }

    // Upsample1D 2x
    ggml_tensor * up_w = find_tensor(m, "flow/encoder/up_layer/conv/w");
    ggml_tensor * up_b = find_tensor(m, "flow/encoder/up_layer/conv/b");
    ggml_tensor * xu = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * xu_3d = ggml_reshape_3d(ctx, xu, 1, xu->ne[0], xu->ne[1]);
    ggml_tensor * xu_2x = ggml_concat(ctx, xu_3d, xu_3d, 0);
    xu = ggml_cont(ctx, ggml_reshape_2d(ctx, xu_2x, xu_3d->ne[1]*2, xu_3d->ne[2]));
    xu = zero_pad_dim0(ctx, xu, 4, 0);
    xu = conv1d_f32(ctx, up_w, xu, 1, 0, 1);
    xu = ggml_add(ctx, xu, ggml_reshape_2d(ctx, up_b, 1, D));
    x = ggml_cont(ctx, ggml_permute(ctx, xu, 1, 0, 2, 3));

    // up_embed
    ggml_tensor * ulw = find_tensor(m, "flow/encoder/up_embed/linear/w");
    ggml_tensor * ulb = find_tensor(m, "flow/encoder/up_embed/linear/b");
    ggml_tensor * unw = find_tensor(m, "flow/encoder/up_embed/norm/w");
    ggml_tensor * unb = find_tensor(m, "flow/encoder/up_embed/norm/b");
    x = ggml_add(ctx, ggml_mul_mat(ctx, ulw, x), ulb);
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_add(ctx, ggml_mul(ctx, x, unw), unb);
    x = ggml_scale(ctx, x, std::sqrt((float)D));

    // 4 up_conformer blocks at length 2T
    for (int i = 0; i < 4; ++i) {
        auto w = load_conformer(m, "flow/encoder/up_block" + std::to_string(i));
        x = conformer_block(ctx, w, x, pos2, D, T2, H, HEAD_DIM);
    }

    // after_norm
    ggml_tensor * anw = find_tensor(m, "flow/encoder/after_norm/w");
    ggml_tensor * anb = find_tensor(m, "flow/encoder/after_norm/b");
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_add(ctx, ggml_mul(ctx, x, anw), anb);

    // encoder_proj: Linear(D -> 80)
    ggml_tensor * epw = find_tensor(m, "flow/encoder_proj/w");
    ggml_tensor * epb = find_tensor(m, "flow/encoder_proj/b");
    ggml_tensor * mu = ggml_add(ctx, ggml_mul_mat(ctx, epw, x), epb);
    ggml_set_name(mu, "mu"); ggml_set_output(mu);
    ggml_build_forward_expand(gf, mu);

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(cache.allocr, gf);
    }  // end build_graph

    ggml_gallocr_alloc_graph(cache.allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), input_embed.data(), 0, input_embed.size()*sizeof(float));

    // Cached positional embeddings — same (T, D) keys reused across every
    // synth at the same chunk size.  compute_pos_emb is ~T*D*5 trig ops
    // per call; for multilingual T=350+ at D=512 that's a real wedge of
    // per-synth host time.
    const std::vector<float> & pe1 = cached_pos_emb(T,  D);
    const std::vector<float> & pe2 = cached_pos_emb(T2, D);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos1"), pe1.data(), 0, pe1.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos2"), pe2.data(), 0, pe2.size()*sizeof(float));
    compute(m.backend, gf);

    ggml_tensor * mu_out = ggml_graph_get_tensor(gf, "mu");
    std::vector<float> mu_data(ggml_nelements(mu_out));
    ggml_backend_tensor_get(mu_out, mu_data.data(), 0, ggml_nbytes(mu_out));
    return mu_data;  // shape ggml ne=[T2, 80] = numpy (80, T2)
}

// ============================================================================
// CFM estimator (single forward) — same graph as stage_G4 of test_s3gen.cpp
// ============================================================================

struct cfm_resnet_w {
    ggml_tensor *b1_conv_w, *b1_conv_b, *b1_ln_w, *b1_ln_b;
    ggml_tensor *b2_conv_w, *b2_conv_b, *b2_ln_w, *b2_ln_b;
    ggml_tensor *mlp_w, *mlp_b, *res_w, *res_b;
};

static cfm_resnet_w load_cfm_resnet(const model_ctx & m, const std::string & pfx) {
    cfm_resnet_w w;
    w.b1_conv_w = find_tensor(m, pfx + "/block1/block/0/weight");
    w.b1_conv_b = find_tensor(m, pfx + "/block1/block/0/bias");
    w.b1_ln_w   = find_tensor(m, pfx + "/block1/block/2/weight");
    w.b1_ln_b   = find_tensor(m, pfx + "/block1/block/2/bias");
    w.b2_conv_w = find_tensor(m, pfx + "/block2/block/0/weight");
    w.b2_conv_b = find_tensor(m, pfx + "/block2/block/0/bias");
    w.b2_ln_w   = find_tensor(m, pfx + "/block2/block/2/weight");
    w.b2_ln_b   = find_tensor(m, pfx + "/block2/block/2/bias");
    w.mlp_w     = find_tensor(m, pfx + "/mlp/1/weight");
    w.mlp_b     = find_tensor(m, pfx + "/mlp/1/bias");
    w.res_w     = find_tensor(m, pfx + "/res_conv/weight");
    w.res_b     = find_tensor(m, pfx + "/res_conv/bias");
    return w;
}

static ggml_tensor * cfm_causal_block(ggml_context * ctx, ggml_tensor * x,
                                      ggml_tensor * conv_w, ggml_tensor * conv_b,
                                      ggml_tensor * ln_w, ggml_tensor * ln_b, int64_t C_out) {
    ggml_tensor * xp = zero_pad_dim0(ctx, x, 2, 0);
    ggml_tensor * y = conv1d_f32(ctx, conv_w, xp, 1, 0, 1);
    y = ggml_add(ctx, y, ggml_reshape_2d(ctx, conv_b, 1, C_out));
    y = layer_norm_on_channel(ctx, y, ln_w, ln_b);
    return ggml_mish_fn(ctx, y);
}

static ggml_tensor * cfm_resnet(ggml_context * ctx, const cfm_resnet_w & w,
                                ggml_tensor * x, ggml_tensor * t_emb, int64_t C_out) {
    ggml_tensor * h = cfm_causal_block(ctx, x, w.b1_conv_w, w.b1_conv_b, w.b1_ln_w, w.b1_ln_b, C_out);
    ggml_tensor * t_feat = ggml_mish_fn(ctx, t_emb);
    ggml_tensor * t_proj = ggml_add(ctx, ggml_mul_mat(ctx, w.mlp_w, t_feat), w.mlp_b);
    h = ggml_add(ctx, h, ggml_reshape_2d(ctx, t_proj, 1, C_out));
    h = cfm_causal_block(ctx, h, w.b2_conv_w, w.b2_conv_b, w.b2_ln_w, w.b2_ln_b, C_out);
    ggml_tensor * res = conv1d_f32(ctx, w.res_w, x, 1, 0, 1);
    res = ggml_add(ctx, res, ggml_reshape_2d(ctx, w.res_b, 1, C_out));
    return ggml_add(ctx, h, res);
}

struct basic_tfm_w {
    ggml_tensor *norm1_w, *norm1_b;
    ggml_tensor *to_q, *to_k, *to_v;
    ggml_tensor *to_out_w, *to_out_b;
    ggml_tensor *norm3_w, *norm3_b;
    ggml_tensor *ff0_w, *ff0_b, *ff2_w, *ff2_b;
};

static basic_tfm_w load_basic_tfm(const model_ctx & m, const std::string & pfx) {
    basic_tfm_w w;
    w.norm1_w = find_tensor(m, pfx + "/norm1/weight");
    w.norm1_b = find_tensor(m, pfx + "/norm1/bias");
    w.to_q = find_tensor(m, pfx + "/attn1/to_q/weight");
    w.to_k = find_tensor(m, pfx + "/attn1/to_k/weight");
    w.to_v = find_tensor(m, pfx + "/attn1/to_v/weight");
    w.to_out_w = find_tensor(m, pfx + "/attn1/to_out/0/weight");
    w.to_out_b = find_tensor(m, pfx + "/attn1/to_out/0/bias");
    w.norm3_w = find_tensor(m, pfx + "/norm3/weight");
    w.norm3_b = find_tensor(m, pfx + "/norm3/bias");
    w.ff0_w = find_tensor(m, pfx + "/ff/net/0/proj/weight");
    w.ff0_b = find_tensor(m, pfx + "/ff/net/0/proj/bias");
    w.ff2_w = find_tensor(m, pfx + "/ff/net/2/weight");
    w.ff2_b = find_tensor(m, pfx + "/ff/net/2/bias");
    return w;
}

static ggml_tensor * basic_tfm(ggml_context * ctx, const basic_tfm_w & w,
                               ggml_tensor * x, int T, int C, bool f16_kv_attn, int H = 8, int HD = 64) {
    int INNER = H * HD;
    ggml_tensor * nx = layer_norm(ctx, x, w.norm1_w, w.norm1_b);
    ggml_tensor * q = ggml_mul_mat(ctx, w.to_q, nx);
    ggml_tensor * k = ggml_mul_mat(ctx, w.to_k, nx);
    ggml_tensor * v = ggml_mul_mat(ctx, w.to_v, nx);
    // Zero-cont Q/K/V for flash-attn (see PROGRESS.md 3.14).  After
    // mul_mat the result is (INNER, T) contiguous; INNER = H * HD.
    // Metal's flash_attn_ext takes strided (HD, T, H) views directly,
    // so drop the reshape+permute+cont triple.
    const size_t col_stride  = (size_t) INNER * sizeof(float);
    const size_t head_stride = (size_t) HD    * sizeof(float);
    q = ggml_view_3d(ctx, q, HD, T, H, col_stride, head_stride, 0);
    k = ggml_view_3d(ctx, k, HD, T, H, col_stride, head_stride, 0);
    v = ggml_view_3d(ctx, v, HD, T, H, col_stride, head_stride, 0);
    if (f16_kv_attn) {
        // Experimental OpenCL/mobile mode: keep Q in F32 but materialise K/V
        // into contiguous F16 so backends with `flash_attn_f32_f16` (e.g.
        // Adreno OpenCL, see PROGRESS.md "OpenCL / Adreno bring-up" §
        // "OpenCL optimization log") dispatch the mixed-precision kernel
        // instead of the F32-only one.  ggml_cpy handles the strided-source
        // → contiguous-F16-dst case across Metal / OpenCL / CPU.
        ggml_tensor * k_f16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, T, H);
        ggml_tensor * v_f16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, T, H);
        k = ggml_cpy(ctx, k, k_f16);
        v = ggml_cpy(ctx, v, v_f16);
    }

    // Fused softmax(QK^T / sqrt(HD)) @ V, streaming (no materialized T x T attn matrix).
    // Output layout is (HD, H, T) internally ((D, H, N) per flash_attn_ext docs).
    ggml_tensor * attn_fa = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr,
                                                /*scale=*/1.0f / std::sqrt((float)HD),
                                                /*max_bias=*/0.0f,
                                                /*logit_softcap=*/0.0f);
    // flash_attn_ext output: ne=[HD, H, T, 1] (contiguous). Reshape to (INNER, T).
    ggml_tensor * flat = ggml_reshape_2d(ctx, attn_fa, INNER, T);
    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, w.to_out_w, flat), w.to_out_b);
    x = ggml_add(ctx, x, attn_out);

    ggml_tensor * nx2 = layer_norm(ctx, x, w.norm3_w, w.norm3_b);
    ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff0_w, nx2), w.ff0_b);
    ff = ggml_gelu_erf(ctx, ff);
    ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff2_w, ff), w.ff2_b);
    return ggml_add(ctx, x, ff);
}

struct cfm_tfm_stack { std::vector<basic_tfm_w> blocks; };
static cfm_tfm_stack load_tfm_stack(const model_ctx & m, const std::string & pfx, int n) {
    cfm_tfm_stack s;
    for (int i = 0; i < n; ++i) s.blocks.push_back(load_basic_tfm(m, pfx + "/" + std::to_string(i)));
    return s;
}

static ggml_tensor * apply_tfm_stack(ggml_context * ctx, const cfm_tfm_stack & s,
                                     ggml_tensor * x, int T, int C, bool f16_kv_attn) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    for (const auto & b : s.blocks) xt = basic_tfm(ctx, b, xt, T, C, f16_kv_attn);
    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
}

static ggml_tensor * cfm_causal_k3(ggml_context * ctx, ggml_tensor * x,
                                   ggml_tensor * w, ggml_tensor * b, int C_out) {
    ggml_tensor * xp = zero_pad_dim0(ctx, x, 2, 0);
    ggml_tensor * y = conv1d_f32(ctx, w, xp, 1, 0, 1);
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, C_out));
}

// --------------------------------------------------------------------------
// Batch-aware CFM estimator helpers.  These mirror the batch=1 helpers but
// preserve an outer batch dim so the non-meanflow CFG path can run cond +
// uncond through the decoder in a single forward call (batch=2), amortising
// the weight-read cost across both passes.
//
// Shape convention: x ne=[T, C, B].  t_emb is (TIME_DIM, B).  Biases are
// (C,) — ggml broadcasts size-1 dims.
// --------------------------------------------------------------------------

static ggml_tensor * cfm_causal_block_b(ggml_context * ctx, ggml_tensor * x,
                                        ggml_tensor * conv_w, ggml_tensor * conv_b,
                                        ggml_tensor * ln_w, ggml_tensor * ln_b, int64_t C_out) {
    ggml_tensor * xp = zero_pad_dim0(ctx, x, 2, 0);
    ggml_tensor * y = conv1d_f32_b(ctx, conv_w, xp, 1, 0, 1);
    y = ggml_add(ctx, y, ggml_reshape_2d(ctx, conv_b, 1, C_out));
    y = layer_norm_on_channel(ctx, y, ln_w, ln_b);
    return ggml_mish_fn(ctx, y);
}

static ggml_tensor * cfm_resnet_b(ggml_context * ctx, const cfm_resnet_w & w,
                                  ggml_tensor * x, ggml_tensor * t_emb_b, int64_t C_out) {
    ggml_tensor * h = cfm_causal_block_b(ctx, x, w.b1_conv_w, w.b1_conv_b, w.b1_ln_w, w.b1_ln_b, C_out);
    ggml_tensor * t_feat = ggml_mish_fn(ctx, t_emb_b);                      // (TIME_DIM, B)
    ggml_tensor * t_proj = ggml_add(ctx, ggml_mul_mat(ctx, w.mlp_w, t_feat),
                                    w.mlp_b);                                // (C, B)
    const int64_t B = t_proj->ne[1];
    h = ggml_add(ctx, h, ggml_reshape_3d(ctx, t_proj, 1, C_out, B));
    h = cfm_causal_block_b(ctx, h, w.b2_conv_w, w.b2_conv_b, w.b2_ln_w, w.b2_ln_b, C_out);
    ggml_tensor * res = conv1d_f32_b(ctx, w.res_w, x, 1, 0, 1);
    res = ggml_add(ctx, res, ggml_reshape_2d(ctx, w.res_b, 1, C_out));
    return ggml_add(ctx, h, res);
}

static ggml_tensor * basic_tfm_b(ggml_context * ctx, const basic_tfm_w & w,
                                 ggml_tensor * x, int T, int C, int B,
                                 bool f16_kv_attn,
                                 int H = 8, int HD = 64) {
    int INNER = H * HD;
    ggml_tensor * nx = layer_norm(ctx, x, w.norm1_w, w.norm1_b);            // (C, T, B)
    ggml_tensor * q = ggml_mul_mat(ctx, w.to_q, nx);                        // (INNER, T, B)
    ggml_tensor * k = ggml_mul_mat(ctx, w.to_k, nx);
    ggml_tensor * v = ggml_mul_mat(ctx, w.to_v, nx);
    // Zero-cont Q/K/V (see PROGRESS.md 3.14): express the
    // (HD, T, H, B) layout expected by flash_attn_ext as a strided view
    // on the already-contiguous (INNER, T, B) mul_mat output.
    const size_t col_stride   = (size_t) INNER   * sizeof(float);
    const size_t head_stride  = (size_t) HD      * sizeof(float);
    const size_t batch_stride = (size_t) INNER * T * sizeof(float);
    q = ggml_view_4d(ctx, q, HD, T, H, B, col_stride, head_stride, batch_stride, 0);
    k = ggml_view_4d(ctx, k, HD, T, H, B, col_stride, head_stride, batch_stride, 0);
    v = ggml_view_4d(ctx, v, HD, T, H, B, col_stride, head_stride, batch_stride, 0);
    if (f16_kv_attn) {
        // Mirror the batch=1 path: optionally materialise K/V as contiguous
        // F16 so backends with `flash_attn_f32_f16` (Adreno OpenCL) dispatch
        // the mixed-precision kernel.  See basic_tfm() for full rationale.
        ggml_tensor * k_f16 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HD, T, H, B);
        ggml_tensor * v_f16 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HD, T, H, B);
        k = ggml_cpy(ctx, k, k_f16);
        v = ggml_cpy(ctx, v, v_f16);
    }
    ggml_tensor * attn_fa = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr,
                                                1.0f / std::sqrt((float)HD), 0.0f, 0.0f);
    // flash_attn_ext output ne=[HD, H, T, B].  Reshape back to (INNER, T, B).
    ggml_tensor * flat = ggml_reshape_3d(ctx, attn_fa, INNER, T, B);
    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, w.to_out_w, flat), w.to_out_b);
    x = ggml_add(ctx, x, attn_out);

    ggml_tensor * nx2 = layer_norm(ctx, x, w.norm3_w, w.norm3_b);
    ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff0_w, nx2), w.ff0_b);
    ff = ggml_gelu_erf(ctx, ff);
    ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff2_w, ff), w.ff2_b);
    return ggml_add(ctx, x, ff);
}

static ggml_tensor * apply_tfm_stack_b(ggml_context * ctx, const cfm_tfm_stack & s,
                                       ggml_tensor * x, int T, int C, int B,
                                       bool f16_kv_attn) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));    // (C, T, B)
    for (const auto & b : s.blocks) xt = basic_tfm_b(ctx, b, xt, T, C, B, f16_kv_attn);
    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));               // (T, C, B)
}

static ggml_tensor * cfm_causal_k3_b(ggml_context * ctx, ggml_tensor * x,
                                     ggml_tensor * w, ggml_tensor * b, int C_out) {
    ggml_tensor * xp = zero_pad_dim0(ctx, x, 2, 0);
    ggml_tensor * y = conv1d_f32_b(ctx, w, xp, 1, 0, 1);
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, C_out));
}

// Compute the time embedding for a single scalar t (or r).
// Returns (TIME_EMB_DIM=1024,) after sinusoidal + 2-layer MLP.
//
// Cached: the graph topology (inputs, weights, output shape) is constant
// across all 10 CFM steps. Previously each call rebuilt the graph,
// reserved a fresh gallocr, computed, and freed — burning ~1 ms of
// dispatch + allocator overhead per step on Metal. Per call (multilingual,
// 10 CFM steps) that's ~10 ms; for meanflow with `compute_time_mixed`
// it's slightly more. The cache is keyed on the backend pointer so a
// fresh model_ctx in another thread doesn't share scaffolding.
struct time_mlp_cache {
    ggml_backend_t  backend = nullptr;
    std::vector<uint8_t> buf;
    ggml_context *  ctx    = nullptr;
    ggml_cgraph *   gf     = nullptr;
    ggml_gallocr_t  allocr = nullptr;
    ggml_tensor *   x_in   = nullptr;
    ggml_tensor *   y_out  = nullptr;
    ~time_mlp_cache() {
        if (allocr) ggml_gallocr_free(allocr);
        if (ctx)    ggml_free(ctx);
    }
};

static std::vector<float> compute_time_mlp(const model_ctx & m, float t_val) {
    const int TDIM = 320;
    std::vector<float> t_sin(TDIM);
    float log_factor = std::log(10000.0f) / (float)(TDIM/2 - 1);
    for (int i = 0; i < TDIM/2; ++i) {
        float freq = std::exp(-(float)i * log_factor);
        float arg = 1000.0f * t_val * freq;
        t_sin[i] = std::sin(arg);
        t_sin[i + TDIM/2] = std::cos(arg);
    }

    thread_local time_mlp_cache cache;
    if (cache.ctx == nullptr || cache.backend != m.backend) {
        if (cache.allocr) { ggml_gallocr_free(cache.allocr); cache.allocr = nullptr; }
        if (cache.ctx)    { ggml_free(cache.ctx); cache.ctx = nullptr; }
        cache.buf.assign(4 * 1024 * 1024, 0);
        ggml_init_params gp = { cache.buf.size(), cache.buf.data(), true };
        cache.ctx = ggml_init(gp);
        cache.gf  = ggml_new_graph(cache.ctx);

        cache.x_in = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, TDIM);
        ggml_set_name(cache.x_in, "x"); ggml_set_input(cache.x_in);
        ggml_tensor * l1w = find_tensor(m, "cfm/time_mlp/linear_1/weight");
        ggml_tensor * l1b = find_tensor(m, "cfm/time_mlp/linear_1/bias");
        ggml_tensor * l2w = find_tensor(m, "cfm/time_mlp/linear_2/weight");
        ggml_tensor * l2b = find_tensor(m, "cfm/time_mlp/linear_2/bias");
        ggml_tensor * y = ggml_add(cache.ctx, ggml_mul_mat(cache.ctx, l1w, cache.x_in), l1b);
        y = ggml_silu(cache.ctx, y);
        y = ggml_add(cache.ctx, ggml_mul_mat(cache.ctx, l2w, y), l2b);
        ggml_set_name(y, "out"); ggml_set_output(y);
        cache.y_out = y;
        ggml_build_forward_expand(cache.gf, cache.y_out);

        cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        ggml_gallocr_reserve(cache.allocr, cache.gf);
        cache.backend = m.backend;
    }

    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
    ggml_backend_tensor_set(cache.x_in, t_sin.data(), 0, t_sin.size() * sizeof(float));
    compute(m.backend, cache.gf);

    std::vector<float> out(ggml_nelements(cache.y_out));
    ggml_backend_tensor_get(cache.y_out, out.data(), 0, ggml_nbytes(cache.y_out));
    return out;
}

// Mix t and r embeddings via time_embed_mixer (Linear(2048 -> 1024), no bias)
static std::vector<float> compute_time_mixed(const model_ctx & m,
                                             const std::vector<float> & t_mlp,
                                             const std::vector<float> & r_mlp) {
    static size_t buf_size = 4 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params gp = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    int TOT = (int)t_mlp.size();
    ggml_tensor * t_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TOT);
    ggml_set_name(t_in, "t_in"); ggml_set_input(t_in);
    ggml_tensor * r_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TOT);
    ggml_set_name(r_in, "r_in"); ggml_set_input(r_in);
    ggml_tensor * cat = ggml_concat(ctx, t_in, r_in, 0);
    ggml_tensor * mix_w = find_tensor(m, "cfm/time_embed_mixer/weight");
    ggml_tensor * mixed = ggml_mul_mat(ctx, mix_w, cat);
    ggml_set_name(mixed, "out"); ggml_set_output(mixed);
    ggml_build_forward_expand(gf, mixed);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_in"), t_mlp.data(), 0, t_mlp.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "r_in"), r_mlp.data(), 0, r_mlp.size()*sizeof(float));
    compute(m.backend, gf);

    std::vector<float> out(ggml_nelements(mixed));
    ggml_backend_tensor_get(mixed, out.data(), 0, ggml_nbytes(mixed));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return out;
}

// Memoised time-embedding pipeline.  Both Turbo (meanflow,
// t_span = [0, 0.5, 1]) and multilingual (cosine-scheduled, 10 steps)
// produce the same set of t-values across all subsequent synth calls —
// the t-embedding outputs are deterministic functions of t (and the
// model weights), so we cache them.  Globals + mutex live in the
// anonymous namespace block earlier in this file.
//
// Bit-exactness: trivially preserved — same compute, just memoised.
static std::vector<float> compute_time_mlp_cached(const model_ctx & m, float t_val) {
    const uint32_t key = g_float_bits(t_val);
    {
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        auto it = g_time_mlp_results.find(key);
        if (it != g_time_mlp_results.end()) return it->second;
    }
    auto out = compute_time_mlp(m, t_val);
    {
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        g_time_mlp_results.try_emplace(key, out);
    }
    return out;
}

// Used only by the meanflow (Turbo) path — multilingual doesn't run
// time_embed_mixer.  Caches the full t_emb pipeline by (t, r) pair.
static std::vector<float> compute_time_emb_cached(const model_ctx & m, float t_val, float r_val) {
    const uint64_t key = g_float_pair_bits(t_val, r_val);
    {
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        auto it = g_time_emb_results.find(key);
        if (it != g_time_emb_results.end()) return it->second;
    }
    auto t_mlp = compute_time_mlp_cached(m, t_val);
    auto r_mlp = compute_time_mlp_cached(m, r_val);
    auto out = compute_time_mixed(m, t_mlp, r_mlp);
    {
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        g_time_emb_results.try_emplace(key, out);
    }
    return out;
}

// `cfm_estimator_cache` struct, its global `g_cfm_estimator_cache`,
// `g_weight_cpu_mirror` + `cached_cpu_weights_f32`, the bit-cast key
// helpers `g_float_bits` / `g_float_pair_bits`, and the
// `s3gen_release_synth_caches()` definition all live in the cache
// block earlier in this file (so they're in scope for run_encoder
// and other users above).  See "CPU-side persistent caches".

// Single estimator forward: (x, mu, t_emb, spks, cond) -> dxdt
// All shapes are numpy (80, T) or (80,) as given, flattened row-major.
static std::vector<float> cfm_estimator_forward(
    const model_ctx & m,
    cfm_estimator_cache & cache,
    const std::vector<float> & x,
    const std::vector<float> & mu,
    const std::vector<float> & t_emb,
    const std::vector<float> & spks,
    const std::vector<float> & cond,
    int T,
    bool f16_kv_attn) {
    const int MEL = 80, CH = 256, TIME_DIM = 1024;
    const int N_MID = 12, N_BLOCKS = 4;

    const bool build_graph = (cache.T != T) || cache.b2;
    if (build_graph) {
        if (cache.allocr) { ggml_gallocr_free(cache.allocr); cache.allocr = nullptr; }
        if (cache.ctx) { ggml_free(cache.ctx); cache.ctx = nullptr; }
        // 64 MB is comfortable headroom for ~1500 tensor headers + 65536-node
        // graph metadata at no_alloc=true (the buffer holds tensor structs
        // and graph book-keeping only, not weight data).  Was 256 MB before;
        // dropped after measuring real usage at <8 MB and noticing that the
        // virtual reservation was inflating RSS on systems without overcommit.
        cache.buf.resize(64 * 1024 * 1024);
        ggml_init_params gp = { cache.buf.size(), cache.buf.data(), true };
        cache.ctx = ggml_init(gp);
        cache.gf = ggml_new_graph_custom(cache.ctx, 65536, false);
        cache.T  = T;
        cache.b2 = false;
    }
    ggml_context * ctx = cache.ctx;
    ggml_cgraph * gf = cache.gf;

    if (build_graph) {

    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL); ggml_set_name(x_in, "x_in"); ggml_set_input(x_in);
    ggml_tensor * mu_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL); ggml_set_name(mu_in, "mu_in"); ggml_set_input(mu_in);
    ggml_tensor * spks_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MEL); ggml_set_name(spks_in, "spks_in"); ggml_set_input(spks_in);
    ggml_tensor * cond_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL); ggml_set_name(cond_in, "cond_in"); ggml_set_input(cond_in);
    ggml_tensor * t_emb_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIME_DIM); ggml_set_name(t_emb_in, "t_emb"); ggml_set_input(t_emb_in);

    ggml_tensor * spks_bc = ggml_repeat(ctx, ggml_reshape_2d(ctx, spks_in, 1, MEL), x_in);
    ggml_tensor * xc = ggml_concat(ctx, x_in, mu_in, 1);
    xc = ggml_concat(ctx, xc, spks_bc, 1);
    xc = ggml_concat(ctx, xc, cond_in, 1);

    auto down_rn = load_cfm_resnet(m, "cfm/down_blocks/0/0");
    auto down_tfms = load_tfm_stack(m, "cfm/down_blocks/0/1", N_BLOCKS);
    ggml_tensor * down_conv_w = find_tensor(m, "cfm/down_blocks/0/2/weight");
    ggml_tensor * down_conv_b = find_tensor(m, "cfm/down_blocks/0/2/bias");

    ggml_tensor * z = cfm_resnet(ctx, down_rn, xc, t_emb_in, CH);
    z = apply_tfm_stack(ctx, down_tfms, z, T, CH, f16_kv_attn);
    ggml_tensor * hidden = z;
    z = cfm_causal_k3(ctx, z, down_conv_w, down_conv_b, CH);

    for (int i = 0; i < N_MID; ++i) {
        auto rn = load_cfm_resnet(m, "cfm/mid_blocks/" + std::to_string(i) + "/0");
        auto tfms = load_tfm_stack(m, "cfm/mid_blocks/" + std::to_string(i) + "/1", N_BLOCKS);
        z = cfm_resnet(ctx, rn, z, t_emb_in, CH);
        z = apply_tfm_stack(ctx, tfms, z, T, CH, f16_kv_attn);
    }

    auto up_rn = load_cfm_resnet(m, "cfm/up_blocks/0/0");
    auto up_tfms = load_tfm_stack(m, "cfm/up_blocks/0/1", N_BLOCKS);
    ggml_tensor * up_conv_w = find_tensor(m, "cfm/up_blocks/0/2/weight");
    ggml_tensor * up_conv_b = find_tensor(m, "cfm/up_blocks/0/2/bias");
    z = ggml_concat(ctx, z, hidden, 1);
    z = cfm_resnet(ctx, up_rn, z, t_emb_in, CH);
    z = apply_tfm_stack(ctx, up_tfms, z, T, CH, f16_kv_attn);
    z = cfm_causal_k3(ctx, z, up_conv_w, up_conv_b, CH);

    ggml_tensor * fb_conv_w = find_tensor(m, "cfm/final_block/block/0/weight");
    ggml_tensor * fb_conv_b = find_tensor(m, "cfm/final_block/block/0/bias");
    ggml_tensor * fb_ln_w   = find_tensor(m, "cfm/final_block/block/2/weight");
    ggml_tensor * fb_ln_b   = find_tensor(m, "cfm/final_block/block/2/bias");
    z = cfm_causal_block(ctx, z, fb_conv_w, fb_conv_b, fb_ln_w, fb_ln_b, CH);

    ggml_tensor * fp_w = find_tensor(m, "cfm/final_proj/weight");
    ggml_tensor * fp_b = find_tensor(m, "cfm/final_proj/bias");
    ggml_tensor * out = conv1d_f32(ctx, fp_w, z, 1, 0, 1);
    out = ggml_add(ctx, out, ggml_reshape_2d(ctx, fp_b, 1, MEL));
    ggml_set_name(out, "out"); ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(cache.allocr, gf);
    }  // end graph-build block

    // Tried: run one dummy compute right after gallocr_reserve to pre-warm
    // Vulkan's first-dispatch state (shader residency / memory pool / command
    // pool warmup), hoping the real step0 would then run at step1 speed
    // (~12 ms instead of ~83 ms).  Outcome: the cold-compute cost is
    // per-dispatch, not per-graph-first-dispatch — adding the warmup just
    // shifted 70 ms from "hidden first-dispatch" to "explicit extra compute"
    // without reducing step0.  S3GEN_INFER went UP by ~13 ms.  Reverted.
    // The 83→12 ms gap appears to be a driver/scheduler warm-up cost on the
    // first command buffer submission that no amount of dummy work inside
    // cfm_estimator_forward removes.  Real fix would need to move the first
    // dispatch elsewhere in the pipeline (e.g. during T3→S3Gen transition)
    // so it overlaps with other host work, which is a bigger refactor.

    ggml_gallocr_alloc_graph(cache.allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), x.data(), 0, x.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mu_in"), mu.data(), 0, mu.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "spks_in"), spks.data(), 0, spks.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cond_in"), cond.data(), 0, cond.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb"), t_emb.data(), 0, t_emb.size()*sizeof(float));
    compute(m.backend, gf);

    ggml_tensor * out_t = ggml_graph_get_tensor(gf, "out");
    std::vector<float> out_data(ggml_nelements(out_t));
    ggml_backend_tensor_get(out_t, out_data.data(), 0, ggml_nbytes(out_t));
    return out_data;
}

// Single estimator forward, batch=2 — runs the conditional and unconditional
// passes through the decoder in one shot.  Inputs are flat F32 vectors of
// shape (T*MEL) or (MEL,) etc.; the `*_u` suffix carries the uncond copy.
// Output is two dxdt vectors (cond, uncond) each of shape (T*MEL).
//
// Used by the non-meanflow (MTL) CFM loop to halve its per-utterance
// estimator-call count — the expensive weight-tensor reads amortise across
// both batch elements, so the pipeline gets close to a 2× speedup on CPU
// where the decoder is memory-bandwidth bound.
static void cfm_estimator_forward_b2(
    const model_ctx & m,
    cfm_estimator_cache & cache,
    const std::vector<float> & x_c,     const std::vector<float> & x_u,
    const std::vector<float> & mu_c,    const std::vector<float> & mu_u,
    const std::vector<float> & t_emb_c, const std::vector<float> & t_emb_u,
    const std::vector<float> & spks_c,  const std::vector<float> & spks_u,
    const std::vector<float> & cond_c,  const std::vector<float> & cond_u,
    std::vector<float> & out_c, std::vector<float> & out_u,
    int T,
    bool f16_kv_attn) {
    const int MEL = 80, CH = 256, TIME_DIM = 1024;
    const int N_MID = 12, N_BLOCKS = 4;
    const int B = 2;

    const bool build_graph = (cache.T != T) || !cache.b2;
    if (build_graph) {
        if (cache.allocr) { ggml_gallocr_free(cache.allocr); cache.allocr = nullptr; }
        if (cache.ctx) { ggml_free(cache.ctx); cache.ctx = nullptr; }
        // 64 MB is plenty for 65536 graph nodes + ~3000 tensor headers (the
        // batch=2 graph roughly doubles tensor count vs the batch=1 path).
        // Was 512 MB before — see cfm_estimator_forward for the rationale on
        // why the original number was overspec.
        cache.buf.resize(64 * 1024 * 1024);
        ggml_init_params gp = { cache.buf.size(), cache.buf.data(), true };
        cache.ctx = ggml_init(gp);
        cache.gf = ggml_new_graph_custom(cache.ctx, 65536, false);
        cache.T  = T;
        cache.b2 = true;
    }
    ggml_context * ctx = cache.ctx;
    ggml_cgraph * gf = cache.gf;

    if (build_graph) {

    ggml_tensor * x_in    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, MEL, B); ggml_set_name(x_in, "x_in");       ggml_set_input(x_in);
    ggml_tensor * mu_in   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, MEL, B); ggml_set_name(mu_in, "mu_in");     ggml_set_input(mu_in);
    ggml_tensor * spks_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MEL, B);    ggml_set_name(spks_in, "spks_in"); ggml_set_input(spks_in);
    ggml_tensor * cond_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, MEL, B); ggml_set_name(cond_in, "cond_in"); ggml_set_input(cond_in);
    ggml_tensor * t_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, TIME_DIM, B); ggml_set_name(t_emb_in, "t_emb"); ggml_set_input(t_emb_in);

    // Broadcast spks (MEL, B) over T → (T, MEL, B).
    ggml_tensor * spks_bc = ggml_repeat(ctx,
        ggml_reshape_3d(ctx, spks_in, 1, MEL, B), x_in);
    ggml_tensor * xc = ggml_concat(ctx, x_in, mu_in, 1);
    xc = ggml_concat(ctx, xc, spks_bc, 1);
    xc = ggml_concat(ctx, xc, cond_in, 1);

    auto down_rn = load_cfm_resnet(m, "cfm/down_blocks/0/0");
    auto down_tfms = load_tfm_stack(m, "cfm/down_blocks/0/1", N_BLOCKS);
    ggml_tensor * down_conv_w = find_tensor(m, "cfm/down_blocks/0/2/weight");
    ggml_tensor * down_conv_b = find_tensor(m, "cfm/down_blocks/0/2/bias");

    ggml_tensor * z = cfm_resnet_b(ctx, down_rn, xc, t_emb_in, CH);
    z = apply_tfm_stack_b(ctx, down_tfms, z, T, CH, B, f16_kv_attn);
    ggml_tensor * hidden = z;
    z = cfm_causal_k3_b(ctx, z, down_conv_w, down_conv_b, CH);

    for (int i = 0; i < N_MID; ++i) {
        auto rn = load_cfm_resnet(m, "cfm/mid_blocks/" + std::to_string(i) + "/0");
        auto tfms = load_tfm_stack(m, "cfm/mid_blocks/" + std::to_string(i) + "/1", N_BLOCKS);
        z = cfm_resnet_b(ctx, rn, z, t_emb_in, CH);
        z = apply_tfm_stack_b(ctx, tfms, z, T, CH, B, f16_kv_attn);
    }

    auto up_rn = load_cfm_resnet(m, "cfm/up_blocks/0/0");
    auto up_tfms = load_tfm_stack(m, "cfm/up_blocks/0/1", N_BLOCKS);
    ggml_tensor * up_conv_w = find_tensor(m, "cfm/up_blocks/0/2/weight");
    ggml_tensor * up_conv_b = find_tensor(m, "cfm/up_blocks/0/2/bias");
    z = ggml_concat(ctx, z, hidden, 1);
    z = cfm_resnet_b(ctx, up_rn, z, t_emb_in, CH);
    z = apply_tfm_stack_b(ctx, up_tfms, z, T, CH, B, f16_kv_attn);
    z = cfm_causal_k3_b(ctx, z, up_conv_w, up_conv_b, CH);

    ggml_tensor * fb_conv_w = find_tensor(m, "cfm/final_block/block/0/weight");
    ggml_tensor * fb_conv_b = find_tensor(m, "cfm/final_block/block/0/bias");
    ggml_tensor * fb_ln_w   = find_tensor(m, "cfm/final_block/block/2/weight");
    ggml_tensor * fb_ln_b   = find_tensor(m, "cfm/final_block/block/2/bias");
    z = cfm_causal_block_b(ctx, z, fb_conv_w, fb_conv_b, fb_ln_w, fb_ln_b, CH);

    ggml_tensor * fp_w = find_tensor(m, "cfm/final_proj/weight");
    ggml_tensor * fp_b = find_tensor(m, "cfm/final_proj/bias");
    ggml_tensor * out = conv1d_f32_b(ctx, fp_w, z, 1, 0, 1);
    out = ggml_add(ctx, out, ggml_reshape_2d(ctx, fp_b, 1, MEL));
    ggml_set_name(out, "out"); ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(cache.allocr, gf);
    }

    ggml_gallocr_alloc_graph(cache.allocr, gf);

    // Stage inputs: cond slice [0, T*MEL), uncond slice [T*MEL, 2*T*MEL).
    const size_t one_tm = (size_t) T * MEL * sizeof(float);
    const size_t one_m  = (size_t) MEL * sizeof(float);
    const size_t one_td = (size_t) TIME_DIM * sizeof(float);

    ggml_tensor * x_t    = ggml_graph_get_tensor(gf, "x_in");
    ggml_tensor * mu_t   = ggml_graph_get_tensor(gf, "mu_in");
    ggml_tensor * spks_t = ggml_graph_get_tensor(gf, "spks_in");
    ggml_tensor * cond_t = ggml_graph_get_tensor(gf, "cond_in");
    ggml_tensor * te_t   = ggml_graph_get_tensor(gf, "t_emb");

    ggml_backend_tensor_set(x_t,     x_c.data(),     0 * one_tm, one_tm);
    ggml_backend_tensor_set(x_t,     x_u.data(),     1 * one_tm, one_tm);
    ggml_backend_tensor_set(mu_t,    mu_c.data(),    0 * one_tm, one_tm);
    ggml_backend_tensor_set(mu_t,    mu_u.data(),    1 * one_tm, one_tm);
    ggml_backend_tensor_set(cond_t,  cond_c.data(),  0 * one_tm, one_tm);
    ggml_backend_tensor_set(cond_t,  cond_u.data(),  1 * one_tm, one_tm);
    ggml_backend_tensor_set(spks_t,  spks_c.data(),  0 * one_m,  one_m);
    ggml_backend_tensor_set(spks_t,  spks_u.data(),  1 * one_m,  one_m);
    ggml_backend_tensor_set(te_t,    t_emb_c.data(), 0 * one_td, one_td);
    ggml_backend_tensor_set(te_t,    t_emb_u.data(), 1 * one_td, one_td);

    compute(m.backend, gf);

    ggml_tensor * out_t = ggml_graph_get_tensor(gf, "out");
    // out_t ne=[T, MEL, B=2], contiguous.  Read cond/uncond halves separately.
    const size_t half_bytes = (size_t) T * MEL * sizeof(float);
    out_c.resize(T * MEL);
    out_u.resize(T * MEL);
    ggml_backend_tensor_get(out_t, out_c.data(), 0,           half_bytes);
    ggml_backend_tensor_get(out_t, out_u.data(), half_bytes,  half_bytes);
}

// ============================================================================
// HiFT vocoder (lifted from mel2wav.cpp)
// ============================================================================

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
    for (int f = 0; f < F; ++f) {
        for (int n = 0; n < n_fft; ++n) {
            double th = two_pi * f * n / n_fft;
            float w = window[n];
            K[n + f       * n_fft] = (float)(std::cos(th) * w);
            K[n + (F + f) * n_fft] = (float)(-std::sin(th) * w);
        }
    }
    return K;
}

static std::vector<float> build_istft_kernel(int n_fft, const std::vector<float> & window) {
    int F = n_fft / 2 + 1;
    std::vector<float> K((size_t)n_fft * 1 * (2 * F), 0.0f);
    const double two_pi = 2.0 * M_PI;
    const double inv_N = 1.0 / (double)n_fft;
    for (int f = 0; f < F; ++f) {
        double coef_re = (f == 0 || f == n_fft/2) ? 1.0 : 2.0;
        double coef_im = (f == 0 || f == n_fft/2) ? 0.0 : 2.0;
        for (int n = 0; n < n_fft; ++n) {
            double th = two_pi * f * n / n_fft;
            float w = window[n];
            K[n + f       * n_fft] = (float)(coef_re * std::cos(th) * w * inv_N);
            K[n + (F + f) * n_fft] = (float)(-coef_im * std::sin(th) * w * inv_N);
        }
    }
    return K;
}

static std::vector<float> build_window_sum(int T_stft, int n_fft, int hop,
                                           const std::vector<float> & window) {
    int L = (T_stft - 1) * hop + n_fft;
    std::vector<float> ws(L, 0.0f);
    for (int t = 0; t < T_stft; ++t) {
        int base = t * hop;
        for (int n = 0; n < n_fft; ++n) ws[base + n] += window[n] * window[n];
    }
    return ws;
}

// Cached HiFT scaffolding helpers.  hann_window + istft_kernel are
// pure functions of n_fft (constant 1920 in the chatterbox HiFT
// path); window_sum additionally depends on T_stft (varies with
// output length, but stable across same-shape synth
// calls).  Caching them eliminates the per-synth host-CPU build cost
// — build_istft_kernel(1920) alone is ~1.85M F32 mults + cos/sin.
static const std::vector<float> & cached_hann_window(int n_fft) {
    {
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        auto it = g_hann_window_cache.find(n_fft);
        if (it != g_hann_window_cache.end()) return it->second;
    }
    auto w = build_hann_window(n_fft, true);
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    auto [it, inserted] = g_hann_window_cache.try_emplace(n_fft, std::move(w));
    return it->second;
}

static const std::vector<float> & cached_istft_kernel(int n_fft) {
    {
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        auto it = g_istft_kernel_cache.find(n_fft);
        if (it != g_istft_kernel_cache.end()) return it->second;
    }
    // Use the cached hann window so we don't re-derive it twice.
    auto k = build_istft_kernel(n_fft, cached_hann_window(n_fft));
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    auto [it, inserted] = g_istft_kernel_cache.try_emplace(n_fft, std::move(k));
    return it->second;
}

// Cached STFT analysis kernel.  Pure function of n_fft (constant 16
// in chatterbox HiFT) and the cached hann window.  Per-build cost is
// small (~144 floats; trig + window scaling) but rebuilding it every
// synth is pointless waste.  Keyed identically
// to `cached_istft_kernel`; both share `g_synth_caches_mu`.
static const std::vector<float> & cached_stft_kernel(int n_fft) {
    {
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        auto it = g_stft_kernel_cache.find(n_fft);
        if (it != g_stft_kernel_cache.end()) return it->second;
    }
    auto k = build_stft_kernel(n_fft, cached_hann_window(n_fft));
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    auto [it, inserted] = g_stft_kernel_cache.try_emplace(n_fft, std::move(k));
    return it->second;
}

static const std::vector<float> & cached_window_sum(int T_stft, int n_fft, int hop) {
    // Pack (n_fft, hop, T_stft) into a single int64 key — n_fft and
    // hop are constants on the chatterbox path but encoding them
    // makes the cache safe against future variant additions.
    const int64_t key =
        ((int64_t)(uint16_t) n_fft << 48) |
        ((int64_t)(uint16_t) hop   << 32) |
        (int64_t)(uint32_t) T_stft;
    {
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        auto it = g_window_sum_cache.find(key);
        if (it != g_window_sum_cache.end()) return it->second;
    }
    auto ws = build_window_sum(T_stft, n_fft, hop, cached_hann_window(n_fft));
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    auto [it, inserted] = g_window_sum_cache.try_emplace(key, std::move(ws));
    return it->second;
}

static ggml_tensor * snake(ggml_context * ctx, ggml_tensor * x,
                           ggml_tensor * alpha, ggml_tensor * inv_alpha) {
    ggml_tensor * a  = ggml_reshape_2d(ctx, alpha,     1, alpha->ne[0]);
    ggml_tensor * ia = ggml_reshape_2d(ctx, inv_alpha, 1, inv_alpha->ne[0]);
    ggml_tensor * ax = ggml_mul(ctx, x, a);
    ggml_tensor * s  = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    return ggml_add(ctx, x, ggml_mul(ctx, s2, ia));
}

static std::vector<float> invert_alpha_cpu(const model_ctx & m, const std::string & name) {
    ggml_tensor * t = find_tensor(m, name);
    std::vector<float> a(ggml_nelements(t));
    ggml_backend_tensor_get(t, a.data(), 0, ggml_nbytes(t));
    std::vector<float> inv(a.size());
    for (size_t i = 0; i < a.size(); ++i) inv[i] = 1.0f / (a[i] + 1e-9f);
    return inv;
}

// invert_alpha_cpu is fired ~72× per HiFT call (12 ResBlocks × 6 alpha
// tensors); each call is a tensor_get + per-element reciprocal.  Alpha
// tensors are constant for the model lifetime, so cache by tensor* —
// invalidation tied to s3gen_release_synth_caches (model-context lifetime).
static const std::vector<float> & cached_inv_alpha(const model_ctx & m,
                                                   const std::string & name) {
    ggml_tensor * t = find_tensor(m, name);
    {
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        auto it = g_inv_alpha_results.find(t);
        if (it != g_inv_alpha_results.end()) return it->second;
    }
    auto inv = invert_alpha_cpu(m, name);
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    auto [it, inserted] = g_inv_alpha_results.try_emplace(t, std::move(inv));
    return it->second;
}

// `cached_pos_emb` lives in the cache block above (right after
// `compute_pos_emb`).  `cached_hann_window`, `cached_istft_kernel`,
// and `cached_window_sum` are defined just above this block
// (alongside `build_hann_window` / `build_istft_kernel` /
// `build_window_sum`).

// F0 predictor (mel (80, T) -> f0 (T,))
//
// Graph + gallocator cached process-wide via g_f0_graph_cache (keyed
// on T_mel).  Same-shape calls (e.g. streaming chunks at constant
// T_mel) skip the rebuild + gallocr_reserve.
static std::vector<float> run_f0_predictor(const model_ctx & m, const std::vector<float> & mel, int T_mel) {
    graph_cache & cache = g_f0_graph_cache;
    const bool build_graph = (cache.key != (int64_t) T_mel) || (cache.ctx == nullptr);
    if (build_graph) {
        if (cache.allocr) { ggml_gallocr_free(cache.allocr); cache.allocr = nullptr; }
        if (cache.ctx)    { ggml_free(cache.ctx);            cache.ctx    = nullptr; }
        cache.buf.resize(8 * 1024 * 1024);
        ggml_init_params gp = { cache.buf.size(), cache.buf.data(), true };
        cache.ctx = ggml_init(gp);
        cache.gf  = ggml_new_graph_custom(cache.ctx, 1024, false);
        cache.key = (int64_t) T_mel;
    }
    ggml_context * ctx = cache.ctx;
    ggml_cgraph * gf = cache.gf;

    if (build_graph) {

    ggml_tensor * mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, 80);
    ggml_set_name(mel_in, "mel_in"); ggml_set_input(mel_in);
    ggml_tensor * x = mel_in;
    for (int i = 0; i < 5; ++i) {
        std::string pfx = "hift/f0_predictor/condnet/" + std::to_string(i * 2);
        ggml_tensor * w = find_tensor(m, pfx + "/weight");
        ggml_tensor * b = find_tensor(m, pfx + "/bias");
        int C_out = (int)w->ne[2];
        x = conv1d_f32(ctx, w, x, 1, 1, 1);
        x = ggml_add(ctx, x, ggml_reshape_2d(ctx, b, 1, C_out));
        x = ggml_unary(ctx, x, GGML_UNARY_OP_ELU);
    }
    // ggml-cpu's mul_mat asserts nb10 == ggml_type_size(src1->type) — the
    // CPU backend rejects a permuted src1 even for f32 matmul, so the
    // cont here is required for CPU correctness.  Vulkan / Metal /
    // CUDA shaders do iterate by stride and would accept the bare
    // permute, so this trades one dispatch on those backends for not
    // aborting on CPU.  A backend-conditional fast path can revisit
    // this later.
    ggml_tensor * xp = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * cw = find_tensor(m, "hift/f0_predictor/classifier/weight");
    ggml_tensor * cb = find_tensor(m, "hift/f0_predictor/classifier/bias");
    ggml_tensor * y = ggml_mul_mat(ctx, cw, xp);
    y = ggml_add(ctx, y, cb);
    y = ggml_abs(ctx, y);
    y = ggml_reshape_1d(ctx, y, T_mel);
    ggml_set_name(y, "out"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(cache.allocr, gf);
    }  // end build_graph

    ggml_gallocr_alloc_graph(cache.allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel_in"), mel.data(), 0, mel.size()*sizeof(float));
    compute(m.backend, gf);
    ggml_tensor * y_out = ggml_graph_get_tensor(gf, "out");
    std::vector<float> f0(T_mel);
    ggml_backend_tensor_get(y_out, f0.data(), 0, ggml_nbytes(y_out));
    return f0;
}

// SineGen + SourceModule (CPU implementation)
static std::vector<float> sinegen_source(const std::vector<float> & f0_wav, int sr,
                                         int harmonic_num, float sine_amp, float noise_std,
                                         float voiced_threshold,
                                         const std::vector<float> & l_w, float l_b,
                                         uint32_t seed) {
    int T_wav = (int)f0_wav.size();
    int H = harmonic_num + 1;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-(float)M_PI, (float)M_PI);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> phase_vec(H, 0.0f);
    for (int h = 1; h < H; ++h) phase_vec[h] = uniform(rng);
    std::vector<float> sine_waves((size_t)H * T_wav, 0.0f);
    std::vector<double> cum_phase(H, 0.0);
    for (int t = 0; t < T_wav; ++t) {
        float f0 = f0_wav[t];
        bool voiced = f0 > voiced_threshold;
        for (int h = 0; h < H; ++h) {
            double inc = (double)f0 * (h + 1) / (double)sr;
            cum_phase[h] += inc;
            double theta = 2.0 * M_PI * (cum_phase[h] - std::floor(cum_phase[h]));
            float sine = sine_amp * std::sin((float)theta + phase_vec[h]);
            float namp = voiced ? noise_std : sine_amp / 3.0f;
            float uv = voiced ? 1.0f : 0.0f;
            sine_waves[(size_t)h * T_wav + t] = sine * uv + namp * gauss(rng);
        }
    }
    std::vector<float> src(T_wav, 0.0f);
    for (int t = 0; t < T_wav; ++t) {
        float s = l_b;
        for (int h = 0; h < H; ++h) s += l_w[h] * sine_waves[(size_t)h * T_wav + t];
        src[t] = std::tanh(s);
    }
    return src;
}

// STFT (time-domain source -> spec)
//
// Graph + analysis kernel cached process-wide via g_stft_graph_cache
// (keyed on T_src) and g_stft_kernel_cache (keyed on n_fft).
// Streaming chunks of varying length still produce correct output —
// the graph cache rebuilds when its T_src diverges; the n_fft-
// keyed kernel cache stays at one entry across all chunks because n_fft
// is constant in the chatterbox HiFT path.  Lifecycle is identical to
// the round-2 graph caches: invalidated together by
// s3gen_release_synth_caches() before ggml_backend_free, so the cached
// gallocator releases against a still-valid backend on backend swap or
// s3gen_unload().
static std::vector<float> run_stft(const model_ctx & m, const std::vector<float> & src) {
    const int n_fft = 16, hop = 4;
    const int F = n_fft / 2 + 1;
    int T_src = (int)src.size();

    const std::vector<float> & kernel = cached_stft_kernel(n_fft);

    graph_cache & cache = g_stft_graph_cache;
    const bool build_graph = (cache.key != (int64_t) T_src) || (cache.ctx == nullptr);
    if (build_graph) {
        if (cache.allocr) { ggml_gallocr_free(cache.allocr); cache.allocr = nullptr; }
        if (cache.ctx)    { ggml_free(cache.ctx);            cache.ctx    = nullptr; }
        // Reuse `buf` across rebuilds — keeping it allocated avoids a
        // 4 MB malloc when streaming chunks rotate through varying T_src
        // values.  graph_cache::destroy() preserves the buf reservation.
        cache.buf.resize(4 * 1024 * 1024);
        ggml_init_params gp = { cache.buf.size(), cache.buf.data(), true };
        cache.ctx = ggml_init(gp);
        cache.gf  = ggml_new_graph_custom(cache.ctx, 8192, false);
        cache.key = (int64_t) T_src;

        ggml_tensor * s = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, T_src, 1);
        ggml_set_name(s, "s"); ggml_set_input(s);
        ggml_tensor * s_pad = reflect_pad_1d(cache.ctx, s, n_fft/2, n_fft/2);
        ggml_tensor * k = ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, n_fft, 1, 2*F);
        ggml_set_name(k, "k"); ggml_set_input(k);
        ggml_tensor * spec = conv1d_f32(cache.ctx, k, s_pad, hop, 0, 1);
        ggml_set_name(spec, "out"); ggml_set_output(spec);
        ggml_build_forward_expand(cache.gf, spec);

        cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        ggml_gallocr_reserve(cache.allocr, cache.gf);
    }

    ggml_gallocr_alloc_graph(cache.allocr, cache.gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(cache.gf, "s"),
                            src.data(), 0, src.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(cache.gf, "k"),
                            kernel.data(), 0, kernel.size() * sizeof(float));
    compute(m.backend, cache.gf);
    ggml_tensor * spec = ggml_graph_get_tensor(cache.gf, "out");
    std::vector<float> out(ggml_nelements(spec));
    ggml_backend_tensor_get(spec, out.data(), 0, ggml_nbytes(spec));
    return out;
}

// Full HiFT decode: mel + s_stft -> wav (inlined from mel2wav.cpp)
// Graph + gallocator cached process-wide via g_hift_graph_cache (keyed
// on pack(T_mel, T_stft)).  Scaffolding (hann_window, istft_kernel,
// window_sum, ~72 inv_alpha tensors) is also cached, so subsequent
// same-shape calls do zero CPU host work outside
// the graph compute itself.  HiFT is the biggest multilingual beneficiary
// because audio length scales with prompt length.
static std::vector<float> run_hift_decode(const model_ctx & m,
                                          const std::vector<float> & mel, int T_mel,
                                          const std::vector<float> & s_stft, int T_stft) {
    const int MEL = 80, NFFT2 = 18, BASE_CH = 512, n_fft = 16, hop = 4;
    const int F = n_fft / 2 + 1;
    std::vector<int> ups_rates  = {8, 5, 3};
    std::vector<int> ups_ksizes = {16, 11, 7};
    std::vector<int> ups_ch     = {256, 128, 64};
    std::vector<int> rb_ksizes  = {3, 7, 11};
    std::vector<std::vector<int>> rb_dils = {{1,3,5},{1,3,5},{1,3,5}};
    std::vector<int> src_rb_ksizes = {7, 7, 11};
    std::vector<std::vector<int>> src_rb_dils = {{1,3,5},{1,3,5},{1,3,5}};

    graph_cache & cache = g_hift_graph_cache;
    const int64_t cache_key = pack_hift_key(T_mel, T_stft);
    const bool build_graph = (cache.key != cache_key) || (cache.ctx == nullptr);
    if (build_graph) {
        if (cache.allocr) { ggml_gallocr_free(cache.allocr); cache.allocr = nullptr; }
        if (cache.ctx)    { ggml_free(cache.ctx);            cache.ctx    = nullptr; }
        // 64 MB arena — same as the pre-cache version.  Reusing the
        // vector across rebuilds avoids a 64 MB malloc churn when (T_mel,
        // T_stft) change between streaming chunks.
        cache.buf.resize(64 * 1024 * 1024);
        ggml_init_params gp = { cache.buf.size(), cache.buf.data(), true };
        cache.ctx = ggml_init(gp);
        cache.gf  = ggml_new_graph_custom(cache.ctx, 131072, false);
        cache.key = cache_key;
        // Wipe and re-populate the alpha-input metadata for the new build.
        // Mutex held briefly; the graph build below runs without the lock
        // because synthesize() is process-serial in practice.
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        g_hift_inv_alpha_entries.clear();
    }
    ggml_context * ctx = cache.ctx;
    ggml_cgraph * gf = cache.gf;

    if (build_graph) {

    ggml_tensor * mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, MEL);
    ggml_set_name(mel_in, "mel_in"); ggml_set_input(mel_in);
    ggml_tensor * s_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_stft, NFFT2);
    ggml_set_name(s_in, "s_in"); ggml_set_input(s_in);

    auto mk_inv = [&](const std::string & pref, int C) {
        // Record the (graph-input-name, source-tensor-ptr) pair so that
        // run_hift_decode can re-feed each alpha-input slot on cache
        // hits.  cached_inv_alpha actually owns the data — we just need
        // a stable handle to look it up later.
        ggml_tensor * src = find_tensor(m, pref);
        (void) cached_inv_alpha(m, pref);  // warm the data cache
        std::string gn = "inv_" + pref;
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        ggml_set_name(t, gn.c_str()); ggml_set_input(t);
        {
            std::lock_guard<std::mutex> lk(g_synth_caches_mu);
            g_hift_inv_alpha_entries.emplace_back(std::move(gn), src);
        }
        return t;
    };

    auto load_rb = [&](const std::string & pref, int C) {
        struct pd { ggml_tensor *a1, *c1w, *c1b, *a2, *c2w, *c2b, *ia1, *ia2; };
        std::vector<pd> p(3);
        for (int i = 0; i < 3; ++i) {
            p[i].a1 = find_tensor(m, pref + "/activations1/" + std::to_string(i) + "/alpha");
            p[i].c1w = find_tensor(m, pref + "/convs1/" + std::to_string(i) + "/weight");
            p[i].c1b = find_tensor(m, pref + "/convs1/" + std::to_string(i) + "/bias");
            p[i].a2 = find_tensor(m, pref + "/activations2/" + std::to_string(i) + "/alpha");
            p[i].c2w = find_tensor(m, pref + "/convs2/" + std::to_string(i) + "/weight");
            p[i].c2b = find_tensor(m, pref + "/convs2/" + std::to_string(i) + "/bias");
            p[i].ia1 = mk_inv(pref + "/activations1/" + std::to_string(i) + "/alpha", C);
            p[i].ia2 = mk_inv(pref + "/activations2/" + std::to_string(i) + "/alpha", C);
        }
        return p;
    };

    auto rb_fwd = [&](auto & rb, ggml_tensor * x, int C, const std::vector<int> & dils, int ks) {
        for (int i = 0; i < 3; ++i) {
            auto & p = rb[i];
            int d = dils[i];
            int pad1 = (ks * d - d) / 2;
            int pad2 = (ks - 1) / 2;
            ggml_tensor * xt = snake(ctx, x, p.a1, p.ia1);
            xt = conv1d_f32(ctx, p.c1w, xt, 1, pad1, d);
            xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, p.c1b, 1, C));
            xt = snake(ctx, xt, p.a2, p.ia2);
            xt = conv1d_f32(ctx, p.c2w, xt, 1, pad2, 1);
            xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, p.c2b, 1, C));
            x = ggml_add(ctx, x, xt);
        }
        return x;
    };

    ggml_tensor * cpw = find_tensor(m, "hift/conv_pre/weight");
    ggml_tensor * cpb = find_tensor(m, "hift/conv_pre/bias");
    ggml_tensor * x = conv1d_f32(ctx, cpw, mel_in, 1, 3, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, cpb, 1, BASE_CH));

    for (int i = 0; i < 3; ++i) {
        x = ggml_leaky_relu(ctx, x, 0.1f, false);
        ggml_tensor * uw = find_tensor(m, "hift/ups/" + std::to_string(i) + "/weight");
        ggml_tensor * ub = find_tensor(m, "hift/ups/" + std::to_string(i) + "/bias");
        int up_pad = (ups_ksizes[i] - ups_rates[i]) / 2;
        x = conv_transpose_1d_f32(ctx, uw, x, ups_rates[i], up_pad);
        x = ggml_add(ctx, x, ggml_reshape_2d(ctx, ub, 1, ups_ch[i]));
        if (i == 2) {
            ggml_tensor * xs = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 1 * x->nb[0]);
            xs = ggml_cont(ctx, xs);
            x = ggml_concat(ctx, xs, x, 0);
        }
        ggml_tensor * sw = find_tensor(m, "hift/source_downs/" + std::to_string(i) + "/weight");
        ggml_tensor * sb = find_tensor(m, "hift/source_downs/" + std::to_string(i) + "/bias");
        int sd_stride = (i == 0) ? 15 : (i == 1) ? 3 : 1;
        int sd_pad    = (i == 0) ? 7  : (i == 1) ? 1 : 0;
        ggml_tensor * si = conv1d_f32(ctx, sw, s_in, sd_stride, sd_pad, 1);
        si = ggml_add(ctx, si, ggml_reshape_2d(ctx, sb, 1, (int)sw->ne[2]));
        auto srb = load_rb("hift/source_resblocks/" + std::to_string(i), ups_ch[i]);
        si = rb_fwd(srb, si, ups_ch[i], src_rb_dils[i], src_rb_ksizes[i]);
        x = ggml_add(ctx, x, si);

        ggml_tensor * xs = nullptr;
        for (int j = 0; j < 3; ++j) {
            auto rb = load_rb("hift/resblocks/" + std::to_string(i * 3 + j), ups_ch[i]);
            ggml_tensor * rb_out = rb_fwd(rb, x, ups_ch[i], rb_dils[j], rb_ksizes[j]);
            xs = (xs == nullptr) ? rb_out : ggml_add(ctx, xs, rb_out);
        }
        x = ggml_scale(ctx, xs, 1.0f / 3.0f);
    }

    x = ggml_leaky_relu(ctx, x, 0.01f, false);
    ggml_tensor * cp2w = find_tensor(m, "hift/conv_post/weight");
    ggml_tensor * cp2b = find_tensor(m, "hift/conv_post/bias");
    x = conv1d_f32(ctx, cp2w, x, 1, 3, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, cp2b, 1, NFFT2));

    // ISTFT
    size_t col_stride = x->nb[1];
    ggml_tensor * mag_log = ggml_cont(ctx, ggml_view_2d(ctx, x, T_stft, F, col_stride, 0));
    mag_log = ggml_clamp(ctx, mag_log, -1e6f, 1e2f);
    ggml_tensor * mag = ggml_exp(ctx, mag_log);
    ggml_tensor * ph_in = ggml_cont(ctx, ggml_view_2d(ctx, x, T_stft, F, col_stride, (size_t)F * col_stride));
    ggml_tensor * ph = ggml_sin(ctx, ph_in);
    ggml_tensor * real = ggml_mul(ctx, mag, ggml_cos(ctx, ph));
    ggml_tensor * imag = ggml_mul(ctx, mag, ggml_sin(ctx, ph));
    ggml_tensor * spec = ggml_concat(ctx, real, imag, 1);

    // Cached scaffolding sizes — pure functions of (n_fft, hop, T_stft).
    // Build the input-tensor declarations against the cached vector sizes.
    const std::vector<float> & ws_for_size = cached_window_sum(T_stft, n_fft, hop);

    ggml_tensor * istft_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_fft, 1, 2 * F);
    ggml_set_name(istft_k, "istft_k"); ggml_set_input(istft_k);
    ggml_tensor * ws_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int)ws_for_size.size(), 1);
    ggml_set_name(ws_in, "w_sum"); ggml_set_input(ws_in);

    ggml_tensor * y = ggml_conv_transpose_1d(ctx, istft_k, spec, hop, 0, 1);
    y = ggml_div(ctx, y, ws_in);
    int pad_amt = n_fft / 2;
    int L_wav = (int)ws_for_size.size() - n_fft;
    // Drop the trailing ggml_cont.  The view's only consumer is
    // ggml_clamp (element-wise, accepts strided src0); clamp's output
    // is a fresh contiguous tensor allocated by the gallocator.
    // ggml_set_output is set on that contig output, so
    // tensor_get reads from a contig buffer.  Saves 1 dispatch / HiFT decode.
    ggml_tensor * y_trim = ggml_view_2d(ctx, y, L_wav, y->ne[1], y->nb[1],
                                        (size_t)pad_amt * y->nb[0]);
    y_trim = ggml_clamp(ctx, y_trim, -0.99f, 0.99f);
    ggml_set_name(y_trim, "wav"); ggml_set_output(y_trim);
    ggml_build_forward_expand(gf, y_trim);

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(cache.allocr, gf);
    }  // end build_graph

    // Cached scaffolding (pulled outside build_graph too — when the graph
    // is reused, ik / ws data still need to be staged into the input
    // tensors).  cached_* helpers are O(1) on hits.
    const std::vector<float> & ik_data = cached_istft_kernel(n_fft);
    const std::vector<float> & ws_data = cached_window_sum(T_stft, n_fft, hop);

    ggml_gallocr_alloc_graph(cache.allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel_in"),  mel.data(),    0, mel.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "s_in"),    s_stft.data(), 0, s_stft.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "istft_k"), ik_data.data(),0, ik_data.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w_sum"),   ws_data.data(),0, ws_data.size()*sizeof(float));
    // Re-feed every alpha-input slot from the cached data.  The (graph-
    // input-name, source-tensor-ptr) pairs were captured during the
    // graph build; cached_inv_alpha is the source of truth for the data
    // (keyed by source tensor pointer, so the entry survives across
    // graph rebuilds — only s3gen_release_synth_caches drops it).
    //
    // Snapshot g_hift_inv_alpha_entries under the mutex (cheap; ~72
    // string + pointer pairs), then iterate WITHOUT the lock.  Each
    // cached_inv_alpha call below takes the same mutex internally, so
    // holding it across the loop would deadlock.
    std::vector<std::pair<std::string, const ggml_tensor *>> entries_snapshot;
    {
        std::lock_guard<std::mutex> lk(g_synth_caches_mu);
        entries_snapshot = g_hift_inv_alpha_entries;
    }
    for (const auto & e : entries_snapshot) {
        ggml_tensor * src = const_cast<ggml_tensor *>(e.second);
        const std::string src_name = ggml_get_name(src);
        const std::vector<float> & inv = cached_inv_alpha(m, src_name);
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, e.first.c_str()),
                                inv.data(), 0, inv.size()*sizeof(float));
    }
    compute(m.backend, gf);

    ggml_tensor * y_trim_out = ggml_graph_get_tensor(gf, "wav");
    std::vector<float> wav(ggml_nelements(y_trim_out));
    ggml_backend_tensor_get(y_trim_out, wav.data(), 0, ggml_nbytes(y_trim_out));
    return wav;
}

// ============================================================================
// WAV writer
// ============================================================================

static void write_wav(const std::string & path, const std::vector<float> & wav, int sr) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open " + path);
    uint32_t n = (uint32_t)wav.size();
    uint32_t byte_rate = sr * 2;
    uint32_t data_size = n * 2;
    uint32_t chunk_size = 36 + data_size;
    std::fwrite("RIFF", 1, 4, f); std::fwrite(&chunk_size, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f); std::fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16; uint16_t fmt = 1, ch = 1, align = 2, bps = 16;
    uint32_t sr_u = (uint32_t)sr;
    std::fwrite(&fmt_size, 4, 1, f); std::fwrite(&fmt, 2, 1, f); std::fwrite(&ch, 2, 1, f);
    std::fwrite(&sr_u, 4, 1, f); std::fwrite(&byte_rate, 4, 1, f);
    std::fwrite(&align, 2, 1, f); std::fwrite(&bps, 2, 1, f);
    std::fwrite("data", 1, 4, f); std::fwrite(&data_size, 4, 1, f);
    for (float x : wav) {
        float c = std::max(-1.0f, std::min(1.0f, x));
        int16_t v = (int16_t)std::lrintf(c * 32767.0f);
        std::fwrite(&v, 2, 1, f);
    }
    std::fclose(f);
}

// ============================================================================
// Token file loader (reads "1,2,3" or newline-separated ints)
// ============================================================================

static std::vector<int32_t> read_tokens_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<int32_t> out;
    std::string token;
    while (std::getline(f, token, ',')) {
        try { out.push_back(std::stoi(token)); } catch (...) {
            // maybe newline-separated, try stoi after trimming
            while (!token.empty() && (token.back() == '\n' || token.back() == ' ' || token.back() == '\r')) token.pop_back();
            if (!token.empty()) out.push_back(std::stoi(token));
        }
    }
    return out;
}

// ============================================================================
// Public entry point — takes pre-generated T3 speech tokens + a voice source
// (either a --ref-dir with .npy files or the built-in voice baked into the
// s3gen GGUF) and writes a 24 kHz wav.
// ============================================================================

#include "tts-cpp/chatterbox/s3gen_pipeline.h"

int s3gen_synthesize_to_wav(
    const std::vector<int32_t> & speech_tokens,
    const s3gen_synthesize_opts & opts)
{
    const std::string & gguf_path = opts.s3gen_gguf_path;
    const std::string & ref_dir   = opts.ref_dir;
    const std::string & out_path  = opts.out_wav_path;
    const int  seed       = opts.seed;
    const int  sr         = opts.sr;
    const bool debug_mode = opts.debug;
    const bool verbose    = opts.verbose;
    const int  pre_lookahead_len = 3;  // Chatterbox default

    // Verbose progress / per-stage timing goes through this helper so all of
    // it can be disabled with `--verbose` unset.  Errors and machine-parseable
    // BENCH: lines stay unconditional below.
    auto vlog = [&](const char * fmt, auto... args) {
        if (!verbose) return;
        // `fmt` is always a string literal at call sites but the compiler
        // can't prove that through the variadic lambda.  Android NDK's
        // default `-Werror=format-security` (together with `_FORTIFY_SOURCE=2`)
        // then refuses to build unless we silence the warning locally.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
        fprintf(stderr, fmt, args...);
#pragma GCC diagnostic pop
    };

    int n_threads = opts.n_threads;
    if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    g_n_threads = n_threads;
    vlog("Using %d threads\n", g_n_threads);

    // Cooperative cancellation: cheap acquire-load on every checkpoint.
    // Returns true once *opts.cancel_flag becomes true (or stays false
    // forever when the caller didn't pass one).  Implemented as a
    // lambda so the no-cancel-flag path stays one branch deep.
    auto is_cancelled = [&]() -> bool {
        return opts.cancel_flag != nullptr &&
               opts.cancel_flag->load(std::memory_order_acquire);
    };

    if (gguf_path.empty()) {
        fprintf(stderr, "s3gen_synthesize_to_wav: s3gen_gguf_path is required\n");
        return 1;
    }
    if (out_path.empty() && opts.pcm_out == nullptr) {
        fprintf(stderr, "s3gen_synthesize_to_wav: at least one of out_wav_path or pcm_out must be set\n");
        return 1;
    }
    if (debug_mode && ref_dir.empty()) {
        fprintf(stderr, "--debug requires --ref-dir (Python-dumped intermediate tensors)\n");
        return 1;
    }
    if (speech_tokens.empty()) {
        fprintf(stderr, "s3gen_synthesize_to_wav: speech_tokens is empty\n");
        return 1;
    }

    // Reference conditioning: prefer GGUF-embedded built-in voice;
    // fall back to .npy files if --ref-dir was provided.
    // Layout: emb (192,) float32; pt (N,) int32; pf (mel_len1, 80) float32.
    std::vector<float>   emb_data;
    std::vector<int32_t> pt_data;
    std::vector<float>   pf_data;
    int pf_rows = 0;  // mel_len1

    // The view fields take precedence over the *_override vectors so a
    // streaming host (chatterbox::Engine) doesn't re-copy MB-sized
    // prompt_feat / embedding tensors per chunk; the vectors stay
    // around for callers that hold their data on the opts struct
    // itself (tts-cli).
    const bool has_prompt_token = (opts.prompt_token_view_data && opts.prompt_token_view_size > 0)
                                  || !opts.prompt_token_override.empty();
    const bool has_embedding    = (opts.embedding_view_data && opts.embedding_view_size > 0)
                                  || !opts.embedding_override.empty();
    const bool has_prompt_feat  = (opts.prompt_feat_view_data && opts.prompt_feat_view_size > 0)
                                  || !opts.prompt_feat_override.empty();

    if (ref_dir.empty() && !has_embedding && !has_prompt_feat && !has_prompt_token) {
        vlog("No --ref-dir given; loading built-in voice from GGUF.\n");
    } else {
        if (!ref_dir.empty()) vlog("Loading ref dict from %s\n", ref_dir.c_str());

        if (opts.prompt_token_view_data && opts.prompt_token_view_size > 0) {
            pt_data.assign(opts.prompt_token_view_data,
                           opts.prompt_token_view_data + opts.prompt_token_view_size);
            vlog("  prompt_token: using C++ override view (%zu tokens, single-copy view path)\n",
                    pt_data.size());
        } else if (!opts.prompt_token_override.empty()) {
            pt_data = opts.prompt_token_override;
            vlog("  prompt_token: using C++ override (S3TokenizerV2, %zu tokens)\n",
                    pt_data.size());
        } else {
            npy_array pt_npy = npy_load(ref_dir + "/prompt_token.npy");
            pt_data.assign((const int32_t*)pt_npy.data.data(),
                           (const int32_t*)pt_npy.data.data() + pt_npy.n_elements());
        }

        if (opts.embedding_view_data && opts.embedding_view_size > 0) {
            emb_data.assign(opts.embedding_view_data,
                            opts.embedding_view_data + opts.embedding_view_size);
            vlog("  embedding: using C++ override view (%zu dims, single-copy view path)\n",
                    emb_data.size());
        } else if (!opts.embedding_override.empty()) {
            emb_data = opts.embedding_override;
            vlog("  embedding: using C++ override (CAMPPlus, %zu dims)\n", emb_data.size());
        } else {
            npy_array emb_npy = npy_load(ref_dir + "/embedding.npy");
            emb_data.assign((const float*)emb_npy.data.data(),
                            (const float*)emb_npy.data.data() + emb_npy.n_elements());
        }

        if (opts.prompt_feat_view_data && opts.prompt_feat_view_size > 0) {
            pf_data.assign(opts.prompt_feat_view_data,
                           opts.prompt_feat_view_data + opts.prompt_feat_view_size);
            pf_rows = opts.prompt_feat_view_rows;
            vlog("  prompt_feat: using C++ override view (%d mel frames, single-copy view path)\n", pf_rows);
        } else if (!opts.prompt_feat_override.empty()) {
            pf_data = opts.prompt_feat_override;
            pf_rows = opts.prompt_feat_rows_override;
            vlog("  prompt_feat: using C++ override (%d mel frames)\n", pf_rows);
        } else {
            npy_array pf_npy = npy_load(ref_dir + "/prompt_feat.npy");
            pf_data.assign((const float*)pf_npy.data.data(),
                           (const float*)pf_npy.data.data() + pf_npy.n_elements());
            pf_rows = (int)pf_npy.shape[0];
        }
    }

    vlog("Speech tokens: %zu\n", speech_tokens.size());

    // Trim tokens >= vocab_size.  In batch / last-chunk (`finalize=true`) we
    // append 3 S3GEN_SIL lookahead tokens to give the encoder right-context
    // for the true ending.  For streaming chunks (`finalize=false`) we skip
    // that: the lookahead will come from real speech tokens in the next
    // chunk, and we'll trim the 6 mel frames corresponding to the pre-
    // lookahead window right after CFM.
    const int32_t S3GEN_SIL = tts_cpp::chatterbox::kS3GenSilenceToken;
    const int32_t VOCAB_SIZE = 6561;
    std::vector<int32_t> padded;
    for (int32_t t : speech_tokens) {
        if (t >= 0 && t < VOCAB_SIZE) padded.push_back(t);
    }
    if (opts.append_lookahead_silence) {
        for (int i = 0; i < pre_lookahead_len; ++i) padded.push_back(S3GEN_SIL);
    }

    // Cache the loaded model across invocations so the streaming driver
    // pays the ~700 ms GGUF-load cost only once.  Keyed on (path,
    // n_gpu_layers) so switching backends still works.  Verbose gates the
    // banner prints but the BENCH line always goes out for perf checks.
    model_ctx & m = *s3gen_model_cache_get(gguf_path, opts.n_gpu_layers, verbose);
    {
        const double load_ms = s3gen_model_cache_last_load_ms();
        // Only emit the BENCH line on an actual GGUF load — on cache hits
        // the value is always 0 and repeating it per chunk adds noise.
        if (load_ms > 0.0 || verbose) {
            fprintf(stderr, "BENCH: S3GEN_LOAD_MS=%.0f\n", load_ms);
        }
    }

    // HiFT-side graphs (f0_predictor, STFT, hift_decode) used to need a
    // dedicated CPU copy of the S3Gen GGUF on Metal because conv_transpose_1d,
    // pad_ext and diag_mask_inf were either missing or pathologically slow in
    // ggml-metal.  After the kernel fixes in ggml/src/ggml-metal/, HiFT runs
    // on the main backend directly on every platform.
    const model_ctx & m_hift = m;

    // If neither --ref-dir nor any C++ override populated the three
    // conditioning tensors above, pull the built-in voice from the GGUF.
    //
    // NB: `ref_dir.empty()` alone is NOT a valid check here — --reference-audio
    // by itself (no --ref-dir) legitimately fills all three via the C++
    // override path, and we must not overwrite those.
    if (pt_data.empty() && emb_data.empty() && pf_data.empty()) {
        ggml_tensor * t_emb = find_tensor(m, "s3gen/builtin/embedding");
        ggml_tensor * t_pt  = find_tensor(m, "s3gen/builtin/prompt_token");
        ggml_tensor * t_pf  = find_tensor(m, "s3gen/builtin/prompt_feat");
        emb_data.resize(ggml_nelements(t_emb));
        pt_data.resize(ggml_nelements(t_pt));
        pf_data.resize(ggml_nelements(t_pf));
        ggml_backend_tensor_get(t_emb, emb_data.data(), 0, ggml_nbytes(t_emb));
        ggml_backend_tensor_get(t_pt,  pt_data.data(),  0, ggml_nbytes(t_pt));
        ggml_backend_tensor_get(t_pf,  pf_data.data(),  0, ggml_nbytes(t_pf));
        // prompt_feat is stored ggml ne=[80, 500] = numpy (500, 80).
        pf_rows = (int)t_pf->ne[1];
        vlog("  built-in voice: embedding=(%zu,) prompt_token=(%zu,) prompt_feat=(%d, %lld)\n",
                emb_data.size(), pt_data.size(), pf_rows, (long long)t_pf->ne[0]);
    }
    double pipeline_t0 = now_ms();

    const int D = 512;
    const int MEL = 80;

    // 1) Concat prompt_token + padded speech_tokens
    int n_prompt = (int)pt_data.size();
    int n_total = n_prompt + (int)padded.size();
    vlog("n_prompt=%d n_speech_padded=%zu n_total=%d\n", n_prompt, padded.size(), n_total);

    std::vector<int32_t> flow_tokens(n_total);
    std::memcpy(flow_tokens.data(), pt_data.data(), n_prompt * sizeof(int32_t));
    std::memcpy(flow_tokens.data() + n_prompt, padded.data(), padded.size() * sizeof(int32_t));

    // 2) input_embedding lookup + multiply by mask
    vlog("Running input_embedding...\n");
    ggml_tensor * emb_w = find_tensor(m, "flow/input_embedding");
    // input_embedding weight is multiple MB on Turbo and ~28 MB on
    // multilingual (vocab=13632 × D=512 × 4 B).  Without caching, each
    // synth call pays the full GPU→CPU download (~600-1000 µs wall on
    // RTX 5090).  Cache the CPU mirror so subsequent calls only pay
    // the cheap row-copy lookup cost.  Cache is bound to the s3gen model
    // lifecycle.
    const float * emb_w_data = cached_cpu_weights_f32(emb_w);
    vlog("  emb_w ne=[%lld, %lld]\n", (long long)emb_w->ne[0], (long long)emb_w->ne[1]);
    int vocab_size = (int)emb_w->ne[1];
    std::vector<float> input_embed(n_total * D);
    for (int i = 0; i < n_total; ++i) {
        int32_t tok = flow_tokens[i];
        if (tok < 0) tok = 0;
        if (tok >= vocab_size) {
            fprintf(stderr, "warning: token %d out of range (vocab=%d), clamping\n", tok, vocab_size);
            tok = vocab_size - 1;
        }
        std::memcpy(input_embed.data() + i * D, emb_w_data + (size_t)tok * D, D * sizeof(float));
    }
    if (debug_mode) {
        fprintf(stderr, "  token[0]=%d lookup: %.6f %.6f %.6f %.6f %.6f\n",
                flow_tokens[0],
                input_embed[0], input_embed[1], input_embed[2], input_embed[3], input_embed[4]);
    }

    if (is_cancelled()) {
        vlog("synthesis cancelled before encoder\n");
        return 2;
    }

    // 3) Run encoder -> mu_T (numpy (T_mu, 80) layout, to match encoder_proj.npy)
    vlog("Running encoder (T=%d)...\n", n_total);
    double encoder_t0 = now_ms();
    std::vector<float> mu_T = run_encoder(m, input_embed, n_total, D);
    vlog("  [encoder] %.1f ms\n", now_ms() - encoder_t0);
    int T_mu = 2 * n_total;
    vlog("  encoder output: (%d, 80) = %zu floats\n", T_mu, mu_T.size());

    // Streaming: trim the last `pre_lookahead_len * token_mel_ratio = 6`
    // frames off the encoder output BEFORE CFM (matches Python's
    // flow.inference(finalize=False) which does `h = h[:, :-6]` pre-
    // decoder).  Doing it here — not post-CFM — keeps mu / cond / z and
    // CFM's internal attention all sized consistently with Python, so
    // a Python-dumped noise tensor produces bit-exact mel output in C++.
    const int TOKEN_MEL_RATIO_PRE = 2;  // each speech token expands to 2 mels
    if (!opts.finalize) {
        const int trim = pre_lookahead_len * TOKEN_MEL_RATIO_PRE;  // 6
        if (T_mu <= trim) {
            fprintf(stderr, "error: streaming chunk too short (T_mu=%d ≤ trim=%d)\n", T_mu, trim);
            return 1;
        }
        T_mu -= trim;
        mu_T.resize((size_t)T_mu * MEL);  // numpy layout (T_mu, 80): drop last `trim` rows
        vlog("  streaming: trimmed %d mel frames -> T_mu=%d\n", trim, T_mu);
    }

    if (debug_mode) {
        npy_array ref_ie = npy_load(ref_dir + "/input_embedded.npy");
        const float * ref = (const float*)ref_ie.data.data();
        size_t n = std::min(input_embed.size(), ref_ie.n_elements());
        float ma = 0, rsum = 0;
        for (size_t i = 0; i < n; ++i) {
            float d = input_embed[i] - ref[i];
            ma = std::max(ma, std::fabs(d));
            rsum += d * d;
        }
        fprintf(stderr, "  [input_embed] max_abs=%.4e rms=%.4e vs ref\n", ma, std::sqrt(rsum / n));

        npy_array ref_mu = npy_load(ref_dir + "/encoder_proj.npy");
        const float * mref = (const float*)ref_mu.data.data();
        size_t mn = std::min(mu_T.size(), ref_mu.n_elements());
        ma = 0; rsum = 0;
        for (size_t i = 0; i < mn; ++i) {
            float d = mu_T[i] - mref[i];
            ma = std::max(ma, std::fabs(d));
            rsum += d * d;
        }
        fprintf(stderr, "  [mu_T (before transpose)] max_abs=%.4e rms=%.4e vs encoder_proj.npy\n", ma, std::sqrt(rsum / mn));
    }

    // Transpose mu_T from numpy (T_mu, 80) layout to numpy (80, T_mu) for CFM.
    // In memory: mu_T has [t0_m0, t0_m1, ..., t0_m79, t1_m0, ...]
    //            mu should be [m0_t0, m0_t1, ..., m0_tTmu-1, m1_t0, ...]
    std::vector<float> mu(T_mu * MEL);
    for (int m2 = 0; m2 < MEL; ++m2)
        for (int t = 0; t < T_mu; ++t)
            mu[m2 * T_mu + t] = mu_T[t * MEL + m2];

    // Streaming debug: optionally dump the encoder_proj output (same as
    // Python's `mu` tensor fed into flow_matching.forward) so the test
    // harness can isolate encoder-side vs CFM-side divergence.
    if (!opts.dump_mel_path.empty()) {
        std::string base = opts.dump_mel_path;
        if (base.size() > 4 && base.substr(base.size() - 4) == ".npy")
            base.resize(base.size() - 4);
        npy_save_f32(base + "_mu.npy", {(int64_t)T_mu, MEL}, mu_T.data());
    }

    // 4) Speaker embedding: F.normalize + spk_embed_affine_layer
    vlog("Computing speaker embedding...\n");
    const float * emb_raw = emb_data.data();
    float norm = 0.0f;
    for (int i = 0; i < 192; ++i) norm += emb_raw[i] * emb_raw[i];
    norm = std::sqrt(norm + 1e-12f);
    std::vector<float> emb_norm(192);
    for (int i = 0; i < 192; ++i) emb_norm[i] = emb_raw[i] / norm;

    ggml_tensor * saw = find_tensor(m, "flow/spk_embed_affine/w");  // (80, 192) numpy -> ne=[192, 80]
    ggml_tensor * sab = find_tensor(m, "flow/spk_embed_affine/b");  // (80,)
    // Cache CPU mirrors of the speaker-affine weights (~60 KB) instead
    // of paying GPU→CPU download per synth.
    const float * saw_data = cached_cpu_weights_f32(saw);
    const float * sab_data = cached_cpu_weights_f32(sab);
    std::vector<float> spks(MEL, 0.0f);
    for (int o = 0; o < MEL; ++o) {
        float acc = sab_data[o];
        for (int i = 0; i < 192; ++i) acc += saw_data[o * 192 + i] * emb_norm[i];
        spks[o] = acc;
    }

    // 5) Build cond: zeros(T_mu, 80), fill first mel_len1 rows with prompt_feat
    int mel_len1 = pf_rows;
    if (mel_len1 > T_mu) {
        fprintf(stderr, "error: mel_len1=%d > T_mu=%d\n", mel_len1, T_mu);
        return 1;
    }
    std::vector<float> cond(T_mu * MEL, 0.0f);
    // pf is (mel_len1, 80) numpy = ne=[80, mel_len1] in ggml. We want cond ne=[T_mu, MEL].
    // In memory ggml ne=[T_mu, MEL] means [t0_m0, t1_m0, ..., t_Tmu-1_m0, t0_m1, ...].
    // So cond[m, t] = pf[t, m].
    const float * pf_raw = pf_data.data();
    for (int m2 = 0; m2 < MEL; ++m2)
        for (int t = 0; t < mel_len1; ++t)
            cond[m2 * T_mu + t] = pf_raw[t * MEL + m2];

    // Streaming debug: dump spks + cond so we can compare against Python's
    // flow.decoder input tensors chunk-by-chunk.
    if (!opts.dump_mel_path.empty()) {
        std::string base = opts.dump_mel_path;
        if (base.size() > 4 && base.substr(base.size() - 4) == ".npy")
            base.resize(base.size() - 4);
        npy_save_f32(base + "_spks.npy", {MEL}, spks.data());
        // C++ stores cond in ggml ne=[T_mu, MEL] layout (T_mu innermost) which
        // Python sees as numpy shape (MEL, T_mu).  Dump in that layout so we
        // can diff directly against Python's (80, T_mu) decoder-input cond.
        npy_save_f32(base + "_cond.npy", {MEL, (int64_t)T_mu}, cond.data());
        fprintf(stderr, "  [stream] dumped spks (%d,) cond (%d, %d) → %s_{spks,cond}.npy\n",
                MEL, MEL, T_mu, base.c_str());
    }

    if (debug_mode) {
        npy_array ref = npy_load(ref_dir + "/cfm_step0_cond.npy");
        const float * r = (const float*)ref.data.data();
        size_t n = std::min(cond.size(), ref.n_elements());
        float ma = 0, rsum = 0;
        for (size_t i = 0; i < n; ++i) { float d = cond[i] - r[i]; ma = std::max(ma, std::fabs(d)); rsum += d*d; }
        fprintf(stderr, "  [cond] max_abs=%.4e rms=%.4e vs ref\n", ma, std::sqrt(rsum / n));

        npy_array ref_s = npy_load(ref_dir + "/cfm_step0_spks.npy");
        const float * rs = (const float*)ref_s.data.data();
        size_t ns = std::min(spks.size(), ref_s.n_elements());
        ma = 0; rsum = 0;
        for (size_t i = 0; i < ns; ++i) { float d = spks[i] - rs[i]; ma = std::max(ma, std::fabs(d)); rsum += d*d; }
        fprintf(stderr, "  [spks] max_abs=%.4e rms=%.4e vs ref\n", ma, std::sqrt(rsum / ns));

        // Also compare mu vs cfm_step0_mu.npy (which should equal encoder_proj)
        npy_array ref_mu = npy_load(ref_dir + "/cfm_step0_mu.npy");
        const float * rm = (const float*)ref_mu.data.data();
        size_t nm = std::min(mu.size(), ref_mu.n_elements());
        ma = 0; rsum = 0;
        for (size_t i = 0; i < nm; ++i) { float d = mu[i] - rm[i]; ma = std::max(ma, std::fabs(d)); rsum += d*d; }
        fprintf(stderr, "  [mu vs step0_mu] max_abs=%.4e rms=%.4e vs ref\n", ma, std::sqrt(rsum / nm));
    }

    // 6) Initial CFM noise.  Layout mirrors Python: z has shape (80, T_mu).
    //
    //  - meanflow (Turbo):   randn everywhere, then replace the speech region
    //                        with a second independent randn draw (matches
    //                        `flow_matching.forward`'s `noised_mels` overlay).
    //  - standard CFM (MTL): randn(80, T_mu) once; no overlay.  The speech
    //                        region is implicitly conditioned through `cond`
    //                        instead of via an extra noise tensor.
    const bool meanflow = m.meanflow;
    vlog("Initializing CFM noise (seed=%d, %s)...\n", seed,
            meanflow ? "meanflow" : "standard CFM + CFG");
    std::vector<float> z(T_mu * MEL);
    int n_speech_part = 2 * (int)padded.size();
    int prompt_len_in_mu = T_mu - n_speech_part;
    vlog("  T_mu=%d prompt_len_in_mu=%d n_speech_part=%d\n",
            T_mu, prompt_len_in_mu, n_speech_part);
    if (!opts.cfm_z0_override.empty()) {
        if ((int64_t)opts.cfm_z0_override.size() != (int64_t)T_mu * MEL) {
            fprintf(stderr, "error: cfm_z0_override has %zu elements but T_mu*MEL=%d\n",
                    opts.cfm_z0_override.size(), T_mu * MEL);
            return 1;
        }
        std::memcpy(z.data(), opts.cfm_z0_override.data(), z.size() * sizeof(float));
        fprintf(stderr, "  [stream] loaded %zu elems of CFM z0 from cfm_z0_override\n",
                opts.cfm_z0_override.size());
    } else if (debug_mode && meanflow) {
        npy_array z_npy = npy_load(ref_dir + "/cfm_z0_raw.npy");
        std::memcpy(z.data(), z_npy.data.data(), z.size() * sizeof(float));
        fprintf(stderr, "  [debug] loaded z from cfm_z0_raw.npy\n");
        npy_array nm_npy = npy_load(ref_dir + "/cfm_noised_mels.npy");
        const float * nm = (const float*)nm_npy.data.data();
        int nm_T = (int)nm_npy.shape[1];
        fprintf(stderr, "  [debug] overlay noised_mels (80, %d) at pos %d\n", nm_T, prompt_len_in_mu);
        for (int m2 = 0; m2 < MEL; ++m2)
            for (int t = 0; t < nm_T; ++t)
                z[m2 * T_mu + (prompt_len_in_mu + t)] = nm[m2 * nm_T + t];
    } else {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (size_t i = 0; i < z.size(); ++i) z[i] = gauss(rng);
        if (meanflow) {
            std::mt19937 rng2(seed + 2);
            std::normal_distribution<float> gauss2(0.0f, 1.0f);
            for (int m2 = 0; m2 < MEL; ++m2)
                for (int t = prompt_len_in_mu; t < T_mu; ++t)
                    z[m2 * T_mu + t] = gauss2(rng2);
        }
    }

    // 7) CFM loop.
    //  - meanflow:      t_span linearly spaced on [0,1], default 2 steps,
    //                   one estimator call per step, t_emb mixed with r via
    //                   the meanflow-only time_embed_mixer.
    //  - standard CFM:  t_span cosine-scheduled, default 10 steps, two
    //                   estimator calls per step (cond + uncond with zeroed
    //                   mu/spks/cond) combined via cfg_rate.
    std::vector<float> t_span;
    const int cfm_steps = opts.cfm_steps > 0 ? opts.cfm_steps :
                          (meanflow ? 2 : m.n_timesteps);
    t_span.reserve(cfm_steps + 1);
    for (int i = 0; i <= cfm_steps; ++i) {
        float tau = (float)i / (float)cfm_steps;
        if (!meanflow) {
            tau = 1.0f - std::cos(tau * 0.5f * (float)M_PI);
        }
        t_span.push_back(tau);
    }

    const float cfg_rate = m.cfg_rate;
    const std::vector<float> zero_mu  (T_mu * MEL, 0.0f);
    const std::vector<float> zero_cond(T_mu * MEL, 0.0f);
    const std::vector<float> zero_spks(MEL, 0.0f);
    // Pack cond + uncond into one batch=2 forward call on GPU backends so
    // their per-dispatch overhead amortises across both passes.  On ggml-cpu
    // the dispatch overhead is already ~zero and the extra permute+cont ops
    // that a batched attention block needs in each layer actually regress
    // throughput (measured +11% S3Gen wall time on M4 CPU), so we keep the
    // two-call path there.  Meanflow has no CFG to begin with.
    const bool use_b2 = (!meanflow) && (cfg_rate != 0.0f) &&
                        !ggml_backend_is_cpu(m.backend);

    // Persistent CFM estimator graph cache.  Re-used across synth
    // calls when T matches — multi-synth chunks 2..N skip the graph
    // build + gallocr_reserve cost.  Lifetime managed by
    // s3gen_model_cache_release.  Works for both batch=1 (Turbo) and
    // batch=2 (multilingual CFG) paths via the cache.b2 flag.
    cfm_estimator_cache & cfm_cache = g_cfm_estimator_cache;
    double cfm_t0 = now_ms();
    for (size_t s = 0; s < t_span.size() - 1; ++s) {
        if (is_cancelled()) {
            vlog("synthesis cancelled at CFM step %zu\n", s);
            return 2;
        }
        float t = t_span[s], r = t_span[s + 1];
        float dt = r - t;
        vlog("CFM step %zu: t=%g r=%g dt=%g...\n", s, t, r, dt);
        // Memoised t-emb pipeline.  Same (t, r) pair always produces
        // the same vector (deterministic function of t, r, and the
        // model weights).  Both Turbo (meanflow) and multilingual
        // (standard) paths benefit; multilingual amortises
        // the cache better since it has 10 steps × 2 sets of {t, r}
        // values that repeat across every subsequent synth call.
        std::vector<float> t_emb;
        if (meanflow) {
            t_emb = compute_time_emb_cached(m, t, r);
        } else {
            t_emb = compute_time_mlp_cached(m, t);
        }

        if (debug_mode && meanflow) {
            npy_array ref = npy_load(ref_dir + "/cfm_t_mix_call" + std::to_string(s) + ".npy");
            const float * r_ = (const float*)ref.data.data();
            size_t n = std::min(t_emb.size(), ref.n_elements());
            float ma = 0, rsum = 0;
            for (size_t i = 0; i < n; ++i) { float d = t_emb[i] - r_[i]; ma = std::max(ma, std::fabs(d)); rsum += d*d; }
            fprintf(stderr, "    [t_emb step%zu] max_abs=%.4e rms=%.4e vs ref\n", s, ma, std::sqrt(rsum / n));

            npy_array ref_x = npy_load(ref_dir + "/cfm_step" + std::to_string(s) + "_x_in.npy");
            const float * rx = (const float*)ref_x.data.data();
            size_t nx = std::min(z.size(), ref_x.n_elements());
            ma = 0; rsum = 0;
            for (size_t i = 0; i < nx; ++i) { float d = z[i] - rx[i]; ma = std::max(ma, std::fabs(d)); rsum += d*d; }
            fprintf(stderr, "    [x_in step%zu] max_abs=%.4e rms=%.4e vs ref\n", s, ma, std::sqrt(rsum / nx));
        }

        double step_t0 = now_ms();
        std::vector<float> dxdt_cond;
        std::vector<float> dxdt_uncond;
        // True when this step needs the CFG combine — both flavours of
        // CFG path (B=2 batched and B=1 two-call) populate dxdt_uncond
        // and require the linear `(1+cfg)*cond - cfg*uncond` mix.
        bool have_cfg_uncond = false;
        if (use_b2) {
            cfm_estimator_forward_b2(m, cfm_cache,
                z, z,
                mu, zero_mu,
                t_emb, t_emb,
                spks, zero_spks,
                cond, zero_cond,
                dxdt_cond, dxdt_uncond, T_mu, opts.cfm_f16_kv_attn);
            have_cfg_uncond = true;
        } else if (!meanflow && cfg_rate != 0.0f) {
            // Non-Metal CFG path (CPU + any backend where use_b2 is false).
            // Run the conditional and unconditional passes back-to-back on
            // the same B=1 graph (cfm_estimator_cache key (T, b2=false)
            // means both calls reuse the same cached graph) and combine
            // with the standard CFG mix.  Restoring this branch fixes a
            // silent regression introduced when the b2 path landed on Metal:
            // previously the else clause computed only the conditional pass
            // and dropped CFG entirely on every non-Metal backend.
            dxdt_cond = cfm_estimator_forward(m, cfm_cache, z, mu, t_emb, spks, cond, T_mu, opts.cfm_f16_kv_attn);
            dxdt_uncond = cfm_estimator_forward(m, cfm_cache, z, zero_mu, t_emb, zero_spks, zero_cond, T_mu, opts.cfm_f16_kv_attn);
            have_cfg_uncond = true;
        } else {
            dxdt_cond = cfm_estimator_forward(m, cfm_cache, z, mu, t_emb, spks, cond, T_mu, opts.cfm_f16_kv_attn);
        }

        // Debug + dump hooks read the post-CFG-combine dxdt; precompute it
        // when the caller actually asks for it, otherwise fold the combine
        // into the Euler step below to save a pass over the array.
        const bool need_full_dxdt = (debug_mode && meanflow) ||
                                    (s == 0 && !opts.dump_mel_path.empty());
        if (have_cfg_uncond && need_full_dxdt) {
            for (size_t i = 0; i < dxdt_cond.size(); ++i) {
                dxdt_cond[i] = (1.0f + cfg_rate) * dxdt_cond[i] - cfg_rate * dxdt_uncond[i];
            }
        }
        auto & dxdt = dxdt_cond;
        vlog("  [cfm_step%zu] %.1f ms\n", s, now_ms() - step_t0);

        if (debug_mode && meanflow) {
            npy_array ref = npy_load(ref_dir + "/cfm_step" + std::to_string(s) + "_dxdt.npy");
            const float * r_ = (const float*)ref.data.data();
            size_t n = std::min(dxdt.size(), ref.n_elements());
            float ma = 0, rsum = 0;
            for (size_t i = 0; i < n; ++i) { float d = dxdt[i] - r_[i]; ma = std::max(ma, std::fabs(d)); rsum += d*d; }
            fprintf(stderr, "    [dxdt step%zu] max_abs=%.4e rms=%.4e vs ref\n", s, ma, std::sqrt(rsum / n));
        }

        if (s == 0 && !opts.dump_mel_path.empty()) {
            std::string base = opts.dump_mel_path;
            if (base.size() > 4 && base.substr(base.size() - 4) == ".npy")
                base.resize(base.size() - 4);
            npy_save_f32(base + "_step0_dxdt.npy", {MEL, (int64_t)T_mu}, dxdt.data());
            fprintf(stderr, "  [stream] dumped step0_dxdt (%d, %d) → %s_step0_dxdt.npy\n",
                    MEL, T_mu, base.c_str());
        }

        // Fused CFG-combine + Euler step.  Saves one pass over
        // `dxdt` per step.  When the debug/dump code-paths above
        // already wrote the combined result back into `dxdt_cond`, we
        // detect it via `need_full_dxdt && have_cfg_uncond` and fall back
        // to the plain `z + dt * dxdt_cond` form so the math stays
        // bit-exact across both branches.
        if (have_cfg_uncond && !need_full_dxdt) {
            const float c1 = (1.0f + cfg_rate);
            const float c0 = -cfg_rate;
            for (size_t i = 0; i < z.size(); ++i) {
                const float d = c1 * dxdt_cond[i] + c0 * dxdt_uncond[i];
                z[i] = z[i] + dt * d;
            }
        } else {
            for (size_t i = 0; i < z.size(); ++i) z[i] = z[i] + dt * dxdt[i];
        }
    }
    vlog("  [cfm_total] %.1f ms\n", now_ms() - cfm_t0);

    // 8) Slice mel = z[:, mel_len1:] -> shape (80, T_mu - mel_len1).
    //
    // For streaming (finalize=false), also drop the last
    //   pre_lookahead_len * token_mel_ratio = 3 * 2 = 6 mel frames
    // — these aren't "safe" yet (they'd change once the next chunk's tokens
    // provide more right-context).  Matches the trim in Python
    // CausalMaskedDiffWithXvec.inference(..., finalize=False).
    // Beyond-prompt mel span: starts at mel_len1 in z, ends at T_mu.
    // Streaming's 6-frame tail trim is already baked into T_mu above (we
    // shrunk it pre-CFM, matching Python).  Here we only apply the caller's
    // skip offset to drop frames already emitted on prior chunks.
    int T_mel_full = T_mu - mel_len1;
    int skip = opts.skip_mel_frames;
    if (skip < 0) skip = 0;
    if (skip > T_mel_full) {
        fprintf(stderr, "error: skip_mel_frames=%d > T_mel_full=%d\n", skip, T_mel_full);
        return 1;
    }
    int T_mel = T_mel_full - skip;
    vlog("Mel slicing: T_mu=%d mel_len1=%d full=%d skip=%d -> T_mel=%d  "
                    "(finalize=%s, pad_silence=%s)\n",
            T_mu, mel_len1, T_mel_full, skip, T_mel,
            opts.finalize ? "true" : "false",
            opts.append_lookahead_silence ? "true" : "false");
    std::vector<float> mel(MEL * T_mel);
    // z has shape ne=[T_mu, MEL]; grab the slice [mel_len1 + skip, mel_len1 + skip + T_mel).
    const int mel_off = mel_len1 + skip;
    for (int m2 = 0; m2 < MEL; ++m2)
        for (int t = 0; t < T_mel; ++t)
            mel[m2 * T_mel + t] = z[m2 * T_mu + (t + mel_off)];

    if (!opts.dump_mel_path.empty()) {
        std::vector<float> mel_tn((size_t)T_mel * MEL);
        for (int t = 0; t < T_mel; ++t)
            for (int m2 = 0; m2 < MEL; ++m2)
                mel_tn[(size_t)t * MEL + m2] = mel[(size_t)m2 * T_mel + t];
        npy_save_f32(opts.dump_mel_path, {(int64_t)T_mel, MEL}, mel_tn.data());
        fprintf(stderr, "  dumped mel (%d, %d) → %s\n", T_mel, MEL, opts.dump_mel_path.c_str());
    }

    if (debug_mode) {
        npy_array ref_mel = npy_load(ref_dir + "/mel_output.npy");
        const float * r = (const float*)ref_mel.data.data();
        size_t n = std::min(mel.size(), ref_mel.n_elements());
        float ma = 0, rsum = 0, max_ref = 0;
        for (size_t i = 0; i < n; ++i) {
            float d = mel[i] - r[i];
            ma = std::max(ma, std::fabs(d));
            rsum += d * d;
            max_ref = std::max(max_ref, std::fabs(r[i]));
        }
        fprintf(stderr, "  [mel] max_abs=%.4e rms=%.4e max|ref|=%.4e rel=%.4e\n",
                ma, std::sqrt(rsum / n), max_ref, ma / std::max(max_ref, 1e-9f));
    }

    // 9) HiFT
    double hift_t0 = now_ms();
    vlog("Running f0_predictor...\n");
    double t0 = now_ms();
    auto f0 = run_f0_predictor(m_hift, mel, T_mel);
    vlog("  [f0_predictor] %.1f ms\n", now_ms() - t0);
    int upsample = 8 * 5 * 3 * 4;
    int T_wav = T_mel * upsample;
    std::vector<float> f0_up(T_wav);
    for (int i = 0; i < T_mel; ++i)
        for (int j = 0; j < upsample; ++j) f0_up[i * upsample + j] = f0[i];

    vlog("Running SineGen...\n");
    t0 = now_ms();
    std::vector<float> l_linear_w(9);
    ggml_tensor * llw = find_tensor(m_hift, "hift/m_source/l_linear/weight");
    ggml_tensor * llb = find_tensor(m_hift, "hift/m_source/l_linear/bias");
    ggml_backend_tensor_get(llw, l_linear_w.data(), 0, 9 * sizeof(float));
    float l_linear_b;
    ggml_backend_tensor_get(llb, &l_linear_b, 0, sizeof(float));
    auto src = sinegen_source(f0_up, sr, 8, 0.1f, 0.003f, 10.0f, l_linear_w, l_linear_b, (uint32_t)(seed + 1));
    vlog("  [sinegen] %.1f ms\n", now_ms() - t0);

    // Streaming: splice in the previous chunk's source tail so the F0 phase
    // (and hence the vocoded waveform) stays continuous at the chunk seam.
    // Python equivalent: `s[:, :, :cache_source.shape[2]] = cache_source`
    // inside HiFTGenerator.inference.
    if (!opts.hift_cache_source.empty()) {
        size_t n = std::min(opts.hift_cache_source.size(), src.size());
        std::memcpy(src.data(), opts.hift_cache_source.data(), n * sizeof(float));
        vlog("  [sinegen] spliced %zu cache_source samples at head\n", n);
    }
    // Export the tail for the next chunk's cache_source BEFORE running STFT
    // (Python returns `s` unmodified; our tail copy matches that convention).
    if (opts.hift_source_tail_out != nullptr && opts.source_tail_samples > 0) {
        int tail_n = std::min((int)src.size(), opts.source_tail_samples);
        opts.hift_source_tail_out->assign(src.end() - tail_n, src.end());
    }

    if (is_cancelled()) {
        vlog("synthesis cancelled before STFT\n");
        return 2;
    }

    vlog("Running STFT...\n");
    t0 = now_ms();
    auto s_stft = run_stft(m_hift, src);
    vlog("  [stft] %.1f ms\n", now_ms() - t0);
    int T_stft = (int)(s_stft.size() / 18);

    if (is_cancelled()) {
        vlog("synthesis cancelled before HiFT decode\n");
        return 2;
    }

    vlog("Running HiFT decode...\n");
    t0 = now_ms();
    auto wav = run_hift_decode(m_hift, mel, T_mel, s_stft, T_stft);
    vlog("  [hift_decode] %.1f ms\n", now_ms() - t0);
    vlog("  [hift_total] %.1f ms\n", now_ms() - hift_t0);
    vlog("  wav: %zu samples (%.3fs @ %d Hz)\n", wav.size(), (float)wav.size() / sr, sr);

    // First-chunk / batch-mode: apply raised-cosine fade-in to mask HiFT's
    // resnet cold start.  Length = 2*(sr/50) = 960 samples (40 ms) at 24 kHz.
    // First half is zero, second half is (cos(π→0)+1)/2 (0→1 ramp).
    // Python equivalent: `output_wavs[:, :len(self.trim_fade)] *= self.trim_fade`.
    if (opts.apply_trim_fade) {
        const int n_trim = sr / 50;  // 480 at 24 kHz
        const int fade_len = 2 * n_trim;
        if ((int)wav.size() >= fade_len) {
            for (int i = 0; i < n_trim; ++i) wav[i] = 0.0f;
            for (int i = 0; i < n_trim; ++i) {
                float theta = (float)M_PI * (1.0f - (float)i / (float)n_trim);
                float w = 0.5f * (std::cos(theta) + 1.0f);
                wav[n_trim + i] *= w;
            }
        }
    }

    double pipeline_total = now_ms() - pipeline_t0;
    double audio_ms = 1000.0 * wav.size() / sr;
    fprintf(stderr, "BENCH: S3GEN_INFER_MS=%.0f AUDIO_MS=%.0f\n", pipeline_total, audio_ms);
    vlog("\n=== pipeline: %.1f ms for %.1f ms of audio (RTF=%.2f, %.1fx %s) ===\n",
         pipeline_total, audio_ms,
         pipeline_total / audio_ms,
         audio_ms / pipeline_total >= 1.0 ? audio_ms / pipeline_total : pipeline_total / audio_ms,
         audio_ms >= pipeline_total ? "faster than real-time" : "slower than real-time");

    if (opts.pcm_out) {
        *opts.pcm_out = wav;
    }
    if (!out_path.empty()) {
        write_wav(out_path, wav, sr);
        fprintf(stderr, "Wrote %s\n", out_path.c_str());
    }
    return 0;
}

int s3gen_preload(const std::string & s3gen_gguf_path, int n_gpu_layers) {
    try {
        (void)s3gen_model_cache_get(s3gen_gguf_path, n_gpu_layers, /*verbose=*/false);
        g_s3gen_cache_refcount.fetch_add(1, std::memory_order_acq_rel);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "s3gen_preload: %s\n", e.what());
        return 1;
    }
}

void s3gen_unload() {
    // Refcount-protected.  Multiple Engine instances can share the
    // cache via repeated s3gen_preload calls; only the last unload
    // actually releases the backend / weights.  Underflow (unload
    // with no matching preload) is clamped at zero and runs a release
    // anyway, which keeps idempotency for callers that follow the
    // historical 'preload-once / unload-once-no-matter-what' pattern.
    int prev = g_s3gen_cache_refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev > 1) {
        return;
    }
    if (prev <= 0) {
        // Reset to zero so subsequent preload/unload pairs continue to
        // work cleanly; without this, a second underflow would push
        // the count further negative and a future preload would never
        // be able to drive it back to zero.
        g_s3gen_cache_refcount.store(0, std::memory_order_release);
    }
    s3gen_model_cache_release();
}

// ============================================================================
// Internal test hooks
// ============================================================================
//
// Implementations of the read-only cache-state queries declared in
// chatterbox_tts_test_hooks.h.  Defined here so they sit in the same
// translation unit as the caches themselves and don't need any extra
// linkage gymnastics.

namespace tts_cpp::chatterbox::test_hooks {

size_t time_mlp_result_cache_size() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_time_mlp_results.size();
}
size_t time_emb_result_cache_size() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_time_emb_results.size();
}
size_t weight_mirror_cache_size() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_weight_cpu_mirror.size();
}
bool cfm_estimator_cache_built() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_cfm_estimator_cache.ctx != nullptr;
}
bool cfm_estimator_cache_b2() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_cfm_estimator_cache.b2;
}
uint32_t float_cache_key(float t_val) {
    return g_float_bits(t_val);
}
uint64_t float_pair_cache_key(float t_val, float r_val) {
    return g_float_pair_bits(t_val, r_val);
}
std::vector<float> peek_time_mlp_cached(float t_val) {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    auto it = g_time_mlp_results.find(g_float_bits(t_val));
    if (it == g_time_mlp_results.end()) return {};
    return it->second;
}

// ---- Round 2 hooks --------------------------------------------------------

bool encoder_graph_cache_built() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_encoder_graph_cache.ctx != nullptr;
}
int encoder_graph_cache_T() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return (int) g_encoder_graph_cache.key;
}
bool hift_graph_cache_built() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_hift_graph_cache.ctx != nullptr;
}
int hift_graph_cache_T_mel() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    if (g_hift_graph_cache.key < 0) return -1;
    return (int) (g_hift_graph_cache.key >> 32);
}
int hift_graph_cache_T_stft() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    if (g_hift_graph_cache.key < 0) return -1;
    return (int) (g_hift_graph_cache.key & 0xffffffffLL);
}
bool f0_graph_cache_built() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_f0_graph_cache.ctx != nullptr;
}
int f0_graph_cache_T_mel() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return (int) g_f0_graph_cache.key;
}
size_t pos_emb_cache_size() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_pos_emb_results.size();
}
size_t inv_alpha_cache_size() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_inv_alpha_results.size();
}
size_t istft_kernel_cache_size() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_istft_kernel_cache.size();
}
size_t hann_window_cache_size() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_hann_window_cache.size();
}
bool stft_graph_cache_built() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_stft_graph_cache.ctx != nullptr;
}
int stft_graph_cache_T_src() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return (int) g_stft_graph_cache.key;
}
size_t stft_kernel_cache_size() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_stft_kernel_cache.size();
}
size_t window_sum_cache_size() {
    std::lock_guard<std::mutex> lk(g_synth_caches_mu);
    return g_window_sum_cache.size();
}

}  // namespace tts_cpp::chatterbox::test_hooks
