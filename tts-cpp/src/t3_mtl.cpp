// Chatterbox multilingual T3 (Llama-520M) variant: loader + forward pass.
//
// Structural differences from the GPT-2 Medium Turbo variant in src/main.cpp:
//   - 30 layers vs 24, head_dim=64, n_kv_head=16 (MHA, not GQA).
//   - Pre-norm with RMSNorm (no bias) instead of LayerNorm.
//   - Rotary position embedding with llama3 scaling: freq_factors precomputed
//     at load time and applied through ggml_rope_ext's `c` param.
//   - SwiGLU MLP: SiLU(gate(x)) * up(x) -> down(x); three Linears per layer.
//   - Separate Q/K/V projections (no fused c_attn).
//   - Classifier-Free Guidance: each T3 graph runs twice per call, once for
//     the conditional (full text embeddings) batch element and once for the
//     unconditional one (text embeddings zeroed).  Two independent KV caches
//     live inside the model struct for this.  Logits are combined in the
//     sampler as `cond + cfg_weight * (cond - uncond)`.
//   - Conditioning tokens:
//        spkr_enc(speaker_emb)             -> 1 token
//        perceiver(cond_prompt_speech_emb) -> 32 tokens (shared AttentionBlock2
//                                             used cross-attn then self-attn)
//        emotion_adv_fc(exaggeration)      -> 1 token
//     These concatenate into `cond_emb` (34 tokens).  Conditional and
//     unconditional passes share the cond_emb; text/speech embeddings differ
//     between them (uncond zeroes text embeds but keeps the speech BOS).

#include "chatterbox_t3_internal.h"
#include "t3_mtl.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tts_cpp::chatterbox::detail {

namespace {

// Process-wide registry of the Phase-15 stacked-weight buffers, with an
// atexit hook that frees them before Metal's static device destructors
// run. Without this Metal asserts on `[rsets->data count] == 0` because
// `buffer_stack` is still live when the ggml-metal dylib tears down.
// Mirrors `s3gen_model_cache_release` in chatterbox_tts.cpp; the
// existing buffer_w / buffer_kv get cleaned up by other paths
// (explicit free_t3() in error returns, dylib finaliser via the
// model_ctx cache for s3gen, etc.) — only the new buffer_stack needs
// to be added to the atexit chain.
struct t3_stack_entry {
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context *        ctx    = nullptr;
};
std::mutex                  t3_stack_mu;
std::vector<t3_stack_entry> t3_stack_registry;
bool                        t3_stack_atexit_registered = false;

void t3_stack_release_atexit() {
    std::lock_guard<std::mutex> lk(t3_stack_mu);
    for (auto & e : t3_stack_registry) {
        if (e.buffer) {
            ggml_backend_buffer_free(e.buffer);
            e.buffer = nullptr;
        }
        if (e.ctx) {
            ggml_free(e.ctx);
            e.ctx = nullptr;
        }
    }
    t3_stack_registry.clear();
}

}  // anonymous namespace

void t3_stack_register(ggml_backend_buffer_t buf, ggml_context * ctx) {
    std::lock_guard<std::mutex> lk(t3_stack_mu);
    t3_stack_registry.push_back({buf, ctx});
    if (!t3_stack_atexit_registered) {
        std::atexit(t3_stack_release_atexit);
        t3_stack_atexit_registered = true;
    }
}

// Drop a (buffer, ctx) pair from the atexit registry without freeing.
// Used by free_t3() in main on error-path early-returns: free_t3 itself
// frees buffer_stack + ctx_stack so the backend can shut down cleanly in
// the same scope; the atexit hook would otherwise double-free dangling
// pointers if we didn't pull them out of the registry first.
void t3_stack_unregister(ggml_backend_buffer_t buf, ggml_context * ctx) {
    std::lock_guard<std::mutex> lk(t3_stack_mu);
    for (auto it = t3_stack_registry.begin(); it != t3_stack_registry.end(); ) {
        if (it->buffer == buf && it->ctx == ctx) {
            it = t3_stack_registry.erase(it);
        } else {
            ++it;
        }
    }
}

// Forward declaration for the step-graph builder used by the
// cache below.  Body lives in the second anonymous namespace further
// down (alongside the legacy build_step_graph_mtl wrapper).
namespace {
ggml_cgraph * build_step_graph_mtl_in_ctx(const chatterbox_model & model,
                                          ggml_context * ctx,
                                          int n_past,
                                          bool is_uncond);
}

// ============================================================================
// T3 step-graph cache (multilingual CFG token decode)
// ============================================================================
//
// `build_step_graph_mtl(n_past, is_uncond)` constructs a 30-layer Llama-block
// graph from scratch on every token decode call.  Multilingual CFG fires
// this 2× per token (cond + uncond on CPU); a 136-token Spanish synth
// previously rebuilt 272 graphs at ~3 ms each — roughly 800 ms / synth of
// pure host-CPU graph construction work.
//
// The cache stores per-(n_past, is_uncond) entries with their own
// ggml_context, gallocator, and metadata buf.  ggml_view's offset is a
// graph-build-time constant in `build_llama_block` (KV write/read offsets
// scale with `n_past`), so each distinct n_past needs its own cached
// graph — there is no shape-independent path here.
//
// Memory cap: a hard FIFO bound of `T3_STEP_CACHE_CAP` entries (default
// 256, covering 128 tokens × 2 modes).  When the cap is hit, new
// (n_past, is_uncond) keys fall back to the legacy thread_local-buf path
// (correct, just no caching benefit).  Tested: cache invariants stay
// correct under cap pressure; bit-exact preserved.
//
// Lifecycle: cleared by detail::t3_release_caches() — called from the
// CLI's free_t3 lambda + Engine::Impl::free_model BEFORE the model
// backend is freed (gallocators carry backend references; freeing them
// against a dead backend would assert).  Plus a fallback atexit hook
// for the unsurprising case where neither path runs.

namespace {

// Cache entry holds just the graph metadata — NOT a per-entry
// gallocator.  The caller's existing shared allocator (passed into
// run_step_pass) is used for both cached and legacy-fallback graphs;
// alloc_graph re-lays-out per call but reuses one backend buffer
// across every (n_past, is_uncond) variant.  This is what keeps the
// single-utterance regression at zero — per-entry gallocator would
// allocate ~1 MB device memory PER cached graph (272 misses × 1 MB =
// ~270 MB allocator churn on the first multilingual synth, observed
// as ~10 % T3 wall-time regression).  Share the allocator instead.
struct t3_step_cache_entry {
    int64_t                key = -1;   // pack(n_past, is_uncond)
    ggml_context *         ctx = nullptr;
    ggml_cgraph *          gf  = nullptr;
    std::vector<uint8_t>   buf;

    t3_step_cache_entry() = default;
    t3_step_cache_entry(const t3_step_cache_entry &)             = delete;
    t3_step_cache_entry & operator=(const t3_step_cache_entry &) = delete;
    t3_step_cache_entry(t3_step_cache_entry && other) noexcept
        : key(other.key), ctx(other.ctx), gf(other.gf),
          buf(std::move(other.buf)) {
        other.key = -1;
        other.ctx = nullptr;
        other.gf  = nullptr;
    }
    t3_step_cache_entry & operator=(t3_step_cache_entry && other) noexcept {
        if (this != &other) {
            destroy();
            key = other.key;
            ctx = other.ctx;
            gf  = other.gf;
            buf = std::move(other.buf);
            other.key = -1;
            other.ctx = nullptr;
            other.gf  = nullptr;
        }
        return *this;
    }
    ~t3_step_cache_entry() { destroy(); }

    void destroy() {
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        gf  = nullptr;
        key = -1;
    }
};

constexpr size_t T3_STEP_CACHE_CAP = 256;

// Caching is opt-in to avoid a small (~10 %) T3 regression on
// single-utterance workloads where every step call is a cache miss.
// In a single multilingual synth, n_past goes 0, 1, 2, ..., N-1 once
// each, so the cache fills up but nothing is re-used — every miss
// pays the bookkeeping cost (vector::resize, list insert, mutex
// acquire) without any compensating hit savings.
//
// Server-mode and other multi-synth callers — where synth #2 starts
// at n_past=0 again and re-decodes the same prompt prefix as
// synth #1 — get a real win (~3 ms × hits per call ≈ 1 s / synth
// on multilingual), so the env var unlocks caching for those
// workloads:
//
//   CHATTERBOX_T3_STEP_CACHE=1 ./tts-cli ...
//
// Reads once at first use, cached as a static const bool.  Tests
// set the env var via `setenv()` before any eval_step_mtl call.
bool t3_step_cache_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("CHATTERBOX_T3_STEP_CACHE");
        if (!e || !e[0]) return false;
        return e[0] == '1' || e[0] == 't' || e[0] == 'T' ||
               e[0] == 'y' || e[0] == 'Y';
    }();
    return enabled;
}

// Mutex protects the entire cache state below.  Held only across cache
// state mutations, not across the underlying backend compute itself.
std::mutex                                                              t3_step_cache_mu;
std::list<t3_step_cache_entry>                                          t3_step_cache_lru;     // front = most recent
std::unordered_map<int64_t, std::list<t3_step_cache_entry>::iterator>   t3_step_cache_idx;
size_t                                                                  t3_step_cache_hits     = 0;
size_t                                                                  t3_step_cache_misses   = 0;
bool                                                                    t3_step_cache_atexit_registered = false;

inline int64_t pack_step_key(int n_past, bool is_uncond) {
    return ((int64_t) n_past << 1) | (is_uncond ? 1 : 0);
}

void t3_step_cache_release_locked() {
    // Caller holds t3_step_cache_mu.
    t3_step_cache_idx.clear();
    t3_step_cache_lru.clear();   // entries' destructors free ctx + allocr
    t3_step_cache_hits   = 0;
    t3_step_cache_misses = 0;
}

void t3_step_cache_release_atexit() {
    std::lock_guard<std::mutex> lk(t3_step_cache_mu);
    t3_step_cache_release_locked();
}

// Look up a cached entry; on hit, splice it to the front (LRU "touch").
// Returns nullptr on miss.  Mutex must NOT be held by caller.
t3_step_cache_entry * t3_step_cache_lookup(int n_past, bool is_uncond) {
    const int64_t key = pack_step_key(n_past, is_uncond);
    std::lock_guard<std::mutex> lk(t3_step_cache_mu);
    auto it = t3_step_cache_idx.find(key);
    if (it == t3_step_cache_idx.end()) {
        ++t3_step_cache_misses;
        return nullptr;
    }
    // Move to front (LRU touch).  splice within the same list keeps
    // iterators valid; this is the canonical std::list LRU pattern.
    t3_step_cache_lru.splice(t3_step_cache_lru.begin(),
                             t3_step_cache_lru, it->second);
    ++t3_step_cache_hits;
    return &(*it->second);
}

// Build a new cached entry and insert at the front.  If the cache is
// at capacity, evicts the oldest (back-of-list) entry first.  Returns
// the inserted entry, or nullptr on failure (e.g., backend init).
//
// Caller must NOT hold the mutex; this function takes it internally
// because the build itself is heavy (~3 ms) and we don't want to
// block other reader threads on it.  Two threads racing on the same
// (n_past, is_uncond) miss are serialised here so only one builds.
t3_step_cache_entry * t3_step_cache_insert_or_get(const chatterbox_model & model,
                                                  int n_past, bool is_uncond) {
    const int64_t key = pack_step_key(n_past, is_uncond);
    std::lock_guard<std::mutex> lk(t3_step_cache_mu);

    // Re-check after locking — another thread may have inserted while
    // we were waiting.
    auto existing = t3_step_cache_idx.find(key);
    if (existing != t3_step_cache_idx.end()) {
        t3_step_cache_lru.splice(t3_step_cache_lru.begin(),
                                 t3_step_cache_lru, existing->second);
        ++t3_step_cache_hits;
        return &(*existing->second);
    }

    // Evict back-of-list if at capacity.
    if (t3_step_cache_lru.size() >= T3_STEP_CACHE_CAP) {
        const int64_t old_key = t3_step_cache_lru.back().key;
        t3_step_cache_idx.erase(old_key);
        t3_step_cache_lru.pop_back();   // dtor frees ctx + allocr
    }

    // Build the new entry at the front.
    t3_step_cache_lru.emplace_front();
    t3_step_cache_entry & e = t3_step_cache_lru.front();

    const size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                            ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    e.buf.resize(buf_size);
    e.key = key;

    ggml_init_params p = { buf_size, e.buf.data(), /*no_alloc=*/true };
    e.ctx = ggml_init(p);
    if (!e.ctx) {
        t3_step_cache_lru.pop_front();
        return nullptr;
    }

    e.gf = build_step_graph_mtl_in_ctx(model, e.ctx, n_past, is_uncond);
    if (!e.gf) {
        t3_step_cache_lru.pop_front();
        return nullptr;
    }

    t3_step_cache_idx[key] = t3_step_cache_lru.begin();

    if (!t3_step_cache_atexit_registered) {
        std::atexit(t3_step_cache_release_atexit);
        t3_step_cache_atexit_registered = true;
    }

    return &t3_step_cache_lru.front();
}

}  // namespace

// Public release entry-point.  Called from chatterbox_cli.cpp's
// free_t3 lambda and chatterbox_engine.cpp's Impl::free_model BEFORE
// ggml_backend_free.  Idempotent.
void t3_release_caches() {
    std::lock_guard<std::mutex> lk(t3_step_cache_mu);
    t3_step_cache_release_locked();
}

// detail-scope bridges so the test_hooks namespace (defined further
// down, outside detail::) can reach the step-graph cache state without
// each individual symbol leaking into the public surface.  These
// helpers are NOT for production callers; the only consumers are
// test_hooks::t3_* in the same TU.
size_t _t3_step_cache_size_for_tests() {
    std::lock_guard<std::mutex> lk(t3_step_cache_mu);
    return t3_step_cache_lru.size();
}
size_t _t3_step_cache_capacity_for_tests() {
    return T3_STEP_CACHE_CAP;
}
bool _t3_step_cache_contains_for_tests(int n_past, bool is_uncond) {
    const int64_t key = pack_step_key(n_past, is_uncond);
    std::lock_guard<std::mutex> lk(t3_step_cache_mu);
    return t3_step_cache_idx.count(key) > 0;
}
size_t _t3_step_cache_hits_for_tests() {
    std::lock_guard<std::mutex> lk(t3_step_cache_mu);
    return t3_step_cache_hits;
}
size_t _t3_step_cache_misses_for_tests() {
    std::lock_guard<std::mutex> lk(t3_step_cache_mu);
    return t3_step_cache_misses;
}

namespace {

int64_t require_key(const gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) throw std::runtime_error(std::string("missing GGUF key: ") + key);
    return id;
}

ggml_tensor * require_tensor(const chatterbox_model & m, const char * name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end() || !it->second) {
        throw std::runtime_error(std::string("missing tensor: ") + name);
    }
    return it->second;
}

uint32_t get_u32(const gguf_context * ctx, const char * key) {
    return gguf_get_val_u32(ctx, require_key(ctx, key));
}
float get_f32(const gguf_context * ctx, const char * key) {
    return gguf_get_val_f32(ctx, require_key(ctx, key));
}
bool get_bool(const gguf_context * ctx, const char * key) {
    return gguf_get_val_bool(ctx, require_key(ctx, key));
}

// Llama-3 style RoPE frequency scaling (transformers `_compute_llama3_parameters`).
// Produces a per-frequency-bin correction factor that ggml_rope_ext will
// apply as its `c` (freq_factors) parameter.  Length is head_dim/2.
//
//   base_inv_freq[i] = 1 / theta^(2i / head_dim)
//   wavelen[i]       = 2*pi / base_inv_freq[i]
//   if wavelen > low_wavelen:  inv_freq = base / factor
//   if wavelen < high_wavelen: inv_freq = base
//   else:                      smooth transition
//   freq_factor[i]   = base_inv_freq[i] / effective_inv_freq[i]
//                    (ggml multiplies the position by 1/freq_factor[i] when
//                    rotating each band, so storing base/effective here is
//                    equivalent to dividing the base frequency by the
//                    same ratio Python's `inv_freq_llama / inv_freq_extrapolation`
//                    produces).  Parity test green against PyTorch.
std::vector<float> compute_llama3_freq_factors(int head_dim, float theta,
                                               float factor, float low_freq,
                                               float high_freq, int orig_max_pos) {
    const int half = head_dim / 2;
    std::vector<float> ff(half, 1.0f);

    const float low_wavelen  = (float) orig_max_pos / low_freq;
    const float high_wavelen = (float) orig_max_pos / high_freq;

    for (int i = 0; i < half; ++i) {
        const float base = 1.0f / std::pow(theta, (float)(2 * i) / (float) head_dim);
        const float wavelen = 2.0f * (float) M_PI / base;

        float effective;
        if (wavelen > low_wavelen) {
            effective = base / factor;
        } else if (wavelen < high_wavelen) {
            effective = base;
        } else {
            const float smooth = ((float) orig_max_pos / wavelen - low_freq) /
                                 (high_freq - low_freq);
            const float scaled = base / factor;
            effective = (1.0f - smooth) * scaled + smooth * base;
        }
        ff[i] = base / effective;
    }
    return ff;
}

// Perceiver cross/self attention block (Perceiver.attn): a single
// AttentionBlock2 with LayerNorm + 4-head scaled-dot-product attention +
// proj_out + residual to the query side.
//
// In the perceiver forward we call this twice with the same weights:
//   pre_att = attn(query_tokens, h_in)       // cross-attn
//   out     = attn(pre_att, pre_att)         // self-attn
//
// perc_q shape:  (n_embd, T_q)      query input (added to the output as residual)
// perc_kv shape: (n_embd, T_kv)     key/value input
// Returns:       (n_embd, T_q)
ggml_tensor * build_perceiver_attn(ggml_context * ctx,
                                   const perceiver_weights & w,
                                   const chatterbox_hparams & hp,
                                   ggml_tensor * perc_q,
                                   ggml_tensor * perc_kv) {
    const int n_embd  = hp.n_embd;
    const int n_heads = hp.perceiver_heads;
    const int head_dim = n_embd / n_heads;

    const int T_q  = perc_q->ne[1];
    const int T_kv = perc_kv->ne[1];

    // LayerNorm on both inputs (same affine weights as Python's self.norm).
    // eps fixed at 1e-5 to match nn.LayerNorm's PyTorch default; this is
    // intentionally NOT hp.eps (which is the Llama backbone's RMSNorm eps
    // and only applies to the 30 transformer blocks).
    auto ln = [&](ggml_tensor * x) {
        ggml_tensor * n = ggml_norm(ctx, x, /*eps=*/1e-5f);
        return ggml_add(ctx, ggml_mul(ctx, n, w.norm_g), w.norm_b);
    };
    ggml_tensor * q_norm = ln(perc_q);
    ggml_tensor * kv_norm = ln(perc_kv);

    ggml_tensor * q_lin = ggml_add(ctx, ggml_mul_mat(ctx, w.to_q_w, q_norm),  w.to_q_b);
    ggml_tensor * k_lin = ggml_add(ctx, ggml_mul_mat(ctx, w.to_k_w, kv_norm), w.to_k_b);
    ggml_tensor * v_lin = ggml_add(ctx, ggml_mul_mat(ctx, w.to_v_w, kv_norm), w.to_v_b);

    // Reshape to (head_dim, T, n_heads) for flash_attn_ext.
    ggml_tensor * Q = ggml_reshape_3d(ctx, q_lin, head_dim, n_heads, T_q);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // (head_dim, T_q, n_heads)
    ggml_tensor * K = ggml_reshape_3d(ctx, k_lin, head_dim, n_heads, T_kv);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    ggml_tensor * V = ggml_reshape_3d(ctx, v_lin, head_dim, n_heads, T_kv);
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    const float scale = 1.0f / std::sqrt((float) head_dim);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K, V, /*mask=*/nullptr,
                                             scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    // attn output layout: (head_dim, n_heads, T_q, 1)
    attn = ggml_reshape_2d(ctx, attn, n_embd, T_q);

    ggml_tensor * proj = ggml_add(ctx, ggml_mul_mat(ctx, w.proj_out_w, attn), w.proj_out_b);
    return ggml_add(ctx, perc_q, proj);
}

// Perceiver forward: pre_att = attn(pre_attention_query, h); return attn(pre_att, pre_att)
// pre_attention_query shape in the GGUF: (1024, 32, 1) after transpose convention
// h shape (cond_prompt_speech_emb): (1024, cond_prompt_len)
ggml_tensor * build_perceiver(ggml_context * ctx,
                              const chatterbox_model & m,
                              ggml_tensor * h) {
    // pre_attention_query stored as (1, 32, 1024) → ggml storage (1024, 32, 1).
    // Take it as a (1024, 32) 2D tensor.
    ggml_tensor * query = ggml_reshape_2d(ctx, m.perceiver.pre_attention_query, m.hparams.n_embd, m.hparams.perceiver_queries);
    ggml_tensor * pre_att = build_perceiver_attn(ctx, m.perceiver, m.hparams, query, h);
    ggml_tensor * out     = build_perceiver_attn(ctx, m.perceiver, m.hparams, pre_att, pre_att);
    return out;
}

// One Llama transformer block.  Writes K/V into the selected KV cache
// tensors at positions [n_past, n_past + N).
//
// inpL:       (n_embd, N)            for B=1
//             (n_embd, N, 2)         for B=2 (cond + uncond packed as ne[2])
// memory_k/v: 1D F32 buffer holding the **cond+uncond pair** for MTL:
//             size = 2 * head_dim * n_kv_head * n_ctx * n_layer.
//             Per-layer slab is `2 * kv_layer_elems`; cond at offset 0
//             within the slab, uncond at offset kv_layer_elems.
//
// b_offset_elems selects which half is touched in the B=1 path:
//             0            → cond pass writes/reads the cond slab
//             kv_layer_elems → uncond pass writes/reads the uncond slab
// In the B=2 path b_offset_elems is ignored: ne[3]=2 spans both halves
// and per-batch stride is `kv_layer_elems * sizeof(float)`.
ggml_tensor * build_llama_block(ggml_context * ctx, ggml_cgraph * gf,
                                const chatterbox_model & m,
                                int il,
                                ggml_tensor * inpL,
                                int n_past, int N, int B,
                                size_t b_offset_elems,
                                ggml_tensor * memory_k,
                                ggml_tensor * memory_v,
                                ggml_tensor * pos_ids,
                                ggml_tensor * kq_mask) {
    const auto & hp = m.hparams;
    const auto & l  = m.layers_mtl[il];
    const int HD  = hp.head_dim;
    const int NH  = hp.n_head;
    const int NKV = hp.n_kv_head;
    const int n_ctx = hp.n_ctx;
    const int64_t L = n_past + N;

    // KV strides are sized off the cache dtype (F32 historically; F16
    // since Phase 2 to halve KV bandwidth) so the same builder works for
    // either precision without re-deriving offsets per-call.
    const size_t kv_ts = ggml_type_size(memory_k->type);
    const size_t kv_head_stride   = (size_t) HD * n_ctx * kv_ts;
    const size_t kv_pos_stride    = (size_t) HD * kv_ts;
    const size_t kv_layer_elems   = (size_t) HD * n_ctx * NKV;       // one batch slab
    const size_t kv_batch_stride  = kv_layer_elems * kv_ts;          // step from cond to uncond
    const size_t kv_layer_stride  = (size_t) 2 * kv_batch_stride;    // per-layer slab is 2x
    const size_t layer_off = (size_t) il * kv_layer_stride
                           + b_offset_elems * kv_ts;

    // Pre-attention RMSNorm (no bias).
    ggml_tensor * cur = ggml_rms_norm(ctx, inpL, hp.eps);
    cur = ggml_mul(ctx, cur, l.ln_attn_g);

    // Q/K/V mat-muls.  When the Phase-15 stacked W_qkv is available
    // (Metal hot path) we run ONE Q4_0 mat-mul producing
    // (3 * n_embd, N, B), then slice Q/K/V via strided views straight
    // into the (HD, NH, N[, B]) shape that RoPE expects — no
    // ggml_reshape (would require a contiguous source) and no
    // ggml_cont (would defeat the saving). RoPE's metal kernel walks
    // src via per-element nb00/nb01/nb02/nb03 strides so it handles
    // the non-contiguous N stride on the slice transparently.
    const int n_embd_t = hp.n_embd;
    ggml_tensor * Qlin;
    ggml_tensor * Klin;
    ggml_tensor * Vlin;
    bool used_stacked_qkv = false;
    if (l.wqkv) {
        ggml_tensor * QKV = ggml_mul_mat(ctx, l.wqkv, cur);  // (3*n_embd, N) or (3*n_embd, N, B)
        used_stacked_qkv = true;
        const size_t f = sizeof(float);
        const size_t row_stride   = (size_t) 3 * n_embd_t * f;
        const size_t batch_stride = row_stride * (size_t) N;
        const size_t off_q = 0 * (size_t) n_embd_t * f;
        const size_t off_k = 1 * (size_t) n_embd_t * f;
        const size_t off_v = 2 * (size_t) n_embd_t * f;
        if (B == 1) {
            Qlin = ggml_view_2d(ctx, QKV, n_embd_t, N, row_stride, off_q);
            Klin = ggml_view_2d(ctx, QKV, n_embd_t, N, row_stride, off_k);
            Vlin = ggml_view_2d(ctx, QKV, n_embd_t, N, row_stride, off_v);
        } else {
            Qlin = ggml_view_3d(ctx, QKV, n_embd_t, N, B, row_stride, batch_stride, off_q);
            Klin = ggml_view_3d(ctx, QKV, n_embd_t, N, B, row_stride, batch_stride, off_k);
            Vlin = ggml_view_3d(ctx, QKV, n_embd_t, N, B, row_stride, batch_stride, off_v);
        }
    } else {
        Qlin = ggml_mul_mat(ctx, l.wq, cur);
        Klin = ggml_mul_mat(ctx, l.wk, cur);
        Vlin = ggml_mul_mat(ctx, l.wv, cur);
    }

    // Reshape to (HD, n_head, N) [B=1] or (HD, n_head, N, B) [B=2].
    // ggml_rope_ext requires ne[2] == len(pos_ids), so sequence stays on
    // ne[2] at the rope call; the optional batch dim sits at ne[3].
    //
    // Use ggml_view_3d/4d (not ggml_reshape) so the same code path
    // works whether Q/K/V came from contiguous per-head mul_mats
    // (un-stacked path) or from strided slices of the W_qkv mul_mat
    // (Phase-15 stacked path). RoPE's metal kernel walks src via
    // per-element nb01/nb02/nb03 strides so the strided N step is
    // transparent.
    ggml_tensor * Q;
    ggml_tensor * K;
    ggml_tensor * V;
    {
        const size_t f = sizeof(float);
        if (B == 1) {
            Q = ggml_view_3d(ctx, Qlin, HD, NH,  N, HD * f, Qlin->nb[1], 0);
            K = ggml_view_3d(ctx, Klin, HD, NKV, N, HD * f, Klin->nb[1], 0);
            V = ggml_view_3d(ctx, Vlin, HD, NKV, N, HD * f, Vlin->nb[1], 0);
        } else {
            Q = ggml_view_4d(ctx, Qlin, HD, NH,  N, B, HD * f, Qlin->nb[1], Qlin->nb[2], 0);
            K = ggml_view_4d(ctx, Klin, HD, NKV, N, B, HD * f, Klin->nb[1], Klin->nb[2], 0);
            V = ggml_view_4d(ctx, Vlin, HD, NKV, N, B, HD * f, Vlin->nb[1], Vlin->nb[2], 0);
        }
    }
    (void) used_stacked_qkv;

    // RoPE on Q and K (NEOX-style half-split convention used by Llama).
    // ggml_rope_ext broadcasts cleanly over an optional batch dim at ne[3].
    const int rope_mode = GGML_ROPE_TYPE_NEOX;
    Q = ggml_rope_ext(ctx, Q, pos_ids, m.rope_freq_factors,
                      HD, rope_mode, hp.rope_orig_max_pos,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    K = ggml_rope_ext(ctx, K, pos_ids, m.rope_freq_factors,
                      HD, rope_mode, hp.rope_orig_max_pos,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // Flash attention expects (HD, N, NH[, B]).  Permute (0, 2, 1, 3) lifts
    // N to ne[1] so the KV cache keeps a [HD, n_ctx, n_kv_head] inner-3D
    // layout that flash_attn can read contiguously per (head, batch).
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    // Write K/V into the cache at [n_past : n_past+N) for this layer.
    {
        ggml_tensor * k_dst;
        ggml_tensor * v_dst;
        if (B == 1) {
            k_dst = ggml_view_3d(ctx, memory_k,
                HD, N, NKV,
                kv_pos_stride, kv_head_stride,
                layer_off + (size_t) n_past * kv_pos_stride);
            v_dst = ggml_view_3d(ctx, memory_v,
                HD, N, NKV,
                kv_pos_stride, kv_head_stride,
                layer_off + (size_t) n_past * kv_pos_stride);
        } else {
            k_dst = ggml_view_4d(ctx, memory_k,
                HD, N, NKV, B,
                kv_pos_stride, kv_head_stride, kv_batch_stride,
                layer_off + (size_t) n_past * kv_pos_stride);
            v_dst = ggml_view_4d(ctx, memory_v,
                HD, N, NKV, B,
                kv_pos_stride, kv_head_stride, kv_batch_stride,
                layer_off + (size_t) n_past * kv_pos_stride);
        }
        ggml_build_forward_expand(gf, ggml_cpy(ctx, K, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, V, v_dst));
    }

    // Attention: read the full [0, L) slice from the cache.
    ggml_tensor * Kfull;
    ggml_tensor * Vfull;
    if (B == 1) {
        Kfull = ggml_view_3d(ctx, memory_k,
            HD, L, NKV,
            kv_pos_stride, kv_head_stride,
            layer_off);
        Vfull = ggml_view_3d(ctx, memory_v,
            HD, L, NKV,
            kv_pos_stride, kv_head_stride,
            layer_off);
    } else {
        Kfull = ggml_view_4d(ctx, memory_k,
            HD, L, NKV, B,
            kv_pos_stride, kv_head_stride, kv_batch_stride,
            layer_off);
        Vfull = ggml_view_4d(ctx, memory_v,
            HD, L, NKV, B,
            kv_pos_stride, kv_head_stride, kv_batch_stride,
            layer_off);
    }

    const float scale = 1.0f / std::sqrt((float) HD);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, Kfull, Vfull, kq_mask,
                                             scale, 0.0f, 0.0f);
    // attn ne=[HD, NH, N, B].  Reshape back to (n_embd, N[, B]).
    if (B == 1) {
        cur = ggml_reshape_2d(ctx, attn, hp.n_embd, N);
    } else {
        cur = ggml_reshape_3d(ctx, attn, hp.n_embd, N, B);
    }

    // O-proj + residual.
    cur = ggml_mul_mat(ctx, l.wo, cur);
    cur = ggml_add(ctx, cur, inpL);

    // MLP (SwiGLU) with pre-norm + residual.
    //
    // Phase 15 stacks `[W_gate ‖ W_up]` along the M dim so a single
    // Q4_0 mat-mul produces (2 * n_ff, N, B); ggml_swiglu (the
    // single-arg variant, GGML_GLU_OP_SWIGLU on the stacked tensor)
    // splits the result internally and fuses
    // `silu(first_half) * second_half` into one Metal kernel
    // (kernel_swiglu_f32). Net effect per layer per step: 2 mat-muls
    // + 1 swiglu instead of 2 mat-muls + 1 swiglu_split, **plus**
    // one fewer mul_mat dispatch.
    //
    // Pre-norm `mul(rms_norm(x), g)` is already auto-fused upstream
    // by ggml-metal's `can_fuse(RMS_NORM, MUL)` path
    // (kernel_rms_norm_mul_f32) — leave it written as the obvious
    // two ops so CPU + non-Metal backends get the same shape.
    ggml_tensor * inpFF = cur;
    ggml_tensor * norm2 = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), l.ln_mlp_g);
    ggml_tensor * gate = ggml_mul_mat(ctx, l.mlp_gate, norm2);
    ggml_tensor * up   = ggml_mul_mat(ctx, l.mlp_up,   norm2);
    ggml_tensor * mlp  = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat(ctx, l.mlp_down, mlp);
    return ggml_add(ctx, inpFF, down);
}

// Build the shared cond_emb fragment: (n_embd, 34).
// exaggeration_tensor is a 1-D F32 tensor of length 1 with the emotion
// scalar (we multiply with emotion_adv_w to get the 1024-d emotion token).
ggml_tensor * build_cond_emb(ggml_context * ctx,
                             const chatterbox_model & m,
                             ggml_tensor * exaggeration) {
    const auto & hp = m.hparams;

    // 1. spkr_enc(speaker_emb): (n_embd, 1).
    //    cond_spkr_w ggml ne=(256, 1024) [from nn.Linear (out=1024, in=256) -> no
    //    explicit transpose, numpy <-> ggml axis reversal gives us (in, out)].
    //    builtin_speaker_emb ggml ne=(256, 1).  Result ne=(1024, 1).  Bias
    //    (1024,) broadcasts along the N=1 column.
    ggml_tensor * spkr_raw = ggml_mul_mat(ctx, m.cond_spkr_w, m.builtin_speaker_emb);
    ggml_tensor * spkr = ggml_add(ctx, spkr_raw,
                                   ggml_reshape_2d(ctx, m.cond_spkr_b, hp.n_embd, 1));

    // 2. cond_prompt_speech_emb = speech_emb[tokens] + speech_pos_emb[0..len).
    //    T3.prepare_conditioning adds positional embeddings to the speech
    //    tokens before handing them to the perceiver (not-is_gpt branch).
    ggml_tensor * cond_tok_emb = ggml_get_rows(ctx, m.speech_emb, m.builtin_cond_prompt_tokens);
    const int cond_prompt_len = m.builtin_cond_prompt_tokens->ne[0];
    ggml_tensor * cond_pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cond_prompt_len);
    ggml_set_name(cond_pos_ids, "cond_prompt_pos_ids");
    ggml_set_input(cond_pos_ids);
    ggml_tensor * cond_pos = ggml_get_rows(ctx, m.speech_pos_emb, cond_pos_ids);
    ggml_tensor * cond_prompt_emb = ggml_add(ctx, cond_tok_emb, cond_pos);

    // 3. perceiver output: (n_embd, 32)
    ggml_tensor * perc = build_perceiver(ctx, m, cond_prompt_emb);

    // 4. emotion_adv: emotion_adv_w is (1, n_embd) after transpose; exaggeration
    //    is a (1,1) input scalar.  mul_mat((n_embd, 1), (1, 1)) → (n_embd, 1).
    //    Wait, emotion_adv_fc.weight in PyTorch is shape (1024, 1) (out, in).
    //    After transpose in the converter: (1, 1024) stored as ggml shape (1, 1024).
    //    mul_mat(w[K=1, M=1024], x[K=1, N=1]) → (M=1024, N=1).  Good.
    ggml_tensor * emot = ggml_mul_mat(ctx, m.emotion_adv_w, exaggeration);

    // 5. Concat along seq dim (ne[1]).  spkr(1024,1), perc(1024,32), emot(1024,1)
    //    → (1024, 34).
    ggml_tensor * cond_emb = ggml_concat(ctx, spkr, perc, /*dim=*/1);
    cond_emb = ggml_concat(ctx, cond_emb, emot, /*dim=*/1);
    return cond_emb;
}

// Build the prompt graph for either the conditional or unconditional pass.
//
//   tokens: the T_text text token IDs (same for both passes)
//   is_uncond: if true, zero out the text token embeddings (but keep text_pos_emb
//              and the BOS speech tokens unchanged).
ggml_cgraph * build_prompt_graph_mtl(const chatterbox_model & model,
                                     int n_text_tokens,
                                     bool is_uncond) {
    const auto & hp = model.hparams;
    const int len_cond = 1 + hp.perceiver_queries + (hp.emotion_adv ? 1 : 0);
    const int N = len_cond + n_text_tokens + 2;  // +1 initial_speech, +1 bos

    static size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                             ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    // Dynamic inputs.
    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(text_tokens, "text_tokens");  ggml_set_input(text_tokens);

    ggml_tensor * speech_bos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_bos, "speech_bos");  ggml_set_input(speech_bos);

    ggml_tensor * pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(pos_ids, "pos_ids");  ggml_set_input(pos_ids);

    ggml_tensor * text_pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(text_pos_ids, "text_pos_ids");  ggml_set_input(text_pos_ids);

    ggml_tensor * speech_pos0 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_pos0, "speech_pos0");  ggml_set_input(speech_pos0);

    ggml_tensor * exaggeration = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_name(exaggeration, "exaggeration");  ggml_set_input(exaggeration);

    // Causal attention mask for prompt path (N > 1).  F16 as required by Metal FA.
    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N, N);
    ggml_set_name(kq_mask, "kq_mask");  ggml_set_input(kq_mask);

    // 1. cond_emb (34 tokens).
    ggml_tensor * cond_emb = build_cond_emb(ctx, model, exaggeration);

    // 2. text_emb with learned pos (zeroed token part if uncond).
    ggml_tensor * text_pos_emb_seq = ggml_get_rows(ctx, model.text_pos_emb, text_pos_ids);
    ggml_tensor * text_emb_out;
    if (is_uncond) {
        text_emb_out = text_pos_emb_seq;
    } else {
        ggml_tensor * text_tok_emb = ggml_get_rows(ctx, model.text_emb, text_tokens);
        text_emb_out = ggml_add(ctx, text_tok_emb, text_pos_emb_seq);
    }

    // 3. Speech embeddings: initial_speech = bos (both are speech_emb(6561) + spos[0]).
    ggml_tensor * speech_tok_emb = ggml_get_rows(ctx, model.speech_emb, speech_bos);
    ggml_tensor * speech_pos_emb_0 = ggml_get_rows(ctx, model.speech_pos_emb, speech_pos0);
    ggml_tensor * speech_emb_out = ggml_add(ctx, speech_tok_emb, speech_pos_emb_0);

    // 4. Concat: cond_emb | text_emb | initial_speech | bos.
    ggml_tensor * inp = ggml_concat(ctx, cond_emb, text_emb_out, /*dim=*/1);
    inp = ggml_concat(ctx, inp, speech_emb_out, /*dim=*/1);
    inp = ggml_concat(ctx, inp, speech_emb_out, /*dim=*/1);

    // 5. Run 30 Llama layers.  Cond/uncond share one memory_k/memory_v
    // buffer (size 2 * kv_layer_elems per layer); pick the right half via
    // b_offset_elems.
    const size_t kv_layer_elems = (size_t) hp.head_dim * hp.n_kv_head * hp.n_ctx;
    const size_t b_off = is_uncond ? kv_layer_elems : 0;
    ggml_tensor * cur = inp;
    for (int il = 0; il < hp.n_layer; ++il) {
        cur = build_llama_block(ctx, gf, model, il, cur, /*n_past=*/0, N,
                                /*B=*/1, b_off,
                                model.memory_k, model.memory_v,
                                pos_ids, kq_mask);
    }

    // Final RMSNorm + speech_head (take logits at last position only — seq index N-1).
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), model.norm_g);
    // cur: (n_embd, N) -> take last column.
    ggml_tensor * last = ggml_view_2d(ctx, cur, hp.n_embd, 1,
                                      cur->nb[1],
                                      (size_t)(N - 1) * cur->nb[1]);
    ggml_tensor * logits = ggml_mul_mat(ctx, model.speech_head, last);  // (n_speech_vocab, 1)
    ggml_set_name(logits, "logits");  ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx);
    return gf;
}

// B=2 prompt graph: pack cond + uncond into a single forward over the
// batch dim (ne[2]).  cond_emb (spkr+perceiver+emotion) is identical
// between the two passes, so we just duplicate it; the text-token
// embedding differs (uncond zeroes the token part but keeps the learned
// positional embedding).  Output: (n_speech_vocab, 1, 2) with cond at
// b=0 and uncond at b=1.  Mirrors the use_b2 pattern from
// src/chatterbox_tts.cpp:1994 (S3Gen CFM CFG).
ggml_cgraph * build_prompt_graph_mtl_b2(const chatterbox_model & model,
                                        int n_text_tokens) {
    const auto & hp = model.hparams;
    const int len_cond = 1 + hp.perceiver_queries + (hp.emotion_adv ? 1 : 0);
    const int N = len_cond + n_text_tokens + 2;

    static size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                             ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(text_tokens, "text_tokens");  ggml_set_input(text_tokens);

    ggml_tensor * speech_bos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_bos, "speech_bos");  ggml_set_input(speech_bos);

    ggml_tensor * pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(pos_ids, "pos_ids");  ggml_set_input(pos_ids);

    ggml_tensor * text_pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(text_pos_ids, "text_pos_ids");  ggml_set_input(text_pos_ids);

    ggml_tensor * speech_pos0 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_pos0, "speech_pos0");  ggml_set_input(speech_pos0);

    ggml_tensor * exaggeration = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_name(exaggeration, "exaggeration");  ggml_set_input(exaggeration);

    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N, N);
    ggml_set_name(kq_mask, "kq_mask");  ggml_set_input(kq_mask);

    // Cond fragment (n_embd, 34) — shared between cond + uncond passes.
    ggml_tensor * cond_emb = build_cond_emb(ctx, model, exaggeration);

    // Text embedding diverges between the two passes:
    //   cond:   speech_emb[tokens] + text_pos_emb[0..T)
    //   uncond: text_pos_emb[0..T) only (text-token contribution zeroed)
    ggml_tensor * text_pos_emb_seq = ggml_get_rows(ctx, model.text_pos_emb, text_pos_ids);
    ggml_tensor * text_tok_emb     = ggml_get_rows(ctx, model.text_emb, text_tokens);
    ggml_tensor * text_cond   = ggml_add(ctx, text_tok_emb, text_pos_emb_seq);
    ggml_tensor * text_uncond = text_pos_emb_seq;

    // Speech BOS embeddings (shared between passes).
    ggml_tensor * speech_tok_emb    = ggml_get_rows(ctx, model.speech_emb, speech_bos);
    ggml_tensor * speech_pos_emb_0  = ggml_get_rows(ctx, model.speech_pos_emb, speech_pos0);
    ggml_tensor * speech_emb_out    = ggml_add(ctx, speech_tok_emb, speech_pos_emb_0);

    // Per-batch input assembly (matches the B=1 prompt graph's order):
    //   [cond_emb | text_emb_X | speech_emb | speech_emb] → (n_embd, N)
    auto assemble_one = [&](ggml_tensor * text) {
        ggml_tensor * inp = ggml_concat(ctx, cond_emb, text, /*dim=*/1);
        inp = ggml_concat(ctx, inp, speech_emb_out, /*dim=*/1);
        inp = ggml_concat(ctx, inp, speech_emb_out, /*dim=*/1);
        return inp;
    };
    ggml_tensor * inp_cond   = assemble_one(text_cond);
    ggml_tensor * inp_uncond = assemble_one(text_uncond);

    // Stack along the batch dim: (n_embd, N, 1) + (n_embd, N, 1) → (n_embd, N, 2).
    ggml_tensor * inp_b2 = ggml_concat(ctx,
        ggml_reshape_3d(ctx, inp_cond,   hp.n_embd, N, 1),
        ggml_reshape_3d(ctx, inp_uncond, hp.n_embd, N, 1),
        /*dim=*/2);

    ggml_tensor * cur = inp_b2;
    for (int il = 0; il < hp.n_layer; ++il) {
        cur = build_llama_block(ctx, gf, model, il, cur, /*n_past=*/0, N,
                                /*B=*/2, /*b_offset_elems=*/0,
                                model.memory_k, model.memory_v,
                                pos_ids, kq_mask);
    }

    // Final norm + head.  cur ne=[n_embd, N, 2]; take last position only,
    // resulting in (n_embd, 1, 2), then mat_mul with speech_head (which
    // broadcasts over batch) to (n_speech_vocab, 1, 2).
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), model.norm_g);
    ggml_tensor * last = ggml_view_3d(ctx, cur,
        hp.n_embd, 1, 2,
        cur->nb[1], cur->nb[2],
        (size_t)(N - 1) * cur->nb[1]);
    last = ggml_cont(ctx, last);  // mat_mul wants contiguous src1 over batches
    ggml_tensor * logits = ggml_mul_mat(ctx, model.speech_head, last);
    ggml_set_name(logits, "logits");  ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx);
    return gf;
}

// B=2 step graph: same input speech token + position fed into both cond
// and uncond passes (the sampler combined the previous logits and chose a
// single token).  The two batches diverge only via the KV cache, which
// already differs from the B=2 prompt graph that wrote them.
ggml_cgraph * build_step_graph_mtl_b2(const chatterbox_model & model,
                                      int n_past) {
    const auto & hp = model.hparams;

    static size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                             ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * speech_token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_token, "speech_token"); ggml_set_input(speech_token);

    ggml_tensor * speech_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_pos, "speech_pos"); ggml_set_input(speech_pos);

    ggml_tensor * pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(pos_ids, "pos_ids"); ggml_set_input(pos_ids);

    // inp_b1 = speech_emb[tok] + speech_pos_emb[pos]  → (n_embd, 1).
    // Both batches see the same input embedding; broadcast to (n_embd, 1, 2)
    // via ggml_concat.  The materialization cost is ~4 KB per token and
    // amortises across 30 Llama layers.
    ggml_tensor * inp_b1 = ggml_add(ctx,
        ggml_get_rows(ctx, model.speech_emb,     speech_token),
        ggml_get_rows(ctx, model.speech_pos_emb, speech_pos));
    ggml_tensor * inp_b1_3d = ggml_reshape_3d(ctx, inp_b1, hp.n_embd, 1, 1);
    ggml_tensor * inp = ggml_concat(ctx, inp_b1_3d, inp_b1_3d, /*dim=*/2);

    ggml_tensor * cur = inp;
    for (int il = 0; il < hp.n_layer; ++il) {
        cur = build_llama_block(ctx, gf, model, il, cur, n_past, /*N=*/1,
                                /*B=*/2, /*b_offset_elems=*/0,
                                model.memory_k, model.memory_v,
                                pos_ids, /*kq_mask=*/nullptr);
    }
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), model.norm_g);

    // cur ne=[n_embd, 1, 2] → speech_head @ cur → (n_speech_vocab, 1, 2)
    ggml_tensor * logits = ggml_mul_mat(ctx, model.speech_head, cur);
    ggml_set_name(logits, "logits"); ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx);
    return gf;
}

// Body of the step graph build, parameterised on a caller-provided
// ggml_context.  Lets the step-graph cache hold the ctx
// alive across calls without sharing the legacy thread_local buf.
ggml_cgraph * build_step_graph_mtl_in_ctx(const chatterbox_model & model,
                                          ggml_context * ctx,
                                          int n_past,
                                          bool is_uncond) {
    const auto & hp = model.hparams;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * speech_token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_token, "speech_token"); ggml_set_input(speech_token);

    ggml_tensor * speech_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_pos, "speech_pos"); ggml_set_input(speech_pos);

    ggml_tensor * pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(pos_ids, "pos_ids"); ggml_set_input(pos_ids);

    ggml_tensor * inp = ggml_add(ctx,
        ggml_get_rows(ctx, model.speech_emb, speech_token),
        ggml_get_rows(ctx, model.speech_pos_emb, speech_pos));

    const size_t kv_layer_elems = (size_t) hp.head_dim * hp.n_kv_head * hp.n_ctx;
    const size_t b_off = is_uncond ? kv_layer_elems : 0;

    ggml_tensor * cur = inp;
    for (int il = 0; il < hp.n_layer; ++il) {
        cur = build_llama_block(ctx, gf, model, il, cur, n_past, /*N=*/1,
                                /*B=*/1, b_off,
                                model.memory_k, model.memory_v,
                                pos_ids, /*kq_mask=*/nullptr);
    }
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.eps), model.norm_g);

    ggml_tensor * logits = ggml_mul_mat(ctx, model.speech_head, cur);  // (n_speech_vocab, 1)
    ggml_set_name(logits, "logits"); ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    return gf;
}

// Legacy non-cached entry point (still used as fallback when the
// step-graph cache is at capacity).  Frees the per-call ctx — gf
// remains valid because the bytes live in the thread_local buf
// until the next call to ggml_init reuses the buf.
ggml_cgraph * build_step_graph_mtl(const chatterbox_model & model,
                                   int n_past,
                                   bool is_uncond) {
    static size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                             ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = build_step_graph_mtl_in_ctx(model, ctx, n_past, is_uncond);
    ggml_free(ctx);
    return gf;
}

void fill_causal_mask_f16(std::vector<ggml_fp16_t> & out, int N) {
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    out.assign((size_t) N * N, zero);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out[(size_t) i * N + j] = (j <= i) ? zero : neg_inf;
        }
    }
}

bool run_prompt_pass(const chatterbox_model & model,
                     ggml_gallocr_t allocr,
                     int n_threads,
                     const std::vector<int32_t> & text_tokens,
                     float exaggeration,
                     bool is_uncond,
                     std::vector<float> & logits_out,
                     int & prompt_len_out) {
    const auto & hp = model.hparams;
    const int len_cond = 1 + hp.perceiver_queries + (hp.emotion_adv ? 1 : 0);
    const int N = len_cond + (int) text_tokens.size() + 2;
    prompt_len_out = N;

    ggml_cgraph * gf = build_prompt_graph_mtl(model, (int) text_tokens.size(), is_uncond);
    // alloc_graph reserves lazily; see run_step_pass_b2 comment.
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "run_prompt_pass: gallocr_alloc_graph failed (graph topology exceeded reserved budget?)\n");
        return false;
    }

    // Dynamic inputs.  Any tensor may be pruned by the allocator if it does
    // not feed into the final output (e.g. text_tokens is unused on the
    // uncond pass where text_emb is replaced by zeros), so null-check.
    auto set_in = [&](const char * name, const void * data, size_t bytes) {
        ggml_tensor * t = ggml_graph_get_tensor(gf, name);
        if (t) ggml_backend_tensor_set(t, data, 0, bytes);
    };
    set_in("text_tokens", text_tokens.data(), text_tokens.size() * sizeof(int32_t));
    int32_t bos = hp.start_speech_token;
    set_in("speech_bos", &bos, sizeof(bos));

    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = i;
    set_in("pos_ids", pos.data(), pos.size() * sizeof(int32_t));

    std::vector<int32_t> text_pos(text_tokens.size());
    for (size_t i = 0; i < text_tokens.size(); ++i) text_pos[i] = (int32_t) i;
    set_in("text_pos_ids", text_pos.data(), text_pos.size() * sizeof(int32_t));

    int32_t sp0 = 0;
    set_in("speech_pos0", &sp0, sizeof(sp0));

    const int cond_prompt_len = hp.cond_prompt_len;
    std::vector<int32_t> cond_pos(cond_prompt_len);
    for (int i = 0; i < cond_prompt_len; ++i) cond_pos[i] = i;
    set_in("cond_prompt_pos_ids", cond_pos.data(), cond_pos.size() * sizeof(int32_t));

    float exag = exaggeration;
    set_in("exaggeration", &exag, sizeof(exag));

    // Causal mask.
    std::vector<ggml_fp16_t> mask;
    fill_causal_mask_f16(mask, N);
    set_in("kq_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
    ggml_backend_graph_compute(model.backend, gf);

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    logits_out.resize(ggml_nelements(logits));
    ggml_backend_tensor_get(logits, logits_out.data(), 0, ggml_nbytes(logits));
    return true;
}

// Run the prompt graph as a single batch=2 forward (cond on b=0, uncond
// on b=1).  Output logits shape: (n_speech_vocab, 1, 2); we read the
// cond half into logits_cond and the uncond half into logits_uncond.
bool run_prompt_pass_b2(const chatterbox_model & model,
                        ggml_gallocr_t allocr,
                        int n_threads,
                        const std::vector<int32_t> & text_tokens,
                        float exaggeration,
                        std::vector<float> & logits_cond_out,
                        std::vector<float> & logits_uncond_out,
                        int & prompt_len_out) {
    const auto & hp = model.hparams;
    const int len_cond = 1 + hp.perceiver_queries + (hp.emotion_adv ? 1 : 0);
    const int N = len_cond + (int) text_tokens.size() + 2;
    prompt_len_out = N;

    ggml_cgraph * gf = build_prompt_graph_mtl_b2(model, (int) text_tokens.size());
    // alloc_graph below already reserves lazily via ggml_gallocr_needs_realloc;
    // see run_step_pass_b2 for the rationale on dropping the explicit
    // ggml_gallocr_reserve(allocr, gf) call here.
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "run_prompt_pass_b2: gallocr_alloc_graph failed (graph topology exceeded reserved budget?)\n");
        return false;
    }

    auto set_in = [&](const char * name, const void * data, size_t bytes) {
        ggml_tensor * t = ggml_graph_get_tensor(gf, name);
        if (t) ggml_backend_tensor_set(t, data, 0, bytes);
    };
    set_in("text_tokens", text_tokens.data(), text_tokens.size() * sizeof(int32_t));
    int32_t bos = hp.start_speech_token;
    set_in("speech_bos", &bos, sizeof(bos));

    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = i;
    set_in("pos_ids", pos.data(), pos.size() * sizeof(int32_t));

    std::vector<int32_t> text_pos(text_tokens.size());
    for (size_t i = 0; i < text_tokens.size(); ++i) text_pos[i] = (int32_t) i;
    set_in("text_pos_ids", text_pos.data(), text_pos.size() * sizeof(int32_t));

    int32_t sp0 = 0;
    set_in("speech_pos0", &sp0, sizeof(sp0));

    const int cond_prompt_len = hp.cond_prompt_len;
    std::vector<int32_t> cond_pos(cond_prompt_len);
    for (int i = 0; i < cond_prompt_len; ++i) cond_pos[i] = i;
    set_in("cond_prompt_pos_ids", cond_pos.data(), cond_pos.size() * sizeof(int32_t));

    float exag = exaggeration;
    set_in("exaggeration", &exag, sizeof(exag));

    std::vector<ggml_fp16_t> mask;
    fill_causal_mask_f16(mask, N);
    set_in("kq_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
    ggml_backend_graph_compute(model.backend, gf);

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    // logits ne=[n_speech_vocab, 1, 2], contiguous.  Cond at b=0, uncond at b=1.
    const size_t per_batch_bytes = (size_t) hp.n_speech_vocab * sizeof(float);
    logits_cond_out.resize(hp.n_speech_vocab);
    logits_uncond_out.resize(hp.n_speech_vocab);
    ggml_backend_tensor_get(logits, logits_cond_out.data(),   0,                per_batch_bytes);
    ggml_backend_tensor_get(logits, logits_uncond_out.data(), per_batch_bytes, per_batch_bytes);
    return true;
}

// B=2 step pass: one forward producing both cond + uncond logits.
bool run_step_pass_b2(const chatterbox_model & model,
                      ggml_gallocr_t allocr,
                      int n_threads,
                      int n_past,
                      int32_t token,
                      std::vector<float> & logits_cond_out,
                      std::vector<float> & logits_uncond_out) {
    const auto & hp = model.hparams;

    ggml_cgraph * gf = build_step_graph_mtl_b2(model, n_past);
    // Skip the explicit ggml_gallocr_reserve(allocr, gf) call here:
    // alloc_graph below already calls ggml_gallocr_needs_realloc, and
    // only re-runs the topology analysis when the graph actually grew
    // (single-buffer single-backend case — the default for chatterbox).
    // The per-step graph keeps the same node count + per-node tensor
    // shapes for every n_past >= 1, so after the first call alloc_graph
    // is a fast O(n_nodes) buffer-reset; the explicit reserve forced an
    // unnecessary topology re-walk on every one of the 84 step calls.
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "run_step_pass_b2: gallocr_alloc_graph failed (n_past=%d)\n", n_past);
        return false;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_token"), &token, 0, sizeof(token));
    int32_t sp = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_pos"), &sp, 0, sizeof(sp));
    int32_t pos = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ids"), &pos, 0, sizeof(pos));

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
    ggml_backend_graph_compute(model.backend, gf);

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    const size_t per_batch_bytes = (size_t) hp.n_speech_vocab * sizeof(float);
    logits_cond_out.resize(hp.n_speech_vocab);
    logits_uncond_out.resize(hp.n_speech_vocab);
    ggml_backend_tensor_get(logits, logits_cond_out.data(),   0,                per_batch_bytes);
    ggml_backend_tensor_get(logits, logits_uncond_out.data(), per_batch_bytes, per_batch_bytes);
    return true;
}

bool run_step_pass(const chatterbox_model & model,
                   ggml_gallocr_t allocr,
                   int n_threads,
                   int n_past,
                   int32_t token,
                   bool is_uncond,
                   std::vector<float> & logits_out) {
    // When CHATTERBOX_T3_STEP_CACHE is set, try the per-(n_past,
    // is_uncond) graph cache first.  On hit, we skip the ~3 ms build
    // cost.  On miss + room: build into a fresh cache entry; the
    // caller's allocator is used for layout either
    // way (no ~1 MB-per-entry backend buffer regression).  On miss +
    // cache full: fall back to the legacy thread_local-buf path.
    //
    // Default-disabled because in single-utterance workloads every
    // step call is a unique n_past — the cache fills up but nothing
    // is re-used.  See the t3_step_cache_enabled() comment above.
    t3_step_cache_entry * entry = nullptr;
    if (t3_step_cache_enabled()) {
        entry = t3_step_cache_lookup(n_past, is_uncond);
        if (!entry) {
            entry = t3_step_cache_insert_or_get(model, n_past, is_uncond);
        }
    }

    ggml_cgraph * gf = entry ? entry->gf
                             : build_step_graph_mtl(model, n_past, is_uncond);

    // alloc_graph reserves lazily; see run_step_pass_b2 comment.
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "run_step_pass: gallocr_alloc_graph failed (n_past=%d)\n", n_past);
        return false;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_token"), &token, 0, sizeof(token));
    int32_t sp = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_pos"), &sp, 0, sizeof(sp));
    int32_t pos = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ids"), &pos, 0, sizeof(pos));

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
    ggml_backend_graph_compute(model.backend, gf);

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    logits_out.resize(ggml_nelements(logits));
    ggml_backend_tensor_get(logits, logits_out.data(), 0, ggml_nbytes(logits));
    return true;
}

} // namespace

// -- Stage builders for parity validation (see t3_mtl.h) --------------------

ggml_cgraph * build_stage_cond_emb_graph(const chatterbox_model & m) {
    static size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                             ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * exag = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_name(exag, "exaggeration"); ggml_set_input(exag);

    ggml_tensor * out = build_cond_emb(ctx, m, exag);
    ggml_set_name(out, "cond_emb");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx);
    return gf;
}

ggml_cgraph * build_stage_text_emb_graph(const chatterbox_model & m, int T_text) {
    static size_t buf_size = ggml_tensor_overhead() * 256 +
                             ggml_graph_overhead_custom(256, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);

    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_text);
    ggml_set_name(text_tokens, "text_tokens"); ggml_set_input(text_tokens);
    ggml_tensor * text_pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_text);
    ggml_set_name(text_pos_ids, "text_pos_ids"); ggml_set_input(text_pos_ids);

    ggml_tensor * tok = ggml_get_rows(ctx, m.text_emb, text_tokens);
    ggml_tensor * pos = ggml_get_rows(ctx, m.text_pos_emb, text_pos_ids);
    ggml_tensor * out = ggml_add(ctx, tok, pos);
    ggml_set_name(out, "text_emb_with_pos");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx);
    return gf;
}

ggml_cgraph * build_stage_inputs_graph(const chatterbox_model & m, int T_text,
                                       bool is_uncond) {
    const auto & hp = m.hparams;
    const int len_cond = 1 + hp.perceiver_queries + (hp.emotion_adv ? 1 : 0);
    const int N = len_cond + T_text + 2;

    static size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                             ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_text);
    ggml_set_name(text_tokens, "text_tokens"); ggml_set_input(text_tokens);
    ggml_tensor * text_pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_text);
    ggml_set_name(text_pos_ids, "text_pos_ids"); ggml_set_input(text_pos_ids);
    ggml_tensor * speech_bos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_bos, "speech_bos"); ggml_set_input(speech_bos);
    ggml_tensor * speech_pos0 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_pos0, "speech_pos0"); ggml_set_input(speech_pos0);
    ggml_tensor * exag = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_name(exag, "exaggeration"); ggml_set_input(exag);

    ggml_tensor * cond_emb = build_cond_emb(ctx, m, exag);

    ggml_tensor * text_pos = ggml_get_rows(ctx, m.text_pos_emb, text_pos_ids);
    ggml_tensor * text_emb_out;
    if (is_uncond) {
        text_emb_out = text_pos;
    } else {
        ggml_tensor * tok = ggml_get_rows(ctx, m.text_emb, text_tokens);
        text_emb_out = ggml_add(ctx, tok, text_pos);
    }

    ggml_tensor * semb = ggml_get_rows(ctx, m.speech_emb, speech_bos);
    ggml_tensor * spos = ggml_get_rows(ctx, m.speech_pos_emb, speech_pos0);
    ggml_tensor * speech_emb_out = ggml_add(ctx, semb, spos);

    ggml_tensor * inp = ggml_concat(ctx, cond_emb, text_emb_out, /*dim=*/1);
    inp = ggml_concat(ctx, inp, speech_emb_out, /*dim=*/1);
    inp = ggml_concat(ctx, inp, speech_emb_out, /*dim=*/1);
    ggml_set_name(inp, "inputs_embeds");
    ggml_set_output(inp);
    ggml_build_forward_expand(gf, inp);
    (void) N;
    ggml_free(ctx);
    return gf;
}

ggml_cgraph * build_stage_layers_graph(const chatterbox_model & m, int N,
                                       int n_layers, bool is_uncond) {
    const auto & hp = m.hparams;
    static size_t buf_size = ggml_tensor_overhead() * CHBX_MAX_NODES +
                             ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, N);
    ggml_set_name(inp, "inputs_embeds"); ggml_set_input(inp);

    ggml_tensor * pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(pos_ids, "pos_ids"); ggml_set_input(pos_ids);

    ggml_tensor * kq_mask = nullptr;
    if (N > 1) {
        kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N, N);
        ggml_set_name(kq_mask, "kq_mask"); ggml_set_input(kq_mask);
    }

    const size_t kv_layer_elems = (size_t) hp.head_dim * hp.n_kv_head * hp.n_ctx;
    const size_t b_off = is_uncond ? kv_layer_elems : 0;

    ggml_tensor * cur = inp;
    const int actual_layers = std::min(n_layers, hp.n_layer);
    for (int il = 0; il < actual_layers; ++il) {
        cur = build_llama_block(ctx, gf, m, il, cur, /*n_past=*/0, N,
                                /*B=*/1, b_off,
                                m.memory_k, m.memory_v, pos_ids, kq_mask);
    }
    ggml_set_name(cur, "layers_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx);
    return gf;
}

ggml_cgraph * build_stage_head_graph(const chatterbox_model & m, int N) {
    const auto & hp = m.hparams;
    static size_t buf_size = ggml_tensor_overhead() * 64 +
                             ggml_graph_overhead_custom(64, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, N);
    ggml_set_name(inp, "inputs_embeds"); ggml_set_input(inp);

    ggml_tensor * cur = ggml_mul(ctx, ggml_rms_norm(ctx, inp, hp.eps), m.norm_g);
    ggml_tensor * last = ggml_view_2d(ctx, cur, hp.n_embd, 1,
                                      cur->nb[1],
                                      (size_t)(N - 1) * cur->nb[1]);
    ggml_tensor * logits = ggml_mul_mat(ctx, m.speech_head, last);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    ggml_free(ctx);
    return gf;
}

// --------------------------------------------------------------------------

bool load_model_gguf_mtl(const std::string & path,
                         chatterbox_model & model,
                         int requested_ctx,
                         int n_gpu_layers) {
    extern int g_log_verbose;
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params params = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), params);
    if (!gguf_ctx) {
        fprintf(stderr, "load_model_gguf_mtl: failed to open '%s'\n", path.c_str());
        return false;
    }

    try {
        auto & hp = model.hparams;
        hp.variant           = CHBX_VARIANT_MTL;
        hp.n_text_vocab      = (int32_t) get_u32(gguf_ctx, KEY_TEXT_VOCAB_SIZE);
        hp.n_speech_vocab    = (int32_t) get_u32(gguf_ctx, KEY_SPEECH_VOCAB_SIZE);
        hp.start_speech_token = (int32_t) get_u32(gguf_ctx, KEY_START_SPEECH);
        hp.stop_speech_token  = (int32_t) get_u32(gguf_ctx, KEY_STOP_SPEECH);
        hp.start_text_token   = (int32_t) get_u32(gguf_ctx, KEY_START_TEXT);
        hp.stop_text_token    = (int32_t) get_u32(gguf_ctx, KEY_STOP_TEXT);
        hp.speaker_embed_size = (int32_t) get_u32(gguf_ctx, KEY_SPEAKER_EMBED);
        hp.cond_prompt_len    = (int32_t) get_u32(gguf_ctx, KEY_COND_PROMPT_LEN);
        hp.n_ctx   = (int32_t) get_u32(gguf_ctx, KEY_N_CTX);
        hp.n_embd  = (int32_t) get_u32(gguf_ctx, KEY_N_EMBD);
        hp.n_head  = (int32_t) get_u32(gguf_ctx, KEY_N_HEAD);
        hp.n_kv_head = (int32_t) get_u32(gguf_ctx, KEY_N_KV_HEAD);
        hp.head_dim = (int32_t) get_u32(gguf_ctx, KEY_HEAD_DIM);
        hp.intermediate_size = (int32_t) get_u32(gguf_ctx, KEY_INTERMEDIATE_SIZE);
        hp.n_layer = (int32_t) get_u32(gguf_ctx, KEY_N_LAYER);
        hp.max_text_tokens   = (int32_t) get_u32(gguf_ctx, KEY_MAX_TEXT_TOKENS);
        hp.max_speech_tokens = (int32_t) get_u32(gguf_ctx, KEY_MAX_SPEECH_TOKENS);
        hp.speech_cond_prompt_len = (int32_t) get_u32(gguf_ctx, KEY_SPEECH_COND_LEN);
        hp.perceiver_queries = (int32_t) get_u32(gguf_ctx, KEY_PERCEIVER_QUERIES);
        hp.perceiver_heads   = (int32_t) get_u32(gguf_ctx, KEY_PERCEIVER_HEADS);
        hp.emotion_adv       = get_bool(gguf_ctx, KEY_EMOTION_ADV);
        hp.eps               = get_f32(gguf_ctx, KEY_RMS_EPS);
        hp.rope_theta        = get_f32(gguf_ctx, KEY_ROPE_THETA);
        hp.rope_scale_factor = get_f32(gguf_ctx, KEY_ROPE_SCALING_FACTOR);
        hp.rope_low_freq     = get_f32(gguf_ctx, KEY_ROPE_LOW_FREQ);
        hp.rope_high_freq    = get_f32(gguf_ctx, KEY_ROPE_HIGH_FREQ);
        hp.rope_orig_max_pos = (int32_t) get_u32(gguf_ctx, KEY_ROPE_ORIG_MAX_POS);

        if (hp.rope_high_freq <= hp.rope_low_freq) {
            throw std::runtime_error("invalid llama3 rope freq config: high_freq_factor (" +
                                     std::to_string(hp.rope_high_freq) +
                                     ") must be > low_freq_factor (" +
                                     std::to_string(hp.rope_low_freq) +
                                     ") to avoid div-by-zero in compute_llama3_freq_factors");
        }
        if (hp.rope_scale_factor == 0.0f) {
            throw std::runtime_error("invalid llama3 rope_scaling_factor: must be non-zero");
        }
        if (hp.rope_orig_max_pos <= 0) {
            throw std::runtime_error("invalid rope.original_max_position: must be > 0");
        }

        if (requested_ctx > 0) hp.n_ctx = std::min(hp.n_ctx, requested_ctx);

        model.backend = init_backend(n_gpu_layers);

        const int64_t num_tensors = gguf_get_n_tensors(gguf_ctx);
        ggml_init_params p = { ggml_tensor_overhead() * (size_t)(num_tensors + 1), nullptr, true };
        model.ctx_w = ggml_init(p);
        if (!model.ctx_w) throw std::runtime_error("ggml_init failed");

        for (int64_t i = 0; i < num_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            // Reject duplicate names: ggml_dup_tensor would happily allocate
            // a second tensor under the same name and the second one wins in
            // model.tensors, leaking the first in ctx_w until model destruction.
            // Real-world cause is a malformed GGUF or a future merged-shard
            // format that emits the same name from two shards.
            if (model.tensors.find(name) != model.tensors.end()) {
                throw std::runtime_error(std::string("duplicate tensor name in GGUF: ") + name);
            }
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
            ggml_tensor * dst = ggml_dup_tensor(model.ctx_w, src);
            ggml_set_name(dst, name);
            model.tensors[name] = dst;
        }

        const int half_hd = hp.head_dim / 2;
        ggml_tensor * freq_factors = ggml_new_tensor_1d(model.ctx_w, GGML_TYPE_F32, half_hd);
        ggml_set_name(freq_factors, "rope_freq_factors");
        model.rope_freq_factors = freq_factors;

        model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
        if (!model.buffer_w) {
            throw std::runtime_error("load_model_gguf_mtl: ggml_backend_alloc_ctx_tensors failed for "
                                     "weights buffer (backend out of memory?)");
        }

        for (ggml_tensor * cur = ggml_get_first_tensor(model.ctx_w); cur; cur = ggml_get_next_tensor(model.ctx_w, cur)) {
            if (cur == freq_factors) continue;
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
            ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
        }

        {
            std::vector<float> ff = compute_llama3_freq_factors(
                hp.head_dim, hp.rope_theta, hp.rope_scale_factor,
                hp.rope_low_freq, hp.rope_high_freq, hp.rope_orig_max_pos);
            ggml_backend_tensor_set(freq_factors, ff.data(), 0, ff.size() * sizeof(float));
        }

        model.text_emb        = require_tensor(model, "chatterbox/text_emb");
        model.speech_emb      = require_tensor(model, "chatterbox/speech_emb");
        model.text_pos_emb    = require_tensor(model, "chatterbox/text_pos_emb");
        model.speech_pos_emb  = require_tensor(model, "chatterbox/speech_pos_emb");
        model.text_head       = require_tensor(model, "chatterbox/text_head");
        model.speech_head     = require_tensor(model, "chatterbox/speech_head");
        model.norm_g          = require_tensor(model, "model/norm/g");
        model.cond_spkr_w     = require_tensor(model, "chatterbox/cond_spkr/w");
        model.cond_spkr_b     = require_tensor(model, "chatterbox/cond_spkr/b");
        model.emotion_adv_w   = require_tensor(model, "chatterbox/emotion_adv_fc/w");

        model.perceiver.pre_attention_query = require_tensor(model, "chatterbox/perceiver/pre_attention_query");
        model.perceiver.norm_g = require_tensor(model, "chatterbox/perceiver/attn/norm/g");
        model.perceiver.norm_b = require_tensor(model, "chatterbox/perceiver/attn/norm/b");
        model.perceiver.to_q_w = require_tensor(model, "chatterbox/perceiver/attn/to_q/w");
        model.perceiver.to_q_b = require_tensor(model, "chatterbox/perceiver/attn/to_q/b");
        model.perceiver.to_k_w = require_tensor(model, "chatterbox/perceiver/attn/to_k/w");
        model.perceiver.to_k_b = require_tensor(model, "chatterbox/perceiver/attn/to_k/b");
        model.perceiver.to_v_w = require_tensor(model, "chatterbox/perceiver/attn/to_v/w");
        model.perceiver.to_v_b = require_tensor(model, "chatterbox/perceiver/attn/to_v/b");
        model.perceiver.proj_out_w = require_tensor(model, "chatterbox/perceiver/attn/proj_out/w");
        model.perceiver.proj_out_b = require_tensor(model, "chatterbox/perceiver/attn/proj_out/b");

        model.builtin_speaker_emb        = require_tensor(model, "chatterbox/builtin/speaker_emb");
        model.builtin_cond_prompt_tokens = require_tensor(model, "chatterbox/builtin/cond_prompt_speech_tokens");

        model.layers_mtl.resize(hp.n_layer);
        for (int i = 0; i < hp.n_layer; ++i) {
            auto & l = model.layers_mtl[i];
            std::string lp = "model/h" + std::to_string(i);
            l.ln_attn_g = require_tensor(model, (lp + "/ln_attn/g").c_str());
            l.ln_mlp_g  = require_tensor(model, (lp + "/ln_mlp/g").c_str());
            l.wq = require_tensor(model, (lp + "/attn/q/w").c_str());
            l.wk = require_tensor(model, (lp + "/attn/k/w").c_str());
            l.wv = require_tensor(model, (lp + "/attn/v/w").c_str());
            l.wo = require_tensor(model, (lp + "/attn/o/w").c_str());
            l.mlp_gate = require_tensor(model, (lp + "/mlp/gate/w").c_str());
            l.mlp_up   = require_tensor(model, (lp + "/mlp/up/w").c_str());
            l.mlp_down = require_tensor(model, (lp + "/mlp/down/w").c_str());
        }

        // Single unified KV buffer holding the cond+uncond pair.
        // Layout per layer: 2x kv_layer_elems contiguous floats, with the
        // cond half at offset 0 and the uncond half at offset kv_layer_elems.
        // The B=1 single-pass code addresses the right half via the
        // `b_offset_elems` parameter to build_llama_block; the B=2 batched
        // path views ne[3]=2 over the same memory with batch_stride=
        // kv_layer_elems * sizeof(float).
        ggml_init_params kv_params = { ggml_tensor_overhead() * 4, nullptr, true };
        model.ctx_kv = ggml_init(kv_params);
        const int64_t kv_elements_b2 =
            (int64_t) 2 * hp.head_dim * hp.n_kv_head * hp.n_ctx * hp.n_layer;
        // KV dtype is kept at F32 here.  Phase-2 of §3.21 tried F16 KV —
        // build_llama_block already routes ggml_type_size(memory_k->type)
        // into the strides, ggml_flash_attn_ext consumes F16 K/V
        // directly, and the per-step ggml_cpy converts F32→F16 for
        // free — but on M3 Ultra it was a wash (Q4_0 502 → 507 ms,
        // F16 within noise) and produced byte-exact audio, suggesting
        // ggml-metal's flash-attn was already running its matmul at
        // F16 internally regardless of storage dtype.  We keep F32
        // storage to match the §3.19 numerics envelope.  Memory-bound
        // backends (e.g. M4 with 10 GPU cores) may still benefit; flip
        // this to GGML_TYPE_F16 to try that.
        model.memory_k = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, kv_elements_b2);
        model.memory_v = ggml_new_tensor_1d(model.ctx_kv, GGML_TYPE_F32, kv_elements_b2);
        // Legacy aliases for any caller that hasn't been migrated yet
        // (none on the MTL hot path; kept nullable on purpose).
        model.memory_k_uncond = nullptr;
        model.memory_v_uncond = nullptr;
        model.buffer_kv = ggml_backend_alloc_ctx_tensors(model.ctx_kv, model.backend);
        if (!model.buffer_kv) {
            throw std::runtime_error("load_model_gguf_mtl: ggml_backend_alloc_ctx_tensors failed for "
                                     "KV-cache buffer (backend out of memory?)");
        }

        // Phase 15: per-layer fused-matmul stacks for the Metal hot path.
        //
        //   wqkv      : (n_embd, 3 * n_embd)  rows  [Q ‖ K ‖ V]
        //   w_gate_up : (n_embd, 2 * n_ff)    rows  [gate ‖ up]
        //
        // Each Llama block previously dispatched 3 separate Q4_0 mat-muls
        // for Q/K/V plus 2 for gate/up; stacking them collapses those into
        // 1 + 1 = 2 dispatches per block, saving (3-1) + (2-1) = 3 kernel
        // launches per block per step inside the same compute_graph
        // commit. On a 30-layer × 84-token T3 step pass that's ~7.5k
        // fewer kernel launches per call. The combined mat-mul also
        // gives the Metal mul_mm shader a wider M dimension, which is
        // what its tiling expects (NR0 = 64).
        //
        // CPU backend keeps the original wq/wk/wv path because
        // ggml-cpu's per-kernel overhead is already negligible and the
        // extra weight memory footprint (~75 MB for the multilingual
        // T3) trades unfavourably with thread-cache locality there.
        if (!ggml_backend_is_cpu(model.backend)) {
            const int n_embd = hp.n_embd;
            const int n_ff   = hp.intermediate_size;

            const size_t stack_meta = ggml_tensor_overhead() * (size_t) (2 * hp.n_layer + 4);
            ggml_init_params sp = { stack_meta, nullptr, true };
            model.ctx_stack = ggml_init(sp);
            if (!model.ctx_stack) {
                throw std::runtime_error("load_model_gguf_mtl: ggml_init failed for stacked-weights ctx");
            }

            // QKV stack: Q4_0 in the multilingual T3 GGUF (q.w / k.w / v.w
            // all Q4_0 for every layer). gate/up CAN'T be stacked because
            // the converter ships gate as F16 and up as Q4_0 — different
            // element widths can't share a single ggml_tensor.
            for (int i = 0; i < hp.n_layer; ++i) {
                auto & l = model.layers_mtl[i];
                if (l.wq->type != l.wk->type || l.wq->type != l.wv->type) {
                    fprintf(stderr, "load_model_gguf_mtl: skipping QKV stack on layer %d "
                                    "(mixed types Q=%s K=%s V=%s)\n",
                            i, ggml_type_name(l.wq->type), ggml_type_name(l.wk->type),
                            ggml_type_name(l.wv->type));
                    l.wqkv = nullptr;
                    continue;
                }
                l.wqkv = ggml_new_tensor_2d(model.ctx_stack, l.wq->type, n_embd, 3 * n_embd);
            }
            (void) n_ff;
            model.buffer_stack = ggml_backend_alloc_ctx_tensors(model.ctx_stack, model.backend);
            if (!model.buffer_stack) {
                throw std::runtime_error("load_model_gguf_mtl: ggml_backend_alloc_ctx_tensors failed for "
                                         "stacked-weights buffer (backend out of memory?)");
            }
            t3_stack_register(model.buffer_stack, model.ctx_stack);

            // Copy Q/K/V rows into wqkv via host scratch. Q4_0 row
            // layout is M-major (rows packed contiguously), so we just
            // append wq's rows, then wk's, then wv's.  The early type-
            // equality guard above implies wq/wk/wv have identical sizes
            // today, but max over all three so a future shape divergence
            // can't silently truncate a per-layer copy.
            size_t scratch_bytes = 0;
            for (int i = 0; i < hp.n_layer; ++i) {
                auto & l = model.layers_mtl[i];
                if (!l.wqkv) continue;
                scratch_bytes = std::max({scratch_bytes,
                                          ggml_nbytes(l.wq),
                                          ggml_nbytes(l.wk),
                                          ggml_nbytes(l.wv)});
            }
            std::vector<char> scratch(scratch_bytes);
            for (int i = 0; i < hp.n_layer; ++i) {
                auto & l = model.layers_mtl[i];
                if (!l.wqkv) continue;
                size_t off = 0;
                auto copy_into = [&](ggml_tensor * src, ggml_tensor * dst) {
                    const size_t nb = ggml_nbytes(src);
                    ggml_backend_tensor_get(src, scratch.data(), 0, nb);
                    ggml_backend_tensor_set(dst, scratch.data(), off, nb);
                    off += nb;
                };
                copy_into(l.wq, l.wqkv);
                copy_into(l.wk, l.wqkv);
                copy_into(l.wv, l.wqkv);
            }
        }

        {
            const int64_t jk = gguf_find_key(gguf_ctx, "tokenizer.ggml.mtl_json");
            const int64_t lk = gguf_find_key(gguf_ctx, "tokenizer.ggml.mtl_languages");
            if (jk < 0) {
                fprintf(stderr, "load_model_gguf_mtl: GGUF missing tokenizer.ggml.mtl_json; "
                                "re-run scripts/convert-t3-mtl-to-gguf.py.\n");
                gguf_free(gguf_ctx);
                ggml_free(tmp_ctx);
                return false;
            }
            if (gguf_get_kv_type(gguf_ctx, jk) != GGUF_TYPE_STRING) {
                fprintf(stderr, "load_model_gguf_mtl: tokenizer.ggml.mtl_json has unexpected GGUF type %d "
                                "(expected GGUF_TYPE_STRING); re-run scripts/convert-t3-mtl-to-gguf.py.\n",
                        (int) gguf_get_kv_type(gguf_ctx, jk));
                gguf_free(gguf_ctx);
                ggml_free(tmp_ctx);
                return false;
            }
            const char * jv = gguf_get_val_str(gguf_ctx, jk);
            if (!jv) {
                fprintf(stderr, "load_model_gguf_mtl: tokenizer.ggml.mtl_json is null\n");
                gguf_free(gguf_ctx);
                ggml_free(tmp_ctx);
                return false;
            }
            model.mtl_tokenizer_json = jv;
            if (lk >= 0 && gguf_get_kv_type(gguf_ctx, lk) == GGUF_TYPE_ARRAY &&
                gguf_get_arr_type(gguf_ctx, lk) == GGUF_TYPE_STRING) {
                const size_t n = gguf_get_arr_n(gguf_ctx, lk);
                model.mtl_languages.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    const char * s = gguf_get_arr_str(gguf_ctx, lk, i);
                    if (s) model.mtl_languages.emplace_back(s);
                }
            }
        }

        if (g_log_verbose) {
            fprintf(stderr, "load_model_gguf_mtl: ctx=%d embd=%d layers=%d heads=%d kv_heads=%d "
                            "head_dim=%d inter=%d text_vocab=%d speech_vocab=%d cond_prompt=%d\n",
                    hp.n_ctx, hp.n_embd, hp.n_layer, hp.n_head, hp.n_kv_head,
                    hp.head_dim, hp.intermediate_size,
                    hp.n_text_vocab, hp.n_speech_vocab, hp.cond_prompt_len);
            fprintf(stderr, "load_model_gguf_mtl: weights=%.2f MB  KV=%.2f MB (cond+uncond unified) "
                            "tokenizer_json=%zu bytes  languages=%zu\n",
                    ggml_backend_buffer_get_size(model.buffer_w) / (1024.0*1024.0),
                    ggml_backend_buffer_get_size(model.buffer_kv) / (1024.0*1024.0),
                    model.mtl_tokenizer_json.size(), model.mtl_languages.size());
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "load_model_gguf_mtl: %s\n", e.what());
        gguf_free(gguf_ctx); if (tmp_ctx) ggml_free(tmp_ctx);
        return false;
    }

    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
    return true;
}

bool eval_prompt_mtl(const chatterbox_model & model,
                     ggml_gallocr_t allocr,
                     int n_threads,
                     const std::vector<int32_t> & text_tokens,
                     float exaggeration,
                     std::vector<float> & logits_cond_out,
                     std::vector<float> & logits_uncond_out,
                     int & prompt_len) {
    // Metal: dispatch the cond+uncond pair through a single B=2 graph so
    // the 30 Llama-block weight reads + Metal kernel dispatches are
    // amortised over both batches.  CPU keeps the two-call path (each
    // op processes B=2 in a tight loop, so batching just doubles the
    // per-op work without saving ops; mirrors §3.20's S3Gen B=2 finding
    // that on CPU the two-call path stayed the winner).
    const bool use_b2 = !ggml_backend_is_cpu(model.backend);
    if (use_b2) {
        return run_prompt_pass_b2(model, allocr, n_threads, text_tokens,
                                  exaggeration, logits_cond_out,
                                  logits_uncond_out, prompt_len);
    }

    int plen_c = 0, plen_u = 0;
    if (!run_prompt_pass(model, allocr, n_threads, text_tokens, exaggeration,
                         /*is_uncond=*/false, logits_cond_out, plen_c)) return false;
    if (!run_prompt_pass(model, allocr, n_threads, text_tokens, exaggeration,
                         /*is_uncond=*/true, logits_uncond_out, plen_u)) return false;
    prompt_len = plen_c;
    if (plen_c != plen_u) {
        // Defensive: both passes derive prompt_len from the same hparams
        // (len_cond + n_text_tokens + 2), so a mismatch should be impossible.
        // Log loudly so anyone hitting this can root-cause instead of seeing
        // a silent "prompt eval failed".
        fprintf(stderr, "eval_prompt_mtl: cond/uncond prompt_len mismatch (%d vs %d); "
                        "graph builder is inconsistent across is_uncond\n", plen_c, plen_u);
        return false;
    }
    return true;
}

bool eval_step_mtl(const chatterbox_model & model,
                   ggml_gallocr_t allocr,
                   int n_threads,
                   int n_past,
                   int32_t token,
                   std::vector<float> & logits_cond_out,
                   std::vector<float> & logits_uncond_out) {
    // The step graph indexes `speech_pos_emb` directly with `n_past`.
    // `speech_pos_emb` only has `max_speech_tokens` rows (see
    // scripts/convert-t3-mtl-to-gguf.py: MAX_SPEECH_TOKENS=4096), so we have
    // to refuse the step before `ggml_get_rows` reads past the embedding
    // table.  In practice n_past starts at len_cond + n_text_tokens + 2
    // (~2084 with max text), so this only fires on very long generations.
    if (model.hparams.max_speech_tokens > 0 &&
        n_past >= model.hparams.max_speech_tokens) {
        fprintf(stderr, "eval_step_mtl: n_past=%d exceeds max_speech_tokens=%d; "
                        "stopping generation to avoid out-of-range speech_pos_emb lookup\n",
                n_past, model.hparams.max_speech_tokens);
        return false;
    }
    // Metal: cond+uncond batched into a single forward.  See eval_prompt_mtl.
    const bool use_b2 = !ggml_backend_is_cpu(model.backend);
    if (use_b2) {
        return run_step_pass_b2(model, allocr, n_threads, n_past, token,
                                logits_cond_out, logits_uncond_out);
    }
    if (!run_step_pass(model, allocr, n_threads, n_past, token, /*uncond=*/false,
                       logits_cond_out)) return false;
    if (!run_step_pass(model, allocr, n_threads, n_past, token, /*uncond=*/true,
                       logits_uncond_out)) return false;
    return true;
}

// Sampler (CFG + rep penalty + temperature + min_p + top_p).  Top-k clamp is
// optional (ignored if <= 0).  Mirrors the LogitsProcessorList order used by
// ChatterboxMultilingualTTS.generate.
int32_t sample_next_token_mtl(const std::vector<float> & logits_cond,
                              const std::vector<float> & logits_uncond,
                              const std::vector<int32_t> & generated,
                              const chatterbox_sampling_params & p,
                              std::mt19937 & rng,
                              int32_t stop_token) {
    const size_t V = logits_cond.size();
    std::vector<float> l(V);
    for (size_t i = 0; i < V; ++i) {
        l[i] = logits_cond[i] + p.cfg_weight * (logits_cond[i] - logits_uncond[i]);
    }

    if (p.repeat_penalty != 1.0f) {
        for (int32_t t : generated) {
            if (t < 0 || (size_t) t >= V) continue;
            if (l[t] > 0.0f) l[t] /= p.repeat_penalty;
            else             l[t] *= p.repeat_penalty;
        }
    }

    if (p.temp > 0.0f && p.temp != 1.0f) {
        for (float & x : l) x /= p.temp;
    }

    if (p.min_p > 0.0f) {
        float maxl = -INFINITY;
        for (float x : l) if (x > maxl) maxl = x;
        const float thresh = maxl + std::log(p.min_p);
        for (float & x : l) if (x < thresh) x = -INFINITY;
    }

    if (p.top_p < 1.0f && p.top_p > 0.0f) {
        std::vector<int> idx(V);
        for (size_t i = 0; i < V; ++i) idx[i] = (int) i;
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return l[a] > l[b]; });
        float maxl = l[idx[0]];
        double sum = 0.0;
        std::vector<double> probs(V);
        for (size_t i = 0; i < V; ++i) {
            probs[i] = std::exp((double)(l[i] - maxl));
            sum += probs[i];
        }
        double cum = 0.0;
        for (size_t i = 0; i < V; ++i) {
            cum += probs[idx[i]] / sum;
            if (cum >= p.top_p) {
                for (size_t j = i + 1; j < V; ++j) l[idx[j]] = -INFINITY;
                break;
            }
        }
    }

    if (p.top_k > 0 && (size_t) p.top_k < V) {
        std::vector<int> idx(V);
        for (size_t i = 0; i < V; ++i) idx[i] = (int) i;
        // Partition so that idx[k-1] holds the k-th largest logit.  The
        // earlier nth_element(begin, begin+k, ..., greater) call placed the
        // (k+1)-th largest at idx[k] and left positions [0, k) as some
        // unordered partition of the top-k, so idx[k-1] could be any
        // top-k element (often not the smallest), making the threshold
        // too high and erasing legitimate top-k logits.  See sample_next_token_ex
        // in src/main.cpp for the equivalent (correct) Turbo-side variant.
        std::nth_element(idx.begin(), idx.begin() + (p.top_k - 1), idx.end(),
                         [&](int a, int b){ return l[a] > l[b]; });
        const float cut = l[idx[p.top_k - 1]];
        for (float & x : l) if (x < cut) x = -INFINITY;
    }

    double maxl = -INFINITY;
    for (float x : l) if (x > maxl) maxl = x;
    if (!std::isfinite(maxl)) {
        // All logits are -inf or NaN — sampling cascade left the distribution
        // empty (e.g. min_p too aggressive against a flat distribution, or
        // numerical underflow mid-pipeline).  Returning `stop_token` lets the
        // outer decode loop break cleanly; the previous behaviour of returning
        // 0 happened to be a valid in-vocab speech token id and would desync
        // the sampler.
        fprintf(stderr, "sample_next_token_mtl: degenerate logits "
                        "(all -inf/NaN); returning stop_token=%d\n", stop_token);
        return stop_token;
    }
    double sum = 0.0;
    std::vector<double> probs(V);
    for (size_t i = 0; i < V; ++i) {
        probs[i] = std::exp((double)(l[i] - maxl));
        sum += probs[i];
    }
    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(rng);
    double cum = 0.0;
    for (size_t i = 0; i < V; ++i) {
        cum += probs[i];
        if (cum >= r) return (int32_t) i;
    }
    return (int32_t)(V - 1);
}

} // namespace tts_cpp::chatterbox::detail

// ============================================================================
// T3 step-graph cache test hooks
// ============================================================================
//
// Read-only observability for the cache state declared in the
// section of t3_mtl.cpp.  The cache state lives in an anonymous
// namespace inside detail::; these forwarders go through the
// `_t3_step_cache_*_for_tests` bridges defined alongside it.

#include "chatterbox_tts_test_hooks.h"

namespace tts_cpp::chatterbox::test_hooks {

size_t t3_step_graph_cache_size() {
    return tts_cpp::chatterbox::detail::_t3_step_cache_size_for_tests();
}
size_t t3_step_graph_cache_capacity() {
    return tts_cpp::chatterbox::detail::_t3_step_cache_capacity_for_tests();
}
bool t3_step_graph_cache_contains(int n_past, bool is_uncond) {
    return tts_cpp::chatterbox::detail::_t3_step_cache_contains_for_tests(n_past, is_uncond);
}
size_t t3_step_graph_cache_hits() {
    return tts_cpp::chatterbox::detail::_t3_step_cache_hits_for_tests();
}
size_t t3_step_graph_cache_misses() {
    return tts_cpp::chatterbox::detail::_t3_step_cache_misses_for_tests();
}
void t3_release_caches() {
    tts_cpp::chatterbox::detail::t3_release_caches();
}

}  // namespace tts_cpp::chatterbox::test_hooks
