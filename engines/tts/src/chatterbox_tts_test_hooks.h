// Internal test hooks for chatterbox_tts.cpp's CPU optimisation caches.
//
// These declarations let tests in src/test_*.cpp inspect cache state that is
// otherwise file-static.  They are deliberately NOT included in
// include/tts-cpp/chatterbox/s3gen_pipeline.h because production callers must
// not depend on cache layout.
//
// The hooks are populated by the persistent-cache work for the
// CPU-side multilingual TTS path (see PROGRESS.md for the design
// notes).
//
// Rules:
//  - Read-only.  Tests must NOT mutate cache state via these hooks; use
//    the public s3gen_unload() helper if a clean slate is required.
//  - Locking is internal.  All hooks acquire the same mutex used by the
//    cache writers, so concurrent calls during a synthesize() in another
//    thread are safe but may briefly block.
//  - Stable: adding new caches must add new hooks rather than reshape
//    existing ones (the test harnesses depend on this surface).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tts_cpp::chatterbox::test_hooks {

// Number of (t_val) entries in the time_mlp result cache populated lazily
// by compute_time_mlp_cached().  Multilingual = up to n_timesteps + 1
// distinct t-values per process; Turbo = up to 3 (t_span = [0, 0.5, 1]).
size_t time_mlp_result_cache_size();

// Number of ((t_val, r_val)) entries in the time_mixed result cache used
// only by the Turbo meanflow path.  Multilingual never populates this.
size_t time_emb_result_cache_size();

// Number of ggml_tensor* entries in the CPU weight mirror cache.
// Populated by cached_cpu_weights_f32(); covers flow/input_embedding +
// spk_embed_affine/{w,b} + any other weight that synthesize() reads via
// ggml_backend_tensor_get on the hot path.
size_t weight_mirror_cache_size();

// True iff the persistent (global) cfm_estimator_cache currently holds
// a built graph.  Initially false; flips to true after the first call to
// cfm_estimator_forward() and stays true until s3gen_unload().
bool cfm_estimator_cache_built();

// Returns true iff the persistent cfm_estimator_cache last built a B=2
// (CFG cond+uncond batched) graph.  Always false on CPU because the
// CPU code path keeps use_b2 = false; useful for verifying that future
// edits don't accidentally flip CPU into the B=2 path.
bool cfm_estimator_cache_b2();

// Cache key generators — exposed so tests can verify the hashing rules
// for floats (bit-cast into uint32_t / uint64_t).  Important because
// std::hash<float> mishandles -0.0 / +0.0 and NaN inconsistently across
// libstdc++/libc++.
uint32_t float_cache_key(float t_val);
uint64_t float_pair_cache_key(float t_val, float r_val);

// Returns the cached time_mlp output for `t_val` if present, or an
// empty vector if there's no entry.  Lets tests probe whether a given
// t-value was actually warmed without re-entering compute_time_mlp.
std::vector<float> peek_time_mlp_cached(float t_val);

// ---------- Round 2 (PROGRESS.md §3.33): graph + scaffolding caches ----

// Persistent encoder graph cache.  Built lazily by run_encoder() and
// invalidated when its key (T) diverges from a streaming chunk.  False
// before any synth and after s3gen_unload().
bool encoder_graph_cache_built();

// Cache key (input length T) currently held by the encoder graph
// cache.  -1 if not built; otherwise the T from the most recent build.
int  encoder_graph_cache_T();

// Persistent HiFT decoder graph cache.  Built lazily by
// run_hift_decode() and invalidated when (T_mel, T_stft) diverge.
bool hift_graph_cache_built();
int  hift_graph_cache_T_mel();
int  hift_graph_cache_T_stft();

// Persistent F0 predictor graph cache.  Built lazily by
// run_f0_predictor(); keyed on T_mel.
bool f0_graph_cache_built();
int  f0_graph_cache_T_mel();

// Sizes of the small scaffolding caches.  Each is process-wide; a
// stable set of n_fft / hop / model parameters means the steady-state
// size is small (1-2 entries each).
size_t pos_emb_cache_size();
size_t inv_alpha_cache_size();
size_t istft_kernel_cache_size();
size_t hann_window_cache_size();
size_t window_sum_cache_size();

// ---------- Round 5 (PROGRESS.md §3.36): STFT graph + kernel caches ---
//
// `run_stft` (called once per synth from the HiFT path, between
// SineGen output and the HiFT decoder) used to allocate a fresh
// 4 MB context buffer + ggml_gallocator + backend buffer + build a
// fresh conv1d graph every synth.  The graph topology depends on
// T_src (= T_mel × 480), so it must rebuild when streaming chunks
// change length.  The forward STFT analysis kernel `build_stft_kernel`
// is a pure function of n_fft (constant 16 in the chatterbox path)
// and depends on `cached_hann_window(n_fft)` — caching it eliminates
// the per-synth ~144-element trig + window build.
//
// Wired into the same s3gen_release_synth_caches() teardown as the
// other graph caches, so backend swap / s3gen_unload() leaves no
// dangling gallocator pointing at a freed backend.

bool   stft_graph_cache_built();
int    stft_graph_cache_T_src();
size_t stft_kernel_cache_size();

// ---------- Round 4 (PROGRESS.md §3.35): T3 step-graph cache ---------
//
// MTL-only.  Caches the per-(n_past, is_uncond) graph that
// `build_step_graph_mtl` constructs from scratch on every token
// decode call.  Multilingual fires this 2× per token (CFG cond +
// uncond), so a 136-token Spanish utterance previously rebuilt 272
// graphs at ~3 ms each ≈ 800 ms / synth of pure host-CPU graph
// construction work.
//
// The cache is OPT-IN at runtime via the env var
// `CHATTERBOX_T3_STEP_CACHE` (default 0).  Enabling it on a single-
// utterance workload pays the bookkeeping cost (~10 % T3
// regression) without any compensating hit benefit because each
// step has a unique n_past — the cache only pays off on synth #2+
// in long-running processes (server mode), where the second synth
// re-decodes from n_past=0 and hits every cached entry.  Tests set
// the env var explicitly.

// Number of cached step graphs currently held; 0 before any
// eval_step_mtl call, 0 after t3_release_caches().  Bounded by the
// LRU cap (`t3_step_graph_cache_capacity()`).
size_t t3_step_graph_cache_size();

// Cache capacity (LRU bound).  Covers e.g. 128 tokens × 2 modes
// out-of-the-box.  If a synth exceeds this, late tokens fall back
// to the build-then-discard path; early tokens stay cached for the
// next synth.
size_t t3_step_graph_cache_capacity();

// True iff the (n_past, is_uncond) entry is currently in the cache.
// Used by tests to verify the LRU eviction rule and to spot-check
// hits without racing on logits comparison.
bool t3_step_graph_cache_contains(int n_past, bool is_uncond);

// Number of cache hits / cache misses since the last
// t3_release_caches().  Tests use these to confirm that re-running
// a step pass with the same shape key actually re-uses the cached
// graph instead of rebuilding it.
size_t t3_step_graph_cache_hits();
size_t t3_step_graph_cache_misses();

// Explicit teardown.  Idempotent; safe to call before/after the
// main t3 backend is freed.  Production callers (CLI, Engine) call
// this from their model-free path BEFORE ggml_backend_free so the
// gallocators in cached entries release against a still-valid
// backend.
void t3_release_caches();

}  // namespace tts_cpp::chatterbox::test_hooks
