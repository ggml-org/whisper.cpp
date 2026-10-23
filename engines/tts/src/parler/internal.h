#pragma once
// Parler-TTS engine internals: hparams, model weights, stage entry points.
// Everything hparam-driven from GGUF metadata — no mini/large branching.

#include "ggml.h"
#include "ggml-backend.h"
#include "sched_dispatch.h"
#include "../cosyvoice_mmap.h"
#include "dac_coreml.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

// graph-node budget for every parler graph builder (t5 / decoder / dac)
constexpr int PARLER_MAX_NODES = 4096;

// Thread count of an EngineOptions::n_threads = 0 engine (capped at the core
// count when that is known).
constexpr unsigned PARLER_DEFAULT_THREADS = 4;

// Frames per DAC decode window. The DAC is fully convolutional, so a whole-sequence
// graph grows the compute buffer without bound (~1.96 MiB/frame); windowing makes the
// peak O(1) in output length. Context of parler_dac_rf_frames() on each side keeps
// every emitted sample bit-identical to a whole-sequence decode.
constexpr int PARLER_DAC_WINDOW_FRAMES = 128;

struct parler_hparams {
    std::string variant;  // optional checkpoint family, used to validate sidecars
    // t5 encoder
    int   t5_n_layer      = 0;
    int   t5_d_model      = 0;
    int   t5_d_ff         = 0;
    int   t5_n_head       = 0;
    int   t5_d_kv         = 0;
    int   t5_rel_buckets  = 0;
    int   t5_rel_max_dist = 0;
    float t5_rms_eps      = 1e-6f;
    int   t5_vocab        = 0;
    // decoder LM
    int   dec_n_layer   = 0;
    int   dec_d_model   = 0;
    int   dec_n_head    = 0;
    int   dec_d_ff      = 0;
    int   n_codebooks   = 0;
    int   dec_vocab     = 0;   // 1088 (codes 0..1023, eos/pad 1024, bos 1025)
    float dec_ln_eps    = 1e-5f;
    int   bos_id        = 0;
    int   eos_id        = 0;
    int   pad_id        = 0;
    int   dec_start_id  = 0;
    int   max_position  = 0;   // sinusoidal table rows
    bool  enc_to_dec    = false;
    // generation defaults (from HF generation_config)
    int   gen_max_length     = 0;
    int   gen_min_new_tokens = 0;
    bool  gen_do_sample      = false;
    float gen_temperature    = 1.0f;
    int   gen_top_k          = 0;
    // dac
    int   dac_sample_rate   = 0;
    int   dac_n_q           = 0;
    int   dac_codebook_size = 0;
    int   dac_latent        = 0;
    int   dac_decoder_dim   = 0;
    int   dac_hop           = 0;
    std::vector<int> dac_rates;
    // runtime geometry
    int   n_ctx = 0;           // decoder KV capacity (prompt + delayed steps)
};

struct parler_t5_layer {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr, * o = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
};

struct parler_dec_layer {
    ggml_tensor * attn_norm_w = nullptr, * attn_norm_b = nullptr;
    ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr, * o = nullptr;
    ggml_tensor * cross_norm_w = nullptr, * cross_norm_b = nullptr;
    ggml_tensor * cq = nullptr, * ck = nullptr, * cv = nullptr, * co = nullptr;
    ggml_tensor * ffn_norm_w = nullptr, * ffn_norm_b = nullptr;
    ggml_tensor * up = nullptr, * down = nullptr;
    ggml_tensor * qkv = nullptr; // fused [d_model, 3*d_model] of q|k|v (one mul_mat)
};

struct parler_dac_residual {
    ggml_tensor * snake1_alpha = nullptr;
    ggml_tensor * snake1_inv   = nullptr;                  // 1 / (alpha + eps), CPU only
    ggml_tensor * conv1_w = nullptr, * conv1_b = nullptr;  // k7, dilated
    ggml_tensor * snake2_alpha = nullptr;
    ggml_tensor * snake2_inv   = nullptr;
    ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;  // k1
};

struct parler_dac_block {
    ggml_tensor * snake_alpha = nullptr;
    ggml_tensor * snake_inv   = nullptr;
    ggml_tensor * convt_w = nullptr, * convt_b = nullptr;  // k=2*stride
    ggml_tensor * convt_cols_w = nullptr;                  // [IC, K*OC], Hexagon only
    int stride = 0;
    parler_dac_residual res[3];                            // dilations 1, 3, 9
};

struct parler_dac_quant {
    ggml_tensor * codebook = nullptr;                      // [codebook_dim, 1024]
    ggml_tensor * out_proj_w = nullptr, * out_proj_b = nullptr;  // 1x1 conv
};

struct parler_model {
    parler_hparams hparams;

    ggml_backend_t        backend  = nullptr;
    ggml_context        * ctx_w    = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;
    // Map-in-place GGUF backing for ctx_w on the CPU backend (null on GPU / when
    // mapping fails). map_buf wraps the whole mapping; freeing it does not munmap.
    tts_cpp::cosyvoice::MappedFile mapped;
    ggml_backend_buffer_t          map_buf = nullptr;
    mutable ::tts_cpp::detail::sched_fallback sched_fb;

    // Flash-attention path for self- and cross-attention, probed at load over
    // parler_fa_kv_type(); PARLER_NO_FA (or a failed probe) keeps the manual
    // softmax(QK^T)V path over an F32 cache.
    bool      use_fa  = false;
    ggml_type kv_type = GGML_TYPE_F32;
    // On a GPU backend the DAC upsampling runs conv_transpose_1d as phase-matmuls
    // (Metal's conv_transpose kernel is ~an order slower); CPU keeps the direct op,
    // or sgemm + col2im on Accelerate builds (parler_dac_accel_enabled).
    bool      on_gpu  = false;
    bool      on_hexagon = false;

    // t5
    ggml_tensor * t5_embed = nullptr;
    ggml_tensor * t5_rel_b = nullptr;          // [n_head, rel_buckets]
    ggml_tensor * t5_output_norm = nullptr;
    std::vector<parler_t5_layer> t5_layers;
    // glue
    ggml_tensor * enc_to_dec_w = nullptr, * enc_to_dec_b = nullptr;  // optional
    ggml_tensor * embed_prompts = nullptr;
    // decoder
    ggml_tensor * embed_positions = nullptr;   // [d_model, max_position]
    std::vector<ggml_tensor *> dec_embed;      // n_codebooks tables
    std::vector<ggml_tensor *> lm_heads;       // n_codebooks heads
    ggml_tensor * lm_head_stacked = nullptr;   // [d_model, vocab*n_codebooks] (one mul_mat)
    ggml_tensor * dec_output_norm_w = nullptr, * dec_output_norm_b = nullptr;
    std::vector<parler_dec_layer> dec_layers;
    // fused-weight buffer (qkv per layer + stacked lm heads), every backend
    ggml_context        * ctx_fused    = nullptr;
    ggml_backend_buffer_t buffer_fused = nullptr;
    // dac
    std::vector<parler_dac_quant> dac_quant;
    ggml_tensor * dac_conv_in_w = nullptr, * dac_conv_in_b = nullptr;
    std::vector<parler_dac_block> dac_blocks;
    ggml_tensor * dac_snake_out_alpha = nullptr;
    ggml_tensor * dac_snake_out_inv   = nullptr;
    // DAC tensors derived from the weights at load: per-channel
    // 1 / (alpha + eps) for every snake on CPU and Hexagon, so each snake runs
    // as one fused ggml_snake node, and on Hexagon the transposed-conv kernels
    // rearranged for the columns GEMM.  Null on GPU backends.
    ggml_context        * ctx_dac_derived    = nullptr;
    ggml_backend_buffer_t buffer_dac_derived = nullptr;
    ggml_tensor * dac_conv_out_w = nullptr, * dac_conv_out_b = nullptr;
    // Reused across windows AND across calls: one bounded arena for the whole
    // process instead of a fresh large allocation per decode (streaming makes one
    // call per chunk). Created on first decode; freed by parler_free_model.
    mutable ggml_gallocr_t dac_allocr = nullptr;
    // Core ML DAC sidecar (dac_coreml.h) and the host quantizer it decodes
    // from; null on ggml. Mutable: a failing sidecar is retired mid-decode.
    mutable std::shared_ptr<parler_dac_sidecar> dac_sidecar;
    mutable bool dac_sidecar_retired = false;  // a failure retired it (STRICT keeps failing)
    bool dac_coreml_strict = false;           // captured at attachment, not read at decode time
    parler_dac_rvq dac_rvq;

    // decoder self-attention KV cache (token-major slab, one per K/V):
    // [d_model, n_ctx] rows per layer, stacked layer-major.
    ggml_context        * ctx_kv    = nullptr;
    ggml_backend_buffer_t buffer_kv = nullptr;
    ggml_tensor * memory_k = nullptr;
    ggml_tensor * memory_v = nullptr;

    // per-description cross-attention K/V (rebuilt when the description changes).
    // cross_k[l]=[d_model,T]; cross_v_t[l]=[d_model,T] non-transposed under FA,
    // else [T,d_model] transposed; both of kv_type.
    ggml_context        * ctx_cross    = nullptr;
    ggml_backend_buffer_t buffer_cross = nullptr;
    std::vector<ggml_tensor *> cross_k;
    std::vector<ggml_tensor *> cross_v_t;
    int cross_len = 0;

    // tokenizer payload (host-side; consumed by parler_tokenizer)
    std::vector<std::string> tok_pieces;
    std::vector<float>       tok_scores;
    std::vector<uint8_t>     tok_charsmap;
    int  tok_unk_id  = 2;
    int  tok_eos_id  = 1;
    bool tok_add_eos = true;

    // optional separate BPE prompt tokenizer (indic-class checkpoints);
    // absent => the unigram tokenizer above serves prompts too
    bool has_prompt_tok = false;
    std::vector<std::string> ptok_pieces;
    std::vector<std::string> ptok_merges;
    int  ptok_unk_id  = 0;
    int  ptok_bos_id  = 1;
    bool ptok_add_bos = true;
};

// ---- parler_gguf.cpp ----
bool parler_load_gguf(const std::string & path, parler_model & model,
                      int n_gpu_layers = 0, std::string * error = nullptr,
                      const std::string & backend = {});
void parler_free_model(parler_model & model);

// Sizes of the load-time buffers, filled by parler_load_gguf_metadata_only.
struct parler_fit_measure {
    size_t weights_bytes     = 0;  // ctx_w on the resolved backend (alloc+stream path)
    size_t fused_bytes       = 0;  // fused qkv stacks + stacked LM heads (every backend)
    size_t kv_bytes          = 0;  // decoder self-KV slab at the resolved kv type
    size_t dac_derived_bytes = 0;  // DAC snake reciprocals and Hexagon convt columns weights
};

// Metadata-only twin of parler_load_gguf: same backend policy, FA probe, KV
// type resolution, tensor wiring, fused-weight eligibility, and shape checks,
// but every buffer the real load allocates is SIZED instead (the size-only
// twin of ggml_backend_alloc_ctx_tensors) and no tensor data leaves the disk.
// The CPU mmap-in-place path is intentionally bypassed: the projection prices
// the allocate-and-stream fallback, which bounds the fully-touched mapping
// from above (wrong only in the strict direction).  All tensors come back
// marked externally allocated so graph pricing counts only compute scratch.
// The model must never have tensor data read or written; free with
// parler_free_model as usual (no buffers exist in measure mode).
bool parler_load_gguf_metadata_only(const std::string & path, parler_model & model,
                                    int n_gpu_layers, parler_fit_measure & measure,
                                    std::string * error = nullptr,
                                    const std::string & backend = {});

// Element type of the self-KV cache and the cross K/V under flash attention:
// F16 on GPU backends; F32 on CPU, whose FA kernel then multiplies and
// accumulates in F32 like the manual path, so it costs no precision.
ggml_type parler_fa_kv_type(bool on_gpu);

// Whether the loader fuses the decode projections: GPU loads, mmap-backed CPU
// loads (the duplicated originals stay file-backed and evictable there), and
// metadata-only measure loads.  The CPU allocate-and-stream fallback stays
// unfused so the projections are never resident twice in anonymous memory.
// Setting PARLER_NO_FUSED disables fusion everywhere.
bool parler_should_fuse_decode_weights(const parler_model & model, bool measuring);

// Fuse the per-layer q|k|v projections and the n_codebooks LM heads into
// row-concatenated tensors (fewer decode dispatches; byte-exact results).
// Exposed so the model-free fused-vs-separate parity test can exercise it on
// a synthetic model.  A non-null `measure` sizes the fused buffer and wires
// the pointers without filling (metadata-only loads have no weight data).
bool parler_fuse_decode_weights(parler_model & model,
                                parler_fit_measure * measure = nullptr);

// Dual-path graph dispatch honoring the sched_dispatch contract (gf must be
// freshly built per call; set inputs AFTER this returns true, via
// ggml_backend_tensor_set on named graph tensors).
bool parler_graph_prepare(const parler_model & model, ggml_cgraph * gf,
                          ggml_gallocr_t allocr, bool & use_sched, const char * caller);
bool parler_graph_compute(const parler_model & model, ggml_cgraph * gf,
                          bool use_sched, int n_threads, const char * caller);

// ---- parler_t5.cpp ----
// Run the T5 encoder on the description ids and produce the cross-attention
// source states [T, dec_d_model] (enc_to_dec projection applied when present),
// then precompute per-layer cross K/V into model.ctx_cross.
bool parler_encode_description(parler_model & model,
                               const std::vector<int32_t> & desc_ids,
                               int n_threads,
                               std::vector<float> * states_out = nullptr);

// (Re)allocate the persistent per-description cross-K/V tensors for a
// description of T tokens -- the exact block parler_encode_description runs
// first.  When `measure_bytes` is non-null the buffer is SIZED instead of
// allocated (added to *measure_bytes) and the tensors come back marked
// externally allocated so the T5 / decoder fit graphs can be built over them.
bool parler_alloc_cross(parler_model & model, int T, size_t * measure_bytes = nullptr);

// Fit-graph twin of the T5 encode: builds (but never runs) the same graph
// parler_encode_description computes, including the cross-K/V writes, so it
// can be priced size-only.  parler_alloc_cross must have been called for T.
// The graph lives in a thread_local arena valid until the next parler graph
// build on this thread.
ggml_cgraph * parler_build_t5_fit_graph(const parler_model & model, int T);

// ---- parler_decoder.cpp ----
// Logits handoff for prefill/step: `view` points into the graph's own compute
// buffer when that buffer is host memory (sampled in place, no download),
// otherwise into `copy`, filled by one device download.  Both the pointer and
// the data stay valid only until the next graph prepare on the same allocator,
// and the logits processors mutate the row in place.
struct parler_step_logits {
    float * view = nullptr;
    std::vector<float> copy;
};

// Prefill: prompt embeds prepended to the BOS start frame; returns logits for
// the LAST position, [n_codebooks * dec_vocab] row-major by codebook.
// n_past_out = prompt_len + 1.
bool parler_dec_prefill(const parler_model & model,
                        const std::vector<int32_t> & prompt_ids,
                        const std::vector<int32_t> & start_frame,
                        ggml_gallocr_t allocr, int n_threads,
                        parler_step_logits & logits_out, int & n_past_out);
// One decode step: frame = n_codebooks token ids (delay-mask applied by the
// caller), position = n_past.
bool parler_dec_step(const parler_model & model,
                     const std::vector<int32_t> & frame,
                     int n_past,
                     ggml_gallocr_t allocr, int n_threads,
                     parler_step_logits & logits_out);

// Fit-graph twins of the decoder graphs: build (but never run) exactly what
// parler_dec_prefill / parler_dec_step would dispatch, so the fit projector
// can price them size-only.  The description cross-K/V tensors must exist
// (parler_alloc_cross).  The graph lives in the decoder's thread_local arena,
// valid until the next parler graph build on this thread.
ggml_cgraph * parler_build_prefill_fit_graph(const parler_model & model, int prompt_tokens);
ggml_cgraph * parler_build_step_fit_graph(const parler_model & model, int n_past);

// ---- parler_gguf.cpp: DAC snake reciprocals ----
// Snake epsilon of the DAC: snake(x) = x + sin^2(alpha x) / (alpha + eps).
constexpr float PARLER_DAC_SNAKE_EPS = 1e-9f;

// Whether the DAC graph runs its snakes as fused ggml_snake nodes over the
// precomputed reciprocals: CPU and Hexagon loads whose snake alphas are F32
// (every shipped GGUF); GPU graphs keep the elementwise chain they were
// validated with.
bool parler_dac_uses_fused_snake(const parler_model & model);

bool parler_hexagon_runs_weight_type(ggml_type type);

void parler_dac_convt_cols_from_kernel(const float * kernel, int64_t K, int64_t OC,
                                       int64_t IC, float * cols);

// ---- parler_dac_accel.cpp ----
// Accelerate (cblas_sgemm) kernels for the DAC conv stack on the CPU backend
// of Apple builds; parler_dac_accel_compiled() is false elsewhere and the
// builders below then return null.  PARLER_DAC_NO_ACCEL keeps the ggml kernels.
bool parler_dac_accel_compiled();
bool parler_dac_accel_enabled(const parler_model & model);

// Stride-1 "same" conv1d (padding (K-1)/2 * dilation) of x [L, IC] with
// w [K, IC, OC], plus bias [OC] and an optional residual [L, OC]: im2col in
// cache-sized row tiles, one sgemm per tile, tiles spread over the threads.
ggml_tensor * parler_dac_accel_conv1d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w,
                                      ggml_tensor * bias, int dilation, ggml_tensor * residual);

// DAC graph pieces, exposed for the model-free kernel test.  The snake runs
// fused when alpha and its reciprocals carry one value per channel of x
// (vForce when `accel`, else ggml_snake) and as the broadcasting elementwise
// chain otherwise; the stride-1 "same" conv (padding (K-1)/2 * dilation, plus
// bias and an optional residual) takes the Accelerate kernel only for a
// per-channel bias, and with `f16_gemm` multiplies F16 im2col columns at
// default precision instead of F32 columns with F32 accumulation.
ggml_tensor * parler_dac_snake(ggml_context * ctx, bool accel, ggml_tensor * x, ggml_tensor * alpha,
                               ggml_tensor * inv, ggml_tensor * eps);
ggml_tensor * parler_dac_conv_same(ggml_context * ctx, bool accel, ggml_tensor * x, ggml_tensor * w,
                                   ggml_tensor * b, int dilation, ggml_tensor * residual,
                                   bool f16_gemm = false);

bool parler_dac_uses_f16_gemm(const parler_model & model);

// Snake over x [T, C] with per-channel alpha and 1 / (alpha + eps) on vForce.
ggml_tensor * parler_dac_accel_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha,
                                     ggml_tensor * inv);

// Host bytes of the conv kernels' im2col tiles (one per worker thread, kept
// for the thread's lifetime) for n_threads workers; 0 off the Accelerate path.
size_t parler_dac_accel_scratch_bytes(const parler_model & model, int n_threads);

// GEMM half of a transposed conv1d: columns [K*OC, IL] = w [K, OC, IC] . x^T,
// for x [IL, IC]; ggml_col2im_1d overlap-adds them into the signal.
ggml_tensor * parler_dac_accel_convt_columns(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w);

// ---- parler_dac.cpp: Hexagon transposed conv ----
ggml_tensor * parler_dac_convt_columns(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w_cols,
                                       int stride, bool f32_precision);

// Whether the DAC's CPU rounding depends on the window shape, so a windowed
// or ranged decode matches a whole-sequence one only within float tolerance:
// tinyBLAS (GGML_LLAMAFILE) and the Accelerate kernels both tile by shape.
bool parler_dac_cpu_is_shape_exact();

// ---- parler_dac.cpp ----
// DAC receptive field in latent frames, derived from the loaded conv geometry.
// Windows carry this much real context per side; the streaming emitter holds back
// the same amount so a chunk matches the whole-sequence decode bit for bit.
int parler_dac_rf_frames(const parler_model & model);

// Peak DAC compute-buffer bytes for a decode of n_frames; must not grow once
// n_frames exceeds one window plus context. Measures without allocating.
size_t parler_dac_compute_buffer_size(const parler_model & model, int n_frames);

// Fit-graph twin of one DAC decode window: builds (but never runs) the graph
// a decode of n_frames dispatches, at the same bounded window
// (min(n_frames, PARLER_DAC_WINDOW_FRAMES + 2*rf)) and the same
// convt-dispatch (phase-GEMM on GPU).  *ctx_out must be freed with ggml_free
// after pricing; returns nullptr on failure.
ggml_cgraph * parler_build_dac_fit_graph(const parler_model & model, int n_frames,
                                         ggml_context ** ctx_out);

// codes: [n_codebooks, n_frames] row-major, values in [0, codebook_size).
// Decodes frames [out_begin, out_end) — the default (0, -1) means the whole
// sequence — and writes (out_end - out_begin) * dac_hop samples. Frames outside
// the requested range are still consulted as convolution context, so the result
// is bit-identical to decoding the whole sequence and slicing.
// latent_out (optional, tests): summed RVQ latent for the same frame range,
// [latent_dim, T] ggml layout (element (c, t) at t*latent_dim + c).
// With a Core ML sidecar attached (model.dac_sidecar) the windows run on it;
// a failing sidecar is retired and the whole range is redone on ggml, unless
// the captured Core ML strict policy is enabled, which fails the call instead. backend_out
// (optional) receives where the range ran: "ggml" or the sidecar's label.
bool parler_dac_decode(const parler_model & model,
                       const int32_t * codes, int n_frames,
                       int n_threads,
                       std::vector<float> & pcm_out,
                       std::vector<float> * latent_out = nullptr,
                       int out_begin = 0, int out_end = -1,
                       std::string * err_out = nullptr,
                       std::string * backend_out = nullptr);

} // namespace detail
} // namespace parler
} // namespace tts_cpp
