#pragma once
// Audio8 engine internals: hyperparameters, weights, stage entry points.
//
// Audio8 is a DualAR model. A 24-layer "slow" transformer reads the ChatML
// prompt one frame at a time and emits a semantic token; a 4-layer "fast"
// transformer then expands that token into the remaining nine codebook values
// for the same frame. A separate DAC-style codec turns the ten codebooks into
// audio, and its encoder turns reference audio back into codes for cloning.
//
// The LM and the two codec halves are separate GGUFs, so a text-only build
// can skip the encoder.

#include "ggml.h"
#include "ggml-backend.h"
#include "sched_dispatch.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "audio8/tokenizer.h"

// The Core ML synthesis sidecar (src/audio8/coreml/codec-synth.h), opaque here
// so the model struct compiles on every platform; only TTS_CPP_USE_COREML
// builds ever hold a non-null one.
struct audio8_coreml_codec_context;

namespace tts_cpp {
namespace audio8 {
namespace detail {

constexpr int AUDIO8_MAX_NODES = 8192;

// Hard ceiling on the codebook count a GGUF may declare. Audio8 ships ten;
// everything from the prompt rows to the fast KV cache to the per-position
// graph set scales with this, so an unbounded value from model metadata would
// size real allocations and real pricing loops.
constexpr int AUDIO8_MAX_CODEBOOKS = 32;

// The fast head is four layers over at most num_codebooks positions, so its
// graphs are two orders of magnitude smaller than the cap above. They are held
// for the life of the model, one per position, and an arena sized for 8192
// tensor headers would hold 30 MB of host memory to describe ~135 nodes.
constexpr int AUDIO8_FAST_MAX_NODES = 512;

// EngineOptions::max_frames == 0 resolves to this, the reference default
// (~24 s of audio). Shared with the fit projector so the projected workload
// is the one the engine would run.
constexpr int AUDIO8_DEFAULT_MAX_FRAMES = 512;

// Adds the wall time of its scope to `sink`, in milliseconds. Stages are timed
// by accumulation because most of them are entered once per generated frame.
class stage_timer {
public:
    explicit stage_timer(double & sink)
        : sink_(sink), start_(std::chrono::steady_clock::now()) {}

    ~stage_timer() {
        const std::chrono::duration<double, std::milli> elapsed =
            std::chrono::steady_clock::now() - start_;
        sink_ += elapsed.count();
    }

    stage_timer(const stage_timer &) = delete;
    stage_timer & operator=(const stage_timer &) = delete;

private:
    double & sink_;
    std::chrono::steady_clock::time_point start_;
};

struct lm_hparams {
    int depth = 0;
    int hidden = 0;
    int n_head = 0;
    int n_kv = 0;
    int head_dim = 0;
    int inter = 0;
    int vocab = 0;
    int fast_depth = 0;
    int fast_hidden = 0;
    int fast_n_head = 0;
    int fast_n_kv = 0;
    int fast_head_dim = 0;
    int fast_inter = 0;
    int num_codebooks = 0;
    int codebook_size = 0;
    int semantic_begin = 0;
    int semantic_end = 0;
    int eos = 0;
    int pad = 0;
    int max_seq_len = 0;
    int ras_window = 0;
    float rope_theta = 0.0f;
    float rms_eps = 0.0f;
    float ras_top_p = 0.0f;
    float ras_temperature = 0.0f;
    bool norm_fast_input = true;
    bool qkv_bias = true;
    bool fast_qkv_bias = false;
};

// The language model's loader stacks q, k and v (and w1 and w3) into one
// weight when their rows share a layout, so a block holds either the fused
// tensor or the separate ones, never both.
struct attention_weights {
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wqkv = nullptr;
    ggml_tensor * wo = nullptr;
    ggml_tensor * wq_b = nullptr;
    ggml_tensor * wk_b = nullptr;
    ggml_tensor * wv_b = nullptr;
    ggml_tensor * wqkv_b = nullptr;
    ggml_tensor * attn_norm = nullptr;
};

struct block_weights {
    attention_weights attn;
    ggml_tensor * w1 = nullptr;
    ggml_tensor * w2 = nullptr;
    ggml_tensor * w3 = nullptr;
    ggml_tensor * w13 = nullptr;
    ggml_tensor * ffn_norm = nullptr;
};

// One transformer's KV slab, stacked layer-major. K is position-major;
// V is channel-major unless F32 flash attention selects position-major V.
struct kv_cache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    int capacity = 0;
    int stride = 0;
    // F32 flash attention reads contiguous head dimensions from both caches.
    // Selected once at load time; masked prefill uses ordinary attention over
    // the same position-major values.
    bool flash_attention = false;
};

struct lm_model {
    lm_hparams hp;

    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;
    // Hexagon's quantized projection layout is separate from canonical
    // embeddings, norms and other weights. Buffers may split at device limits.
    std::vector<ggml_backend_buffer_t> repacked_projection_buffers;
    ::tts_cpp::detail::sched_fallback sched;

    ggml_tensor * tok_emb = nullptr;
    ggml_tensor * codebook_emb = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * sem_head = nullptr;
    ggml_tensor * rope_cos = nullptr;
    ggml_tensor * rope_sin = nullptr;
    std::vector<block_weights> blocks;

    ggml_tensor * fast_emb = nullptr;
    ggml_tensor * fast_norm = nullptr;
    ggml_tensor * fast_out = nullptr;
    ggml_tensor * fast_rope_cos = nullptr;
    ggml_tensor * fast_rope_sin = nullptr;
    std::vector<block_weights> fast_blocks;

    kv_cache slow_kv;
    kv_cache fast_kv;
    // Separate arenas: the two graphs have different shapes, and alternating
    // them through one allocator would resize it on every call. fast_allocr
    // serves the per-call path only; a replayed position uses its own.
    ggml_gallocr_t slow_allocr = nullptr;
    ggml_gallocr_t fast_allocr = nullptr;
    ggml_gallocr_t frame_allocr = nullptr;

    // One built graph per fast-AR position, kept for the life of the model.
    // Every frame replays the same shapes, so rebuilding them per frame costs a
    // context, a support sweep and an allocation each time, and hands the
    // backend a graph it cannot recognise as the one it just ran. Each position
    // owns its allocator: one shared arena would move under the graphs already
    // built against it the first time a later position reserved something
    // bigger, and a graph whose tensors moved cannot be replayed either.
    struct fast_graph {
        ggml_context * ctx = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * logits = nullptr;
        ggml_gallocr_t allocr = nullptr;
        bool use_sched = false;
    };
    std::vector<fast_graph> fast_graphs;
    // Set when a build lands on the scheduler fallback, which reallocates one
    // shared arena per graph and so cannot hand out memory a graph may keep.
    bool fast_cache_off = false;
    // Whether this backend can pick codes itself, decided once at load time.
    bool picks_codes = false;
    bool precise_outputs = false;

    TokenizerData tokenizer;
};

struct transformer_spec {
    int n_layer = 0;
    int n_head = 0;
    int n_kv = 0;
    int dim = 0;
    int inter = 0;
    int window = 0;
    int head_dim = 0;
    float rope_theta = 0.0f;
    float norm_eps = 0.0f;
};

// Pre-norm block with the learned residual scales the codec's window
// transformers use in place of Audio8's plain residual add.
struct scaled_block_weights {
    attention_weights attn;
    ggml_tensor * attn_scale = nullptr;
    ggml_tensor * w1 = nullptr;
    ggml_tensor * w2 = nullptr;
    ggml_tensor * w3 = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_scale = nullptr;
};

struct window_transformer {
    transformer_spec spec;
    std::vector<scaled_block_weights> blocks;
    ggml_tensor * norm = nullptr;
    ggml_tensor * rope_cos = nullptr;
    ggml_tensor * rope_sin = nullptr;
};

// A causal conv1d and its bias. Weight norm is already folded in, so the
// weight is the materialised kernel.
struct conv_weights {
    ggml_tensor * w = nullptr;
    ggml_tensor * b = nullptr;
};

struct convnext_weights {
    conv_weights dwconv;
    ggml_tensor * norm_w = nullptr;
    ggml_tensor * norm_b = nullptr;
    conv_weights pwconv1;
    conv_weights pwconv2;
    ggml_tensor * gamma = nullptr;
};

struct resample_stage {
    conv_weights conv;
    convnext_weights convnext;
    int stride = 0;
};

// Both halves carry the codebook, but only the encoder needs the projection
// into it and only the decoder the projection back out.
struct quantizer_weights {
    ggml_tensor * codebook = nullptr;
    conv_weights in_proj;
    conv_weights out_proj;
};

struct residual_unit {
    ggml_tensor * alpha1 = nullptr;
    conv_weights conv1;
    ggml_tensor * alpha2 = nullptr;
    conv_weights conv2;
};

// Encoder: three residual units, Snake, strided conv, and on the last stage a
// window transformer. Decoder: Snake, transposed conv, three residual units.
struct dac_stage {
    ggml_tensor * alpha = nullptr;
    conv_weights conv;
    int stride = 0;
    std::vector<residual_unit> units;
    window_transformer transformer;
    bool has_transformer = false;
};

struct codec_hparams {
    int sample_rate = 0;
    int frame_size = 0;
    int num_codebooks = 0;
    int latent_dim = 0;
    int codebook_dim = 0;
    int max_frames = 0;
    int semantic_codebook_size = 0;
    int residual_codebook_size = 0;
    int residual_codebooks = 0;
    float convnext_norm_eps = 0.0f;
    float snake_epsilon = 0.0f;
    std::vector<int> encoder_strides;
    std::vector<int> decoder_strides;
    std::vector<int> residual_dilations;
};

struct codec_conv_options {
    using tiled_fn = ggml_tensor * (*)(ggml_context *, ggml_backend_t, ggml_tensor *, ggml_tensor *, int, int);
    ggml_backend_t backend = nullptr;
    tiled_fn tiled = nullptr;
    ggml_prec precision = GGML_PREC_F32;
};

struct codec_model {
    codec_hparams hp;
    bool has_encoder = false;
    bool has_decoder = false;

    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;
    ::tts_cpp::detail::sched_fallback sched;
    bool precise_outputs = false;
    codec_conv_options synthesis_convs;
    // Both directions run in two graphs: one over the whole sequence at one
    // column per frame, and one over blocks of the sample-rate stack. They get
    // an arena each, because alternating them through one would resize it on
    // every switch.
    ggml_gallocr_t allocr = nullptr;
    ggml_gallocr_t block_allocr = nullptr;

    std::vector<quantizer_weights> semantic_quantizers;
    std::vector<quantizer_weights> residual_quantizers;
    window_transformer post;
    std::vector<resample_stage> upsample;

    conv_weights dec_in;
    std::vector<dac_stage> dec_stages;
    ggml_tensor * dec_out_alpha = nullptr;
    conv_weights dec_out;

    // How much of the sample-rate stack each block covers: frames of output
    // when synthesising, columns of the encoder's 512-fold reduction when
    // analysing. Smaller blocks cost less scratch and repeat more work on each
    // block's context.
    //
    // Synthesis takes zero to mean the widest block whose scratch fits
    // synthesis_scratch_budget, which is the fastest width available at that
    // budget; a positive value pins it instead. A zero budget takes a share of
    // what the backend reports free.
    int synthesis_block_frames = 0;
    size_t synthesis_scratch_budget = 0;
    int analysis_block_columns = 128;

    // Core ML sidecar for the synthesis stack (decoder only; TTS_CPP_COREML
    // builds, compiled model next to the GGUF). Synthesis runs in windows of
    // the exported width and falls back to the ggml blocks when it cannot.
    audio8_coreml_codec_context * coreml = nullptr;
    bool synthesis_on_coreml = false;

    conv_weights enc_in;
    std::vector<dac_stage> enc_stages;
    ggml_tensor * enc_out_alpha = nullptr;
    conv_weights enc_out;
    window_transformer pre;
    std::vector<resample_stage> downsample;
};

bool load_lm(const std::string & path, int n_gpu_layers,
             const std::string & backend, lm_model & model,
             std::string * error);
void free_lm(lm_model & model);

// Backend-selector matcher: shared with tests so the alias ("hexagon" ->
// HTP/HTP0), exact-device ("HTP0", "CUDA0"), and policy ("auto", "cpu",
// "opencl") rules can be covered without having to spin up a ggml device.
// `reg` and `device` may be nullptr; nullptrs never match.
bool backend_selection_matches(const std::string & requested,
                               const char * reg, const char * device,
                               enum ggml_backend_dev_type type);

// ── Memory-fit measurement (include/tts-cpp/audio8/fit.h) ──────────────────
// Metadata-only twins of load_lm / load_codec: same backend resolution, same
// hparam reads and tensor wiring, but every allocation the real path makes is
// SIZED into `measure` instead of performed, and no tensor data is read.
// Weight and KV tensors come back marked externally allocated (dummy non-null
// data, the same trick ggml's own measure paths use) so graph pricing
// excludes them from compute buffers; a model loaded this way must never have
// tensor data read or written, and must only be used to build graphs for
// size-only pricing.
struct fit_load_measure {
    size_t weights_bytes = 0;
    size_t kv_bytes      = 0;  // LM only: the slow + fast KV slabs
};

bool load_lm_metadata_only(const std::string & path, int n_gpu_layers,
                           const std::string & backend, lm_model & model,
                           fit_load_measure & measure, std::string * error);
bool load_codec_metadata_only(const std::string & path, int n_gpu_layers,
                              const std::string & backend, codec_model & model,
                              fit_load_measure & measure, std::string * error);

// What a codec GGUF says about itself: which half it holds and the shapes it
// was converted with.
struct codec_header {
    codec_hparams hp;
    std::string part;  // "encoder" or "decoder"
};

// Reads that header and nothing else, so a set of GGUFs can be checked for
// belonging together before any of the weights are pulled off disk.
bool peek_codec_header(const std::string & path, codec_header & header,
                       std::string * error);

bool load_codec(const std::string & path, int n_gpu_layers,
                const std::string & backend, codec_model & model,
                std::string * error);
void free_codec(codec_model & model);

// Prompt frames, laid out the way the model reads them: row 0 is the semantic
// or text token, rows 1..num_codebooks the reference codebooks (zero outside a
// cloning prompt). Column-major, one column per position.
struct prompt_frames {
    std::vector<int32_t> values;
    int width = 0;
    int rows = 0;
};

prompt_frames build_frames(const lm_hparams & hp, const PromptSegments & segments,
                           const std::vector<int32_t> & reference_codes, int reference_len);

ggml_tensor * build_conv_stack(ggml_context * ctx, const codec_model & model,
                               int n_samples);

struct scratch;
struct slow_graph_outputs {
    ggml_tensor * logits  = nullptr;
    ggml_tensor * carried = nullptr;
};
void build_slow_graph(lm_model & model, scratch & build, int width, int n_past,
                      slow_graph_outputs & outs);
void set_slow_graph_inputs(const lm_model & model, ggml_cgraph * graph,
                           const int32_t * frames, int width, int n_past);

bool slow_step(lm_model & model, const int32_t * frames, int width, int n_past,
               int n_threads, std::vector<float> & sem_logits,
               std::vector<float> & fast_input, std::string * error);

int argmax_of(const std::vector<float> & values);

// Turns one position's fast-head logits into a codebook value.
using code_picker = std::function<int(const std::vector<float> & logits, int position)>;

// Expands one semantic token into num_codebooks values. codes_out is sized
// num_codebooks; entry 0 is derived from `semantic` without a forward pass.
bool fast_step(lm_model & model, const std::vector<float> & fast_input, int semantic,
               int n_threads, const code_picker & pick, std::vector<int32_t> & codes_out,
               std::string * error);

// Fit builders: the exact graphs fast_pass / fast_frame build (position 0
// primes from the slow transformer's hidden state; later positions read a
// code embedding), exposed so the fit projector can price them without
// running. build_slow_graph above already serves the slow transformer.
void build_fast_fit_graph(lm_model & model, scratch & build, int position, bool prime);
void build_fast_frame_fit_graph(lm_model & model, scratch & build);

// Codec graph measurement for the fit projector. Decode prices the
// full-sequence latent graph plus the synthesis block the runtime's own
// planner (plan_blocks) would pick against the memory available right now;
// encode prices the worst convolution block and the full-reference analysis
// graph. The device/host split follows the same dual-path dispatch
// (direct gallocr vs [backend, CPU] scheduler) prepare_graph uses.
// Returns false on a build/measure failure.
struct codec_fit_measure {
    uint64_t device_bytes = 0;
    uint64_t host_bytes   = 0;
    int      block_frames = 0;  // decode only: the planned synthesis block width
    int      block_span   = 0;  // decode only: width + causal context actually priced
};
bool measure_decode_memory(codec_model & model, int n_frames, codec_fit_measure & out);
bool measure_encode_memory(codec_model & model, int n_frames, codec_fit_measure & out);

// Encoder shape queries the fit projector sizes host slabs and RoPE-table
// checks with: transformer positions for an n_frames reference (4 per frame),
// and the width of one feature column the convolution stack hands over.
int encode_positions(const codec_model & model, int n_frames);
int encode_feature_width(const codec_model & model);

// fast_step's greedy equivalent in one graph instead of num_codebooks graphs,
// picking each code on the backend. One submission rather than ten: worth roughly
// a quarter of a millisecond per position on Metal, where a submission costs far
// more than the kernels it carries. Only call it when `model.picks_codes`.
//
// Picking on the device means ggml's argmax settles a tie between two equal
// logits differently from argmax_of, which keeps the first. That is why
// picks_codes excludes the CPU backend, whose per-position path is the reference
// for the tie-break and which has no submissions to collapse anyway.
bool fast_frame(lm_model & model, const std::vector<float> & fast_input, int semantic,
                int n_threads, std::vector<int32_t> & codes_out, std::string * error);

// Stage boundaries the parity test compares one at a time, so a divergence
// names the module that caused it instead of only showing a wrong waveform.
// Each is channels-inner: [latent_dim, frames], and [latent_dim, 4 * frames]
// for the upsampled latent.
struct decode_taps {
    std::vector<float> semantic;
    std::vector<float> residual;
    std::vector<float> post;
    std::vector<float> latent;
};

// Asked between blocks of the sample-rate stack, the one place a codec pass
// can stop: once ggml has a graph it runs to completion. Returning true ends
// the pass with CANCELLED as the error. An empty hook never cancels.
using cancel_hook = std::function<bool()>;

constexpr const char * CANCELLED = "audio8: synthesis cancelled";

// The decode pass runs as two graphs with very different shapes, so they are
// timed apart: the post transformer works one column per frame, the sample-rate
// stack expands each column into frame_size samples.
struct decode_timing {
    double latent_ms = 0.0;
    double synthesis_ms = 0.0;
    // The block width synthesis settled on and what the allocator priced it at,
    // which is the only view of a width chosen from a memory budget. On the
    // Core ML sidecar the width is the exported window and the scratch zero.
    int block_frames = 0;
    size_t block_scratch = 0;
    std::string synthesis_backend = "ggml";  // or the sidecar's compute label
    // Core ML windows synthesised while the language model was still running.
    int streamed_windows = 0;
};

// Frames of history a synthesis block or Core ML window needs before its own.
int synthesis_context_frames(const codec_model & model);

// What synthesis_block_frames == 0 resolves the scratch budget to, given a
// configured budget and what the backend reports for the device. Separate from
// the query so the arithmetic can be checked against figures a real device will
// not produce on demand.
size_t synthesis_scratch_budget(size_t configured, size_t free_bytes, size_t total_bytes);

// codes: [num_codebooks, n_frames] row-major. Writes n_frames * frame_size
// samples at the codec sample rate.
bool decode_codes(codec_model & model, const int32_t * codes, int n_frames,
                  int n_threads, const cancel_hook & cancel,
                  std::vector<float> & pcm_out, std::string * error,
                  decode_taps * taps = nullptr, decode_timing * timing = nullptr);

// The post transformer a chunk at a time: each call computes frames [done,
// end) from their codes and the keys and values the previous chunk left for its
// last window - 1 positions, so an utterance costs one pass however it is cut.
struct post_history {
    int done = 0;
    int held = 0;
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
};

// codes: [num_codebooks, end - history.done]. Appends [latent_dim, end -
// history.done] to `post`. Refuses a post transformer without a finite
// attention window, whose history would be the whole utterance.
bool extend_post(codec_model & model, const int32_t * codes, int end, int n_threads,
                 post_history & history, std::vector<float> & post, std::string * error);
bool post_has_window(const codec_model & model);

// Frame-major codes, as the language model emits them, to the codec's
// [num_codebooks, end - first] rows for frames [first, end).
std::vector<int32_t> codebook_rows(const std::vector<int32_t> & frames, int books, int first,
                                   int end);

// Synthesis that overlaps generation (window_stream.h): null unless a Core ML
// sidecar is attached. `frames` grows frame-major while the stream lives.
class window_stream;
std::unique_ptr<window_stream> open_synthesis_stream(codec_model & model,
                                                     const std::vector<int32_t> & frames,
                                                     int n_threads, const cancel_hook & cancel);

// decode_codes for a streamed utterance: the remaining windows, then the
// stitched waveform. A sidecar failure retires it and reruns the whole
// utterance on the ggml blocks, as decode_codes does.
bool finish_synthesis_stream(codec_model & model, window_stream & stream,
                             const int32_t * codes, int n_frames, int n_threads,
                             const cancel_hook & cancel, std::vector<float> & pcm_out,
                             std::string * error, decode_timing * timing);

// The convolutional encoder's output, [latent_dim, 4 * frames], and what the
// downsampling stages and the pre-module make of it, [latent_dim, frames].
struct encode_taps {
    std::vector<float> encoded;
    std::vector<float> downsampled;
};

// pcm: mono at the codec sample rate. Writes [num_codebooks, n_frames].
bool encode_audio(codec_model & model, const float * pcm, int n_samples,
                  int n_threads, const cancel_hook & cancel,
                  std::vector<int32_t> & codes_out, int & n_frames,
                  std::string * error, encode_taps * taps = nullptr);

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
