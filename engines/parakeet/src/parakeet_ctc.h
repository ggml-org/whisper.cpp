#pragma once

// GGUF-backed FastConformer encoder: loader, ggml encoder graph, CTC head, greedy decode.
//
// Holds shared configuration and tensor handles for CTC, RNN-T, TDT, EOU, and Sortformer GGUFs.

#include "ggml.h"
#include "mel_preprocess.h"
#include "sentencepiece_bpe.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_backend_sched;
typedef struct ggml_backend_sched * ggml_backend_sched_t;

namespace parakeet {

// Encoder + per-engine head hyperparameters loaded from GGUF metadata.
//
// Field naming convention:
//   - encoder fields (no prefix): apply to the FastConformer encoder
//     shared by every engine.
//   - tdt_*: TDT decoder-only fields (LSTM prediction net + joint MLP +
//     duration head). Ignored for CTC and Sortformer GGUFs.
//   - sortformer_*: Sortformer head-only fields. The `sortformer_fc_*`
//     prefix refers to the FastConformer encoder dimensions as seen by
//     the Sortformer head (`fc_d_model` is the encoder output dim, fed
//     into the encoder_proj down to `tf_d_model` for the transformer).
//     The `sortformer_tf_*` prefix refers to the post-projection
//     transformer block stack. (Runtime weight structs in
//     parakeet_sortformer.h use shorter names `D_enc` / `tf_d` for the
//     same two dimensions; treat them as synonyms for `sortformer_fc_d_model`
//     and `sortformer_tf_d_model` respectively.)
//
// Conv-module normalisation in a Conformer block.
//   - BatchNorm  -- pre-fused into (scale, shift) at convert time
//                   (CTC, TDT, offline Sortformer). Inference graph is
//                   `mul + add`, no running stats needed.
//   - LayerNorm  -- gamma/beta stored under the same `conv.batch_norm.*`
//                   keys in the original NeMo state dict; converter
//                   writes them as `conv.norm.{weight,bias}` instead
//                   of fusing. Used by the streaming-trained EOU
//                   FastConformer-RNN-T 120M.
enum class ConvNormType {
    BatchNorm,
    LayerNorm,
};

struct EncoderConfig {
    int  d_model                  = 1024;
    int  n_layers                 = 24;
    int  n_heads                  = 8;
    int  head_dim                 = 128;
    int  ff_dim                   = 4096;
    int  conv_kernel              = 9;
    int  subsampling_factor       = 8;
    int  subsampling_channels     = 256;
    int  subsampling_freq_bins    = 10;
    int  pos_emb_max_len          = 5000;
    bool xscaling                 = true;
    bool untie_biases             = true;
    bool use_bias                 = true;
    float layer_norm_eps          = 1.0e-5f;

    // Streaming / cache-aware encoder knobs (EOU and Unified metadata; CTC/TDT
    // GGUFs leave these at the offline defaults). `att_context_left/right`
    // are in **post-subsampling encoder frames**, matching NeMo's
    // `att_context_size`. `conv_causal` and `causal_downsampling` flip
    // the depthwise conv module / subsampler from symmetric padding to
    // left-only padding when the GGUF was trained that way.
    ConvNormType conv_norm_type   = ConvNormType::BatchNorm;
    bool causal_downsampling      = false;
    bool conv_causal              = false;
    int  att_context_left         = -1;     // -1 = unrestricted
    int  att_context_right        = -1;
    bool att_chunked_limited      = false;
    bool att_dynamic_chunking     = false;
    bool conv_dynamic_chunking    = false;

    int  tdt_pred_hidden          = 640;
    int  tdt_pred_rnn_layers      = 2;
    int  tdt_joint_hidden         = 640;
    int  tdt_num_durations        = 5;

    int  rnnt_pred_hidden          = 640;
    int  rnnt_pred_rnn_layers      = 2;
    int  rnnt_joint_hidden         = 640;
    int  rnnt_max_symbols_per_step = 10;

    // EOU-specific (parakeet_realtime_eou_120m-v1).
    // Predictor + joint dims mirror TDT's, but EOU has 1 LSTM layer
    // (vs 2 for TDT) and no duration head. Cache shapes + chunk size
    // come from the converter's metadata block (matching the
    // upstream NeMo `RNNTBPEModel.cache_aware_*` configuration).
    int  eou_pred_hidden              = 640;
    int  eou_pred_rnn_layers          = 1;
    int  eou_joint_hidden             = 640;
    int  eou_chunk_mel_frames         = 25;
    int  eou_cache_lookback_frames    = 70;
    int  eou_cache_time_steps         = 8;
    int  eou_max_symbols_per_step     = 5;

    int  sortformer_num_spks      = 4;
    int  sortformer_fc_d_model    = 512;
    int  sortformer_tf_d_model    = 192;
    int  sortformer_tf_n_layers   = 18;
    int  sortformer_tf_n_heads    = 8;
    int  sortformer_tf_inner_size = 768;
    bool sortformer_tf_pre_ln     = false;
};

struct SubsamplingWeights {
    ggml_tensor * conv0_w    = nullptr;
    ggml_tensor * conv0_b    = nullptr;
    ggml_tensor * conv1_dw_w = nullptr;
    ggml_tensor * conv1_dw_b = nullptr;
    ggml_tensor * conv1_pw_w = nullptr;
    ggml_tensor * conv1_pw_b = nullptr;
    ggml_tensor * conv2_dw_w = nullptr;
    ggml_tensor * conv2_dw_b = nullptr;
    ggml_tensor * conv2_pw_w = nullptr;
    ggml_tensor * conv2_pw_b = nullptr;
    ggml_tensor * out_w      = nullptr;
    ggml_tensor * out_b      = nullptr;
};

struct BlockWeights {
    ggml_tensor * norm_ff1_w = nullptr;
    ggml_tensor * norm_ff1_b = nullptr;
    ggml_tensor * ff1_l1_w   = nullptr;
    ggml_tensor * ff1_l1_b   = nullptr;
    ggml_tensor * ff1_l2_w   = nullptr;
    ggml_tensor * ff1_l2_b   = nullptr;

    ggml_tensor * norm_attn_w = nullptr;
    ggml_tensor * norm_attn_b = nullptr;
    ggml_tensor * attn_q_w    = nullptr;
    ggml_tensor * attn_q_b    = nullptr;
    ggml_tensor * attn_k_w    = nullptr;
    ggml_tensor * attn_k_b    = nullptr;
    ggml_tensor * attn_v_w    = nullptr;
    ggml_tensor * attn_v_b    = nullptr;
    ggml_tensor * attn_qkv_w  = nullptr;
    ggml_tensor * attn_qkv_b  = nullptr;
    ggml_tensor * attn_out_w  = nullptr;
    ggml_tensor * attn_out_b  = nullptr;
    ggml_tensor * attn_pos_w  = nullptr;
    ggml_tensor * pos_bias_u  = nullptr;
    ggml_tensor * pos_bias_v  = nullptr;

    ggml_tensor * norm_conv_w = nullptr;
    ggml_tensor * norm_conv_b = nullptr;
    ggml_tensor * conv_pw1_w  = nullptr;
    ggml_tensor * conv_pw1_b  = nullptr;
    ggml_tensor * conv_dw_w   = nullptr;
    ggml_tensor * conv_dw_b   = nullptr;
    // BatchNorm path (CTC / TDT / offline Sortformer): pre-fused.
    ggml_tensor * conv_bn_scale = nullptr;
    ggml_tensor * conv_bn_shift = nullptr;
    // LayerNorm path (EOU): gamma/beta over the channel dim.
    ggml_tensor * conv_norm_w   = nullptr;
    ggml_tensor * conv_norm_b   = nullptr;
    ggml_tensor * conv_pw2_w  = nullptr;
    ggml_tensor * conv_pw2_b  = nullptr;

    ggml_tensor * norm_ff2_w = nullptr;
    ggml_tensor * norm_ff2_b = nullptr;
    ggml_tensor * ff2_l1_w   = nullptr;
    ggml_tensor * ff2_l1_b   = nullptr;
    ggml_tensor * ff2_l2_w   = nullptr;
    ggml_tensor * ff2_l2_b   = nullptr;

    ggml_tensor * norm_out_w = nullptr;
    ggml_tensor * norm_out_b = nullptr;
};

struct CtcHeadWeights {
    ggml_tensor * w = nullptr;
    ggml_tensor * b = nullptr;
};

struct TdtLstmLayer {
    ggml_tensor * w_ih = nullptr;
    ggml_tensor * w_hh = nullptr;
    ggml_tensor * b_ih = nullptr;
    ggml_tensor * b_hh = nullptr;
};

struct TdtWeights {
    ggml_tensor * predict_embed = nullptr;
    std::vector<TdtLstmLayer> lstm;

    ggml_tensor * joint_enc_w  = nullptr;
    ggml_tensor * joint_enc_b  = nullptr;
    ggml_tensor * joint_pred_w = nullptr;
    ggml_tensor * joint_pred_b = nullptr;
    ggml_tensor * joint_out_w  = nullptr;
    ggml_tensor * joint_out_b  = nullptr;
};

using RnntWeights = TdtWeights;

struct NemotronLocalePrompt {
    std::string alias;
    int32_t prompt_id = -1;
};

struct UnifiedStreamingConfig {
    bool available = false;
    int left_context_frames = 0;
    int cache_time_steps = 0;
    std::vector<int32_t> allowed_chunk_frames;
    std::vector<int32_t> allowed_right_context_frames;
};

struct NemotronConfig {
    int pred_hidden = 0;
    int pred_rnn_layers = 0;
    int joint_hidden = 0;
    int max_symbols_per_step = 0;

    int num_prompts = 0;
    int prompt_width = 0;
    int prompt_input_width = 0;
    // Streaming cache geometry from GGUF metadata. The cache-aware
    // encoder reads these rather than hard-coded 56 / 8 frame counts.
    int left_context_frames = 0;
    int cache_time_steps = 0;

    std::vector<int32_t> allowed_right_context_frames;
    std::vector<int32_t> allowed_chunk_ms;
    std::vector<NemotronLocalePrompt> locale_prompts;
    std::string default_locale;
};

struct NemotronPromptWeights {
    ggml_tensor * proj_0_w = nullptr;
    ggml_tensor * proj_0_b = nullptr;
    ggml_tensor * proj_2_w = nullptr;
    ggml_tensor * proj_2_b = nullptr;
};

struct NemotronWeights {
    RnntWeights rnnt;
    NemotronPromptWeights prompt;
};

struct TdtRuntimeWeights;
struct ParakeetCtcModel;

struct NemotronStreamStepResult {
    std::vector<float> encoder_raw;
    std::vector<float> encoder_conditioned;
    std::vector<int32_t> new_token_ids;
    std::string text;
    int encoder_frames = 0;
    int decoder_steps = 0;
    double encoder_ms = 0.0;
    double decoder_ms = 0.0;
};

struct NemotronStreamState {
    struct Impl;
    std::unique_ptr<Impl> impl;

    std::vector<float> cache_channel;
    std::vector<float> cache_time;
    std::vector<int32_t> token_ids;

    int cache_length = 0;
    int prompt_id = -1;
    int right_context_frames = -1;
    int step_index = 0;
    int64_t emitted_encoder_frames = 0;
    int max_graph_encoder_frames = 0;
    bool cancelled = false;
    bool finalized = false;

    NemotronStreamState();
    ~NemotronStreamState();
    NemotronStreamState(NemotronStreamState &&) noexcept;
    NemotronStreamState & operator=(NemotronStreamState &&) noexcept;
    NemotronStreamState(const NemotronStreamState &) = delete;
    NemotronStreamState & operator=(const NemotronStreamState &) = delete;
};

struct NemotronOfflineResult {
    std::string text;
    std::vector<int32_t> token_ids;
    int encoder_frames = 0;
    double encoder_ms = 0.0;
    double decoder_ms = 0.0;
};

int run_nemotron_cache_aware_offline(
        ParakeetCtcModel & model,
        TdtRuntimeWeights & runtime,
        const float * mel,
        int n_mel_frames,
        int n_mels,
        const std::string & language,
        int right_context_frames,
        std::atomic<bool> & cancel_flag,
        NemotronOfflineResult & result);

enum class ParakeetModelType {
    CTC,
    RNNT,
    TDT,
    EOU,
    NEMOTRON,
    SORTFORMER,
    NEMOTRON_DIARIZATION,
};

struct NemotronDiarizationConfig {
    int encoder_width = 0;
    int encoder_layers = 0;
    int attention_heads = 0;
    int feed_forward_width = 0;
    int output_width = 0;
    int speakers = 0;
    int subsampling_factor = 0;
    int output_stride = 0;
    int position_limit = 0;
    float rope_base = 0.0f;
    float rotary_fraction = 0.0f;
};

struct NemotronDiarizationLayer {
    ggml_tensor * norm1_w = nullptr;
    ggml_tensor * norm1_b = nullptr;
    ggml_tensor * qkv_w = nullptr;
    ggml_tensor * attention_w = nullptr;
    ggml_tensor * attention_b = nullptr;
    ggml_tensor * norm2_w = nullptr;
    ggml_tensor * norm2_b = nullptr;
    ggml_tensor * feed_forward_in_w = nullptr;
    ggml_tensor * feed_forward_in_b = nullptr;
    ggml_tensor * feed_forward_out_w = nullptr;
    ggml_tensor * feed_forward_out_b = nullptr;
};

struct NemotronDiarizationWeights {
    ggml_tensor * feature_projection = nullptr;
    ggml_tensor * embedding_norm_w = nullptr;
    ggml_tensor * embedding_norm_b = nullptr;
    std::vector<NemotronDiarizationLayer> layers;
    ggml_tensor * final_norm_w = nullptr;
    ggml_tensor * final_norm_b = nullptr;
    ggml_tensor * encoder_projection_w = nullptr;
    ggml_tensor * encoder_projection_b = nullptr;
    ggml_tensor * upsample_w = nullptr;
    ggml_tensor * upsample_b = nullptr;
    ggml_tensor * hidden_w = nullptr;
    ggml_tensor * hidden_b = nullptr;
    ggml_tensor * speakers_w = nullptr;
    ggml_tensor * speakers_b = nullptr;
    ggml_tensor * silence_embedding = nullptr;
};

void validate_nemotron_silence_embedding_type(const ggml_tensor * tensor);

// EOU prediction-net + joint weights. Same shape as TdtWeights minus the
// duration head: `joint.out` is (vocab+1, joint_hidden) -- where vocab
// here counts the BPE pieces + `<EOU>` + `<EOB>` and the +1 is the
// transducer blank as the last index. Stored as `ggml_tensor *` into
// the GGUF mmap; dequantised once at Engine load via
// `eou_prepare_runtime` (parakeet_eou.h).
struct EouWeights {
    ggml_tensor * predict_embed = nullptr;
    std::vector<TdtLstmLayer> lstm;

    ggml_tensor * joint_enc_w  = nullptr;
    ggml_tensor * joint_enc_b  = nullptr;
    ggml_tensor * joint_pred_w = nullptr;
    ggml_tensor * joint_pred_b = nullptr;
    ggml_tensor * joint_out_w  = nullptr;
    ggml_tensor * joint_out_b  = nullptr;
};

struct SortformerTransformerBlock {
    ggml_tensor * attn_q_w  = nullptr;
    ggml_tensor * attn_q_b  = nullptr;
    ggml_tensor * attn_k_w  = nullptr;
    ggml_tensor * attn_k_b  = nullptr;
    ggml_tensor * attn_v_w  = nullptr;
    ggml_tensor * attn_v_b  = nullptr;
    ggml_tensor * attn_o_w  = nullptr;
    ggml_tensor * attn_o_b  = nullptr;
    ggml_tensor * ln1_w     = nullptr;
    ggml_tensor * ln1_b     = nullptr;
    ggml_tensor * ffn_in_w  = nullptr;
    ggml_tensor * ffn_in_b  = nullptr;
    ggml_tensor * ffn_out_w = nullptr;
    ggml_tensor * ffn_out_b = nullptr;
    ggml_tensor * ln2_w     = nullptr;
    ggml_tensor * ln2_b     = nullptr;
};

struct SortformerWeights {
    ggml_tensor * encoder_proj_w = nullptr;
    ggml_tensor * encoder_proj_b = nullptr;
    std::vector<SortformerTransformerBlock> transformer;
    ggml_tensor * head_h2h_w = nullptr;
    ggml_tensor * head_h2h_b = nullptr;
    ggml_tensor * head_h2s_w = nullptr;
    ggml_tensor * head_h2s_b = nullptr;
};

// Universal Parakeet model object. Carries the encoder + decoder
// weights for whichever engine the GGUF declares (CTC, RNN-T, TDT, or
// Sortformer); `model_type` selects which decoder fields are populated.
// Named `ParakeetCtcModel` for historical reasons (the CTC pipeline
// landed first); `ParakeetModel` is the recommended new name and is
// provided as a typedef alias below.
struct ParakeetCtcModel {
    ParakeetModelType model_type = ParakeetModelType::CTC;

    // Optional GGUF metadata tag (key `parakeet.model_variant`). Carries
    // a stable identifier for the converted checkpoint that the engine
    // can match against -- preferred over shape-based heuristics where
    // two variants share the same encoder shape (e.g. sortformer-v2 vs
    // sortformer-v2.1-aosc). Empty if the GGUF predates the key.
    std::string model_variant;

    EncoderConfig encoder_cfg;
    MelConfig     mel_cfg;
    BpeVocab      vocab;

    int32_t blank_id   = 1024;
    int32_t vocab_size = 1025;

    // Optional CTC language masks (IndicConformer multilingual aggregate
    // vocab). Empty for monolingual Parakeet CTC GGUFs. When non-empty,
    // EngineOptions::language / CLI --language selects a [start, end)
    // slice; greedy CTC also always considers blank_id.
    struct CtcLangRange {
        std::string id;
        int32_t token_start = 0;  // inclusive
        int32_t token_end   = 0;  // exclusive
    };
    std::vector<CtcLangRange> ctc_lang_ranges;

    bool supports_streaming = false;

    // EOU-specific token IDs (resolved from the GGUF's `parakeet.eou.*`
    // metadata; -1 if missing). The decoder pipeline keys on `eou_id`
    // for the segment-flush + LSTM-state-reset behaviour and treats
    // `eob_id` as a block-boundary "no-op" emitted during training.
    int32_t eou_id = -1;
    int32_t eob_id = -1;

    std::vector<int32_t> tdt_durations;

    SubsamplingWeights       subsampling;
    std::vector<BlockWeights> blocks;
    CtcHeadWeights            ctc;
    RnntWeights               rnnt;
    TdtWeights                tdt;
    EouWeights                eou;
    SortformerWeights         sortformer;
    // CPU-resident copies of the Sortformer head weights; populated at load only
    // on Mali-Vulkan, where the head runs on CPU while the encoder stays on GPU.
    SortformerWeights         sortformer_cpu;

    NemotronDiarizationConfig nemotron_diarization_cfg;
    NemotronDiarizationWeights nemotron_diarization;

    NemotronConfig nemotron_cfg;
    NemotronWeights nemotron;
    UnifiedStreamingConfig unified_cfg;

    ggml_tensor * mel_filterbank = nullptr;
    ggml_tensor * window         = nullptr;

    struct Impl;
    std::shared_ptr<Impl> impl;

    // Accessors for callers that build their own ggml graphs against the
    // GGUF-resident tensors (e.g. parakeet_tdt's per-step LSTM/joint graphs).
    // Both return `nullptr` until `load_from_gguf` succeeds.
    ggml_backend_t backend_active() const;  // Metal / CUDA / Vulkan if compiled & enabled, else CPU
    ggml_context * weights_ctx()    const;  // ggml_context that owns the GGUF tensor metadata
};

// Forward-looking name. New code should use `ParakeetModel`. The
// `ParakeetCtcModel` name is retained for backward compatibility and
// will be removed once internal call sites are migrated. A future
// pass should also split `EncoderConfig` (currently carrying both
// TDT- and Sortformer-specific fields) into `EncoderConfig` +
// `TdtConfig` + `SortformerConfig`.
using ParakeetModel = ParakeetCtcModel;

// Backend init configuration. Call before the first `load_from_gguf`
// (or Engine construction) in the process. Both are no-ops once the
// ggml-backend registry has been populated (the registry is a
// process-wide singleton); see implementation comments for the
// detailed lifetime contract.
void set_backends_directory(const std::string & dir);

ggml_backend_t init_engine_backend(bool use_gpu);
ggml_backend_t init_engine_backend(bool use_gpu, const std::string & requested);
ggml_backend_t init_engine_cpu_backend();
void set_opencl_cache_dir(const std::string & dir);

int load_from_gguf(const std::string & gguf_path,
                   ParakeetCtcModel  & out_model,
                   int                 n_threads,
                   int                 n_gpu_layers,
                   bool                verbose);

int load_from_gguf(const std::string & gguf_path,
                   ParakeetCtcModel & out_model,
                   int n_threads, int n_gpu_layers, bool verbose,
                   const std::string & backend);

// ── Memory-fit measurement (see include/parakeet/fit.h) ───────────────────

// Byte totals collected by load_from_gguf_metadata_only.
struct GgufLoadMeasure {
    size_t weights_bytes        = 0;  // default-buft weight buffers on the active backend
    size_t repack_bytes         = 0;  // CPU extra-buft (repack) weight buffers, CPU runs only
    size_t sortformer_cpu_bytes = 0;  // Mali-Vulkan CPU head weight copies
};

// Metadata-only twin of load_from_gguf: identical backend resolution, GGUF
// parsing, and tensor wiring, but every allocation a real load makes is
// *sized* into `out_measure` instead of performed, and no tensor data is ever
// read (weight upload, mel filterbank/window, Core ML sidecar, and the
// Sortformer CPU head copy are all skipped). On success every weight tensor
// is marked externally-allocated (dummy non-NULL `data`), so the model can
// build and measure compute graphs -- ggml_gallocr then excludes the weights
// from the measured compute buffers -- but it must NEVER be used for
// inference or have tensor data read/written through it.
int load_from_gguf_metadata_only(const std::string & gguf_path,
                                 ParakeetCtcModel  & out_model,
                                 int                 n_threads,
                                 int                 n_gpu_layers,
                                 bool                verbose,
                                 GgufLoadMeasure   & out_measure);

// Measure the compute buffer the offline encoder graph at `n_mel_frames`
// would allocate on the model's active backend, without allocating or
// executing it. Requires a model from load_from_gguf_metadata_only (a
// real-loaded model works too, but pays nothing less). Returns the
// build_encoder_graph rc contract (0 = ok).
int measure_encoder_compute(ParakeetCtcModel & model,
                            int                n_mel_frames,
                            int                n_mels,
                            size_t           & out_bytes);

// Byte totals for a decoder runtime (transducer LSTM/joint or EOU predictor),
// measured without allocating. Filled by tdt_measure_runtime /
// eou_measure_runtime against a load_from_gguf_metadata_only model.
struct DecoderFitMeasure {
    size_t device_state_bytes   = 0;  // persistent decoder state (h/c/pred/enc_proj)
    size_t device_compute_bytes = 0;  // fixed-shape decode graphs + worst-case enc_proj graph
    size_t host_bytes           = 0;  // host-dequantised f32 weights (CPU decode path)
};

// ── Nemotron fit measurement (see include/parakeet/fit.h) ──────────────────

// Measure the compute buffer of the Nemotron locale-prompt projection graph
// at `n_frames` encoder frames -- the graph run_nemotron_prompt_projection
// builds and caches (one graph resident at a time, keyed by frame count) --
// without allocating it. Requires a Nemotron model (metadata-only or real).
int measure_nemotron_prompt_compute(ParakeetCtcModel & model,
                                    int                n_frames,
                                    size_t           & out_bytes);

// Measure the run_subsampling graph at `n_mel_frames` (per-chunk Nemotron
// streaming builds a fresh subsampling graph through the shared scheduler
// every step). `out_active_bytes` is the active-backend reservation, sized
// with a gallocr on its default buffer type -- the subsampling ops run
// single-split on every active backend, so this matches the scheduler's
// device buffer. `out_host_input_bytes` is the CPU-side scheduler buffer
// holding the graph-input originals on GPU runs (the scheduler assigns
// inputs to the CPU backend and copies them in; zero on CPU-only runs).
// test-fit-params asserts the sum equals the scheduler's real reservation
// byte for byte; see the implementation comment for why the scheduler's own
// size-only reserve cannot run on a metadata-only model.
int measure_subsampling_compute(ParakeetCtcModel & model,
                                int                n_mel_frames,
                                int                n_mels,
                                size_t           & out_active_bytes,
                                size_t           & out_host_input_bytes);

// Byte totals for ONE cache-aware Nemotron streaming session at a given
// right-context operating point (StreamingOptions::chunk_ms resolves to one
// of parakeet.nemotron.allowed_right_context_frames). Filled by
// nemotron_measure_stream against a load_from_gguf_metadata_only model.
struct NemotronStreamFitMeasure {
    // Steady-state step-graph gallocr buffer on the active backend: encoder
    // input, per-layer channel/time cache I/O tensors, and the compute
    // transients, all owned by one allocator for the session's life.
    size_t device_step_graph_bytes  = 0;
    // Per-chunk pre-encode (subsampling) graph on the active backend (the
    // shared scheduler's device reservation; see measure_subsampling_compute).
    size_t device_subsampling_bytes = 0;
    // CPU-side scheduler buffer holding the subsampling graph-input originals
    // on GPU runs (zero on CPU-only runs). Already INCLUDED in host_bytes;
    // broken out so parity tests can reconstruct the scheduler's full
    // reservation as device_subsampling_bytes + this.
    size_t host_subsampling_input_bytes = 0;
    // Session state living in host RAM: per-layer channel/time caches, the
    // step graph's position/mask mirrors, mel chunk staging, the per-step
    // encoder/prompt staging vectors, the RNN-T decode state, and the
    // scheduler's CPU-side subsampling input buffer.
    size_t host_bytes               = 0;
};

// Size everything one live Nemotron stream session keeps resident at the
// given operating point, without allocating. Returns 0 on success.
int nemotron_measure_stream(ParakeetCtcModel         & model,
                            int                        right_context_frames,
                            NemotronStreamFitMeasure & out);

// Actually-allocated byte counts on a REAL-loaded model, for fit-parity tests
// and stats. Both return 0 on a metadata-only model.
//
// Total bytes of the allocated weight buffers (default buffer + CPU repack
// extra buffers; excludes the Mali Sortformer CPU head copy).
size_t model_weights_buffer_bytes(const ParakeetCtcModel & m);
// Backend bytes held by the most recently built cached encoder graph: its
// allocator's buffer plus its positional-projection cache (0 before any build).
size_t model_encoder_compute_buffer_bytes(const ParakeetCtcModel & m);
// Backend buffer held by the cached Nemotron prompt-projection graph's
// allocator (0 until run_nemotron_prompt_projection has built one).
size_t model_nemotron_prompt_buffer_bytes(const ParakeetCtcModel & m);
// Backend buffer held by a live Nemotron stream session's step-graph
// allocator (0 before the first stream step builds it).
size_t nemotron_stream_graph_buffer_bytes(const NemotronStreamState & state);

void print_model_summary(const ParakeetCtcModel & m);
const char * model_type_name(ParakeetModelType model_type);

void validate_nemotron_model(const ParakeetCtcModel & model);
void validate_unified_streaming_model(const ParakeetCtcModel & model);
int32_t resolve_nemotron_prompt_id(
    const ParakeetCtcModel & model,
    const std::string & language);

bool        model_has_gpu_backend(const ParakeetCtcModel & m);
// True when a GPU was detected but routed to CPU as a known-bad backend (Mali).
// Lets hosts treat the CPU backend as expected, not a GPU regression.
bool        model_gpu_unsupported(const ParakeetCtcModel & m);
std::string model_active_backend_name(const ParakeetCtcModel & m);
ggml_backend_t model_active_backend(ParakeetCtcModel & m);

// True when the FastConformer encoder runs on the Apple Core ML sidecar (ANE/GPU)
// instead of the ggml backend. Always false on non-Apple / non-Core ML builds.
bool        model_encoder_on_coreml(const ParakeetCtcModel & m);
bool        model_bypass_encoder_on_coreml(const ParakeetCtcModel & m);
// Fixed mel-frame capacity advertised by the active Core ML sidecar, or 0 when
// Core ML is unavailable, disabled, flexible-shape, or has an unknown layout.
int         model_coreml_fixed_mel_frames(const ParakeetCtcModel & m);
int         model_coreml_bypass_fixed_frames(const ParakeetCtcModel & m);
// Encoder compute-backend label for stats/logging: the Core ML label (e.g. "coreml")
// when the sidecar is active, otherwise the ggml active-backend name.
std::string model_encoder_backend_name(const ParakeetCtcModel & m);

// Backend for the Sortformer head: the active backend normally, but CPU on
// Mali-Vulkan (its transformer block 0 miscomputes to NaN; encoder stays on GPU).
ggml_backend_t model_sortformer_backend(const ParakeetCtcModel & m);

// True when the head is routed to CPU (Mali-Vulkan); the graph then reads the
// CPU-resident weight copies (model.sortformer_cpu), not the GPU originals.
bool model_sortformer_on_cpu(const ParakeetCtcModel & m);

// The shared compute scheduler (active backend + CPU). Graphs run through it get
// per-op CPU fallback. Returns nullptr if the model is not loaded.
ggml_backend_sched_t model_sched(const ParakeetCtcModel & m);

int run_subsampling(ParakeetCtcModel   & model,
                    const float        * mel,
                    int                  n_mel_frames,
                    int                  n_mels,
                    std::vector<float> & out_feats,
                    int                & out_n_frames);

struct EncoderOutputs {
    std::vector<float> subsampling_out;
    std::vector<float> block_0_post_ff1;
    std::vector<float> block_0_post_attn;
    std::vector<float> block_0_post_conv;
    std::vector<float> block_0_post_ff2;
    std::vector<float> block_0_out;
    std::vector<float> block_last_out;
    std::vector<float> encoder_out;
    std::vector<float> logits;

    // Diagnostic: fine-grained snapshots inside block 0's
    // relative-position multi-head attention. Populated only when
    // capture_intermediates=true. Empty on the flash-attn path (the
    // whole attention collapses into one FLASH_ATTN_EXT node — nothing
    // to intercept). Order: qkv → ac (content×query) → bd (pos×query) →
    // scores (ac + rel_shift(bd)) → softmax → out (y before residual).
    std::vector<float> block_0_attn_qkv;
    std::vector<float> block_0_attn_k_raw;
    std::vector<float> block_0_attn_v_raw;
    std::vector<float> block_0_attn_q_perm;
    std::vector<float> block_0_attn_k_perm;
    std::vector<float> block_0_attn_q_u;
    std::vector<float> block_0_attn_ac;
    std::vector<float> block_0_attn_bd;
    std::vector<float> block_0_attn_scores;
    std::vector<float> block_0_attn_softmax;
    std::vector<float> block_0_attn_out;

    int n_enc_frames = 0;
    int d_model      = 0;
    int vocab_size   = 0;

    // True only when this invocation completed through the Core ML sidecar.
    // Loading a sidecar is insufficient: an incompatible shape or prediction
    // failure can still make run_encoder fall back to ggml.
    bool used_coreml = false;
};

// `capture_intermediates`: when true (default, kept for backward compat with
// the per-stage parity harnesses such as `test-encoder` /
// `test-tdt-encoder-parity` / `test-sortformer-parity`), every per-stage
// capture tensor (subsampling_out, block_0_post_*, block_0_out,
// block_last_out) is copied back to `out`. When false, only `encoder_out`
// and (CTC GGUFs only) `logits` are copied -- the production path
// (`Engine::transcribe()`, `StreamSession::process_window()`,
// `Engine::diarize()` etc.) doesn't need the intermediates and pays a
// 5+ MB host-copy round-trip per inference today, which is real
// per-call cost on GPU/OpenCL backends and negligible-but-noisy on
// CPU. The graph topology is unchanged either way -- only the
// host-copy step is gated, so this is safe regardless of backend
// scheduling. `allow_coreml_padded` is reserved for offline callers that know
// trailing zero mel frames represent padding, not an in-progress streaming
// window; it lets a fixed-shape Core ML sidecar consume that padded input.
int run_encoder(ParakeetCtcModel   & model,
                const float        * mel,
                int                  n_mel_frames,
                int                  n_mels,
                EncoderOutputs     & out,
                int                  max_layers = -1,
                bool                 capture_intermediates = true,
                bool                 allow_coreml_padded = false);

// Apply Nemotron's locale one-hot concatenation and two-layer projection to
// row-major encoder output shaped (n_frames, d_model).
int run_nemotron_prompt_projection(
    ParakeetCtcModel & model,
    const float * encoder_out,
    int n_frames,
    int d_model,
    int32_t prompt_id,
    std::vector<float> & projected);

int init_nemotron_stream_state(
    const ParakeetCtcModel & model,
    const std::string & language,
    int right_context_frames,
    NemotronStreamState & state);

int append_nemotron_mel_frames(
    NemotronStreamState & state,
    const float * mel,
    int n_frames,
    int n_mels);

int append_nemotron_pcm(
    const ParakeetCtcModel & model,
    NemotronStreamState & state,
    const float * samples,
    int n_samples,
    bool finalize);

int next_nemotron_processed_signal(
    NemotronStreamState & state,
    int n_mels,
    bool finalize,
    std::vector<float> & processed_signal,
    int & n_frames);

int run_nemotron_stream_step(
    ParakeetCtcModel & model,
    TdtRuntimeWeights & runtime,
    const float * processed_signal,
    int n_mel_frames,
    int n_mels,
    bool finalize,
    NemotronStreamState & state,
    NemotronStreamStepResult & result);

void cancel_nemotron_stream(NemotronStreamState & state);
void reset_nemotron_stream(NemotronStreamState & state);
int nemotron_pending_mel_frames(const NemotronStreamState & state);

// Run the conformer block stack on pre-subsampled embeddings, skipping the
// subsampling/pre_encode block. Used by the v2.1 streaming (AOSC) path where
// the speaker cache + FIFO + new chunk are concatenated in pre-encode space and
// re-contextualised by the conformer layers in a single forward.
//
//   pre_encode_in: row-major (n_pre_encode_frames, d_model)
//   d_model:       must equal model.encoder_cfg.d_model
//   out.encoder_out is filled with the post-encoder (n_pre_encode_frames, d_model) slab.
//
// Capture-intermediate fields on `out` are always cleared (no per-stage capture
// in this path -- the production AOSC path only consumes `encoder_out`).
int run_encoder_bypass_pre_encode(
    ParakeetCtcModel   & model,
    const float        * pre_encode_in,
    int                  n_pre_encode_frames,
    int                  d_model,
    EncoderOutputs     & out,
    int                  max_layers = -1);

// Optional language mask for multilingual CTC. token_end < 0 means
// full-vocab greedy (default / monolingual). When token_end > token_start,
// argmax considers [token_start, token_end) and always blank_id.
struct CtcDecodeOptions {
    int32_t token_start = 0;
    int32_t token_end   = -1;
};

bool find_ctc_language_range(const ParakeetCtcModel & model,
                             const std::string      & language,
                             int32_t                & out_start,
                             int32_t                & out_end);

// Resolve EngineOptions::language / CLI --language into CTC decode options.
// Empty language with masks present throws (required). Non-empty language with
// no masks returns full-vocab options (ignored). Unknown id with masks throws.
CtcDecodeOptions resolve_ctc_decode_options(const ParakeetCtcModel & model,
                                            const std::string      & language);

std::vector<int32_t> ctc_greedy_decode(const float * logits,
                                       int           n_frames,
                                       int           vocab_size,
                                       int32_t       blank_id,
                                       const CtcDecodeOptions * opts = nullptr);

void ctc_greedy_decode_window(const float * logits,
                              int           start_frame,
                              int           end_frame,
                              int           vocab_size,
                              int32_t       blank_id,
                              int32_t     & inout_prev_token,
                              std::vector<int32_t> & out_tokens,
                              std::vector<int>     * out_first_frame = nullptr,
                              const CtcDecodeOptions * opts = nullptr);

struct BlockSubstageTimes {
    double ff1_ms  = 0.0;
    double attn_ms = 0.0;
    double conv_ms = 0.0;
    double ff2_ms  = 0.0;
    double norm_out_ms = 0.0;
    double block_full_ms = 0.0;
};

int profile_block_substages(ParakeetCtcModel & model,
                            int T_enc,
                            int warmup_runs,
                            int timed_runs,
                            BlockSubstageTimes & out);

// Test diagnostics: whether fused attention is compiled in, whether the encoder
// activation stream is compiled to run in F16 (PARAKEET_F16_ACTIVATIONS build
// flag), and whether the encoder graph built for this model contains a given
// op.
bool flash_attn_compiled();
bool f16_activations_compiled();
bool encoder_graph_uses_op(const ParakeetCtcModel & model, enum ggml_op op);

}
