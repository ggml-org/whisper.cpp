#pragma once

// GGUF-backed FastConformer encoder: loader, ggml encoder graph, CTC head, greedy decode.
//
// Holds shared configuration and tensor handles for CTC, TDT, EOU, and Sortformer GGUFs.

#include "mel_preprocess.h"
#include "sentencepiece_bpe.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

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

    // Streaming / cache-aware encoder knobs (currently EOU-only; CTC/TDT
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

    int  tdt_pred_hidden          = 640;
    int  tdt_pred_rnn_layers      = 2;
    int  tdt_joint_hidden         = 640;
    int  tdt_num_durations        = 5;

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

enum class ParakeetModelType {
    CTC,
    TDT,
    EOU,
    SORTFORMER,
};

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
// weights for whichever engine the GGUF declares (CTC, TDT, or
// Sortformer); `model_type` selects which decoder fields are populated.
// Named `ParakeetCtcModel` for historical reasons (the CTC pipeline
// landed first); `ParakeetModel` is the recommended new name and is
// provided as a typedef alias below.
struct ParakeetCtcModel {
    ParakeetModelType model_type = ParakeetModelType::CTC;

    EncoderConfig encoder_cfg;
    MelConfig     mel_cfg;
    BpeVocab      vocab;

    int32_t blank_id   = 1024;
    int32_t vocab_size = 1025;

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
    TdtWeights                tdt;
    EouWeights                eou;
    SortformerWeights         sortformer;

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

int load_from_gguf(const std::string & gguf_path,
                   ParakeetCtcModel  & out_model,
                   int                 n_threads,
                   int                 n_gpu_layers,
                   bool                verbose);

void print_model_summary(const ParakeetCtcModel & m);

bool        model_has_gpu_backend(const ParakeetCtcModel & m);
std::string model_active_backend_name(const ParakeetCtcModel & m);
ggml_backend_t model_active_backend(ParakeetCtcModel & m);

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
    int n_enc_frames = 0;
    int d_model      = 0;
    int vocab_size   = 0;
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
// scheduling.
int run_encoder(ParakeetCtcModel   & model,
                const float        * mel,
                int                  n_mel_frames,
                int                  n_mels,
                EncoderOutputs     & out,
                int                  max_layers = -1,
                bool                 capture_intermediates = true);

std::vector<int32_t> ctc_greedy_decode(const float * logits,
                                       int           n_frames,
                                       int           vocab_size,
                                       int32_t       blank_id);

void ctc_greedy_decode_window(const float * logits,
                              int           start_frame,
                              int           end_frame,
                              int           vocab_size,
                              int32_t       blank_id,
                              int32_t     & inout_prev_token,
                              std::vector<int32_t> & out_tokens,
                              std::vector<int>     * out_first_frame = nullptr);

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

}
