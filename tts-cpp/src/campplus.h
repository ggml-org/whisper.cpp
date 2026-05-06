#pragma once

// Native C++ port of CAMPPlus (FunASR / 3D-Speaker).  Takes an 80-channel
// Kaldi-fbank spectrogram at 16 kHz and produces a 192-d speaker embedding
// that's used by S3Gen's spk_embed_affine layer (the `embedding` tensor that
// prepare-voice.py currently dumps).
//
// Architecture (see chatterbox.models.s3gen.xvector):
//   FCM (Conv2d head: 4× Conv2d + 4× BasicResBlock + down-conv)
//   → xvector.tdnn (Conv1d k=5 stride=2)
//   → block1 (12 × CAMDenseTDNNLayer, kernel=3, dilation=1)
//   → transit1 (Conv1d 1x1, halves channels)
//   → block2 (24 × CAMDenseTDNNLayer, dilation=2)
//   → transit2
//   → block3 (16 × CAMDenseTDNNLayer, dilation=2)
//   → transit3
//   → out_nonlinear (BN + ReLU)
//   → stats_pool (mean + unbiased std over time, concatenated)
//   → dense (Conv1d 1x1 to 192, BN affine=False)
//
// Every BatchNorm has been pre-fused in the converter to a (scale, shift)
// vector pair, so the C++ side never sees gamma/beta/mean/var directly.

#include <cstdint>
#include <string>
#include <vector>

typedef struct ggml_backend * ggml_backend_t;

// -----------------------------------------------------------------------------
// Weight containers
// -----------------------------------------------------------------------------

struct campplus_conv {
    std::vector<float> w;          // weight, row-major as PyTorch stores it
    std::vector<float> b;          // optional bias; empty if the layer had bias=False
    int C_out = 0, C_in = 0, k = 0;         // for Conv1d
    int kH = 0, kW = 0;                     // for Conv2d (k == kH*kW omitted)
    int stride_h = 1, stride_w = 1;
    int pad_h = 0, pad_w = 0;
    int dilation_h = 1, dilation_w = 1;
    bool is_2d = false;
};

struct campplus_bn {
    std::vector<float> scale;      // gamma / sqrt(var + eps)  (or 1/sqrt for affine=False)
    std::vector<float> shift;      // beta - mean*scale        (or -mean*scale for affine=False)
};

// FCM's BasicResBlock:
//   bn1(in_ch=32) + conv1 + bn2 + conv2 + shortcut (optional)
struct campplus_res_block {
    campplus_conv conv1;
    campplus_bn   bn1;
    campplus_conv conv2;
    campplus_bn   bn2;
    // Shortcut is only present on stride>1 / channel-change blocks: conv1x1 + BN.
    // When absent, shortcut_conv.w is empty.
    campplus_conv shortcut_conv;
    campplus_bn   shortcut_bn;
    int stride_h = 1;
};

struct campplus_fcm {
    campplus_conv conv1;           // (1 → 32, k=3, s=1, p=1)
    campplus_bn   bn1;
    std::vector<campplus_res_block> layer1;  // 2 blocks, first has stride=2
    std::vector<campplus_res_block> layer2;  // 2 blocks, first has stride=2
    campplus_conv conv2;           // (32 → 32, k=3, s=(2,1), p=1)
    campplus_bn   bn2;
};

// CAMDenseTDNNLayer:
//   nonlinear1 (BN + ReLU) → linear1 (Conv1x1) → nonlinear2 (BN + ReLU)
//   → cam_layer (linear_local + 2-stage context attention)
struct campplus_cam_dense_tdnn_layer {
    campplus_bn   bn1;
    campplus_conv linear1;         // Conv1x1 (C_in → bn_channels=128)
    campplus_bn   bn2;
    // CAMLayer
    campplus_conv cam_linear_local;     // Conv1d (128 → 32, k=3, dilation)
    campplus_conv cam_linear1;          // Conv1x1 (128 → 64) with bias
    campplus_conv cam_linear2;          // Conv1x1 (64 → 32)  with bias
};

struct campplus_cam_block {
    int num_layers = 0;
    int kernel_size = 3;
    int dilation = 1;
    std::vector<campplus_cam_dense_tdnn_layer> layers;
};

struct campplus_transit {
    campplus_bn   bn;
    campplus_conv linear;          // Conv1x1
};

struct campplus_weights {
    int feat_dim      = 80;
    int embedding_size= 192;
    int seg_pool_len  = 100;
    int sample_rate   = 16000;

    campplus_fcm head;

    // xvector sub-modules.
    campplus_conv tdnn_linear;     // Conv1d (320 → 128, k=5, s=2, padding=2)
    campplus_bn   tdnn_bn;

    campplus_cam_block block1;     // 12 layers, dilation=1
    campplus_transit   transit1;
    campplus_cam_block block2;     // 24 layers, dilation=2
    campplus_transit   transit2;
    campplus_cam_block block3;     // 16 layers, dilation=2
    campplus_transit   transit3;

    campplus_bn   out_nonlinear_bn;

    campplus_conv dense_linear;    // Conv1x1 (1024 → 192)
    campplus_bn   dense_bn;        // BN affine=False (scale = 1/sqrt(var+eps))
};

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

// Loads CAMPPlus weights from an s3gen GGUF that was produced with the Phase 2d
// converter.  Returns false if the GGUF predates the CAMPPlus embedding, or if
// any expected tensor is missing.
bool campplus_load(const std::string & s3gen_gguf_path,
                   campplus_weights & out);

// Runs the whole CAMPPlus forward pass on a single utterance's Kaldi-fbank
// feature matrix.
//
//   fbank_t_by_c : row-major (T, 80) log-fbank at 16 kHz.  Per-utterance mean
//                  over T must ALREADY be subtracted (that's what
//                  extract_feature() in xvector.py does before forwarding).
//   backend      : ggml backend used to run the forward graph.  Pass nullptr
//                  for the legacy scalar CPU path (used by test harnesses);
//                  pass the main inference backend for Metal / Vulkan / CUDA.
//   out          : 192-d speaker embedding (raw, NOT L2-normalised — matches
//                  what prepare-voice.py stores in embedding.npy).
bool campplus_embed(const std::vector<float> & fbank_t_by_c,
                    int T,
                    const campplus_weights & w,
                    ggml_backend_t backend,
                    std::vector<float> & out);
