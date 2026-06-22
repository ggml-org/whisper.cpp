#pragma once

// Analytic backward pass for the CAMPPlus speaker encoder — voice-clone roadmap,
// ticket "GGML backward pass: CAMPPlus speaker encoder" (QVAC-20984).
//
// Scope: CAMPPlus maps an 80-channel Kaldi-fbank spectrogram to a 192-d speaker
// embedding. In the enrollment loop it provides the speaker-similarity loss
// between the (constant) target-WAV embedding and the generated-audio embedding.
// Only the generated-audio path needs gradients, so the gradient this class
// produces is `d(loss)/d(fbank)` — the input gradient with the model weights
// frozen. The fbank itself is differentiated further back to the waveform by a
// separate stage; this module stops at the CAMPPlus input.
//
// Why analytic (not ggml autodiff): the CAMPPlus forward leans on ops whose
// backward is not implemented in the vendored ggml (im2col-based conv1d/conv2d,
// `ggml_mean`, `ggml_sqrt`, `ggml_sigmoid`, `ggml_pad`, the seg-pool reshape /
// sum_rows / repeat chain, ...). The math here is the standard conv / batch-norm
// (pre-fused affine) / pooling / gating backward, computed in double for a
// well-conditioned reference and validated component-wise against central finite
// differences by the voiceclone gradcheck harness (Task 2 / QVAC-20979).
//
// Layout convention (mirrors `campplus_embed_cpu` in campplus.cpp):
//   1-D feature map: channel-major (C, T), access x[c * T + t].
//   2-D feature map: channel-major (C, H, W), access x[c * H * W + h * W + w].
// The public fbank in/out uses the (T, feat_dim) row-major layout of the public
// `campplus_embed` API; the transpose to/from channel-major happens internally.
//
// `CampplusBackward` owns the frozen weights and caches the per-call activations
// as state: `forward(fbank)` runs the chain and stores the activations needed by
// `backward(d_emb)`. The class has no dependency on the ggml graph or the GGUF
// loader; a thin adapter binds the real weights into `CpWeights` elsewhere.

#include <vector>

namespace tts_cpp {
namespace cp_grad {

// --- Plain data holders (double mirror of campplus.h structs) ---------------

// Conv weight in PyTorch row-major layout: Conv1d (C_out, C_in, k) flattened as
// ((co * C_in) + ci) * k + kk; Conv2d (C_out, C_in, kH, kW). `b` empty => no bias.
struct CpConv {
    std::vector<double> w;
    std::vector<double> b;
    int C_out = 0, C_in = 0;
    int k = 0;                       // Conv1d kernel
    int kH = 0, kW = 0;              // Conv2d kernel
    int stride = 1, pad = 0, dilation = 1;          // Conv1d
    int stride_h = 1, stride_w = 1, pad_h = 0, pad_w = 0;  // Conv2d
};

// Pre-fused affine batch norm: y[c] = x[c] * scale[c] + shift[c]. Frozen at
// inference, so the input-gradient is a per-channel scale.
struct CpBn {
    std::vector<double> scale;       // [C]
    std::vector<double> shift;       // [C]
};

// FCM BasicResBlock: conv1 + bn1 + relu + conv2 + bn2 (+ optional shortcut),
// residual add then relu. Conv2d, stride only on H.
struct CpResBlock {
    CpConv conv1;  CpBn bn1;
    CpConv conv2;  CpBn bn2;
    CpConv sc;     CpBn sc_bn;       // shortcut; sc.w empty => identity
    int stride_h = 1;
    bool has_shortcut = false;
};

struct CpFcm {
    CpConv conv1;  CpBn bn1;
    std::vector<CpResBlock> layer1;  // 2 blocks, first stride 2
    std::vector<CpResBlock> layer2;  // 2 blocks, first stride 2
    CpConv conv2;  CpBn bn2;         // stride (sH=2, sW=1)
};

// CAMDenseTDNNLayer: bn1+relu -> linear1(1x1) -> bn2+relu -> CAMLayer, then
// dense concat [x_in, cam_out].
struct CpCamLayer {
    CpBn bn1;
    CpConv linear1;                  // 1x1 (C_in -> bn_channels)
    CpBn bn2;
    CpConv loc;                      // linear_local (bn_channels -> growth, k, dil), no bias
    CpConv cam1;                     // 1x1 (bn_channels -> bn_channels/2), bias
    CpConv cam2;                     // 1x1 (bn_channels/2 -> growth), bias
};

struct CpCamBlock {
    int num_layers = 0;
    int kernel_size = 3;
    int dilation = 1;
    int growth = 32;
    int bn_channels = 128;
    int C_in = 0;                    // channels entering layer 0
    std::vector<CpCamLayer> layers;
};

struct CpTransit {
    CpBn bn;
    CpConv linear;                   // 1x1
};

struct CpWeights {
    int feat_dim       = 80;
    int embedding_size = 192;
    int seg_pool_len   = 100;

    CpFcm head;

    CpConv tdnn;  CpBn tdnn_bn;      // Conv1d (fcm_out -> init_channels, k=5, s=2, p=2)

    CpCamBlock block1;  CpTransit transit1;
    CpCamBlock block2;  CpTransit transit2;
    CpCamBlock block3;  CpTransit transit3;

    CpBn out_bn;                     // out_nonlinear BN
    CpConv dense;                    // 1x1 (final*2 -> embedding)
    CpBn dense_bn;                   // affine-less BN (scale = 1/sqrt(var+eps))
};

// --- Activation caches -------------------------------------------------------

// ReLU is recovered from the cached pre-activation (relu input). Conv / BN
// input-gradients need only the frozen weights, so the input tensors are not
// cached; only the values the nonlinearities and poolings need are kept.
struct CpResBlockActs {
    std::vector<double> relu1_in;    // bn1(conv1(x)) pre-relu, (planes, Ho*Wo)
    std::vector<double> relu_out_in; // (conv2 path + shortcut) pre-final-relu
    int H_in = 0, W_in = 0;          // block input dims (for conv backward)
    int H_out = 0, W_out = 0;        // block output dims
};

struct CpCamLayerActs {
    std::vector<double> relu1_in;    // bn1(x_in) pre-relu, (C_in, T)
    std::vector<double> relu2_in;    // bn2(linear1(.)) pre-relu, (bn_channels, T)
    std::vector<double> y_local;     // linear_local output, (growth, T)
    std::vector<double> h1_in;       // cam1(context)+b pre-relu, (bn_channels/2, T)
    std::vector<double> gate;        // sigmoid output, (growth, T)
    int C_in = 0;                    // layer input channels
};

struct CpFcmActs {
    std::vector<double> conv1_relu_in;          // (32, 80*T) pre-relu
    std::vector<CpResBlockActs> layer1;
    std::vector<CpResBlockActs> layer2;
    std::vector<double> conv2_relu_in;          // (32, 10*T) pre-relu
    int T = 0;                                  // FCM width (== input T)
    int H_after = 0;                            // 10
};

struct CpActs {
    int T = 0;                                  // input frames
    CpFcmActs fcm;
    std::vector<double> tdnn_relu_in;           // (init_channels, T_cam) pre-relu
    int T_cam = 0;
    std::vector<CpCamLayerActs> block1, block2, block3;
    std::vector<double> tr1_relu_in, tr2_relu_in, tr3_relu_in;  // pre-relu of each transit BN
    int tr1_Cin = 0, tr2_Cin = 0, tr3_Cin = 0;
    std::vector<double> out_relu_in;            // out_nonlinear BN pre-relu, (final, T_cam)
    std::vector<double> stats_x;                // out_nonlinear output (final, T_cam) post-relu
    std::vector<double> stats_mean;             // (final)
    std::vector<double> stats_std;              // (final)
    int final_ch = 0;
};

// --- CAMPPlus backward -------------------------------------------------------
//
// Stateful: construct with the frozen weights, call `forward(fbank, T)` (caches
// activations), then `backward(d_emb)` (consumes them). The stateless math
// primitives are private; the gradcheck self-tests reach them through a friend
// tester so each is validated individually against finite differences.
class CampplusBackward {
public:
    explicit CampplusBackward(CpWeights weights);

    const CpWeights & weights() const { return weights_; }

    // Forward: `fbank_t_by_c` is row-major (T, feat_dim). Runs the chain, caches
    // activations and returns the raw 192-d embedding.
    std::vector<double> forward(const std::vector<double> & fbank_t_by_c, int T);

    // Backward: from d_emb (embedding_size) return d_fbank in the (T, feat_dim)
    // row-major layout the forward consumes. Uses the most recent forward cache.
    std::vector<double> backward(const std::vector<double> & d_emb) const;

private:
    friend struct CampplusBackwardTester;

    // --- elementwise / pooling primitives (channel-major (C, T)) -------------
    static std::vector<double> bn_forward(const std::vector<double> & x, int C, int T,
                                          const std::vector<double> & scale,
                                          const std::vector<double> & shift);
    static std::vector<double> bn_backward_input(const std::vector<double> & d_y, int C, int T,
                                                 const std::vector<double> & scale);

    static std::vector<double> relu_forward(const std::vector<double> & x);
    // d_x = d_y * (relu_in > 0)
    static std::vector<double> relu_backward(const std::vector<double> & relu_in,
                                             const std::vector<double> & d_y);

    // d_x = d_y * s * (1 - s), s = sigmoid output (cached)
    static std::vector<double> sigmoid_backward(const std::vector<double> & s,
                                                const std::vector<double> & d_y);

    static std::vector<double> conv1d_forward(const std::vector<double> & x, int C_in, int T_in,
                                              const std::vector<double> & w, const std::vector<double> & b,
                                              int C_out, int k, int stride, int pad, int dilation,
                                              int T_out);
    static std::vector<double> conv1d_backward_input(const std::vector<double> & d_y, int C_in, int T_in,
                                                     const std::vector<double> & w, int C_out, int k,
                                                     int stride, int pad, int dilation, int T_out);

    static std::vector<double> conv2d_forward(const std::vector<double> & x, int C_in, int H, int W,
                                              const std::vector<double> & w, const std::vector<double> & b,
                                              int C_out, int kH, int kW, int sH, int sW, int pH, int pW,
                                              int H_out, int W_out);
    static std::vector<double> conv2d_backward_input(const std::vector<double> & d_y, int C_in, int H, int W,
                                                     const std::vector<double> & w, int C_out, int kH, int kW,
                                                     int sH, int sW, int pH, int pW, int H_out, int W_out);

    // mean over T (per channel): m[c] = mean_t x[c, t]
    static std::vector<double> mean_T_forward(const std::vector<double> & x, int C, int T);
    static std::vector<double> mean_T_backward(const std::vector<double> & d_m, int C, int T);

    // seg-pool then expand back to (C, T): each ceil-mode bin of seg_len holds
    // the average of its members and is tiled across them.
    static std::vector<double> seg_pool_forward(const std::vector<double> & x, int C, int T, int seg_len);
    static std::vector<double> seg_pool_backward(const std::vector<double> & d_out, int C, int T, int seg_len);

    // stats pool: (C, T) -> (2C) = concat(mean, unbiased std).
    static std::vector<double> stats_pool_forward(const std::vector<double> & x, int C, int T,
                                                  std::vector<double> & mean_out,
                                                  std::vector<double> & std_out);
    static std::vector<double> stats_pool_backward_input(const std::vector<double> & d_out,
                                                         const std::vector<double> & x, int C, int T,
                                                         const std::vector<double> & mean,
                                                         const std::vector<double> & std_);

    // --- module forward/backward -------------------------------------------
    std::vector<double> fcm_resblock_forward(const CpResBlock & blk, const std::vector<double> & x,
                                             int C_in, int H, int W, int & H_out, int & W_out,
                                             CpResBlockActs & acts) const;
    std::vector<double> fcm_resblock_backward(const CpResBlock & blk, const CpResBlockActs & acts,
                                              const std::vector<double> & d_out, int C_in) const;

    std::vector<double> fcm_forward(const std::vector<double> & fbank_ct, int T, int & T_out,
                                    CpFcmActs & acts) const;
    std::vector<double> fcm_backward(const std::vector<double> & d_out, const CpFcmActs & acts) const;

    std::vector<double> cam_layer_forward(const CpCamLayer & L, const std::vector<double> & x_in, int C_in,
                                          int T, int growth, int kernel_size, int dilation, int bn_channels,
                                          int seg_pool_len, CpCamLayerActs & acts) const;
    std::vector<double> cam_layer_backward(const CpCamLayer & L, const CpCamLayerActs & acts,
                                           const std::vector<double> & d_out, int C_in, int T, int growth,
                                           int kernel_size, int dilation, int bn_channels,
                                           int seg_pool_len) const;

    CpWeights weights_;
    mutable CpActs acts_;
};

}  // namespace cp_grad
}  // namespace tts_cpp
