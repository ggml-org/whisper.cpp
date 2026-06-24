#pragma once

// Native C++ port of S3TokenizerV2 (FunASR speech-to-token encoder, FSMN-
// attention + FSQ codebook).  Takes a 16 kHz mono float waveform and
// produces a stream of 25 Hz speech tokens — the same tokens that
// prepare-voice.py's `prompt_token` and `cond_prompt_speech_tokens`
// tensors carry.
//
// Architecture mirrors s3tokenizer.model_v2.S3TokenizerV2:
//   log_mel_spectrogram (128 mels @ 16 kHz, n_fft=400, hop=160)
//   → AudioEncoderV2:
//       Conv1d(128 → 1280, k=3, s=2, p=1) + GELU
//       Conv1d(1280 → 1280, k=3, s=2, p=1) + GELU
//       6 × ResidualAttentionBlock:
//           LayerNorm → FSMNMultiHeadAttention (q/k/v + RoPE
//               + depth-wise Conv1d "fsmn" over v, k=31)
//           LayerNorm → MLP (Linear 1280→5120 + GELU + Linear 5120→1280)
//   → FSQCodebook:
//       Linear(1280 → 8) → tanh × 0.999 → round + 1
//       → sum(h[i] * 3^i for i in 0..7)      → int token in [0, 6560]
//
// The forward pass uses a ggml graph (transformer matmuls benefit from BLAS /
// AVX); log-mel and FSQ are plain C++.

#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;

// Hyperparameters + the few small tensors host code touches directly
// (mel filterbank for log-mel, FSQ project_down for the codebook math).
//
// The ~458 MB of encoder weights (convs + 6 transformer blocks) are NOT
// mirrored here: build_encoder_ctx streams them straight from the GGUF into
// the backend weight buffer in 8 MiB chunks (see s3tokenizer.cpp /
// gguf_stream.h), so they are never dual-resident on the host.  `gguf_path`
// is the file those streamed reads target.
struct s3tokv2_weights {
    int n_mels       = 128;
    int n_state      = 1280;
    int n_head       = 20;
    int n_layer      = 6;
    int head_dim     = 64;
    int mlp_ratio    = 4;
    int fsmn_kernel  = 31;
    int fsq_levels   = 3;
    int fsq_dim      = 8;
    int codebook_size= 6561;
    int conv_stride  = 2;
    int n_fft        = 400;
    int hop          = 160;
    int sample_rate  = 16000;
    float rope_theta = 10000.0f;
    int rope_max_pos = 2048;

    // GGUF file the encoder weights are streamed from at ctx-build time.
    std::string gguf_path;

    // Mel filterbank (n_mels, n_fft/2 + 1) = (128, 201).  Used by log-mel.
    std::vector<float> mel_fb;

    // FSQ quantizer's project_down (dim=1280 → 8).  Used by the FSQ codebook
    // math on the host (s3tokv2_tokenize).
    std::vector<float> fsq_w;
    std::vector<float> fsq_b;
};

// Load from chatterbox-s3gen.gguf.  Returns false silently if the GGUF is
// pre-Phase 2e (no s3tokv2.* tensors).
bool s3tokv2_load(const std::string & s3gen_gguf_path,
                  s3tokv2_weights & out);

// Compute the log-mel spectrogram S3TokenizerV2 consumes:
//   STFT(n_fft=400, hop=160, hann) → drop last time frame →
//   |.|^2 → mel_fb @ power → log10(clip(., 1e-10)) → max(x, x.max() - 8) →
//   (x + 4) / 4
// `wav_16k` is float32 mono in [-1, 1].  Output is row-major (T, n_mels).
std::vector<float> s3tokv2_log_mel(const std::vector<float> & wav_16k,
                                   const s3tokv2_weights & w,
                                   int & out_T);

// Run the full encoder + FSQ and return a flat vector of speech tokens.
// If `max_tokens > 0`, the output is clipped to at most that many tokens
// (matching s3tokenizer.forward(max_len=...)).
//
// `backend` controls where the conformer encoder graph runs.  Pass nullptr
// to fall back on an internal ggml-cpu backend (the prior hard-coded path).
// When voice cloning is done at request time, pass the main inference
// backend to avoid shipping weights through an extra CPU-side allocation
// and to get Metal/Vulkan/CUDA kernels for the 6 transformer blocks.
bool s3tokv2_tokenize(const std::vector<float> & wav_16k,
                      const s3tokv2_weights & w,
                      int max_tokens,
                      std::vector<int32_t> & out_tokens,
                      int n_threads = 0,
                      ggml_backend_t backend = nullptr);
