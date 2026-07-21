#pragma once

// Native C++ port of chatterbox.models.voice_encoder.VoiceEncoder — a 3-layer
// unidirectional LSTM + Linear projection + ReLU + L2-normalise that turns a
// variable-length 16 kHz waveform into a 256-d speaker embedding.
//
// Used by main.cpp when --reference-audio is set, to produce the T3-side
// speaker_emb tensor without calling into Python.

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;

// Weights for a single LSTM layer, PyTorch convention:
//   w_ih: (4*H, I) float32,  gates stacked as [i, f, g, o]
//   w_hh: (4*H, H) float32
//   b_ih: (4*H,)   float32
//   b_hh: (4*H,)   float32
struct voice_encoder_lstm_layer {
    std::vector<float> w_ih;   // 4H * I
    std::vector<float> w_hh;   // 4H * H
    std::vector<float> b_ih;   // 4H
    std::vector<float> b_hh;   // 4H
    int H = 0;
    int I = 0;
};

struct voice_encoder_weights {
    int n_layers  = 3;
    int n_mels    = 40;
    int hidden    = 256;
    int embedding = 256;          // proj output dim
    std::vector<voice_encoder_lstm_layer> lstm;   // n_layers entries
    std::vector<float> proj_w;     // (embedding, hidden), row-major
    std::vector<float> proj_b;     // (embedding,)
    std::vector<float> mel_fb;     // (n_mels, 201) librosa mel filterbank for 16k/400
    // Inference knobs (loaded from GGUF metadata when available, otherwise the
    // Python defaults matching VoiceEncConfig + embeds_from_wavs).
    int   partial_frames = 160;    // ve_partial_frames
    float overlap        = 0.5f;
    float rate           = 1.3f;   // resemble's default for embeds_from_wavs
    float min_coverage   = 0.8f;
};

// Load the VE weights + mel filterbank from a chatterbox-t3-turbo.gguf.
// Returns false if the GGUF was produced by an older converter (pre-Phase 2c).
bool voice_encoder_load(const std::string & t3_gguf_path,
                        voice_encoder_weights & out);

// Run the whole embedding pipeline on a reference waveform.
//
//   wav_16k        : mono, 16 kHz, float32 in [-1, 1]
//   backend        : main ggml backend (Metal / Vulkan / CUDA / CPU).  The
//                    3-layer LSTM + final linear projection run as a single
//                    unrolled ggml graph per partial window on this backend.
//                    Pass nullptr to fall back on an internal CPU backend.
//   out            : 256-d L2-normalised speaker embedding
//
// Internally: compute 40-ch power mel → split into overlapping 160-frame
// partials → 3-layer LSTM → Linear(256,256) → ReLU → L2-norm per partial →
// mean across partials → L2-norm.  Exactly matches
// VoiceEncoder.embeds_from_wavs(..., as_spk=False) for a single utterance.
bool voice_encoder_embed(const std::vector<float> & wav_16k,
                         const voice_encoder_weights & w,
                         ggml_backend_t backend,
                         std::vector<float> & out);
