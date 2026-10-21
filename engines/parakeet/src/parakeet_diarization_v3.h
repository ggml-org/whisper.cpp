#pragma once

#include "parakeet_ctc.h"

#include <cstddef>
#include <vector>

namespace parakeet {

constexpr int kMillisecondsPerSecond = 1000;
constexpr int kNemotronChunkMs = 1040;
constexpr int kNemotronSpeakerCacheFrames = 264;
constexpr int kNemotronFifoFrames = 80;
constexpr int kNemotronRightContextMs = 80;
constexpr int kNemotronUpdateFrames = 40;
constexpr int kNemotronLongFormAutoFrames = 1125;
constexpr int kNemotronLongFormChunkMs = 30000;
constexpr int kNemotronLongFormContextMs = 1040;

enum class NemotronAttention {
    Automatic,
    Unfused,
};

struct NemotronDiarizationChunk {
    std::vector<float> probabilities;
    std::vector<float> embeddings;
    int state_frames = 0;
    int chunk_frames = 0;
    bool used_coreml = false;
};

struct NemotronDiarizationFitMeasure {
    size_t device_compute_bytes = 0;
    size_t host_input_bytes = 0;
};

int run_nemotron_diarization_chunk(
    const ParakeetCtcModel & model,
    const float * mel,
    int mel_frames,
    const float * state,
    int state_frames,
    NemotronDiarizationChunk & output,
    NemotronAttention attention = NemotronAttention::Automatic);

int run_nemotron_diarization(
    const ParakeetCtcModel & model,
    const float * mel,
    int mel_frames,
    std::vector<float> & probabilities,
    NemotronAttention attention = NemotronAttention::Automatic);

int prewarm_nemotron_diarization(
    const ParakeetCtcModel & model,
    float audio_seconds);

long long nemotron_diarization_mel_frames(
    const ParakeetCtcModel & model,
    long long samples);

int nemotron_diarization_encoder_frames(
    const ParakeetCtcModel & model,
    long long mel_frames);

int nemotron_diarization_stream_mel_frames(const ParakeetCtcModel & model);

int nemotron_diarization_stream_state_frames();

int nemotron_diarization_long_form_mel_frames(const ParakeetCtcModel & model);

int nemotron_diarization_long_form_frames(
    const ParakeetCtcModel & model,
    int requested_frames);

bool nemotron_diarization_uses_long_form(
    const ParakeetCtcModel & model,
    int requested_frames,
    long long mel_frames);

bool nemotron_diarization_uses_fused_attention(const ParakeetCtcModel & model);

bool nemotron_diarization_uses_f32_matmuls(const ParakeetCtcModel & model);

int measure_nemotron_diarization(
    const ParakeetCtcModel & model,
    int mel_frames,
    int state_frames,
    NemotronDiarizationFitMeasure & output);

}
