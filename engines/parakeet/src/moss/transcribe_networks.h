#pragma once

#include "moss/transcribe_model.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace parakeet::moss::detail {

void validate_transcribe_encoder(const TranscribeModel & model);
void validate_transcribe_decoder(const TranscribeModel & model);

std::vector<float> read_mel_filters(const TranscribeModel & model);

struct TranscribeChunkEncoding {
    std::vector<float> encoder_states;
    std::vector<float> embeddings;
};

TranscribeMemory measure_audio_chunk(TranscribeModel & model, int tokens);

// The decoder's causal mask, in the element type of its graph input: F16 for
// fused attention, F32 otherwise. The host holds this one buffer per batch.
ggml_type transcribe_mask_type(const TranscribeModel & model);
uint64_t transcribe_mask_bytes(const TranscribeModel & model, int64_t positions, int64_t tokens);
std::vector<uint8_t> build_causal_mask(const TranscribeModel & model, int64_t position, int64_t tokens);
std::vector<uint8_t> build_window_mask(const TranscribeModel & model, int64_t position, int64_t tokens,
                                       int64_t columns);

TranscribeChunkEncoding encode_audio_chunk(TranscribeModel & model, const std::vector<float> & mel, int tokens,
                                           bool keep_encoder_states);

class TranscribeDecoder {
public:
    TranscribeDecoder(TranscribeModel & model, int n_ctx);
    ~TranscribeDecoder();
    TranscribeDecoder(const TranscribeDecoder &) = delete;
    TranscribeDecoder & operator=(const TranscribeDecoder &) = delete;

    uint64_t measure_cache() const;
    TranscribeMemory measure_batch(int position, int tokens, bool has_audio, bool want_logits);
    uint64_t host_state_bytes() const;
    std::vector<float> prefill(const std::vector<int32_t> & ids, const std::vector<float> & audio_embeddings,
                               int batch_tokens);
    std::vector<float> step(int32_t id);
    int position() const;
    int context() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parakeet::moss::detail
