#pragma once

#include "moss/backend_request.h"
#include "moss/coreml_sidecar.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

struct SpeechVqConfig {
    int sample_rate = 0;
    int n_fft = 0;
    int hop_length = 0;
    int n_mels = 0;
    int chunk_samples = 0;
    int n_layers = 0;
    int n_embd = 0;
    int n_ff = 0;
    int n_heads = 0;
    int n_ctx = 0;
    int pooling_kernel = 0;
    int codebook_size = 0;
    int samples_per_token() const;
};

size_t speech_segment_tokens(const SpeechVqConfig & config, size_t samples);
size_t speech_padded_samples(const SpeechVqConfig & config, size_t samples);
void normalize_whisper_log_mel(std::vector<float> & mel);

struct SpeechSegmentCapacity {
    int mel_frames = 0;
    int tokens = 0;
};

SpeechSegmentCapacity speech_segment_capacity(const SpeechVqConfig & config);

class SpeechTokenizerSidecar {
public:
    virtual ~SpeechTokenizerSidecar() = default;
    virtual bool pooled_states(const std::vector<float> & mel, std::vector<float> & states) = 0;
    virtual const char * label() const = 0;
};

std::unique_ptr<SpeechTokenizerSidecar> open_speech_tokenizer_sidecar(const std::string & codec_path,
                                                                      const SpeechVqConfig & config,
                                                                      const CoremlPolicy & policy);

class SpeechTokenizer {
public:
    SpeechTokenizer(const std::string & codec_path, const BackendRequest & backend, int n_threads,
                    AttentionLayout attention = AttentionLayout::Automatic);
    ~SpeechTokenizer();
    SpeechTokenizer(const SpeechTokenizer &) = delete;
    SpeechTokenizer & operator=(const SpeechTokenizer &) = delete;

    const SpeechVqConfig & config() const;
    const char * backend_name() const;
    bool fused_attention() const;
    std::vector<float> log_mel(const float * samples, size_t count) const;
    std::vector<int32_t> encode_segment(const float * samples, size_t count);
    std::vector<int32_t> encode(const std::vector<float> & pcm_16k, const std::function<bool()> & stop = {});

    void attach_sidecar(std::unique_ptr<SpeechTokenizerSidecar> sidecar, bool strict);
    bool on_coreml() const;
    void begin_run();
    std::string run_backend() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss::detail
