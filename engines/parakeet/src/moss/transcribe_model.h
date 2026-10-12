#pragma once

#include "moss/transcribe_runtime.h"
#include "parakeet/fit.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend;
struct ggml_context;
struct ggml_tensor;

namespace parakeet::moss::detail {

struct TranscribeAudioConfig {
    int sample_rate = 0;
    int n_fft = 0;
    int hop_length = 0;
    int n_mels = 0;
    int chunk_samples = 0;
    int chunk_frames = 0;
};

struct TranscribeEncoderConfig {
    int n_layers = 0;
    int n_embd = 0;
    int n_ff = 0;
    int n_heads = 0;
    int n_ctx = 0;
    float eps = 1e-5f;
};

struct TranscribeTextConfig {
    int n_layers = 0;
    int n_embd = 0;
    int n_ff = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int n_ctx_train = 0;
    int vocab = 0;
    float rope_base = 1000000.0f;
    float rms_eps = 1e-6f;
};

struct TranscribeTokens {
    int32_t audio_start = 0;
    int32_t audio_end = 0;
    int32_t audio_pad = 0;
    int32_t im_start = 0;
    int32_t im_end = 0;
    int32_t pad = 0;
};

struct TranscribeConfig {
    TranscribeAudioConfig audio;
    TranscribeEncoderConfig encoder;
    TranscribeTextConfig text;
    TranscribeTokens tokens;
    int merge_size = 0;
    float adaptor_eps = 1e-6f;
    float audio_tokens_per_second = 0.0f;
    int time_marker_every_seconds = 0;
    bool time_markers = true;
    std::string system_prompt;
    std::string default_prompt;
    std::string hotword_prefix;
    std::string hotword_separator;
    std::vector<int32_t> default_prompt_ids;
    int default_max_new_tokens = 0;
    int samples_per_token() const;
};

class TranscribeModel {
public:
    TranscribeModel(const std::string & path, bool use_gpu, int n_threads,
                    const std::string & backend = "auto", bool measure_only = false);
    ~TranscribeModel();
    TranscribeModel(const TranscribeModel &) = delete;
    TranscribeModel & operator=(const TranscribeModel &) = delete;

    const TranscribeConfig & config() const;
    const char * backend_name() const;
    ggml_backend * backend() const;
    ggml_tensor * tensor(const std::string & name) const;
    std::vector<std::string> tokenizer_tokens() const;
    std::vector<std::string> tokenizer_merges() const;
    std::vector<int32_t> tokenizer_types() const;

    FitResult measure_weights();
    uint64_t weight_bytes() const;
    TranscribeMemory allocated_memory() const;
    TranscribeMemory measure(TranscribeGraph & graph);
    bool measure_only() const;
    void allocate(TranscribeGraph & graph);
    void compute(TranscribeGraph & graph);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parakeet::moss::detail
