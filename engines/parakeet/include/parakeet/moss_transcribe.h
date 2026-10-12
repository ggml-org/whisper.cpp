#pragma once
#include "parakeet/export.h"
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace parakeet::moss {

struct TranscribeOptions {
    std::string model_path;
    int n_threads = 4;
    bool use_gpu = false;
    std::string backends_dir;
    // "auto" follows use_gpu. An explicit backend (e.g. "cuda", "vulkan" or
    // "CUDA0") overrides use_gpu and fails if unavailable.
    std::string backend = "auto";
};

struct TranscribeRequest {
    std::string prompt;
    std::vector<std::string> hotwords;
    int max_new_tokens = 0;
};

struct TranscriptSegment {
    double start_s = 0;
    double end_s = 0;
    std::string speaker;
    std::string text;
};

struct TranscribeResult {
    std::string text;
    std::vector<TranscriptSegment> segments;
    bool cancelled = false;
    int audio_tokens = 0;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double encode_ms = 0;
    double prefill_ms = 0;
    double decode_ms = 0;
    std::string encoder_backend;
};

using TranscribeProgress = std::function<bool(int generated_tokens, int max_new_tokens)>;

class PARAKEET_API TranscribeEngine {
public:
    explicit TranscribeEngine(const TranscribeOptions & options);
    ~TranscribeEngine();
    TranscribeEngine(const TranscribeEngine &) = delete;
    TranscribeEngine & operator=(const TranscribeEngine &) = delete;
    TranscribeResult transcribe(const float * pcm, size_t samples, int sample_rate,
                                const TranscribeRequest & request = {}, const TranscribeProgress & progress = {});
    void cancel() noexcept;
    int sample_rate() const noexcept;
    const char * backend_name() const noexcept;
    bool encoder_on_coreml() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace parakeet::moss
