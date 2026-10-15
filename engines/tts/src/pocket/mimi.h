#pragma once
#include "pocket/memory.h"
#include <memory>
#include <string>
#include <vector>

struct ggml_backend_device;

namespace tts_cpp::pocket::detail {
class Mimi {
public:
    explicit Mimi(const std::string & path, int threads = 2, ggml_backend_device * device = nullptr);
    ~Mimi();
    Mimi(const Mimi &) = delete;
    Mimi & operator=(const Mimi &) = delete;
    static MemoryMeasure measure(const std::string & path, int threads, bool include_encoder,
                                 ggml_backend_device * device = nullptr);
    // Independent decoder/encoder histories. Decoder accepts [T,latent_dim]
    // and emits mono PCM, retaining causal state between chunks.
    void reset_decoder();
    std::vector<float> decode(const std::vector<float> & latents);
    // A whole mono reference at the native sample rate; zero-padded to the
    // next frame. Each call starts an independent encoding session.
    std::vector<float> encode(const std::vector<float> & pcm);
    // Version 1 codec geometry, shared by decoding and metadata-only fit.
    static constexpr int sample_rate() { return 24000; }
    static constexpr int frame_samples() { return 1920; }
    static constexpr int latent_dim() { return 32; }
    static constexpr int max_decode_frames() { return 16; }
    const std::string & source_hash() const;
    const char * backend_name() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tts_cpp::pocket::detail
