#pragma once

#include "moss/backend_request.h"
#include "moss/codec_coreml.h"
#include "tts-cpp/fit.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

class Codec {
public:
    Codec(const std::string & path, const BackendRequest & backend, int n_threads, bool measure_only = false,
          AttentionLayout attention = AttentionLayout::Automatic);
    ~Codec();
    Codec(const Codec &) = delete;
    Codec & operator=(const Codec &) = delete;

    bool is_encoder() const;
    int sample_rate() const;
    int num_quantizers() const;
    int samples_per_frame() const;
    const char * backend_name() const;
    bool fused_attention() const;

    std::vector<int32_t> encode(const std::vector<float> & pcm);
    std::vector<float> decode(const std::vector<int32_t> & codes, int n_channels = 0);

    void begin_decode_stream(int n_channels = 0, int piece_frames = 0);
    std::vector<float> decode_stream(const std::vector<int32_t> & codes);
    int64_t frames_decoded() const;
    FitResult measure(int64_t length, int n_channels, bool streaming);

    CodecGeometry sidecar_geometry() const;
    void attach_sidecar(std::unique_ptr<CodecSidecarModel> sidecar, bool strict);
    bool on_coreml() const;
    void begin_run();
    std::string run_backend() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<float> decode_in_windows(Codec & codec, const std::vector<int32_t> & codes,
                                     int64_t window_frames, int n_channels = 0);
std::vector<float> decode_segments(Codec & codec, const std::vector<std::vector<int32_t>> & segments,
                                   int64_t max_single_frames, int n_channels = 0);
std::vector<float> decode_after_context(Codec & codec, const std::vector<std::vector<int32_t>> & segments,
                                        int64_t context_frames, int64_t max_single_frames,
                                        int n_channels = 0);

} // namespace tts_cpp::moss::detail
