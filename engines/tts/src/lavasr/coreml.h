#pragma once

#include "tts-cpp/lavasr/coreml.h"
#include "denoiser_core.h"
#include "enhancer_core.h"

#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

bool coreml_requested(const CoreMLOptions & options);

class CoreMLNetwork {
public:
    static std::unique_ptr<CoreMLNetwork> load(const CoreMLOptions & options, const EnhancerWeights & weights);
    static std::unique_ptr<CoreMLNetwork> load(const CoreMLOptions & options, const DenoiserWeights & weights);
    ~CoreMLNetwork();
    std::string backend_name() const;
    void enhance(const std::vector<float> & mel, int frames, std::vector<float> & real, std::vector<float> & imag);
    void denoise(const std::vector<float> & real, const std::vector<float> & imag, int frames,
                 std::vector<float> & real_out, std::vector<float> & imag_out);
    void denoise_batch(const std::vector<float> & real, const std::vector<float> & imag, int frames, int chunks,
                       std::vector<float> & real_out, std::vector<float> & imag_out);

private:
    CoreMLNetwork(const CoreMLOptions & options);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
