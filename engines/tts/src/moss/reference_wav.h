#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

struct WavShape {
    uint64_t frames;
    uint32_t sample_rate;
    uint32_t channels;
};

void validate_reference_shape(const WavShape & shape, int expected_rate);
void require_reference_total(size_t samples, int sample_rate);
std::vector<float> normalize_reference_pcm(const std::vector<float> & pcm);
std::vector<float> read_reference_wav(const std::string & path, int expected_rate);

} // namespace tts_cpp::moss::detail
