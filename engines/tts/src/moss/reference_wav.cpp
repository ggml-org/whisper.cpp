#include "moss/reference_wav.h"

#include "dr_wav.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

constexpr uint64_t MAX_REFERENCE_WAV_BYTES = 64ull * 1024 * 1024;
constexpr uint64_t MAX_REFERENCE_SECONDS   = 60;
constexpr uint32_t MAX_REFERENCE_CHANNELS  = 64;
constexpr uint64_t MAX_REFERENCE_SAMPLES   = 60ull * 192000ull;
constexpr size_t REFERENCE_READ_BLOCK_FRAMES = 4096;
constexpr double REFERENCE_TARGET_DBFS = -20.0;
constexpr double REFERENCE_GAIN_LIMIT_DB = 3.0;
constexpr double REFERENCE_POWER_FLOOR = 1e-9;
constexpr double POWER_DECIBEL_FACTOR = 10.0;
constexpr double AMPLITUDE_DECIBEL_FACTOR = 20.0;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss engine: " + message);
}

void require_reference_file_size(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const std::streamoff bytes = input ? (std::streamoff) input.tellg() : -1;
    if (bytes <= 0) {
        fail("cannot read reference WAV: " + path);
    }
    if ((uint64_t) bytes > MAX_REFERENCE_WAV_BYTES) {
        fail("reference WAV must be at most 64 MiB on disk");
    }
}

void downmix_frames(const std::vector<float> & interleaved, size_t count, unsigned int channels,
                    std::vector<float> & mono, size_t start) {
    for (size_t frame = 0; frame < count; ++frame) {
        float sum = 0.0f;
        for (unsigned int channel = 0; channel < channels; ++channel) {
            sum += interleaved[frame * channels + channel];
        }
        mono[start + frame] = sum / (float) channels;
    }
}

std::vector<float> read_mono_frames(drwav & wav) {
    std::vector<float> mono((size_t) wav.totalPCMFrameCount, 0.0f);
    std::vector<float> interleaved(REFERENCE_READ_BLOCK_FRAMES * wav.channels);
    for (size_t start = 0; start < mono.size();) {
        const size_t count = std::min(REFERENCE_READ_BLOCK_FRAMES, mono.size() - start);
        if (drwav_read_pcm_frames_f32(&wav, count, interleaved.data()) != count) {
            fail("reference WAV is truncated");
        }
        downmix_frames(interleaved, count, wav.channels, mono, start);
        start += count;
    }
    return mono;
}

} // namespace

void validate_reference_shape(const WavShape & shape, int expected_rate) {
    if ((int) shape.sample_rate != expected_rate) {
        fail("reference WAV must be sampled at " + std::to_string(expected_rate) +
             " Hz; there is no resampling");
    }
    if (shape.channels == 0 || shape.channels > MAX_REFERENCE_CHANNELS) {
        fail("reference WAV channel count is unsupported");
    }
    if (shape.frames == 0 || shape.frames > (uint64_t) shape.sample_rate * MAX_REFERENCE_SECONDS) {
        fail("reference WAV must hold between one sample and " +
             std::to_string(MAX_REFERENCE_SECONDS) + " seconds of audio");
    }
    if (shape.frames > MAX_REFERENCE_SAMPLES) {
        fail("reference WAV declares more samples than the engine decodes");
    }
}

void require_reference_total(size_t samples, int sample_rate) {
    if ((uint64_t) samples > (uint64_t) sample_rate * MAX_REFERENCE_SECONDS ||
        (uint64_t) samples > MAX_REFERENCE_SAMPLES) {
        fail("dialogue references must total at most " + std::to_string(MAX_REFERENCE_SECONDS) +
             " seconds of audio");
    }
}

std::vector<float> normalize_reference_pcm(const std::vector<float> & pcm) {
    if (pcm.empty()) {
        return pcm;
    }
    const double power = std::inner_product(pcm.begin(), pcm.end(), pcm.begin(), 0.0) / pcm.size();
    const double dbfs = POWER_DECIBEL_FACTOR * std::log10(power + REFERENCE_POWER_FLOOR);
    const double gain = std::clamp(REFERENCE_TARGET_DBFS - dbfs,
            -REFERENCE_GAIN_LIMIT_DB, REFERENCE_GAIN_LIMIT_DB);
    const float factor = (float) std::pow(POWER_DECIBEL_FACTOR, gain / AMPLITUDE_DECIBEL_FACTOR);
    std::vector<float> normalized(pcm.size());
    std::transform(pcm.begin(), pcm.end(), normalized.begin(),
            [factor](float sample) { return sample * factor; });
    return normalized;
}

std::vector<float> read_reference_wav(const std::string & path, int expected_rate) {
    require_reference_file_size(path);
    drwav wav{};
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        fail("cannot read reference WAV: " + path);
    }
    try {
        validate_reference_shape({wav.totalPCMFrameCount, wav.sampleRate, wav.channels}, expected_rate);
        std::vector<float> mono = read_mono_frames(wav);
        drwav_uninit(&wav);
        return mono;
    } catch (...) {
        drwav_uninit(&wav);
        throw;
    }
}

} // namespace tts_cpp::moss::detail
