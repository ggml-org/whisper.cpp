#include "parakeet/engine.h"
#include "test_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kSampleRate = 16000;
constexpr int kThreads = 8;
constexpr int kRuns = 5;
constexpr int kSpeakers = 8;
constexpr size_t kFeedSamples = kSampleRate / 10;
using Clock = std::chrono::steady_clock;

void validate(const parakeet::DiarizationResult & result) {
    if (result.num_spks != kSpeakers || result.n_frames <= 0 ||
        result.speaker_probs.size() != static_cast<size_t>(result.n_frames) * kSpeakers) {
        throw std::runtime_error("Invalid diarization output dimensions");
    }
    for (float value : result.speaker_probs) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            throw std::runtime_error("Invalid speaker probability");
        }
    }
}

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

parakeet::DiarizationResult infer(parakeet::Engine & engine, const std::vector<float> & audio) {
    auto result = engine.diarize_samples(audio.data(), static_cast<int>(audio.size()), kSampleRate, {});
    validate(result);
    return result;
}

double benchmark_offline(parakeet::Engine & engine, const std::vector<float> & audio) {
    std::vector<double> timings;
    for (int run = 0; run < kRuns; ++run) {
        const auto start = Clock::now();
        infer(engine, audio);
        timings.push_back(elapsed_ms(start));
    }
    std::sort(timings.begin(), timings.end());
    return timings[kRuns / 2];
}

double benchmark_stream(parakeet::Engine & engine, const std::vector<float> & audio) {
    auto session = engine.diarize_start({}, {});
    const auto start = Clock::now();
    for (size_t offset = 0; offset < audio.size(); offset += kFeedSamples) {
        session->feed_pcm_f32(audio.data() + offset,
            static_cast<int>(std::min(kFeedSamples, audio.size() - offset)));
    }
    session->finalize();
    return elapsed_ms(start);
}

void save_probabilities(const char * path, const std::vector<float> & probabilities) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(probabilities.data()),
        static_cast<std::streamsize>(probabilities.size() * sizeof(float)));
    if (!output) throw std::runtime_error("Cannot write probability output");
}

int run_benchmark(char ** argv) {
    std::vector<float> audio;
    int sample_rate = 0;
    if (!parakeet_test::load_wav_pcm16le_mono(argv[2], audio, sample_rate) || sample_rate != kSampleRate) {
        throw std::runtime_error("Expected a readable 16 kHz mono PCM16 WAV");
    }
    parakeet::EngineOptions options;
    options.model_gguf_path = argv[1];
    options.n_gpu_layers = std::stoi(argv[4]);
    options.n_threads = kThreads;
    options.prewarm = true;
    parakeet::Engine engine(options);
    const auto result = infer(engine, audio);
    save_probabilities(argv[3], result.speaker_probs);
    const double offline_ms = benchmark_offline(engine, audio);
    const double stream_ms = benchmark_stream(engine, audio);
    std::printf("{\"frames\":%d,\"speakers\":%d,\"offline_median_ms\":%.6f,\"stream_ms\":%.6f}\n",
        result.n_frames, result.num_spks, offline_ms, stream_ms);
    return 0;
}
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "Usage: %s MODEL WAV PROBABILITIES GPU_LAYERS\n", argv[0]);
        return 2;
    }
    try {
        return run_benchmark(argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
