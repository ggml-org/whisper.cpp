// Short-input Sortformer batch diarization must remain on ggml when an
// unmasked fixed-capacity Core ML sidecar is present.

#include "parakeet/engine.h"

#include <exception>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef _WIN32
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kSampleRate = 16000;
constexpr int kShortAudioSeconds = 3;
constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
constexpr const char * kDiarizerDisableEnv = "PARAKEET_COREML_DIARIZER_DISABLE";

template <typename T>
void hash_value(uint64_t & hash, const T & value) {
    const auto * bytes = reinterpret_cast<const unsigned char *>(&value);
    for (size_t i = 0; i < sizeof(T); ++i) {
        hash ^= bytes[i];
        hash *= kFnvPrime;
    }
}

std::vector<float> create_short_input() {
    std::vector<float> samples(kSampleRate * kShortAudioSeconds);
    for (size_t i = 0; i < samples.size(); ++i) {
        const double t = static_cast<double>(i) / kSampleRate;
        const double frequency = t < 1.5 ? 220.0 : 330.0;
        samples[i] = static_cast<float>(0.25 * std::sin(2.0 * kPi * frequency * t));
    }
    return samples;
}

uint64_t fingerprint(const parakeet::DiarizationResult & result) {
    uint64_t hash = kFnvOffsetBasis;
    hash_value(hash, result.n_frames);
    hash_value(hash, result.num_spks);
    hash_value(hash, result.frame_stride_s);
    for (float probability : result.speaker_probs) hash_value(hash, probability);
    for (const auto & segment : result.segments) {
        hash_value(hash, segment.speaker_id);
        hash_value(hash, segment.start_s);
        hash_value(hash, segment.end_s);
    }
    return hash;
}

std::string diarize_in_child(const std::string & gguf, bool disable_coreml) {
    int fds[2];
    if (pipe(fds) != 0) return {};

    const pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        setenv(kDiarizerDisableEnv, "1", 1);
        if (disable_coreml) setenv("PARAKEET_COREML_DISABLE", "1", 1);

        std::string payload;
        try {
            parakeet::EngineOptions options;
            options.model_gguf_path = gguf;
            options.n_gpu_layers = 999;
            parakeet::Engine engine(options);
            const std::vector<float> samples = create_short_input();
            const parakeet::DiarizationResult result = engine.diarize_samples(
                samples.data(), static_cast<int>(samples.size()), kSampleRate);

            std::ostringstream out;
            out << "coreml_available=" << (engine.encoder_on_coreml() ? 1 : 0)
                << "\nfingerprint=" << std::hex << fingerprint(result)
                << "\nframes=" << std::dec << result.n_frames
                << "\nprobabilities=" << result.speaker_probs.size()
                << "\nsegments=" << result.segments.size();
            payload = out.str();
        } catch (const std::exception & error) {
            payload = std::string("error=") + error.what();
        }

        for (size_t offset = 0; offset < payload.size();) {
            const ssize_t written =
                write(fds[1], payload.data() + offset, payload.size() - offset);
            if (written <= 0) break;
            offset += static_cast<size_t>(written);
        }
        close(fds[1]);
        _exit(0);
    }

    close(fds[1]);
    std::string output;
    char buffer[1024];
    ssize_t count = 0;
    while ((count = read(fds[0], buffer, sizeof(buffer))) > 0) {
        output.append(buffer, static_cast<size_t>(count));
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return output;
}

bool parse_result(const std::string & raw, bool & coreml_available,
                  std::string & result) {
    constexpr const char * kPrefix = "coreml_available=";
    if (raw.rfind(kPrefix, 0) != 0) return false;
    const size_t newline = raw.find('\n');
    if (newline == std::string::npos) return false;
    coreml_available = raw.substr(std::strlen(kPrefix),
                                  newline - std::strlen(kPrefix)) == "1";
    result = raw.substr(newline + 1);
    return true;
}

}  // namespace
#endif

int main(int argc, char ** argv) {
#ifdef _WIN32
    (void) argc;
    (void) argv;
    std::fprintf(stderr, "[diarize-coreml-short-parity] SKIP: Core ML is Apple-only.\n");
    return 0;
#else
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <sortformer-v2.1.gguf>\n", argv[0]);
        return 2;
    }

    bool coreml_available = false;
    std::string normal_result;
    if (!parse_result(diarize_in_child(argv[1], false),
                      coreml_available, normal_result)) {
        std::fprintf(stderr, "[diarize-coreml-short-parity] FAIL: normal child failed\n");
        return 1;
    }
    if (!coreml_available) {
        std::fprintf(stderr, "[diarize-coreml-short-parity] SKIP: batch sidecar inactive\n");
        return 0;
    }

    bool disabled_available = true;
    std::string disabled_result;
    if (!parse_result(diarize_in_child(argv[1], true),
                      disabled_available, disabled_result) ||
        disabled_available) {
        std::fprintf(stderr, "[diarize-coreml-short-parity] FAIL: forced-ggml child failed\n");
        return 1;
    }
    if (normal_result != disabled_result) {
        std::fprintf(stderr,
            "[diarize-coreml-short-parity] FAIL: short-input results differ\n"
            "normal:\n%s\nforced ggml:\n%s\n",
            normal_result.c_str(), disabled_result.c_str());
        return 1;
    }

    std::fprintf(stderr,
        "[diarize-coreml-short-parity] PASS: %d-second input used ggml-equivalent routing\n",
        kShortAudioSeconds);
    return 0;
#endif
}
