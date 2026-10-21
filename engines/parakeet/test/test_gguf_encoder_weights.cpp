#include "parakeet_ctc.h"

#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>

namespace {

constexpr const char * kArchitectureKey = "general.architecture";
constexpr const char * kDiarizationArchitecture = "sortformer";
constexpr const char * kParakeetArchitecture = "parakeet-ctc";
constexpr const char * kDisableEnv = "PARAKEET_COREML_DISABLE";
constexpr int kLoadThrew = -1;

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

std::string write_gguf(const char * architecture) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
        (std::string("parakeet-encoder-weights-") + architecture + ".gguf");
    gguf_context * context = gguf_init_empty();
    gguf_set_val_str(context, kArchitectureKey, architecture);
    const bool written = gguf_write_to_file(context, path.string().c_str(), false);
    gguf_free(context);
    return written ? path.string() : std::string();
}

int load_status(const std::string & path) {
    parakeet::ParakeetCtcModel model;
    try {
        return parakeet::load_from_gguf(path, model, 1, 0, false);
    } catch (const std::exception &) {
        return kLoadThrew;
    }
}

bool expect(bool condition, const char * what) {
    std::printf("[%s] %s\n", condition ? "ok  " : "FAIL", what);
    return condition;
}

}

int main() {
    set_env(kDisableEnv, "1");
    const std::string diarization = write_gguf(kDiarizationArchitecture);
    const std::string parakeet = write_gguf(kParakeetArchitecture);
    if (diarization.empty() || parakeet.empty()) {
        std::fprintf(stderr, "cannot write the synthetic GGUFs\n");
        return 1;
    }
    bool passed = expect(load_status(diarization) != parakeet::kGgufLoadNoEncoderWeights,
        "a diarization-architecture GGUF is not rejected for missing Parakeet encoder weights");
    passed = expect(load_status(parakeet) == parakeet::kGgufLoadNoEncoderWeights,
        "a Parakeet GGUF without encoder weights is rejected without its sidecar") && passed;
    std::filesystem::remove(diarization);
    std::filesystem::remove(parakeet);
    return passed ? 0 : 1;
}
