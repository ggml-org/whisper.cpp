// Fixture test for the loader's Core ML-owned encoder mode: a GGUF without
// encoder weights (the TranscribePsy Core ML distribution) and the
// PARAKEET_COREML_ONLY switch for a full GGUF.
//
// Usage:
//   test-coreml-encoder-owner --coreml-gguf <encoder-less.gguf>
//                             [--full-gguf <q8.gguf> --wav <input.wav>]
//
// On every platform it asserts that the encoder-less GGUF is rejected with
// the documented code when no sidecar can take the encoder over (Core ML
// disabled through PARAKEET_COREML_DISABLE, which also covers builds without
// Core ML) and that the memory-fit loader rejects it the same way.
//
// On a Core ML build with the encoder sidecar next to the GGUF it further
// asserts that the GGUF loads with Core ML owning the encoder, that the
// encoder runs, that a request for the ggml encoder is refused instead of
// reading unallocated tensors, and that PARAKEET_COREML_ONLY=1 with a full
// GGUF skips the encoder weights (a smaller weights buffer) while the same
// GGUF without the switch loads them. Without the sidecar that part exits 3
// (skipped).
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "mel_preprocess.h"
#include "parakeet_coreml_path.h"
#include "parakeet_ctc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace parakeet;

namespace {

constexpr int kSkipExitCode = 3;
constexpr const char * kDisableEnv = "PARAKEET_COREML_DISABLE";
constexpr const char * kOnlyEnv    = "PARAKEET_COREML_ONLY";

int g_failures = 0;

void expect(bool cond, const std::string & what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

int load(const std::string & gguf, ParakeetCtcModel & model) {
    return load_from_gguf(gguf, model, /*n_threads=*/0, /*n_gpu_layers=*/0, /*verbose=*/false);
}

void check_rejected_without_sidecar(const std::string & coreml_gguf) {
    set_env(kDisableEnv, "1");
    ParakeetCtcModel model;
    expect(load(coreml_gguf, model) == kGgufLoadNoEncoderWeights,
           "an encoder-less GGUF must be rejected with kGgufLoadNoEncoderWeights without a sidecar");

    ParakeetCtcModel measured;
    GgufLoadMeasure measure;
    expect(load_from_gguf_metadata_only(coreml_gguf, measured, 0, 0, false, measure) ==
               kGgufLoadNoEncoderWeights,
           "the memory-fit loader must reject an encoder-less GGUF");
    unset_env(kDisableEnv);
}

#ifdef PARAKEET_USE_COREML

bool sidecar_present(const std::string & gguf) {
    return std::filesystem::is_directory(coreml_encoder_sidecar_path(gguf));
}

std::vector<float> mel_of(const ParakeetCtcModel & model, const std::string & wav, int & n_frames) {
    std::vector<float> samples;
    int sample_rate = 0;
    expect(load_wav_mono_f32(wav, samples, sample_rate) == 0, "could not load " + wav);
    std::vector<float> mel;
    expect(compute_log_mel(samples.data(), (int) samples.size(), model.mel_cfg, mel, n_frames) == 0,
           "mel failed");
    return mel;
}

void check_owned_encoder_runs(const std::string & coreml_gguf, const std::string & wav) {
    ParakeetCtcModel model;
    expect(load(coreml_gguf, model) == 0, "the encoder-less GGUF must load next to its sidecar");
    if (!model.impl) return;
    expect(model_encoder_owned_by_coreml(model), "Core ML must own the encoder");
    expect(model_encoder_on_coreml(model), "the encoder sidecar must be loaded");

    int n_frames = 0;
    const std::vector<float> mel = mel_of(model, wav, n_frames);

    EncoderOutputs out;
    expect(run_encoder(model, mel.data(), n_frames, model.mel_cfg.n_mels, out,
                       /*max_layers=*/-1, /*capture_intermediates=*/false,
                       /*allow_coreml_padded=*/true) == 0 && out.n_enc_frames > 0 && out.used_coreml,
           "the owned encoder must run on Core ML");

    // capture_intermediates routes every model to the ggml encoder, whose
    // weights were never allocated: the request must fail, not read them.
    EncoderOutputs refused;
    expect(run_encoder(model, mel.data(), n_frames, model.mel_cfg.n_mels, refused,
                       /*max_layers=*/-1, /*capture_intermediates=*/true) != 0,
           "a ggml encoder request must be refused when Core ML owns the encoder");
}

// A directory where the full GGUF carries the Core ML distribution's names, so
// its sidecar is found next to it.
std::filesystem::path stage_full_gguf_with_sidecar(const std::string & full_gguf,
                                                   const std::string & coreml_gguf) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "parakeet-coreml-owner-test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::filesystem::path staged = dir / "staged.gguf";
    std::filesystem::create_symlink(std::filesystem::absolute(full_gguf), staged);
    std::filesystem::create_directory_symlink(
        std::filesystem::absolute(coreml_encoder_sidecar_path(coreml_gguf)),
        coreml_encoder_sidecar_path(staged.string()));
    return staged;
}

void check_coreml_only_skips_encoder_weights(const std::string & full_gguf,
                                             const std::string & coreml_gguf) {
    const std::filesystem::path staged = stage_full_gguf_with_sidecar(full_gguf, coreml_gguf);

    ParakeetCtcModel full;
    expect(load(staged.string(), full) == 0, "the full GGUF must load without the switch");
    expect(full.impl && !model_encoder_owned_by_coreml(full),
           "without PARAKEET_COREML_ONLY the ggml encoder weights are loaded");

    set_env(kOnlyEnv, "1");
    ParakeetCtcModel owned;
    expect(load(staged.string(), owned) == 0, "the full GGUF must load under PARAKEET_COREML_ONLY");
    unset_env(kOnlyEnv);
    expect(owned.impl && model_encoder_owned_by_coreml(owned),
           "PARAKEET_COREML_ONLY must hand the encoder to Core ML");
    if (full.impl && owned.impl) {
        expect(model_weights_buffer_bytes(owned) < model_weights_buffer_bytes(full),
               "skipping the encoder weights must shrink the weights buffer");
    }
    std::filesystem::remove_all(staged.parent_path());
}

#endif  // PARAKEET_USE_COREML

}  // namespace

int main(int argc, char ** argv) {
    std::string coreml_gguf, full_gguf, wav;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--coreml-gguf") == 0) coreml_gguf = argv[i + 1];
        else if (std::strcmp(argv[i], "--full-gguf") == 0) full_gguf = argv[i + 1];
        else if (std::strcmp(argv[i], "--wav") == 0) wav = argv[i + 1];
    }
    if (coreml_gguf.empty()) {
        std::fprintf(stderr, "usage: test-coreml-encoder-owner --coreml-gguf <gguf> "
                             "[--full-gguf <gguf> --wav <wav>]\n");
        return 2;
    }

    check_rejected_without_sidecar(coreml_gguf);

    const bool want_owner_checks = !full_gguf.empty() && !wav.empty();
#ifdef PARAKEET_USE_COREML
    if (want_owner_checks) {
        if (!sidecar_present(coreml_gguf)) {
            std::printf("SKIP: no Core ML encoder sidecar next to %s\n", coreml_gguf.c_str());
            return g_failures == 0 ? kSkipExitCode : 1;
        }
        check_owned_encoder_runs(coreml_gguf, wav);
        check_coreml_only_skips_encoder_weights(full_gguf, coreml_gguf);
    }
#else
    if (want_owner_checks) {
        std::printf("note: Core ML is not compiled in; only the rejection checks ran\n");
    }
#endif

    if (g_failures == 0) {
        std::printf("test-coreml-encoder-owner: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test-coreml-encoder-owner: %d failure(s)\n", g_failures);
    return 1;
}
