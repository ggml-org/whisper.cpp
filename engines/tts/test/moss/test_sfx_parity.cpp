#include "moss/sfx_model.h"
#include "moss/sfx_networks.h"
#include "moss/sfx_prompt.h"
#include "moss/sfx_sampler.h"
#include "moss/sfx_tokenizer.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::moss::detail;

namespace {

constexpr int SKIP = 77;
constexpr double STAGE_COSINE = 0.999;
constexpr double FOUR_BIT_DIT_COSINE = 0.998;
constexpr const char * TEXT_PREFIX = "text.";
constexpr const char * DIT_PREFIX = "dit.";
constexpr double LOOP_COSINE = 0.99;
constexpr int REFERENCE_STEPS = 8;
constexpr int REFERENCE_TENTHS = 50;
constexpr double SCHEDULE_COSINE = 0.999999;
constexpr int PARITY_THREADS = 8;
constexpr float REFERENCE_GUIDANCE = 4.0f;
constexpr float REFERENCE_SHIFT = 5.0f;
constexpr int DECODE_WINDOW = 256;
const char * REFERENCE_PROMPT = "A glass falls and shatters on a tile floor.";

int failures = 0;

void check(bool condition, const std::string & label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
        failures++;
    }
}

template <typename T>
std::vector<T> read_bin(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot read " + path.string());
    }
    const std::streamsize bytes = input.tellg();
    input.seekg(0);
    std::vector<T> data((size_t) bytes / sizeof(T));
    input.read(reinterpret_cast<char *>(data.data()), bytes);
    return data;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return -2.0;
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

double bytes_per_weight(ggml_type type) {
    return (double) ggml_type_size(type) / (double) ggml_blck_size(type);
}

ggml_type coarsest_type(const gguf_context * file, const std::string & prefix) {
    ggml_type coarsest = GGML_TYPE_F32;
    for (int64_t i = 0; i < gguf_get_n_tensors(file); ++i) {
        const ggml_type type = gguf_get_tensor_type(file, i);
        if (std::string(gguf_get_tensor_name(file, i)).rfind(prefix, 0) == 0 &&
                bytes_per_weight(type) < bytes_per_weight(coarsest)) {
            coarsest = type;
        }
    }
    return coarsest;
}

struct WeightTypes {
    ggml_type text;
    ggml_type dit;
};

WeightTypes read_weight_types(const char * model_path) {
    gguf_context * file = gguf_init_from_file(model_path, {true, nullptr});
    if (!file) {
        throw std::runtime_error(std::string("cannot read ") + model_path);
    }
    const WeightTypes types{coarsest_type(file, TEXT_PREFIX), coarsest_type(file, DIT_PREFIX)};
    gguf_free(file);
    return types;
}

double dit_threshold(ggml_type dit) {
    return dit == GGML_TYPE_Q4_0 ? FOUR_BIT_DIT_COSINE : STAGE_COSINE;
}

void report(const char * stage, double value, double threshold) {
    std::printf("%-22s cosine %.6f (threshold %.3f)\n", stage, value, threshold);
    check(value >= threshold, std::string(stage) + " diverges from the PyTorch reference");
}

void test_tokenizer(const SfxModel & model, const std::filesystem::path & dir) {
    const SfxTokenizer tokenizer(model);
    check(tokenizer.encode(clean_prompt(duration_prompt(REFERENCE_PROMPT, REFERENCE_TENTHS))) ==
            read_bin<int32_t>(dir / "ids.bin"),
            "tokenizer reproduces the Hugging Face ids");
}

void test_text_encoder(SfxModel & model, const std::filesystem::path & dir) {
    const std::vector<float> context = encode_text(model, read_bin<int32_t>(dir / "ids.bin"));
    report("text encoder", cosine(context, read_bin<float>(dir / "context_pos.bin")), STAGE_COSINE);
}

void test_dit_velocity(SfxModel & model, const std::filesystem::path & dir, double threshold) {
    const std::vector<float> noise = read_bin<float>(dir / "noise.bin");
    const std::vector<float> sigmas = read_bin<float>(dir / "sigmas.bin");
    const float timestep = sigmas[0] * (float) model.config().train_timesteps;
    SfxDitSession dit(model, model.config().latent_frames());
    const std::vector<float> positive = dit.velocity(noise, timestep, read_bin<float>(dir / "context_pos.bin"));
    const std::vector<float> negative = dit.velocity(noise, timestep, read_bin<float>(dir / "context_neg.bin"));
    report("dit positive", cosine(positive, read_bin<float>(dir / "v_pos0.bin")), threshold);
    report("dit negative", cosine(negative, read_bin<float>(dir / "v_neg0.bin")), threshold);
    const std::vector<float> again = dit.velocity(noise, timestep, read_bin<float>(dir / "context_pos.bin"));
    check(again == positive, "a reused DiT session is deterministic");
}

void test_sampling_loop(SfxModel & model, const std::filesystem::path & dir) {
    const std::vector<float> sigmas = flow_sigmas(REFERENCE_STEPS, REFERENCE_SHIFT);
    report("sigma schedule", cosine(sigmas, read_bin<float>(dir / "sigmas.bin")), SCHEDULE_COSINE);
    const std::vector<float> positive = read_bin<float>(dir / "context_pos.bin");
    const std::vector<float> negative = read_bin<float>(dir / "context_neg.bin");
    std::vector<float> latents = read_bin<float>(dir / "noise.bin");
    SfxDitSession dit(model, model.config().latent_frames());
    for (int step = 0; step < REFERENCE_STEPS; ++step) {
        const float timestep = sigmas[(size_t) step] * (float) model.config().train_timesteps;
        const std::vector<float> velocity = guided_velocity(dit.velocity(latents, timestep, positive),
                dit.velocity(latents, timestep, negative), REFERENCE_GUIDANCE);
        euler_step(latents, velocity, sigmas[(size_t) step],
                step + 1 < REFERENCE_STEPS ? sigmas[(size_t) step + 1] : 0.0f);
    }
    report("sampling loop", cosine(latents, read_bin<float>(dir / "latents.bin")), LOOP_COSINE);
}

void test_vae_decode(SfxModel & model, const std::filesystem::path & dir) {
    const SfxConfig & config = model.config();
    const std::vector<float> reference = read_bin<float>(dir / "audio.bin");
    const int hop = config.vae.hop_length();
    const int keep = (int) ((reference.size() + hop - 1) / hop);
    std::vector<float> pcm = decode_latents(model, read_bin<float>(dir / "latents.bin"),
            config.latent_frames(), keep, DECODE_WINDOW, {});
    pcm.resize(reference.size());
    report("vae decode", cosine(pcm, reference), STAGE_COSINE);
}

} // namespace

int main() {
    const char * model_path = std::getenv("MOSS_SFX_MODEL");
    const char * reference_dir = std::getenv("MOSS_SFX_REFERENCE_DIR");
    if (!model_path || !reference_dir) {
        std::printf("SKIP: set MOSS_SFX_MODEL and MOSS_SFX_REFERENCE_DIR\n");
        return SKIP;
    }
    try {
        const char * backend = std::getenv("MOSS_SFX_BACKEND");
        const bool use_gpu = std::getenv("MOSS_SFX_GPU") != nullptr;
        const WeightTypes types = read_weight_types(model_path);
        std::printf("weights: text %s, DiT %s\n", ggml_type_name(types.text), ggml_type_name(types.dit));
        SfxModel model(model_path, {backend ? backend : "", use_gpu}, PARITY_THREADS);
        std::printf("backend: %s\n", model.backend_name());
        const std::filesystem::path dir(reference_dir);
        test_tokenizer(model, dir);
        test_text_encoder(model, dir);
        test_vae_decode(model, dir);
        test_dit_velocity(model, dir, dit_threshold(types.dit));
        test_sampling_loop(model, dir);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (failures == 0) {
        std::printf("moss sfx parity: OK\n");
        return 0;
    }
    std::fprintf(stderr, "moss sfx parity: %d failures\n", failures);
    return 1;
}
