#define TTS_CPP_BUILD
#include "tts-cpp/supertonic/engine.h"

#include "supertonic_internal.h"
#include "npy.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace tts_cpp::supertonic {

using namespace detail;

namespace {

std::string supertonic_setup_hint(const std::string & path) {
    return "Supertonic GGUF not found: " + path + "\n"
           "Create the local model first, for example:\n"
           "  bash scripts/setup-supertonic2.sh\n"
           "or for the English-only bundle:\n"
           "  bash scripts/setup-supertonic2.sh --arch supertonic\n"
           "Model GGUFs live under models/ and are intentionally ignored by git.";
}

std::vector<float> read_tensor_f32(ggml_tensor * t) {
    std::vector<float> out((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

// NumPy RandomState-compatible MT19937 + standard_normal().  This matches the
// legacy np.random.seed(seed); np.random.randn(...) sequence used by the ONNX
// reference dumper.  std::normal_distribution is intentionally not used here:
// its transform is implementation-defined and produced audibly different
// Supertonic samples for the same seed.
class numpy_random_state {
public:
    explicit numpy_random_state(uint32_t seed) {
        mt_[0] = seed;
        for (int i = 1; i < N; ++i) {
            mt_[i] = 1812433253U * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + (uint32_t)i;
        }
        index_ = N;
    }

    float standard_normal() {
        if (has_gauss_) {
            has_gauss_ = false;
            return (float) gauss_;
        }
        double x1 = 0.0, x2 = 0.0, r2 = 0.0;
        do {
            x1 = 2.0 * uniform_double() - 1.0;
            x2 = 2.0 * uniform_double() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        const double f = std::sqrt(-2.0 * std::log(r2) / r2);
        gauss_ = x1 * f;
        has_gauss_ = true;
        return (float)(x2 * f);
    }

private:
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfU;
    static constexpr uint32_t UPPER_MASK = 0x80000000U;
    static constexpr uint32_t LOWER_MASK = 0x7fffffffU;

    uint32_t mt_[N]{};
    int index_ = N + 1;
    bool has_gauss_ = false;
    double gauss_ = 0.0;

    uint32_t uint32() {
        static const uint32_t mag01[2] = {0x0U, MATRIX_A};
        if (index_ >= N) {
            int kk = 0;
            for (; kk < N - M; ++kk) {
                uint32_t y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
                mt_[kk] = mt_[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            for (; kk < N - 1; ++kk) {
                uint32_t y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
                mt_[kk] = mt_[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            uint32_t y = (mt_[N - 1] & UPPER_MASK) | (mt_[0] & LOWER_MASK);
            mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
            index_ = 0;
        }
        uint32_t y = mt_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }

    double uniform_double() {
        const uint32_t a = uint32() >> 5;
        const uint32_t b = uint32() >> 6;
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }
};

} // namespace

struct Engine::Impl {
    EngineOptions    opts;
    supertonic_model model;
    std::atomic<bool> cancel_flag{false};

    explicit Impl(const EngineOptions & o)
        : opts(o) {
        if (opts.model_gguf_path.empty()) {
            throw std::runtime_error("Supertonic Engine: model_gguf_path is required");
        }
        if (!std::filesystem::exists(opts.model_gguf_path)) {
            throw std::runtime_error(supertonic_setup_hint(opts.model_gguf_path));
        }
        if (!load_supertonic_gguf(opts.model_gguf_path, model, opts.n_gpu_layers, false)) {
            throw std::runtime_error("Supertonic Engine: failed to load GGUF: " +
                                     opts.model_gguf_path);
        }
        try {
            supertonic_set_n_threads(model, opts.n_threads);

            // Validate voice up front so we throw at construction
            // rather than mid-synthesize().
            const std::string voice = opts.voice.empty()
                ? model.hparams.default_voice
                : opts.voice;
            if (model.voices.find(voice) == model.voices.end()) {
                throw std::runtime_error("Supertonic Engine: unknown voice: " + voice);
            }
        } catch (...) {
            free_supertonic_model(model);
            throw;
        }
    }

    ~Impl() {
        free_supertonic_model(model);
    }

    Impl(const Impl &)             = delete;
    Impl & operator=(const Impl &) = delete;

    SynthesisResult synthesize(const std::string & text) {
        if (text.empty()) {
            throw std::runtime_error("Supertonic Engine: text is empty");
        }

        const std::string voice = opts.voice.empty()
            ? model.hparams.default_voice
            : opts.voice;
        const int   steps = opts.steps > 0 ? opts.steps : model.hparams.default_steps;
        const float speed = opts.speed > 0.0f ? opts.speed : model.hparams.default_speed;
        if (steps <= 0) throw std::runtime_error("Supertonic Engine: steps must be positive");
        if (speed <= 0.0f) throw std::runtime_error("Supertonic Engine: speed must be positive");

        auto vit = model.voices.find(voice);
        if (vit == model.voices.end()) {
            // Re-validated here in case opts.voice was hot-swapped after
            // construction (not currently supported but guard anyway).
            throw std::runtime_error("Supertonic Engine: unknown voice: " + voice);
        }
        std::vector<float> style_ttl = read_tensor_f32(vit->second.ttl);
        std::vector<float> style_dp  = read_tensor_f32(vit->second.dp);

        std::vector<int32_t> text_ids_i32;
        std::string normalized;
        std::string error;
        if (!supertonic_text_to_ids(model, text, opts.language, text_ids_i32, &normalized, &error)) {
            throw std::runtime_error("Supertonic Engine: text preprocessing failed: " + error);
        }
        std::vector<int64_t> text_ids(text_ids_i32.begin(), text_ids_i32.end());

        if (cancel_flag.load(std::memory_order_acquire)) {
            throw std::runtime_error("Supertonic Engine: cancelled before duration");
        }

        float duration_raw = 0.0f;
        if (!supertonic_duration_forward_ggml(model, text_ids.data(), (int) text_ids.size(),
                                              style_dp.data(), duration_raw, &error)) {
            throw std::runtime_error("Supertonic Engine: duration failed: " + error);
        }
        const float duration_s  = duration_raw / speed;
        const int   sample_rate = model.hparams.sample_rate;
        const int   chunk = model.hparams.base_chunk_size *
                            model.hparams.ttl_chunk_compress_factor;
        int wav_len = (int) (duration_s * sample_rate);
        int latent_len = std::max(1, (wav_len + chunk - 1) / chunk);

        std::vector<float> latent;
        if (!opts.noise_npy_path.empty()) {
            npy_array noise = npy_load(opts.noise_npy_path);
            if (noise.dtype != "<f4" || noise.shape.size() != 3 || noise.shape[0] != 1 ||
                noise.shape[1] != model.hparams.latent_channels) {
                throw std::runtime_error("Supertonic Engine: noise npy must be float32 [1, latent_channels, L]");
            }
            latent_len = (int) noise.shape[2];
            wav_len = latent_len * chunk;
            latent.resize(noise.n_elements());
            std::memcpy(latent.data(), npy_as_f32(noise), latent.size() * sizeof(float));
        } else {
            numpy_random_state rng((uint32_t) opts.seed);
            latent.assign((size_t) model.hparams.latent_channels * latent_len, 0.0f);
            for (float & v : latent) v = rng.standard_normal();
        }

        if (cancel_flag.load(std::memory_order_acquire)) {
            throw std::runtime_error("Supertonic Engine: cancelled before text encoder");
        }

        std::vector<float> text_emb;
        if (!supertonic_text_encoder_forward_ggml(model, text_ids.data(), (int) text_ids.size(),
                                                  style_ttl.data(), text_emb, &error)) {
            throw std::runtime_error("Supertonic Engine: text encoder failed: " + error);
        }

        std::vector<float> latent_mask((size_t) latent_len, 1.0f);

        std::vector<float> next;
        for (int step = 0; step < steps; ++step) {
            if (cancel_flag.load(std::memory_order_acquire)) {
                throw std::runtime_error("Supertonic Engine: cancelled at vector step "
                                         + std::to_string(step));
            }
            if (!supertonic_vector_step_ggml(model, latent.data(), latent_len,
                                             text_emb.data(), (int) text_ids.size(),
                                             style_ttl.data(), latent_mask.data(),
                                             step, steps, next, &error)) {
                throw std::runtime_error("Supertonic Engine: vector estimator failed: " + error);
            }
            latent.swap(next);
        }

        if (cancel_flag.load(std::memory_order_acquire)) {
            throw std::runtime_error("Supertonic Engine: cancelled before vocoder");
        }

        std::vector<float> wav_full;
        if (!supertonic_vocoder_forward_ggml(model, latent.data(), latent_len, wav_full, &error)) {
            throw std::runtime_error("Supertonic Engine: vocoder failed: " + error);
        }

        SynthesisResult result;
        result.sample_rate = sample_rate;
        result.duration_s  = duration_s;
        result.pcm.assign(wav_full.begin(),
                          wav_full.begin() + std::min((size_t) wav_len, wav_full.size()));
        return result;
    }

    std::string backend_name() const {
        if (!model.backend) return "(unknown)";
        if (const char * name = ggml_backend_name(model.backend)) {
            return std::string(name);
        }
        return "(unknown)";
    }
};

Engine::Engine(const EngineOptions & opts)
    : pimpl_(std::make_unique<Impl>(opts)) {}

Engine::~Engine() = default;

Engine::Engine(Engine &&) noexcept            = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

SynthesisResult Engine::synthesize(const std::string & text) {
    return pimpl_->synthesize(text);
}

void Engine::cancel() {
    pimpl_->cancel_flag.store(true, std::memory_order_release);
}

const EngineOptions & Engine::options() const {
    return pimpl_->opts;
}

std::string Engine::backend_name() const {
    return pimpl_->backend_name();
}

BackendDevice Engine::backend_device() const {
    ggml_backend_t b = pimpl_ ? pimpl_->model.backend : nullptr;
    if (!b) return BackendDevice::CPU;
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    if (!dev) return BackendDevice::CPU;
    return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU
               ? BackendDevice::CPU
               : BackendDevice::GPU;
}

// Convenience one-shot wrapper.  Pays the full GGUF load + free per
// call; use Engine directly for repeated synthesis.
SynthesisResult synthesize(const EngineOptions & opts, const std::string & text) {
    Engine engine(opts);
    return engine.synthesize(text);
}

} // namespace tts_cpp::supertonic
