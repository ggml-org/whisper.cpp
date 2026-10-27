#include "tts-cpp/moss/sound_effect.h"

#include "moss/sfx_coreml.h"
#include "moss/sfx_model.h"
#include "moss/sfx_networks.h"
#include "moss/sfx_request.h"
#include "moss/sfx_sampler.h"
#include "moss/sfx_tokenizer.h"

#include "backend_selection.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace tts_cpp::moss {
namespace {

using detail::SfxDitRunner;
using detail::SfxModel;
using detail::SfxTokenizer;
using detail::ResolvedRequest;
constexpr int TENTHS_PER_SECOND = 10;
constexpr float UNGUIDED = 1.0f;

struct PromptContexts {
    std::vector<float> positive;
    std::vector<float> negative;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss sfx: " + message);
}

double elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

} // namespace

struct SoundEffectEngine::Impl {
    SoundEffectOptions options;
    std::unique_ptr<SfxModel> model;
    std::unique_ptr<SfxTokenizer> tokenizer;
    std::unique_ptr<detail::SfxDitSidecar> dit_sidecar;
    std::unique_ptr<detail::SfxVaeSidecar> vae_sidecar;
    bool strict = false;
    detail::StageBackends dit_backends;
    detail::StageBackends vae_backends;
    std::atomic<bool> cancel_requested{false};
    std::mutex generation_mutex;

    void load() {
        if (options.model_path.empty()) {
            fail("model_path is required");
        }
        if (!options.backends_dir.empty()) {
            ::tts_cpp::detail::set_backends_directory(options.backends_dir);
        }
        model = std::make_unique<SfxModel>(options.model_path, detail::backend_request(options), options.n_threads);
        validate_networks();
        tokenizer = std::make_unique<SfxTokenizer>(*model);
        attach_sidecars();
    }

    void attach_sidecars() {
        const detail::CoremlPolicy policy = detail::read_coreml_policy();
        dit_sidecar = detail::open_sfx_dit_sidecar(options.model_path, model->config(), policy);
        vae_sidecar = detail::open_sfx_vae_sidecar(options.model_path, model->config(), policy);
        strict = policy.strict;
    }

    void validate_networks() const {
        detail::validate_text_encoder(*model);
        detail::validate_dit(*model);
        detail::validate_vae(*model);
    }

    std::vector<float> step_velocity(SfxDitRunner & dit, const std::vector<float> & latents, float timestep,
                                     const std::vector<float> & positive, const std::vector<float> & negative,
                                     float guidance) {
        std::vector<float> conditioned = dit.velocity(latents, timestep, positive);
        if (guidance == UNGUIDED) {
            return conditioned;
        }
        return detail::guided_velocity(conditioned, dit.velocity(latents, timestep, negative), guidance);
    }

    bool keep_going(const SoundEffectProgress & progress, int step, int total) const {
        if (cancel_requested) {
            return false;
        }
        return !progress || progress(step, total);
    }

    void denoise_step(SfxDitRunner & dit, std::vector<float> & latents, const ResolvedRequest & request,
                      const std::vector<float> & sigmas, int step, const std::vector<float> & positive,
                      const std::vector<float> & negative) {
        const float sigma = sigmas[(size_t) step];
        const float next = step + 1 < request.steps ? sigmas[(size_t) step + 1] : 0.0f;
        const float timestep = sigma * (float) model->config().train_timesteps;
        detail::euler_step(latents, step_velocity(dit, latents, timestep, positive, negative, request.guidance),
                sigma, next);
    }

    bool denoise(std::vector<float> & latents, const ResolvedRequest & request, const std::vector<float> & positive,
                 const std::vector<float> & negative, const SoundEffectProgress & progress) {
        const std::vector<float> sigmas = detail::flow_sigmas(request.steps, request.shift);
        SfxDitRunner dit(*model, model->config().latent_frames(), dit_sidecar, strict, dit_backends);
        if (request.guidance == UNGUIDED) dit.prepare_contexts({positive});
        else dit.prepare_contexts({positive, negative});
        return run_steps(dit, latents, request, sigmas, positive, negative, progress);
    }

    bool run_steps(SfxDitRunner & dit, std::vector<float> & latents, const ResolvedRequest & request,
                   const std::vector<float> & sigmas, const std::vector<float> & positive,
                   const std::vector<float> & negative, const SoundEffectProgress & progress) {
        for (int step = 0; step < request.steps; ++step) {
            denoise_step(dit, latents, request, sigmas, step, positive, negative);
            if (!keep_going(progress, step + 1, request.steps)) {
                return false;
            }
        }
        return true;
    }

    int64_t output_samples(int tenths) const {
        return (int64_t) model->config().vae.sample_rate * tenths / TENTHS_PER_SECOND;
    }

    int kept_frames(int64_t samples) const {
        const detail::SfxConfig & config = model->config();
        const int hop = config.vae.hop_length();
        return (int) std::min<int64_t>(config.latent_frames(), (samples + hop - 1) / hop);
    }

    std::vector<float> decode(const std::vector<float> & latents, int tenths) {
        const int64_t samples = output_samples(tenths);
        std::vector<float> pcm = detail::decode_sfx_latents(*model, vae_sidecar, strict, vae_backends, latents,
                model->config().latent_frames(), kept_frames(samples), detail::SFX_DECODE_WINDOW_FRAMES,
                [this](int, int) { return !cancel_requested.load(); });
        pcm.resize(std::min(pcm.size(), (size_t) samples));
        return pcm;
    }

    PromptContexts encode_prompts(const ResolvedRequest & request) {
        PromptContexts contexts;
        contexts.positive = detail::encode_text(*model, tokenizer->encode(request.conditioning));
        contexts.negative = detail::encode_text(*model, tokenizer->encode(request.negative_prompt));
        return contexts;
    }

    std::vector<float> initial_noise(uint32_t seed) const {
        const detail::SfxConfig & config = model->config();
        return detail::gaussian_noise((size_t) config.latent_frames() * config.dit.in_channels, seed);
    }

    void finish_decode(SoundEffectResult & result, const std::vector<float> & latents, int tenths) {
        const auto decode_start = std::chrono::steady_clock::now();
        result.pcm = decode(latents, tenths);
        result.decode_ms = elapsed_ms(decode_start);
        result.cancelled = cancel_requested;
        if (result.cancelled) {
            result.pcm.clear();
        }
    }

    SoundEffectResult run(const ResolvedRequest & request, const SoundEffectProgress & progress) {
        SoundEffectResult result;
        result.sample_rate = model->config().vae.sample_rate;
        const auto text_start = std::chrono::steady_clock::now();
        const PromptContexts contexts = encode_prompts(request);
        result.text_ms = elapsed_ms(text_start);

        const auto diffusion_start = std::chrono::steady_clock::now();
        std::vector<float> latents = initial_noise(request.seed);
        const bool finished = denoise(latents, request, contexts.positive, contexts.negative, progress);
        result.diffusion_ms = elapsed_ms(diffusion_start);
        result.cancelled = !finished;
        if (finished) {
            finish_decode(result, latents, request.tenths);
        }
        return result;
    }

    SoundEffectResult generate(const SoundEffectRequest & request, const SoundEffectProgress & progress) {
        std::unique_lock<std::mutex> lock(generation_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            fail("generation already in progress on this instance");
        }
        cancel_requested = false;
        const ResolvedRequest resolved = detail::resolve_sfx_request(model->config(), request);
        dit_backends.begin();
        vae_backends.begin();
        SoundEffectResult result = run(resolved, progress);
        result.dit_backend = dit_backends.summary();
        result.vae_backend = vae_backends.summary();
        return result;
    }
};

SoundEffectEngine::SoundEffectEngine(const SoundEffectOptions & options) : impl_(new Impl) {
    impl_->options = options;
    impl_->load();
}

SoundEffectEngine::~SoundEffectEngine() = default;

SoundEffectResult SoundEffectEngine::generate(const SoundEffectRequest & request,
                                              const SoundEffectProgress & progress) {
    return impl_->generate(request, progress);
}

void SoundEffectEngine::cancel() noexcept {
    impl_->cancel_requested = true;
}

int SoundEffectEngine::sample_rate() const noexcept {
    return impl_->model->config().vae.sample_rate;
}

float SoundEffectEngine::max_seconds() const noexcept {
    return impl_->model->config().max_seconds;
}

const char * SoundEffectEngine::backend_name() const noexcept {
    return impl_->model->backend_name();
}

bool SoundEffectEngine::dit_on_coreml() const noexcept {
    return impl_->dit_sidecar != nullptr;
}

bool SoundEffectEngine::vae_on_coreml() const noexcept {
    return impl_->vae_sidecar != nullptr;
}

} // namespace tts_cpp::moss
