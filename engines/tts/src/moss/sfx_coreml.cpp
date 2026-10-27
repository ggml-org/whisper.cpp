#include "moss/sfx_coreml.h"

#include <cmath>

namespace tts_cpp::moss::detail {
namespace {

constexpr const char * OWNER = "moss sfx";
constexpr float TIMESTEP_MAX_PERIOD = 10000.0f;
constexpr int SINGLE_BATCH = 1;
constexpr int MONO = 1;
constexpr int LATENT_RANK = 3;

class CoremlSfxDit final : public SfxDitSidecar {
public:
    CoremlSfxDit(std::unique_ptr<CoremlModel> model, std::vector<CoremlTensor> inputs, CoremlTensor output)
        : model_(std::move(model)), inputs_(std::move(inputs)), output_(std::move(output)) {}

    bool velocity(const std::vector<float> & latents, const std::vector<float> & timestep,
                  const std::vector<float> & context, std::vector<float> & velocity) override {
        if (!sizes_match({&latents, &timestep, &context})) {
            return false;
        }
        velocity.assign((size_t) tensor_elements(output_), 0.0f);
        return model_->predict(inputs_, {latents.data(), timestep.data(), context.data()}, {output_},
                {velocity.data()});
    }

    const char * label() const override {
        return model_->label();
    }

private:
    bool sizes_match(const std::vector<const std::vector<float> *> & data) const {
        for (size_t i = 0; i < data.size(); ++i) {
            if ((int64_t) data[i]->size() != tensor_elements(inputs_[i])) {
                return false;
            }
        }
        return true;
    }

    std::unique_ptr<CoremlModel> model_;
    std::vector<CoremlTensor> inputs_;
    CoremlTensor output_;
};

class CoremlSfxVae final : public SfxVaeSidecar {
public:
    CoremlSfxVae(std::unique_ptr<CoremlModel> model, CoremlTensor input, CoremlTensor output, int window)
        : model_(std::move(model)), input_(std::move(input)), output_(std::move(output)), window_(window) {}

    int window() const override {
        return window_;
    }

    bool decode(const std::vector<float> & latents, std::vector<float> & pcm) override {
        if ((int64_t) latents.size() != tensor_elements(input_)) {
            return false;
        }
        pcm.assign((size_t) tensor_elements(output_), 0.0f);
        return model_->predict({input_}, {latents.data()}, {output_}, {pcm.data()});
    }

    const char * label() const override {
        return model_->label();
    }

private:
    std::unique_ptr<CoremlModel> model_;
    CoremlTensor input_;
    CoremlTensor output_;
    int window_;
};

std::vector<CoremlTensor> dit_inputs(const SfxConfig & config) {
    return {{"latents", {SINGLE_BATCH, config.dit.in_channels, config.latent_frames()}},
            {"timestep", {SINGLE_BATCH, config.dit.freq_dim}},
            {"context", {SINGLE_BATCH, config.text.max_tokens, config.dit.text_dim}}};
}

int declared_window(const CoremlModel & model, const SfxConfig & config) {
    const std::vector<int64_t> dims = model.dims("latents");
    if (dims.size() != LATENT_RANK || dims[0] != SINGLE_BATCH || dims[1] != config.vae.latent_dim || dims[2] < 1 ||
        dims[2] > config.latent_frames()) {
        return 0;
    }
    return (int) dims[2];
}

void fill_sinusoid(std::vector<float> & out, float timestep, int half) {
    for (int j = 0; j < half; ++j) {
        const float freq = expf(-logf(TIMESTEP_MAX_PERIOD) * (float) j / (float) half);
        const float arg = timestep * freq;
        out[(size_t) j] = cosf(arg);
        out[(size_t) (j + half)] = sinf(arg);
    }
}

} // namespace

std::vector<float> timestep_sinusoid(float timestep, int dim) {
    std::vector<float> out((size_t) dim, 0.0f);
    fill_sinusoid(out, timestep, dim / 2);
    return out;
}

std::unique_ptr<SfxDitSidecar> open_sfx_dit_sidecar(const std::string & model_path, const SfxConfig & config,
                                                    const CoremlPolicy & policy) {
    std::unique_ptr<CoremlModel> model = CoremlModel::open(sfx_dit_sidecar_path(model_path), policy);
    if (!model) {
        return nullptr;
    }
    std::vector<CoremlTensor> inputs = dit_inputs(config);
    CoremlTensor output{"velocity", {SINGLE_BATCH, config.dit.out_channels, config.latent_frames()}};
    if (!model->declares(inputs) || !model->declares(output)) {
        return nullptr;
    }
    return std::make_unique<CoremlSfxDit>(std::move(model), std::move(inputs), std::move(output));
}

std::unique_ptr<SfxVaeSidecar> open_sfx_vae_sidecar(const std::string & model_path, const SfxConfig & config,
                                                    const CoremlPolicy & policy) {
    std::unique_ptr<CoremlModel> model = CoremlModel::open(sfx_vae_sidecar_path(model_path), policy);
    if (!model) {
        return nullptr;
    }
    const int window = declared_window(*model, config);
    CoremlTensor input{"latents", {SINGLE_BATCH, config.vae.latent_dim, window}};
    CoremlTensor output{"pcm", {SINGLE_BATCH, MONO, (int64_t) window * config.vae.hop_length()}};
    if (window < 1 || !model->declares(output)) {
        return nullptr;
    }
    return std::make_unique<CoremlSfxVae>(std::move(model), std::move(input), std::move(output), window);
}

SfxDitRunner::SfxDitRunner(SfxModel & model, int frames, std::unique_ptr<SfxDitSidecar> & sidecar, bool strict,
                           StageBackends & backends)
    : model_(model), frames_(frames), sidecar_(sidecar), strict_(strict), backends_(backends) {}

void SfxDitRunner::prepare_contexts(std::vector<std::vector<float>> contexts) {
    session_.reset();
    contexts_ = std::move(contexts);
}

std::vector<float> SfxDitRunner::velocity(const std::vector<float> & latents, float timestep,
                                          const std::vector<float> & context) {
    std::vector<float> result;
    if (try_sidecar(latents, timestep, context, result)) {
        return result;
    }
    if (strict_) {
        fail_strict(OWNER, "DiT");
    }
    return on_ggml(latents, timestep, context);
}

bool SfxDitRunner::try_sidecar(const std::vector<float> & latents, float timestep, const std::vector<float> & context,
                               std::vector<float> & velocity) {
    if (!sidecar_) {
        return false;
    }
    if (!sidecar_->velocity(latents, timestep_sinusoid(timestep, model_.config().dit.freq_dim), context, velocity)) {
        sidecar_.reset();
        return false;
    }
    backends_.note(sidecar_->label());
    return true;
}

std::vector<float> SfxDitRunner::on_ggml(const std::vector<float> & latents, float timestep,
                                         const std::vector<float> & context) {
    if (!session_) {
        session_ = std::make_unique<SfxDitSession>(model_, frames_, false, (int) contexts_.size());
        if (!contexts_.empty()) session_->prepare_contexts(std::move(contexts_));
    }
    backends_.note(GGML_STAGE_BACKEND);
    return session_->velocity(latents, timestep, context);
}

std::vector<float> decode_sfx_latents(SfxModel & model, std::unique_ptr<SfxVaeSidecar> & sidecar, bool strict,
                                      StageBackends & backends, const std::vector<float> & latents, int frames,
                                      int keep_frames, int window_frames, const SfxDecodeProgress & progress) {
    if (sidecar) {
        std::vector<float> pcm;
        const SidecarDecode outcome = decode_latents_on_sidecar(*sidecar, model.config().vae, latents, frames,
                keep_frames, progress, pcm);
        if (outcome == SidecarDecode::Done) {
            backends.note(sidecar->label());
            return pcm;
        }
        if (outcome == SidecarDecode::Cancelled) {
            return {};
        }
        sidecar.reset();
    }
    if (strict) {
        fail_strict(OWNER, "VAE");
    }
    backends.note(GGML_STAGE_BACKEND);
    return decode_latents(model, latents, frames, keep_frames, window_frames, progress);
}

} // namespace tts_cpp::moss::detail
