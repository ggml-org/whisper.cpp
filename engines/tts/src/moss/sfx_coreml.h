#pragma once

#include "moss/coreml_sidecar.h"
#include "moss/sfx_model.h"
#include "moss/sfx_networks.h"

#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

class SfxDitSidecar {
public:
    virtual ~SfxDitSidecar() = default;
    virtual bool velocity(const std::vector<float> & latents, const std::vector<float> & timestep,
                          const std::vector<float> & context, std::vector<float> & velocity) = 0;
    virtual const char * label() const = 0;
};

class SfxVaeSidecar {
public:
    virtual ~SfxVaeSidecar() = default;
    virtual int window() const = 0;
    virtual bool decode(const std::vector<float> & latents, std::vector<float> & pcm) = 0;
    virtual const char * label() const = 0;
};

std::vector<float> timestep_sinusoid(float timestep, int dim);

std::unique_ptr<SfxDitSidecar> open_sfx_dit_sidecar(const std::string & model_path, const SfxConfig & config,
                                                    const CoremlPolicy & policy);
std::unique_ptr<SfxVaeSidecar> open_sfx_vae_sidecar(const std::string & model_path, const SfxConfig & config,
                                                    const CoremlPolicy & policy);

class SfxDitRunner {
public:
    SfxDitRunner(SfxModel & model, int frames, std::unique_ptr<SfxDitSidecar> & sidecar, bool strict,
                 StageBackends & backends);

    void prepare_contexts(std::vector<std::vector<float>> contexts);
    std::vector<float> velocity(const std::vector<float> & latents, float timestep, const std::vector<float> & context);

private:
    bool try_sidecar(const std::vector<float> & latents, float timestep, const std::vector<float> & context,
                     std::vector<float> & velocity);
    std::vector<float> on_ggml(const std::vector<float> & latents, float timestep, const std::vector<float> & context);

    SfxModel & model_;
    int frames_;
    std::unique_ptr<SfxDitSidecar> & sidecar_;
    bool strict_;
    StageBackends & backends_;
    std::unique_ptr<SfxDitSession> session_;
    std::vector<std::vector<float>> contexts_;
};

std::vector<float> decode_sfx_latents(SfxModel & model, std::unique_ptr<SfxVaeSidecar> & sidecar, bool strict,
                                      StageBackends & backends, const std::vector<float> & latents, int frames,
                                      int keep_frames, int window_frames, const SfxDecodeProgress & progress);

} // namespace tts_cpp::moss::detail
