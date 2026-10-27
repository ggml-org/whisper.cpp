#pragma once

#include "moss/sfx_model.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace tts_cpp::moss::detail {

void validate_text_encoder(const SfxModel & model);
void validate_dit(const SfxModel & model);
void validate_vae(const SfxModel & model);

std::vector<float> encode_text(SfxModel & model, const std::vector<int32_t> & ids);
SfxMemory measure_text(SfxModel & model, const std::vector<int32_t> & ids);

class SfxDitSession {
public:
    SfxDitSession(SfxModel & model, int frames, bool measure_only = false, int cached_contexts = 0);
    ~SfxDitSession();
    SfxDitSession(const SfxDitSession &) = delete;
    SfxDitSession & operator=(const SfxDitSession &) = delete;

    void prepare_contexts(std::vector<std::vector<float>> contexts);
    std::vector<float> velocity(const std::vector<float> & latents, float timestep,
                                const std::vector<float> & context);
    SfxMemory measure();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

using SfxDecodeProgress = std::function<bool(int done, int total)>;

constexpr int SFX_VAE_CONTEXT_FRAMES = 32;

struct SfxVaeWindow {
    int first = 0;
    int count = 0;
    int keep_first = 0;
    int keep_count = 0;
};

enum class SidecarDecode { Done, Cancelled, Failed };

class SfxVaeSidecar;

std::vector<float> decode_latents(SfxModel & model, const std::vector<float> & latents, int frames,
                                  int keep_frames, int window_frames, const SfxDecodeProgress & progress);
std::vector<SfxVaeWindow> plan_fixed_vae_windows(int frames, int keep_frames, int window, int context);
SidecarDecode decode_latents_on_sidecar(SfxVaeSidecar & sidecar, const SfxVaeConfig & config,
                                        const std::vector<float> & latents, int frames, int keep_frames,
                                        const SfxDecodeProgress & progress, std::vector<float> & pcm);
SfxMemory measure_decode(SfxModel & model, int frames, int keep_frames, int window_frames);

} // namespace tts_cpp::moss::detail
