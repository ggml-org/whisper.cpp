#include "tts-cpp/moss/sound_effect_fit.h"
#include "moss/sfx_model.h"
#include "moss/sfx_networks.h"
#include "moss/sfx_request.h"
#include "moss/sfx_tokenizer.h"
#include "backend_selection.h"
#include "fit_util.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tts_cpp::moss {
namespace {

using fitutil::sat_add;
using fitutil::sat_mul;
constexpr int TENTHS_PER_SECOND = 10;
constexpr int GUIDED_LATENT_BUFFERS = 4;
constexpr int UNGUIDED_LATENT_BUFFERS = 2;
constexpr int PROMPT_CONTEXTS = 2;

uint64_t host_rows_payload(const detail::SfxModel & model, size_t tokens) {
    return model.table_on_host() ? sat_mul(sat_mul(tokens, model.config().text.n_embd), sizeof(float)) : 0;
}

uint64_t text_payload(const detail::SfxModel & model, size_t tokens) {
    const uint64_t indices = sat_add(sat_mul(tokens, sizeof(int32_t) * 2), sat_mul(sat_mul(tokens, tokens), sizeof(float)));
    return sat_add(indices, host_rows_payload(model, tokens));
}

detail::SfxMemory measure_texts(detail::SfxModel & model, detail::SfxTokenizer & tokenizer,
                       const detail::ResolvedRequest & request) {
    const auto positive = tokenizer.encode(request.conditioning);
    const auto negative = tokenizer.encode(request.negative_prompt);
    const auto first = detail::measure_text(model, positive);
    const auto second = detail::measure_text(model, negative);
    detail::SfxMemory result;
    result.device_bytes = std::max(first.device_bytes, second.device_bytes);
    result.host_compute_bytes = std::max(first.host_compute_bytes, second.host_compute_bytes);
    result.host_bytes = std::max(sat_add(first.host_bytes, text_payload(model, positive.size())),
                                 sat_add(second.host_bytes, text_payload(model, negative.size())));
    return result;
}

detail::SfxMemory measure_dit(detail::SfxModel & model, const detail::ResolvedRequest & request) {
    const auto & config = model.config();
    detail::SfxDitSession session(model, config.latent_frames(), true, request.guidance == 1.0f ? 1 : PROMPT_CONTEXTS);
    auto result = session.measure();
    const auto latents = sat_mul(config.latent_frames(), config.dit.in_channels);
    const auto buffers = request.guidance == 1.0f ? UNGUIDED_LATENT_BUFFERS : GUIDED_LATENT_BUFFERS;
    result.host_bytes = sat_add(result.host_bytes, sat_mul(sat_mul(latents, buffers), sizeof(float)));
    result.host_bytes = sat_add(result.host_bytes, sat_mul(config.latent_frames(), sizeof(int32_t)));
    result.host_bytes = sat_add(result.host_bytes, sat_mul(request.steps, sizeof(float)));
    return result;
}

detail::SfxMemory measure_audio(detail::SfxModel & model, const detail::ResolvedRequest & request) {
    const auto & config = model.config();
    const int64_t samples = int64_t(config.vae.sample_rate) * request.tenths / TENTHS_PER_SECOND;
    const int frames = config.latent_frames();
    const int hop = config.vae.hop_length();
    const int keep = (int) std::min<int64_t>(frames, (samples + hop - 1) / hop);
    auto result = detail::measure_decode(model, frames, keep, detail::SFX_DECODE_WINDOW_FRAMES);
    const uint64_t latents = sat_mul(sat_mul(frames, config.dit.in_channels), sizeof(float));
    const uint64_t waveform = sat_mul(sat_mul(keep, hop), sizeof(float));
    result.host_bytes = sat_add(result.host_bytes, sat_add(latents, waveform));
    return result;
}

void measure_workload(FitResult & result, detail::SfxModel & model,
                      const detail::ResolvedRequest & request) {
    detail::SfxTokenizer tokenizer(model);
    const auto text = measure_texts(model, tokenizer, request);
    const auto dit = measure_dit(model, request);
    const auto audio = measure_audio(model, request);
    const auto text_peak = sat_add(dit.persistent_device_bytes, text.device_bytes);
    const auto audio_peak = sat_add(dit.persistent_device_bytes, audio.device_bytes);
    result.device.lm_compute_bytes = std::max(text_peak, dit.device_bytes);
    result.device.codec_compute_bytes =
        audio_peak > result.device.lm_compute_bytes ? audio_peak - result.device.lm_compute_bytes : 0;
    const uint64_t contexts = sat_mul(sat_mul(model.config().text.max_tokens, model.config().text.n_embd),
                                     sizeof(float) * PROMPT_CONTEXTS);
    const uint64_t host_compute = std::max({text.host_compute_bytes, dit.host_compute_bytes, audio.host_compute_bytes});
    result.host_bytes = sat_add(result.host_bytes, host_compute);
    const uint64_t peak = std::max({text.host_bytes, dit.host_bytes, audio.host_bytes});
    result.host_bytes = sat_add(result.host_bytes, sat_add(tokenizer.storage_bytes(), sat_add(contexts, peak)));
}

void finish(FitResult & result, uint64_t margin_bytes) {
    result.device.total_bytes = sat_add(result.device.weights_bytes,
        sat_add(result.device.lm_compute_bytes, result.device.codec_compute_bytes));
    uint64_t required = sat_add(result.device.total_bytes, margin_bytes);
    if (result.device_shares_host_memory) required = sat_add(required, result.host_bytes);
    result.fits = required != std::numeric_limits<uint64_t>::max() && required <= result.device_free_bytes;
    result.status = result.fits ? FitStatus::Success : FitStatus::Failure;
    result.reason = result.fits ? "fits" : "does-not-fit";
    std::ostringstream report;
    report << "model: moss-sfx\ndevice: " << result.device_name
           << "\nweights: " << result.device.weights_bytes
           << " bytes\ntext / diffusion compute peak: " << result.device.lm_compute_bytes
           << " bytes\nadditional decode compute: " << result.device.codec_compute_bytes
           << " bytes\nhost: " << result.host_bytes << " bytes\nmargin: " << margin_bytes
           << " bytes\nverdict: " << result.reason << '\n';
    result.report = report.str();
}

}

FitResult fit_params(const SoundEffectOptions & options, const SoundEffectRequest & request,
                     uint64_t margin_bytes) {
    FitResult result;
    result.model_variant = "moss-sfx";
    if (options.model_path.empty()) {
        result.reason = "invalid-arguments";
        return result;
    }
    if (!options.backends_dir.empty()) ::tts_cpp::detail::set_backends_directory(options.backends_dir);
    try {
        detail::SfxModel model(options.model_path, detail::backend_request(options), options.n_threads, true);
        detail::validate_text_encoder(model);
        detail::validate_dit(model);
        detail::validate_vae(model);
        detail::ResolvedRequest resolved;
        try {
            resolved = detail::resolve_sfx_request(model.config(), request);
        } catch (const std::runtime_error & error) {
            result.reason = "invalid-arguments";
            result.report = error.what();
            return result;
        }
        result = model.measure_weights();
        result.model_variant = "moss-sfx";
        measure_workload(result, model, resolved);
        finish(result, margin_bytes);
    } catch (const std::exception & error) {
        const std::string message = error.what();
        result.reason = message.find("no compute backend") != std::string::npos ? "no-backend-device"
            : message.find("measurement") != std::string::npos ? "measurement-failed" : "model-unreadable";
        result.report = message;
    }
    return result;
}

}
