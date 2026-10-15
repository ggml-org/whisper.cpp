#include "tts-cpp/moss/fit.h"
#include "moss/codec.h"
#include "moss/delay_lm.h"
#include "backend_selection.h"
#include "fit_util.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tts_cpp::moss {
namespace {

constexpr int CONTEXT_HEADROOM = 8;
constexpr int MIN_CONTEXT = 32;
constexpr int MAX_CONTEXT = 32768;
constexpr int MAX_THREADS = 1024;
constexpr int SINGLE_DECODE_MAX_FRAMES = 750;
constexpr int TERMINATION_ROWS = 2;
constexpr int MAX_REFERENCE_SECONDS = 60;

using fitutil::sat_add;
using fitutil::sat_mul;

bool valid_options(const EngineOptions & options, const FitWorkload & workload) {
    return !options.backbone_path.empty() && !options.decoder_path.empty() &&
        options.context >= MIN_CONTEXT && options.context <= MAX_CONTEXT &&
        options.n_threads > 0 && options.n_threads <= MAX_THREADS &&
        options.max_new_tokens > 0 && options.duration_tokens >= 0 &&
        workload.prompt_rows > 0 && workload.reference_samples >= 0 &&
        (workload.reference_samples == 0 || !options.encoder_path.empty()) &&
        (options.reference_audio_path.empty() || options.dialogue_reference_paths.empty());
}

bool valid_budget(const EngineOptions & options, const FitWorkload & workload, int channels) {
    const int64_t total = int64_t(workload.prompt_rows) + options.max_new_tokens + CONTEXT_HEADROOM;
    const int64_t duration = int64_t(options.duration_tokens) + channels - 1 + TERMINATION_ROWS;
    return total <= options.context &&
        (options.duration_tokens == 0 || duration <= options.max_new_tokens);
}

void add_component(FitResult & result, const FitResult & component) {
    if (result.device_name != component.device_name) {
        throw std::runtime_error("component backend mismatch");
    }
    result.device_free_bytes = std::min(result.device_free_bytes, component.device_free_bytes);
    result.device.weights_bytes = sat_add(result.device.weights_bytes, component.device.weights_bytes);
    result.device.state_bytes = sat_add(result.device.state_bytes, component.device.state_bytes);
    result.device.codec_compute_bytes = sat_add(result.device.codec_compute_bytes, component.device.codec_compute_bytes);
    result.host_bytes = sat_add(result.host_bytes, component.host_bytes);
}

void validate_codec(const detail::Codec & codec, bool encoder, int channels) {
    if (codec.is_encoder() != encoder || codec.num_quantizers() < channels) {
        throw std::runtime_error("codec role or quantizer count does not match the backbone");
    }
}

void measure_reference(FitResult & result, const EngineOptions & options,
                       const FitWorkload & workload, int channels, int sample_rate, int hop) {
    if (workload.reference_samples == 0) return;
    detail::Codec encoder(options.encoder_path, detail::backend_request(options), options.n_threads, true);
    validate_codec(encoder, true, channels);
    if (encoder.sample_rate() != sample_rate || encoder.samples_per_frame() != hop) {
        throw std::runtime_error("codec frame geometry does not match");
    }
    if (workload.reference_samples < hop ||
        workload.reference_samples > int64_t(sample_rate) * MAX_REFERENCE_SECONDS) {
        throw std::invalid_argument("reference workload is outside the codec range");
    }
    add_component(result, encoder.measure(workload.reference_samples, encoder.num_quantizers(), false));
}

void measure_decoder(FitResult & result, detail::Codec & decoder,
                     const EngineOptions & options, const FitWorkload & workload, int channels) {
    const int64_t reference_frames = workload.reference_samples / decoder.samples_per_frame();
    const int64_t frames = sat_add(options.max_new_tokens, reference_frames);
    const bool streaming = workload.streaming || frames > SINGLE_DECODE_MAX_FRAMES;
    const int64_t window = workload.streaming ? std::max(1, options.stream_chunk_frames)
                                            : SINGLE_DECODE_MAX_FRAMES;
    const int64_t measured_frames = streaming ? std::min(frames, window) : frames;
    add_component(result, decoder.measure(measured_frames, channels, streaming));
    const uint64_t waveform = sat_mul(sat_mul(frames, decoder.samples_per_frame()), sizeof(float));
    const uint64_t codes = sat_mul(sat_mul(frames + workload.prompt_rows, channels), sizeof(int32_t));
    result.host_bytes = sat_add(result.host_bytes, sat_add(sat_mul(waveform, 2), sat_mul(codes, 4)));
    result.host_bytes = sat_add(result.host_bytes, sat_mul(workload.reference_samples, sizeof(float)));
}

void finish_result(FitResult & result, const FitWorkload & workload) {
    result.device.total_bytes = sat_add(
        sat_add(result.device.weights_bytes, result.device.state_bytes),
        sat_add(result.device.lm_compute_bytes, result.device.codec_compute_bytes));
    uint64_t required = sat_add(result.device.total_bytes, workload.margin_bytes);
    if (result.device_shares_host_memory) required = sat_add(required, result.host_bytes);
    result.fits = required != std::numeric_limits<uint64_t>::max() && required <= result.device_free_bytes;
    result.status = result.fits ? FitStatus::Success : FitStatus::Failure;
    result.reason = result.fits ? "fits" : "does-not-fit";
    std::ostringstream report;
    report << "model: moss (TTS / TTSD)\ndevice: " << result.device_name
           << "\nweights: " << result.device.weights_bytes << " bytes\nstate: " << result.device.state_bytes
           << " bytes\nlm compute: " << result.device.lm_compute_bytes
           << " bytes\ncodec compute: " << result.device.codec_compute_bytes
           << " bytes\nhost: " << result.host_bytes << " bytes\nmargin: " << workload.margin_bytes
           << " bytes\nverdict: " << result.reason << '\n';
    result.report = report.str();
}

}

FitResult fit_params(const EngineOptions & options, const FitWorkload & workload) {
    FitResult result;
    result.model_variant = "moss";
    if (!valid_options(options, workload)) {
        result.reason = "invalid-arguments";
        return result;
    }
    if (!options.backends_dir.empty()) ::tts_cpp::detail::set_backends_directory(options.backends_dir);
    try {
        detail::DelayLM backbone(options.backbone_path, detail::backend_request(options), options.n_threads,
                options.context, true);
        const int channels = backbone.config().n_vq;
        if (!valid_budget(options, workload, channels)) {
            result.reason = "workload-too-large";
            return result;
        }
        detail::Codec decoder(options.decoder_path, detail::backend_request(options), options.n_threads, true);
        validate_codec(decoder, false, channels);
        result = backbone.measure(workload.prompt_rows, options.max_new_tokens);
        result.model_variant = "moss";
        measure_reference(result, options, workload, channels, decoder.sample_rate(), decoder.samples_per_frame());
        measure_decoder(result, decoder, options, workload, channels);
        finish_result(result, workload);
    } catch (const std::invalid_argument &) {
        result.reason = "invalid-arguments";
    } catch (const std::exception & error) {
        const std::string message = error.what();
        result.reason = message.find("no compute backend") != std::string::npos ? "no-backend-device"
            : message.find("measurement") != std::string::npos ? "measurement-failed" : "model-unreadable";
        result.report = message;
    }
    return result;
}

}
