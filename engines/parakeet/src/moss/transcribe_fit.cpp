#include "moss/transcribe_fit.h"
#include "fit_util.h"
#include "moss/transcribe_audio.h"
#include "moss/transcribe_networks.h"
#include "moss/transcribe_request.h"
#include "parakeet/moss_transcribe_fit.h"
#include "parakeet_ctc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace parakeet::moss {
namespace {

using detail::TranscribeMemory;
using fitutil::sat_add;
using fitutil::sat_mul;
constexpr uint64_t VECTOR_CAPACITY_FACTOR = 2;
constexpr uint64_t TRANSCRIPT_COPIES = 4;
constexpr int MAX_THREADS = 1024;

struct Workload {
    size_t samples;
    int audio_tokens;
    int limit;
    std::vector<int32_t> prompt;
};

void merge_memory(TranscribeMemory & peak, const TranscribeMemory & phase) {
    peak.device_bytes = std::max(peak.device_bytes, phase.device_bytes);
    peak.host_compute_bytes = std::max(peak.host_compute_bytes, phase.host_compute_bytes);
    peak.host_bytes = std::max(peak.host_bytes, phase.host_bytes);
    peak.cpu_work_bytes = std::max(peak.cpu_work_bytes, phase.cpu_work_bytes);
}

bool resolve_workload(FitResult & result, const detail::TranscribeConfig & config,
                      const detail::TranscribeTokenizer & tokenizer, const TranscribeRequest & request,
                      double seconds, Workload & workload) {
    const long double samples = std::ceil((long double) seconds * config.audio.sample_rate);
    const uint64_t max_samples = sat_mul(config.text.n_ctx_train, config.samples_per_token());
    if (samples > max_samples || samples >= (long double) std::numeric_limits<size_t>::max()) {
        result.reason = "workload-too-large";
        return false;
    }
    workload.samples = (size_t) samples;
    workload.audio_tokens = detail::transcribe_audio_tokens(config, workload.samples);
    workload.limit = detail::resolved_transcribe_limit(config, request);
    if (workload.limit > config.text.n_ctx_train || workload.audio_tokens > config.text.n_ctx_train - workload.limit) {
        result.reason = "workload-too-large";
        return false;
    }
    workload.prompt = detail::transcribe_prompt(config, tokenizer, workload.audio_tokens,
        detail::resolved_transcribe_prompt(config, request));
    if (workload.prompt.size() + (uint64_t) workload.limit > (uint64_t) config.text.n_ctx_train) {
        result.reason = "workload-too-large";
        return false;
    }
    return true;
}

TranscribeMemory measure_encoder(detail::TranscribeModel & model, const Workload & workload) {
    const size_t chunk = (size_t) model.config().audio.chunk_samples;
    const int tokens = detail::transcribe_chunk_tokens(model.config(), std::min(workload.samples, chunk));
    auto peak = detail::measure_audio_chunk(model, tokens);
    const size_t tail = workload.samples % chunk;
    if (workload.samples > chunk && tail > 0) {
        merge_memory(peak, detail::measure_audio_chunk(model, detail::transcribe_chunk_tokens(model.config(), tail)));
    }
    return peak;
}

TranscribeMemory measure_prefill(detail::TranscribeDecoder & decoder, const detail::TranscribeConfig & config,
                                 const Workload & workload) {
    TranscribeMemory peak;
    constexpr size_t batch = detail::TRANSCRIBE_PREFILL_BATCH_TOKENS;
    for (size_t first = 0; first < workload.prompt.size(); first += batch) {
        const size_t count = std::min(batch, workload.prompt.size() - first);
        const auto begin = workload.prompt.begin() + (std::ptrdiff_t) first;
        const bool audio = std::find(begin, begin + (std::ptrdiff_t) count, config.tokens.audio_pad)
            != begin + (std::ptrdiff_t) count;
        merge_memory(peak, decoder.measure_batch((int) first, (int) count, audio,
            first + count == workload.prompt.size()));
    }
    if (workload.limit > 1) {
        merge_memory(peak, decoder.measure_batch((int) workload.prompt.size() + workload.limit - 2, 1, false, true));
    }
    return peak;
}

uint64_t mel_storage(const detail::TranscribeAudioConfig & audio) {
    const uint64_t bins = audio.n_fft / 2 + 1;
    return sat_add(sat_mul(sat_mul(bins, audio.n_mels), sizeof(float)),
        sat_mul(audio.n_fft, sizeof(float) + sizeof(std::complex<float>)));
}

uint64_t encoder_payload(const detail::TranscribeConfig & config) {
    const auto & audio = config.audio;
    const uint64_t bins = audio.n_fft / 2 + 1;
    const uint64_t mel = sat_mul(audio.chunk_frames, audio.n_mels);
    const uint64_t power = sat_mul(audio.chunk_frames, bins);
    const uint64_t padded = sat_add(audio.chunk_samples, audio.n_fft);
    const uint64_t fft = sat_mul(audio.n_fft, sizeof(float) + sizeof(std::complex<float>) * 2);
    const uint64_t embeddings = sat_mul(config.encoder.n_ctx / config.merge_size, config.text.n_embd);
    return sat_add(fft, sat_mul(sat_add(sat_add(mel, power), sat_add(padded, embeddings)), sizeof(float)));
}

uint64_t decoder_payload(const detail::TranscribeConfig & config, const Workload & workload,
                         const detail::TranscribeTokenizer & tokenizer) {
    const uint64_t batch = std::min<size_t>(detail::TRANSCRIBE_PREFILL_BATCH_TOKENS, workload.prompt.size());
    const uint64_t mask = sat_mul(batch, workload.prompt.size());
    const uint64_t input = sat_mul(batch, config.text.n_embd + 2);
    uint64_t bytes = sat_add(sat_mul(sat_add(mask, input), sizeof(float)), sat_mul(batch, sizeof(int32_t) * 2));
    bytes = sat_add(bytes, sat_mul(config.text.vocab, sizeof(float) * 2));
    bytes = sat_add(bytes, sat_mul(workload.limit, sizeof(int32_t) * VECTOR_CAPACITY_FACTOR));
    bytes = sat_add(bytes, sat_mul(sat_mul(workload.limit, tokenizer.max_piece_bytes()), TRANSCRIPT_COPIES));
    bytes = sat_add(bytes, sat_mul(workload.limit, sizeof(TranscriptSegment) * VECTOR_CAPACITY_FACTOR));
    return bytes;
}

void measure_workload(FitResult & result, detail::TranscribeModel & model,
                      const detail::TranscribeTokenizer & tokenizer, const Workload & workload) {
    const auto & config = model.config();
    const auto encoder = measure_encoder(model, workload);
    detail::TranscribeDecoder decoder(model, (int) workload.prompt.size() + workload.limit);
    result.device.decoder_state_bytes = decoder.measure_cache();
    const auto decode = measure_prefill(decoder, config, workload);
    result.device.encoder_compute_bytes = encoder.device_bytes;
    result.device.decoder_compute_bytes = decode.device_bytes > encoder.device_bytes
        ? decode.device_bytes - encoder.device_bytes : 0;
    uint64_t resident = sat_add(tokenizer.storage_bytes(), mel_storage(config.audio));
    resident = sat_add(resident, sat_mul(workload.samples, sizeof(float)));
    resident = sat_add(resident, sat_mul(workload.prompt.capacity(), sizeof(int32_t)));
    resident = sat_add(resident, sat_mul(sat_mul(workload.audio_tokens, config.text.n_embd),
        sizeof(float) * VECTOR_CAPACITY_FACTOR));
    const uint64_t encode_host = sat_add(encoder.host_bytes, encoder_payload(config));
    const uint64_t decode_host = sat_add(decode.host_bytes,
        sat_add(decoder.host_state_bytes(), decoder_payload(config, workload, tokenizer)));
    resident = sat_add(resident, std::max(encode_host, decode_host));
    resident = sat_add(resident, std::max(encoder.host_compute_bytes, decode.host_compute_bytes));
    resident = sat_add(resident,
                       std::max(encoder.cpu_work_bytes, decode.cpu_work_bytes));
    result.host_bytes = sat_add(result.host_bytes, resident);
}

} // namespace

void detail::finish_transcribe_fit(FitResult &result, uint64_t margin) {
  result.device.total_bytes =
      sat_add(result.device.weights_bytes,
              sat_add(result.device.decoder_state_bytes,
                      sat_add(result.device.encoder_compute_bytes,
                              result.device.decoder_compute_bytes)));
  uint64_t required = sat_add(result.device.total_bytes, margin);
  if (result.device_shares_host_memory)
    required = sat_add(required, result.host_bytes);
  result.fits = required != std::numeric_limits<uint64_t>::max() &&
                required <= result.device_free_bytes;
  result.status = result.fits ? FitStatus::Success : FitStatus::Failure;
  result.reason = result.fits ? "fits" : "does-not-fit";
  std::ostringstream report;
  report << "model: moss-transcribe\ndevice: " << result.device_name
         << "\nweights: " << result.device.weights_bytes
         << " bytes\nencoder compute: " << result.device.encoder_compute_bytes
         << " bytes\nKV cache: " << result.device.decoder_state_bytes
         << " bytes\nadditional decoder compute: "
         << result.device.decoder_compute_bytes
         << " bytes\nhost: " << result.host_bytes
         << " bytes\nmargin: " << margin << " bytes\nverdict: " << result.reason
         << '\n';
  result.report = report.str();
}

FitResult fit_params(const TranscribeOptions & options, const TranscribeRequest & request,
                     double audio_seconds, uint64_t margin_bytes) {
    FitResult result;
    result.model_type = "moss-transcribe";
    if (options.model_path.empty() || options.n_threads < 1 || options.n_threads > MAX_THREADS ||
        !std::isfinite(audio_seconds) || audio_seconds <= 0 || request.max_new_tokens < 0 ||
        request.prompt.size() > detail::TRANSCRIBE_MAX_PROMPT_BYTES) {
        result.reason = "invalid-arguments";
        return result;
    }
    if (!options.backends_dir.empty()) ::parakeet::set_backends_directory(options.backends_dir);
    try {
        detail::TranscribeModel model(options.model_path, options.use_gpu, options.n_threads, options.backend, true);
        detail::validate_transcribe_encoder(model);
        detail::validate_transcribe_decoder(model);
        detail::TranscribeTokenizer tokenizer(model);
        Workload workload{};
        try {
            if (!resolve_workload(result, model.config(), tokenizer, request, audio_seconds, workload)) return result;
        } catch (const std::runtime_error & error) {
            result.reason = "invalid-arguments";
            result.report = error.what();
            return result;
        }
        result = model.measure_weights();
        result.model_type = "moss-transcribe";
        measure_workload(result, model, tokenizer, workload);
        detail::finish_transcribe_fit(result, margin_bytes);
    } catch (const std::exception & error) {
        const std::string message = error.what();
        result.reason = message.find("no compute backend") != std::string::npos ? "no-backend-device"
            : message.find("measurement") != std::string::npos ? "measurement-failed" : "model-unreadable";
        result.report = message;
    }
    return result;
}

}
