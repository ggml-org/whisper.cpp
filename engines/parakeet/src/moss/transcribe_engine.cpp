#include "parakeet/moss_transcribe.h"

#include "moss/transcribe_audio.h"
#include "moss/transcribe_coreml.h"
#include "moss/transcribe_model.h"
#include "moss/transcribe_networks.h"
#include "moss/transcribe_text.h"
#include "moss/transcribe_request.h"

#include "parakeet_ctc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace parakeet::moss {
namespace {

using detail::TranscribeDecoder;
using detail::TranscribeMel;
using detail::TranscribeModel;
using detail::TranscribeTokenizer;


[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss transcribe: " + message);
}

double elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

int32_t greedy_token(const std::vector<float> & logits) {
    return (int32_t) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

} // namespace

struct TranscribeEngine::Impl {
    TranscribeOptions options;
    std::unique_ptr<TranscribeModel> model;
    std::unique_ptr<TranscribeTokenizer> tokenizer;
    std::unique_ptr<TranscribeMel> mel;
    std::unique_ptr<detail::TranscribeAudioEncoder> audio_encoder;
    std::atomic<bool> cancel_requested{false};
    std::mutex transcription_mutex;

    void load() {
        if (options.model_path.empty()) {
            fail("model_path is required");
        }
        if (!options.backends_dir.empty()) {
            ::parakeet::set_backends_directory(options.backends_dir);
        }
        model = std::make_unique<TranscribeModel>(options.model_path, options.use_gpu, options.n_threads, options.backend);
        detail::validate_transcribe_encoder(*model);
        detail::validate_transcribe_decoder(*model);
        tokenizer = std::make_unique<TranscribeTokenizer>(*model);
        mel = std::make_unique<TranscribeMel>(model->config().audio, detail::read_mel_filters(*model));
        attach_audio_encoder();
    }

    void attach_audio_encoder() {
        const detail::CoremlPolicy policy = detail::read_coreml_policy();
        audio_encoder = std::make_unique<detail::TranscribeAudioEncoder>(*model,
                detail::open_transcribe_encoder_sidecar(options.model_path, model->config(), policy), policy.strict);
    }

    void require_context(int prompt_tokens, int limit) const {
        const int64_t needed = (int64_t) prompt_tokens + limit;
        if (needed > model->config().text.n_ctx_train) {
            fail("audio and max_new_tokens need " + std::to_string(needed) + " positions; the decoder holds " +
                 std::to_string(model->config().text.n_ctx_train));
        }
    }

    void validate(const float * pcm, size_t samples, int sample_rate, const TranscribeRequest & request) const {
        if (sample_rate != model->config().audio.sample_rate) {
            fail("audio must be sampled at " + std::to_string(model->config().audio.sample_rate) + " Hz");
        }
        if (pcm == nullptr || samples == 0) {
            fail("audio must not be empty");
        }
        if (request.prompt.size() > detail::TRANSCRIBE_MAX_PROMPT_BYTES) {
            fail("prompt must be at most " + std::to_string(detail::TRANSCRIBE_MAX_PROMPT_BYTES) + " bytes");
        }
    }

    std::vector<float> chunk_embeddings(const float * pcm, size_t samples, size_t chunk) {
        const size_t length = (size_t) model->config().audio.chunk_samples;
        const size_t first = chunk * length;
        const size_t count = std::min(length, samples - first);
        const int tokens = detail::transcribe_chunk_tokens(model->config(), count);
        return audio_encoder->encode(mel->chunk(pcm + first, count), tokens);
    }

    bool encode_audio(const float * pcm, size_t samples, std::vector<float> & embeddings) {
        const size_t chunks = detail::transcribe_chunk_count(model->config().audio, samples);
        for (size_t chunk = 0; chunk < chunks; ++chunk) {
            if (cancel_requested) {
                return false;
            }
            const std::vector<float> encoded = chunk_embeddings(pcm, samples, chunk);
            embeddings.insert(embeddings.end(), encoded.begin(), encoded.end());
        }
        return true;
    }

    bool keep_going(const TranscribeProgress & progress, int generated, int limit) const {
        if (cancel_requested) {
            return false;
        }
        return !progress || progress(generated, limit);
    }

    bool generate(TranscribeDecoder & decoder, std::vector<float> logits, int limit,
                  const TranscribeProgress & progress, std::vector<int32_t> & generated) {
        const int32_t end = model->config().tokens.im_end;
        for (int count = 0; count < limit; ++count) {
            const int32_t token = greedy_token(logits);
            if (token == end) {
                return true;
            }
            generated.push_back(token);
            if (!keep_going(progress, (int) generated.size(), limit)) {
                return false;
            }
            if (count + 1 < limit) {
                logits = decoder.step(token);
            }
        }
        return true;
    }

    void finish(TranscribeResult & result, const std::vector<int32_t> & generated) const {
        result.generated_tokens = (int) generated.size();
        result.text = detail::strip_whitespace(tokenizer->decode(generated));
        result.segments = detail::parse_transcript(result.text);
    }

    TranscribeResult run(const float * pcm, size_t samples, const TranscribeRequest & request,
                         const TranscribeProgress & progress) {
        TranscribeResult result;
        const int limit = detail::resolved_transcribe_limit(model->config(), request);
        result.audio_tokens = detail::transcribe_audio_tokens(model->config(), samples);
        const std::vector<int32_t> prompt = detail::transcribe_prompt(model->config(), *tokenizer,
                result.audio_tokens, detail::resolved_transcribe_prompt(model->config(), request));
        result.prompt_tokens = (int) prompt.size();
        require_context(result.prompt_tokens, limit);

        const auto encode_start = std::chrono::steady_clock::now();
        std::vector<float> embeddings;
        audio_encoder->begin_run();
        result.cancelled = !encode_audio(pcm, samples, embeddings);
        result.encode_ms = elapsed_ms(encode_start);
        result.encoder_backend = audio_encoder->run_backend();
        if (result.cancelled) {
            return result;
        }

        const auto prefill_start = std::chrono::steady_clock::now();
        TranscribeDecoder decoder(*model, result.prompt_tokens + limit);
        std::vector<float> logits = decoder.prefill(prompt, embeddings, detail::TRANSCRIBE_PREFILL_BATCH_TOKENS);
        result.prefill_ms = elapsed_ms(prefill_start);

        const auto decode_start = std::chrono::steady_clock::now();
        std::vector<int32_t> generated;
        result.cancelled = !generate(decoder, std::move(logits), limit, progress, generated);
        result.decode_ms = elapsed_ms(decode_start);
        finish(result, generated);
        return result;
    }

    TranscribeResult transcribe(const float * pcm, size_t samples, int sample_rate, const TranscribeRequest & request,
                                const TranscribeProgress & progress) {
        std::unique_lock<std::mutex> lock(transcription_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            fail("transcription already in progress on this instance");
        }
        cancel_requested = false;
        validate(pcm, samples, sample_rate, request);
        return run(pcm, samples, request, progress);
    }
};

TranscribeEngine::TranscribeEngine(const TranscribeOptions & options) : impl_(new Impl) {
    impl_->options = options;
    impl_->load();
}

TranscribeEngine::~TranscribeEngine() = default;

TranscribeResult TranscribeEngine::transcribe(const float * pcm, size_t samples, int sample_rate,
                                              const TranscribeRequest & request,
                                              const TranscribeProgress & progress) {
    return impl_->transcribe(pcm, samples, sample_rate, request, progress);
}

void TranscribeEngine::cancel() noexcept {
    impl_->cancel_requested = true;
}

int TranscribeEngine::sample_rate() const noexcept {
    return impl_->model->config().audio.sample_rate;
}

const char * TranscribeEngine::backend_name() const noexcept {
    return impl_->model->backend_name();
}

bool TranscribeEngine::encoder_on_coreml() const noexcept {
    return impl_->audio_encoder->on_coreml();
}

} // namespace parakeet::moss
