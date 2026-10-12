#include "tts-cpp/moss/speech.h"

#include "moss/speech_codec.h"
#include "moss/speech_generation.h"
#include "moss/speech_lm.h"
#include "moss/speech_prompt.h"
#include "moss/speech_request.h"

#include "backend_selection.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace tts_cpp::moss {
namespace {

using detail::SpeechCodec;
using detail::SpeechLM;
using detail::SpeechRow;
using detail::SpeechTextTokenizer;
using detail::SpeechTurn;
using detail::SpeechTurnRole;

constexpr int MIN_NEW_TOKENS = 10;
constexpr int PREFILL_BATCH_TOKENS = 256;
constexpr size_t MAX_MESSAGES = 256;
constexpr size_t MAX_TEXT_BYTES = 16384;
constexpr int MIN_SAMPLE_RATE = 8000;
constexpr int MAX_SAMPLE_RATE = 192000;
constexpr double MAX_AUDIO_SECONDS = 600.0;
constexpr double MAX_VOICE_SECONDS = 60.0;
constexpr float MAX_TEMPERATURE = 10.0f;
constexpr float MAX_REPLY_SECONDS = 3600.0f;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss speech: " + message);
}

double elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

SpeechTurnRole turn_role(SpeechRole role) {
    switch (role) {
        case SpeechRole::System:    return SpeechTurnRole::System;
        case SpeechRole::Assistant: return SpeechTurnRole::Assistant;
        case SpeechRole::User:      break;
    }
    return SpeechTurnRole::User;
}

void validate_audio(const std::vector<float> & audio, int sample_rate, double max_seconds, const char * what) {
    if (sample_rate < MIN_SAMPLE_RATE || sample_rate > MAX_SAMPLE_RATE) {
        fail(std::string(what) + " sample rate must be in [" + std::to_string(MIN_SAMPLE_RATE) + ", " +
             std::to_string(MAX_SAMPLE_RATE) + "] Hz");
    }
    if ((double) audio.size() / sample_rate > max_seconds) {
        fail(std::string(what) + " is longer than " + std::to_string((int) max_seconds) + " s");
    }
}

void validate_message(const SpeechMessage & message) {
    if (message.audio.empty() == message.text.empty()) {
        fail("each message carries exactly one of text or audio");
    }
    if (message.text.size() > MAX_TEXT_BYTES) {
        fail("message text must be at most " + std::to_string(MAX_TEXT_BYTES) + " bytes");
    }
    if (!message.audio.empty()) {
        validate_audio(message.audio, message.sample_rate, MAX_AUDIO_SECONDS, "message audio");
    }
}

void validate_messages(const std::vector<SpeechMessage> & messages) {
    if (messages.empty() || messages.size() > MAX_MESSAGES) {
        fail("a request needs 1.." + std::to_string(MAX_MESSAGES) + " messages");
    }
    for (const SpeechMessage & message : messages) {
        validate_message(message);
    }
    if (messages.back().role != SpeechRole::User) {
        fail("the last message must be the user's turn");
    }
}

void validate_sampling(const SpeechRequest & request) {
    if (!request.greedy && !(request.temperature > 0.0f && request.temperature <= MAX_TEMPERATURE)) {
        fail("temperature must be in (0, " + std::to_string((int) MAX_TEMPERATURE) + "]");
    }
    if (!(request.top_p > 0.0f && request.top_p <= 1.0f) || request.top_k < 0) {
        fail("top_p must be in (0, 1] and top_k must not be negative");
    }
    if (request.max_new_tokens < 0) {
        fail("max_new_tokens must not be negative");
    }
    if (!std::isfinite(request.max_reply_seconds) || request.max_reply_seconds < 0.0f ||
        request.max_reply_seconds > MAX_REPLY_SECONDS) {
        fail("max_reply_seconds must be in [0, " + std::to_string((int) MAX_REPLY_SECONDS) + "]");
    }
}

detail::SpeechSampling sampling_of(const SpeechRequest & request) {
    return {!request.greedy, request.temperature, request.top_p, request.top_k};
}

} // namespace

namespace detail {

void validate_speech_request(const SpeechRequest & request) {
    validate_messages(request.messages);
    validate_sampling(request);
    if (!request.voice.empty()) {
        validate_audio(request.voice, request.voice_sample_rate, MAX_VOICE_SECONDS, "voice prompt");
    }
}

void check_speech_context(size_t prompt_rows, int max_new_tokens, int n_ctx_train) {
    if ((int64_t) prompt_rows + max_new_tokens > n_ctx_train) {
        fail("the conversation and max_new_tokens exceed the model context");
    }
}

int speech_generation_budget(size_t prompt_rows, int requested_tokens, int n_ctx_train) {
    if (requested_tokens < 0) {
        fail("max_new_tokens must not be negative");
    }
    if (n_ctx_train <= 0 || prompt_rows >= (size_t) n_ctx_train) {
        fail("the conversation leaves no room for a reply in the model context");
    }
    const int budget = requested_tokens > 0 ? requested_tokens : n_ctx_train - (int) prompt_rows;
    check_speech_context(prompt_rows, budget, n_ctx_train);
    return budget;
}

} // namespace detail

struct SpeechEngine::Impl {
    SpeechOptions options;
    std::unique_ptr<SpeechLM> lm;
    std::unique_ptr<SpeechCodec> codec;
    std::unique_ptr<SpeechTextTokenizer> tokenizer;
    std::atomic<bool> cancel_requested{false};
    std::mutex respond_mutex;

    void load() {
        if (options.model_path.empty() || options.codec_path.empty()) {
            fail("model_path and codec_path are required");
        }
        if (!options.backends_dir.empty()) {
            ::tts_cpp::detail::set_backends_directory(options.backends_dir);
        }
        lm = std::make_unique<SpeechLM>(options.model_path, options.use_gpu, options.n_threads);
        codec = std::make_unique<SpeechCodec>(options.codec_path, options.use_gpu, options.n_threads);
        tokenizer = std::make_unique<SpeechTextTokenizer>(lm->tokenizer_tokens(), lm->tokenizer_merges(),
                lm->tokenizer_types());
    }

    float tokens_per_second() const {
        const detail::SpeechVqConfig & vq = codec->tokenizer().config();
        return (float) vq.sample_rate / (float) vq.samples_per_token();
    }

    bool cancelled() const {
        return cancel_requested.load();
    }

    detail::SpeechStop stop_signal() const {
        return [this] { return cancelled(); };
    }

    SpeechTurn turn_of(const SpeechMessage & message) {
        SpeechTurn turn{turn_role(message.role), message.text, {}, !message.audio.empty()};
        if (turn.is_audio) {
            turn.audio_codes = codec->encode(message.audio, message.sample_rate, stop_signal());
        }
        return turn;
    }

    std::vector<SpeechTurn> turns_of(const std::vector<SpeechMessage> & messages) {
        std::vector<SpeechTurn> turns;
        for (size_t i = 0; i < messages.size() && !cancelled(); ++i) {
            turns.push_back(turn_of(messages[i]));
        }
        return turns;
    }

    detail::SpeechLimits limits_of(const SpeechRequest & request, size_t prompt_rows) const {
        const int max_new = detail::speech_generation_budget(prompt_rows, request.max_new_tokens, lm->config().n_ctx_train);
        const int reply_codes = (int) std::ceil(request.max_reply_seconds * tokens_per_second());
        return {max_new, MIN_NEW_TOKENS, reply_codes};
    }

    bool keep_going(const SpeechProgress & progress, int generated, int limit) const {
        return !cancel_requested && (!progress || progress(generated, limit));
    }

    bool generate(detail::SpeechGenerationState & state, detail::SpeechLogits logits, const SpeechRequest & request,
                  const detail::SpeechLimits & limits, const SpeechProgress & progress) {
        return detail::run_speech_generation(*lm, state, std::move(logits), sampling_of(request), request.seed,
                [&](int generated) { return keep_going(progress, generated, limits.max_new_tokens); });
    }

    const detail::SpeechVoice & reply_voice(const SpeechRequest & request, detail::SpeechVoice & custom) {
        if (request.voice.empty()) {
            return codec->default_voice();
        }
        custom = codec->voice(request.voice, request.voice_sample_rate);
        return custom;
    }

    void speak(SpeechResult & result, const std::vector<int32_t> & codes, const SpeechRequest & request) {
        const auto decode_start = std::chrono::steady_clock::now();
        detail::SpeechVoice custom;
        detail::SpeechDecodeOptions decode;
        decode.cancel = &cancel_requested;
        result.pcm = codec->decode(codes, reply_voice(request, custom), decode);
        result.decode_ms = elapsed_ms(decode_start);
        result.cancelled = cancel_requested.load();
    }

    std::vector<SpeechRow> encode_prompt(const SpeechRequest & request, SpeechResult & result) {
        const auto encode_start = std::chrono::steady_clock::now();
        std::vector<SpeechTurn> turns = turns_of(request.messages);
        result.encode_ms = elapsed_ms(encode_start);
        if (cancelled()) {
            return {};
        }
        return detail::speech_prompt(lm->config(), *tokenizer, turns, !request.text_reply);
    }

    detail::SpeechLogits prefill_prompt(const std::vector<SpeechRow> & prompt, const detail::SpeechLimits & limits,
                                        SpeechResult & result) {
        const auto prefill_start = std::chrono::steady_clock::now();
        lm->begin(result.prompt_tokens + limits.max_new_tokens);
        detail::SpeechLogits logits = lm->prefill(prompt, PREFILL_BATCH_TOKENS, stop_signal());
        result.prefill_ms = elapsed_ms(prefill_start);
        return logits;
    }

    static SpeechResult cancelled_result(SpeechResult result) {
        result.cancelled = true;
        return result;
    }

    SpeechResult run(const SpeechRequest & request, const SpeechProgress & progress) {
        SpeechResult result;
        result.sample_rate = codec->sample_rate();
        const std::vector<SpeechRow> prompt = encode_prompt(request, result);
        if (cancelled()) {
            return cancelled_result(result);
        }
        result.prompt_tokens = (int) prompt.size();
        const detail::SpeechLimits limits = limits_of(request, prompt.size());
        detail::SpeechLogits logits = prefill_prompt(prompt, limits, result);
        if (cancelled()) {
            return cancelled_result(result);
        }
        return reply(request, progress, prompt.back(), limits, std::move(logits), result);
    }

    SpeechResult reply(const SpeechRequest & request, const SpeechProgress & progress, const SpeechRow & last_prompt_row,
                       const detail::SpeechLimits & limits, detail::SpeechLogits logits, SpeechResult result) {
        const auto generate_start = std::chrono::steady_clock::now();
        detail::SpeechGenerationState state(lm->config().tokens, last_prompt_row, limits);
        result.cancelled = !generate(state, std::move(logits), request, limits, progress);
        result.generate_ms = elapsed_ms(generate_start);
        result.generated_tokens = (int) state.generated().size();
        result.truncated = state.truncated();
        result.text = detail::speech_reply_text(*tokenizer, state.generated(), lm->config().tokens);
        const std::vector<int32_t> codes = detail::speech_reply_codes(state.generated(), lm->config().tokens);
        result.reply_tokens = (int) codes.size();
        // Each request begins with a fresh prompt; the completed LM's KV and
        // prefill graph buffers are no longer needed during S3Gen/HiFT decode.
        if (options.use_gpu && !result.cancelled && !request.text_reply) {
            // The full-precision LM and S3Gen decoder exceed smaller GPUs when
            // resident together. Keep metadata/backend identity, but reload LM
            // weights at the next begin() after the decoder has been released.
            lm->release_weights();
        } else {
            lm->release_generation();
        }
        if (!result.cancelled && !request.text_reply) {
            speak(result, codes, request);
        }
        return result;
    }

    SpeechResult respond(const SpeechRequest & request, const SpeechProgress & progress) {
        std::unique_lock<std::mutex> lock(respond_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            fail("a response is already in progress on this instance");
        }
        cancel_requested = false;
        detail::validate_speech_request(request);
        codec->tokenizer().begin_run();
        SpeechResult result = run(request, progress);
        result.tokenizer_backend = codec->tokenizer().run_backend();
        return result;
    }
};

SpeechEngine::SpeechEngine(const SpeechOptions & options) : impl_(new Impl) {
    impl_->options = options;
    impl_->load();
}

SpeechEngine::~SpeechEngine() = default;

SpeechResult SpeechEngine::respond(const SpeechRequest & request, const SpeechProgress & progress) {
    return impl_->respond(request, progress);
}

void SpeechEngine::cancel() noexcept {
    impl_->cancel_requested = true;
}

int SpeechEngine::sample_rate() const noexcept {
    return impl_->codec->sample_rate();
}

float SpeechEngine::tokens_per_second() const noexcept {
    return impl_->tokens_per_second();
}

const char * SpeechEngine::backend_name() const noexcept {
    return impl_->lm->backend_name();
}

bool SpeechEngine::tokenizer_on_coreml() const noexcept {
    return impl_->codec->tokenizer().on_coreml();
}

} // namespace tts_cpp::moss
