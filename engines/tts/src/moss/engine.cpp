#include "tts-cpp/moss/engine.h"

#include "moss/codec.h"
#include "moss/delay_lm.h"
#include "moss/frontend.h"
#include "moss/generation.h"
#include "moss/reference_wav.h"

#include "backend_selection.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <random>
#include <stdexcept>

namespace tts_cpp::moss {
namespace {

using detail::AudioSegments;
using detail::Codec;
using detail::DelayLM;
using detail::DelayLogits;
using detail::DelayRow;
using detail::DelayState;
using detail::Frontend;
using detail::SamplingConfig;
using detail::read_reference_wav;
using detail::require_reference_total;

constexpr int CONTEXT_HEADROOM = 8;
constexpr int SINGLE_DECODE_MAX_FRAMES = 750;
constexpr int TERMINATION_ROWS = 2;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss engine: " + message);
}

double elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

} // namespace

struct Engine::Impl {
    EngineOptions options;
    SamplingConfig sampling;
    std::unique_ptr<DelayLM> backbone;
    std::unique_ptr<Frontend> frontend;
    std::unique_ptr<Codec> decoder;
    detail::PromptAudio prompt_audio;
    int continuation_frames = 0;
    std::atomic<bool> cancel_requested{false};
    std::mutex synthesis_mutex;

    void load() {
        if (options.backbone_path.empty() || options.decoder_path.empty()) {
            fail("backbone_path and decoder_path are required");
        }
        if (options.max_new_tokens < 1) {
            fail("max_new_tokens must be positive");
        }
        if (options.duration_tokens < 0) {
            fail("duration_tokens must not be negative");
        }
        if (!options.reference_audio_path.empty() && !options.dialogue_reference_paths.empty()) {
            fail("reference_audio_path and dialogue_reference_paths are exclusive");
        }
        if (!options.backends_dir.empty()) {
            ::tts_cpp::detail::set_backends_directory(options.backends_dir);
        }
        sampling.text_temperature  = options.text_temperature;
        sampling.text_top_p        = options.text_top_p;
        sampling.text_top_k        = options.text_top_k;
        sampling.audio_temperature = options.audio_temperature;
        sampling.audio_top_p       = options.audio_top_p;
        sampling.audio_top_k       = options.audio_top_k;
        sampling.audio_repetition_penalty = options.audio_repetition_penalty;
        backbone.reset(new DelayLM(options.backbone_path, options.use_gpu, options.n_threads,
                options.context));
        frontend.reset(new Frontend(*backbone));
        validate_duration_budget();
        // Reference encoding only needs the backbone's channel count. Finish
        // and destroy its temporary encoder before loading decoder weights so
        // cloning does not require both codec checkpoints resident at once.
        load_reference();
        decoder.reset(new Codec(options.decoder_path, options.use_gpu, options.n_threads));
        if (decoder->is_encoder()) {
            fail("decoder_path points at an encoder checkpoint");
        }
        if (decoder->num_quantizers() < backbone->config().n_vq) {
            fail("the decoder has fewer quantizers than the backbone channels");
        }
    }

    void validate_duration_budget() const {
        if (options.duration_tokens == 0) {
            return;
        }
        const int needed = options.duration_tokens + backbone->config().n_vq - 1 + TERMINATION_ROWS;
        if (needed > options.max_new_tokens) {
            fail("duration_tokens needs " + std::to_string(needed) +
                 " generation steps with the delay drain; raise max_new_tokens");
        }
    }

    void load_reference() {
        if (options.reference_audio_path.empty() && options.dialogue_reference_paths.empty()) {
            return;
        }
        if (options.encoder_path.empty()) {
            fail("voice cloning needs encoder_path next to the reference audio");
        }
        Codec encoder(options.encoder_path, options.use_gpu, options.n_threads);
        if (!encoder.is_encoder()) {
            fail("encoder_path points at a decoder checkpoint");
        }
        if (encoder.num_quantizers() < backbone->config().n_vq) {
            fail("the encoder has fewer quantizers than the backbone channels");
        }
        if (!options.reference_audio_path.empty()) {
            encode_single_reference(encoder);
        } else {
            encode_dialogue_references(encoder);
        }
    }

    std::vector<int32_t> truncate_channels(const std::vector<int32_t> & codes, int from, int to) {
        std::vector<int32_t> narrowed;
        narrowed.reserve(codes.size() / (size_t) from * (size_t) to);
        for (size_t frame = 0; frame < codes.size() / (size_t) from; ++frame) {
            narrowed.insert(narrowed.end(), codes.begin() + (std::ptrdiff_t) (frame * (size_t) from),
                    codes.begin() + (std::ptrdiff_t) (frame * (size_t) from + (size_t) to));
        }
        return narrowed;
    }

    std::vector<int32_t> encode_checked(Codec & encoder, const std::vector<float> & pcm) {
        std::vector<int32_t> codes = encoder.encode(pcm);
        if (codes.empty()) {
            fail("reference audio is shorter than one codec frame");
        }
        const int n_vq = backbone->config().n_vq;
        if (encoder.num_quantizers() != n_vq) {
            codes = truncate_channels(codes, encoder.num_quantizers(), n_vq);
        }
        return codes;
    }

    void encode_single_reference(Codec & encoder) {
        const std::vector<float> pcm = read_reference_wav(options.reference_audio_path,
                encoder.sample_rate());
        prompt_audio.speaker_codes.push_back(encode_checked(encoder, pcm));
    }

    std::vector<float> read_dialogue_references(Codec & encoder) {
        std::vector<float> concatenated;
        for (const std::string & path : options.dialogue_reference_paths) {
            const std::vector<float> pcm = read_reference_wav(path, encoder.sample_rate());
            prompt_audio.speaker_codes.push_back(encode_checked(encoder, pcm));
            concatenated.insert(concatenated.end(), pcm.begin(), pcm.end());
            require_reference_total(concatenated.size(), encoder.sample_rate());
        }
        return concatenated;
    }

    void encode_dialogue_references(Codec & encoder) {
        const std::vector<float> concatenated = read_dialogue_references(encoder);
        prompt_audio.continuation_codes = encode_checked(encoder, concatenated);
        continuation_frames = (int) (prompt_audio.continuation_codes.size() /
                (size_t) backbone->config().n_vq);
    }

    void generate_rows(const std::vector<DelayRow> & prompt, DelayState & state,
                       std::mt19937 & rng, SynthesisResult & result,
                       const std::function<bool()> & after_step) {
        DelayLogits logits = backbone->prefill(prompt);
        for (int step = 0; step < options.max_new_tokens; ++step) {
            if (cancel_requested) {
                result.cancelled = true;
                return;
            }
            const DelayRow row = state.step(logits, sampling, rng);
            result.generated_frames += 1;
            if (after_step && !after_step()) {
                result.cancelled = true;
                return;
            }
            if (state.stopping()) {
                return;
            }
            logits = backbone->step(row);
        }
    }

    struct StreamProgress {
        std::vector<int32_t> pending;
        int scanned_frames = 0;
        size_t context_samples = 0;
        bool in_segment = false;
        bool audio_emitted = false;
    };

    size_t context_sample_count() const {
        return (size_t) continuation_frames * (size_t) decoder->samples_per_frame();
    }

    static std::vector<float> drop_context(std::vector<float> pcm, size_t & context_samples) {
        const size_t dropped = std::min(context_samples, pcm.size());
        pcm.erase(pcm.begin(), pcm.begin() + (std::ptrdiff_t) dropped);
        context_samples -= dropped;
        return pcm;
    }

    int pending_frames(const StreamProgress & progress) const {
        return (int) (progress.pending.size() / (size_t) backbone->config().n_vq);
    }

    std::vector<int32_t> take_pending(StreamProgress & progress, int frames) const {
        const size_t count = (size_t) frames * (size_t) backbone->config().n_vq;
        std::vector<int32_t> taken(progress.pending.begin(),
                progress.pending.begin() + (std::ptrdiff_t) count);
        progress.pending.erase(progress.pending.begin(), progress.pending.begin() + (std::ptrdiff_t) count);
        return taken;
    }

    bool deliver(std::vector<float> decoded, StreamProgress & progress, const AudioCallback & callback,
                 const std::chrono::steady_clock::time_point & start, SynthesisResult & result) {
        const std::vector<float> pcm = drop_context(std::move(decoded), progress.context_samples);
        if (pcm.empty()) {
            return true;
        }
        if (!progress.audio_emitted) {
            progress.audio_emitted = true;
            result.first_audio_ms = elapsed_ms(start);
        }
        return callback(pcm.data(), pcm.size(), decoder->sample_rate());
    }

    bool decode_pending(StreamProgress & progress, int frames, const AudioCallback & callback,
                        const std::chrono::steady_clock::time_point & start, SynthesisResult & result) {
        const std::vector<int32_t> codes = take_pending(progress, frames);
        const auto decode_start = std::chrono::steady_clock::now();
        std::vector<float> pcm = decoder->decode_stream(codes);
        result.decode_ms += elapsed_ms(decode_start);
        return deliver(std::move(pcm), progress, callback, start, result);
    }

    int stream_piece_frames() const {
        return std::max(1, options.stream_chunk_frames);
    }

    bool emit_full_chunks(StreamProgress & progress, const AudioCallback & callback,
                          const std::chrono::steady_clock::time_point & start, SynthesisResult & result) {
        const int chunk = stream_piece_frames();
        while (pending_frames(progress) >= chunk) {
            if (!decode_pending(progress, chunk, callback, start, result)) {
                return false;
            }
        }
        return true;
    }

    bool close_segment(StreamProgress & progress, const AudioCallback & callback,
                       const std::chrono::steady_clock::time_point & start, SynthesisResult & result) {
        progress.in_segment = false;
        const int remaining = pending_frames(progress);
        return remaining == 0 || decode_pending(progress, remaining, callback, start, result);
    }

    bool consume_frame(const std::vector<int32_t> & frame, StreamProgress & progress,
                       const AudioCallback & callback,
                       const std::chrono::steady_clock::time_point & start, SynthesisResult & result) {
        const int n_vq = backbone->config().n_vq;
        if (detail::frame_is_pad(frame, 0, n_vq, backbone->config().audio_pad_code)) {
            return !progress.in_segment || close_segment(progress, callback, start, result);
        }
        if (!progress.in_segment) {
            decoder->begin_decode_stream(n_vq, stream_piece_frames());
            progress.in_segment = true;
        }
        progress.pending.insert(progress.pending.end(), frame.begin(), frame.end());
        return emit_full_chunks(progress, callback, start, result);
    }

    bool stream_ready_frames(const DelayState & state, int base_row, StreamProgress & progress,
                             const AudioCallback & callback,
                             const std::chrono::steady_clock::time_point & start, SynthesisResult & result) {
        const int available = state.available_frames(base_row);
        for (; progress.scanned_frames < available; ++progress.scanned_frames) {
            if (!consume_frame(state.frame_codes(base_row, progress.scanned_frames), progress,
                    callback, start, result)) {
                return false;
            }
        }
        return true;
    }

    void decode_batch(const AudioSegments & segments, SynthesisResult & result) {
        const auto decode_start = std::chrono::steady_clock::now();
        result.pcm = detail::decode_after_context(*decoder, segments, continuation_frames,
                SINGLE_DECODE_MAX_FRAMES, backbone->config().n_vq);
        result.decode_ms = elapsed_ms(decode_start);
    }

    std::vector<DelayRow> build_checked_prompt(const std::string & text) {
        const std::vector<DelayRow> prompt = frontend->build_prompt(backbone->config(), text,
                options.language, options.duration_tokens, prompt_audio);
        if ((int) prompt.size() + options.max_new_tokens + CONTEXT_HEADROOM > backbone->context()) {
            fail("prompt plus max_new_tokens exceeds the context; raise the context option");
        }
        return prompt;
    }

    int delayed_stream_base(const std::vector<DelayRow> & prompt) const {
        return (int) prompt.size() - continuation_frames;
    }

    SynthesisResult run(const std::string & text) {
        std::unique_lock<std::mutex> lock(synthesis_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            fail("synthesis already in progress on this instance");
        }
        if (text.empty()) {
            fail("text must not be empty");
        }
        cancel_requested = false;
        decoder->begin_run();
        SynthesisResult result = synthesize_locked(text);
        result.codec_backend = decoder->run_backend();
        return result;
    }

    SynthesisResult synthesize_locked(const std::string & text) {
        const std::vector<DelayRow> prompt = build_checked_prompt(text);
        std::mt19937 rng(options.seed);
        DelayState state(backbone->config(), prompt, frontend->tokens().pad,
                frontend->tokens().im_end);
        SynthesisResult result;

        const auto generation_start = std::chrono::steady_clock::now();
        backbone->reset();
        generate_rows(prompt, state, rng, result, nullptr);
        result.generation_ms = elapsed_ms(generation_start);
        if (result.cancelled) {
            return result;
        }

        const AudioSegments segments = state.generated_audio(delayed_stream_base(prompt));
        if (cancel_requested) {
            result.cancelled = true;
            return result;
        }
        decode_batch(segments, result);
        if (result.pcm.empty()) {
            fail("the model produced no audio frames");
        }
        result.sample_rate = decoder->sample_rate();
        result.cancelled = cancel_requested;
        return result;
    }

    SynthesisResult run_stream(const std::string & text, const AudioCallback & callback) {
        std::unique_lock<std::mutex> lock(synthesis_mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            fail("synthesis already in progress on this instance");
        }
        if (text.empty()) {
            fail("text must not be empty");
        }
        if (!callback) {
            fail("streaming synthesis needs a callback");
        }
        cancel_requested = false;
        decoder->begin_run();
        SynthesisResult result = stream_locked(text, callback);
        result.codec_backend = decoder->run_backend();
        return result;
    }

    SynthesisResult stream_locked(const std::string & text, const AudioCallback & callback) {
        const std::vector<DelayRow> prompt = build_checked_prompt(text);
        const int stream_base = delayed_stream_base(prompt);
        std::mt19937 rng(options.seed);
        DelayState state(backbone->config(), prompt, frontend->tokens().pad,
                frontend->tokens().im_end);
        SynthesisResult result;
        StreamProgress progress;
        progress.context_samples = context_sample_count();

        const auto start = std::chrono::steady_clock::now();
        backbone->reset();
        generate_rows(prompt, state, rng, result, [&]() {
            return stream_ready_frames(state, stream_base, progress, callback, start, result);
        });
        if (!result.cancelled) {
            const bool delivered =
                    stream_ready_frames(state, stream_base, progress, callback, start, result) &&
                    close_segment(progress, callback, start, result);
            result.cancelled = !delivered;
        }
        result.generation_ms = elapsed_ms(start) - result.decode_ms;
        result.sample_rate = decoder->sample_rate();
        result.cancelled = result.cancelled || cancel_requested;
        if (!result.cancelled && !progress.audio_emitted) {
            fail("the model produced no audio frames");
        }
        return result;
    }
};

Engine::Engine(const EngineOptions & options) : impl_(new Impl) {
    impl_->options = options;
    impl_->load();
}

Engine::~Engine() = default;

SynthesisResult Engine::synthesize(const std::string & text) {
    return impl_->run(text);
}

SynthesisResult Engine::synthesize_stream(const std::string & text, const AudioCallback & callback) {
    return impl_->run_stream(text, callback);
}

void Engine::cancel() noexcept {
    impl_->cancel_requested = true;
}

int Engine::sample_rate() const noexcept {
    return impl_->decoder->sample_rate();
}

const char * Engine::backend_name() const noexcept {
    return impl_->backbone->backend_name();
}

bool Engine::codec_on_coreml() const noexcept {
    return impl_->decoder->on_coreml();
}

} // namespace tts_cpp::moss
