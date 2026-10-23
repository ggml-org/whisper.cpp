#include "tts-cpp/audio8/engine.h"

#include "audio8/internal.h"
#include "audio8/sampling.h"
#include "audio8/window_stream.h"
#include "backend_selection.h"
#include "backend_util.h"
#include "voice_features.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>

namespace tts_cpp {
namespace audio8 {

using namespace ::tts_cpp::audio8::detail;

namespace {

constexpr int DEFAULT_MAX_FRAMES = AUDIO8_DEFAULT_MAX_FRAMES;
constexpr int DEFAULT_THREADS = 4;

int resolve_threads(int requested) {
    if (requested > 0) return requested;
    const unsigned hardware = std::thread::hardware_concurrency();
    return static_cast<int>(std::min(hardware ? hardware : 4u,
                                     static_cast<unsigned>(DEFAULT_THREADS)));
}

sampling_params resolve_sampling(const EngineOptions & opts) {
    sampling_params params;
    params.greedy = opts.greedy;
    params.temperature = opts.temperature;
    params.top_k = opts.top_k;
    params.top_p = opts.top_p;
    return params;
}

sampling_params narrow_sampling(const lm_hparams & hp) {
    sampling_params params;
    params.temperature = hp.ras_temperature;
    params.top_p = hp.ras_top_p;
    return params;
}

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("audio8: " + message);
}

// Names both sides of a comparison so a mismatch says which two files
// disagree and by how much.
struct agreement {
    const char * left;
    const char * right;

    void on(const char * field, int left_value, int right_value) const {
        require(left_value == right_value,
                std::string(field) + " disagrees: " + left + " says " +
                    std::to_string(left_value) + ", " + right + " says " +
                    std::to_string(right_value));
    }
};

codec_header read_header(const std::string & path, const char * expected_part) {
    codec_header header;
    std::string error;
    if (!peek_codec_header(path, header, &error)) throw std::runtime_error(error);
    require(header.part == expected_part,
            path + " holds the codec's " + header.part + " half, not the " +
                expected_part);
    return header;
}

// The language model emits one code per quantizer, so a half that disagrees
// with its own bank would hand the other side frames of the wrong width.
void check_bank(const codec_hparams & hp, const char * half) {
    require(hp.num_codebooks == hp.residual_codebooks + 1,
            std::string("the codec ") + half + " declares " +
                std::to_string(hp.num_codebooks) + " codebooks but " +
                std::to_string(hp.residual_codebooks) +
                " residual quantizers alongside the semantic one");
}

// Codes cross from the encoder into the prompt and from the language model
// into the decoder, so files from different checkpoints would index one's
// tables with the other's codes.
void check_against_lm(const lm_hparams & lm, const codec_hparams & codec,
                      const char * half) {
    const agreement pair{"the language model", half};
    pair.on("the codebook count", lm.num_codebooks, codec.num_codebooks);
    pair.on("the semantic codebook size", lm.codebook_size, codec.semantic_codebook_size);
}

void check_halves(const codec_hparams & encoder, const codec_hparams & decoder) {
    const agreement pair{"the codec encoder", "the decoder"};
    pair.on("the codebook count", encoder.num_codebooks, decoder.num_codebooks);
    pair.on("the semantic codebook size", encoder.semantic_codebook_size,
            decoder.semantic_codebook_size);
    pair.on("the residual codebook size", encoder.residual_codebook_size,
            decoder.residual_codebook_size);
    pair.on("the residual quantizer count", encoder.residual_codebooks,
            decoder.residual_codebooks);
    pair.on("the codebook width", encoder.codebook_dim, decoder.codebook_dim);
    pair.on("the latent width", encoder.latent_dim, decoder.latent_dim);
    pair.on("the frame size", encoder.frame_size, decoder.frame_size);
    pair.on("the sample rate", encoder.sample_rate, decoder.sample_rate);
}

std::vector<float> at_codec_rate(const VoicePrompt & voice, int codec_rate) {
    require(voice.sample_rate > 0, "the voice prompt has no sample rate");
    if (voice.sample_rate == codec_rate) return voice.pcm;
    return resample_sinc(voice.pcm, voice.sample_rate, codec_rate);
}

// Identifies a voice prompt well enough to skip re-encoding the same one, and
// cheaply enough not to matter next to the codec pass it saves.
uint64_t fingerprint(const VoicePrompt & voice) {
    uint64_t hash = 1469598103934665603ull;
    const auto absorb = [&hash](const void * data, size_t bytes) {
        const unsigned char * cursor = static_cast<const unsigned char *>(data);
        for (size_t index = 0; index < bytes; ++index) {
            hash = (hash ^ cursor[index]) * 1099511628211ull;
        }
    };
    absorb(voice.pcm.data(), voice.pcm.size() * sizeof(float));
    absorb(voice.transcript.data(), voice.transcript.size());
    absorb(&voice.sample_rate, sizeof(voice.sample_rate));
    return hash;
}

}  // namespace

struct Engine::Impl {
    EngineOptions opts;
    lm_model lm;
    codec_model decoder;
    codec_model encoder;
    bool has_encoder = false;
    std::unique_ptr<Tokenizer> tokenizer;
    sampling_params sampling;
    std::atomic<bool> cancel_requested{false};
    StageTimings timings;

    std::vector<int32_t> cached_codes;
    int cached_frames = 0;
    uint64_t cached_voice = 0;
    bool has_cached_voice = false;

    ~Impl() {
        free_lm(lm);
        free_codec(decoder);
        free_codec(encoder);
    }

    bool wants_cancel() const {
        return cancel_requested.load(std::memory_order_relaxed);
    }

    void check_cancel() const {
        if (wants_cancel()) throw std::runtime_error(CANCELLED);
    }

    // What the codec asks between blocks, so a long pass stops with the rest
    // of the pipeline rather than running to the end.
    cancel_hook cancel_probe() const {
        return [this] { return wants_cancel(); };
    }

    int frame_budget(int prompt_width) const {
        const int requested = opts.max_frames > 0 ? opts.max_frames : DEFAULT_MAX_FRAMES;
        return std::min(requested, lm.hp.max_seq_len - prompt_width);
    }

    // The reference's reference_codes: [num_codebooks, frames] row-major.
    const std::vector<int32_t> & reference_codes(const VoicePrompt & voice,
                                                 int n_threads, int & frames) {
        require(has_encoder,
                "cloning needs codec_encoder_gguf_path, which was not configured");
        const uint64_t key = fingerprint(voice);
        if (has_cached_voice && cached_voice == key) {
            frames = cached_frames;
            return cached_codes;
        }
        const std::vector<float> pcm = at_codec_rate(voice, encoder.hp.sample_rate);
        require(!pcm.empty(), "the voice prompt is empty after resampling");
        std::string error;
        stage_timer measure(timings.voice_encode_ms);
        if (!encode_audio(encoder, pcm.data(), static_cast<int>(pcm.size()), n_threads,
                          cancel_probe(), cached_codes, cached_frames, &error)) {
            has_cached_voice = false;
            throw std::runtime_error(error);
        }
        cached_voice = key;
        has_cached_voice = true;
        frames = cached_frames;
        return cached_codes;
    }

    // The reference encode is timed as its own stage, so it runs before the
    // prompt timer opens and the two stages stay disjoint.
    prompt_frames prompt_for(const std::string & text, const VoicePrompt & voice,
                             int n_threads) {
        const bool cloning = !voice.empty();
        int frames = 0;
        const std::vector<int32_t> empty;
        const std::vector<int32_t> & codes =
            cloning ? reference_codes(voice, n_threads, frames) : empty;

        stage_timer measure(timings.prompt_ms);
        const PromptSegments segments =
            build_prompt(*tokenizer, text, voice.transcript, cloning);
        if (!cloning) return build_frames(lm.hp, segments, {}, 0);
        require(codes.size() ==
                    static_cast<size_t>(lm.hp.num_codebooks) * static_cast<size_t>(frames),
                "the encoder returned " + std::to_string(codes.size()) +
                    " codes for " + std::to_string(frames) + " frames, not the " +
                    std::to_string(lm.hp.num_codebooks) + " rows the prompt needs");
        return build_frames(lm.hp, segments, codes, frames);
    }

    // Semantic logits arrive already restricted to the rows the sampler may
    // pick: codebook_size semantic values followed by EOS. Index and codebook
    // value therefore coincide, and EOS is the one index past the end.
    bool is_eos(int index) const { return index == lm.hp.codebook_size; }

    void run_slow(const int32_t * frames, int width, int n_past, int n_threads,
                  double & sink, std::vector<float> & sem_logits,
                  std::vector<float> & fast_input) {
        stage_timer measure(sink);
        std::string error;
        if (!slow_step(lm, frames, width, n_past, n_threads, sem_logits, fast_input,
                       &error)) {
            throw std::runtime_error(error);
        }
    }

    // Greedy expands a whole frame in one graph, but only where the backend can
    // pick the codes itself; every other picker has to come back to the host
    // between codebooks, so it keeps the per-position path.
    void run_fast(const std::vector<float> & fast_input, int semantic, int n_threads,
                  const code_picker & pick, std::vector<int32_t> & frame) {
        stage_timer measure(timings.fast_decode_ms);
        std::string error;
        const bool chained = sampling.greedy && lm.picks_codes;
        const bool ok = chained
                            ? fast_frame(lm, fast_input, semantic, n_threads, frame, &error)
                            : fast_step(lm, fast_input, semantic, n_threads, pick, frame,
                                        &error);
        if (!ok) throw std::runtime_error(error);
    }

    int pick_semantic(RepetitionAwareSampler & sampler, const std::vector<float> & logits,
                      std::mt19937 & rng) {
        stage_timer measure(timings.sample_ms);
        return sampler.pick(logits, sampling, rng);
    }

    void offer_to_codec(window_stream * stream, int n_frames) {
        if (!stream) return;
        std::string error;
        if (!stream->advance(n_frames, &error)) throw std::runtime_error(error);
    }

    void generate_codes(const prompt_frames & prompt, int n_threads, window_stream * stream,
                        std::vector<int32_t> & codes, int & n_frames) {
        require(prompt.width < lm.hp.max_seq_len,
                "the prompt is " + std::to_string(prompt.width) +
                    " frames, which does not fit the model's context of " +
                    std::to_string(lm.hp.max_seq_len));
        const int budget = frame_budget(prompt.width);
        std::mt19937 rng(static_cast<uint32_t>(opts.seed));
        RepetitionAwareSampler semantic_sampler(lm.hp.ras_window, 0,
                                                lm.hp.codebook_size - 1,
                                                narrow_sampling(lm.hp));
        const code_picker pick = [&](const std::vector<float> & logits, int) {
            return sample_token(logits, sampling, rng);
        };

        std::vector<float> sem_logits;
        std::vector<float> fast_input;
        run_slow(prompt.values.data(), prompt.width, 0, n_threads, timings.prefill_ms,
                 sem_logits, fast_input);

        codes.clear();
        std::vector<int32_t> frame;
        std::vector<int32_t> column(lm.hp.num_codebooks + 1);
        n_frames = 0;
        for (int step = 0; step < budget; ++step) {
            check_cancel();
            const int index = pick_semantic(semantic_sampler, sem_logits, rng);
            if (is_eos(index)) break;

            const int semantic = lm.hp.semantic_begin + index;
            run_fast(fast_input, semantic, n_threads, pick, frame);
            codes.insert(codes.end(), frame.begin(), frame.end());
            ++n_frames;
            offer_to_codec(stream, n_frames);
            if (step + 1 == budget) break;

            column[0] = semantic;
            std::copy(frame.begin(), frame.end(), column.begin() + 1);
            run_slow(column.data(), 1, prompt.width + step, n_threads,
                     timings.slow_decode_ms, sem_logits, fast_input);
        }
    }

    bool decode_frames(window_stream * stream, const std::vector<int32_t> & frames,
                       int n_frames, int n_threads, std::vector<float> & pcm,
                       decode_timing & timing, std::string & error) {
        const std::vector<int32_t> rows =
            codebook_rows(frames, lm.hp.num_codebooks, 0, n_frames);
        if (stream) {
            return finish_synthesis_stream(decoder, *stream, rows.data(), n_frames, n_threads,
                                           cancel_probe(), pcm, &error, &timing);
        }
        return decode_codes(decoder, rows.data(), n_frames, n_threads, cancel_probe(), pcm,
                            &error, nullptr, &timing);
    }

    void run_codec(window_stream * stream, const std::vector<int32_t> & frames, int n_frames,
                   int n_threads, SynthesisResult & result) {
        decode_timing timing;
        std::string error;
        if (!decode_frames(stream, frames, n_frames, n_threads, result.pcm, timing, error)) {
            throw std::runtime_error(error);
        }
        result.codec_synthesis_backend = timing.synthesis_backend;
        timings.codec_latent_ms = timing.latent_ms;
        timings.codec_synth_ms = timing.synthesis_ms;
        if (opts.verbose) {
            std::fprintf(stderr,
                         "[audio8-timing] codec block %d frames, %.0f MB scratch, synthesis on "
                         "%s, %d windows during generation\n",
                         timing.block_frames, timing.block_scratch / (1024.0 * 1024.0),
                         timing.synthesis_backend.c_str(), timing.streamed_windows);
        }
    }

    // resample_for_output is a passthrough at the native rate, and a stage that
    // did nothing has to report nothing, so the timer only covers real work.
    void resample(std::vector<float> & pcm, int native) {
        const int target = opts.output_sample_rate;
        if (target <= 0 || target == native) return;
        stage_timer measure(timings.resample_ms);
        pcm = resample_for_output(std::move(pcm), native, target);
    }

    std::vector<float> resample_chunk(OutputResampler & resampler, const std::vector<float> & pcm) {
        if (resampler.passthrough()) return pcm;
        stage_timer measure(timings.resample_ms);
        return resampler.process(pcm);
    }

    std::vector<float> finish_resampling(OutputResampler & resampler) {
        if (resampler.passthrough()) return {};
        stage_timer measure(timings.resample_ms);
        return resampler.finish();
    }

    SynthesisResult run(const std::string & text, const VoicePrompt & voice,
                        const AudioCallback & callback = {}) {
        timings = StageTimings();
        SynthesisResult result;
        {
            stage_timer whole(timings.total_ms);
            result = generate(text, voice, callback);
        }
        result.timings = timings;
        if (opts.verbose) report_timings(result);
        return result;
    }

    SynthesisResult generate(const std::string & text, const VoicePrompt & voice,
                             const AudioCallback & callback) {
        require(!text.empty(), "the text to speak must not be empty");
        cancel_requested.store(false, std::memory_order_relaxed);
        const int n_threads = resolve_threads(opts.n_threads);

        const prompt_frames prompt = prompt_for(text, voice, n_threads);
        std::vector<int32_t> frames;
        int n_frames = 0;
        OutputResampler resampler(decoder.hp.sample_rate, opts.output_sample_rate);
        std::vector<float> delivered;
        bool received = false;
        const auto deliver = [&](const std::vector<float> & pcm) {
            if (pcm.empty()) return;
            check_cancel();
            if (!callback(pcm.data(), pcm.size(), resampler.target_rate())) {
                cancel_requested.store(true, std::memory_order_relaxed);
                check_cancel();
            }
            delivered.insert(delivered.end(), pcm.begin(), pcm.end());
        };
        std::function<void(const std::vector<float> &)> emit;
        if (callback) {
            emit = [&](const std::vector<float> & pcm) {
                received = true;
                deliver(resample_chunk(resampler, pcm));
            };
        }
        const std::unique_ptr<window_stream> stream =
            open_synthesis_stream(decoder, frames, n_threads, cancel_probe(), emit);
        generate_codes(prompt, n_threads, stream.get(), frames, n_frames);
        require(n_frames > 0, "the model emitted no audio frames");
        check_cancel();

        SynthesisResult result;
        result.frames = n_frames;
        result.codes.assign(frames.begin(), frames.end());
        run_codec(stream.get(), frames, n_frames, n_threads, result);

        const int native = decoder.hp.sample_rate;
        if (callback) {
            if (!received) emit(result.pcm);
            deliver(finish_resampling(resampler));
            result.pcm = std::move(delivered);
        } else {
            resample(result.pcm, native);
        }
        result.sample_rate =
            opts.output_sample_rate > 0 ? opts.output_sample_rate : native;
        result.duration_s =
            static_cast<float>(result.pcm.size()) / static_cast<float>(result.sample_rate);
        return result;
    }

    static void report_stages(const StageTimings & t) {
        const std::pair<const char *, double> stages[] = {
            {"voice-encode", t.voice_encode_ms}, {"prompt", t.prompt_ms},
            {"prefill", t.prefill_ms},           {"sample", t.sample_ms},
            {"fast-decode", t.fast_decode_ms},   {"slow-decode", t.slow_decode_ms},
            {"codec-latent", t.codec_latent_ms}, {"codec-synth", t.codec_synth_ms},
            {"resample", t.resample_ms},
        };
        for (const std::pair<const char *, double> & stage : stages) {
            std::fprintf(stderr, "[audio8-timing] %-14s %8.1f ms\n", stage.first,
                         stage.second);
        }
    }

    void report_timings(const SynthesisResult & result) const {
        const StageTimings & t = result.timings;
        report_stages(t);
        const double realtime =
            t.total_ms > 0.0 ? 1000.0 * result.duration_s / t.total_ms : 0.0;
        std::fprintf(stderr,
                     "[audio8-timing] %-14s %8.1f ms for %d frames, %.2f s audio "
                     "(%.2fx realtime)\n",
                     "total", t.total_ms, result.frames, result.duration_s, realtime);
    }
};

Engine::Engine(const EngineOptions & opts) : pimpl_(new Impl()) {
    pimpl_->opts = opts;
    validate_output_sample_rate(opts.output_sample_rate, "audio8");
    if (!opts.backends_dir.empty()) {
        ::tts_cpp::detail::set_backends_directory(opts.backends_dir);
    }
    // EngineOptions::backend is threaded through load_lm / load_codec /
    // init_backend. "" / "auto" takes the tier-based GPU selection honouring
    // n_gpu_layers; "cpu" short-circuits to the CPU backend; "hexagon" or an
    // exact ggml device name (e.g. "HTP0", "CUDA0") selects that device, with
    // init_backend returning nullptr and emitting a one-line stderr note on
    // an unresolved name so the caller sees a clean "failed to init a
    // compute backend" error.
    require(!opts.lm_gguf_path.empty(), "lm_gguf_path is required");
    require(!opts.codec_decoder_gguf_path.empty(), "codec_decoder_gguf_path is required");

    // Headers first: a set of GGUFs that does not belong together is rejected
    // before a gigabyte of weights is read.
    const bool cloning = !opts.codec_encoder_gguf_path.empty();
    const codec_header decoder = read_header(opts.codec_decoder_gguf_path, "decoder");
    check_bank(decoder.hp, "decoder");
    codec_header encoder;
    if (cloning) {
        encoder = read_header(opts.codec_encoder_gguf_path, "encoder");
        check_bank(encoder.hp, "encoder");
        check_halves(encoder.hp, decoder.hp);
    }

    std::string error;
    if (!load_lm(opts.lm_gguf_path, opts.n_gpu_layers, opts.backend, pimpl_->lm, &error)) {
        throw std::runtime_error(error);
    }
    const lm_hparams & lm = pimpl_->lm.hp;
    check_against_lm(lm, decoder.hp, "the codec decoder");
    if (cloning) check_against_lm(lm, encoder.hp, "the codec encoder");

    if (!load_codec(opts.codec_decoder_gguf_path, opts.n_gpu_layers, opts.backend,
                    pimpl_->decoder, &error)) {
        throw std::runtime_error(error);
    }
    if (opts.verbose) {
        std::fprintf(stderr, "[audio8] codec synthesis on %s\n",
                     pimpl_->decoder.synthesis_on_coreml ? "the Core ML sidecar"
                                                         : ggml_backend_name(pimpl_->decoder.backend));
    }
    if (cloning) {
        if (!load_codec(opts.codec_encoder_gguf_path, opts.n_gpu_layers, opts.backend,
                        pimpl_->encoder,
                        &error)) {
            throw std::runtime_error(error);
        }
        pimpl_->has_encoder = true;
    }

    pimpl_->tokenizer.reset(new Tokenizer(pimpl_->lm.tokenizer));
    pimpl_->sampling = resolve_sampling(opts);
}

Engine::~Engine() = default;
Engine::Engine(Engine &&) noexcept = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

SynthesisResult Engine::synthesize(const std::string & text) {
    return pimpl_->run(text, VoicePrompt{});
}

SynthesisResult Engine::synthesize(const std::string & text, const VoicePrompt & voice) {
    return pimpl_->run(text, voice);
}

SynthesisResult Engine::synthesize_stream(const std::string & text, const AudioCallback & callback) {
    return synthesize_stream(text, VoicePrompt{}, callback);
}

SynthesisResult Engine::synthesize_stream(const std::string & text, const VoicePrompt & voice,
                                          const AudioCallback & callback) {
    require(static_cast<bool>(callback), "the audio callback must not be empty");
    return pimpl_->run(text, voice, callback);
}

void Engine::cancel() {
    pimpl_->cancel_requested.store(true, std::memory_order_relaxed);
}

const EngineOptions & Engine::options() const { return pimpl_->opts; }

int Engine::sample_rate() const {
    return pimpl_->opts.output_sample_rate > 0 ? pimpl_->opts.output_sample_rate
                                               : pimpl_->decoder.hp.sample_rate;
}

std::string Engine::backend_name() const {
    if (!pimpl_->lm.backend) return "(unknown)";
    return ggml_backend_name(pimpl_->lm.backend);
}

BackendDevice Engine::backend_device() const {
    return ::tts_cpp::detail::backend_is_cpu(pimpl_->lm.backend) ? BackendDevice::CPU
                                                                 : BackendDevice::GPU;
}

bool Engine::codec_on_coreml() const {
    return pimpl_->decoder.synthesis_on_coreml;
}

VoicePrompt load_voice_prompt(const std::string & wav_path,
                              const std::string & transcript) {
    VoicePrompt voice;
    if (!wav_load(wav_path, voice.pcm, voice.sample_rate)) {
        throw std::runtime_error("audio8: cannot read the reference wav at " + wav_path);
    }
    voice.transcript = transcript;
    return voice;
}

SynthesisResult synthesize(const EngineOptions & opts, const std::string & text) {
    Engine engine(opts);
    return engine.synthesize(text);
}

}  // namespace audio8
}  // namespace tts_cpp
