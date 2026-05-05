// Engine: GGUF load, transcribe, streaming sessions, diarization, timing and options.

#include "parakeet/engine.h"
#include "parakeet/streaming.h"
#include "parakeet/diarization.h"
#include "parakeet/attributed.h"

#include "parakeet_ctc.h"
#include "parakeet_tdt.h"
#include "parakeet_eou.h"
#include "parakeet_sortformer.h"
#include "mel_preprocess.h"
#include "sentencepiece_bpe.h"
#include "energy_vad.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace parakeet {

namespace {

// Encoder frame stride in milliseconds, derived from the GGUF's mel hop
// length, encoder subsampling factor and sample rate. All shipped models
// happen to land at 80 ms (16 kHz x hop=160 x sub=8) but new GGUFs may
// differ -- e.g. a 24 kHz checkpoint or a 4x subsampling variant.
inline double encoder_frame_stride_ms(const ParakeetCtcModel & model) {
    const int hop = model.mel_cfg.hop_length;
    const int sub = model.encoder_cfg.subsampling_factor > 0
                  ? model.encoder_cfg.subsampling_factor : 8;
    const int sr  = model.mel_cfg.sample_rate > 0 ? model.mel_cfg.sample_rate : 16000;
    return 1000.0 * (double) (hop * sub) / (double) sr;
}

double ms_since(std::chrono::steady_clock::time_point a) {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now() - a).count() / 1000.0;
}

}

struct Engine::Impl {
    EngineOptions       opts;
    ParakeetCtcModel    model;
    std::atomic<bool>   cancel_flag{false};

    TdtRuntimeWeights   tdt_rt;
    bool                tdt_ready = false;

    EouRuntimeWeights   eou_rt;
    bool                eou_ready = false;

    bool                     sortformer_ready = false;

    // Reusable mel preprocess scratch buffers. Engine APIs are
    // documented as single-threaded per-instance (see engine.h
    // `Engine::transcribe_*` notes), so a single state member is
    // sufficient. StreamSession holds its own MelState (see below)
    // because the encoder + decoder pipelines run independently.
    MelState            mel_state;

    Impl() = default;
};

// Opt-in encoder prewarm. Runs one synthetic
// forward pass through the encoder so the cold-graph-build cost is
// amortised into Engine construction instead of warmup_1.
//
// What this catches across backends:
//   * Metal: triggers MSL → MTLPipelineState compile.
//   * OpenCL: triggers clBuildProgram for every kernel variant the
//             encoder graph touches; binaries get cached via the
//             program-binary-cache patch when GGML_OPENCL_CACHE_DIR 
//             is set, so subsequent processes skip even this prewarm cost.
//   * Vulkan: triggers vkCreateGraphicsPipelines.
//   * CUDA: triggers cuGraphInstantiate.
//   * CPU: pre-builds the ggml graph nodes + scratch + caches them
//          via the same encoder_graphs LRU as a real call.
//
// Mel input is all-zero (filterbank output for a silent buffer is
// the model's mel_floor / log_zero_guard, but for warmup we just
// need any valid-shape input that flows through every node — zeros
// are fine; log-mel of true zeros is effectively `log(eps)`, no NaN
// risk because compute_log_mel + run_encoder both apply the model's
// log_zero_guard from the GGUF metadata).
//
// Shape: `prewarm_audio_seconds * sample_rate` samples mapped to
// `(prewarm_audio_seconds * sr / hop)` mel frames. Encoder cache
// is shape-keyed on `(T_mel, layers, all_valid)`, so a real call
// with a different T_mel will trigger a fresh graph build — but on
// Metal/OpenCL/Vulkan the *kernel pipeline cache* is keyed on
// kernel signature, not graph shape, so prewarm with any shape
// still warms the relevant compile cost.
//
// Cost: typically 50-300 ms on the first construction; subsequent
// constructions in the same process land near zero (encoder cache
// + GPU pipeline cache hit).
static void prewarm_encoder(ParakeetCtcModel & model, float audio_seconds) {
    if (audio_seconds <= 0.0f) audio_seconds = 1.0f;

    const int sr  = model.mel_cfg.sample_rate > 0 ? model.mel_cfg.sample_rate : 16000;
    const int hop = model.mel_cfg.hop_length   > 0 ? model.mel_cfg.hop_length   : 160;

    // Frame count derived directly from the audio-seconds knob; we
    // skip compute_log_mel entirely (the host-side mel pipeline
    // doesn't have any cold-build state worth amortising) and feed
    // the encoder zeros at the correct shape.
    const int n_frames = std::max(8,
        (int) std::lround((double) audio_seconds * (double) sr / (double) hop));
    const int n_mels   = model.mel_cfg.n_mels > 0 ? model.mel_cfg.n_mels : 80;

    std::vector<float> zeros((size_t) n_frames * (size_t) n_mels, 0.0f);

    EncoderOutputs out;
    // capture_intermediates=false: production-shape call (no
    // per-stage host roundtrips); same `false` the audit
    // wired into Engine::transcribe_*. capture=false keeps the
    // graph topology identical to a real call so the kernel
    // pipeline cache hit is real.
    if (int rc = run_encoder(model, zeros.data(), n_frames, n_mels, out,
                             /*max_layers=*/-1,
                             /*capture_intermediates=*/false); rc != 0) {
        // Don't fail construction — the user's first transcribe
        // call will surface the same error with full context.
        // Just log so the field is observable.
        std::fprintf(stderr,
            "[parakeet] prewarm_encoder: run_encoder rc=%d (T_mel=%d, n_mels=%d) -- "
            "first transcribe will pay the cold cost the prewarm was meant to cover\n",
            rc, n_frames, n_mels);
    }
}

Engine::Engine(const EngineOptions & opts) : pimpl_(std::make_unique<Impl>()) {
    pimpl_->opts = opts;

    const int rc = load_from_gguf(opts.model_gguf_path,
                                  pimpl_->model,
                                  opts.n_threads,
                                  opts.n_gpu_layers,
                                  opts.verbose);
    if (rc != 0) {
        throw std::runtime_error("parakeet::Engine: failed to load GGUF '" +
                                 opts.model_gguf_path +
                                 "' (rc=" + std::to_string(rc) + ")");
    }

    if (pimpl_->model.model_type == ParakeetModelType::TDT) {
        if (tdt_prepare_runtime(pimpl_->model, pimpl_->tdt_rt) != 0) {
            throw std::runtime_error("Engine: tdt_prepare_runtime failed");
        }
        pimpl_->tdt_ready = true;
    }
    if (pimpl_->model.model_type == ParakeetModelType::EOU) {
        if (eou_prepare_runtime(pimpl_->model, pimpl_->eou_rt) != 0) {
            throw std::runtime_error("Engine: eou_prepare_runtime failed");
        }
        pimpl_->eou_ready = true;
    }
    if (pimpl_->model.model_type == ParakeetModelType::SORTFORMER) {
        pimpl_->sortformer_ready = true;
    }

    if (opts.prewarm) {
        prewarm_encoder(pimpl_->model, opts.prewarm_audio_seconds);
    }
}


Engine::~Engine() = default;
Engine::Engine(Engine &&) noexcept = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

const EngineOptions & Engine::options() const {
    return pimpl_->opts;
}

std::string Engine::model_type() const {
    switch (pimpl_->model.model_type) {
        case ParakeetModelType::TDT:        return "tdt";
        case ParakeetModelType::EOU:        return "eou";
        case ParakeetModelType::SORTFORMER: return "sortformer";
        case ParakeetModelType::CTC:
        default:                            return "ctc";
    }
}

bool Engine::is_diarization_model() const {
    return pimpl_->model.model_type == ParakeetModelType::SORTFORMER;
}

bool Engine::is_transcription_model() const {
    return pimpl_->model.model_type == ParakeetModelType::CTC ||
           pimpl_->model.model_type == ParakeetModelType::TDT  ||
           pimpl_->model.model_type == ParakeetModelType::EOU;
}

BackendDevice Engine::backend_device() const {
    return model_has_gpu_backend(pimpl_->model) ? BackendDevice::GPU
                                                : BackendDevice::CPU;
}

std::string Engine::backend_name() const {
    return model_active_backend_name(pimpl_->model);
}

void Engine::cancel() {
    pimpl_->cancel_flag.store(true);
}

EngineResult Engine::transcribe(const std::string & wav_path) {
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        throw std::runtime_error("parakeet::Engine::transcribe: failed to load wav '" +
                                 wav_path + "' (rc=" + std::to_string(rc) + ")");
    }
    return transcribe_samples(samples.data(), (int) samples.size(), sr);
}

EngineResult Engine::transcribe_samples(const float * samples, int n_samples, int sample_rate) {
    if (!samples || n_samples <= 0) {
        throw std::runtime_error("parakeet::Engine::transcribe_samples: empty input");
    }
    if (sample_rate != pimpl_->model.mel_cfg.sample_rate) {
        throw std::runtime_error("parakeet::Engine::transcribe_samples: input is " +
                                 std::to_string(sample_rate) + " Hz but model expects " +
                                 std::to_string(pimpl_->model.mel_cfg.sample_rate) + " Hz");
    }
    if (pimpl_->model.model_type == ParakeetModelType::SORTFORMER) {
        throw std::runtime_error(
            "parakeet::Engine::transcribe_samples: loaded GGUF is a Sortformer "
            "diarization model; call Engine::diarize() (or transcribe_with_speakers "
            "with a separate ASR engine) instead.");
    }

    pimpl_->cancel_flag.store(false);

    using clock = std::chrono::steady_clock;
    const auto t_total = clock::now();

    const auto t_mel = clock::now();
    std::vector<float> mel;
    int n_mel_frames = 0;
    if (int rc = compute_log_mel(samples, n_samples, pimpl_->model.mel_cfg,
                                 pimpl_->mel_state, mel, n_mel_frames); rc != 0) {
        throw std::runtime_error("parakeet::Engine::transcribe_samples: compute_log_mel failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const double preprocess_ms = ms_since(t_mel);

    const auto t_enc = clock::now();
    EncoderOutputs enc_out;
    if (int rc = run_encoder(pimpl_->model, mel.data(), n_mel_frames,
                             pimpl_->model.mel_cfg.n_mels, enc_out,
                             /*max_layers=*/-1,
                             /*capture_intermediates=*/false); rc != 0) {
        throw std::runtime_error("parakeet::Engine::transcribe_samples: run_encoder failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const double encoder_ms = ms_since(t_enc);

    const auto t_dec = clock::now();
    std::vector<int32_t> ids;
    std::string text;
    if (pimpl_->model.model_type == ParakeetModelType::TDT) {
        TdtDecodeOptions dopts;
        TdtDecodeResult  dres;
        if (int rc = tdt_greedy_decode(pimpl_->model, pimpl_->tdt_rt,
                                       enc_out.encoder_out.data(),
                                       enc_out.n_enc_frames, enc_out.d_model,
                                       dopts, dres); rc != 0) {
            throw std::runtime_error("parakeet::Engine::transcribe_samples: tdt_greedy_decode failed (rc=" +
                                     std::to_string(rc) + ")");
        }
        ids  = std::move(dres.token_ids);
        text = std::move(dres.text);
    } else if (pimpl_->model.model_type == ParakeetModelType::EOU) {
        EouDecodeOptions dopts;
        dopts.max_symbols_per_step = pimpl_->model.encoder_cfg.eou_max_symbols_per_step;
        EouDecodeResult dres;
        if (int rc = eou_greedy_decode(pimpl_->model, pimpl_->eou_rt,
                                       enc_out.encoder_out.data(),
                                       enc_out.n_enc_frames, enc_out.d_model,
                                       dopts, dres); rc != 0) {
            throw std::runtime_error("parakeet::Engine::transcribe_samples: eou_greedy_decode failed (rc=" +
                                     std::to_string(rc) + ")");
        }
        ids  = std::move(dres.token_ids);
        text = std::move(dres.text);
    } else {
        ids  = ctc_greedy_decode(enc_out.logits.data(), enc_out.n_enc_frames,
                                 pimpl_->model.vocab_size, pimpl_->model.blank_id);
        text = detokenize(pimpl_->model.vocab, ids);
    }
    const double decode_ms = ms_since(t_dec);

    EngineResult result;
    result.text           = std::move(text);
    result.token_ids      = std::move(ids);
    result.preprocess_ms  = preprocess_ms;
    result.encoder_ms     = encoder_ms;
    result.decode_ms      = decode_ms;
    result.total_ms       = ms_since(t_total);
    result.audio_samples  = n_samples;
    result.sample_rate    = sample_rate;
    result.mel_frames     = n_mel_frames;
    result.encoder_frames = enc_out.n_enc_frames;
    return result;
}

EngineResult Engine::transcribe_stream(const std::string & wav_path,
                                       const StreamingOptions & opts,
                                       StreamingCallback on_segment) {
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        throw std::runtime_error("parakeet::Engine::transcribe_stream: failed to load wav '" +
                                 wav_path + "' (rc=" + std::to_string(rc) + ")");
    }
    return transcribe_samples_stream(samples.data(), (int) samples.size(), sr,
                                     opts, std::move(on_segment));
}

EngineResult Engine::transcribe_samples_stream(const float * samples,
                                               int n_samples,
                                               int sample_rate,
                                               const StreamingOptions & opts,
                                               StreamingCallback on_segment) {
    if (!samples || n_samples <= 0) {
        throw std::runtime_error("parakeet::Engine::transcribe_samples_stream: empty input");
    }
    if (sample_rate != pimpl_->model.mel_cfg.sample_rate) {
        throw std::runtime_error("parakeet::Engine::transcribe_samples_stream: input is " +
                                 std::to_string(sample_rate) + " Hz but model expects " +
                                 std::to_string(pimpl_->model.mel_cfg.sample_rate) + " Hz");
    }
    if (opts.sample_rate != 0 && opts.sample_rate != sample_rate) {
        throw std::runtime_error("parakeet::Engine::transcribe_samples_stream: "
                                 "StreamingOptions.sample_rate must match the input sample_rate");
    }
    if (opts.chunk_ms <= 0) {
        throw std::runtime_error("parakeet::Engine::transcribe_samples_stream: "
                                 "StreamingOptions.chunk_ms must be > 0");
    }
    if (pimpl_->model.model_type == ParakeetModelType::SORTFORMER) {
        throw std::runtime_error(
            "transcribe_samples_stream: streaming is for transcription models only; "
            "Sortformer is a diarization model. Use Engine::diarize().");
    }

    pimpl_->cancel_flag.store(false);

    using clock = std::chrono::steady_clock;
    const auto t_total = clock::now();

    const auto t_mel = clock::now();
    std::vector<float> mel;
    int n_mel_frames = 0;
    if (int rc = compute_log_mel(samples, n_samples, pimpl_->model.mel_cfg,
                                 pimpl_->mel_state, mel, n_mel_frames); rc != 0) {
        throw std::runtime_error("parakeet::Engine::transcribe_samples_stream: compute_log_mel failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const double preprocess_ms = ms_since(t_mel);

    const auto t_enc = clock::now();
    EncoderOutputs enc_out;
    if (int rc = run_encoder(pimpl_->model, mel.data(), n_mel_frames,
                             pimpl_->model.mel_cfg.n_mels, enc_out,
                             /*max_layers=*/-1,
                             /*capture_intermediates=*/false); rc != 0) {
        throw std::runtime_error("parakeet::Engine::transcribe_samples_stream: run_encoder failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const double encoder_ms = ms_since(t_enc);

    const int T_enc = enc_out.n_enc_frames;
    const int vocab = pimpl_->model.vocab_size;
    const int blank = pimpl_->model.blank_id;

    const double frame_stride_ms = encoder_frame_stride_ms(pimpl_->model);
    int frames_per_window = (int) std::floor(opts.chunk_ms / frame_stride_ms);
    if (frames_per_window < 1) frames_per_window = 1;

    EngineResult result;
    result.preprocess_ms  = preprocess_ms;
    result.encoder_ms     = encoder_ms;
    result.audio_samples  = n_samples;
    result.sample_rate    = sample_rate;
    result.mel_frames     = n_mel_frames;
    result.encoder_frames = T_enc;

    const auto t_dec = clock::now();

    const bool is_tdt = (pimpl_->model.model_type == ParakeetModelType::TDT);
    const bool is_eou = (pimpl_->model.model_type == ParakeetModelType::EOU);

    int32_t prev_token = -1;
    TdtDecodeState tdt_state;
    EouDecodeState eou_state;
    if (is_tdt) tdt_init_state(pimpl_->tdt_rt, (int) pimpl_->model.blank_id, tdt_state);
    if (is_eou) eou_init_state(pimpl_->eou_rt, eou_state);

    int chunk_index = 0;
    bool first_segment = true;

    for (int start = 0; start < T_enc; start += frames_per_window) {
        if (pimpl_->cancel_flag.load()) break;

        int end = start + frames_per_window;
        if (end > T_enc) end = T_enc;

        const auto t_win = clock::now();

        std::vector<int32_t> win_tokens;
        int eou_boundaries_in_chunk = 0;
        if (is_tdt) {
            TdtDecodeOptions dopts;
            int steps = 0;
            const float * win_enc = enc_out.encoder_out.data()
                                  + static_cast<size_t>(start) * enc_out.d_model;
            if (int rc = tdt_decode_window(pimpl_->model, pimpl_->tdt_rt,
                                           win_enc, end - start, enc_out.d_model,
                                           dopts, tdt_state, win_tokens, steps);
                rc != 0) {
                throw std::runtime_error("parakeet::Engine::transcribe_samples_stream: "
                                         "tdt_decode_window failed (rc=" + std::to_string(rc) + ")");
            }
        } else if (is_eou) {
            EouDecodeOptions dopts;
            dopts.max_symbols_per_step = pimpl_->model.encoder_cfg.eou_max_symbols_per_step;
            std::vector<EouSegmentBoundary> win_segments;
            int steps = 0;
            const float * win_enc = enc_out.encoder_out.data()
                                  + static_cast<size_t>(start) * enc_out.d_model;
            if (int rc = eou_decode_window(pimpl_->model, pimpl_->eou_rt,
                                           win_enc, end - start, enc_out.d_model,
                                           dopts, eou_state,
                                           win_tokens, win_segments, steps);
                rc != 0) {
                throw std::runtime_error("parakeet::Engine::transcribe_samples_stream: "
                                         "eou_decode_window failed (rc=" + std::to_string(rc) + ")");
            }
            eou_boundaries_in_chunk = static_cast<int>(win_segments.size());
        } else {
            ctc_greedy_decode_window(enc_out.logits.data(),
                                     start, end, vocab, blank,
                                     prev_token, win_tokens, nullptr);
        }

        const size_t prev_cumulative_len = result.text.size();
        result.token_ids.insert(result.token_ids.end(),
                                win_tokens.begin(), win_tokens.end());
        result.text = detokenize(pimpl_->model.vocab, result.token_ids);
        const std::string win_text = result.text.substr(prev_cumulative_len);

        const double win_decode_ms = ms_since(t_win);

        const double seg_end_s = static_cast<double>(end) * frame_stride_ms / 1000.0;
        if (on_segment) {
            StreamingSegment seg;
            seg.text        = win_text;
            seg.token_ids   = win_tokens;
            seg.start_s     = static_cast<double>(start) * frame_stride_ms / 1000.0;
            seg.end_s       = seg_end_s;
            seg.chunk_index = chunk_index;
            seg.is_final    = true;
            seg.is_eou_boundary = eou_boundaries_in_chunk > 0;
            seg.encoder_ms  = first_segment ? encoder_ms : 0.0;
            seg.decode_ms   = win_decode_ms;
            on_segment(seg);
        }

        // EOU Mode 2: emit EndOfTurn when EOU boundaries appear in this chunk (same idea as Mode 3).
        if (opts.on_event && eou_boundaries_in_chunk > 0) {
            StreamEvent ev;
            ev.type           = StreamEventType::EndOfTurn;
            ev.timestamp_s    = seg_end_s;
            ev.chunk_index    = chunk_index;
            ev.eot_confidence = 1.0f;
            for (int i = 0; i < eou_boundaries_in_chunk; ++i) {
                opts.on_event(ev);
            }
        }

        ++chunk_index;
        first_segment = false;
    }

    result.decode_ms = ms_since(t_dec);
    result.total_ms  = ms_since(t_total);
    return result;
}

DiarizationResult Engine::diarize(const std::string & wav_path,
                                  const DiarizationOptions & opts) {
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        throw std::runtime_error("Engine::diarize: failed to load wav '" + wav_path +
                                 "' (rc=" + std::to_string(rc) + ")");
    }
    return diarize_samples(samples.data(), (int) samples.size(), sr, opts);
}

static DiarizationResult engine_impl_diarize_helper(Engine::Impl & impl,
                                                    const float * samples,
                                                    int n_samples,
                                                    int sample_rate,
                                                    const DiarizationOptions & opts) {
    if (!samples || n_samples <= 0) {
        throw std::runtime_error("diarize: empty input");
    }
    if (sample_rate != impl.model.mel_cfg.sample_rate) {
        throw std::runtime_error("diarize: input is " +
                                 std::to_string(sample_rate) + " Hz but model expects " +
                                 std::to_string(impl.model.mel_cfg.sample_rate) + " Hz");
    }
    if (impl.model.model_type != ParakeetModelType::SORTFORMER || !impl.sortformer_ready) {
        throw std::runtime_error("diarize: loaded GGUF is not a Sortformer model");
    }

    impl.cancel_flag.store(false);

    using clock = std::chrono::steady_clock;
    const auto t_total = clock::now();

    std::vector<float> work(samples, samples + n_samples);
    float peak = 0.0f;
    for (float v : work) {
        const float a = std::fabs(v);
        if (a > peak) peak = a;
    }
    constexpr float NORM_FLOOR = 1e-3f;
    if (peak > NORM_FLOOR) {
        const float inv = 1.0f / peak;
        for (float & v : work) v *= inv;
    }

    const auto t_mel = clock::now();
    std::vector<float> mel;
    int n_mel_frames = 0;
    if (int rc = compute_log_mel(work.data(), n_samples, impl.model.mel_cfg,
                                 impl.mel_state, mel, n_mel_frames); rc != 0) {
        throw std::runtime_error("diarize: compute_log_mel failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const double preprocess_ms = ms_since(t_mel);

    const auto t_enc = clock::now();
    EncoderOutputs enc_out;
    if (int rc = run_encoder(impl.model, mel.data(), n_mel_frames,
                             impl.model.mel_cfg.n_mels, enc_out,
                             /*max_layers=*/-1,
                             /*capture_intermediates=*/false); rc != 0) {
        throw std::runtime_error("diarize: run_encoder failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const double encoder_ms = ms_since(t_enc);

    SortformerDiarizationOptions sopts;
    sopts.threshold = opts.threshold;
    SortformerDiarizationResult dres;

    ggml_backend_t active_backend = model_active_backend(impl.model);
    if (!active_backend) {
        throw std::runtime_error("diarize: no active ggml backend");
    }

    int diarize_rc = sortformer_diarize_ggml(impl.model,
                                             enc_out.encoder_out.data(),
                                             enc_out.n_enc_frames, enc_out.d_model,
                                             active_backend, sopts, dres);
    if (diarize_rc != 0) {
        throw std::runtime_error("diarize: sortformer_diarize failed (rc=" +
                                 std::to_string(diarize_rc) + ")");
    }

    DiarizationResult result;
    result.n_frames       = dres.n_frames;
    result.num_spks       = dres.num_spks;
    result.frame_stride_s = dres.frame_stride_s;
    result.speaker_probs  = std::move(dres.speaker_probs);
    result.audio_samples  = n_samples;
    result.sample_rate    = sample_rate;
    result.preprocess_ms  = preprocess_ms;
    result.encoder_ms     = encoder_ms;
    result.decode_ms      = dres.decode_ms;
    result.total_ms       = ms_since(t_total);

    const double min_dur = opts.min_segment_ms / 1000.0;
    for (const auto & s : dres.segments) {
        if ((s.end_s - s.start_s) < min_dur) continue;
        DiarizationSegment d;
        d.speaker_id = s.speaker_id;
        d.start_s    = s.start_s;
        d.end_s      = s.end_s;
        result.segments.push_back(d);
    }

    return result;
}

DiarizationResult Engine::diarize_samples(const float * samples,
                                          int n_samples,
                                          int sample_rate,
                                          const DiarizationOptions & opts) {
    return engine_impl_diarize_helper(*pimpl_, samples, n_samples, sample_rate, opts);
}

AttributedTranscriptionResult transcribe_with_speakers(
    Engine & sortformer_engine,
    Engine & asr_engine,
    const std::string & wav_path,
    const AttributedTranscriptionOptions & opts) {
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        throw std::runtime_error("transcribe_with_speakers: failed to load wav '" +
                                 wav_path + "' (rc=" + std::to_string(rc) + ")");
    }
    return transcribe_samples_with_speakers(sortformer_engine, asr_engine,
                                            samples.data(), (int) samples.size(),
                                            sr, opts);
}

AttributedTranscriptionResult transcribe_samples_with_speakers(
    Engine & sortformer_engine,
    Engine & asr_engine,
    const float * samples,
    int n_samples,
    int sample_rate,
    const AttributedTranscriptionOptions & opts) {
    if (!samples || n_samples <= 0) {
        throw std::runtime_error("transcribe_samples_with_speakers: empty input");
    }
    if (!sortformer_engine.is_diarization_model()) {
        throw std::runtime_error("transcribe_samples_with_speakers: first engine "
                                 "is not a Sortformer diarization model");
    }
    if (!asr_engine.is_transcription_model()) {
        throw std::runtime_error("transcribe_samples_with_speakers: second engine "
                                 "is not an ASR transcription model");
    }

    using clock = std::chrono::steady_clock;
    const auto t_total = clock::now();

    AttributedTranscriptionResult out;
    out.audio_samples = n_samples;
    out.sample_rate   = sample_rate;

    out.diarization = sortformer_engine.diarize_samples(samples, n_samples, sample_rate, opts.diarization);

    const double pad_s = opts.pad_segment_ms / 1000.0;
    const double min_s = opts.min_segment_ms / 1000.0;

    std::vector<AttributedSegment> raw;
    raw.reserve(out.diarization.segments.size());
    for (const auto & seg : out.diarization.segments) {
        const double slice_start = std::max(0.0, seg.start_s - pad_s);
        const double slice_end   = std::min((double) n_samples / sample_rate, seg.end_s + pad_s);
        if ((slice_end - slice_start) < min_s) continue;

        const int start_sample = (int) std::floor(slice_start * sample_rate);
        const int end_sample   = std::min((int) std::ceil(slice_end * sample_rate), n_samples);
        const int n_slice      = end_sample - start_sample;
        if (n_slice <= 0) continue;

        EngineResult er = asr_engine.transcribe_samples(
            samples + start_sample, n_slice, sample_rate);
        ++out.asr_calls;

        AttributedSegment a;
        a.speaker_id = seg.speaker_id;
        a.text       = std::move(er.text);
        a.start_s    = seg.start_s;
        a.end_s      = seg.end_s;
        raw.push_back(std::move(a));
    }

    if (!opts.merge_same_speaker) {
        out.segments = std::move(raw);
    } else {
        for (auto & s : raw) {
            if (!out.segments.empty() &&
                out.segments.back().speaker_id == s.speaker_id) {
                if (!out.segments.back().text.empty() && !s.text.empty()) {
                    out.segments.back().text += ' ';
                }
                out.segments.back().text += s.text;
                out.segments.back().end_s = s.end_s;
            } else {
                out.segments.push_back(std::move(s));
            }
        }
    }

    out.total_ms = ms_since(t_total);
    return out;
}

struct StreamSession::Impl {
    Engine::Impl *      engine_impl = nullptr;
    StreamingOptions    opts;
    StreamingCallback   on_segment;

    int chunk_samples           = 0;
    int left_context_samples    = 0;
    int right_lookahead_samples = 0;

    std::vector<float>   left_history;
    std::vector<float>   pending;

    int     chunk_index    = 0;
    int64_t emitted_samples = 0;
    int32_t prev_token     = -1;
    TdtDecodeState tdt_state;
    EouDecodeState eou_state;

    std::string             cumulative_text;
    std::vector<int32_t>    cumulative_token_ids;

    bool finalized = false;
    bool cancelled = false;

    // Optional EnergyVad for CTC/TDT when enable_energy_vad and no native VAD exists.
    std::unique_ptr<EnergyVad> energy_vad;
    int64_t total_pcm_seen = 0;

    // Reusable mel preprocess scratch. Carrying it on the session
    // means every Mode 2 / Mode 3 chunk skips the 6-vector allocation
    // in `compute_log_mel` after the first call -- the dominant
    // per-chunk allocator pressure on streaming workloads.
    MelState mel_state;

    void process_window(const float * window_samples, int window_n,
                        int center_start_sample,
                        int center_end_sample,
                        bool is_final_chunk);
    void try_emit_chunks();
    void flush_remainder();
};

void StreamSession::Impl::process_window(const float * window_samples, int window_n,
                                         int center_start_sample,
                                         int center_end_sample,
                                         bool is_final_chunk) {
    if (cancelled) return;
    if (window_n <= 0) return;

    using clock = std::chrono::steady_clock;
    const auto t_chunk = clock::now();

    std::vector<float> mel;
    int n_mel_frames = 0;
    if (int rc = compute_log_mel(window_samples, window_n,
                                 engine_impl->model.mel_cfg,
                                 mel_state, mel, n_mel_frames); rc != 0) {
        throw std::runtime_error("StreamSession: compute_log_mel failed (rc=" +
                                 std::to_string(rc) + ")");
    }

    EncoderOutputs enc_out;
    if (int rc = run_encoder(engine_impl->model, mel.data(), n_mel_frames,
                             engine_impl->model.mel_cfg.n_mels, enc_out,
                             /*max_layers=*/-1,
                             /*capture_intermediates=*/false); rc != 0) {
        throw std::runtime_error("StreamSession: run_encoder failed (rc=" +
                                 std::to_string(rc) + ")");
    }

    const double encoder_ms = ms_since(t_chunk);

    const int T_enc = enc_out.n_enc_frames;
    const int sr    = opts.sample_rate;
    const double frame_stride_ms = encoder_frame_stride_ms(engine_impl->model);
    const int frame_samples = (int) std::round(sr * frame_stride_ms / 1000.0);

    int left_drop_frames     = center_start_sample / frame_samples;
    int center_frame_count   = (center_end_sample - center_start_sample) / frame_samples;
    int right_drop_frames    = T_enc - left_drop_frames - center_frame_count;
    if (is_final_chunk) {
        right_drop_frames = 0;
        center_frame_count = T_enc - left_drop_frames;
    }

    if (left_drop_frames < 0) left_drop_frames = 0;
    if (left_drop_frames > T_enc) left_drop_frames = T_enc;
    if (right_drop_frames < 0) right_drop_frames = 0;
    if (right_drop_frames > T_enc - left_drop_frames) {
        right_drop_frames = T_enc - left_drop_frames;
    }

    const int center_end_frame = T_enc - right_drop_frames;

    const auto t_dec = clock::now();
    std::vector<int32_t> win_tokens;
    int  eou_boundaries_in_chunk = 0;
    if (engine_impl->model.model_type == ParakeetModelType::TDT) {
        TdtDecodeOptions dopts;
        int steps = 0;
        const int n_frames = std::max(0, center_end_frame - left_drop_frames);
        const float * win_enc = enc_out.encoder_out.data()
                              + static_cast<size_t>(left_drop_frames) * enc_out.d_model;
        if (int rc = tdt_decode_window(engine_impl->model, engine_impl->tdt_rt,
                                       win_enc, n_frames, enc_out.d_model,
                                       dopts, tdt_state, win_tokens, steps);
            rc != 0) {
            throw std::runtime_error("StreamSession: tdt_decode_window failed (rc=" +
                                     std::to_string(rc) + ")");
        }
    } else if (engine_impl->model.model_type == ParakeetModelType::EOU) {
        EouDecodeOptions dopts;
        dopts.max_symbols_per_step =
            engine_impl->model.encoder_cfg.eou_max_symbols_per_step;
        std::vector<EouSegmentBoundary> win_segments;
        int steps = 0;
        const int n_frames = std::max(0, center_end_frame - left_drop_frames);
        const float * win_enc = enc_out.encoder_out.data()
                              + static_cast<size_t>(left_drop_frames) * enc_out.d_model;
        if (int rc = eou_decode_window(engine_impl->model, engine_impl->eou_rt,
                                       win_enc, n_frames, enc_out.d_model,
                                       dopts, eou_state,
                                       win_tokens, win_segments, steps);
            rc != 0) {
            throw std::runtime_error("StreamSession: eou_decode_window failed (rc=" +
                                     std::to_string(rc) + ")");
        }
        eou_boundaries_in_chunk = static_cast<int>(win_segments.size());
    } else {
        ctc_greedy_decode_window(enc_out.logits.data(),
                                 left_drop_frames, center_end_frame,
                                 engine_impl->model.vocab_size,
                                 engine_impl->model.blank_id,
                                 prev_token, win_tokens, nullptr);
    }

    const size_t prev_cumulative_len = cumulative_text.size();
    cumulative_token_ids.insert(cumulative_token_ids.end(),
                                win_tokens.begin(), win_tokens.end());
    cumulative_text = detokenize(engine_impl->model.vocab, cumulative_token_ids);
    const std::string win_text = cumulative_text.substr(prev_cumulative_len);

    const double decode_ms = ms_since(t_dec);

    const double chunk_start_s = static_cast<double>(emitted_samples) / sr;
    const double chunk_end_s   = static_cast<double>(emitted_samples +
                                                     (center_end_sample - center_start_sample)) / sr;
    if (on_segment) {
        StreamingSegment seg;
        seg.text        = win_text;
        seg.token_ids   = win_tokens;
        seg.start_s     = chunk_start_s;
        seg.end_s       = chunk_end_s;
        seg.chunk_index = chunk_index;
        seg.is_final    = true;
        seg.is_eou_boundary = eou_boundaries_in_chunk > 0;
        seg.encoder_ms  = encoder_ms;
        seg.decode_ms   = decode_ms;
        on_segment(seg);
    }

    // EOU: EndOfTurn when `<EOU>` appears this chunk. Confidence is 1.0 for this discrete signal.
    if (opts.on_event && eou_boundaries_in_chunk > 0) {
        StreamEvent ev;
        ev.type           = StreamEventType::EndOfTurn;
        ev.timestamp_s    = chunk_end_s;
        ev.chunk_index    = chunk_index;
        ev.eot_confidence = 1.0f;
        for (int i = 0; i < eou_boundaries_in_chunk; ++i) {
            opts.on_event(ev);
        }
    }

    emitted_samples += (center_end_sample - center_start_sample);
    ++chunk_index;
}

void StreamSession::Impl::try_emit_chunks() {
    if (cancelled) return;
    while (!cancelled &&
           static_cast<int>(pending.size()) >= chunk_samples + right_lookahead_samples) {
        std::vector<float> window;
        window.reserve(left_history.size() + chunk_samples + right_lookahead_samples);
        window.insert(window.end(), left_history.begin(), left_history.end());
        window.insert(window.end(), pending.begin(),
                      pending.begin() + chunk_samples + right_lookahead_samples);

        const int center_start = static_cast<int>(left_history.size());
        const int center_end   = center_start + chunk_samples;

        process_window(window.data(), static_cast<int>(window.size()),
                       center_start, center_end, /*is_final_chunk=*/false);

        left_history.insert(left_history.end(),
                            pending.begin(), pending.begin() + chunk_samples);
        if (static_cast<int>(left_history.size()) > left_context_samples) {
            left_history.erase(left_history.begin(),
                               left_history.end() - left_context_samples);
        }

        pending.erase(pending.begin(), pending.begin() + chunk_samples);
    }
}

void StreamSession::Impl::flush_remainder() {
    if (cancelled) return;
    if (pending.empty()) return;

    std::vector<float> window;
    window.reserve(left_history.size() + pending.size());
    window.insert(window.end(), left_history.begin(), left_history.end());
    window.insert(window.end(), pending.begin(), pending.end());

    const int center_start = static_cast<int>(left_history.size());
    const int center_end   = center_start + static_cast<int>(pending.size());

    process_window(window.data(), static_cast<int>(window.size()),
                   center_start, center_end, /*is_final_chunk=*/true);

    pending.clear();
    left_history.clear();
}

StreamSession::StreamSession(std::unique_ptr<Impl> impl)
    : pimpl_(std::move(impl)) {}

StreamSession::~StreamSession() {
    if (pimpl_ && !pimpl_->finalized && !pimpl_->cancelled) {
        try { pimpl_->cancelled = true; } catch (...) {}
    }
}
StreamSession::StreamSession(StreamSession &&) noexcept = default;
StreamSession & StreamSession::operator=(StreamSession &&) noexcept = default;

const StreamingOptions & StreamSession::options() const {
    return pimpl_->opts;
}

// Feed PCM into optional EnergyVad and emit VadStateChanged when its state flips.
static void stream_drive_energy_vad(StreamSession::Impl & impl,
                                    const float * samples, int n_samples,
                                    int64_t start_sample) {
    if (!impl.energy_vad || !impl.opts.on_event) return;
    EnergyVad::Transition tr = impl.energy_vad->process(samples, n_samples,
                                                        start_sample);
    if (tr.to_state == EnergyVad::State::Unknown) return;
    StreamEvent ev;
    ev.type        = StreamEventType::VadStateChanged;
    ev.timestamp_s = (double) tr.at_sample / (double) impl.opts.sample_rate;
    ev.chunk_index = -1;
    ev.vad_state   = (tr.to_state == EnergyVad::State::Speaking)
                         ? VadState::Speaking : VadState::Silent;
    ev.vad_score   = tr.rms;
    impl.opts.on_event(ev);
}

void StreamSession::feed_pcm_f32(const float * samples, int n_samples) {
    if (!pimpl_) throw std::runtime_error("StreamSession: moved-from session");
    if (pimpl_->finalized) {
        throw std::runtime_error("StreamSession::feed_pcm_f32: session already finalized");
    }
    if (pimpl_->cancelled) return;
    if (!samples || n_samples <= 0) return;
    stream_drive_energy_vad(*pimpl_, samples, n_samples,
                            pimpl_->total_pcm_seen);
    pimpl_->total_pcm_seen += n_samples;
    pimpl_->pending.insert(pimpl_->pending.end(), samples, samples + n_samples);
    pimpl_->try_emit_chunks();
}

void StreamSession::feed_pcm_i16(const int16_t * samples, int n_samples) {
    if (!pimpl_) throw std::runtime_error("StreamSession: moved-from session");
    if (pimpl_->finalized) {
        throw std::runtime_error("StreamSession::feed_pcm_i16: session already finalized");
    }
    if (pimpl_->cancelled) return;
    if (!samples || n_samples <= 0) return;
    const size_t prev = pimpl_->pending.size();
    pimpl_->pending.resize(prev + n_samples);
    constexpr float inv = 1.0f / 32768.0f;
    for (int i = 0; i < n_samples; ++i) {
        pimpl_->pending[prev + i] = static_cast<float>(samples[i]) * inv;
    }
    stream_drive_energy_vad(*pimpl_, pimpl_->pending.data() + prev, n_samples,
                            pimpl_->total_pcm_seen);
    pimpl_->total_pcm_seen += n_samples;
    pimpl_->try_emit_chunks();
}

void StreamSession::finalize() {
    if (!pimpl_) return;
    if (pimpl_->finalized) return;
    pimpl_->finalized = true;
    pimpl_->try_emit_chunks();
    pimpl_->flush_remainder();
}

void StreamSession::cancel() {
    if (!pimpl_) return;
    pimpl_->cancelled = true;
}

std::unique_ptr<StreamSession> Engine::stream_start(const StreamingOptions & opts,
                                                    StreamingCallback on_segment) {
    if (opts.sample_rate != pimpl_->model.mel_cfg.sample_rate) {
        throw std::runtime_error(
            "Engine::stream_start: opts.sample_rate=" + std::to_string(opts.sample_rate) +
            " does not match model rate=" + std::to_string(pimpl_->model.mel_cfg.sample_rate));
    }
    if (opts.chunk_ms <= 0) {
        throw std::runtime_error("Engine::stream_start: chunk_ms must be > 0");
    }
    if (opts.left_context_ms < 0 || opts.right_lookahead_ms < 0) {
        throw std::runtime_error("Engine::stream_start: left_context_ms and right_lookahead_ms must be >= 0");
    }
    if (pimpl_->model.model_type == ParakeetModelType::SORTFORMER) {
        throw std::runtime_error(
            "Engine::stream_start: streaming is for transcription models only; "
            "Sortformer is a diarization model. Use Engine::diarize().");
    }

    auto impl = std::make_unique<StreamSession::Impl>();
    impl->engine_impl  = pimpl_.get();
    impl->opts         = opts;
    impl->on_segment   = std::move(on_segment);

    const int sr = opts.sample_rate;
    impl->chunk_samples           = sr * opts.chunk_ms / 1000;
    impl->left_context_samples    = sr * opts.left_context_ms / 1000;
    impl->right_lookahead_samples = sr * opts.right_lookahead_ms / 1000;

    impl->left_history.reserve(impl->left_context_samples);
    impl->pending.reserve(impl->chunk_samples + impl->right_lookahead_samples);

    if (pimpl_->model.model_type == ParakeetModelType::TDT) {
        tdt_init_state(pimpl_->tdt_rt, (int) pimpl_->model.blank_id, impl->tdt_state);
    }
    // Optional EnergyVad for CTC/TDT only (EOU uses `<EOU>`; Sortformer uses SortformerStreamSession).
    if (opts.enable_energy_vad &&
        pimpl_->model.model_type != ParakeetModelType::EOU) {
        impl->energy_vad = std::make_unique<EnergyVad>(
            sr,
            opts.energy_vad_window_ms,
            opts.energy_vad_hangover_ms,
            opts.energy_vad_threshold_db);
    }
    if (pimpl_->model.model_type == ParakeetModelType::EOU) {
        eou_init_state(pimpl_->eou_rt, impl->eou_state);
    }

    return std::make_unique<StreamSession>(std::move(impl));
}

struct SortformerStreamSession::Impl {
    Engine::Impl *              engine_impl = nullptr;
    SortformerStreamingOptions  opts;
    SortformerSegmentCallback   on_segment;

    int    chunk_samples   = 0;
    int    history_samples = 0;

    std::vector<float> ring;
    int64_t            ring_origin_sample = 0;

    int64_t            emitted_samples    = 0;
    int                chunk_index        = 0;

    bool finalized = false;
    bool cancelled = false;

    std::vector<StreamingDiarizationSegment> last_pending;

    // Speaking vs silent from Sortformer probs: max probability above opts.threshold.
    // Initial Unknown forces a transition on the first chunk.
    VadState vad_state = VadState::Unknown;

    void try_emit_chunks();
    void process_chunk(int64_t window_start_sample,
                       int64_t window_end_sample,
                       int64_t emit_start_sample,
                       int64_t emit_end_sample,
                       bool    is_final_chunk);
};

void SortformerStreamSession::Impl::process_chunk(int64_t window_start_sample,
                                                  int64_t window_end_sample,
                                                  int64_t emit_start_sample,
                                                  int64_t emit_end_sample,
                                                  bool    is_final_chunk) {
    if (cancelled) return;
    if (window_end_sample <= window_start_sample) return;

    const size_t off  = (size_t) (window_start_sample - ring_origin_sample);
    const int    n    = (int) (window_end_sample - window_start_sample);

    DiarizationOptions diopts;
    diopts.threshold      = opts.threshold;
    diopts.min_segment_ms = opts.min_segment_ms;

    DiarizationResult diar;
    {
        const float * win = ring.data() + off;
        diar = engine_impl_diarize_helper(*engine_impl, win, n, opts.sample_rate, diopts);
    }

    const double window_offset_s = (double) window_start_sample / opts.sample_rate;
    const double emit_lo_s = (double) emit_start_sample / opts.sample_rate;
    const double emit_hi_s = (double) emit_end_sample   / opts.sample_rate;

    std::vector<StreamingDiarizationSegment> emitted;
    emitted.reserve(diar.segments.size());

    for (const auto & s : diar.segments) {
        const double abs_start = window_offset_s + s.start_s;
        const double abs_end   = window_offset_s + s.end_s;
        if (abs_end <= emit_lo_s) continue;
        if (abs_start >= emit_hi_s) continue;

        StreamingDiarizationSegment out;
        out.speaker_id  = s.speaker_id;
        out.start_s     = std::max(abs_start, emit_lo_s);
        out.end_s       = std::min(abs_end,   emit_hi_s);
        out.chunk_index = chunk_index;
        out.is_final    = is_final_chunk;
        if (out.end_s - out.start_s < opts.min_segment_ms / 1000.0) continue;
        emitted.push_back(out);
    }

    if (on_segment) {
        for (const auto & seg : emitted) on_segment(seg);
    }
    last_pending = std::move(emitted);

    // VadStateChanged from speaker_probs: a frame speaks if any speaker exceeds threshold;
    // the chunk speaks if any emitting-frame qualifies; dominant speaker from mean probs.
    if (opts.on_event) {
        const int num_spks = diar.num_spks;
        const int n_frames = diar.n_frames;
        const double frame_stride_s = diar.frame_stride_s > 0.0
                                          ? diar.frame_stride_s : 0.08;
        bool   any_speaking = false;
        int    dominant_spk = -1;
        float  best_score   = 0.0f;
        std::vector<double> spk_score_sum(num_spks, 0.0);
        int    spk_count = 0;
        for (int t = 0; t < n_frames; ++t) {
            const double frame_t_s = window_offset_s + frame_stride_s * t;
            if (frame_t_s < emit_lo_s || frame_t_s >= emit_hi_s) continue;
            ++spk_count;
            for (int s = 0; s < num_spks; ++s) {
                const float p = diar.speaker_probs[(size_t) t * num_spks + s];
                spk_score_sum[s] += p;
                if (p > opts.threshold) any_speaking = true;
                if (p > best_score) best_score = p;
            }
        }
        if (any_speaking) {
            double best_mean = -1.0;
            for (int s = 0; s < num_spks; ++s) {
                const double mean = spk_score_sum[s] / std::max(1, spk_count);
                if (mean > best_mean) { best_mean = mean; dominant_spk = s; }
            }
        }
        const VadState new_state = any_speaking ? VadState::Speaking : VadState::Silent;
        if (new_state != vad_state) {
            StreamEvent ev;
            ev.type        = StreamEventType::VadStateChanged;
            ev.timestamp_s = emit_lo_s;
            ev.chunk_index = chunk_index;
            ev.vad_state   = new_state;
            ev.speaker_id  = (new_state == VadState::Speaking) ? dominant_spk : -1;
            ev.vad_score   = best_score;
            opts.on_event(ev);
            vad_state = new_state;
        }
    }

    emitted_samples = emit_end_sample;
    ++chunk_index;
}

void SortformerStreamSession::Impl::try_emit_chunks() {
    if (cancelled) return;
    while (!cancelled) {
        const int64_t available_end = ring_origin_sample + (int64_t) ring.size();
        if (available_end - emitted_samples < chunk_samples) return;

        const int64_t emit_end = emitted_samples + chunk_samples;

        const int64_t window_end   = emit_end;
        const int64_t window_start = std::max(ring_origin_sample,
                                              window_end - history_samples);
        process_chunk(window_start, window_end, emitted_samples, emit_end, /*is_final=*/false);

        if (history_samples > 0) {
            const int64_t keep_from = std::max(ring_origin_sample,
                                               emit_end - history_samples);
            if (keep_from > ring_origin_sample) {
                const size_t drop = (size_t) (keep_from - ring_origin_sample);
                ring.erase(ring.begin(), ring.begin() + drop);
                ring_origin_sample = keep_from;
            }
        }
    }
}

SortformerStreamSession::SortformerStreamSession(std::unique_ptr<Impl> impl)
    : pimpl_(std::move(impl)) {}

SortformerStreamSession::~SortformerStreamSession() {
    if (pimpl_ && !pimpl_->finalized && !pimpl_->cancelled) {
        try { pimpl_->cancelled = true; } catch (...) {}
    }
}
SortformerStreamSession::SortformerStreamSession(SortformerStreamSession &&) noexcept = default;
SortformerStreamSession & SortformerStreamSession::operator=(SortformerStreamSession &&) noexcept = default;

const SortformerStreamingOptions & SortformerStreamSession::options() const {
    return pimpl_->opts;
}

void SortformerStreamSession::feed_pcm_f32(const float * samples, int n_samples) {
    if (!pimpl_) throw std::runtime_error("SortformerStreamSession: moved-from session");
    if (pimpl_->finalized) throw std::runtime_error("feed_pcm_f32: session already finalized");
    if (pimpl_->cancelled) return;
    if (!samples || n_samples <= 0) return;
    pimpl_->ring.insert(pimpl_->ring.end(), samples, samples + n_samples);
    pimpl_->try_emit_chunks();
}

void SortformerStreamSession::feed_pcm_i16(const int16_t * samples, int n_samples) {
    if (!pimpl_) throw std::runtime_error("SortformerStreamSession: moved-from session");
    if (pimpl_->finalized) throw std::runtime_error("feed_pcm_i16: session already finalized");
    if (pimpl_->cancelled) return;
    if (!samples || n_samples <= 0) return;
    const size_t prev = pimpl_->ring.size();
    pimpl_->ring.resize(prev + n_samples);
    constexpr float inv = 1.0f / 32768.0f;
    for (int i = 0; i < n_samples; ++i) pimpl_->ring[prev + i] = (float) samples[i] * inv;
    pimpl_->try_emit_chunks();
}

void SortformerStreamSession::finalize() {
    if (!pimpl_) return;
    if (pimpl_->finalized) return;
    pimpl_->finalized = true;
    pimpl_->try_emit_chunks();

    const int64_t available_end = pimpl_->ring_origin_sample + (int64_t) pimpl_->ring.size();
    if (available_end > pimpl_->emitted_samples) {
        const int64_t window_end   = available_end;
        const int64_t window_start = std::max(pimpl_->ring_origin_sample,
                                              window_end - pimpl_->history_samples);
        pimpl_->process_chunk(window_start, window_end,
                              pimpl_->emitted_samples, available_end,
                              /*is_final_chunk=*/true);
        return;
    }

    if (!pimpl_->cancelled && pimpl_->on_segment) {
        StreamingDiarizationSegment terminator;
        terminator.speaker_id  = -1;
        terminator.start_s     = (double) pimpl_->emitted_samples / pimpl_->opts.sample_rate;
        terminator.end_s       = terminator.start_s;
        terminator.chunk_index = pimpl_->chunk_index;
        terminator.is_final    = true;
        pimpl_->on_segment(terminator);
    }
}

void SortformerStreamSession::cancel() {
    if (!pimpl_) return;
    pimpl_->cancelled = true;
}

std::unique_ptr<SortformerStreamSession> Engine::diarize_start(
    const SortformerStreamingOptions & opts,
    SortformerSegmentCallback on_segment) {
    if (pimpl_->model.model_type != ParakeetModelType::SORTFORMER || !pimpl_->sortformer_ready) {
        throw std::runtime_error("Engine::diarize_start: loaded GGUF is not a Sortformer model");
    }
    if (opts.sample_rate != pimpl_->model.mel_cfg.sample_rate) {
        throw std::runtime_error("Engine::diarize_start: sample_rate mismatch");
    }
    if (opts.chunk_ms <= 0)   throw std::runtime_error("Engine::diarize_start: chunk_ms must be > 0");
    if (opts.history_ms <= 0) throw std::runtime_error("Engine::diarize_start: history_ms must be > 0");
    if (opts.history_ms < opts.chunk_ms) {
        throw std::runtime_error("Engine::diarize_start: history_ms must be >= chunk_ms");
    }

    auto impl = std::make_unique<SortformerStreamSession::Impl>();
    impl->engine_impl     = pimpl_.get();
    impl->opts            = opts;
    impl->on_segment      = std::move(on_segment);
    impl->chunk_samples   = opts.sample_rate * opts.chunk_ms   / 1000;
    impl->history_samples = opts.sample_rate * opts.history_ms / 1000;
    impl->ring.reserve(impl->history_samples);
    return std::make_unique<SortformerStreamSession>(std::move(impl));
}

}
