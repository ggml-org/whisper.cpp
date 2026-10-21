// Engine: GGUF load, transcribe, streaming sessions, diarization, timing and options.

#include "parakeet/engine.h"
#include "parakeet/streaming.h"
#include "parakeet/diarization.h"
#include "parakeet/attributed.h"

#include "parakeet_ctc.h"
#include "parakeet_tdt.h"
#include "parakeet_unified.h"
#include "parakeet_eou.h"
#include "parakeet_sortformer.h"
#include "parakeet_diarization_v3.h"
#include "mel_preprocess.h"
#include "sentencepiece_bpe.h"
#include "energy_vad.h"
#include "independent_segments.h"
#include "long_form.h"
#include "long_form_encoder.h"
#include "parakeet_log.h"
#include "sortformer_finalize.h"
#include "backend_util.h"
#include "ggml-backend.h"

#ifdef PARAKEET_USE_COREML
#include "coreml/parakeet-decoder.h"
#endif

#include <algorithm>
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

bool is_transducer(ParakeetModelType model_type) {
    return model_type == ParakeetModelType::RNNT ||
           model_type == ParakeetModelType::TDT;
}

std::string format_int_list(const std::vector<int32_t> & values) {
    if (values.empty()) {
        return "";
    }
    if (values.size() == 1) {
        return std::to_string(values.front());
    }
    std::string text;
    for (size_t index = 0; index + 1 < values.size(); ++index) {
        if (index > 0) {
            text += ", ";
        }
        text += std::to_string(values[index]);
    }
    text += ", and " + std::to_string(values.back());
    return text;
}

std::vector<int32_t> frames_to_ms(const std::vector<int32_t> & frames, double frame_stride_ms) {
    std::vector<int32_t> ms;
    ms.reserve(frames.size());
    for (int32_t value : frames) {
        ms.push_back(static_cast<int32_t>(std::lround(value * frame_stride_ms)));
    }
    return ms;
}

int largest_allowed_at_most(const std::vector<int32_t> & allowed, int requested) {
    int best = -1;
    for (int32_t value : allowed) {
        if (value <= requested && value > best) best = value;
    }
    return best;
}

int smallest_allowed(const std::vector<int32_t> & allowed) {
    return *std::min_element(allowed.begin(), allowed.end());
}

int resolve_unified_chunk_frames(const ParakeetCtcModel & model, const StreamingOptions & opts) {
    const double frame_stride_ms = encoder_frame_stride_ms(model);
    const auto & allowed = model.unified_cfg.allowed_chunk_frames;
    const int requested = static_cast<int>(opts.chunk_ms / frame_stride_ms);
    const int snapped = largest_allowed_at_most(allowed, requested);
    const int resolved = snapped > 0 ? snapped : smallest_allowed(allowed);
    if (std::lround(resolved * frame_stride_ms) != opts.chunk_ms) {
        PARAKEET_LOG_INFO("Engine::stream_start: Unified chunk_ms=%d snapped to %ld ms (trained chunks: %s)\n",
                          opts.chunk_ms, std::lround(resolved * frame_stride_ms),
                          format_int_list(frames_to_ms(allowed, frame_stride_ms)).c_str());
    }
    return resolved;
}

int resolve_unified_right_context_frames(const ParakeetCtcModel & model, const StreamingOptions & opts) {
    const double frame_stride_ms = encoder_frame_stride_ms(model);
    const auto & allowed = model.unified_cfg.allowed_right_context_frames;
    const int requested = static_cast<int>(opts.right_lookahead_ms / frame_stride_ms);
    const int snapped = largest_allowed_at_most(allowed, requested);
    return snapped >= 0 ? snapped : smallest_allowed(allowed);
}

TdtDecodeOptions transducer_decode_options(const ParakeetCtcModel & model) {
    TdtDecodeOptions options;
    if (model.model_type == ParakeetModelType::RNNT) {
        options.max_symbols_per_step =
            model.encoder_cfg.rnnt_max_symbols_per_step;
    }
    return options;
}

int decode_transducer_window(const ParakeetCtcModel & model,
                             TdtRuntimeWeights & weights,
                             const float * encoder_out,
                             int frames,
                             int encoder_dim,
                             TdtDecodeState & state,
                             std::vector<int32_t> & tokens,
                             int & steps) {
    const TdtDecodeOptions options = transducer_decode_options(model);
    if (model.model_type == ParakeetModelType::RNNT) {
        return rnnt_decode_window(
            model, weights, encoder_out, frames, encoder_dim,
            options, state, tokens, steps);
    }
    return tdt_decode_window(
        model, weights, encoder_out, frames, encoder_dim,
        options, state, tokens, steps);
}

int decode_transducer(const ParakeetCtcModel & model,
                       TdtRuntimeWeights & weights,
                       const float * encoder_out,
                       int frames,
                       int encoder_dim,
                       TdtDecodeResult & result) {
    const TdtDecodeOptions options = transducer_decode_options(model);
    if (model.model_type == ParakeetModelType::RNNT) {
        return rnnt_greedy_decode(
            model, weights, encoder_out, frames, encoder_dim,
            options, result);
    }
    return tdt_greedy_decode(
        model, weights, encoder_out, frames, encoder_dim,
        options, result);
}

}  // namespace

// ── Long-form offline encoder windowing ───────────────────────────────────
// See EngineOptions::long_form_window_frames and src/long_form.h. The window
// policy itself (constants + LongFormPlan + the pure resolver) lives in
// long_form.h so the fit projector (parakeet_fit.cpp) shares it.

// Resolve the effective long-form window from EngineOptions + model config and
// decide whether the mel-spectrogram is long enough to need windowing at all.
// Inputs that fit within a single window keep the bit-identical single-pass path.
LongFormPlan resolve_long_form_plan(const ParakeetCtcModel & model,
                                    const EngineOptions & opts,
                                    int n_mel_frames) {
    int requested_window  = opts.long_form_window_frames;
    int requested_context = opts.long_form_context_frames;
    apply_long_form_hint(model.encoder_cfg.long_form_window_hint,
                         model.encoder_cfg.long_form_context_hint,
                         requested_window, requested_context);
    const LongFormPlan normal =
        resolve_long_form_plan_frames(requested_window, requested_context,
                                      model.encoder_cfg.pos_emb_max_len,
                                      model.encoder_cfg.subsampling_factor,
                                      n_mel_frames);

    // A negative window request explicitly disables automatic windowing. An
    // oversized fixed-shape Core ML call will then fail its capacity check and
    // run_encoder() will preserve the existing ggml single-pass fallback.
    if (opts.long_form_window_frames < 0) {
        return normal;
    }

    // A sidecar validated with its own seam context (the fixed-window Neural
    // Engine export) supplies it when the caller left the context automatic.
    const int sidecar_context = model_coreml_seam_context_frames(model);
    const int coreml_context_frames =
        opts.long_form_context_frames == 0 && sidecar_context > 0
            ? sidecar_context
            : opts.long_form_context_frames;

    LongFormPlan coreml;
    if (model.model_type == ParakeetModelType::CTC ||
        model.model_type == ParakeetModelType::RNNT ||
        model.model_type == ParakeetModelType::TDT) {
        coreml = resolve_coreml_fixed_shape_plan(
            model_coreml_fixed_mel_frames(model),
            coreml_context_frames,
            model.encoder_cfg.subsampling_factor,
            n_mel_frames);
    } else if (model.model_type == ParakeetModelType::EOU) {
        // Exact-shape EOU calls may use Core ML directly, but oversized inputs
        // stay on ggml until windowed causal/chunked attention parity is proven.
        coreml = resolve_coreml_exact_shape_plan(
            model_coreml_fixed_mel_frames(model),
            opts.long_form_context_frames,
            model.encoder_cfg.subsampling_factor,
            n_mel_frames);
    }

    // Respect a smaller user/model long-form limit, but force fixed-shape
    // windowing when that is the only way the complete input fits the sidecar.
    if (coreml.enabled &&
        (!normal.enabled || coreml.window_frames < normal.window_frames)) {
        return coreml;
    }
    return normal;
}

// Bounded-memory replacement for a single full-length run_encoder() call on long
// inputs. The caller computes the mel once (so per-feature CMVN statistics are
// global, matching the single-pass path); this slides the encoder over that mel
// in overlapping windows (plan_long_form_windows), trims the shared context at
// the seams (compute_window_trim), and concatenates the committed encoder frames
// (and CTC logits) into `out` so the decoder consumes them exactly as a
// single-pass result. Returns the run_encoder rc contract (0 = ok), or -6 if a
// committed range ever escapes its window buffer; honours `cancel_flag` between
// windows.
namespace {

// Holds the ggml encoder graph cache at one graph for the duration of a
// windowed pass. The windows share one shape, so a second slot would only
// keep an earlier, differently shaped graph's compute buffer alive.
struct SingleEncoderGraphScope {
    ParakeetCtcModel & model;
    size_t previous_limit;

    explicit SingleEncoderGraphScope(ParakeetCtcModel & m)
        : model(m), previous_limit(set_encoder_graph_cache_limit(m, 1)) {}
    ~SingleEncoderGraphScope() { set_encoder_graph_cache_limit(model, previous_limit); }
};

// Size the concatenation buffers once d_model is known. The centres tile
// [0, n_mel_frames) gap-free, so the committed total is n_mel_frames / sub up
// to one frame of rounding per window; reserving avoids geometric regrowth,
// whose last reallocation holds the old and new buffers at once.
void reserve_committed_frames(EncoderOutputs & out, int n_mel_frames, int sub,
                              size_t n_windows, bool with_logits) {
    const size_t est_frames = (size_t) (n_mel_frames / std::max(1, sub)) + n_windows + 1;
    out.encoder_out.reserve(est_frames * (size_t) out.d_model);
    if (with_logits && out.vocab_size > 0) {
        out.logits.reserve(est_frames * (size_t) out.vocab_size);
    }
}

bool same_span(const LongFormWindow & a, const LongFormWindow & b) {
    return a.window_start == b.window_start && a.window_len == b.window_len;
}

}  // namespace

int run_encoder_windowed(ParakeetCtcModel & model,
                         const float * mel, int n_mel_frames, int n_mels,
                         const LongFormPlan & plan,
                         std::atomic<bool> & cancel_flag,
                         EncoderOutputs & out,
                         WindowedEncoderStats & stats) {
    using clock = std::chrono::steady_clock;

    const int center_mel = plan.center_frames * plan.sub;
    const int left_ctx_mel = plan.left_context_frames * plan.sub;
    const int right_ctx_mel = plan.right_context_frames * plan.sub;
    const std::vector<LongFormWindow> windows =
        plan_long_form_windows_asymmetric(
            n_mel_frames, center_mel, left_ctx_mel, right_ctx_mel,
            plan.exact_mel_frames);

    out = EncoderOutputs{};
    SingleEncoderGraphScope one_graph(model);

    // A final centre shorter than the context gives the last two windows the
    // same span; it is encoded once and trimmed twice.
    EncoderOutputs win_out;
    const LongFormWindow * encoded = nullptr;
    for (const LongFormWindow & w : windows) {
        if (cancel_flag.load()) {
            break;
        }

        if (encoded == nullptr || !same_span(*encoded, w)) {
            const float * win_mel = mel + (size_t) w.window_start * (size_t) n_mels;

            const auto t_enc = clock::now();
            win_out = EncoderOutputs{};
            if (int rc = run_encoder(model, win_mel, w.window_len, n_mels, win_out,
                                     /*max_layers=*/-1,
                                     /*capture_intermediates=*/false,
                                     /*allow_coreml_padded=*/true);
                rc != 0) {
                return rc;
            }
            stats.encoder_ms += ms_since(t_enc);
            if (win_out.used_coreml) {
                ++stats.coreml_windows;
            } else {
                ++stats.ggml_windows;
            }
            encoded = &w;
        }

        if (out.d_model == 0) {
            out.d_model    = win_out.d_model;
            out.vocab_size = win_out.vocab_size;
            reserve_committed_frames(out, n_mel_frames, plan.sub, windows.size(),
                                     !win_out.logits.empty());
        }

        const WindowTrim trim =
            compute_window_trim(w, win_out.n_enc_frames, plan.sub);
        if (trim.center_cnt <= 0) {
            continue;
        }

        if (!append_committed_frames(out.encoder_out, win_out.encoder_out,
                                     trim.left_drop, trim.center_cnt, out.d_model)) {
            return -6;
        }
        if (!win_out.logits.empty() && out.vocab_size > 0 &&
            !append_committed_frames(out.logits, win_out.logits,
                                     trim.left_drop, trim.center_cnt, out.vocab_size)) {
            return -6;
        }
        out.n_enc_frames += trim.center_cnt;
    }

    out.used_coreml = stats.coreml_windows > 0 && stats.ggml_windows == 0;
    PARAKEET_LOG_INFO(
        "parakeet: long-form encoder windows=%zu coreml=%d ggml=%d\n",
        windows.size(), stats.coreml_windows, stats.ggml_windows);

    return 0;
}

namespace {

// ── Independent-segment batched decode (Core ML decoder sidecar) ─────────
//
// The Core ML TDT decoder decodes `lanes` windows per dispatch, so it needs
// windows with no dependency between them. The mel is cut at pauses into
// segments no longer than the encoder sidecar's window
// (src/independent_segments.h), each segment is encoded in one dispatch, and
// each segment's encoder output is decoded from a fresh prediction state.
// Note: the output can differ from the sequential decoder's.

// Segment cap when the encoder sidecar does not report a window of its own.
constexpr int    kSegmentMelFramesFallback = 1500;
constexpr double kSegmentPauseSearchMs     = 6000.0;
constexpr const char * kSegmentBatchDecodeEnv = "PARAKEET_ANE_BATCH_DECODE";
constexpr const char * kSegmentMsEnv          = "PARAKEET_ANE_SEGMENT_MS";
constexpr const char * kSegmentPauseSearchEnv = "PARAKEET_ANE_PAUSE_SEARCH_MS";
constexpr const char * kSegmentHostDecodeEnv  = "PARAKEET_ANE_DECODE_HOST";

struct IndependentSegmentsOutcome {
    std::vector<int32_t> ids;
    std::string          text;
    int                  encoder_frames = 0;
    bool                 used_coreml    = false;
    double encoder_ms = 0.0, decode_ms = 0.0, segment_plan_ms = 0.0, assemble_ms = 0.0;
};

bool env_flag_disabled(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && std::atoi(v) == 0;
}

bool env_flag_enabled(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && std::atoi(v) != 0;
}

// Segment cap in mel frames: the encoder sidecar's own dispatch window, so
// each segment is exactly one encoder dispatch, unless PARAKEET_ANE_SEGMENT_MS
// asks for another length.
int resolve_segment_frames(const ParakeetCtcModel & model, double mel_stride_ms) {
    int segment_frames = model_coreml_fixed_mel_frames(model);
    if (segment_frames <= 0) segment_frames = kSegmentMelFramesFallback;
    if (const char * env = std::getenv(kSegmentMsEnv)) {
        const double requested = std::atof(env);
        if (requested > 0.0) {
            segment_frames = std::max(1, (int) std::lround(requested / mel_stride_ms));
        }
    }
    return segment_frames;
}

int resolve_pause_search_frames(double mel_stride_ms) {
    double search_ms = kSegmentPauseSearchMs;
    if (const char * env = std::getenv(kSegmentPauseSearchEnv)) {
        const double requested = std::atof(env);
        if (requested >= 0.0) search_ms = requested;
    }
    return std::max(0, (int) std::lround(search_ms / mel_stride_ms));
}

// Encodes each segment on its own and appends every frame to `out`: a decoder
// starting from a fresh state needs frames that look like the start of an
// utterance, and unlike the overlapping long-form windows, independent
// segments have no seam context to trim. Returns false on an encoder failure
// or a cancel.
bool encode_segments(ParakeetCtcModel & model, const float * mel, int n_mels,
                     const std::vector<SegmentRange> & segments, int n_mel_frames,
                     std::atomic<bool> & cancel_flag, EncoderOutputs & out,
                     std::vector<TdtLaneWindow> & windows, IndependentSegmentsOutcome & outcome) {
    using clock = std::chrono::steady_clock;
    out = EncoderOutputs{};
    std::vector<int> frame_counts;
    frame_counts.reserve(segments.size());
    int coreml_segments = 0;

    for (const SegmentRange & r : segments) {
        if (cancel_flag.load()) return false;

        const auto t_enc = clock::now();
        EncoderOutputs seg;
        if (int rc = run_encoder(model, mel + (size_t) r.begin * (size_t) n_mels,
                                 r.end - r.begin, n_mels, seg,
                                 /*max_layers=*/-1,
                                 /*capture_intermediates=*/false,
                                 /*allow_coreml_padded=*/true); rc != 0) {
            PARAKEET_LOG_WARN("parakeet: independent-segment encode failed (rc=%d); "
                              "falling back to the single-pass path\n", rc);
            return false;
        }
        outcome.encoder_ms += ms_since(t_enc);
        if (seg.used_coreml) ++coreml_segments;

        const auto t_asm = clock::now();
        if (out.d_model == 0) {
            out.d_model    = seg.d_model;
            out.vocab_size = seg.vocab_size;
            reserve_committed_frames(out, n_mel_frames, model.encoder_cfg.subsampling_factor,
                                     segments.size(), /*with_logits=*/false);
        }
        if (!append_committed_frames(out.encoder_out, seg.encoder_out, 0, seg.n_enc_frames,
                                     out.d_model)) {
            return false;
        }
        frame_counts.push_back(seg.n_enc_frames);
        out.n_enc_frames += seg.n_enc_frames;
        outcome.assemble_ms += ms_since(t_asm);
    }

    // Window pointers are taken once the buffer is complete, so reallocation
    // during the appends cannot invalidate them.
    size_t start = 0;
    for (int frames : frame_counts) {
        windows.push_back({out.encoder_out.data() + start * (size_t) out.d_model, frames});
        start += (size_t) frames;
    }
    outcome.used_coreml = !segments.empty() && coreml_segments == (int) segments.size();
    return out.n_enc_frames > 0;
}

// Concatenates each window's transcript end to end. The windows tile the
// encoder output without overlap, so there is nothing to splice.
void stitch_segment_transcripts(const ParakeetCtcModel & model,
                                const std::vector<std::vector<int32_t>> & per_window,
                                IndependentSegmentsOutcome & outcome) {
    for (const std::vector<int32_t> & tokens : per_window) {
        outcome.ids.insert(outcome.ids.end(), tokens.begin(), tokens.end());
        const std::string piece = detokenize(model.vocab, tokens);
        if (piece.empty()) continue;
        if (!outcome.text.empty()) outcome.text.push_back(' ');
        outcome.text += piece;
    }
}

// PARAKEET_ANE_DECODE_HOST=1: the same windows through the sequential host
// decoder, for parity checks of the lane walk with identical windowing.
int decode_windows_on_host(const ParakeetCtcModel & model, TdtRuntimeWeights & rt,
                           const std::vector<TdtLaneWindow> & windows, int d_model,
                           const TdtDecodeOptions & dopts,
                           std::vector<std::vector<int32_t>> & per_window) {
    per_window.resize(windows.size());
    for (size_t i = 0; i < windows.size(); ++i) {
        TdtDecodeResult wres;
        if (int rc = tdt_greedy_decode(model, rt, windows[i].enc, windows[i].n_frames, d_model,
                                       dopts, wres); rc != 0) {
            return rc;
        }
        per_window[i] = std::move(wres.token_ids);
    }
    return 0;
}

#ifdef PARAKEET_USE_COREML
// The Core ML TDT decoder sidecar behind the lane walk's stepper interface.
struct CoremlLaneStepper final : TdtLaneStepper {
    parakeet_coreml_decoder * dec;
    int64_t n_lanes = 0, n_h_pred = 0, n_h_dim = 0, n_d_enc = 0, n_width = 0, n_v_out = 0;

    explicit CoremlLaneStepper(parakeet_coreml_decoder * d) : dec(d) {
        parakeet_coreml_decoder_geometry(dec, &n_lanes, &n_h_pred, &n_h_dim, &n_d_enc,
                                         &n_width, &n_v_out);
    }
    // width > 1 would mean the graph scores several frames per dispatch
    // against one prediction state, which needs a block-cache invalidation
    // rule the lane walk does not implement.
    bool usable() const { return n_width == 1; }

    int64_t lanes()  const override { return n_lanes; }
    int64_t h_pred() const override { return n_h_pred; }
    int64_t h_dim()  const override { return n_h_dim; }
    int64_t d_enc()  const override { return n_d_enc; }
    int64_t v_out()  const override { return n_v_out; }
    int step(const float * embed, const float * h_in, const float * c_in, const float * enc_step,
             float * logits, float * h_out, float * c_out) override {
        return parakeet_coreml_decoder_step(dec, embed, h_in, c_in, enc_step, logits, h_out, c_out);
    }
};

void log_segment_stats(const ParakeetCtcModel & model, const std::vector<SegmentRange> & segments,
                       const std::vector<TdtLaneWindow> & windows,
                       const std::vector<std::vector<int32_t>> & per_window,
                       const TdtLaneStats & stats, bool host) {
    size_t emitted = 0;
    for (const std::vector<int32_t> & tokens : per_window) emitted += tokens.size();
    PARAKEET_LOG_INFO("parakeet: independent segments=%zu dispatches=%d emitted=%zu rescued=%d host=%d\n",
                      windows.size(), stats.dispatches, emitted, stats.rescued, (int) host);
    for (size_t i = 0; i < windows.size(); ++i) {
        std::string piece = detokenize(model.vocab, per_window[i]);
        if (piece.size() > 70) piece = piece.substr(0, 70) + "...";
        PARAKEET_LOG_INFO("parakeet:   segment %2zu mel=[%6d,%6d) enc=%4d emit=%zu  %s\n",
                          i, segments[i].begin, segments[i].end, windows[i].n_frames,
                          per_window[i].size(), piece.c_str());
    }
}

// Pause-cut segments, one encoder dispatch each, then the batched lane decode
// on the Core ML decoder sidecar. Returns false, leaving `outcome` to be
// discarded, when there is no usable decoder sidecar, PARAKEET_ANE_BATCH_DECODE=0
// is set, or any step fails; the caller then runs the sequential path.
bool transcribe_independent_segments(ParakeetCtcModel & model, TdtRuntimeWeights & rt,
                                     const float * mel, int n_mel_frames, int n_mels,
                                     bool verbose, std::atomic<bool> & cancel_flag,
                                     IndependentSegmentsOutcome & outcome) {
    using clock = std::chrono::steady_clock;
    parakeet_coreml_decoder * dec = model_coreml_decoder(model);
    if (dec == nullptr || env_flag_disabled(kSegmentBatchDecodeEnv) || n_mel_frames <= 0) {
        return false;
    }
    CoremlLaneStepper stepper(dec);
    if (!stepper.usable()) {
        PARAKEET_LOG_WARN("parakeet: Core ML decoder was exported at width=%lld; only width=1 "
                          "is driven\n", (long long) stepper.n_width);
        return false;
    }
    const double mel_stride_ms = model.mel_cfg.hop_length > 0 && model.mel_cfg.sample_rate > 0
        ? 1000.0 * (double) model.mel_cfg.hop_length / (double) model.mel_cfg.sample_rate
        : 0.0;
    if (mel_stride_ms <= 0.0) return false;

    const auto t_plan = clock::now();
    const std::vector<float> energy = mel_frame_energy(mel, n_mel_frames, n_mels);
    const std::vector<SegmentRange> segments = plan_independent_segments(
        energy, n_mel_frames, resolve_segment_frames(model, mel_stride_ms),
        resolve_pause_search_frames(mel_stride_ms));
    outcome.segment_plan_ms = ms_since(t_plan);
    if (segments.empty()) return false;

    EncoderOutputs enc_out;
    std::vector<TdtLaneWindow> windows;
    if (!encode_segments(model, mel, n_mels, segments, n_mel_frames, cancel_flag, enc_out,
                         windows, outcome)) {
        return false;
    }

    const TdtDecodeOptions dopts;
    const bool host = env_flag_enabled(kSegmentHostDecodeEnv);
    std::vector<std::vector<int32_t>> per_window;
    TdtLaneStats stats;
    const auto t_dec = clock::now();
    const int rc = host
        ? decode_windows_on_host(model, rt, windows, enc_out.d_model, dopts, per_window)
        : tdt_decode_lanes(tdt_lane_geometry(model, rt), rt.embed, stepper, windows, dopts,
                           per_window, &stats);
    if (rc != 0) {
        PARAKEET_LOG_WARN("parakeet: Core ML batched decode failed (rc=%d); falling back to the "
                          "ggml decoder\n", rc);
        return false;
    }
    outcome.decode_ms = ms_since(t_dec);

    const auto t_stitch = clock::now();
    stitch_segment_transcripts(model, per_window, outcome);
    outcome.encoder_frames = enc_out.n_enc_frames;
    outcome.assemble_ms += ms_since(t_stitch);

    if (verbose) log_segment_stats(model, segments, windows, per_window, stats, host);
    return true;
}

EngineResult segments_result(IndependentSegmentsOutcome && seg, double preprocess_ms,
                             double total_ms, int n_samples, int sample_rate, int n_mel_frames) {
    EngineResult result;
    result.text            = std::move(seg.text);
    result.token_ids       = std::move(seg.ids);
    result.preprocess_ms   = preprocess_ms;
    result.encoder_ms      = seg.encoder_ms;
    result.decode_ms       = seg.decode_ms;
    result.segment_plan_ms = seg.segment_plan_ms;
    result.assemble_ms     = seg.assemble_ms;
    result.total_ms        = total_ms;
    result.audio_samples   = n_samples;
    result.sample_rate     = sample_rate;
    result.mel_frames      = n_mel_frames;
    result.encoder_frames  = seg.encoder_frames;
    result.encoder_used_coreml = seg.used_coreml;
    return result;
}
#endif  // PARAKEET_USE_COREML

}  // namespace

struct Engine::Impl {
    EngineOptions       opts;
    ParakeetCtcModel    model;
    std::atomic<bool>   cancel_flag{false};

    TdtRuntimeWeights   transducer_rt;
    bool                transducer_ready = false;
    int32_t             nemotron_prompt_id = -1;

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

static void prewarm_model(ParakeetCtcModel & model, float audio_seconds) {
    if (model.model_type != ParakeetModelType::NEMOTRON_DIARIZATION) {
        prewarm_encoder(model, audio_seconds);
        return;
    }
    if (const int rc = prewarm_nemotron_diarization(model, audio_seconds); rc != 0) {
        std::fprintf(stderr,
            "[parakeet] prewarm_nemotron_diarization: rc=%d -- "
            "first diarize will pay the cold cost the prewarm was meant to cover\n", rc);
    }
}

Engine::Engine(const EngineOptions & opts) : pimpl_(std::make_unique<Impl>()) {
    pimpl_->opts = opts;

    // Apply backend-init knobs before the first ggml call. Both are
    // process-singleton-scoped (the ggml-backend registry only ever
    // gets populated once per process; `$GGML_OPENCL_CACHE_DIR` is
    // read once by ggml-opencl at first init), so this is effectively
    // a "first Engine wins" race -- a second Engine with a different
    // backends_dir is logged + ignored by set_backends_directory().
    // Hosts that need per-Engine isolation should run each Engine in
    // its own subprocess.
    if (!opts.backends_dir.empty()) {
        set_backends_directory(opts.backends_dir);
    }
    if (!opts.opencl_cache_dir.empty()) {
        set_opencl_cache_dir(opts.opencl_cache_dir);
    }

    const int rc = load_from_gguf(opts.model_gguf_path,
                                  pimpl_->model,
                                  opts.n_threads,
                                  opts.n_gpu_layers,
                                  opts.verbose,
                                  opts.backend);
    if (rc != 0) {
        throw std::runtime_error("parakeet::Engine: failed to load GGUF '" +
                                 opts.model_gguf_path +
                                 "' with backend '" + opts.backend + "' (rc=" + std::to_string(rc) + ")");
    }

    if (pimpl_->model.model_type == ParakeetModelType::NEMOTRON) {
        pimpl_->nemotron_prompt_id = resolve_nemotron_prompt_id(
            pimpl_->model, opts.language);
    }
    if (is_transducer(pimpl_->model.model_type) ||
        pimpl_->model.model_type == ParakeetModelType::NEMOTRON) {
        if (tdt_prepare_runtime(pimpl_->model, pimpl_->transducer_rt) != 0) {
            throw std::runtime_error("Engine: transducer runtime preparation failed");
        }
        pimpl_->transducer_ready = true;
    }
    if (pimpl_->model.model_type == ParakeetModelType::EOU) {
        if (eou_prepare_runtime(pimpl_->model, pimpl_->eou_rt) != 0) {
            throw std::runtime_error("Engine: eou_prepare_runtime failed");
        }
        pimpl_->eou_ready = true;
    }
    if (pimpl_->model.model_type == ParakeetModelType::SORTFORMER ||
        pimpl_->model.model_type == ParakeetModelType::NEMOTRON_DIARIZATION) {
        pimpl_->sortformer_ready = true;
    }

    if (opts.prewarm) {
        prewarm_model(pimpl_->model, opts.prewarm_audio_seconds);
    }
}


Engine::~Engine() = default;
Engine::Engine(Engine &&) noexcept = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

const EngineOptions & Engine::options() const {
    return pimpl_->opts;
}

std::string Engine::model_type() const {
    return model_type_name(pimpl_->model.model_type);
}

bool Engine::is_diarization_model() const {
    return pimpl_->model.model_type == ParakeetModelType::SORTFORMER ||
           pimpl_->model.model_type == ParakeetModelType::NEMOTRON_DIARIZATION;
}

bool Engine::is_transcription_model() const {
    return pimpl_->model.model_type == ParakeetModelType::CTC ||
           pimpl_->model.model_type == ParakeetModelType::RNNT ||
           pimpl_->model.model_type == ParakeetModelType::TDT  ||
           pimpl_->model.model_type == ParakeetModelType::EOU  ||
           pimpl_->model.model_type == ParakeetModelType::NEMOTRON;
}

BackendDevice Engine::backend_device() const {
    if (backend_is_hexagon(pimpl_->model.backend_active())) return BackendDevice::NPU;
    return model_has_gpu_backend(pimpl_->model) ? BackendDevice::GPU
                                                : BackendDevice::CPU;
}

std::string Engine::backend_name() const {
    return model_active_backend_name(pimpl_->model);
}

std::string Engine::encoder_backend() const {
    return model_encoder_backend_name(pimpl_->model);
}

bool Engine::encoder_on_coreml() const {
    return model_encoder_on_coreml(pimpl_->model);
}

bool Engine::gpu_unsupported() const {
    return model_gpu_unsupported(pimpl_->model);
}

void Engine::cancel() {
    pimpl_->cancel_flag.store(true);
}

namespace {

std::vector<float> load_wav_or_throw(const std::string & wav_path, const char * where, int & sr) {
    std::vector<float> samples;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        throw std::runtime_error(std::string("parakeet::Engine::") + where +
                                 ": failed to load wav '" + wav_path +
                                 "' (rc=" + std::to_string(rc) + ")");
    }
    return samples;
}

void require_audio_for_model(const ParakeetCtcModel & model, const float * samples,
                             int n_samples, int sample_rate, const char * where) {
    if (!samples || n_samples <= 0) {
        throw std::runtime_error(std::string("parakeet::Engine::") + where + ": empty input");
    }
    if (sample_rate != model.mel_cfg.sample_rate) {
        throw std::runtime_error(std::string("parakeet::Engine::") + where + ": input is " +
                                 std::to_string(sample_rate) + " Hz but model expects " +
                                 std::to_string(model.mel_cfg.sample_rate) + " Hz");
    }
}

void require_transcription_model(const ParakeetCtcModel & model, const char * where) {
    if (model.model_type == ParakeetModelType::SORTFORMER) {
        throw std::runtime_error(
            std::string("parakeet::Engine::") + where + ": loaded GGUF is a Sortformer "
            "diarization model; call Engine::diarize() (or transcribe_with_speakers "
            "with a separate ASR engine) instead.");
    }
}

// The mel stage as a detached value. `state` is whichever scratch the caller
// owns: the Engine's for the transcribe entry points, a thread-local one for
// compute_mel, which may run concurrently with a transcription.
PrecomputedMel build_precomputed_mel(const ParakeetCtcModel & model, MelState & state,
                                     const float * samples, int n_samples, int sample_rate,
                                     const char * where) {
    const auto t_mel = std::chrono::steady_clock::now();
    PrecomputedMel out;
    if (int rc = compute_log_mel(samples, n_samples, model.mel_cfg, state,
                                 out.mel, out.n_frames); rc != 0) {
        throw std::runtime_error(std::string("parakeet::Engine::") + where +
                                 ": compute_log_mel failed (rc=" + std::to_string(rc) + ")");
    }
    out.n_mels        = model.mel_cfg.n_mels;
    out.audio_samples = n_samples;
    out.sample_rate   = sample_rate;
    out.preprocess_ms = ms_since(t_mel);
    return out;
}

}  // namespace

EngineResult Engine::transcribe(const std::string & wav_path) {
    int sr = 0;
    std::vector<float> samples = load_wav_or_throw(wav_path, "transcribe", sr);
    require_audio_for_model(pimpl_->model, samples.data(), (int) samples.size(), sr, "transcribe");
    require_transcription_model(pimpl_->model, "transcribe");
    pimpl_->cancel_flag.store(false);

    PrecomputedMel mel = build_precomputed_mel(pimpl_->model, pimpl_->mel_state,
                                               samples.data(), (int) samples.size(), sr,
                                               "transcribe");
    // Everything from here on reads the mel, never the audio, so neither the
    // waveform nor the STFT scratch is held across the encoder and decoder.
    std::vector<float>().swap(samples);
    pimpl_->mel_state.release_scratch();
    return transcribe_mel_impl(std::move(mel), mel.preprocess_ms);
}

EngineResult Engine::transcribe_samples(const float * samples, int n_samples, int sample_rate) {
    require_audio_for_model(pimpl_->model, samples, n_samples, sample_rate, "transcribe_samples");
    require_transcription_model(pimpl_->model, "transcribe_samples");
    pimpl_->cancel_flag.store(false);

    PrecomputedMel mel = build_precomputed_mel(pimpl_->model, pimpl_->mel_state,
                                               samples, n_samples, sample_rate,
                                               "transcribe_samples");
    pimpl_->mel_state.release_scratch();
    return transcribe_mel_impl(std::move(mel), mel.preprocess_ms);
}

PrecomputedMel Engine::compute_mel(const float * samples, int n_samples, int sample_rate) const {
    require_audio_for_model(pimpl_->model, samples, n_samples, sample_rate, "compute_mel");
    // Not pimpl_->mel_state: the Engine may be using that on another thread.
    // A thread_local keeps buffer reuse across calls without sharing state,
    // since the expected caller is one long-lived prefetch worker.
    static thread_local MelState tls_mel_state;
    return build_precomputed_mel(pimpl_->model, tls_mel_state, samples, n_samples, sample_rate,
                                 "compute_mel");
}

PrecomputedMel Engine::compute_mel_wav(const std::string & wav_path) const {
    const auto t_read = std::chrono::steady_clock::now();
    int sr = 0;
    const std::vector<float> samples = load_wav_or_throw(wav_path, "compute_mel_wav", sr);
    const double read_ms = ms_since(t_read);

    PrecomputedMel out = compute_mel(samples.data(), (int) samples.size(), sr);
    // The read is prefetched along with the mel, so it belongs to the same
    // hidden stage; folding it in keeps total_ms end-to-end on this path too.
    out.preprocess_ms += read_ms;
    return out;
}

namespace {

// A PrecomputedMel is a plain public struct, so the frame count, the bin count
// and the buffer it carries are checked against each other before the encoder
// reads it. The product is formed in size_t and checked by division so an
// oversized frame count cannot wrap into a plausible size.
void require_mel_buffer(const PrecomputedMel & mel, int expected_n_mels) {
    if (mel.empty()) {
        throw std::runtime_error("parakeet::Engine::transcribe_mel: empty mel");
    }
    if (mel.n_mels != expected_n_mels) {
        throw std::runtime_error("parakeet::Engine::transcribe_mel: mel has " +
                                 std::to_string(mel.n_mels) + " bins but model expects " +
                                 std::to_string(expected_n_mels));
    }
    const size_t frames   = (size_t) mel.n_frames;
    const size_t bins     = (size_t) mel.n_mels;
    const size_t expected = frames * bins;
    if (expected / bins != frames || mel.mel.size() != expected) {
        throw std::runtime_error("parakeet::Engine::transcribe_mel: mel buffer holds " +
                                 std::to_string(mel.mel.size()) + " values but " +
                                 std::to_string(mel.n_frames) + " frames x " +
                                 std::to_string(mel.n_mels) + " bins were declared");
    }
}

}  // namespace

EngineResult Engine::transcribe_mel(PrecomputedMel mel) {
    require_mel_buffer(mel, pimpl_->model.mel_cfg.n_mels);
    require_transcription_model(pimpl_->model, "transcribe_mel");
    pimpl_->cancel_flag.store(false);
    // The mel was timed on whichever thread built it; only the stages from
    // here on are this call's to measure.
    return transcribe_mel_impl(std::move(mel), mel.preprocess_ms);
}

EngineResult Engine::transcribe_mel_impl(PrecomputedMel && precomputed, double prior_ms) {
    using clock = std::chrono::steady_clock;
    const auto t_total = clock::now();

    std::vector<float> mel = std::move(precomputed.mel);
    const int    n_mel_frames  = precomputed.n_frames;
    const int    n_samples     = precomputed.audio_samples;
    const int    sample_rate   = precomputed.sample_rate;
    const double preprocess_ms = precomputed.preprocess_ms;

#ifdef PARAKEET_USE_COREML
    // With a TDT decoder sidecar the whole pass runs as pause-cut segments,
    // batched on the Neural Engine; anything below is the fallback.
    if (pimpl_->model.model_type == ParakeetModelType::TDT) {
        IndependentSegmentsOutcome seg;
        if (transcribe_independent_segments(pimpl_->model, pimpl_->transducer_rt, mel.data(),
                                            n_mel_frames, pimpl_->model.mel_cfg.n_mels,
                                            pimpl_->opts.verbose, pimpl_->cancel_flag, seg)) {
            return segments_result(std::move(seg), preprocess_ms, prior_ms + ms_since(t_total),
                                   n_samples, sample_rate, n_mel_frames);
        }
    }
#endif

    double encoder_ms = 0.0;
    EncoderOutputs enc_out;
    const LongFormPlan lf =
        resolve_long_form_plan(pimpl_->model, pimpl_->opts, n_mel_frames);

    const int exact_mel_frames = model_coreml_fixed_mel_frames(pimpl_->model);
    const bool cache_aware_nemotron =
        pimpl_->model.model_type == ParakeetModelType::NEMOTRON &&
        should_use_nemotron_cache_aware_offline(
                exact_mel_frames,
                pimpl_->opts.long_form_window_frames,
                lf.enabled,
                n_mel_frames);

    if (cache_aware_nemotron) {
        const auto & right_contexts =
            pimpl_->model.nemotron_cfg.allowed_right_context_frames;

        if (right_contexts.empty()) {
            throw std::runtime_error(
                    "Nemotron has no cache-aware operating point");
        }

        NemotronOfflineResult offline;
        const int rc = run_nemotron_cache_aware_offline(
                pimpl_->model,
                pimpl_->transducer_rt,
                mel.data(),
                n_mel_frames,
                pimpl_->model.mel_cfg.n_mels,
                pimpl_->opts.language,
                right_contexts.back(),
                pimpl_->cancel_flag,
                offline);
        if (rc != 0) {
            throw std::runtime_error(
                    "Nemotron cache-aware offline transcription failed (rc=" +
                    std::to_string(rc) + ")");
        }

        EngineResult result;
        result.text = std::move(offline.text);
        result.token_ids = std::move(offline.token_ids);
        result.preprocess_ms = preprocess_ms;
        result.encoder_ms = offline.encoder_ms;
        result.decode_ms = offline.decoder_ms;
        result.total_ms = prior_ms + ms_since(t_total);
        result.audio_samples = n_samples;
        result.sample_rate = sample_rate;
        result.mel_frames = n_mel_frames;
        result.encoder_frames = offline.encoder_frames;
        result.encoder_used_coreml = false;
        return result;
    }

    if (lf.enabled) {
        // Long input: window the encoder over the (global-CMVN) mel so the
        // O(T_enc^2) attention never allocates the full-length score tensor.
        WindowedEncoderStats wstats;
        if (int rc = run_encoder_windowed(pimpl_->model, mel.data(), n_mel_frames,
                                          pimpl_->model.mel_cfg.n_mels, lf,
                                          pimpl_->cancel_flag, enc_out, wstats);
            rc != 0) {
            throw std::runtime_error("parakeet::Engine::transcribe_samples: "
                                     "run_encoder_windowed failed (rc=" +
                                     std::to_string(rc) + ")");
        }
        encoder_ms = wstats.encoder_ms;
    } else {
        const auto t_enc = clock::now();
        if (int rc = run_encoder(pimpl_->model, mel.data(), n_mel_frames,
                                 pimpl_->model.mel_cfg.n_mels, enc_out,
                                 /*max_layers=*/-1,
                                 /*capture_intermediates=*/false,
                                 /*allow_coreml_padded=*/true); rc != 0) {
            throw std::runtime_error("parakeet::Engine::transcribe_samples: run_encoder failed (rc=" +
                                     std::to_string(rc) + ")");
        }
        encoder_ms = ms_since(t_enc);
    }

    // A cancel() during the windowed encoder loop can leave no committed
    // frames; decoding zero frames is meaningless, so return an empty result.
    if (enc_out.n_enc_frames <= 0) {
        EngineResult result;
        result.preprocess_ms  = preprocess_ms;
        result.encoder_ms     = encoder_ms;
        result.total_ms       = prior_ms + ms_since(t_total);
        result.audio_samples  = n_samples;
        result.sample_rate    = sample_rate;
        result.mel_frames     = n_mel_frames;
        result.encoder_frames = 0;
        result.encoder_used_coreml = enc_out.used_coreml;
        return result;
    }

    const auto t_dec = clock::now();
    std::vector<int32_t> ids;
    std::string text;
    std::vector<float> conditioned_encoder;
    const float * decoder_input = enc_out.encoder_out.data();
    if (pimpl_->model.model_type == ParakeetModelType::NEMOTRON) {
        if (int rc = run_nemotron_prompt_projection(
                pimpl_->model,
                enc_out.encoder_out.data(),
                enc_out.n_enc_frames,
                enc_out.d_model,
                pimpl_->nemotron_prompt_id,
                conditioned_encoder); rc != 0) {
            throw std::runtime_error(
                "parakeet::Engine::transcribe_samples: Nemotron prompt "
                "projection failed (rc=" + std::to_string(rc) + ")");
        }
        decoder_input = conditioned_encoder.data();

        RnntDecodeOptions options;
        options.max_symbols_per_step =
            pimpl_->model.nemotron_cfg.max_symbols_per_step;
        RnntDecodeResult result;
        const int rc = lf.enabled
            ? rnnt_greedy_decode_chunked(
                pimpl_->model, pimpl_->transducer_rt, decoder_input,
                enc_out.n_enc_frames, enc_out.d_model,
                std::max(1, lf.center_frames), options, result)
            : rnnt_greedy_decode(
                pimpl_->model, pimpl_->transducer_rt, decoder_input,
                enc_out.n_enc_frames, enc_out.d_model, options, result);
        if (rc != 0) {
            throw std::runtime_error(
                "parakeet::Engine::transcribe_samples: Nemotron RNNT decode "
                "failed (rc=" + std::to_string(rc) + ")");
        }
        ids = std::move(result.token_ids);
        text = std::move(result.text);
    } else if (is_transducer(pimpl_->model.model_type)) {
        TdtDecodeResult  dres;
        if (int rc = decode_transducer(
                pimpl_->model, pimpl_->transducer_rt,
                decoder_input,
                enc_out.n_enc_frames, enc_out.d_model,
                dres); rc != 0) {
            throw std::runtime_error("parakeet::Engine::transcribe_samples: transducer decode failed (rc=" +
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
        const CtcDecodeOptions dopts =
            resolve_ctc_decode_options(pimpl_->model, pimpl_->opts.language);
        ids  = ctc_greedy_decode(enc_out.logits.data(), enc_out.n_enc_frames,
                                 pimpl_->model.vocab_size, pimpl_->model.blank_id,
                                 &dopts);
        text = detokenize(pimpl_->model.vocab, ids);
    }
    const double decode_ms = ms_since(t_dec);

    EngineResult result;
    result.text           = std::move(text);
    result.token_ids      = std::move(ids);
    result.preprocess_ms  = preprocess_ms;
    result.encoder_ms     = encoder_ms;
    result.decode_ms      = decode_ms;
    result.total_ms       = prior_ms + ms_since(t_total);
    result.audio_samples  = n_samples;
    result.sample_rate    = sample_rate;
    result.mel_frames     = n_mel_frames;
    result.encoder_frames = enc_out.n_enc_frames;
    result.encoder_used_coreml = enc_out.used_coreml;
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
    if (pimpl_->model.model_type == ParakeetModelType::NEMOTRON ||
        pimpl_->model.unified_cfg.available) {
        pimpl_->cancel_flag.store(false);

        const auto started = std::chrono::steady_clock::now();
        const double frame_stride_ms =
            encoder_frame_stride_ms(pimpl_->model);

        EngineResult result;
        result.audio_samples = n_samples;
        result.sample_rate = sample_rate;
        result.mel_frames = std::max(
            1,
            n_samples / pimpl_->model.mel_cfg.hop_length);

        StreamingOptions session_options = opts;
        session_options.sample_rate = sample_rate;

        auto session = stream_start(
            session_options,
            [&](const StreamingSegment & segment) {
                result.token_ids.insert(
                    result.token_ids.end(),
                    segment.token_ids.begin(),
                    segment.token_ids.end());

                result.text = detokenize(
                    pimpl_->model.vocab,
                    result.token_ids);

                result.encoder_ms += segment.encoder_ms;
                result.decode_ms += segment.decode_ms;
                result.encoder_frames = std::max(
                    result.encoder_frames,
                    static_cast<int>(std::llround(
                        segment.end_s * 1000.0 /
                        frame_stride_ms)));

                if (on_segment) {
                    on_segment(segment);
                }
            });

        session->feed_pcm_f32(samples, n_samples);
        session->finalize();

        result.total_ms = ms_since(started);
        return result;
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

    double encoder_ms = 0.0;
    EncoderOutputs enc_out;
    const LongFormPlan lf =
        resolve_long_form_plan(pimpl_->model, pimpl_->opts, n_mel_frames);
    if (lf.enabled) {
        // Long input: window the encoder over the (global-CMVN) mel for bounded
        // memory. The per-window decode below walks the stitched encoder output.
        WindowedEncoderStats wstats;
        if (int rc = run_encoder_windowed(pimpl_->model, mel.data(), n_mel_frames,
                                          pimpl_->model.mel_cfg.n_mels, lf,
                                          pimpl_->cancel_flag, enc_out, wstats);
            rc != 0) {
            throw std::runtime_error("parakeet::Engine::transcribe_samples_stream: "
                                     "run_encoder_windowed failed (rc=" +
                                     std::to_string(rc) + ")");
        }
        encoder_ms = wstats.encoder_ms;
    } else {
        const auto t_enc = clock::now();
        if (int rc = run_encoder(pimpl_->model, mel.data(), n_mel_frames,
                                 pimpl_->model.mel_cfg.n_mels, enc_out,
                                 /*max_layers=*/-1,
                                 /*capture_intermediates=*/false,
                                 /*allow_coreml_padded=*/true); rc != 0) {
            throw std::runtime_error("parakeet::Engine::transcribe_samples_stream: run_encoder failed (rc=" +
                                     std::to_string(rc) + ")");
        }
        encoder_ms = ms_since(t_enc);
    }

    const int T_enc = enc_out.n_enc_frames;
    const int vocab = pimpl_->model.vocab_size;
    const int blank = pimpl_->model.blank_id;
    const CtcDecodeOptions ctc_dopts =
        resolve_ctc_decode_options(pimpl_->model, pimpl_->opts.language);

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
    result.encoder_used_coreml = enc_out.used_coreml;

    const auto t_dec = clock::now();

    const bool has_transducer = is_transducer(pimpl_->model.model_type);
    const bool is_eou = (pimpl_->model.model_type == ParakeetModelType::EOU);

    int32_t prev_token = -1;
    TdtDecodeState transducer_state;
    EouDecodeState eou_state;
    if (has_transducer) {
        tdt_init_state(
            pimpl_->transducer_rt,
            (int) pimpl_->model.blank_id,
            transducer_state);
    }
    if (is_eou && eou_init_state(pimpl_->eou_rt, eou_state) != 0) {
        throw std::runtime_error("Engine: eou_init_state failed");
    }

    int chunk_index = 0;
    bool first_segment = true;

    for (int start = 0; start < T_enc; start += frames_per_window) {
        if (pimpl_->cancel_flag.load()) break;

        int end = start + frames_per_window;
        if (end > T_enc) end = T_enc;

        const auto t_win = clock::now();

        std::vector<int32_t> win_tokens;
        int eou_boundaries_in_chunk = 0;
        if (has_transducer) {
            int steps = 0;
            const float * win_enc = enc_out.encoder_out.data()
                                  + static_cast<size_t>(start) * enc_out.d_model;
            if (int rc = decode_transducer_window(
                    pimpl_->model, pimpl_->transducer_rt,
                    win_enc, end - start, enc_out.d_model,
                    transducer_state, win_tokens, steps);
                rc != 0) {
                throw std::runtime_error("parakeet::Engine::transcribe_samples_stream: "
                                         "transducer decode failed (rc=" + std::to_string(rc) + ")");
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
                                     prev_token, win_tokens, nullptr,
                                     &ctc_dopts);
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
            seg.starts_word = win_tokens.empty()
                ? true
                : token_is_word_start(pimpl_->model.vocab, win_tokens.front());
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

    SortformerDiarizationOptions sopts;
    sopts.threshold = opts.threshold;
    SortformerDiarizationResult dres;

    const auto t_enc = clock::now();
    double encoder_ms = 0.0;
    if (sortformer_diarize_mel_coreml(impl.model, mel.data(), n_mel_frames, sopts, dres)) {
        encoder_ms = ms_since(t_enc);
        dres.decode_ms = 0.0;
    } else {
        EncoderOutputs enc_out;
        if (int rc = run_encoder(impl.model, mel.data(), n_mel_frames,
                                 impl.model.mel_cfg.n_mels, enc_out,
                                 /*max_layers=*/-1,
                                 /*capture_intermediates=*/false,
                                 /*allow_coreml_padded=*/false); rc != 0) {
            throw std::runtime_error("diarize: run_encoder failed (rc=" +
                                     std::to_string(rc) + ")");
        }
        encoder_ms = ms_since(t_enc);

        int diarize_rc = sortformer_diarize_ggml(impl.model,
                                                 enc_out.encoder_out.data(),
                                                 enc_out.n_enc_frames, enc_out.d_model,
                                                 sopts, dres);
        if (diarize_rc != 0) {
            throw std::runtime_error("diarize: sortformer_diarize failed (rc=" +
                                     std::to_string(diarize_rc) + ")");
        }
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

// AOSC streaming variant of engine_impl_diarize_helper. NeMo-faithful port of
// `forward_streaming_step` + `streaming_update`:
//   1. compute_log_mel on the chunk audio (which already includes lc/rc context)
//   2. run_subsampling -> chunk_pre_encode_embs (post-subsampling, 512-d)
//   3. sortformer_aosc_step assembles [spkcache | fifo | chunk_pre_encode],
//      runs the conformer layers via run_encoder_bypass_pre_encode, then the
//      diariser head, then streaming_update on the resulting preds + new chunk
//
// Returned segments are chunk-relative (start_s == 0 at the START OF THE
// committed chunk -- the lc_enc frames at the head of the encoder output are
// dropped before thresholding).
static DiarizationResult engine_impl_diarize_streaming_helper(
    Engine::Impl & impl,
    const float * samples, int n_samples,
    int sample_rate,
    const DiarizationOptions & opts,
    SortformerSpeakerCache & cache,
    const SortformerStreamingConfig & cfg,
    int lc_enc_frames_expected, int rc_enc_frames_expected) {
    if (!samples || n_samples <= 0) {
        throw std::runtime_error("diarize_streaming: empty input");
    }
    if (sample_rate != impl.model.mel_cfg.sample_rate) {
        throw std::runtime_error("diarize_streaming: input is " +
                                 std::to_string(sample_rate) + " Hz but model expects " +
                                 std::to_string(impl.model.mel_cfg.sample_rate) + " Hz");
    }
    if (impl.model.model_type != ParakeetModelType::SORTFORMER || !impl.sortformer_ready) {
        throw std::runtime_error("diarize_streaming: loaded GGUF is not a Sortformer model");
    }

    impl.cancel_flag.store(false);

    using clock = std::chrono::steady_clock;
    const auto t_total = clock::now();

    // No per-chunk peak normalisation: amplitude consistency across chunks
    // matters for the cache embeddings to remain in-distribution.
    std::vector<float> work(samples, samples + n_samples);

    const auto t_mel = clock::now();
    std::vector<float> mel;
    int n_mel_frames = 0;
    if (int rc = compute_log_mel(work.data(), n_samples, impl.model.mel_cfg,
                                 impl.mel_state, mel, n_mel_frames); rc != 0) {
        throw std::runtime_error("diarize_streaming: compute_log_mel failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const double preprocess_ms = ms_since(t_mel);

    // Subsampling only -- the cache concat happens BEFORE the conformer layers.
    const auto t_enc = clock::now();
    std::vector<float> pre_encode;
    int n_pre_encode_frames = 0;
    if (int rc = run_subsampling(impl.model, mel.data(), n_mel_frames,
                                 impl.model.mel_cfg.n_mels,
                                 pre_encode, n_pre_encode_frames); rc != 0) {
        throw std::runtime_error("diarize_streaming: run_subsampling failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const int D = impl.model.encoder_cfg.d_model;

    // Reconcile expected lc/rc encoder frames with what subsampling actually
    // produced. If subsampling returned fewer frames than expected (tail-chunk
    // with insufficient right context), shrink rc to what fits and let
    // chunk_len_eff absorb the leftover.
    int lc = lc_enc_frames_expected;
    int rc = rc_enc_frames_expected;
    if (lc + rc > n_pre_encode_frames) {
        rc = std::max(0, n_pre_encode_frames - lc);
        if (lc + rc > n_pre_encode_frames) {
            lc = std::max(0, n_pre_encode_frames - rc);
        }
    }
    int chunk_len_eff = n_pre_encode_frames - lc - rc;
    if (chunk_len_eff <= 0) {
        DiarizationResult result;
        result.n_frames       = 0;
        result.num_spks       = impl.model.encoder_cfg.sortformer_num_spks;
        result.frame_stride_s = (double)(impl.model.mel_cfg.hop_length *
                                         impl.model.encoder_cfg.subsampling_factor) /
                                (double)impl.model.mel_cfg.sample_rate;
        result.audio_samples  = n_samples;
        result.sample_rate    = sample_rate;
        result.preprocess_ms  = preprocess_ms;
        result.encoder_ms     = ms_since(t_enc);
        result.total_ms       = ms_since(t_total);
        return result;
    }

    SortformerDiarizationOptions s_opts;
    s_opts.threshold = opts.threshold;
    SortformerDiarizationResult dres;

    if (int rc_ = sortformer_aosc_step(impl.model,
                                       pre_encode.data(),
                                       n_pre_encode_frames, D,
                                       lc, rc, chunk_len_eff,
                                       cache, cfg, s_opts, dres);
        rc_ != 0) {
        throw std::runtime_error("diarize_streaming: sortformer_aosc_step failed (rc=" +
                                 std::to_string(rc_) + ")");
    }

    const double encoder_ms = ms_since(t_enc) - dres.decode_ms;

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

static void append_nemotron_diarization_segments(
    DiarizationResult & result, const DiarizationOptions & options) {
    std::vector<SortformerSegment> segments;
    sf_threshold_segments(result.speaker_probs, result.n_frames,
        result.num_spks, result.frame_stride_s, options.threshold, segments);
    const double minimum_seconds = options.min_segment_ms / 1000.0;
    for (const auto & segment : segments) {
        if (segment.end_s - segment.start_s < minimum_seconds) continue;
        result.segments.push_back({segment.speaker_id, segment.start_s, segment.end_s});
    }
}

static float nemotron_audio_gain(const float * samples, int n_samples) {
    const float peak = std::fabs(*std::max_element(samples, samples + n_samples,
        [](float lhs, float rhs) { return std::fabs(lhs) < std::fabs(rhs); }));
    return 1.0f / (peak + 1.0e-3f);
}

static std::vector<float> normalize_nemotron_audio(
    const float * samples, int n_samples, float gain) {
    std::vector<float> normalized(samples, samples + n_samples);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
        [gain](float sample) { return sample * gain; });
    return normalized;
}

static DiarizationResult engine_impl_diarize_nemotron_with_gain(
    Engine::Impl & impl, const float * samples, int n_samples,
    int sample_rate, const DiarizationOptions & options, float gain) {
    if (!samples || n_samples <= 0 || sample_rate != impl.model.mel_cfg.sample_rate) {
        throw std::runtime_error("Nemotron diarization requires nonempty 16 kHz audio");
    }
    const auto start = std::chrono::steady_clock::now();
    const std::vector<float> normalized = normalize_nemotron_audio(samples, n_samples, gain);
    std::vector<float> mel;
    int mel_frames = 0;
    if (compute_log_mel(normalized.data(), n_samples, impl.model.mel_cfg,
        impl.mel_state, mel, mel_frames) != 0) {
        throw std::runtime_error("Nemotron diarization mel preprocessing failed");
    }
    const double preprocess_ms = ms_since(start);
    const auto inference_start = std::chrono::steady_clock::now();
    DiarizationResult result;
    if (run_nemotron_diarization(impl.model, mel.data(), mel_frames,
        result.speaker_probs) != 0) {
        throw std::runtime_error("Nemotron diarization inference failed");
    }
    result.decode_ms = ms_since(inference_start);
    result.n_frames = mel_frames;
    result.num_spks = impl.model.nemotron_diarization_cfg.speakers;
    result.speaker_probs.resize(static_cast<size_t>(result.n_frames) * result.num_spks);
    result.frame_stride_s = static_cast<double>(impl.model.mel_cfg.hop_length) / sample_rate;
    result.audio_samples = n_samples;
    result.sample_rate = sample_rate;
    result.preprocess_ms = preprocess_ms;
    append_nemotron_diarization_segments(result, options);
    result.total_ms = ms_since(start);
    return result;
}

static DiarizationResult engine_impl_diarize_nemotron(
    Engine::Impl & impl, const float * samples, int n_samples,
    int sample_rate, const DiarizationOptions & options) {
    if (!samples || n_samples <= 0) {
        throw std::runtime_error("Nemotron diarization requires nonempty 16 kHz audio");
    }
    return engine_impl_diarize_nemotron_with_gain(
        impl, samples, n_samples, sample_rate, options,
        nemotron_audio_gain(samples, n_samples));
}

static void average_nemotron_coarse_probabilities(
    const NemotronDiarizationChunk & chunk, int speakers, int factor,
    std::vector<float> & coarse) {
    const int frames = chunk.state_frames + chunk.chunk_frames;
    coarse.assign(static_cast<size_t>(frames) * speakers, 0.0f);
    for (int frame = 0; frame < frames; ++frame) {
        for (int step = 0; step < factor; ++step) {
            for (int speaker = 0; speaker < speakers; ++speaker) {
                coarse[static_cast<size_t>(frame) * speakers + speaker] +=
                    chunk.probabilities[
                        static_cast<size_t>(frame * factor + step) * speakers + speaker] /
                    factor;
            }
        }
    }
}

static void append_nemotron_cache_state(
    const SortformerSpeakerCache & cache, std::vector<float> & state) {
    state = cache.spkcache;
    state.insert(state.end(), cache.fifo.begin(), cache.fifo.end());
}

static DiarizationResult engine_impl_diarize_nemotron_streaming(
    Engine::Impl & impl, const float * samples, int n_samples,
    int sample_rate, const DiarizationOptions & options,
    SortformerSpeakerCache & cache, const SortformerStreamingConfig & config,
    int actual_left_context_samples, int right_context, int committed_samples,
    float gain) {
    const auto started = std::chrono::steady_clock::now();
    const std::vector<float> normalized = normalize_nemotron_audio(samples, n_samples, gain);
    std::vector<float> mel;
    int mel_frames = 0;
    if (compute_log_mel(normalized.data(), n_samples, impl.model.mel_cfg,
        impl.mel_state, mel, mel_frames) != 0) {
        throw std::runtime_error("Nemotron streaming mel preprocessing failed");
    }
    const int hop = impl.model.mel_cfg.hop_length;
    mel_frames = std::min(mel_frames, (n_samples + hop - 1) / hop);
    const int factor = impl.model.nemotron_diarization_cfg.subsampling_factor;
    const int chunk_frames = (mel_frames + factor - 1) / factor;
    const int left = std::min(
        (actual_left_context_samples + hop * factor - 1) / (hop * factor), chunk_frames);
    const int requested_committed =
        (committed_samples + hop * factor - 1) / (hop * factor);
    const int right = std::min(right_context,
        std::max(0, chunk_frames - left - requested_committed));
    const int committed = std::min(requested_committed,
        chunk_frames - left - right);
    DiarizationResult result;
    result.num_spks = impl.model.nemotron_diarization_cfg.speakers;
    result.frame_stride_s = static_cast<double>(hop) / sample_rate;
    result.sample_rate = sample_rate;
    result.audio_samples = committed_samples;
    result.preprocess_ms = ms_since(started);
    if (committed <= 0) return result;

    const int previous_speaker_frames = cache.n_rows;
    const int previous_fifo_frames = cache.n_fifo;
    std::vector<float> state;
    append_nemotron_cache_state(cache, state);
    NemotronDiarizationChunk chunk;
    const float * state_data = state.empty() ? nullptr : state.data();
    if (run_nemotron_diarization_chunk(impl.model, mel.data(), mel_frames,
        state_data, previous_speaker_frames + previous_fifo_frames, chunk) != 0) {
        throw std::runtime_error("Nemotron streaming inference failed");
    }
    std::vector<float> coarse;
    average_nemotron_coarse_probabilities(chunk, result.num_spks, factor, coarse);
    const int committed_offset = chunk.state_frames + left;
    const int native_offset = committed_offset * factor;
    result.n_frames = std::min(committed * factor,
        (committed_samples + hop - 1) / hop);
    result.speaker_probs.assign(
        chunk.probabilities.begin() + static_cast<size_t>(native_offset) * result.num_spks,
        chunk.probabilities.begin() +
            static_cast<size_t>(native_offset + result.n_frames) * result.num_spks);
    append_nemotron_diarization_segments(result, options);
    sortformer_update_speaker_cache(cache,
        chunk.embeddings.data() + static_cast<size_t>(left) *
            impl.model.nemotron_diarization_cfg.encoder_width,
        committed, coarse.data(), previous_speaker_frames,
        previous_fifo_frames, left, result.num_spks,
        impl.model.nemotron_diarization_cfg.encoder_width, config);
    result.total_ms = ms_since(started);
    result.encoder_ms = result.total_ms - result.preprocess_ms;
    return result;
}

static DiarizationResult engine_impl_diarize_nemotron_long_form(
    Engine::Impl & impl, const float * samples, int n_samples,
    int sample_rate, const DiarizationOptions & options);

static bool nemotron_offline_uses_long_form(
    const Engine::Impl & impl, const float * samples, int n_samples) {
    return samples && n_samples > 0 && nemotron_diarization_uses_long_form(
        impl.model, impl.opts.long_form_window_frames,
        nemotron_diarization_mel_frames(impl.model, n_samples));
}

DiarizationResult Engine::diarize_samples(const float * samples,
                                          int n_samples,
                                          int sample_rate,
                                          const DiarizationOptions & opts) {
    if (pimpl_->model.model_type == ParakeetModelType::NEMOTRON_DIARIZATION) {
        if (nemotron_offline_uses_long_form(*pimpl_, samples, n_samples)) {
            return engine_impl_diarize_nemotron_long_form(
                *pimpl_, samples, n_samples, sample_rate, opts);
        }
        return engine_impl_diarize_nemotron(*pimpl_, samples, n_samples, sample_rate, opts);
    }
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
    TdtDecodeState transducer_state;
    EouDecodeState eou_state;
    std::unique_ptr<NemotronStreamState> nemotron_state;
    std::unique_ptr<UnifiedStreamState> unified_state;

    std::string             cumulative_text;
    std::vector<int32_t>    cumulative_token_ids;

    bool finalized = false;
    bool cancelled = false;

    // Optional EnergyVad for CTC/RNN-T/TDT when enable_energy_vad and no native VAD exists.
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

    void drain_nemotron(bool finalize);
    void drain_unified(bool finalize);
    void emit_native_segment(int64_t start_frame,
                             int64_t end_frame,
                             const std::vector<int32_t> & new_token_ids,
                             const std::string & text,
                             double encoder_ms);

    void cancel_session() {
        cancelled = true;

        if (nemotron_state) {
            cancel_nemotron_stream(*nemotron_state);
        }
        if (unified_state) {
            cancel_unified_stream(*unified_state);
        }
    }
};

void StreamSession::Impl::drain_nemotron(bool finalize) {
    if (!nemotron_state) {
        throw std::runtime_error(
            "StreamSession: Nemotron stream state is not initialized");
    }

    auto & model = engine_impl->model;
    auto & state = *nemotron_state;

    while (!cancelled) {
        if (engine_impl->cancel_flag.load()) {
            cancel_session();
            break;
        }

        std::vector<float> processed_signal;
        int n_mel_frames = 0;

        const int ready = next_nemotron_processed_signal(
            state,
            model.mel_cfg.n_mels,
            finalize,
            processed_signal,
            n_mel_frames);

        if (ready < 0) {
            throw std::runtime_error(
                "StreamSession: preparing Nemotron chunk failed "
                "(rc=" + std::to_string(ready) + ")");
        }
        if (ready == 0) {
            break;
        }

        const int64_t start_encoder_frame =
            state.emitted_encoder_frames;
        const auto started = std::chrono::steady_clock::now();

        const bool last_chunk =
            finalize && nemotron_pending_mel_frames(state) == 0;

        NemotronStreamStepResult result;
        const int rc = run_nemotron_stream_step(
            model,
            engine_impl->transducer_rt,
            processed_signal.data(),
            n_mel_frames,
            model.mel_cfg.n_mels,
            last_chunk,
            state,
            result);

        if (rc != 0) {
            throw std::runtime_error(
                "StreamSession: Nemotron stream step failed "
                "(rc=" + std::to_string(rc) + ")");
        }

        emit_native_segment(
            start_encoder_frame,
            state.emitted_encoder_frames,
            result.new_token_ids,
            result.text,
            ms_since(started));

        if (last_chunk) {
            break;
        }
    }
}

void StreamSession::Impl::emit_native_segment(int64_t start_frame,
                                              int64_t end_frame,
                                              const std::vector<int32_t> & new_token_ids,
                                              const std::string & text,
                                              double encoder_ms) {
    const size_t previous_text_size = cumulative_text.size();
    cumulative_token_ids.insert(
        cumulative_token_ids.end(), new_token_ids.begin(), new_token_ids.end());
    cumulative_text = text;
    ++chunk_index;
    if (!on_segment) return;

    const auto & model = engine_impl->model;
    const double frame_stride_s = encoder_frame_stride_ms(model) / 1000.0;
    StreamingSegment segment;
    segment.text = cumulative_text.substr(previous_text_size);
    segment.token_ids = new_token_ids;
    segment.start_s = static_cast<double>(start_frame) * frame_stride_s;
    segment.end_s = static_cast<double>(end_frame) * frame_stride_s;
    segment.chunk_index = chunk_index - 1;
    segment.is_final = true;
    segment.starts_word = new_token_ids.empty()
        ? true
        : token_is_word_start(model.vocab, new_token_ids.front());
    segment.encoder_ms = encoder_ms;
    segment.decode_ms = 0.0;
    on_segment(segment);
}

void StreamSession::Impl::drain_unified(bool finalize) {
    if (!unified_state) {
        throw std::runtime_error("StreamSession: Unified stream state is not initialized");
    }
    auto & model = engine_impl->model;
    auto & state = *unified_state;
    std::vector<float> processed_signal;

    while (!cancelled) {
        if (engine_impl->cancel_flag.load()) {
            cancel_session();
            break;
        }
        int n_mel_frames = 0;
        const int ready = next_unified_processed_signal(
            state, model.mel_cfg.n_mels, finalize, processed_signal, n_mel_frames);
        if (ready < 0) {
            throw std::runtime_error(
                "StreamSession: preparing Unified chunk failed (rc=" + std::to_string(ready) + ")");
        }
        if (ready == 0) break;

        const int64_t start_frame = state.emitted_encoder_frames;
        const auto started = std::chrono::steady_clock::now();
        const bool last_chunk = finalize && unified_pending_mel_frames(state) == 0;

        UnifiedStreamStepResult result;
        const int rc = run_unified_stream_step(
            model, engine_impl->transducer_rt, processed_signal.data(), n_mel_frames,
            model.mel_cfg.n_mels, last_chunk, state, result);
        if (rc != 0) {
            throw std::runtime_error(
                "StreamSession: Unified stream step failed (rc=" + std::to_string(rc) + ")");
        }
        emit_native_segment(start_frame, state.emitted_encoder_frames,
                            result.new_token_ids, result.text, ms_since(started));
        if (last_chunk) break;
    }
}

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
    if (is_transducer(engine_impl->model.model_type)) {
        int steps = 0;
        const int n_frames = std::max(0, center_end_frame - left_drop_frames);
        const float * win_enc = enc_out.encoder_out.data()
                              + static_cast<size_t>(left_drop_frames) * enc_out.d_model;
        if (int rc = decode_transducer_window(
                engine_impl->model, engine_impl->transducer_rt,
                win_enc, n_frames, enc_out.d_model,
                transducer_state, win_tokens, steps);
            rc != 0) {
            throw std::runtime_error("StreamSession: transducer decode failed (rc=" +
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
        const CtcDecodeOptions ctc_dopts =
            resolve_ctc_decode_options(engine_impl->model, engine_impl->opts.language);
        ctc_greedy_decode_window(enc_out.logits.data(),
                                 left_drop_frames, center_end_frame,
                                 engine_impl->model.vocab_size,
                                 engine_impl->model.blank_id,
                                 prev_token, win_tokens, nullptr,
                                 &ctc_dopts);
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
        seg.starts_word = win_tokens.empty()
            ? true
            : token_is_word_start(engine_impl->model.vocab, win_tokens.front());
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
        try { pimpl_->cancel_session(); } catch (...) {}
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

void StreamSession::feed_pcm_f32(
        const float * samples,
        int n_samples) {
    if (!pimpl_) {
        throw std::runtime_error(
            "StreamSession: moved-from session");
    }
    if (pimpl_->finalized) {
        throw std::runtime_error(
            "StreamSession::feed_pcm_f32: session already finalized");
    }
    if (pimpl_->cancelled) return;
    if (!samples || n_samples <= 0) return;

    stream_drive_energy_vad(
        *pimpl_,
        samples,
        n_samples,
        pimpl_->total_pcm_seen);
    pimpl_->total_pcm_seen += n_samples;

    if (pimpl_->nemotron_state) {
        const int rc = append_nemotron_pcm(
            pimpl_->engine_impl->model,
            *pimpl_->nemotron_state,
            samples,
            n_samples,
            false);

        if (rc != 0) {
            throw std::runtime_error(
                "StreamSession::feed_pcm_f32: appending Nemotron PCM failed "
                "(rc=" + std::to_string(rc) + ")");
        }

        pimpl_->drain_nemotron(false);
        return;
    }

    if (pimpl_->unified_state) {
        const int rc = append_unified_pcm(
            pimpl_->engine_impl->model, *pimpl_->unified_state, samples, n_samples, false);
        if (rc != 0) {
            throw std::runtime_error(
                "StreamSession::feed_pcm_f32: appending Unified PCM failed "
                "(rc=" + std::to_string(rc) + ")");
        }
        pimpl_->drain_unified(false);
        return;
    }

    pimpl_->pending.insert(
        pimpl_->pending.end(),
        samples,
        samples + n_samples);
    pimpl_->try_emit_chunks();
}

void StreamSession::feed_pcm_i16(const int16_t * samples, int n_samples) {
    if (!pimpl_) throw std::runtime_error("StreamSession: moved-from session");
    if (pimpl_->finalized) {
        throw std::runtime_error("StreamSession::feed_pcm_i16: session already finalized");
    }
    if (pimpl_->cancelled) return;
    if (!samples || n_samples <= 0) return;

    if (pimpl_->nemotron_state || pimpl_->unified_state) {
        std::vector<float> converted(
            static_cast<size_t>(n_samples));

        constexpr float inv = 1.0f / 32768.0f;
        for (int i = 0; i < n_samples; ++i) {
            converted[static_cast<size_t>(i)] =
                static_cast<float>(samples[i]) * inv;
        }

        feed_pcm_f32(converted.data(), n_samples);
        return;
    }

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

    if (pimpl_->cancelled) return;

    if (pimpl_->nemotron_state) {
        auto & model = pimpl_->engine_impl->model;
        auto & state = *pimpl_->nemotron_state;

        int rc = append_nemotron_pcm(
            model,
            state,
            nullptr,
            0,
            true);

        if (rc != 0) {
            throw std::runtime_error(
                "StreamSession::finalize: flushing Nemotron PCM failed "
                "(rc=" + std::to_string(rc) + ")");
        }

        pimpl_->drain_nemotron(false);
        pimpl_->drain_nemotron(true);

        if (!state.finalized) {
            NemotronStreamStepResult result;
            rc = run_nemotron_stream_step(
                model,
                pimpl_->engine_impl->transducer_rt,
                nullptr,
                0,
                model.mel_cfg.n_mels,
                true,
                state,
                result);

            if (rc != 0) {
                throw std::runtime_error(
                    "StreamSession::finalize: finalizing Nemotron state failed "
                    "(rc=" + std::to_string(rc) + ")");
            }
        }

        return;
    }

    if (pimpl_->unified_state) {
        auto & model = pimpl_->engine_impl->model;
        auto & state = *pimpl_->unified_state;
        int rc = append_unified_pcm(model, state, nullptr, 0, true);
        if (rc != 0) {
            throw std::runtime_error(
                "StreamSession::finalize: flushing Unified PCM failed "
                "(rc=" + std::to_string(rc) + ")");
        }
        pimpl_->drain_unified(false);
        pimpl_->drain_unified(true);
        if (!state.finalized && !pimpl_->cancelled) {
            UnifiedStreamStepResult result;
            rc = run_unified_stream_step(
                model, pimpl_->engine_impl->transducer_rt, nullptr, 0,
                model.mel_cfg.n_mels, true, state, result);
            if (rc != 0) {
                throw std::runtime_error(
                    "StreamSession::finalize: finalizing Unified state failed "
                    "(rc=" + std::to_string(rc) + ")");
            }
        }
        return;
    }

    pimpl_->try_emit_chunks();
    pimpl_->flush_remainder();
}

void StreamSession::cancel() {
    if (!pimpl_) return;
    pimpl_->cancel_session();
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

    pimpl_->cancel_flag.store(false);

    auto impl = std::make_unique<StreamSession::Impl>();
    impl->engine_impl  = pimpl_.get();
    impl->opts         = opts;
    impl->on_segment   = std::move(on_segment);

    if (pimpl_->model.model_type == ParakeetModelType::NEMOTRON) {
        const auto & chunk_sizes =
            pimpl_->model.nemotron_cfg.allowed_chunk_ms;
        const auto & right_contexts =
            pimpl_->model.nemotron_cfg.allowed_right_context_frames;
        const auto chunk_it = std::find(
            chunk_sizes.begin(), chunk_sizes.end(), opts.chunk_ms);

        if (chunk_it == chunk_sizes.end() ||
            chunk_sizes.size() != right_contexts.size()) {
            throw std::runtime_error(
                "Engine::stream_start: unsupported Nemotron chunk_ms; "
                "supported values are " + format_int_list(chunk_sizes));
        }

        const size_t operating_point = static_cast<size_t>(
            chunk_it - chunk_sizes.begin());
        const int right_context_frames =
            right_contexts[operating_point];

        impl->nemotron_state = std::make_unique<NemotronStreamState>();

        const int rc = init_nemotron_stream_state(
            pimpl_->model,
            pimpl_->opts.language,
            right_context_frames,
            *impl->nemotron_state);

        if (rc != 0) {
            throw std::runtime_error(
                "Engine::stream_start: Nemotron stream initialization failed "
                "(rc=" + std::to_string(rc) + ")");
        }
    } else if (pimpl_->model.unified_cfg.available) {
        impl->unified_state = std::make_unique<UnifiedStreamState>();
        const int rc = init_unified_stream_state(
            pimpl_->model,
            resolve_unified_chunk_frames(pimpl_->model, opts),
            resolve_unified_right_context_frames(pimpl_->model, opts),
            *impl->unified_state);
        if (rc != 0) {
            throw std::runtime_error(
                "Engine::stream_start: Unified stream initialization failed "
                "(rc=" + std::to_string(rc) + ")");
        }
    }

    const int sr = opts.sample_rate;
    impl->chunk_samples           = sr * opts.chunk_ms / 1000;
    impl->left_context_samples    = sr * opts.left_context_ms / 1000;
    impl->right_lookahead_samples = sr * opts.right_lookahead_ms / 1000;

    impl->left_history.reserve(impl->left_context_samples);
    impl->pending.reserve(impl->chunk_samples + impl->right_lookahead_samples);

    if (is_transducer(pimpl_->model.model_type)) {
        tdt_init_state(
            pimpl_->transducer_rt,
            (int) pimpl_->model.blank_id,
            impl->transducer_state);
    }
    // Optional EnergyVad for CTC/RNN-T/TDT/Nemotron only (EOU uses `<EOU>`;
    // Sortformer uses SortformerStreamSession).
    if (opts.enable_energy_vad &&
        pimpl_->model.model_type != ParakeetModelType::EOU) {
        impl->energy_vad = std::make_unique<EnergyVad>(
            sr,
            opts.energy_vad_window_ms,
            opts.energy_vad_hangover_ms,
            opts.energy_vad_threshold_db);
    }
    if (pimpl_->model.model_type == ParakeetModelType::EOU &&
        eou_init_state(pimpl_->eou_rt, impl->eou_state) != 0) {
        throw std::runtime_error("Engine::open_stream: eou_init_state failed");
    }

    return std::make_unique<StreamSession>(std::move(impl));
}

struct SortformerStreamSession::Impl {
    Engine::Impl *              engine_impl = nullptr;
    SortformerStreamingOptions  opts;
    SortformerSegmentCallback   on_segment;

    int    chunk_samples   = 0;
    int    history_samples = 0;

    // AOSC audio-context budgets, in samples and post-subsampling encoder frames.
    // Populated in diarize_start when cache_active is true; zero otherwise.
    int    chunk_left_context_samples  = 0;
    int    chunk_right_context_samples = 0;
    int    lc_enc_frames_expected      = 0;
    int    rc_enc_frames_expected      = 0;

    // AOSC compression policy + cache geometry (NeMo defaults). Populated from
    // opts in diarize_start when cache_active is true.
    SortformerStreamingConfig  sortformer_cfg;

    std::vector<float> ring;
    int64_t            ring_origin_sample = 0;

    int64_t            emitted_samples    = 0;
    int                chunk_index        = 0;

    bool finalized = false;
    bool cancelled = false;

    std::vector<StreamingDiarizationSegment> last_pending;

    // Full segments from the previous chunk in ABSOLUTE time, with the
    // session-stable speaker IDs that this session has emitted. Used by
    // compute_slot_remap_ to find an overlap-based remap that anchors
    // slot identity across chunks even when the visible voice set
    // changes (e.g. a speaker ages out of the rolling history window).
    // Empty before the first chunk emits. Unused on the cache_active path.
    std::vector<StreamingDiarizationSegment> prev_chunk_full_segments;

    SortformerSpeakerCache cache;
    bool                   cache_active = false;
    float                  nemotron_stream_gain = 0.0f;
    std::vector<float> *   probability_sink = nullptr;

    // Speaking vs silent from Sortformer probs: max probability above opts.threshold.
    // Initial Unknown forces a transition on the first chunk.
    VadState vad_state = VadState::Unknown;

    void try_emit_chunks();
    void process_chunk(int64_t window_start_sample,
                       int64_t window_end_sample,
                       int64_t emit_start_sample,
                       int64_t emit_end_sample);

    // Compute remap[local_id] -> session_id by maximising overlap of
    // current chunk's local-ID segments against prev_chunk_full_segments
    // (which carry session IDs). Greedy: highest-overlap pairs first;
    // unmatched local slots get the lowest unused session ID. Identity
    // mapping when prev_chunk_full_segments is empty (first chunk).
    std::vector<int> compute_slot_remap_(
        const std::vector<StreamingDiarizationSegment> & cur_full,
        int num_spks) const;
};

std::vector<int> SortformerStreamSession::Impl::compute_slot_remap_(
    const std::vector<StreamingDiarizationSegment> & cur_full,
    int num_spks) const {
    std::vector<int> remap(num_spks, -1);
    if (num_spks <= 0) return remap;
    if (prev_chunk_full_segments.empty()) {
        for (int i = 0; i < num_spks; ++i) remap[i] = i;
        return remap;
    }
    // Build num_spks x num_spks overlap matrix: O[local_id][session_id]
    // = total absolute-time overlap between current chunk's segments
    // labelled `local_id` and previous chunk's session-stable segments
    // labelled `session_id`. Consecutive windows share `history_ms -
    // chunk_ms` of audio, so every active speaker has plenty of
    // co-occurring segments to match.
    std::vector<std::vector<double>> O(
        num_spks, std::vector<double>(num_spks, 0.0));
    for (const auto & c : cur_full) {
        if (c.speaker_id < 0 || c.speaker_id >= num_spks) continue;
        for (const auto & p : prev_chunk_full_segments) {
            if (p.speaker_id < 0 || p.speaker_id >= num_spks) continue;
            const double a = std::max(c.start_s, p.start_s);
            const double b = std::min(c.end_s,   p.end_s);
            if (b > a) O[c.speaker_id][p.speaker_id] += (b - a);
        }
    }
    // Greedy assignment over O: order local IDs by their best available
    // overlap (descending), then for each pick the highest-overlap
    // un-taken session ID.
    std::vector<bool> taken(num_spks, false);
    std::vector<std::pair<double, int>> order;
    order.reserve((size_t) num_spks);
    for (int i = 0; i < num_spks; ++i) {
        double m = 0.0;
        for (int j = 0; j < num_spks; ++j) m = std::max(m, O[i][j]);
        order.emplace_back(m, i);
    }
    std::sort(order.begin(), order.end(),
              [](const std::pair<double, int> & a,
                 const std::pair<double, int> & b) {
                  return a.first > b.first;
              });
    for (const auto & pr : order) {
        if (pr.first <= 0.0) continue;
        const int  i = pr.second;
        int        best   = -1;
        double     best_o = 0.0;
        for (int j = 0; j < num_spks; ++j) {
            if (taken[j]) continue;
            if (O[i][j] > best_o) { best_o = O[i][j]; best = j; }
        }
        if (best >= 0) { remap[i] = best; taken[best] = true; }
    }
    // Unmatched locals (no overlap with any prev segment, or no prev
    // segments at all) take the lowest unused session ID. This keeps
    // session IDs stable and predictable across long streams.
    int next = 0;
    for (int i = 0; i < num_spks; ++i) {
        if (remap[i] != -1) continue;
        while (next < num_spks && taken[next]) ++next;
        if (next < num_spks) {
            remap[i] = next;
            taken[next] = true;
            ++next;
        } else {
            remap[i] = i; // safety; shouldn't fire when num_spks is consistent
        }
    }
    return remap;
}

void SortformerStreamSession::Impl::process_chunk(int64_t window_start_sample,
                                                  int64_t window_end_sample,
                                                  int64_t emit_start_sample,
                                                  int64_t emit_end_sample) {
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
        if (engine_impl->model.model_type == ParakeetModelType::NEMOTRON_DIARIZATION &&
            nemotron_stream_gain == 0.0f) {
            nemotron_stream_gain = nemotron_audio_gain(win, n);
        }
        if (cache_active) {
            diar = engine_impl->model.model_type == ParakeetModelType::NEMOTRON_DIARIZATION
                ? engine_impl_diarize_nemotron_streaming(
                    *engine_impl, win, n, opts.sample_rate, diopts, cache,
                    sortformer_cfg, static_cast<int>(emit_start_sample - window_start_sample),
                    rc_enc_frames_expected,
                    static_cast<int>(emit_end_sample - emit_start_sample),
                    nemotron_stream_gain)
                : engine_impl_diarize_streaming_helper(
                    *engine_impl, win, n, opts.sample_rate, diopts, cache,
                    sortformer_cfg, lc_enc_frames_expected, rc_enc_frames_expected);
        } else {
            // v1 path: full history_ms re-encoded each chunk; overlap-based
            // slot remap downstream.
            diar = engine_impl->model.model_type == ParakeetModelType::NEMOTRON_DIARIZATION
                ? engine_impl_diarize_nemotron_with_gain(
                    *engine_impl, win, n, opts.sample_rate, diopts, nemotron_stream_gain)
                : engine_impl_diarize_helper(*engine_impl, win, n, opts.sample_rate, diopts);
        }
    }

    if (probability_sink && cache_active) {
        probability_sink->insert(probability_sink->end(),
            diar.speaker_probs.begin(), diar.speaker_probs.end());
    }

    // AOSC's `sortformer_aosc_step` returns segments + speaker_probs spanning
    // ONLY the committed chunk (chunk_len_eff frames), with time 0 = start of
    // committed chunk. v1's helper returns segments + probs over the FULL
    // rolling window, with time 0 = start of window. The "window_offset_s"
    // used downstream must match the helper's frame-0 origin.
    const double emit_lo_s = (double) emit_start_sample / opts.sample_rate;
    const double emit_hi_s = (double) emit_end_sample   / opts.sample_rate;
    const double window_offset_s = cache_active
        ? emit_lo_s
        : (double) window_start_sample / opts.sample_rate;

    // Materialise the FULL window's segments in absolute-time coordinates
    // (local speaker IDs from Sortformer's per-chunk output). This is the
    // input both to the slot-remap computation and to the storage that
    // anchors the next chunk's IDs.
    std::vector<StreamingDiarizationSegment> cur_full;
    cur_full.reserve(diar.segments.size());
    for (const auto & s : diar.segments) {
        StreamingDiarizationSegment f;
        f.speaker_id  = s.speaker_id;
        f.start_s     = window_offset_s + s.start_s;
        f.end_s       = window_offset_s + s.end_s;
        f.chunk_index = chunk_index;
        f.is_final    = false;
        cur_full.push_back(f);
    }

    // AOSC anchors slot identity via the cache + Sort Loss, so the local
    // speaker IDs already match session IDs across chunks. Identity remap.
    // On v1 the overlap-based remap reconciles any per-chunk slot
    // permutation against prev_chunk_full_segments.
    std::vector<int> slot_remap;
    if (cache_active) {
        slot_remap.resize((size_t) diar.num_spks);
        for (int i = 0; i < diar.num_spks; ++i) slot_remap[i] = i;
    } else {
        slot_remap = compute_slot_remap_(cur_full, diar.num_spks);
    }

    auto remap_id = [&slot_remap, num_spks = diar.num_spks](int local) -> int {
        if (local < 0 || local >= num_spks) return local;
        return slot_remap[local];
    };

    std::vector<StreamingDiarizationSegment> emitted;
    emitted.reserve(diar.segments.size());

    for (const auto & s : diar.segments) {
        const double abs_start = window_offset_s + s.start_s;
        const double abs_end   = window_offset_s + s.end_s;
        if (abs_end <= emit_lo_s) continue;
        if (abs_start >= emit_hi_s) continue;

        StreamingDiarizationSegment out;
        out.speaker_id  = remap_id(s.speaker_id);
        out.start_s     = std::max(abs_start, emit_lo_s);
        out.end_s       = std::min(abs_end,   emit_hi_s);
        out.chunk_index = chunk_index;
        out.is_final    = false;
        if (out.end_s - out.start_s < opts.min_segment_ms / 1000.0) continue;
        emitted.push_back(out);
    }

    if (on_segment) {
        for (const auto & seg : emitted) on_segment(seg);
    }
    last_pending = std::move(emitted);

    // Remap cur_full into session-stable IDs and store as the new
    // baseline so the next chunk's `compute_slot_remap_` can match
    // against today's emitted identity scheme. AOSC anchors slot
    // identity through the speaker cache, so `compute_slot_remap_`
    // is never consulted on that path -- skip the storage and the
    // identity-remap loop entirely.
    if (!cache_active) {
        for (auto & f : cur_full) {
            f.speaker_id = remap_id(f.speaker_id);
        }
        prev_chunk_full_segments = std::move(cur_full);
    }

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

        int64_t window_start;
        int64_t window_end;
        if (cache_active) {
            // AOSC: window = [emit_start - lc_samples, emit_end + rc_samples].
            // Wait for rc audio to arrive after the committed chunk before emitting.
            const int64_t needed_end = emit_end + chunk_right_context_samples;
            if (available_end < needed_end) return;
            window_start = std::max(ring_origin_sample,
                                    emitted_samples - chunk_left_context_samples);
            window_end   = needed_end;
        } else {
            // v1 path: full rolling history_ms window, no right context.
            window_end   = emit_end;
            window_start = std::max(ring_origin_sample,
                                    window_end - history_samples);
        }
        process_chunk(window_start, window_end, emitted_samples, emit_end);

        // Trim the ring. v1 keeps the trailing history_ms. AOSC needs to keep
        // chunk_left_context_samples ahead of emit_end so the NEXT chunk's
        // window_start (emit_end - lc_samples) is still in the ring.
        const int64_t keep_min_from = cache_active
            ? std::max(ring_origin_sample, emit_end - chunk_left_context_samples)
            : std::max(ring_origin_sample, emit_end - history_samples);
        if (keep_min_from > ring_origin_sample) {
            const size_t drop = (size_t) (keep_min_from - ring_origin_sample);
            ring.erase(ring.begin(), ring.begin() + drop);
            ring_origin_sample = keep_min_from;
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

bool SortformerStreamSession::aosc_active() const {
    return pimpl_ && pimpl_->cache_active;
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
    detail::finalize_sortformer_stream(
        pimpl_->finalized,
        pimpl_->cancelled,
        [&] {
            pimpl_->try_emit_chunks();
        },
        [&] {
            const int64_t available_end =
                pimpl_->ring_origin_sample + (int64_t) pimpl_->ring.size();
            return available_end > pimpl_->emitted_samples;
        },
        [&] {
            const int64_t available_end =
                pimpl_->ring_origin_sample + (int64_t) pimpl_->ring.size();
            // Tail chunk: drain whatever remains. AOSC also picks up left
            // context from before emit_start; right context is whatever is
            // left when no more audio is coming.
            int64_t window_start;
            int64_t window_end;
            if (pimpl_->cache_active) {
                window_start = std::max(
                    pimpl_->ring_origin_sample,
                    pimpl_->emitted_samples - pimpl_->chunk_left_context_samples);
                window_end = available_end;
            } else {
                window_end = available_end;
                window_start = std::max(
                    pimpl_->ring_origin_sample,
                    window_end - pimpl_->history_samples);
            }
            pimpl_->process_chunk(
                window_start,
                window_end,
                pimpl_->emitted_samples,
                available_end);
        },
        [&] {
            if (!pimpl_->on_segment) return;
            StreamingDiarizationSegment terminator;
            terminator.speaker_id  = -1;
            terminator.start_s     =
                (double) pimpl_->emitted_samples / pimpl_->opts.sample_rate;
            terminator.end_s       = terminator.start_s;
            terminator.chunk_index = pimpl_->chunk_index;
            terminator.is_final    = true;
            pimpl_->on_segment(terminator);
        });
}

void SortformerStreamSession::cancel() {
    if (!pimpl_) return;
    pimpl_->cancelled = true;
}

static std::unique_ptr<SortformerStreamSession::Impl> make_diarization_session(
    Engine::Impl & engine, const SortformerStreamingOptions & opts,
    SortformerSegmentCallback on_segment) {
    constexpr int kNemotronSilenceFramesPerSpeaker = 1;
    constexpr int kSortformerLeftContextMs = 80;
    const bool is_nemotron =
        engine.model.model_type == ParakeetModelType::NEMOTRON_DIARIZATION;
    SortformerStreamingOptions effective = opts;
    const SortformerStreamingOptions defaults;
    if (is_nemotron) {
        if (effective.chunk_ms == defaults.chunk_ms) effective.chunk_ms = kNemotronChunkMs;
        if (effective.spkcache_len == defaults.spkcache_len) {
            effective.spkcache_len = kNemotronSpeakerCacheFrames;
        }
        if (effective.fifo_len == defaults.fifo_len) effective.fifo_len = kNemotronFifoFrames;
        if (effective.chunk_left_context_ms ==
            SortformerStreamingOptions::unspecified_left_context_ms) {
            effective.chunk_left_context_ms = 0;
        }
        if (effective.chunk_right_context_ms == defaults.chunk_right_context_ms) {
            effective.chunk_right_context_ms = kNemotronRightContextMs;
        }
        if (effective.spkcache_update_period == defaults.spkcache_update_period) {
            effective.spkcache_update_period = kNemotronUpdateFrames;
        }
        const int coarse_frame_ms =
            1000 * engine.model.mel_cfg.hop_length *
            engine.model.nemotron_diarization_cfg.subsampling_factor /
            engine.model.mel_cfg.sample_rate;
        if (effective.spkcache_enable &&
            (effective.chunk_ms % coarse_frame_ms != 0 ||
             effective.chunk_left_context_ms % coarse_frame_ms != 0 ||
             effective.chunk_right_context_ms % coarse_frame_ms != 0)) {
            throw std::runtime_error(
                "Nemotron AOSC chunk and context durations must align to encoder frames");
        }
    } else if (effective.chunk_left_context_ms ==
        SortformerStreamingOptions::unspecified_left_context_ms) {
        effective.chunk_left_context_ms = kSortformerLeftContextMs;
    }
    if (effective.history_ms < effective.chunk_ms) {
        throw std::runtime_error("Engine::diarize_start: history_ms must be >= chunk_ms");
    }
    auto impl = std::make_unique<SortformerStreamSession::Impl>();
    impl->engine_impl     = &engine;
    impl->opts            = effective;
    impl->on_segment      = std::move(on_segment);
    impl->chunk_samples   = effective.sample_rate * effective.chunk_ms / 1000;
    impl->history_samples = effective.sample_rate * effective.history_ms / 1000;
    impl->ring.reserve(impl->history_samples);

    const std::string & variant = engine.model.model_variant;
    const bool model_is_v2_1 = !variant.empty()
        ? (variant == "sortformer-streaming-v2.1-aosc")
        : (engine.model.encoder_cfg.n_layers == 17 &&
           engine.model.mel_cfg.n_mels == 128);
    impl->cache_active = opts.spkcache_enable && (model_is_v2_1 || is_nemotron);

    if (impl->cache_active) {
        // Populate AOSC config from public options.
        impl->sortformer_cfg.spkcache_len             = effective.spkcache_len;
        impl->sortformer_cfg.fifo_len                 = effective.fifo_len;
        impl->sortformer_cfg.spkcache_update_period   = effective.spkcache_update_period;
        if (is_nemotron) {
            impl->sortformer_cfg.spkcache_sil_frames_per_spk =
                kNemotronSilenceFramesPerSpeaker;
        }
        // chunk_len in encoder frames; derived from chunk_ms.
        const int enc_frame_ms =
            1000 * engine.model.mel_cfg.hop_length *
            (is_nemotron
                ? engine.model.nemotron_diarization_cfg.subsampling_factor
                : engine.model.encoder_cfg.subsampling_factor) /
            engine.model.mel_cfg.sample_rate;
        impl->sortformer_cfg.chunk_len =
            std::max(1, effective.chunk_ms / std::max(1, enc_frame_ms));
        const int lc_ms = std::max(0, effective.chunk_left_context_ms);
        const int rc_ms = std::max(0, effective.chunk_right_context_ms);
        impl->sortformer_cfg.chunk_left_context  = lc_ms / std::max(1, enc_frame_ms);
        impl->sortformer_cfg.chunk_right_context = rc_ms / std::max(1, enc_frame_ms);

        impl->chunk_left_context_samples  = opts.sample_rate * lc_ms / 1000;
        impl->chunk_right_context_samples = opts.sample_rate * rc_ms / 1000;
        impl->lc_enc_frames_expected = impl->sortformer_cfg.chunk_left_context;
        impl->rc_enc_frames_expected = impl->sortformer_cfg.chunk_right_context;

        // Reset cache to a clean state with mean_sil_emb zeros at the model's
        // fc_d_model dimension.
        sortformer_cache_reset(impl->cache, is_nemotron
            ? engine.model.nemotron_diarization_cfg.encoder_width
            : engine.model.encoder_cfg.d_model);
        if (is_nemotron && engine.model.nemotron_diarization.silence_embedding) {
            ggml_backend_tensor_get(
                engine.model.nemotron_diarization.silence_embedding,
                impl->cache.mean_sil_emb.data(), 0,
                impl->cache.mean_sil_emb.size() * sizeof(float));
        }

        std::fprintf(stderr,
            "[parakeet] %s AOSC enabled (spkcache_len=%d fifo_len=%d "
            "chunk=%d lc=%d rc=%d update_period=%d)\n",
            is_nemotron ? "Nemotron 3 Diarization" : "Sortformer v2.1",
            impl->sortformer_cfg.spkcache_len,
            impl->sortformer_cfg.fifo_len,
            impl->sortformer_cfg.chunk_len,
            impl->sortformer_cfg.chunk_left_context,
            impl->sortformer_cfg.chunk_right_context,
            impl->sortformer_cfg.spkcache_update_period);
    }
    return impl;
}

static SortformerStreamingOptions nemotron_long_form_options(int sample_rate) {
    SortformerStreamingOptions options;
    options.sample_rate = sample_rate;
    options.chunk_ms = kNemotronLongFormChunkMs;
    options.chunk_left_context_ms = kNemotronLongFormContextMs;
    options.chunk_right_context_ms = kNemotronLongFormContextMs;
    options.history_ms = std::max(options.history_ms, options.chunk_ms);
    return options;
}

static void hold_last_frame(std::vector<float> & probabilities, int speakers, int frames) {
    const size_t target = static_cast<size_t>(frames) * speakers;
    if (probabilities.size() < static_cast<size_t>(speakers)) {
        probabilities.resize(target, 0.0f);
        return;
    }
    const std::vector<float> last(probabilities.end() - speakers, probabilities.end());
    while (probabilities.size() < target) {
        probabilities.insert(probabilities.end(), last.begin(), last.end());
    }
    probabilities.resize(target);
}

static int nemotron_long_form_block_samples(int sample_rate) {
    return static_cast<int>(
        static_cast<long long>(sample_rate) * kNemotronLongFormChunkMs / kMillisecondsPerSecond);
}

static void feed_in_blocks(SortformerStreamSession & session, const float * samples,
                           int n_samples, int block_samples) {
    for (long long offset = 0; offset < n_samples; offset += block_samples) {
        const int block = static_cast<int>(std::min<long long>(block_samples, n_samples - offset));
        session.feed_pcm_f32(samples + offset, block);
    }
}

static std::vector<float> nemotron_long_form_probabilities(
    Engine::Impl & impl, const float * samples, int n_samples, int sample_rate) {
    std::vector<float> probabilities;
    auto session_impl = make_diarization_session(
        impl, nemotron_long_form_options(sample_rate), nullptr);
    session_impl->probability_sink = &probabilities;
    session_impl->nemotron_stream_gain = nemotron_audio_gain(samples, n_samples);
    SortformerStreamSession session(std::move(session_impl));
    feed_in_blocks(session, samples, n_samples, nemotron_long_form_block_samples(sample_rate));
    session.finalize();
    return probabilities;
}

static DiarizationResult engine_impl_diarize_nemotron_long_form(
    Engine::Impl & impl, const float * samples, int n_samples,
    int sample_rate, const DiarizationOptions & options) {
    if (sample_rate != impl.model.mel_cfg.sample_rate) {
        throw std::runtime_error("Nemotron diarization requires nonempty 16 kHz audio");
    }
    const auto start = std::chrono::steady_clock::now();
    DiarizationResult result;
    result.num_spks = impl.model.nemotron_diarization_cfg.speakers;
    result.n_frames = static_cast<int>(nemotron_diarization_mel_frames(impl.model, n_samples));
    result.speaker_probs = nemotron_long_form_probabilities(impl, samples, n_samples, sample_rate);
    hold_last_frame(result.speaker_probs, result.num_spks, result.n_frames);
    result.frame_stride_s = static_cast<double>(impl.model.mel_cfg.hop_length) / sample_rate;
    result.audio_samples = n_samples;
    result.sample_rate = sample_rate;
    result.decode_ms = ms_since(start);
    append_nemotron_diarization_segments(result, options);
    result.total_ms = ms_since(start);
    return result;
}

std::unique_ptr<SortformerStreamSession> Engine::diarize_start(
    const SortformerStreamingOptions & opts,
    SortformerSegmentCallback on_segment) {
    if (!is_diarization_model() || !pimpl_->sortformer_ready) {
        throw std::runtime_error("Engine::diarize_start: loaded GGUF is not a Sortformer model");
    }
    if (opts.sample_rate != pimpl_->model.mel_cfg.sample_rate) {
        throw std::runtime_error("Engine::diarize_start: sample_rate mismatch");
    }
    if (opts.chunk_ms <= 0)   throw std::runtime_error("Engine::diarize_start: chunk_ms must be > 0");
    if (opts.history_ms <= 0) throw std::runtime_error("Engine::diarize_start: history_ms must be > 0");
    return std::make_unique<SortformerStreamSession>(
        make_diarization_session(*pimpl_, opts, std::move(on_segment)));
}

}
