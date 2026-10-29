// Audio8 memory-fit preflight (include/tts-cpp/audio8/fit.h): project one
// Engine load + one synthesize() against the device memory available right
// now, without reading weight data.
//
// The projection mirrors the real runtime path by construction rather than by
// formula: it drives the same loaders (metadata-only), the same graph
// builders, and the same allocation policies -- direct gallocr vs the
// [backend, CPU-last] scheduler fallback, and the codec's memory-budgeted
// synthesis block plan -- through ggml's size-only APIs
// (ggml_backend_alloc_ctx_tensors_from_buft_size / ggml_gallocr_reserve_n_size /
// ggml_backend_sched_reserve_size), so it tracks runtime changes instead of
// drifting from them.

#include "tts-cpp/audio8/fit.h"

#include "audio8/graph.h"
#include "audio8/internal.h"
#include "backend_selection.h"
#include "backend_util.h"
#include "fit_price.h"
#include "fit_util.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace tts_cpp {
namespace audio8 {

namespace {

using detail::AUDIO8_DEFAULT_MAX_FRAMES;
using detail::AUDIO8_MAX_NODES;
using fitutil::sat_add;
using fitutil::sat_mul;
using fitutil::sat_u64_from_double;

std::string fmt_mib(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", (double) bytes / (1024.0 * 1024.0));
    return buf;
}

// Frees whatever a partial metadata-only load left behind, in the loaders'
// own teardown order.
struct model_guard {
    detail::lm_model    lm;
    detail::codec_model decoder;
    detail::codec_model encoder;
    ~model_guard() {
        detail::free_lm(lm);
        detail::free_codec(decoder);
        detail::free_codec(encoder);
    }
};

// Price one freshly built LM graph through the same dual-path dispatch
// prepare_graph allocates with. Returns false on a pricing failure.
bool price_lm_graph(detail::lm_model & lm, detail::scratch & build,
                    ::tts_cpp::detail::fit_graph_price & out) {
    if (!build.ok()) return false;
    // 2 * AUDIO8_MAX_NODES mirrors prepare_graph's sched_fallback_ensure size.
    return ::tts_cpp::detail::fit_price_graph(lm.backend, build.graph,
                                              2 * AUDIO8_MAX_NODES, out);
}

bool price_fast_position(detail::lm_model & lm, int position,
                         ::tts_cpp::detail::fit_graph_price & out) {
    detail::scratch build(detail::AUDIO8_FAST_MAX_NODES);
    if (!build.ok()) return false;
    detail::build_fast_fit_graph(lm, build, position, /*prime=*/position == 0);
    return price_lm_graph(lm, build, out);
}

// Position 0 primes from the slow transformer's hidden state and every later
// one reads the code before it; how their prices combine lives with
// fit_price_aggregate, next to the pricing it aggregates.
bool price_fast_positions(detail::lm_model & lm, int num_codebooks,
                          ::tts_cpp::detail::fit_price_aggregate & prices) {
    for (int position = 0; position < num_codebooks; ++position) {
        ::tts_cpp::detail::fit_graph_price one;
        if (!price_fast_position(lm, position, one)) return false;
        prices.add(one);
    }
    return true;
}

}  // namespace

FitResult fit_params(const FitOptions & opts) {
    FitResult r;
    r.model_variant = "audio8";

    const bool cloning = !opts.codec_encoder_gguf_path.empty();
    if (opts.lm_gguf_path.empty() || opts.codec_decoder_gguf_path.empty() ||
        opts.prompt_tokens <= 0 || opts.max_frames < 0 ||
        (cloning && !(opts.reference_seconds > 0.0f))) {
        r.reason = "invalid-arguments";
        return r;
    }

    if (!opts.backends_dir.empty()) {
        ::tts_cpp::detail::set_backends_directory(opts.backends_dir);
    }

    // An empty device registry is always Error, never Success: a projection
    // made against a machine the fitter cannot see is worse than none.
    {
        ggml_backend_t probe = ::tts_cpp::detail::init_cpu_backend();
        if (!probe) {
            r.reason = "no-backend-device";
            return r;
        }
        ggml_backend_free(probe);
    }

    // ── Metadata-only loads: backend resolution + tensor wiring + sizing ────
    model_guard m;
    detail::fit_load_measure lm_load, dec_load, enc_load;
    std::string error;
    // Mirror the engine's backend selector so the projection measures the
    // same device the real load will land on. An explicit name that cannot
    // be initialised (e.g. "hexagon" on a non-HTP host) returns a distinct
    // "backend-unavailable" reason instead of being conflated with
    // model-unreadable; init_backend emits "failed to init a compute
    // backend" in that case.
    auto backend_init_failed = [](const std::string & e) {
        return e.find("failed to init a compute backend") != std::string::npos;
    };
    if (!detail::load_lm_metadata_only(opts.lm_gguf_path, opts.n_gpu_layers,
                                       opts.backend, m.lm, lm_load, &error) ||
        !detail::load_codec_metadata_only(opts.codec_decoder_gguf_path, opts.n_gpu_layers,
                                          opts.backend, m.decoder, dec_load, &error) ||
        !m.decoder.has_decoder) {
        r.reason = backend_init_failed(error) ? "backend-unavailable" : "model-unreadable";
        return r;
    }
    if (cloning) {
        if (!detail::load_codec_metadata_only(opts.codec_encoder_gguf_path, opts.n_gpu_layers,
                                              opts.backend, m.encoder, enc_load, &error) ||
            !m.encoder.has_encoder) {
            r.reason = backend_init_failed(error) ? "backend-unavailable" : "model-unreadable";
            return r;
        }
    }

    // All three sub-models resolve their backend through the same
    // validated-GPU policy (init_backend), so they land on the same device;
    // the LM's handle stands in for all of them.
    ggml_backend_t backend = m.lm.backend;
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    if (!dev) {
        r.reason = "no-backend-device";
        return r;
    }
    r.device_name   = ggml_backend_dev_name(dev);
    r.device_is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    // Unified-memory devices draw the "device" figure from the same physical
    // RAM the host-side buffers live in, so the verdict charges both against it.
    r.device_shares_host_memory = ::tts_cpp::detail::backend_shares_host_memory(backend);
    {
        // A device that shares host memory and reports none of its own
        // (Hexagon) takes the host's figures.
        size_t free_b = 0, total_b = 0;
        ::tts_cpp::detail::backend_memory(backend, free_b, total_b);
        r.device_free_bytes  = free_b;
        r.device_total_bytes = total_b;
        // A discrete device that reports no memory cannot be compared
        // against, so it is an error rather than a refusal.
        if (total_b == 0) {
            r.reason = "device-memory-unknown";
            return r;
        }
    }

    // ── Workload → graph shapes (saturating; an unrepresentable workload
    //    must surface as workload-too-large, never wrap) ────────────────────
    const detail::lm_hparams & hp = m.lm.hp;
    int ref_frames = 0;
    if (cloning) {
        const detail::codec_hparams & ehp = m.encoder.hp;
        const uint64_t ref_samples = sat_u64_from_double(
            std::ceil((double) opts.reference_seconds * ehp.sample_rate));
        const uint64_t frames_u =
            (sat_add(ref_samples, (uint64_t) ehp.frame_size - 1)) / ehp.frame_size;
        if (frames_u > (uint64_t) std::numeric_limits<int>::max()) {
            r.reason = "workload-too-large";
            return r;
        }
        ref_frames = std::max(1, (int) frames_u);
        // encode_positions multiplies n_frames * frame_size in int, so widen
        // the product check first: a near-INT_MAX ref_frames would
        // sign-overflow (UB) and a wrapped-negative position count would slip
        // PAST the max_frames rejection below into a ggml abort. Anything
        // whose sample count exceeds int is over every RoPE table anyway.
        if ((long long) ref_frames * ehp.frame_size >
            (long long) std::numeric_limits<int>::max()) {
            r.reason = "workload-too-large";
            return r;
        }
        // The encoder's own transformer is the first RoPE table to run out
        // (check_encodable); a reference it cannot encode is not projectable.
        if (detail::encode_positions(m.encoder, ref_frames) > ehp.max_frames) {
            r.reason = "workload-too-large";
            return r;
        }
    }
    const uint64_t prompt_u = sat_add((uint64_t) opts.prompt_tokens, (uint64_t) ref_frames);
    if (prompt_u >= (uint64_t) hp.max_seq_len) {
        r.reason = "workload-too-large";  // Engine::generate_codes would refuse it
        return r;
    }
    const int prompt_width = (int) prompt_u;
    const int requested =
        opts.max_frames > 0 ? opts.max_frames : AUDIO8_DEFAULT_MAX_FRAMES;
    const int budget = std::min(requested, hp.max_seq_len - prompt_width);
    if (budget <= 0) {
        r.reason = "workload-too-large";
        return r;
    }
    const int n_frames = budget;  // generation priced to the full budget
    // decode_codes refuses more frames than its baked RoPE table, so the
    // codec side of a runnable synthesis is bounded by it.
    const int codec_frames = std::min(n_frames, m.decoder.hp.max_frames);

    uint64_t lm_compute = 0;
    uint64_t host_extra = 0;  // CPU-fallback portions of sched-priced graphs
    bool fast_positions_replayed = false;  // false once one lands on the scheduler

    // ── LM graphs: the resident arena set ───────────────────────────────────
    // slow_allocr serves both the prompt prefill and every decode step and
    // grows to the larger; fast_allocr and frame_allocr are separate arenas
    // that stay resident alongside it, so they add. (On the scheduler
    // fallback path all graphs share one sched; summing there over-counts,
    // which is the permitted -- strict -- direction.)
    {
        ::tts_cpp::detail::fit_graph_price prefill, decode;
        {
            detail::scratch build(AUDIO8_MAX_NODES);
            detail::slow_graph_outputs outs;
            if (!build.ok()) { r.reason = "measurement-failed"; return r; }
            detail::build_slow_graph(m.lm, build, prompt_width, /*n_past=*/0, outs);
            if (!price_lm_graph(m.lm, build, prefill)) {
                r.reason = "measurement-failed";
                return r;
            }
        }
        {
            const int n_past = std::min(prompt_width + n_frames - 1, hp.max_seq_len - 1);
            detail::scratch build(AUDIO8_MAX_NODES);
            detail::slow_graph_outputs outs;
            if (!build.ok()) { r.reason = "measurement-failed"; return r; }
            detail::build_slow_graph(m.lm, build, /*width=*/1, n_past, outs);
            if (!price_lm_graph(m.lm, build, decode)) {
                r.reason = "measurement-failed";
                return r;
            }
        }
        lm_compute = sat_add(lm_compute,
                             std::max(prefill.device_bytes, decode.device_bytes));
        host_extra = sat_add(host_extra,
                             std::max(prefill.host_bytes, decode.host_bytes));

        // Fast head: every position keeps its own graph and its own arena for
        // the life of the model, so they all stay resident and the projection
        // adds them up.
        ::tts_cpp::detail::fit_price_aggregate fast_prices;
        if (!price_fast_positions(m.lm, hp.num_codebooks, fast_prices)) {
            r.reason = "measurement-failed";
            return r;
        }
        fast_positions_replayed = fast_prices.all_replayed();
        const ::tts_cpp::detail::fit_graph_price fast = fast_prices.total();
        lm_compute = sat_add(lm_compute, fast.device_bytes);
        host_extra = sat_add(host_extra, fast.host_bytes);

        // The chained whole-frame graph only runs where the backend can pick
        // codes itself (greedy on a validated GPU); its arena then coexists
        // with the others.
        if (m.lm.picks_codes) {
            detail::scratch build(AUDIO8_MAX_NODES);
            if (!build.ok()) { r.reason = "measurement-failed"; return r; }
            detail::build_fast_frame_fit_graph(m.lm, build);
            ::tts_cpp::detail::fit_graph_price frame;
            if (!price_lm_graph(m.lm, build, frame)) {
                r.reason = "measurement-failed";
                return r;
            }
            lm_compute = sat_add(lm_compute, frame.device_bytes);
            host_extra = sat_add(host_extra, frame.host_bytes);
        }
    }

    // ── Codec graphs ────────────────────────────────────────────────────────
    uint64_t codec_compute = 0;
    detail::codec_fit_measure dec_fit;
    if (!detail::measure_decode_memory(m.decoder, codec_frames, dec_fit)) {
        r.reason = "measurement-failed";
        return r;
    }
    codec_compute = sat_add(codec_compute, dec_fit.device_bytes);
    host_extra    = sat_add(host_extra, dec_fit.host_bytes);
    detail::codec_fit_measure enc_fit;
    if (cloning) {
        if (!detail::measure_encode_memory(m.encoder, ref_frames, enc_fit)) {
            r.reason = "measurement-failed";
            return r;
        }
        codec_compute = sat_add(codec_compute, enc_fit.device_bytes);
        host_extra    = sat_add(host_extra, enc_fit.host_bytes);
    }

    // ── Host-side slabs (scale with the workload, live in host RAM) ─────────
    {
        uint64_t host = host_extra;
        const uint64_t f32 = sizeof(float);
        const uint64_t books = (uint64_t) hp.num_codebooks;
        // Prompt frames + the prefill's causal mask (the largest mask built).
        host = sat_add(host, sat_mul(sat_mul(books + 1, (uint64_t) prompt_width), 4));
        host = sat_add(host, sat_mul(sat_mul((uint64_t) prompt_width,
                                             (uint64_t) prompt_width), f32));
        // Generated codes: frame-major vector + codebook-major transpose +
        // SynthesisResult::codes.
        host = sat_add(host, sat_mul(sat_mul((uint64_t) n_frames, books), 3 * 4));
        // Semantic logits + carried fast input, per step.
        host = sat_add(host, sat_mul((uint64_t) hp.codebook_size + 1 + hp.hidden, f32));
        // A replayed fast-AR position also keeps its graph's arena -- the
        // tensor headers and the graph itself -- resident in host RAM. The
        // scheduler path keeps none, because it cannot replay one.
        if (fast_positions_replayed) {
            host = sat_add(host, sat_mul(books, (uint64_t) detail::scratch_arena_bytes(
                                                    detail::AUDIO8_FAST_MAX_NODES)));
        }
        // Latent-graph host side: post slab + its window mask.
        host = sat_add(host, sat_mul(sat_mul((uint64_t) codec_frames,
                                             (uint64_t) m.decoder.hp.latent_dim), f32));
        host = sat_add(host, sat_mul(sat_mul((uint64_t) codec_frames,
                                             (uint64_t) codec_frames), f32));
        // The waveform, plus one synthesis block's raw output before its
        // context columns are dropped.
        host = sat_add(host, sat_mul(sat_mul((uint64_t) codec_frames,
                                             (uint64_t) m.decoder.hp.frame_size), f32));
        host = sat_add(host, sat_mul(sat_mul((uint64_t) dec_fit.block_span,
                                             (uint64_t) m.decoder.hp.frame_size), f32));
        if (cloning) {
            const detail::codec_hparams & ehp = m.encoder.hp;
            const int positions = detail::encode_positions(m.encoder, ref_frames);
            // Reference audio: caller's buffer + the padded copy the encoder
            // works on (at_codec_rate may add a resampled third; two is what
            // an already-at-rate reference costs).
            host = sat_add(host, sat_mul(sat_mul((uint64_t) ref_frames,
                                                 (uint64_t) ehp.frame_size), 2 * f32));
            // Feature slab between the convolution blocks and the analysis
            // graph, plus the two window masks the analysis pass fills.
            host = sat_add(host, sat_mul(sat_mul((uint64_t) positions,
                                                 (uint64_t) detail::encode_feature_width(
                                                     m.encoder)), f32));
            host = sat_add(host, sat_mul(sat_mul((uint64_t) positions,
                                                 (uint64_t) positions), f32));
            host = sat_add(host, sat_mul(sat_mul((uint64_t) ref_frames,
                                                 (uint64_t) ref_frames), f32));
        }
        r.host_bytes = host;
    }

    r.device.weights_bytes = sat_add(sat_add(lm_load.weights_bytes, dec_load.weights_bytes),
                                     enc_load.weights_bytes);
    r.device.state_bytes         = lm_load.kv_bytes;
    r.device.lm_compute_bytes    = lm_compute;
    r.device.codec_compute_bytes = codec_compute;
    r.device.total_bytes =
        sat_add(sat_add(r.device.weights_bytes, r.device.state_bytes),
                sat_add(r.device.lm_compute_bytes, r.device.codec_compute_bytes));

    // ── Verdict ─────────────────────────────────────────────────────────────
    // On a unified-memory device the "device" buffers and the host extras
    // compete for the same physical RAM, so both count against the free
    // figure. Saturating arithmetic: an overflow must surface as
    // DOES-NOT-FIT, never wrap into a false FITS.
    uint64_t required = sat_add(r.device.total_bytes, opts.margin_bytes);
    if (r.device_shares_host_memory) {
        required = sat_add(required, r.host_bytes);
    }
    r.fits   = required <= r.device_free_bytes;
    r.status = r.fits ? FitStatus::Success : FitStatus::Failure;
    r.reason = r.fits ? "fits" : "does-not-fit";

    // ── Report ──────────────────────────────────────────────────────────────
    {
        std::string s;
        char line[256];
        std::snprintf(line, sizeof(line), "model:    audio8 (LM + decoder%s)\n",
                      cloning ? " + encoder" : "");
        s += line;
        std::snprintf(line, sizeof(line), "device:   %s (%s), free %s / total %s\n",
                      r.device_name.c_str(), r.device_is_cpu ? "CPU" : "GPU",
                      fmt_mib(r.device_free_bytes).c_str(),
                      fmt_mib(r.device_total_bytes).c_str());
        s += line;
        const std::string ref_note =
            cloning ? " + " + std::to_string(ref_frames) + " reference frames" : "";
        std::snprintf(line, sizeof(line),
                      "workload: %d prompt tokens%s -> %d prompt frames, %d generated "
                      "frames (codec %d, block %d)\n",
                      opts.prompt_tokens, ref_note.c_str(), prompt_width, n_frames,
                      codec_frames, dec_fit.block_frames);
        s += line;
        s += "device projection:\n";
        std::snprintf(line, sizeof(line), "  weights:       %14s\n",
                      fmt_mib(r.device.weights_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  kv state:      %14s\n",
                      fmt_mib(r.device.state_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  lm compute:    %14s\n",
                      fmt_mib(r.device.lm_compute_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  codec compute: %14s\n",
                      fmt_mib(r.device.codec_compute_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  total:         %14s\n",
                      fmt_mib(r.device.total_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "host extras:     %14s%s\n",
                      fmt_mib(r.host_bytes).c_str(),
                      r.device_shares_host_memory
                          ? " (same RAM pool as the device projection)" : "");
        s += line;
        std::snprintf(line, sizeof(line), "margin:          %14s\n",
                      fmt_mib(opts.margin_bytes).c_str());
        s += line;
        if (r.fits) {
            std::snprintf(line, sizeof(line), "verdict: FITS (headroom %s)\n",
                          fmt_mib(r.device_free_bytes - required).c_str());
        } else {
            std::snprintf(line, sizeof(line), "verdict: DOES NOT FIT (short by %s)\n",
                          fmt_mib(required - r.device_free_bytes).c_str());
        }
        s += line;
        r.report = std::move(s);
    }

    return r;
}

}  // namespace audio8
}  // namespace tts_cpp
