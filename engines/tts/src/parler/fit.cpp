// Parler memory-fit preflight (include/tts-cpp/parler/fit.h): project one
// Engine load + one synthesize() against the device memory available right
// now, without reading weight data.
//
// The projection mirrors the real runtime path by construction rather than by
// formula: it drives the same loader (metadata-only: same backend policy, FA
// probe, KV-type resolution, fused-weight eligibility), the same graph
// builders (T5 encode, decoder prefill / deepest step, one DAC window), and
// the same allocation policies -- direct gallocr vs the [backend, CPU-last]
// scheduler fallback -- through ggml's size-only APIs, so it tracks runtime
// changes instead of drifting from them.

#include "tts-cpp/parler/fit.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "fit_price.h"
#include "fit_util.h"
#include "parler/internal.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>

namespace tts_cpp {
namespace parler {

namespace {

using detail::PARLER_DAC_WINDOW_FRAMES;
using detail::PARLER_MAX_NODES;
using fitutil::sat_add;
using fitutil::sat_mul;

std::string fmt_mib(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", (double) bytes / (1024.0 * 1024.0));
    return buf;
}

// Frees whatever a metadata-only load left behind, in the loader's own
// teardown order (no buffers exist in measure mode).
struct model_guard {
    detail::parler_model model;
    ~model_guard() { detail::parler_free_model(model); }
};

// Price one freshly built graph through the same dual-path dispatch
// parler_graph_prepare allocates with (2 * PARLER_MAX_NODES mirrors its
// sched_fallback_ensure size).
bool price_graph(const detail::parler_model & model, ggml_cgraph * gf,
                 ::tts_cpp::detail::fit_graph_price & out) {
    if (!gf) return false;
    return ::tts_cpp::detail::fit_price_graph(model.backend, gf,
                                              2 * PARLER_MAX_NODES, out);
}

}  // namespace

FitResult fit_params(const FitOptions & opts) {
    FitResult r;
    r.model_variant = "parler";

    if (opts.model_gguf_path.empty() || opts.description_tokens <= 0 ||
        opts.prompt_tokens <= 0 || opts.max_frames < 0) {
        r.reason = "invalid-arguments";
        return r;
    }

#ifdef TTS_CPP_USE_COREML
    if (opts.coreml_decoder && !detail::parler_coreml_disabled()) {
        r.reason = "coreml-memory-estimate-unavailable";
        r.report = "The requested Core ML decoder adds resident weights, state and prediction buffers "
                   "outside the native fitter; disable it to project the ggml runtime.\n";
        return r;
    }
#endif

    // An enabled Core ML DAC sidecar runs outside ggml: its resident model and
    // prediction buffers cannot be projected (CosyVoice's rule).
    if (detail::parler_dac_sidecar_enabled(opts.model_gguf_path)) {
        r.reason = "coreml-memory-estimate-unavailable";
        r.report = "The Core ML DAC sidecar's resident model and prediction buffers cannot be "
                   "projected; set PARLER_COREML_DISABLE=1 to project the ggml DAC instead.\n";
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

    // ── Metadata-only load: backend resolution + tensor wiring + sizing ─────
    model_guard m;
    detail::parler_fit_measure load;
    std::string error;
    if (!detail::parler_load_gguf_metadata_only(opts.model_gguf_path, m.model,
                                                opts.n_gpu_layers, load, &error, opts.backend)) {
        r.reason = "model-unreadable";
        return r;
    }

    ggml_backend_t backend = m.model.backend;
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    if (!dev) {
        r.reason = "no-backend-device";
        return r;
    }
    r.device_name   = ggml_backend_dev_name(dev);
    r.device_is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    // Unified-memory devices (CPU, integrated GPUs, Apple Metal, Hexagon)
    // draw the "device" figure from the same physical RAM the host-side
    // buffers live in, so the verdict must charge both against it.
    r.device_shares_host_memory = ::tts_cpp::detail::backend_shares_host_memory(backend);
    {
        size_t free_b = 0, total_b = 0;
        ::tts_cpp::detail::backend_memory(backend, free_b, total_b);
        r.device_free_bytes  = free_b;
        r.device_total_bytes = total_b;
    }

    // ── Workload → graph shapes, as the Engine derives them ─────────────────
    // Widen before comparing: near-INT_MAX inputs must be rejected strictly,
    // never sign-overflow into a wrapped graph shape.
    const detail::parler_hparams & hp = m.model.hparams;
    const int T = opts.description_tokens;
    // The T5 relative-position buckets tensor (and its host mirror) is T*T
    // I32; a product over int is over every real description anyway.
    if ((long long) T * T > (long long) std::numeric_limits<int>::max()) {
        r.reason = "workload-too-large";
        return r;
    }
    // Engine::run refuses max_frames in (0, n_codebooks]: fewer delayed steps
    // cannot yield one audio frame.
    if (opts.max_frames > 0 && opts.max_frames <= hp.n_codebooks) {
        r.reason = "workload-too-large";
        return r;
    }
    const long long N_ll = (long long) opts.prompt_tokens + 1;  // prompt + BOS start frame
    // parler_dec_prefill refuses when N + gen_max_length > n_ctx (the check
    // uses the GGUF's own budget regardless of a lowered max_frames).
    if (N_ll + hp.gen_max_length > hp.n_ctx) {
        r.reason = "workload-too-large";
        return r;
    }
    const int N = (int) N_ll;
    // Delayed decoder steps of one generation: the Engine caps the GGUF's
    // max_length by opts.max_frames when set.
    const int total_steps = (opts.max_frames > 0 && opts.max_frames < hp.gen_max_length)
                                ? opts.max_frames : hp.gen_max_length;
    // The deepest decode step sees every earlier token; parler_dec_step
    // refuses past n_ctx / max_position, so the runnable worst case is:
    const int n_past_max = std::min({N + total_steps - 1, hp.n_ctx - 1, hp.max_position - 1});
    // DAC frames cannot exceed the delayed steps (the delay pattern loses a
    // few slots at the seams; the bound is wrong only in the strict
    // direction, and the DAC arena saturates at one window regardless).
    const int n_frames = total_steps;

    // ── State: KV slab (sized by the loader) + per-description cross K/V ────
    uint64_t state = load.kv_bytes;
    {
        size_t cross_bytes = 0;
        if (!detail::parler_alloc_cross(m.model, T, &cross_bytes)) {
            r.reason = "measurement-failed";
            return r;
        }
        state = sat_add(state, cross_bytes);
    }

    // ── Graphs: the resident arena set ──────────────────────────────────────
    // The Engine's gallocr serves both the prefill and every decode step and
    // grows to the larger; the T5 encode's transient allocr and the
    // persistent DAC window arena coexist with it (a later synthesize() with
    // a new description runs the T5 encode while both are still resident), so
    // they add.  (On the scheduler fallback path all graphs share one sched;
    // summing there over-counts, which is the permitted -- strict --
    // direction.)
    uint64_t lm_compute = 0, codec_compute = 0, host_extra = 0;
    {
        ::tts_cpp::detail::fit_graph_price t5, prefill, step;
        if (!price_graph(m.model, detail::parler_build_t5_fit_graph(m.model, T), t5)) {
            r.reason = "measurement-failed";
            return r;
        }
        if (!price_graph(m.model,
                         detail::parler_build_prefill_fit_graph(m.model, opts.prompt_tokens),
                         prefill)) {
            r.reason = "measurement-failed";
            return r;
        }
        if (!price_graph(m.model, detail::parler_build_step_fit_graph(m.model, n_past_max),
                         step)) {
            r.reason = "measurement-failed";
            return r;
        }
        lm_compute = sat_add(t5.device_bytes,
                             std::max(prefill.device_bytes, step.device_bytes));
        host_extra = sat_add(t5.host_bytes,
                             std::max(prefill.host_bytes, step.host_bytes));
    }
    {
        ggml_context * dac_ctx = nullptr;
        ggml_cgraph * gf = detail::parler_build_dac_fit_graph(m.model, n_frames, &dac_ctx);
        ::tts_cpp::detail::fit_graph_price dac;
        const bool ok = price_graph(m.model, gf, dac);
        if (dac_ctx) ggml_free(dac_ctx);
        if (!ok) {
            r.reason = "measurement-failed";
            return r;
        }
        codec_compute = dac.device_bytes;
        host_extra    = sat_add(host_extra, dac.host_bytes);
    }

    // ── Host-side slabs (scale with the workload, live in host RAM) ─────────
    {
        uint64_t host = host_extra;
        const uint64_t f32   = sizeof(float);
        const uint64_t books = (uint64_t) hp.n_codebooks;
        // Resident tokenizer payload (unigram pieces/scores/charsmap, plus
        // the optional BPE prompt tokenizer).
        uint64_t tok_bytes = m.model.tok_scores.size() * f32 + m.model.tok_charsmap.size();
        for (const std::string & s : m.model.tok_pieces) tok_bytes = sat_add(tok_bytes, s.size());
        for (const std::string & s : m.model.ptok_pieces) tok_bytes = sat_add(tok_bytes, s.size());
        for (const std::string & s : m.model.ptok_merges) tok_bytes = sat_add(tok_bytes, s.size());
        host = sat_add(host, tok_bytes);
        // T5 relative-position buckets, built per encode.
        host = sat_add(host, sat_mul(sat_mul((uint64_t) T, (uint64_t) T), 4));
        // Prefill staging: the causal mask (F16 under FA, F32 otherwise) and
        // the position row.
        host = sat_add(host, sat_mul(sat_mul((uint64_t) N, (uint64_t) N),
                                     m.model.use_fa ? 2 : 4));
        host = sat_add(host, sat_mul((uint64_t) N, 4));
        // Per-step logits and the sampler's per-codebook working vectors
        // (values + sorted copy + index row + double-precision CDF).
        host = sat_add(host, sat_mul(sat_mul(books, (uint64_t) hp.dec_vocab), f32));
        host = sat_add(host, sat_mul((uint64_t) hp.dec_vocab, 20));
        // Delay bookkeeping: the static mask plus the grow-by-swap sequence
        // (two copies at the swap peak), then undelay's rows + codes copies.
        host = sat_add(host, sat_mul(sat_mul(books, (uint64_t) total_steps), 3 * 4));
        host = sat_add(host, sat_mul(sat_mul(books, (uint64_t) total_steps), 2 * 4));
        // DAC window gather + the waveform.
        const int rf    = detail::parler_dac_rf_frames(m.model);
        const int n_win = std::min(n_frames, PARLER_DAC_WINDOW_FRAMES + 2 * rf);
        host = sat_add(host, sat_mul(sat_mul(books, (uint64_t) n_win), 4));
        host = sat_add(host, sat_mul(sat_mul((uint64_t) n_frames,
                                             (uint64_t) hp.dac_hop), f32));
        // Accelerate DAC: per-thread im2col tiles, bounded by every core (and
        // by the engine's default count where the core count is unknown).
        host = sat_add(host, detail::parler_dac_accel_scratch_bytes(
            m.model, (int) std::max(std::thread::hardware_concurrency(),
                                    detail::PARLER_DEFAULT_THREADS)));
        r.host_bytes = host;
    }

    r.device.weights_bytes       = sat_add(sat_add(load.weights_bytes, load.fused_bytes),
                                           load.dac_derived_bytes);
    r.device.state_bytes         = state;
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
        std::snprintf(line, sizeof(line), "model:    parler (kv %s, n_ctx %d%s)\n",
                      ggml_type_name(m.model.kv_type), hp.n_ctx,
                      m.model.use_fa ? ", FA" : "");
        s += line;
        std::snprintf(line, sizeof(line), "device:   %s (%s), free %s / total %s\n",
                      r.device_name.c_str(), r.device_is_cpu ? "CPU" : "GPU",
                      fmt_mib(r.device_free_bytes).c_str(),
                      fmt_mib(r.device_total_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line),
                      "workload: %d description tokens, %d prompt tokens -> prefill %d, "
                      "%d delayed steps (deepest n_past %d)\n",
                      T, opts.prompt_tokens, N, total_steps, n_past_max);
        s += line;
        s += "device projection:\n";
        std::snprintf(line, sizeof(line), "  weights:       %14s\n",
                      fmt_mib(r.device.weights_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  kv state:      %14s (KV slab + cross K/V)\n",
                      fmt_mib(r.device.state_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  lm compute:    %14s (T5 + decoder arenas)\n",
                      fmt_mib(r.device.lm_compute_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  codec compute: %14s (DAC window arena)\n",
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

}  // namespace parler
}  // namespace tts_cpp
