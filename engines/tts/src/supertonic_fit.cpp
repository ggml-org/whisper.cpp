// Supertonic memory-fit preflight (include/tts-cpp/supertonic/fit.h): project
// one Engine load + one synthesize() against the device memory available
// right now, without reading weight data.
//
// The projection drives the real loader in measure mode (same backend policy,
// capability probes, per-tensor storage-type decisions, pre-baked audit
// tensors and the GPU pre-transpose roster) and the real per-stage graph
// builders through ggml's size-only APIs (supertonic_fit_measure_* in the
// stage TUs). Fused GPU/CPU graph paths are modelled; the CPU pointwise
// multi-cache path is refused rather than understated (see fit.h).

#include "tts-cpp/supertonic/fit.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "fit_util.h"
#include "supertonic_internal.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace tts_cpp {
namespace supertonic {

namespace {

namespace det = ::tts_cpp::supertonic::detail;

using fitutil::sat_add;
using fitutil::sat_mul;
using fitutil::sat_u64_from_double;

std::string fmt_mib(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", (double) bytes / (1024.0 * 1024.0));
    return buf;
}

struct model_guard {
    det::supertonic_model model;
    bool armed = false;
    ~model_guard() {
        if (armed) det::free_supertonic_model(model);
    }
};

}  // namespace

FitResult fit_params(const FitOptions & opts) {
    FitResult r;
    r.model_variant = "supertonic";

    det::supertonic_precision precision = det::supertonic_precision::Auto;
    if (opts.precision == "f16") {
        precision = det::supertonic_precision::F16;
    } else if (opts.precision == "q8_0") {
        precision = det::supertonic_precision::Q8_0;
    } else if (opts.precision == "f32") {
        precision = det::supertonic_precision::F32;
    } else if (opts.precision != "auto" && !opts.precision.empty()) {
        r.reason = "invalid-arguments";
        return r;
    }
    if (opts.model_gguf_path.empty() || opts.text_tokens <= 0 ||
        !(opts.audio_seconds > 0.0f) || opts.steps < 0 || opts.vulkan_device < 0) {
        r.reason = "invalid-arguments";
        return r;
    }

    if (!opts.backends_dir.empty()) {
        ::tts_cpp::detail::set_backends_directory(opts.backends_dir);
    }

    // An empty device registry is always Error, never Success.
    {
        ggml_backend_t probe = ::tts_cpp::detail::init_cpu_backend();
        if (!probe) {
            r.reason = "no-backend-device";
            return r;
        }
        ggml_backend_free(probe);
    }

    // ── Metadata-only load: backend + storage-type resolution + sizing ──────
    model_guard m;
    det::supertonic_fit_load_measure load;
    if (!det::load_supertonic_gguf_metadata_only(opts.model_gguf_path, m.model,
                                                 opts.n_gpu_layers, opts.f16_weights,
                                                 precision, opts.vulkan_device,
                                                 /*f16_weights_deny_list=*/{}, load,
                                                 opts.backend)) {
        r.reason = "model-unreadable";
        return r;
    }
    m.armed = true;
    r.model_variant = m.model.hparams.arch;

    ggml_backend_t backend = m.model.backend;
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    if (!dev) {
        r.reason = "no-backend-device";
        return r;
    }
    r.device_name   = ggml_backend_dev_name(dev);
    r.device_is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    r.device_shares_host_memory = ::tts_cpp::detail::backend_shares_host_memory(backend);
    {
        size_t free_b = 0, total_b = 0;
        ::tts_cpp::detail::backend_memory(backend, free_b, total_b);
        r.device_free_bytes  = free_b;
        r.device_total_bytes = total_b;
    }

    // ── Workload → shapes, exactly as Engine::Impl::run_single_chunk ────────
    // Validated BEFORE the compute-path refusal below: an oversized workload
    // is invalid on every backend, so it reports workload-too-large even
    // where the CPU refusal would also apply.
    const det::supertonic_hparams & hp = m.model.hparams;
    const int L_text = opts.text_tokens;
    // The text encoder's one-graph cache holds a persistent L x L rel_band;
    // widen the product before it can wrap.
    if ((long long) L_text * L_text > (long long) std::numeric_limits<int>::max() / 64) {
        r.reason = "workload-too-large";
        return r;
    }
    const int steps = opts.steps > 0 ? opts.steps : hp.default_steps;
    const int chunk = hp.base_chunk_size * hp.ttl_chunk_compress_factor;
    const uint64_t wav_len_u =
        sat_u64_from_double((double) opts.audio_seconds * hp.sample_rate);
    const uint64_t latent_u = std::max<uint64_t>(1, (wav_len_u + chunk - 1) / chunk);
    if (latent_u > (uint64_t) std::numeric_limits<int>::max() / 8192 ||
        (long long) steps * 16384 > (long long) std::numeric_limits<int>::max() / 2) {
        r.reason = "workload-too-large";
        return r;
    }
    const int latent_len = (int) latent_u;
    const int T_wav      = latent_len * chunk;

    // Only the fused graph path is modelled: the CPU pointwise vector-estimator
    // multi-cache set is a documented follow-up. Refusing (Error) beats a
    // wrong FITS.
    if (det::model_prefers_cpu_kernels(m.model)) {
        r.reason = "compute-path-not-supported";
        return r;
    }

    // ── Stage graph-cache arenas (all resident at once, so they sum) ────────
    std::string error;
    uint64_t text_bytes = 0, dur_bytes = 0, ve_bytes = 0;
    uint64_t voc_dev = 0, voc_host = 0;
    // Each measure refuses unmodelled dispatch paths (one-graph / loop graph
    // disabled by env); distinguish that from a real failure.
    const auto measure_reason = [&error]() {
        return error.find("not modelled") != std::string::npos
                   ? "compute-path-not-supported" : "measurement-failed";
    };
    if (!det::supertonic_fit_measure_text_encoder(m.model, L_text, text_bytes, &error) ||
        !det::supertonic_fit_measure_duration(m.model, L_text + 1, dur_bytes, &error)) {
        r.reason = measure_reason();
        return r;
    }
    if (!det::supertonic_fit_measure_vector(m.model, latent_len, L_text, steps,
                                            ve_bytes, &error)) {
        r.reason = measure_reason();
        return r;
    }
    if (!det::supertonic_fit_measure_vocoder(m.model, latent_len, voc_dev, voc_host,
                                             &error)) {
        r.reason = "measurement-failed";
        return r;
    }

    // ── Host-side slabs ──────────────────────────────────────────────────────
    // Steady state: the load's persistent host caches plus one synthesis'
    // working set (text ids + embedding, the encoder's host round-trip
    // vectors, one L x L mask staging row, latents + mask + final latent, the
    // waveform and its result copy).  The load transient (gguf full copy +
    // conversion staging next to the freshly allocated weights) often exceeds
    // it; the verdict takes the larger.
    const uint64_t f32 = sizeof(float);
    uint64_t synth_host = load.host_bytes;
    synth_host = sat_add(synth_host, sat_mul((uint64_t) L_text, 12));
    synth_host = sat_add(synth_host, sat_mul(sat_mul((uint64_t) L_text, 256), f32));
    synth_host = sat_add(synth_host, sat_mul(sat_mul((uint64_t) L_text, 256), 6 * f32));
    synth_host = sat_add(synth_host, sat_mul(sat_mul((uint64_t) L_text, (uint64_t) L_text), f32));
    synth_host = sat_add(synth_host,
                         sat_mul(sat_mul((uint64_t) latent_len, (uint64_t) hp.latent_channels),
                                 2 * f32));
    synth_host = sat_add(synth_host, sat_mul((uint64_t) latent_len, f32));
    synth_host = sat_add(synth_host, sat_mul((uint64_t) T_wav, 2 * f32));
    if (hp.cfg_enabled()) {
        // supertonic3's unconditional-pass host staging in the [C, T] loop
        // (supertonic_vector_loop_one_graph_ggml): text_u [256 x L_text], the
        // style special-token layouts sv_u / sk_u [256 x 50] each alongside
        // their read_f32 sources (~4 x 50 x 256 + 256 floats), and the
        // batch=2 mask / pos_q / cfg-scale upload rows (~2 x latent_len
        // elements each).
        synth_host = sat_add(synth_host, sat_mul(sat_mul((uint64_t) L_text, 256), f32));
        synth_host = sat_add(synth_host, sat_mul((uint64_t) 4 * 50 * 256 + 256, f32));
        synth_host = sat_add(synth_host, sat_mul((uint64_t) latent_len, 6 * f32));
    }
    synth_host = sat_add(synth_host, voc_host);
    const uint64_t load_host = sat_add(load.host_bytes, load.host_transient_bytes);
    r.host_bytes = std::max(synth_host, load_host);

    r.device.weights_bytes       = sat_add(load.weights_bytes, load.extra_bytes);
    r.device.state_bytes         = 0;  // non-autoregressive: no KV-style state
    r.device.lm_compute_bytes    = sat_add(sat_add(text_bytes, dur_bytes), ve_bytes);
    r.device.codec_compute_bytes = voc_dev;
    r.device.total_bytes =
        sat_add(sat_add(r.device.weights_bytes, r.device.state_bytes),
                sat_add(r.device.lm_compute_bytes, r.device.codec_compute_bytes));

    // ── Verdict ─────────────────────────────────────────────────────────────
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
        std::snprintf(line, sizeof(line), "model:    %s (%s%s%s)\n",
                      r.model_variant.c_str(), opts.precision.c_str(),
                      m.model.use_f16_weights ? ", f16 weights" : "",
                      hp.cfg_enabled() ? ", CFG" : "");
        s += line;
        std::snprintf(line, sizeof(line), "device:   %s (%s), free %s / total %s\n",
                      r.device_name.c_str(), r.device_is_cpu ? "CPU" : "GPU",
                      fmt_mib(r.device_free_bytes).c_str(),
                      fmt_mib(r.device_total_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line),
                      "workload: %d text tokens, %.1f s audio -> latent %d, %d CFM steps\n",
                      L_text, (double) opts.audio_seconds, latent_len, steps);
        s += line;
        s += "device projection:\n";
        std::snprintf(line, sizeof(line), "  weights:       %14s (buffer_w + pretransposed extra)\n",
                      fmt_mib(r.device.weights_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line),
                      "  stage caches:  %14s (text %s + duration %s + estimator %s)\n",
                      fmt_mib(r.device.lm_compute_bytes).c_str(),
                      fmt_mib(text_bytes).c_str(), fmt_mib(dur_bytes).c_str(),
                      fmt_mib(ve_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  vocoder cache: %14s\n",
                      fmt_mib(r.device.codec_compute_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  total:         %14s\n",
                      fmt_mib(r.device.total_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "host extras:     %14s (%s)%s\n",
                      fmt_mib(r.host_bytes).c_str(),
                      load_host > synth_host ? "load transient dominates"
                                             : "synthesis working set",
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

}  // namespace supertonic
}  // namespace tts_cpp
