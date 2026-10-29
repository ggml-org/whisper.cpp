// CosyVoice3 memory-fit preflight (include/tts-cpp/cosyvoice/fit.h): project
// one Engine construction + one synthesize() against the device memory
// available right now, without reading weight data.
//
// The pipeline is STAGED (cosyvoice_engine.cpp loads and frees llm / flow /
// hift one at a time), so the projection prices each phase with the real
// loaders (measure mode) and the real graph builders (size-only pricing in
// cosyvoice_pipeline.cpp), and takes the verdict from the PEAK phase.

#include "tts-cpp/cosyvoice/fit.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "cosyvoice_fit_internal.h"
#include "cosyvoice_coreml.h"
#include "fit_util.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>

namespace tts_cpp {
namespace cosyvoice {

namespace {

using fitutil::sat_add;
using fitutil::sat_mul;

std::string fmt_mib(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", (double) bytes / (1024.0 * 1024.0));
    return buf;
}

// One staged phase of the pipeline.  total() is what is resident on the
// device while the phase runs; the verdict takes the max across phases.
struct phase_projection {
    const char * name = "";
    uint64_t weights = 0;   // the phase's weight buffer on the device
    uint64_t state   = 0;   // persistent state while the phase runs (LM KV)
    uint64_t arena   = 0;   // the phase's compute arena(s)
    uint64_t host    = 0;   // phase-local host slabs (+ sched CPU portions)
    uint64_t device_total() const {
        return fitutil::sat_add(fitutil::sat_add(weights, state), arena);
    }
};

struct model_guard {
    model_ctx m;
    bool armed = false;
    ~model_guard() {
        if (armed) cosyvoice_free(m);
    }
};

}  // namespace

FitResult fit_params(const FitOptions & opts) {
    FitResult r;
    r.model_variant = "cosyvoice3";

    if (opts.llm_gguf_path.empty() || opts.flow_gguf_path.empty() ||
        opts.hift_gguf_path.empty() || opts.voice_gguf_path.empty() ||
        opts.text_tokens <= 0 || opts.speech_tokens < 0) {
        r.reason = "invalid-arguments";
        return r;
    }

    if (cosyvoice_coreml::has_sidecar(opts.llm_gguf_path) ||
        cosyvoice_coreml::has_sidecar(opts.flow_gguf_path) ||
        cosyvoice_coreml::has_sidecar(opts.flow_gguf_path, "-frontend") ||
        cosyvoice_coreml::has_sidecar(opts.hift_gguf_path) ||
        cosyvoice_coreml::has_sidecar(opts.hift_gguf_path, "-f0")) {
        r.reason = "coreml-memory-estimate-unavailable";
        r.report = "Core ML resident models and prediction buffers cannot be projected.\n";
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

    // Same backend policy as Engine::Impl (validated-backend allowlist,
    // Mali refused, per-platform Vulkan/CUDA gate, or the explicit device).
    ggml_backend_t backend = cosyvoice_init_backend(opts.backend, opts.n_gpu_layers,
                                                    opts.vulkan_device,
                                                    /*gpu_present_but_unused=*/nullptr);
    if (!backend) {
        r.reason = "no-backend-device";
        return r;
    }
    struct backend_guard {
        ggml_backend_t b;
        ~backend_guard() { if (b) ggml_backend_free(b); }
    } bg{backend};

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
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

    // ── Metadata-only loads (all four GGUFs; headers only, so holding them
    //    together costs nothing) ──────────────────────────────────────────────
    model_guard voice_m, llm_m, flow_m, hift_m;
    cosyvoice_fit_load_measure voice_meas, llm_meas, flow_meas, hift_meas;
    int n_prompt_stok = 0, n_ptok = 0, mel_len1 = 0, SPK = 0;
    int64_t RNW = 0;
    qwen_hp hp;
    int VS = 0, D = 0;
    try {
        // The engine reads the voice GGUF on the shared backend only when
        // that backend is the CPU; on a GPU engine it spins up its own owned
        // CPU backend (the tensors are copied to host vectors and freed).
        voice_m.m = cosyvoice_load_gguf_metadata_only(
            opts.voice_gguf_path, r.device_is_cpu ? backend : nullptr, voice_meas);
        voice_m.armed = true;
        n_prompt_stok = (int) ggml_nelements(cosyvoice_get(voice_m.m, "voice/prompt_stok"));
        n_ptok        = (int) ggml_nelements(cosyvoice_get(voice_m.m, "voice/prompt_token"));
        mel_len1      = (int) (ggml_nelements(cosyvoice_get(voice_m.m, "voice/prompt_feat")) / 80);
        SPK           = (int) ggml_nelements(cosyvoice_get(voice_m.m, "voice/embedding"));

        llm_m.m = cosyvoice_load_gguf_metadata_only(opts.llm_gguf_path, backend, llm_meas);
        llm_m.armed = true;
        hp = cosyvoice_qwen_hp(llm_m.m);
        VS = (int) cosyvoice_get(llm_m.m, "lm/llm_decoder/weight")->ne[1];
        D  = hp.hidden;

        flow_m.m = cosyvoice_load_gguf_metadata_only(opts.flow_gguf_path, backend, flow_meas);
        flow_m.armed = true;
        RNW = cosyvoice_get(flow_m.m, "flow/rand_noise")->ne[0];

        hift_m.m = cosyvoice_load_gguf_metadata_only(opts.hift_gguf_path, backend, hift_meas);
        hift_m.armed = true;
    } catch (const std::exception &) {
        r.reason = "model-unreadable";
        return r;
    }

    // ── Workload → shapes, exactly as Engine::Impl::run derives them ────────
    // Widen before summing: near-INT_MAX text must be rejected strictly,
    // never sign-overflow into a wrapped graph shape.
    const long long L0_ll        = 1ll + opts.text_tokens + 1 + n_prompt_stok;
    const long long max_steps_ll = 50ll * ((long long) opts.text_tokens + 1);
    const long long max_P_ll     = L0_ll + max_steps_ll + 1;
    if (max_P_ll > (long long) std::numeric_limits<int>::max() / 4) {
        r.reason = "workload-too-large";
        return r;
    }
    const int L0        = (int) L0_ll;
    const int max_steps = (int) max_steps_ll;
    const int n_gen     = (opts.speech_tokens > 0)
                              ? std::min(opts.speech_tokens, max_steps) : max_steps;
    const long long T_tok_ll = (long long) n_ptok + n_gen;
    const long long TM_ll    = 2 * T_tok_ll;   // token_mel_ratio = 2
    // cosyvoice_flow_run refuses a mel length past the baked rand_noise
    // frames (~5 min of speech); mirror the same gate.
    if (TM_ll > RNW || TM_ll <= mel_len1) {
        r.reason = "workload-too-large";
        return r;
    }
    const int T_tok    = (int) T_tok_ll;
    const int TM       = (int) TM_ll;
    const int mel_len2 = TM - mel_len1;
    const int T_mel    = mel_len2;
    const int T_wav    = T_mel * 480;          // prod(upsample_rates) * hop_len
    const int T_stft   = T_wav / 4 + 1;

    // Persistent host baseline: the baked-voice vectors the Engine keeps for
    // its whole life.  (The Qwen tokenizer maps live outside the GGUF set and
    // are not projected -- see fit.h.)
    const uint64_t f32 = sizeof(float);
    uint64_t base_host = 0;
    base_host = sat_add(base_host, sat_mul((uint64_t) n_prompt_stok + n_ptok, 4));
    base_host = sat_add(base_host, sat_mul(sat_mul((uint64_t) mel_len1, 80), f32));
    base_host = sat_add(base_host, sat_mul((uint64_t) SPK, f32));

    // ── Phase projections ────────────────────────────────────────────────────
    phase_projection ph_llm{"llm"}, ph_flow{"flow"}, ph_hift{"hift"};
    std::string error;

    {   // LM phase
        uint64_t kv_bytes = 0;
        cosyvoice_fit_price arena;
        if (!cosyvoice_fit_measure_llm(llm_m.m, hp, L0, max_steps, kv_bytes, arena, &error)) {
            r.reason = "measurement-failed";
            return r;
        }
        ph_llm.weights = llm_meas.device_bytes;
        ph_llm.state   = kv_bytes;
        ph_llm.arena   = arena.device_bytes;
        uint64_t host  = sat_add(llm_meas.host_bytes, arena.host_bytes);
        // lm_input row [D, L0], the prefill causal mask [L0, L0], positions.
        host = sat_add(host, sat_mul(sat_mul((uint64_t) L0, (uint64_t) D), f32));
        host = sat_add(host, sat_mul(sat_mul((uint64_t) L0, (uint64_t) L0), f32));
        host = sat_add(host, sat_mul((uint64_t) L0, 4));
        // Per-step logits + the RAS sampler's prob/order/p2 vectors, the
        // token trajectory, and the per-step embedding row.
        host = sat_add(host, sat_mul((uint64_t) VS, f32 + f32 + 4 + 8));
        host = sat_add(host, sat_mul((uint64_t) max_steps, 4));
        host = sat_add(host, sat_mul((uint64_t) D, f32));
        ph_llm.host = host;
    }
    {   // flow phase
        cosyvoice_fit_price arena;
        if (!cosyvoice_fit_measure_flow(flow_m.m, T_tok, TM, SPK, arena, &error)) {
            r.reason = "measurement-failed";
            return r;
        }
        ph_flow.weights = flow_meas.device_bytes;
        ph_flow.arena   = arena.device_bytes;
        uint64_t host   = sat_add(flow_meas.host_bytes, arena.host_bytes);
        const uint64_t mel_tm = sat_mul(sat_mul((uint64_t) 80, (uint64_t) TM), f32);
        // mu_host / cond_host / x_host, the CFG-batched mu/cond/x/dphi
        // uploads (B=2 each), the baked-noise host mirror, positions/tokids,
        // and the trimmed mel handed to the HiFT phase.
        host = sat_add(host, sat_mul(mel_tm, 3));
        host = sat_add(host, sat_mul(mel_tm, 2 * 4));
        host = sat_add(host, sat_mul(sat_mul((uint64_t) RNW, 80), f32));
        host = sat_add(host, sat_mul((uint64_t) TM + T_tok, 4));
        host = sat_add(host, sat_mul(sat_mul((uint64_t) 80, (uint64_t) mel_len2), f32));
        host = sat_add(host, sat_mul((uint64_t) n_gen, 4));  // speech tokens
        ph_flow.host = host;
    }
    {   // hift phase
        cosyvoice_fit_price arena;
        if (!cosyvoice_fit_measure_hift(hift_m.m, T_mel, arena, &error)) {
            r.reason = "measurement-failed";
            return r;
        }
        ph_hift.weights = hift_meas.device_bytes;
        ph_hift.arena   = arena.device_bytes;
        uint64_t host   = sat_add(hift_meas.host_bytes, arena.host_bytes);
        // The mel from the flow phase stays alive, plus f0, the upsampled f0,
        // the 9-harmonic SineGen scratch (sines, the recorded gauss draws, and
        // per-worker phase accumulators — one rad per harmonic worker, up to
        // all 9 concurrently), the excitation, its STFT, the iSTFT window-sum,
        // and the waveform.
        host = sat_add(host, sat_mul(sat_mul((uint64_t) 80, (uint64_t) T_mel), f32));
        host = sat_add(host, sat_mul((uint64_t) T_mel, f32));
        host = sat_add(host, sat_mul((uint64_t) T_wav, f32));            // f0_up
        host = sat_add(host, sat_mul((uint64_t) T_wav, 9 * f32));        // sines
        host = sat_add(host, sat_mul((uint64_t) T_wav, 9 * f32));        // noise draws
        host = sat_add(host, sat_mul((uint64_t) T_wav, 9 * 8));          // rad (doubles, per worker)
        host = sat_add(host, sat_mul((uint64_t) T_wav / 480 + 1, 9 * 2 * 8));
        host = sat_add(host, sat_mul((uint64_t) T_wav, f32));            // source
        host = sat_add(host, sat_mul(sat_mul((uint64_t) T_stft, 18), f32));
        host = sat_add(host, sat_mul((uint64_t) T_wav + 16, f32));       // w_sum
        host = sat_add(host, sat_mul((uint64_t) T_wav, f32));            // wav
        ph_hift.host = host;
    }

    // ── Peak phase → verdict ─────────────────────────────────────────────────
    const phase_projection * phases[3] = { &ph_llm, &ph_flow, &ph_hift };
    const phase_projection * peak = phases[0];
    for (const phase_projection * p : phases) {
        const uint64_t cand = r.device_shares_host_memory
            ? sat_add(p->device_total(), sat_add(p->host, base_host))
            : p->device_total();
        const uint64_t best = r.device_shares_host_memory
            ? sat_add(peak->device_total(), sat_add(peak->host, base_host))
            : peak->device_total();
        if (cand > best) peak = p;
    }

    r.device.weights_bytes       = peak->weights;
    r.device.state_bytes         = peak->state;
    r.device.lm_compute_bytes    = (peak == &ph_llm) ? peak->arena : 0;
    r.device.codec_compute_bytes = (peak == &ph_llm) ? 0 : peak->arena;
    r.device.total_bytes         = peak->device_total();
    r.host_bytes                 = sat_add(base_host, peak->host);

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
        std::snprintf(line, sizeof(line), "model:    cosyvoice3 (staged pipeline: peak phase decides)\n");
        s += line;
        std::snprintf(line, sizeof(line), "device:   %s (%s), free %s / total %s\n",
                      r.device_name.c_str(), r.device_is_cpu ? "CPU" : "GPU",
                      fmt_mib(r.device_free_bytes).c_str(),
                      fmt_mib(r.device_total_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line),
                      "workload: %d text tokens -> L0 %d, KV capacity %d, %d speech tokens "
                      "(TM %d, mel %d)\n",
                      opts.text_tokens, L0, L0 + max_steps + 1, n_gen, TM, mel_len2);
        s += line;
        s += "phases (weights + state + compute | host):\n";
        for (const phase_projection * p : phases) {
            std::snprintf(line, sizeof(line),
                          "  %-5s %14s + %10s + %10s | %10s%s\n",
                          p->name, fmt_mib(p->weights).c_str(), fmt_mib(p->state).c_str(),
                          fmt_mib(p->arena).c_str(), fmt_mib(p->host).c_str(),
                          p == peak ? "   <- peak" : "");
            s += line;
        }
        std::snprintf(line, sizeof(line), "device projection (peak phase '%s'):\n", peak->name);
        s += line;
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
        std::snprintf(line, sizeof(line), "host extras:     %14s (voice vectors + peak phase)%s\n",
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

}  // namespace cosyvoice
}  // namespace tts_cpp
