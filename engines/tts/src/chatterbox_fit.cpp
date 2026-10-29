// Chatterbox memory-fit preflight (include/tts-cpp/chatterbox/fit.h):
// project one Engine load + one synthesize() against the device memory
// available right now, without reading weight data.
//
// The projection mirrors the real runtime by construction: the T3 loader
// runs metadata-only (same variant peek, backend policy, KV-type resolution
// with its per-backend downgrades, MTL wqkv stack sizing), the S3Gen loader
// likewise, and the graphs are the real builders priced through ggml's
// size-only APIs.  See src/chatterbox_fit_internal.h (S3Gen side) and
// src/chatterbox_t3_internal.h (T3 side) for the measured pieces.

#include "tts-cpp/chatterbox/fit.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "chatterbox_fit_internal.h"
#include "chatterbox_t3_internal.h"
#include "fit_util.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace tts_cpp {
namespace chatterbox {

namespace {

using fitutil::sat_add;
using fitutil::sat_mul;

std::string fmt_mib(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", (double) bytes / (1024.0 * 1024.0));
    return buf;
}

// Frees whatever a metadata-only T3 load left behind, in the Engine's own
// teardown order (sched before backend; no buffers exist in measure mode).
struct t3_guard {
    detail::chatterbox_model model;
    ~t3_guard() {
        ::tts_cpp::detail::sched_fallback_free(model.sched_fb);
        if (model.ctx_w)     ggml_free(model.ctx_w);
        if (model.ctx_kv)    ggml_free(model.ctx_kv);
        if (model.ctx_stack) ggml_free(model.ctx_stack);
        if (model.backend)   ggml_backend_free(model.backend);
    }
};

}  // namespace

FitResult fit_params(const FitOptions & opts) {
    FitResult r;

    if (opts.t3_gguf_path.empty() || opts.s3gen_gguf_path.empty() ||
        opts.text_tokens <= 0 || opts.n_predict <= 0 || opts.n_ctx < 0) {
        r.reason = "invalid-arguments";
        return r;
    }

    if (!opts.backends_dir.empty()) {
        ::tts_cpp::detail::set_backends_directory(opts.backends_dir);
    }

    // An empty device registry, or an explicit backend that is missing, is
    // always Error, never Success.
    {
        ggml_backend_t probe = ::tts_cpp::detail::backend_request_is_auto(opts.backend)
            ? ::tts_cpp::detail::init_cpu_backend()
            : ::tts_cpp::detail::init_requested_backend(opts.backend, /*verbose=*/false, "chatterbox-fit");
        if (!probe) {
            r.reason = "no-backend-device";
            return r;
        }
        ggml_backend_free(probe);
    }

    // ── T3: metadata-only load (variant peek + backend + KV resolve) ───────
    const ggml_type kv_requested =
        detail::chatterbox_kv_type_from_str(opts.kv_cache_type);
    t3_guard t3;
    detail::t3_load_measure t3m;
    if (!detail::load_model_gguf(opts.t3_gguf_path, t3.model, opts.n_ctx,
                                 opts.n_gpu_layers, kv_requested, &t3m, opts.backend)) {
        r.reason = "model-unreadable";
        return r;
    }
    const detail::chatterbox_hparams & hp = t3.model.hparams;
    const bool mtl = hp.variant == detail::CHBX_VARIANT_MTL;
    r.model_variant = mtl ? "chatterbox-t3-mtl" : "chatterbox-t3-turbo";

    ggml_backend_t backend = t3.model.backend;
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

    // ── Workload → graph shapes, as the eval functions derive them ─────────
    // Widen before summing: a near-INT_MAX --text-tokens would sign-overflow
    // the int sum (UB) before the n_ctx comparison could reject it. n_ctx is
    // clamped to the model's own table by the loader, so a prompt that passes
    // this check always fits int.
    const long long prompt_len_ll = mtl
        ? (1ll + hp.perceiver_queries + (hp.emotion_adv ? 1 : 0)) + opts.text_tokens + 2
        : 1ll + hp.cond_prompt_len + opts.text_tokens + 1;
    if (prompt_len_ll >= (long long) hp.n_ctx) {
        r.reason = "workload-too-large";  // eval_prompt would refuse it
        return r;
    }
    const int prompt_len = (int) prompt_len_ll;
    // The decode loop appends one token per step until the budget or the
    // context runs out; the deepest step sees every earlier token.
    const int n_gen      = std::min(opts.n_predict, hp.n_ctx - prompt_len);
    const int n_past_max = std::min(hp.n_ctx - 1, prompt_len + n_gen - 1);

    uint64_t lm_device = 0, host_extra = 0;
    {
        uint64_t dev_b = 0, host_b = 0;
        if (!detail::t3_measure_compute(t3.model, opts.text_tokens, n_past_max,
                                        dev_b, host_b)) {
            r.reason = "measurement-failed";
            return r;
        }
        lm_device  = dev_b;
        host_extra = sat_add(host_extra, host_b);
    }

    // ── S3Gen: metadata-only load + the five resident stage arenas ─────────
    detail::s3gen_fit_measure sm;
    {
        // Distinguish an unreadable GGUF from a measurement failure with the
        // same metadata-only peek the loaders use.
        gguf_init_params peek = { /*no_alloc=*/true, /*ctx=*/nullptr };
        gguf_context * g = gguf_init_from_file(opts.s3gen_gguf_path.c_str(), peek);
        if (!g) {
            r.reason = "model-unreadable";
            return r;
        }
        gguf_free(g);
        std::string error;
        if (!detail::s3gen_measure_fit(opts.s3gen_gguf_path, opts.n_gpu_layers, opts.backend,
                                       n_gen, sm, &error)) {
            r.reason = "measurement-failed";
            return r;
        }
    }

    // ── Host-side extras ────────────────────────────────────────────────────
    {
        uint64_t host = host_extra;
        host = sat_add(host, sm.host_compute_bytes);
        host = sat_add(host, sm.host_arena_bytes);
        host = sat_add(host, sm.host_slab_bytes);
        // T3-side host slabs: per-step logits (cond+uncond on MTL), the token
        // trajectory, the prompt graph's F16 causal mask, and the resident
        // tokenizer payload (HF JSON on MTL; token+merge tables on Turbo).
        host = sat_add(host, sat_mul((uint64_t) (mtl ? 2 : 1) * hp.n_speech_vocab,
                                     sizeof(float)));
        host = sat_add(host, sat_mul((uint64_t) n_gen, sizeof(int32_t)));
        host = sat_add(host, sat_mul(sat_mul((uint64_t) prompt_len, (uint64_t) prompt_len), 2));
        uint64_t tok_bytes = t3.model.mtl_tokenizer_json.size();
        for (const std::string & s : t3.model.tok_tokens) tok_bytes = sat_add(tok_bytes, s.size());
        for (const std::string & s : t3.model.tok_merges) tok_bytes = sat_add(tok_bytes, s.size());
        host = sat_add(host, tok_bytes);
        // The Engine accumulates the utterance PCM on top of the S3Gen-side
        // wav/pcm_out copies already counted in host_slab_bytes.
        host = sat_add(host, sat_mul((uint64_t) sm.T_mel * 480, sizeof(float)));
        r.host_bytes = host;
    }

    r.device.weights_bytes = sat_add(sat_add((uint64_t) t3m.weights_bytes,
                                             (uint64_t) t3m.stack_bytes),
                                     sm.weights_bytes);
    r.device.state_bytes         = t3m.kv_bytes;
    r.device.lm_compute_bytes    = lm_device;
    r.device.codec_compute_bytes = sm.device_compute_bytes;
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
        std::snprintf(line, sizeof(line), "model:    %s (kv %s, n_ctx %d)\n",
                      r.model_variant.c_str(), ggml_type_name(hp.kv_type), hp.n_ctx);
        s += line;
        std::snprintf(line, sizeof(line), "device:   %s (%s), free %s / total %s\n",
                      r.device_name.c_str(), r.device_is_cpu ? "CPU" : "GPU",
                      fmt_mib(r.device_free_bytes).c_str(),
                      fmt_mib(r.device_total_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line),
                      "workload: %d text tokens -> prompt %d, %d speech tokens "
                      "(s3gen T=%d, T_mel=%d%s)\n",
                      opts.text_tokens, prompt_len, n_gen, sm.n_total, sm.T_mel,
                      sm.used_b2 ? ", CFG B=2" : "");
        s += line;
        s += "device projection:\n";
        std::snprintf(line, sizeof(line), "  weights:       %14s\n",
                      fmt_mib(r.device.weights_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  kv state:      %14s\n",
                      fmt_mib(r.device.state_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  t3 compute:    %14s\n",
                      fmt_mib(r.device.lm_compute_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  s3gen compute: %14s (5 resident stage arenas)\n",
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

}  // namespace chatterbox
}  // namespace tts_cpp
