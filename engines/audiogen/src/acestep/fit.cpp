// Memory-fit preflight (include/audiogen-cpp/acestep/fit.h): project the
// ACE-Step stage GGUFs + workload against the memory available right now,
// without reading weight data.
//
// The projection mirrors the real runtime path by construction rather than by
// formula: it resolves the same backends (engine_backends.h), drives the same
// stage loaders in metadata-only mode, and prices the same compute graphs the
// pipeline builds -- at the workload's shapes -- through ggml's size-only APIs
// (ggml_backend_alloc_ctx_tensors_from_buft_size, ggml_gallocr_reserve_n_size),
// so it tracks runtime changes instead of drifting from them.
//
// Residency model (see Engine::Impl in engine.cpp): by default generate()
// time-shares the stages -- each phase loads its stage and frees it right
// after, with one deliberate overlap (the cond encoder is resident while the
// text encoder runs). The projection therefore takes the PEAK PHASE per memory
// pool. ACESTEP_KEEP_STAGES keeps everything resident, and the projection
// becomes the sum of all persistent buffers plus the largest ephemeral graph.

#include "audiogen-cpp/acestep/fit.h"

#include "audiogen-cpp/acestep/engine.h"

#include "acestep/backend_registry.h"
#include "acestep/bpe_tokenizer.h"  // TOKEN_IM_END / AUDIO_CODE_BASE (LM head ranges)
#include "acestep/cond_ggml.h"
#include "acestep/detok_ggml.h"
#include "acestep/dit_ggml.h"
#include "acestep/dit_gguf.h"
#include "acestep/engine_backends.h"
#include "acestep/engine_paths.h"
#include "acestep/fit_measure.h"
#include "acestep/fit_pools.h"
#include "acestep/fit_util.h"
#include "acestep/lm_ggml.h"
#include "acestep/stage_placement.h"
#include "acestep/textenc_ggml.h"
#include "acestep/vae_encode_windows.h"
#include "acestep/vae_ggml.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

const char * fit_status_name(FitStatus status) {
    switch (status) {
        case FitStatus::Success: return "success";
        case FitStatus::Failure: return "failure";
        case FitStatus::Error:   return "error";
    }
    return "error";
}

namespace {

using fitutil::sat_add;
using fitutil::sat_mul;
using fitutil::sat_u64_from_double;

std::string fmt_mib(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", (double) bytes / (1024.0 * 1024.0));
    return buf;
}

uint64_t f32_bytes(uint64_t n) { return sat_mul(n, sizeof(float)); }
uint64_t f16_bytes(uint64_t n) { return sat_mul(n, 2); }

// The engine loads the LM with these (engine.cpp ensure_lm); the KV budget is
// fixed by them, independent of the workload.
constexpr int LM_MAX_SEQ_LEN = 2048;
constexpr int LM_N_KV_SETS   = 2;

// Pipeline constants mirrored from engine.cpp (AUDIO_* block). The 25 Hz
// latent rate is implied by CODE_FRAME_RATIO (5 Hz codes x 5 frames/code).
constexpr int SAMPLE_RATE      = 48000;
constexpr int LATENT_CHANNELS  = 64;
constexpr int CODE_FRAME_RATIO = 5;
constexpr int VAE_UPSAMPLE     = 1920;
constexpr int COND_HIDDEN      = 2048;  // cond encoder output width (cond_ggml.cpp COND_H)
constexpr int TEXTENC_HIDDEN   = 1024;  // Qwen3-Embedding hidden size (textenc_ggml.h)

// Same truthy parse as Engine::create's ACESTEP_KEEP_STAGES read.
bool env_keep_stages() {
    const char * e = std::getenv("ACESTEP_KEEP_STAGES");
    return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T' || e[0] == 'y' || e[0] == 'Y');
}

PoolCharge charge_backend(const std::vector<FitPool> & pools, ggml_backend_t backend, uint64_t bytes) {
    return fit_charge(pools, backend ? ggml_backend_get_device(backend) : nullptr, bytes);
}

void register_backend_pools(std::vector<FitPool> & pools, std::initializer_list<ggml_backend_t> backends) {
    for (ggml_backend_t backend : backends) {
        fit_pool_register(pools, fit_pool_for_device(ggml_backend_get_device(backend)));
    }
}

std::vector<FitDevicePool> extra_device_pools(const std::vector<FitPool> & pools, const PoolCharge & peak,
                                              size_t primary) {
    std::vector<FitDevicePool> extras;
    for (size_t i = 0; i < pools.size(); ++i) {
        if (i == FIT_HOST_POOL || i == primary) continue;
        FitDevicePool d;
        d.name               = ggml_backend_dev_name(pools[i].device);
        d.shares_host_memory = pools[i].shares_host_memory;
        d.free_bytes         = pools[i].free_bytes;
        d.total_bytes        = pools[i].total_bytes;
        d.peak_bytes         = peak.bytes[i];
        extras.push_back(std::move(d));
    }
    return extras;
}

std::string format_extra_device_lines(const std::vector<FitDevicePool> & extras) {
    std::string s;
    char        line[320];
    for (const FitDevicePool & d : extras) {
        std::snprintf(line, sizeof(line), "extra:    %s, free %s / total %s%s\n", d.name.c_str(),
                      fmt_mib(d.free_bytes).c_str(), fmt_mib(d.total_bytes).c_str(),
                      d.shares_host_memory ? " (shares host RAM)" : "");
        s += line;
    }
    return s;
}

std::string format_extra_device_peaks(const std::vector<FitDevicePool> & extras) {
    std::string s;
    for (const FitDevicePool & d : extras) {
        s += ", " + d.name + " " + fmt_mib(d.peak_bytes);
    }
    return s;
}

}  // namespace

static FitResult fit_params_unguarded(const FitOptions & opts) {
    FitResult r;

    if (!(opts.duration_seconds > 0.0f) || opts.text_tokens < 1 || opts.lyric_tokens < 1 ||
        opts.lm_prompt_tokens < 0 || opts.lm_max_new_tokens < 0 ||
        opts.keep_stages < -1 || opts.keep_stages > 1) {
        r.reason = "invalid-arguments";
        return r;
    }
    // Bound the token-derived shapes like the T/max_codes guard below bounds
    // the duration-derived ones: parse_i32 accepts up to INT_MAX, and enc_S /
    // the derived LM prompt sum several counts, so unbounded inputs would
    // overflow int (UB) and hand the graph builders negative dimensions -- a
    // preflight must refuse, not crash or wrap. Checked in uint64_t; the cap
    // matches the T guard so every projected dimension stays comfortably
    // inside int arithmetic.
    {
        const uint64_t token_cap = (uint64_t) std::numeric_limits<int>::max() / 4;
        const uint64_t enc_s_u64 = sat_add(sat_add((uint64_t) opts.lyric_tokens, 1),
                                           (uint64_t) opts.text_tokens);
        if ((uint64_t) opts.text_tokens > token_cap || (uint64_t) opts.lyric_tokens > token_cap ||
            (uint64_t) opts.lm_prompt_tokens > token_cap ||
            (uint64_t) opts.lm_max_new_tokens > token_cap || enc_s_u64 > token_cap) {
            r.reason = "workload-too-large";
            return r;
        }
    }

    // ── Paths: same classification as Engine::create ────────────────────────
    EngineOptions paths;
    paths.models_dir          = opts.models_dir;
    paths.text_enc_model_path = opts.text_enc_model_path;
    paths.lm_model_path       = opts.lm_model_path;
    paths.dit_model_path      = opts.dit_model_path;
    paths.vae_model_path      = opts.vae_model_path;
    resolve_stage_paths(paths);
    if (paths.text_enc_model_path.empty() || paths.lm_model_path.empty() ||
        paths.dit_model_path.empty() || paths.vae_model_path.empty()) {
        r.reason = "invalid-arguments";  // a stage GGUF is missing from the request
        return r;
    }

    // ── Backends: same resolution + placement as Engine::create ────────────
    load_backends(opts.backends_dir);

    BackendRequest request;
    request.n_gpu_layers = opts.n_gpu_layers;
    request.backend      = opts.backend;
    request.lm_backend   = opts.lm_backend;
    request.n_threads    = opts.n_threads;
    request.verbose      = opts.verbose;
    AcestepBackends rb;
    if (!resolve_acestep_backends(request, rb)) {
        r.reason = "no-backend-device";
        return r;
    }

    // The VAE creates its own backend inside Vae::load with the same GPU
    // request (engine.cpp saves it into vae_opts; ACESTEP_VAE_GPU overrides).
    // Resolution shared with Vae::load via engine_backends.h, by construction.
    ggml_backend_t vae_backend = resolve_vae_backend(vae_backend_request(request));

    struct Cleanup {
        AcestepBackends * rb;
        ggml_backend_t *  vae_backend;
        TextEncModel * textenc = nullptr;
        CondModel *    cond    = nullptr;
        DetokModel *   detok   = nullptr;
        LMModel *      lm      = nullptr;
        DitModel *     dit     = nullptr;
        VaeModel *     vae     = nullptr;
        ~Cleanup() {
            if (textenc) textenc_model_free(textenc);
            if (cond)    cond_model_free(cond);
            if (detok)   detok_model_free(detok);
            if (lm)      lm_model_free(lm);
            if (dit)     dit_model_free(dit);
            if (vae)     vae_model_free(vae);
            if (*vae_backend) ggml_backend_free(*vae_backend);
            free_acestep_backends(*rb);
        }
    } guard{ &rb, &vae_backend };

    if (!vae_backend) {
        r.reason = "no-backend-device";
        return r;
    }

    if (!ggml_backend_get_device(rb.backend_cpu) || !ggml_backend_get_device(rb.backend)) {
        r.reason = "no-backend-device";
        return r;
    }
    std::vector<FitPool> pools;
    register_backend_pools(pools, { rb.backend_cpu, rb.backend, rb.enc, rb.detok, rb.lm, vae_backend });
    const size_t    primary      = fit_pool_of(pools, ggml_backend_get_device(rb.backend));
    const FitPool & primary_pool = pools[primary];

    r.device_name               = ggml_backend_name(rb.backend);
    r.device_is_cpu             = !rb.on_gpu;
    r.device_shares_host_memory = primary_pool.shares_host_memory;
    r.device_free_bytes         = primary_pool.free_bytes;
    r.device_total_bytes        = primary_pool.total_bytes;
    r.host_free_bytes           = pools[FIT_HOST_POOL].free_bytes;
    r.host_total_bytes          = pools[FIT_HOST_POOL].total_bytes;

    const bool keep_stages =
        opts.keep_stages == 1 || (opts.keep_stages == -1 && env_keep_stages());
    r.stages_resident = keep_stages;

    // ── Workload -> shapes (the derivation chain of engine.cpp generate) ────
    // duration -> LM code budget (lm_pipeline.cpp: duration*5 + 100 cap; the
    // LM may stop earlier at EOS, so the cap is the strict worst case)
    // -> 25 Hz latent frames -> DiT T (padded to patch_size) -> VAE PCM.
    DitConfig dit_cfg;
    {
        DitGGUF g;
        if (!dit_gguf_open(g, paths.dit_model_path)) {
            r.reason = "model-unreadable";
            return r;
        }
        dit_gguf_require_hexagon_weights(g, paths.dit_model_path, rb.backend, "acestep-fit");
        const bool ok = dit_gguf_read_config(g, dit_cfg);
        dit_gguf_close(g);
        if (!ok) {
            r.reason = "model-unreadable";
            return r;
        }
    }
    r.model_name = dit_cfg.model_name;
    r.is_turbo   = dit_cfg.is_turbo;

    const uint64_t max_codes_u64 =
        opts.lm_max_new_tokens > 0
            ? (uint64_t) opts.lm_max_new_tokens
            : sat_add(sat_u64_from_double(std::floor((double) opts.duration_seconds * CODE_FRAME_RATIO)), 100);
    const uint64_t latent_frames_u64 = sat_mul(max_codes_u64, CODE_FRAME_RATIO);
    const uint64_t patch             = (uint64_t) std::max(1, dit_cfg.patch_size);
    const uint64_t T_u64 = sat_mul(sat_add(latent_frames_u64, patch - 1) / patch, patch);
    if (T_u64 > (uint64_t) std::numeric_limits<int>::max() / 4 ||
        max_codes_u64 > (uint64_t) std::numeric_limits<int>::max()) {
        r.reason = "workload-too-large";
        return r;
    }
    const int max_codes     = (int) max_codes_u64;
    const int latent_frames = (int) latent_frames_u64;
    const int T             = (int) T_u64;

    const int S_text  = opts.text_tokens;
    const int S_lyric = opts.lyric_tokens;
    const int S_ref   = 1;  // text2music timbre input: one silence frame (generation_conditioning.h)
    const int enc_S   = S_lyric + 1 + S_text;  // [lyric | timbre token | text] (cond_ggml.cpp)

    // LM prompt: caption + lyrics + template overhead, prefilled into a KV
    // window capped at the engine's fixed max_seq_len.
    int lm_prompt = opts.lm_prompt_tokens > 0 ? opts.lm_prompt_tokens : (S_text + S_lyric + 64);
    lm_prompt     = std::min(lm_prompt, LM_MAX_SEQ_LEN);

    const float guidance = opts.guidance_scale > 0.0f ? opts.guidance_scale
                                                      : (dit_cfg.is_turbo ? 1.0f : 7.0f);
    const bool dit_cfg_on = guidance > 1.0f;
    const bool lm_cfg_on  = opts.lm_cfg_scale > 1.0f;

    // ── Stage loads (metadata-only) + graph pricing ─────────────────────────
    auto fail_unreadable = [&r]() { r.reason = "model-unreadable"; return r; };
    auto fail_measure    = [&r]() { r.reason = "measurement-failed"; return r; };

    // textenc + cond (the "encoders" phase; both resident at its peak).
    AcestepStageMeasure textenc_w{}, cond_w{}, detok_w{}, lm_w{}, dit_w{}, vae_w{};
    guard.textenc = textenc_model_load_metadata_only(paths.text_enc_model_path, rb.enc, opts.verbose, textenc_w);
    if (!guard.textenc) return fail_unreadable();
    size_t textenc_fwd = 0, textenc_lookup = 0;
    {
        std::vector<float> dummy;
        if (!textenc_model_forward(guard.textenc, nullptr, S_text, dummy, &textenc_fwd) ||
            !textenc_model_embed_lookup(guard.textenc, nullptr, S_lyric, dummy, &textenc_lookup)) {
            return fail_measure();
        }
    }

    guard.cond = cond_model_load_metadata_only(paths.dit_model_path, rb.enc, opts.verbose, cond_w);
    if (!guard.cond) return fail_unreadable();
    size_t cond_fwd = 0;
    {
        std::vector<float> dummy;
        int                dummy_s = 0;
        // Non-null timbre pointer selects the timbre path (its bytes are never
        // read in measure mode); text2music always feeds the silence frame.
        const float * timbre_tag = reinterpret_cast<const float *>(&dummy_s);
        if (!cond_model_forward(guard.cond, nullptr, S_text, nullptr, S_lyric,
                                timbre_tag, S_ref, dummy, &dummy_s, &cond_fwd)) {
            return fail_measure();
        }
    }

    // detok
    guard.detok = detok_model_load_metadata_only(paths.dit_model_path, rb.detok, opts.verbose, detok_w);
    if (!guard.detok) return fail_unreadable();
    size_t detok_graph = 0;
    if (detok_model_decode(guard.detok, nullptr, /*T_5Hz=*/1, nullptr, &detok_graph) < 0) {
        return fail_measure();
    }

    // LM: weights + KV + compact tied-head copies + the graphs of every phase.
    guard.lm = lm_model_load_metadata_only(paths.lm_model_path, rb.lm, LM_MAX_SEQ_LEN, opts.verbose,
                                           LM_N_KV_SETS, lm_w);
    if (!guard.lm) return fail_unreadable();
    const LMConfig & lm_cfg = lm_model_config(guard.lm);
    // Compact-head row ranges (lm_pipeline.cpp): Phase 1 projects the prefix
    // [0, AUDIO_CODE_BASE), Phase 2's batched CFG the suffix [TOKEN_IM_END, V).
    // One compact head is alive at a time (rebuilt on a range change), so the
    // projection takes the larger.
    uint64_t lm_head_bytes = 0;
    if (AUDIO_CODE_BASE < lm_cfg.vocab_size) {
        lm_head_bytes = std::max<uint64_t>(
            lm_head_bytes, lm_measure_partial_head_bytes(guard.lm, AUDIO_CODE_BASE));
    }
    if (lm_cfg_on && TOKEN_IM_END > 0 && TOKEN_IM_END < lm_cfg.vocab_size) {
        lm_head_bytes = std::max<uint64_t>(
            lm_head_bytes, lm_measure_partial_head_bytes(guard.lm, lm_cfg.vocab_size - TOKEN_IM_END));
    }
    // Graphs: Phase-1/2 prefill at the prompt length (full head is the larger
    // graph, so measure limit=0), the worst single-stream decode at the full
    // KV window, and -- where the real pipeline batches (flash attention +
    // 2 KV sets) -- the batched CFG decode step. One graph is cached at a
    // time (LMGraphCache), so the phase holds the max, not the sum.
    size_t lm_graph = 0;
    {
        size_t s = 0;
        if (!lm_model_measure_prefill(guard.lm, lm_prompt, /*logit_limit=*/0, s)) return fail_measure();
        lm_graph = std::max(lm_graph, s);
        if (!lm_model_measure_decode(guard.lm, LM_MAX_SEQ_LEN, /*logit_limit=*/0, s)) return fail_measure();
        lm_graph = std::max(lm_graph, s);
        if (AUDIO_CODE_BASE < lm_cfg.vocab_size) {
            if (!lm_model_measure_decode(guard.lm, LM_MAX_SEQ_LEN, AUDIO_CODE_BASE, s)) return fail_measure();
            lm_graph = std::max(lm_graph, s);
        }
        if (lm_cfg_on && lm_model_supports_batched_decode(guard.lm)) {
            if (!lm_model_measure_decode_batch(guard.lm, 2, LM_MAX_SEQ_LEN,
                                               TOKEN_IM_END < lm_cfg.vocab_size ? TOKEN_IM_END : 0, s)) {
                return fail_measure();
            }
            lm_graph = std::max(lm_graph, s);
        }
    }

    // DiT
    guard.dit = dit_model_load_metadata_only(paths.dit_model_path, rb.backend, opts.verbose, dit_w);
    if (!guard.dit) return fail_unreadable();
    size_t dit_graph = 0;
    {
        DitForwardInputs fin;
        fin.T     = T;
        fin.N     = 1;  // CFG runs the uncond pass on the same cached graph (dit_sample)
        fin.enc_S = enc_S;
        fin.H_enc = dit_model_config(guard.dit).enc_hidden_size;
        // dit_sample always passes both masks; non-nullness selects the shape.
        fin.sa_mask_sw = reinterpret_cast<const void *>(&fin);
        fin.ca_mask    = reinterpret_cast<const void *>(&fin);
        std::vector<float> dummy;
        if (!dit_model_forward(guard.dit, fin, dummy, &dit_graph)) return fail_measure();
    }

    // VAE: decoder always; encoder only for source/reference workloads
    // (ensure_vae(true) in prepare_audio_conditioning).
    guard.vae = vae_model_load_metadata_only(paths.vae_model_path, vae_backend,
                                             /*with_encoder=*/opts.with_source_audio, opts.verbose, vae_w);
    if (!guard.vae) return fail_unreadable();
    if (opts.with_source_audio && !vae_model_has_encoder(guard.vae)) {
        // Decoder-only GGUF cannot serve a source/reference workload's encode
        // phase; the real ensure_vae(true) degrades the same way, so project
        // without the encoder rather than failing.
    }
    size_t vae_dec_backend = 0, vae_dec_host_in = 0;
    if (!vae_model_measure_decode(guard.vae, T, vae_dec_backend, vae_dec_host_in)) return fail_measure();
    size_t vae_enc_backend = 0, vae_enc_host_in = 0;
    uint64_t source_frames = 0;
    if (opts.with_source_audio && vae_model_has_encoder(guard.vae)) {
        source_frames = sat_u64_from_double(std::ceil((double) opts.duration_seconds * SAMPLE_RATE));
        const int enc_frames = (int) std::min<uint64_t>(source_frames, VAE_AUDIO_CHUNK_FRAMES);
        if (!vae_model_measure_encode(guard.vae, enc_frames, vae_enc_backend, vae_enc_host_in)) {
            return fail_measure();
        }
    }

    // ── Per-stage rows ──────────────────────────────────────────────────────
    auto stage_row = [&](const char * name, ggml_backend_t backend, const AcestepStageMeasure & w,
                         uint64_t extra_weights, uint64_t state, uint64_t compute,
                         uint64_t host) {
        FitStageProjection s;
        s.name               = name;
        s.device_name        = ggml_backend_name(backend);
        s.on_gpu             = !backend_is_cpu_device(backend);
        s.weights_bytes      = sat_add(w.weights_alloc_bytes, extra_weights);
        s.weights_mmap_bytes = w.weights_mapped_bytes;
        s.state_bytes        = state;
        s.compute_bytes      = compute;
        s.host_bytes         = host;
        r.stages.push_back(std::move(s));
    };

    // Host-side workload buffers per phase, mirroring the std::vector slabs
    // generate() / dit_sample / vae_model_decode hold (see fit.h). Saturating
    // throughout: host_bytes feeds the verdict, so a wrap could flip a real
    // DOES-NOT-FIT into FITS.
    const uint64_t bytes_context_latents = f32_bytes(sat_mul((uint64_t) LATENT_CHANNELS, (uint64_t) latent_frames));
    const uint64_t bytes_enc_hidden      = f32_bytes(sat_mul((uint64_t) COND_HIDDEN, (uint64_t) enc_S));
    const uint64_t bytes_dit_context =
        f32_bytes(sat_mul((uint64_t) (dit_cfg.in_channels - dit_cfg.out_channels), (uint64_t) T));

    uint64_t lm_host = f32_bytes((uint64_t) lm_cfg.vocab_size);          // logits
    if (lm_cfg_on) lm_host = sat_mul(lm_host, 2);                        // cond + uncond
    lm_host = sat_add(lm_host, f16_bytes(sat_mul((uint64_t) LM_MAX_SEQ_LEN, (uint64_t) lm_prompt)));  // prefill mask

    uint64_t detok_host = sat_add(bytes_context_latents,                                  // context latents
                                  f32_bytes(sat_mul((uint64_t) max_codes, 6)));           // fsq_decoded

    uint64_t enc_host = bytes_context_latents;                                             // still held in state
    enc_host = sat_add(enc_host, f32_bytes(sat_mul((uint64_t) TEXTENC_HIDDEN, (uint64_t) S_text)));   // text_hidden
    enc_host = sat_add(enc_host, f32_bytes(sat_mul((uint64_t) TEXTENC_HIDDEN, (uint64_t) S_lyric)));  // lyric_embed
    enc_host = sat_add(enc_host, bytes_enc_hidden);                                        // cond output
    enc_host = sat_add(enc_host, bytes_dit_context);                                       // make_dit_context
    enc_host = sat_add(enc_host, f16_bytes(sat_mul((uint64_t) S_lyric, (uint64_t) S_lyric)));  // lyric slide mask

    // dit_sample's slabs (dit_ggml.cpp): masks, x_t, the [in_ch, T] input
    // staging, velocity, the DCW pair, and under CFG the null-hidden /
    // null-mask / f64 APG momentum / uncond velocity.
    const uint64_t S_patch = (uint64_t) (T / std::max(1, dit_cfg.patch_size));
    const uint64_t n_per   = sat_mul((uint64_t) dit_cfg.out_channels, (uint64_t) T);
    uint64_t dit_host = bytes_context_latents;                       // state.context_latents
    dit_host = sat_add(dit_host, bytes_enc_hidden);                  // conditioning.hidden
    dit_host = sat_add(dit_host, bytes_dit_context);                 // conditioning.context
    dit_host = sat_add(dit_host, f16_bytes(sat_mul(S_patch, S_patch)));                    // sa_mask
    dit_host = sat_add(dit_host, f16_bytes(sat_mul((uint64_t) enc_S, S_patch)));           // ca_mask
    dit_host = sat_add(dit_host, f32_bytes(n_per));                                        // xt
    dit_host = sat_add(dit_host, f32_bytes(sat_mul((uint64_t) dit_cfg.in_channels, (uint64_t) T)));  // input_buf
    dit_host = sat_add(dit_host, f32_bytes(n_per));                                        // vt
    dit_host = sat_add(dit_host, f32_bytes(sat_mul(n_per, 2)));                            // xt_before + denoised (DCW)
    if (dit_cfg_on) {
        dit_host = sat_add(dit_host, f32_bytes(sat_mul((uint64_t) enc_S,
                                                       (uint64_t) dit_model_config(guard.dit).enc_hidden_size)));
        dit_host = sat_add(dit_host, f16_bytes(sat_mul((uint64_t) enc_S, S_patch)));       // null_ca_mask
        dit_host = sat_add(dit_host, sat_mul(n_per, sizeof(double)));                      // apg_momentum
        dit_host = sat_add(dit_host, f32_bytes(n_per));                                    // vt_uncond
    }
    dit_host = sat_add(dit_host, f32_bytes(n_per));                                        // latent_out

    // VAE decode: the stitched full-length PCM plus one window's staging
    // (channel-major input copy in, planar + interleaved copies out).
    const uint64_t pcm_samples = sat_mul(sat_mul((uint64_t) T, (uint64_t) VAE_UPSAMPLE), 2);
    const int      dec_core    = vae_model_decode_window_frames(guard.vae);
    const uint64_t T_win       = (uint64_t) (T <= dec_core ? T : std::min(T, dec_core + 2 * 48));
    uint64_t vae_host = f32_bytes(n_per);                                                  // latent (still held)
    vae_host = sat_add(vae_host, f32_bytes(pcm_samples));                                  // pcm_out
    vae_host = sat_add(vae_host, f32_bytes(sat_mul(T_win, (uint64_t) LATENT_CHANNELS)));   // lin
    vae_host = sat_add(vae_host, f32_bytes(sat_mul(sat_mul(T_win, (uint64_t) VAE_UPSAMPLE), 4)));  // planar + pcm_win

    uint64_t vae_enc_host = 0;
    if (opts.with_source_audio) {
        // Source PCM + its engine-held copy + the encoded latent + one encode
        // window's channel-major staging.
        const uint64_t src_samples = sat_mul(source_frames, 2);
        vae_enc_host = sat_add(f32_bytes(src_samples), f32_bytes(src_samples));
        vae_enc_host = sat_add(vae_enc_host,
                               f32_bytes(sat_mul(sat_add(source_frames / VAE_UPSAMPLE, 1),
                                                 (uint64_t) LATENT_CHANNELS)));
        vae_enc_host = sat_add(vae_enc_host,
                               f32_bytes(sat_mul(std::min<uint64_t>(src_samples,
                                                                    sat_mul((uint64_t) VAE_AUDIO_CHUNK_FRAMES, 2)), 2)));
    }

    stage_row("textenc", rb.enc, textenc_w, 0, 0,
              std::max(textenc_fwd, textenc_lookup), 0);
    stage_row("cond", rb.enc, cond_w, 0, 0, cond_fwd, 0);
    stage_row("lm", rb.lm, lm_w, lm_head_bytes, lm_w.kv_bytes, lm_graph, lm_host);
    stage_row("detok", rb.detok, detok_w, 0, 0, detok_graph, detok_host);
    stage_row("dit", rb.backend, dit_w, 0, 0, dit_graph, dit_host);
    // On a GPU VAE backend the real sched stages the graph input on its CPU
    // (last) backend before copying it in; that portion is host RAM.
    stage_row("vae", vae_backend, vae_w, 0, 0,
              std::max<uint64_t>(vae_dec_backend, vae_enc_backend),
              sat_add(vae_host, std::max<uint64_t>(vae_dec_host_in, vae_enc_host_in)));

    // ── Phase peaks per pool ────────────────────────────────────────────────
    // Stage weight charges (weights+mmap+state on the stage's backend pool;
    // mmapped pages are file-backed but resident while the stage computes, so
    // they charge the host pool -- strict direction).
    auto stage_static = [&](ggml_backend_t backend, const AcestepStageMeasure & w,
                            uint64_t extra_weights, uint64_t state) {
        PoolCharge c = charge_backend(pools, backend, sat_add(sat_add(w.weights_alloc_bytes, extra_weights), state));
        c.add_host(w.weights_mapped_bytes);
        return c;
    };
    const PoolCharge textenc_static = stage_static(rb.enc, textenc_w, 0, 0);
    const PoolCharge cond_static    = stage_static(rb.enc, cond_w, 0, 0);
    const PoolCharge detok_static   = stage_static(rb.detok, detok_w, 0, 0);
    const PoolCharge lm_static      = stage_static(rb.lm, lm_w, lm_head_bytes, lm_w.kv_bytes);
    const PoolCharge dit_static     = stage_static(rb.backend, dit_w, 0, 0);
    const PoolCharge vae_static     = stage_static(vae_backend, vae_w, 0, 0);

    // Phase compute charges.
    const PoolCharge lm_compute      = charge_backend(pools, rb.lm, lm_graph);
    const PoolCharge detok_compute   = charge_backend(pools, rb.detok, detok_graph);
    const PoolCharge enc_compute     = charge_backend(pools, rb.enc,
                                                      std::max({ textenc_fwd, textenc_lookup, cond_fwd }));
    const PoolCharge dit_compute     = charge_backend(pools, rb.backend, dit_graph);
    PoolCharge vae_dec_compute = charge_backend(pools, vae_backend, vae_dec_backend);
    vae_dec_compute.add_host(vae_dec_host_in);
    PoolCharge vae_enc_compute = charge_backend(pools, vae_backend, vae_enc_backend);
    vae_enc_compute.add_host(vae_enc_host_in);

    PoolCharge peak(pools.size());
    if (keep_stages) {
        // Everything resident: all stage buffers, the two persistent graph
        // caches (DiT forward, LM decode), plus the largest ephemeral graph.
        PoolCharge resident(pools.size());
        resident.add(textenc_static).add(cond_static).add(detok_static)
                .add(lm_static).add(dit_static).add(vae_static)
                .add(dit_compute).add(lm_compute);
        PoolCharge ephemeral(pools.size());  // per-call graphs: encoders, detok, VAE windows
        ephemeral.max_with(enc_compute).max_with(detok_compute)
                 .max_with(vae_dec_compute).max_with(vae_enc_compute);
        PoolCharge host_only(pools.size());
        host_only.add_host(std::max({ lm_host, detok_host, enc_host,
                                      dit_host, vae_host, vae_enc_host }));
        peak = resident;
        peak.add(ephemeral).add(host_only);
    } else {
        // Lazy (default): one phase at a time; the text and cond encoders
        // overlap inside the conditioning phase (ensure_cond before
        // encode_prompt -- engine.cpp prepare_encoder_conditioning).
        auto phase = [](PoolCharge stat, PoolCharge comp, uint64_t host) {
            PoolCharge p = stat;
            p.add(comp).add_host(host);
            return p;
        };
        PoolCharge enc_static = textenc_static;  // both encoders resident together
        enc_static.add(cond_static);

        peak.max_with(phase(lm_static, lm_compute, lm_host));
        peak.max_with(phase(detok_static, detok_compute, detok_host));
        peak.max_with(phase(enc_static, enc_compute, enc_host));
        peak.max_with(phase(dit_static, dit_compute, dit_host));
        peak.max_with(phase(vae_static, vae_dec_compute, vae_host));
        if (opts.with_source_audio) {
            peak.max_with(phase(vae_static, vae_enc_compute, vae_enc_host));
        }
    }
    r.peak_device_bytes = peak.bytes[primary];
    r.peak_host_bytes   = peak.bytes[FIT_HOST_POOL];
    r.extra_devices     = extra_device_pools(pools, peak, primary);

    // ── Verdict ─────────────────────────────────────────────────────────────
    const std::vector<FitBudgetCheck> checks = fit_budget_checks(pools, peak, primary, opts.margin_bytes);
    const bool                        fits   = fit_checks_hold(checks);
    r.fits   = fits;
    r.status = fits ? FitStatus::Success : FitStatus::Failure;
    r.reason = fits ? "fits" : "does-not-fit";

    // ── Report ──────────────────────────────────────────────────────────────
    {
        std::string s;
        char line[320];
        std::snprintf(line, sizeof(line), "model:    %s (%s)\n",
                      r.model_name.empty() ? "acestep" : r.model_name.c_str(),
                      r.is_turbo ? "turbo" : "base/sft");
        s += line;
        std::snprintf(line, sizeof(line), "device:   %s (%s), free %s / total %s%s\n",
                      r.device_name.c_str(), r.device_is_cpu ? "CPU" : "GPU",
                      fmt_mib(r.device_free_bytes).c_str(), fmt_mib(r.device_total_bytes).c_str(),
                      r.device_shares_host_memory ? " (shares host RAM)" : "");
        s += line;
        if (primary != FIT_HOST_POOL && fit_shared_budget_pool(pools, primary) == FIT_HOST_POOL) {
            std::snprintf(line, sizeof(line), "host:     free %s / total %s\n",
                          fmt_mib(r.host_free_bytes).c_str(), fmt_mib(r.host_total_bytes).c_str());
            s += line;
        }
        s += format_extra_device_lines(r.extra_devices);
        std::snprintf(line, sizeof(line),
                      "workload: %.1f s -> <=%d LM codes, %d latent frames, DiT T=%d, enc_S=%d%s%s\n",
                      (double) opts.duration_seconds, max_codes, latent_frames, T, enc_S,
                      dit_cfg_on ? ", DiT CFG" : "", opts.with_source_audio ? ", source audio" : "");
        s += line;
        std::snprintf(line, sizeof(line), "stages (%s):\n",
                      keep_stages ? "all resident, ACESTEP_KEEP_STAGES" : "time-shared, low-mem default");
        s += line;
        for (const FitStageProjection & st : r.stages) {
            std::snprintf(line, sizeof(line),
                          "  %-8s %-8s weights %12s%s%s%s%s  compute %12s%s\n",
                          st.name.c_str(), st.device_name.c_str(),
                          fmt_mib(st.weights_bytes).c_str(),
                          st.weights_mmap_bytes ? " (+" : "",
                          st.weights_mmap_bytes ? fmt_mib(st.weights_mmap_bytes).c_str() : "",
                          st.weights_mmap_bytes ? " mmap)" : "",
                          st.state_bytes ? (" kv " + fmt_mib(st.state_bytes)).c_str() : "",
                          fmt_mib(st.compute_bytes).c_str(),
                          st.host_bytes ? (" host " + fmt_mib(st.host_bytes)).c_str() : "");
            s += line;
        }
        std::snprintf(line, sizeof(line), "peak:     device %s, host %s%s, margin %s\n",
                      fmt_mib(r.peak_device_bytes).c_str(), fmt_mib(r.peak_host_bytes).c_str(),
                      format_extra_device_peaks(r.extra_devices).c_str(), fmt_mib(opts.margin_bytes).c_str());
        s += line;
        if (r.fits) {
            const uint64_t headroom = fit_checks_headroom(checks);
            std::snprintf(line, sizeof(line), "verdict: FITS (headroom %s)\n", fmt_mib(headroom).c_str());
        } else {
            std::snprintf(line, sizeof(line), "verdict: DOES NOT FIT\n");
        }
        s += line;
        r.report = std::move(s);
    }

    return r;
}

FitResult fit_params(const FitOptions & opts) {
    try {
        return fit_params_unguarded(opts);
    } catch (const std::exception & e) {
        FitResult r;
        r.reason = "measurement-failed";
        r.report = e.what();
        return r;
    }
}

}  // namespace tts_cpp::acestep
