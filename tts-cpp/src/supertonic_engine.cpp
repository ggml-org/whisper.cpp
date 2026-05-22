#define TTS_CPP_BUILD
#include "tts-cpp/supertonic/engine.h"

#include "backend_selection.h"
#include "supertonic_chunker.h"
#include "supertonic_internal.h"
#include "npy.h"

#ifdef GGML_USE_VULKAN
// QVAC-18605 — needed for `ggml_backend_vk_get_device_description`
// in the `backend_name()` annotator (Vulkan-only).
#include "ggml-vulkan.h"
#endif

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace tts_cpp::supertonic {

using namespace detail;

namespace {

std::string supertonic_setup_hint(const std::string & path) {
    return "Supertonic GGUF not found: " + path + "\n"
           "Create the local model first, for example:\n"
           "  bash scripts/setup-supertonic2.sh\n"
           "or for the English-only bundle:\n"
           "  bash scripts/setup-supertonic2.sh --arch supertonic\n"
           "Model GGUFs live under models/ and are intentionally ignored by git.";
}

std::vector<float> read_tensor_f32(ggml_tensor * t) {
    std::vector<float> out((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

// NumPy RandomState-compatible MT19937 + standard_normal().  This matches the
// legacy np.random.seed(seed); np.random.randn(...) sequence used by the ONNX
// reference dumper.  std::normal_distribution is intentionally not used here:
// its transform is implementation-defined and produced audibly different
// Supertonic samples for the same seed.
class numpy_random_state {
public:
    explicit numpy_random_state(uint32_t seed) {
        mt_[0] = seed;
        for (int i = 1; i < N; ++i) {
            mt_[i] = 1812433253U * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + (uint32_t)i;
        }
        index_ = N;
    }

    float standard_normal() {
        if (has_gauss_) {
            has_gauss_ = false;
            return (float) gauss_;
        }
        double x1 = 0.0, x2 = 0.0, r2 = 0.0;
        do {
            x1 = 2.0 * uniform_double() - 1.0;
            x2 = 2.0 * uniform_double() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        const double f = std::sqrt(-2.0 * std::log(r2) / r2);
        gauss_ = x1 * f;
        has_gauss_ = true;
        return (float)(x2 * f);
    }

private:
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfU;
    static constexpr uint32_t UPPER_MASK = 0x80000000U;
    static constexpr uint32_t LOWER_MASK = 0x7fffffffU;

    uint32_t mt_[N]{};
    int index_ = N + 1;
    bool has_gauss_ = false;
    double gauss_ = 0.0;

    uint32_t uint32() {
        static const uint32_t mag01[2] = {0x0U, MATRIX_A};
        if (index_ >= N) {
            int kk = 0;
            for (; kk < N - M; ++kk) {
                uint32_t y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
                mt_[kk] = mt_[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            for (; kk < N - 1; ++kk) {
                uint32_t y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
                mt_[kk] = mt_[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            uint32_t y = (mt_[N - 1] & UPPER_MASK) | (mt_[0] & LOWER_MASK);
            mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
            index_ = 0;
        }
        uint32_t y = mt_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }

    double uniform_double() {
        const uint32_t a = uint32() >> 5;
        const uint32_t b = uint32() >> 6;
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }
};

// Heuristic: does this chunk end at a natural sentence terminator?
// Used by streaming to decide whether to skip the auto-appended period
// (continuation chunks) or keep it (complete-sentence chunks).  Commas
// and other clause punctuation are NOT counted here — chunks ending in
// a comma still want is_continuation=true so the model hears them as
// a continuation, not a mini-sentence.
//
// Trims trailing whitespace, then decodes the final UTF-8 code point
// and delegates to the chunker's `is_sentence_end_cp` so the
// terminator table is defined in exactly one place (see
// supertonic_chunker.cpp).
bool chunk_ends_with_sentence_term(const std::string & s) {
    size_t i = s.size();
    while (i > 0 && (s[i - 1] == ' ' || s[i - 1] == '\t' ||
                     s[i - 1] == '\n' || s[i - 1] == '\r')) --i;
    if (i == 0) return false;
    // Walk back to the leading byte of the final UTF-8 sequence.
    size_t pos = i - 1;
    while (pos > 0 && ((uint8_t) s[pos] & 0xC0) == 0x80) --pos;
    const size_t bytes = i - pos;
    uint32_t cp = 0;
    if      (bytes == 1) cp = (uint8_t) s[pos];
    else if (bytes == 2) cp = ((s[pos] & 0x1F) << 6) | (s[pos + 1] & 0x3F);
    else if (bytes == 3) cp = ((s[pos] & 0x0F) << 12) |
                              ((s[pos + 1] & 0x3F) << 6) |
                              (s[pos + 2] & 0x3F);
    else if (bytes == 4) cp = ((s[pos] & 0x07) << 18) |
                              ((s[pos + 1] & 0x3F) << 12) |
                              ((s[pos + 2] & 0x3F) << 6) |
                              (s[pos + 3] & 0x3F);
    return detail::is_sentence_end_cp(cp);
}

} // namespace

struct Engine::Impl {
    EngineOptions    opts;
    supertonic_model model;
    std::atomic<bool> cancel_flag{false};
    // QVAC-18605 round 7 — voice ttl/dp host cache.  Populated
    // lazily on first `synthesize()` call per voice; subsequent
    // calls hit the cache and skip the GPU→host download (2 sync
    // points per call eliminated on Vulkan / OpenCL).  See the
    // contract on `voice_host_cache` in supertonic_internal.h.
    voice_host_cache voices_host;

    explicit Impl(const EngineOptions & o)
        : opts(o) {
        if (opts.model_gguf_path.empty()) {
            throw std::runtime_error("Supertonic Engine: model_gguf_path is required");
        }
        if (!std::filesystem::exists(opts.model_gguf_path)) {
            throw std::runtime_error(supertonic_setup_hint(opts.model_gguf_path));
        }
        // Wire backends_dir + opencl_cache_dir BEFORE any backend
        // init. First-Engine-wins across the whole process; second
        // and later Engines reuse the already-loaded registry. See
        // backend_selection.cpp.
        if (!opts.backends_dir.empty()) {
            ::tts_cpp::detail::set_backends_directory(opts.backends_dir);
        }
        if (!opts.opencl_cache_dir.empty()) {
            ::tts_cpp::detail::set_opencl_cache_dir(opts.opencl_cache_dir);
        }

        // Map the public Precision enum onto the internal one (separate
        // declaration so the engine header doesn't pull in internal.h).
        supertonic_precision internal_precision = supertonic_precision::F32;
        switch (opts.precision) {
            case Precision::F32:  internal_precision = supertonic_precision::F32;  break;
            case Precision::F16:  internal_precision = supertonic_precision::F16;  break;
            case Precision::Q8_0: internal_precision = supertonic_precision::Q8_0; break;
        }
        // QVAC-18605 round 7 — apply Vulkan env-var overrides
        // BEFORE `load_supertonic_gguf` (which calls
        // `init_supertonic_backend`).  ggml-vulkan reads its
        // GGML_VK_* env vars at backend init, so the overrides
        // need to land in the environment before that point.
        // Throws on any key without `GGML_VK_` prefix (operator-
        // config typo guard); the throw propagates up to the
        // caller (no model loaded yet, no cleanup needed).
        apply_vulkan_env_overrides(opts.vulkan_env_overrides);
        if (!load_supertonic_gguf(opts.model_gguf_path, model,
                                  opts.n_gpu_layers, /*verbose=*/false,
                                  opts.f16_weights, internal_precision,
                                  opts.vulkan_device,
                                  opts.f16_weights_deny_list)) {
            throw std::runtime_error("Supertonic Engine: failed to load GGUF: " +
                                     opts.model_gguf_path);
        }
        try {
            supertonic_set_n_threads(model, opts.n_threads);

            // F16 K/V attention dispatch: auto-enable on GPU backends,
            // disable on CPU; user can override either way.  Captured
            // into the model so supertonic_op_dispatch_scope picks it
            // up on every synthesize() call.  See model.use_f16_attn
            // in supertonic_internal.h.
            //
            // QVAC-18605 — auto-policy is now backend-capability-gated.
            // Probes `ggml_backend_supports_op` for a Supertonic-
            // shaped F16-K/V flash_attn graph node before flipping
            // the flag.  A backend that compiles `flash_attn_ext`
            // but rejects the F16 K/V variant for our shape (head_dim
            // = 64, n_heads = 4) keeps the F32 path — slower but
            // guaranteed to not crash at first synth call.  Manual
            // override via `--f16-attn 1` still forces dispatch
            // (useful for debug-shim backends).
            if (opts.f16_attn < 0) {
                model.use_f16_attn = !model.backend_is_cpu &&
                                     supertonic_backend_supports_f16_kv_flash_attn(model.backend);
            } else {
                model.use_f16_attn = opts.f16_attn != 0;
            }

            // QVAC-18605 round 4 — multi-dtype K/V dispatch resolution.
            //
            // Layered ON TOP of the round-1 `use_f16_attn` boolean:
            // when `opts.kv_attn_type == -1` (the default), the
            // resolver falls back to the boolean's value, so every
            // existing operator config sees zero behaviour change.
            //
            // When the operator opts in to a non-default dtype, the
            // resolved enum drives the vector-estimator dispatch
            // and the boolean is updated to mirror the F16 case
            // (so any external code still keying on the boolean
            // — currently none in tree but kept for forward-compat
            // — stays consistent).  Out-of-range opts.kv_attn_type
            // throws inside the resolver; we let the throw
            // propagate up to the Engine ctor (which already wraps
            // the body in try/catch and frees the model).
            //
            // Probes are advisory: an explicit BF16 / Q8_0 request
            // on an adapter that doesn't support it falls back to
            // F32 — same advisory-probe pattern as the round-1
            // F16 auto-policy fallback above.
            //
            // PR #18 reviewer (Omar) follow-up: the silent
            // fallback was masking operator surprise — someone
            // pinning `--kv-attn-type bf16` in their production
            // config on a mixed fleet (some adapters support
            // BF16 K/V, some don't) would silently see F32 on
            // the unsupported subset.  The resolver's
            // `out_was_downgraded` out-param surfaces the
            // explicit-request + missing-probe case so we can
            // emit a one-line stderr warning (auto path stays
            // silent — the operator didn't ask for a specific
            // dtype, so there's nothing to surprise them with).
            bool kv_dtype_downgraded = false;
            model.kv_attn_type = resolve_kv_attn_type(
                opts.kv_attn_type,
                model.use_f16_attn,
                supertonic_backend_supports_f16_kv_flash_attn(model.backend),
                supertonic_backend_supports_bf16_kv_flash_attn(model.backend),
                supertonic_backend_supports_q8_0_kv_flash_attn(model.backend),
                &kv_dtype_downgraded);
            if (kv_dtype_downgraded) {
                static const char * const kv_label[] = {
                    "f32", "f16", "bf16", "q8_0"
                };
                std::fprintf(stderr,
                    "supertonic: warning: requested --kv-attn-type %s but the "
                    "resolved backend's flash-attn probe rejected it; falling "
                    "back to f32 (set --kv-attn-type auto to silence)\n",
                    (opts.kv_attn_type >= 0 && opts.kv_attn_type <= 3)
                        ? kv_label[opts.kv_attn_type] : "?");
            }
            // Keep the boolean consistent with the resolved enum.
            // No-op for the default `kv_attn_type == -1` path (the
            // resolver already mirrors the boolean).  Becomes a
            // no-op for explicit `--kv-attn-type 1` too.
            model.use_f16_attn = (model.kv_attn_type == kv_attn_dtype::f16);

            // Validate voice up front so we throw at construction
            // rather than mid-synthesize().
            const std::string voice = opts.voice.empty()
                ? model.hparams.default_voice
                : opts.voice;
            if (model.voices.find(voice) == model.voices.end()) {
                throw std::runtime_error("Supertonic Engine: unknown voice: " + voice);
            }

            // QVAC-18605 follow-up — opt-in first-synth pre-warm.
            // Skipped on CPU (no shader-compile cost to amortise)
            // and on empty `prewarm_text` (the caller didn't ask).
            // On Vulkan / OpenCL this runs one throwaway synth to
            // force every per-stage graph cache to populate and
            // every shader pipeline to compile, so the first
            // operator-visible `synthesize()` call hits steady-
            // state latency instead of paying the ~hundreds-of-ms
            // cold-start hit chatterbox PROGRESS.md measured on
            // Adreno + RADV.
            if (!opts.prewarm_text.empty() && !model.backend_is_cpu) {
                synthesize(opts.prewarm_text);  // discard result
            }
        } catch (...) {
            free_supertonic_model(model);
            throw;
        }
    }

    ~Impl() {
        free_supertonic_model(model);
    }

    Impl(const Impl &)             = delete;
    Impl & operator=(const Impl &) = delete;

    // Single-chunk synthesis worker.  Runs the full Supertonic pipeline
    // (preprocess → duration → noise → text encoder → vector estimator
    // CFM loop → vocoder) on `text` with the given seed.  When
    // `is_continuation` is true the preprocess skips the auto-appended
    // terminal period — used by streaming for mid-utterance chunks so
    // the model isn't told "this is a complete sentence" when it isn't.
    SynthesisResult run_single_chunk(const std::string & text, int seed,
                                     bool is_continuation = false) {
        const std::string voice = opts.voice.empty()
            ? model.hparams.default_voice
            : opts.voice;
        const int   steps = opts.steps > 0 ? opts.steps : model.hparams.default_steps;
        const float speed = opts.speed > 0.0f ? opts.speed : model.hparams.default_speed;
        if (steps <= 0) throw std::runtime_error("Supertonic Engine: steps must be positive");
        if (speed <= 0.0f) throw std::runtime_error("Supertonic Engine: speed must be positive");

        auto vit = model.voices.find(voice);
        if (vit == model.voices.end()) {
            // Re-validated here in case opts.voice was hot-swapped after
            // construction (not currently supported but guard anyway).
            throw std::runtime_error("Supertonic Engine: unknown voice: " + voice);
        }
        // QVAC-18605 round 7 — `voices_host.get_or_load` returns
        // a stable reference into the per-engine cache.  First
        // call per voice does the 2 GPU→host downloads + caches;
        // subsequent calls return the cached entry without
        // touching the backend.  Pointers + size below are valid
        // for the duration of this `synthesize()` call (cache is
        // never `clear()`ed during synthesis).
        const auto & voice_entry = voices_host.get_or_load(voice, vit->second.ttl, vit->second.dp);
        const float * style_ttl  = voice_entry.ttl.data();
        const float * style_dp   = voice_entry.dp.data();

        std::vector<int32_t> text_ids_i32;
        std::string normalized;
        std::string error;
        if (!supertonic_text_to_ids(model, text, opts.language, text_ids_i32,
                                    &normalized, &error, is_continuation)) {
            throw std::runtime_error("Supertonic Engine: text preprocessing failed: " + error);
        }
        std::vector<int64_t> text_ids(text_ids_i32.begin(), text_ids_i32.end());

        if (cancel_flag.load(std::memory_order_acquire)) {
            throw std::runtime_error("Supertonic Engine: cancelled before duration");
        }

        float duration_raw = 0.0f;
        if (!supertonic_duration_forward_ggml(model, text_ids.data(), (int) text_ids.size(),
                                              style_dp, duration_raw, &error)) {
            throw std::runtime_error("Supertonic Engine: duration failed: " + error);
        }
        const float duration_s  = duration_raw / speed;
        const int   sample_rate = model.hparams.sample_rate;
        const int   chunk = model.hparams.base_chunk_size *
                            model.hparams.ttl_chunk_compress_factor;
        int wav_len = (int) (duration_s * sample_rate);
        int latent_len = std::max(1, (wav_len + chunk - 1) / chunk);

        std::vector<float> latent;
        if (!opts.noise_npy_path.empty()) {
            npy_array noise = npy_load(opts.noise_npy_path);
            if (noise.dtype != "<f4" || noise.shape.size() != 3 || noise.shape[0] != 1 ||
                noise.shape[1] != model.hparams.latent_channels) {
                throw std::runtime_error("Supertonic Engine: noise npy must be float32 [1, latent_channels, L]");
            }
            latent_len = (int) noise.shape[2];
            wav_len = latent_len * chunk;
            latent.resize(noise.n_elements());
            std::memcpy(latent.data(), npy_as_f32(noise), latent.size() * sizeof(float));
        } else {
            numpy_random_state rng((uint32_t) seed);
            latent.assign((size_t) model.hparams.latent_channels * latent_len, 0.0f);
            for (float & v : latent) v = rng.standard_normal();
        }

        if (cancel_flag.load(std::memory_order_acquire)) {
            throw std::runtime_error("Supertonic Engine: cancelled before text encoder");
        }

        std::vector<float> text_emb;
        if (!supertonic_text_encoder_forward_ggml(model, text_ids.data(), (int) text_ids.size(),
                                                  style_ttl, text_emb, &error)) {
            throw std::runtime_error("Supertonic Engine: text encoder failed: " + error);
        }

        std::vector<float> latent_mask((size_t) latent_len, 1.0f);

        // Master's CFM loop unrolling (Phase A1+A2) replaced the
        // round-7 per-step `supertonic_vector_step_ggml` loop with
        // a single `supertonic_vector_loop_ggml` call below.  The
        // per-step cancellation hook from round 7 collapses into
        // this single pre-synth check (cancel granularity moves
        // from per-step to per-synth on the GPU path; the CPU
        // path's per-step fallback inside `supertonic_vector_loop_ggml`
        // retains finer cancellation if needed).
        if (cancel_flag.load(std::memory_order_acquire)) {
            throw std::runtime_error("Supertonic Engine: cancelled before vector estimator");
        }
        // Phase A1+A2: run all CFM steps as ONE ggml graph on non-CPU
        // backends.  Latent flows step-to-step in GPU memory; on CPU this
        // falls back to a per-step loop over `supertonic_vector_step_ggml`.
        // Override via SUPERTONIC_DISABLE_LOOP_GRAPH=1.
        // NOTE: cancellation granularity is now per-synth on the GPU path
        // (worst-case cancel latency = whole CFM loop).  CPU keeps per-step
        // cancellation via the fallback.
        std::vector<float> final_latent;
        if (!supertonic_vector_loop_ggml(model, latent.data(), latent_len,
                                          text_emb.data(), (int) text_ids.size(),
                                          style_ttl, latent_mask.data(),
                                          steps, final_latent, &error)) {
            throw std::runtime_error("Supertonic Engine: vector estimator failed: " + error);
        }
        latent = std::move(final_latent);

        if (cancel_flag.load(std::memory_order_acquire)) {
            throw std::runtime_error("Supertonic Engine: cancelled before vocoder");
        }

        std::vector<float> wav_full;
        if (!supertonic_vocoder_forward_ggml(model, latent.data(), latent_len, wav_full, &error)) {
            throw std::runtime_error("Supertonic Engine: vocoder failed: " + error);
        }

        SynthesisResult result;
        result.sample_rate = sample_rate;
        result.duration_s  = duration_s;
        result.pcm.assign(wav_full.begin(),
                          wav_full.begin() + std::min((size_t) wav_len, wav_full.size()));
        return result;
    }

    SynthesisResult synthesize(const std::string & text) {
        if (text.empty()) {
            throw std::runtime_error("Supertonic Engine: text is empty");
        }
        return run_single_chunk(text, opts.seed);
    }

    // Streaming path: chunk text via the multilingual splitter, run the
    // full per-chunk pipeline, apply an anti-click raised-cosine fade
    // across inter-chunk seams, invoke `on_chunk` synchronously, and
    // accumulate the full PCM in the returned result (callback is an
    // *addition*, not a replacement — matches Chatterbox semantics).
    SynthesisResult synthesize_streaming(const std::string & text,
                                         const StreamCallback & on_chunk) {
        if (text.empty()) {
            throw std::runtime_error("Supertonic Engine: text is empty");
        }

        std::vector<std::string> chunks = detail::split_for_streaming(
            text,
            opts.stream_chunk_tokens,
            opts.stream_first_chunk_tokens,
            opts.stream_chunk_tolerance_pct,
            opts.stream_min_chunk_tokens);

        if (chunks.empty()) {
            throw std::runtime_error("Supertonic Engine: chunker produced no chunks");
        }

        // Optional chunk-boundary trace for debugging the multilingual
        // splitter.  Off by default; opt-in via env var so production
        // synthesis isn't slowed by stderr writes.
        if (const char * env = std::getenv("SUPERTONIC_LOG_CHUNKS"); env && env[0] == '1') {
            for (size_t i = 0; i < chunks.size(); ++i) {
                std::fprintf(stderr, "chunk[%zu] (%zu bytes): %s\n",
                             i, chunks[i].size(), chunks[i].c_str());
            }
        }

        SynthesisResult full;
        full.duration_s = 0.0f;

        const int n_chunks = (int) chunks.size();
        for (int k = 0; k < n_chunks; ++k) {
            if (cancel_flag.load(std::memory_order_acquire)) {
                throw std::runtime_error(
                    "Supertonic Engine: cancelled during streaming chunk "
                    + std::to_string(k));
            }

            // Use opts.seed for every chunk.  Each chunk has a different
            // predicted latent_len (driven by its own text and duration
            // model), so the RNG produces different-length noise tensors
            // for each chunk even with the same seed — there's no risk
            // of identical starting noise across chunks.  An earlier
            // version perturbed the seed per chunk (opts.seed + k) as
            // a defensive measure, but that landed some chunks on
            // nearby seeds where the model produces phantom phoneme
            // artifacts ("park.K" tail).  Keeping the user's chosen
            // seed across chunks gives consistent, controllable output.
            //
            // is_continuation: chunks that DON'T end on a natural
            // sentence terminator (.?! and the CJK / Devanagari / Urdu
            // equivalents) need preprocess to skip the auto-appended
            // period.  Otherwise the model hears the stub as a complete
            // sentence with falling intonation + trailing artifacts —
            // the failure mode that originally restricted us to
            // sentence-only chunking.  With the flag, mid-clause /
            // mid-word chunk endings flow through with their natural
            // (un-punctuated) tail so the model treats them as a
            // continuation.
            const bool is_continuation = !chunk_ends_with_sentence_term(chunks[k]);
            if (const char * env = std::getenv("SUPERTONIC_LOG_CHUNKS");
                env && env[0] == '1') {
                std::fprintf(stderr, "chunk[%d] is_continuation=%d\n",
                             k, (int) is_continuation);
            }
            SynthesisResult chunk_res = run_single_chunk(chunks[k], opts.seed,
                                                         is_continuation);

            // Anti-click raised-cosine fade across inter-chunk seams.
            // Without HiFT cache continuity (Supertonic runs each chunk
            // as a fresh independent pipeline), plain concatenation can
            // produce a faint click at the boundary.  ~10 ms is enough
            // to hide the click without audibly attenuating speech.
            // Applied to the start of every non-first chunk and the end
            // of every non-last chunk.  The very-first chunk start and
            // very-last chunk end are left untouched so the streamed
            // output is acoustically equivalent to the batch output at
            // those endpoints.
            const int    sr      = chunk_res.sample_rate;
            const size_t fade_n  = std::min<size_t>(
                                       (size_t)(sr * 10 / 1000),
                                       chunk_res.pcm.size() / 2);
            const bool   is_first = (k == 0);
            const bool   is_last  = (k == n_chunks - 1);

            if (!is_first && fade_n > 0) {
                for (size_t i = 0; i < fade_n; ++i) {
                    const float t = (float) i / (float) fade_n;
                    const float w = 0.5f * (1.0f - std::cos((float) M_PI * t));
                    chunk_res.pcm[i] *= w;
                }
            }
            if (!is_last && fade_n > 0) {
                const size_t n = chunk_res.pcm.size();
                for (size_t i = 0; i < fade_n; ++i) {
                    const float t = (float) i / (float) fade_n;
                    const float w = 0.5f * (1.0f - std::cos((float) M_PI * t));
                    chunk_res.pcm[n - 1 - i] *= w;
                }
            }

            // Fire callback before accumulating, so the consumer sees
            // the same buffer it would receive in pure-streaming mode.
            on_chunk(chunk_res.pcm.data(), chunk_res.pcm.size(), k, is_last);

            full.pcm.insert(full.pcm.end(), chunk_res.pcm.begin(), chunk_res.pcm.end());
            full.duration_s  += chunk_res.duration_s;
            full.sample_rate  = chunk_res.sample_rate;
        }

        return full;
    }

    std::string backend_name() const {
        if (!model.backend) return "(unknown)";
        const char * name = ggml_backend_name(model.backend);
        std::string out = name ? std::string(name) : "(unknown)";
        // QVAC-18605 — append device description when Vulkan is the
        // resolved backend.  Mirrors chatterbox's bench output so a
        // log line like "backend: Vulkan (device 0: NVIDIA RTX 5090)"
        // is unambiguous when triaging multi-GPU machines.
#ifdef GGML_USE_VULKAN
        if (model.backend_is_vk) {
            char desc[256] = {0};
            ggml_backend_vk_get_device_description(opts.vulkan_device < 0 ? 0 : opts.vulkan_device,
                                                   desc, sizeof(desc) - 1);
            if (desc[0]) {
                out += " (device " + std::to_string(opts.vulkan_device < 0 ? 0 : opts.vulkan_device) +
                       ": " + desc + ")";
            }
        }
#endif
        return out;
    }
};

Engine::Engine(const EngineOptions & opts)
    : pimpl_(std::make_unique<Impl>(opts)) {}

Engine::~Engine() = default;

Engine::Engine(Engine &&) noexcept            = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

SynthesisResult Engine::synthesize(const std::string & text) {
    return pimpl_->synthesize(text);
}

SynthesisResult Engine::synthesize(const std::string & text,
                                   const StreamCallback & on_chunk) {
    // Fall through to the batch path when streaming is disabled or no
    // callback is wired up.  Both conditions match the Chatterbox
    // semantics — callers can pass a no-op callback safely.
    if (!on_chunk || pimpl_->opts.stream_chunk_tokens <= 0) {
        return pimpl_->synthesize(text);
    }
    return pimpl_->synthesize_streaming(text, on_chunk);
}

void Engine::cancel() {
    pimpl_->cancel_flag.store(true, std::memory_order_release);
}

// QVAC-18605 follow-up — explicit first-synth pre-warm.
// Forwards to the in-place `synthesize` and discards the PCM,
// gated on the same `backend_is_cpu` short-circuit the auto-
// invoked path at the end of `Impl::Impl` uses.  See the
// declaration in `tts-cpp/supertonic/engine.h` for the full
// rationale; the implementation here intentionally keeps the
// no-op CPU fast path so callers don't have to branch on
// `backend_device()` themselves.
void Engine::warm_up(const std::string & text) {
    if (text.empty()) return;
    if (pimpl_->model.backend_is_cpu) return;
    pimpl_->synthesize(text);  // discard result
}

const EngineOptions & Engine::options() const {
    return pimpl_->opts;
}

std::string Engine::backend_name() const {
    return pimpl_->backend_name();
}

BackendDevice Engine::backend_device() const {
    ggml_backend_t b = pimpl_ ? pimpl_->model.backend : nullptr;
    if (!b) return BackendDevice::CPU;
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    if (!dev) return BackendDevice::CPU;
    return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU
               ? BackendDevice::CPU
               : BackendDevice::GPU;
}

// Convenience one-shot wrapper.  Pays the full GGUF load + free per
// call; use Engine directly for repeated synthesis.
SynthesisResult synthesize(const EngineOptions & opts, const std::string & text) {
    Engine engine(opts);
    return engine.synthesize(text);
}

} // namespace tts_cpp::supertonic
