#include "tts-cpp/chatterbox/engine.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "chatterbox_t3_internal.h"
#include "gpt2_bpe.h"
#include "mtl_tokenizer.h"
#include "npy.h"
#include "t3_mtl.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"
#include "voice_encoder.h"
#include "voice_features.h"

namespace tts_cpp::chatterbox {

using namespace detail;

namespace {

int resolve_thread_count(int requested) {
    if (requested > 0) return requested;
    const int hw = (int) std::thread::hardware_concurrency();
    return hw > 0 ? std::min(hw, 4) : 4;
}

void wait_for_preload(std::thread & t) {
    if (t.joinable()) t.join();
}

} // namespace

struct Engine::Impl {
    EngineOptions opts;

    chatterbox_model     model{};
    ggml_gallocr_t       allocr = nullptr;
    std::thread          s3gen_preload_thread;

    // Baked voice-conditioning state.  Populated at construction when
    // `reference_audio` or `voice_dir` is set, then reused by every
    // synthesize() call so we never re-run VoiceEncoder / CAMPPlus /
    // S3TokenizerV2 / mel extraction more than once.
    bool                 voice_overridden = false;
    std::vector<float>   s3gen_prompt_feat;
    int                  s3gen_prompt_feat_rows = 0;
    std::vector<float>   s3gen_embedding;
    std::vector<int32_t> s3gen_prompt_token;

    // Tokenizer cache.  Exactly one of bpe / mtl_tok is populated based
    // on the loaded GGUF's variant metadata; the other stays default-
    // constructed.  Constructed once in the ctor and reused per
    // synthesize() call (gpt2_bpe rebuilds amortise poorly on Turbo's
    // 64K vocab + ~50K merges; mtl_tokenizer JSON parse is heavier
    // still).
    gpt2_bpe             bpe;
    mtl_tokenizer        mtl_tok;

    std::atomic<bool>    cancel_flag{false};

    explicit Impl(const EngineOptions & o)
        : opts(o) {
        if (opts.t3_gguf_path.empty()) {
            throw std::runtime_error("Engine: t3_gguf_path is required");
        }
        if (opts.s3gen_gguf_path.empty()) {
            throw std::runtime_error("Engine: s3gen_gguf_path is required");
        }
        if (!std::filesystem::exists(opts.t3_gguf_path)) {
            throw std::runtime_error("Engine: T3 GGUF not found: " + opts.t3_gguf_path);
        }
        if (!std::filesystem::exists(opts.s3gen_gguf_path)) {
            throw std::runtime_error("Engine: S3Gen GGUF not found: " + opts.s3gen_gguf_path);
        }
        if (!opts.reference_audio.empty() &&
            !std::filesystem::exists(opts.reference_audio)) {
            throw std::runtime_error("Engine: reference_audio not found: " + opts.reference_audio);
        }
        if (!opts.voice_dir.empty() &&
            !std::filesystem::is_directory(opts.voice_dir)) {
            throw std::runtime_error("Engine: voice_dir not found: " + opts.voice_dir);
        }

        ggml_time_init();
        g_log_verbose = opts.verbose ? 1 : 0;
        // Note: we deliberately do NOT call ggml_log_set here.  The
        // process-global sink is owned by the host application via
        // tts_cpp_log_set (see <tts-cpp/log.h>); installing one
        // unconditionally per Engine ctor would clobber whatever
        // structured-logging callback the host (Bare addon, telemetry
        // pipeline, ...) already registered for the process.

        if (!opts.reference_audio.empty() &&
            !validate_reference_audio(opts.reference_audio)) {
            throw std::runtime_error("Engine: reference_audio failed validation: " + opts.reference_audio);
        }

        if (!load_model_gguf(opts.t3_gguf_path, model, opts.n_ctx, opts.n_gpu_layers)) {
            throw std::runtime_error("Engine: failed to load T3 GGUF: " + opts.t3_gguf_path);
        }

        // Variant dispatch.  Turbo uses the GPT-2 BPE (built from the
        // model's tok_tokens / tok_merges arrays); multilingual uses
        // the HuggingFace MTL tokenizer JSON embedded in the GGUF.
        // synthesize() / run_t3() switch graph + sampling paths on the
        // same flag.
        //
        // The whole block is wrapped in one try/catch so any throw -
        // whether from the BPE/MTL tokenizer load, an unsupported
        // language, or the unknown-variant case - calls free_model()
        // before unwinding.  Without this the Turbo path's
        // bpe.load_from_arrays would leak the partially-constructed
        // model (backend, ctx_w, buffer_w) on a future malformed-vocab
        // throw, since ~Impl never runs on a partial construction.
        try {
            if (model.hparams.variant == CHBX_VARIANT_TURBO) {
                bpe.load_from_arrays(model.tok_tokens, model.tok_merges);
            } else if (model.hparams.variant == CHBX_VARIANT_MTL) {
                if (model.mtl_tokenizer_json.empty()) {
                    throw std::runtime_error(
                        "Engine: T3 GGUF reports chatterbox.variant == t3_mtl "
                        "but does not embed an mtl_tokenizer.json blob; "
                        "regenerate with scripts/convert-t3-mtl-to-gguf.py.  "
                        "Path: " + opts.t3_gguf_path);
                }
                if (!mtl_tok.load_from_json(model.mtl_tokenizer_json)) {
                    throw std::runtime_error(
                        "Engine: failed to parse embedded mtl_tokenizer.json "
                        "for: " + opts.t3_gguf_path);
                }
                if (!opts.language.empty() && !mtl_tok.is_language_supported(opts.language)) {
                    throw std::runtime_error(
                        "Engine: language '" + opts.language +
                        "' not in the multilingual tokenizer's tier-1 set");
                }
            } else {
                throw std::runtime_error(
                    "Engine: T3 GGUF reports unknown chatterbox.variant; "
                    "supported: t3_turbo, t3_mtl.  Path: " + opts.t3_gguf_path);
            }
        } catch (...) {
            free_model();
            throw;
        }

        s3gen_preload_thread = std::thread([path = opts.s3gen_gguf_path,
                                            ngpu = opts.n_gpu_layers]() {
            s3gen_preload(path, ngpu);
        });

        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) {
            wait_for_preload(s3gen_preload_thread);
            s3gen_unload();
            free_model();
            throw std::runtime_error("Engine: ggml_gallocr_new failed");
        }

        try {
            bake_voice_conditioning();
        } catch (...) {
            wait_for_preload(s3gen_preload_thread);
            s3gen_unload();
            if (allocr) { ggml_gallocr_free(allocr); allocr = nullptr; }
            free_model();
            throw;
        }
    }

    ~Impl() {
        wait_for_preload(s3gen_preload_thread);
        // Release the S3Gen cache (which holds its own backend + buffers)
        // BEFORE freeing the T3 backend.  If we don't, the cache's
        // backend resources get torn down by static destructors at
        // process exit, after ggml-metal's global device has already
        // been finalised, tripping its "rsets count == 0" assertion.
        s3gen_unload();
        if (allocr) {
            ggml_gallocr_free(allocr);
            allocr = nullptr;
        }
        free_model();
    }

    Impl(const Impl &)             = delete;
    Impl & operator=(const Impl &) = delete;

    void free_model() {
        // Pull (buffer_stack, ctx_stack) out of the process-wide t3_stack_registry
        // BEFORE freeing them locally — otherwise the atexit hook installed
        // by load_model_gguf_mtl on non-CPU backends would later double-free
        // (or, worse, ggml_backend_buffer_free a buffer whose backend has
        // already been destroyed below).  Mirrors the free_t3 lambda in
        // src/chatterbox_cli.cpp.  No-op on CPU backends and on Turbo
        // (those code paths never allocate buffer_stack / ctx_stack).
        if (model.buffer_stack || model.ctx_stack) {
            t3_stack_unregister(model.buffer_stack, model.ctx_stack);
        }
        // Drop the T3 step-graph cache BEFORE freeing the backend.
        // Cached gallocators carry backend references; freeing them
        // against a dead backend asserts inside the GPU-backend dylib
        // finalisers.
        tts_cpp::chatterbox::detail::t3_release_caches();
        if (model.buffer_w)        { ggml_backend_buffer_free(model.buffer_w);        model.buffer_w        = nullptr; }
        if (model.buffer_kv)       { ggml_backend_buffer_free(model.buffer_kv);       model.buffer_kv       = nullptr; }
        if (model.buffer_stack)    { ggml_backend_buffer_free(model.buffer_stack);    model.buffer_stack    = nullptr; }
        if (model.buffer_override) { ggml_backend_buffer_free(model.buffer_override); model.buffer_override = nullptr; }
        if (model.backend)         { ggml_backend_free(model.backend);                model.backend         = nullptr; }
        if (model.ctx_w)           { ggml_free(model.ctx_w);                          model.ctx_w           = nullptr; }
        if (model.ctx_kv)          { ggml_free(model.ctx_kv);                         model.ctx_kv          = nullptr; }
        if (model.ctx_stack)       { ggml_free(model.ctx_stack);                      model.ctx_stack       = nullptr; }
        if (model.ctx_override)    { ggml_free(model.ctx_override);                   model.ctx_override    = nullptr; }
    }

    // Loads speaker_emb + cond_prompt_speech_tokens from voice_dir when
    // available, computes them from reference_audio otherwise, and writes
    // the results into model.builtin_speaker_emb / builtin_cond_prompt_tokens
    // so subsequent T3 graphs pick up the cloned voice.  Also stashes the
    // three S3Gen-side tensors (prompt_feat, embedding, prompt_token) on
    // the Impl for reuse in synthesize().
    void bake_voice_conditioning() {
        if (opts.reference_audio.empty() && opts.voice_dir.empty()) {
            return;
        }

        const int n_threads = resolve_thread_count(opts.n_threads);

        bool have_se = false;
        bool have_ct = false;
        std::vector<float>   se_data;
        std::vector<int32_t> ct_data;

        if (!opts.voice_dir.empty()) {
            const std::string se_path = opts.voice_dir + "/speaker_emb.npy";
            const std::string ct_path = opts.voice_dir + "/cond_prompt_speech_tokens.npy";
            const std::string emb_path = opts.voice_dir + "/embedding.npy";
            const std::string pt_path  = opts.voice_dir + "/prompt_token.npy";
            const std::string pf_path  = opts.voice_dir + "/prompt_feat.npy";

            if (std::filesystem::exists(se_path)) {
                npy_array a = npy_load(se_path);
                se_data.assign((const float *) a.data.data(),
                               (const float *) a.data.data() + a.n_elements());
                have_se = true;
            }
            if (std::filesystem::exists(ct_path)) {
                npy_array a = npy_load(ct_path);
                ct_data.assign((const int32_t *) a.data.data(),
                               (const int32_t *) a.data.data() + a.n_elements());
                have_ct = true;
            }
            if (std::filesystem::exists(emb_path)) {
                npy_array a = npy_load(emb_path);
                s3gen_embedding.assign((const float *) a.data.data(),
                                      (const float *) a.data.data() + a.n_elements());
            }
            if (std::filesystem::exists(pt_path)) {
                npy_array a = npy_load(pt_path);
                s3gen_prompt_token.assign((const int32_t *) a.data.data(),
                                          (const int32_t *) a.data.data() + a.n_elements());
            }
            if (std::filesystem::exists(pf_path)) {
                npy_array a = npy_load(pf_path);
                s3gen_prompt_feat.assign((const float *) a.data.data(),
                                         (const float *) a.data.data() + a.n_elements());
                // prompt_feat.npy shape is (T_mel, 80)
                if (a.shape.size() >= 1) {
                    s3gen_prompt_feat_rows = (int) a.shape[0];
                }
            }
        }

        if (!have_se && !opts.reference_audio.empty()) {
            voice_encoder_weights vew;
            if (voice_encoder_load(opts.t3_gguf_path, vew)) {
                std::vector<float> wav;
                int sr = 0;
                if (!wav_load(opts.reference_audio, wav, sr)) {
                    throw std::runtime_error("Engine: failed to load reference_audio");
                }
                normalise_lufs(wav, sr, -27.0);
                if (sr != 16000) wav = resample_sinc(wav, sr, 16000);
                if (!voice_encoder_embed(wav, vew, model.backend, se_data)) {
                    throw std::runtime_error("Engine: VoiceEncoder forward failed");
                }
                have_se = true;
            }
        }

        std::vector<int32_t> prompt_token_from_ref;
        if (!have_ct && !opts.reference_audio.empty()) {
            std::vector<int32_t> cond_tokens;
            if (compute_speech_tokens_native(
                    opts.reference_audio, opts.s3gen_gguf_path,
                    /*max_cond_tokens=*/ model.hparams.cond_prompt_len,
                    prompt_token_from_ref, cond_tokens,
                    n_threads, /*backend=*/ model.backend, opts.verbose)) {
                ct_data = std::move(cond_tokens);
                have_ct = true;
            }
        }

        if (have_se) {
            if ((int64_t) se_data.size() != ggml_nelements(model.builtin_speaker_emb)) {
                throw std::runtime_error(
                    "Engine: speaker_emb size mismatch with builtin tensor");
            }
            ggml_backend_tensor_set(
                model.builtin_speaker_emb, se_data.data(), 0,
                ggml_nbytes(model.builtin_speaker_emb));
            voice_overridden = true;
        }

        if (have_ct) {
            if ((int64_t) ct_data.size() == ggml_nelements(model.builtin_cond_prompt_tokens)) {
                ggml_backend_tensor_set(
                    model.builtin_cond_prompt_tokens, ct_data.data(), 0,
                    ggml_nbytes(model.builtin_cond_prompt_tokens));
            } else {
                ggml_init_params op = { ggml_tensor_overhead() * 2, nullptr, true };
                model.ctx_override = ggml_init(op);
                if (!model.ctx_override) {
                    throw std::runtime_error("Engine: ggml_init(ctx_override) failed");
                }
                ggml_tensor * new_ct = ggml_new_tensor_1d(
                    model.ctx_override, GGML_TYPE_I32, (int64_t) ct_data.size());
                ggml_set_name(new_ct,
                              "chatterbox/builtin/cond_prompt_speech_tokens_override");
                model.buffer_override = ggml_backend_alloc_ctx_tensors(
                    model.ctx_override, model.backend);
                if (!model.buffer_override) {
                    throw std::runtime_error("Engine: alloc override buffer failed");
                }
                ggml_backend_tensor_set(
                    new_ct, ct_data.data(), 0, ct_data.size() * sizeof(int32_t));
                model.builtin_cond_prompt_tokens = new_ct;
                model.hparams.cond_prompt_len = (int32_t) ct_data.size();
            }
            voice_overridden = true;
        }

        if (!opts.reference_audio.empty()) {
            if (s3gen_prompt_feat.empty()) {
                int rows = 0;
                if (!compute_prompt_feat_native(
                        opts.reference_audio, opts.s3gen_gguf_path,
                        s3gen_prompt_feat, rows, opts.verbose)) {
                    throw std::runtime_error(
                        "Engine: failed to compute prompt_feat from reference_audio");
                }
                s3gen_prompt_feat_rows = rows;
            }
            if (s3gen_embedding.empty()) {
                if (!compute_embedding_native(
                        opts.reference_audio, opts.s3gen_gguf_path,
                        s3gen_embedding,
                        /*backend=*/ model.backend, opts.verbose)) {
                    // CAMPPlus tensors predate Phase 2d-a in this GGUF.
                    // `compute_embedding_native` already logged the concrete
                    // error to stderr; callers that want a hard failure can
                    // re-run after re-exporting the S3Gen GGUF with a
                    // current scripts/convert-s3gen-to-gguf.py.
                    fprintf(stderr,
                            "Engine: voice-cloning embedding unavailable; "
                            "falling back to built-in speaker embedding for %s\n",
                            opts.reference_audio.c_str());
                }
            }
            if (s3gen_prompt_token.empty() && !prompt_token_from_ref.empty()) {
                s3gen_prompt_token = std::move(prompt_token_from_ref);
            }
        }
    }

    std::vector<int32_t> run_t3(const std::string & text) {
        const bool is_mtl = (model.hparams.variant == CHBX_VARIANT_MTL);

        // Tokenise via the variant-appropriate tokenizer.  Both paths
        // produce a flat vector<int32_t> the rest of the loop is
        // agnostic to.
        std::vector<int32_t> text_tokens;
        if (is_mtl) {
            text_tokens = mtl_tok.encode(text, opts.language);
        } else {
            if (model.tok_tokens.empty()) {
                throw std::runtime_error(
                    "Engine: T3 GGUF has no embedded tokenizer; "
                    "re-run scripts/convert-t3-turbo-to-gguf.py");
            }
            const std::string normalised = gpt2_bpe::punc_norm(text);
            text_tokens = bpe.tokenize(normalised);
        }
        if (text_tokens.empty()) {
            throw std::runtime_error("Engine: text tokenised to empty sequence");
        }

        chatterbox_sampling_params sp;
        sp.top_k          = opts.top_k;
        sp.top_p          = opts.top_p;
        sp.temp           = opts.temperature;
        sp.repeat_penalty = opts.repeat_penalty;
        sp.min_p          = opts.min_p;
        sp.cfg_weight     = opts.cfg_weight;

        const int n_threads = resolve_thread_count(opts.n_threads);
        std::mt19937 rng(opts.seed);

        // Prompt eval.  Turbo: single-pass; MTL: cond + uncond batched
        // into one B=2 forward, with the eval_*_mtl helpers returning
        // both logit slices.
        std::vector<float> logits;
        std::vector<float> logits_c, logits_u;
        int prompt_len = 0;
        if (is_mtl) {
            if (!eval_prompt_mtl(model, allocr, n_threads,
                                 text_tokens, opts.exaggeration,
                                 logits_c, logits_u, prompt_len)) {
                throw std::runtime_error("Engine: T3 prompt eval failed (mtl)");
            }
        } else {
            if (!eval_prompt(model, allocr, n_threads,
                             text_tokens, logits, prompt_len)) {
                throw std::runtime_error("Engine: T3 prompt eval failed");
            }
        }

        int n_past = prompt_len;
        std::vector<int32_t> generated;
        generated.reserve((size_t) opts.n_predict + 1);

        int32_t current = is_mtl
            ? sample_next_token_mtl(logits_c, logits_u, generated, sp, rng,
                                    model.hparams.stop_speech_token)
            : sample_next_token_ex(logits, generated, sp, rng);
        generated.push_back(current);

        for (int i = 0; i < opts.n_predict; ++i) {
            if (cancel_flag.load(std::memory_order_relaxed)) {
                throw std::runtime_error("Engine: synthesis cancelled during T3 decode");
            }
            if (current == model.hparams.stop_speech_token) break;
            if (n_past + 1 > model.hparams.n_ctx) break;
            const bool step_ok = is_mtl
                ? eval_step_mtl(model, allocr, n_threads, n_past, current,
                                logits_c, logits_u)
                : eval_step(model, allocr, n_threads, n_past, current, logits);
            if (!step_ok) {
                throw std::runtime_error("Engine: T3 step eval failed");
            }
            ++n_past;
            current = is_mtl
                ? sample_next_token_mtl(logits_c, logits_u, generated, sp, rng,
                                        model.hparams.stop_speech_token)
                : sample_next_token_ex(logits, generated, sp, rng);
            generated.push_back(current);
        }

        if (!generated.empty() && generated.back() == model.hparams.stop_speech_token) {
            generated.pop_back();
        }
        return generated;
    }

    // Populate the fixed fields of s3gen_synthesize_opts (paths, threads,
    // seed, voice overrides, backend hints).  Streaming-specific fields
    // (finalize, hift_cache_source, skip_mel_frames, ...) are set per
    // chunk by `synthesize_streaming` below.
    void fill_common_s3gen_opts(s3gen_synthesize_opts & sopts) {
        sopts.s3gen_gguf_path = opts.s3gen_gguf_path;
        sopts.out_wav_path    = "";
        sopts.seed            = opts.seed;
        sopts.n_threads       = resolve_thread_count(opts.n_threads);
        sopts.verbose         = opts.verbose;
        sopts.n_gpu_layers    = opts.n_gpu_layers;

        // Use the non-owning view fields rather than the *_override
        // vectors so the streaming path doesn't pay a per-chunk MB-
        // sized value-copy.  Engine::Impl owns the underlying storage
        // for the duration of the synthesize() / synthesize_streaming()
        // call; the views are valid as long as 'this' is.
        if (!s3gen_prompt_feat.empty()) {
            sopts.prompt_feat_view_data = s3gen_prompt_feat.data();
            sopts.prompt_feat_view_size = s3gen_prompt_feat.size();
            sopts.prompt_feat_view_rows = s3gen_prompt_feat_rows;
        }
        if (!s3gen_embedding.empty()) {
            sopts.embedding_view_data = s3gen_embedding.data();
            sopts.embedding_view_size = s3gen_embedding.size();
        }
        if (!s3gen_prompt_token.empty()) {
            sopts.prompt_token_view_data = s3gen_prompt_token.data();
            sopts.prompt_token_view_size = s3gen_prompt_token.size();
        }
        // Cooperative-cancel hook so a single Engine::cancel() reaches
        // both the T3 decode loop (handled directly) and the S3Gen +
        // HiFT path (cooperatively checked between CFM steps and
        // before HiFT decode).
        sopts.cancel_flag = &cancel_flag;
    }

    SynthesisResult synthesize_batch(const std::vector<int32_t> & speech_tokens,
                                     SynthesisResult && partial) {
        s3gen_synthesize_opts sopts;
        fill_common_s3gen_opts(sopts);
        sopts.cfm_steps = opts.cfm_steps;

        // apply_trim_fade defaults to true at the s3gen layer to mask
        // reference-audio bleed-through across the prompt-feat / first
        // synthesized-mel boundary -- the prompt audio context can leak
        // ~20-40 ms of HiFT state into the start of "real" output. When
        // there's no reference audio (built-in voice baked into the S3Gen
        // GGUF, default for chatterbox::Engine), there's nothing to bleed
        // through: the prompt_feat tensor is well-formed pre-recorded mel
        // and HiFT primes cleanly. Leaving apply_trim_fade=true in that
        // mode silently zeros + fades the first 40 ms of legitimate
        // speech, which empirically clipped the leading consonant of the
        // first word ("Hello" -> "lo", "El" -> "l", etc.) for the
        // chatterbox-mtl variant whose built-in conds.pt produces audio
        // with zero leading silence. Gate on the actual presence of a
        // reference-audio source so the existing reference-audio path is
        // unaffected.
        const bool has_voice_override =
            !opts.reference_audio.empty() || !opts.voice_dir.empty();
        sopts.apply_trim_fade = has_voice_override;

        SynthesisResult result = std::move(partial);
        sopts.pcm_out = &result.pcm;

        const auto s3_t0 = std::chrono::steady_clock::now();
        const int rc = s3gen_synthesize_to_wav(speech_tokens, sopts);
        const auto s3_t1 = std::chrono::steady_clock::now();
        if (rc == 2) {
            throw std::runtime_error("Engine: synthesis cancelled during S3Gen+HiFT");
        }
        if (rc != 0) {
            throw std::runtime_error("Engine: s3gen_synthesize_to_wav failed with code "
                                     + std::to_string(rc));
        }

        result.sample_rate   = 24000;
        result.t3_tokens     = (int) speech_tokens.size();
        result.audio_samples = (int) result.pcm.size();
        result.s3gen_ms      = std::chrono::duration<double, std::milli>(s3_t1 - s3_t0).count();
        return result;
    }

    // Ports main.cpp's --stream-chunk-tokens loop.  Splits speech_tokens
    // into chunks of stream_chunk_tokens (with an optional smaller first
    // chunk), carries `hift_cache_source` and `skip_mel_frames` across
    // chunks for phase-continuous seams, and invokes `on_chunk` with
    // each chunk's PCM as it's produced.  Accumulates the full PCM in
    // the returned SynthesisResult so batch callers get the same shape.
    SynthesisResult synthesize_streaming(
        const std::vector<int32_t> & speech_tokens,
        const StreamCallback & on_chunk,
        SynthesisResult && partial) {

        std::vector<int32_t> seg_toks = speech_tokens;
        for (int i = 0; i < kS3GenLookaheadTokens; ++i) {
            seg_toks.push_back(kS3GenSilenceToken);
        }
        const int total_n = (int) seg_toks.size();

        const int chunk_n       = opts.stream_chunk_tokens;
        const int first_chunk_n = opts.stream_first_chunk_tokens > 0
                                    ? opts.stream_first_chunk_tokens
                                    : chunk_n;

        std::vector<int> boundaries = {0};
        int cursor = std::min(first_chunk_n, total_n);
        boundaries.push_back(cursor);
        while (cursor < total_n) {
            cursor = std::min(cursor + chunk_n, total_n);
            boundaries.push_back(cursor);
        }
        // Absorb a tiny trailing chunk into the previous one (avoids
        // paying the full encoder+CFM cost for a handful of new tokens;
        // matches main.cpp's tail-merge heuristic).
        const int min_tail = std::max(6, chunk_n / 3);
        if (boundaries.size() >= 3) {
            const int tail_len = boundaries.back() - boundaries[boundaries.size() - 2];
            if (tail_len < min_tail) boundaries.erase(boundaries.end() - 2);
        }

        std::vector<float> hift_cache_source;
        int prev_mels_emitted = 0;

        SynthesisResult result = std::move(partial);
        result.pcm.clear();

        const int n_chunks = (int) boundaries.size() - 1;
        double s3gen_ms_total = 0.0;

        for (int k = 1; k <= n_chunks; ++k) {
            if (cancel_flag.load(std::memory_order_relaxed)) {
                throw std::runtime_error("Engine: synthesis cancelled during streaming");
            }
            const int end              = boundaries[k];
            const bool is_last_in_seg  = (end == total_n);
            std::vector<int32_t> toks(seg_toks.begin(), seg_toks.begin() + end);

            s3gen_synthesize_opts copts;
            fill_common_s3gen_opts(copts);
            std::vector<float> chunk_pcm;
            copts.pcm_out                   = &chunk_pcm;
            copts.append_lookahead_silence  = false;
            copts.finalize                  = is_last_in_seg;
            copts.skip_mel_frames           = prev_mels_emitted;
            copts.apply_trim_fade           = (k == 1);
            copts.hift_cache_source         = hift_cache_source;
            std::vector<float> tail_out;
            copts.hift_source_tail_out      = &tail_out;
            copts.source_tail_samples       = 480;
            copts.cfm_steps                 = opts.stream_cfm_steps;

            const auto s3_t0 = std::chrono::steady_clock::now();
            const int rc = s3gen_synthesize_to_wav(toks, copts);
            const auto s3_t1 = std::chrono::steady_clock::now();
            if (rc == 2) {
                throw std::runtime_error(
                    "Engine: synthesis cancelled during S3Gen+HiFT chunk "
                    + std::to_string(k));
            }
            if (rc != 0) {
                throw std::runtime_error(
                    "Engine: streaming chunk " + std::to_string(k) +
                    " failed with code " + std::to_string(rc));
            }
            s3gen_ms_total += std::chrono::duration<double, std::milli>(s3_t1 - s3_t0).count();

            on_chunk(chunk_pcm.data(), chunk_pcm.size(), k - 1, is_last_in_seg);

            result.pcm.insert(result.pcm.end(), chunk_pcm.begin(), chunk_pcm.end());
            hift_cache_source = std::move(tail_out);
            const size_t chunk_samples = chunk_pcm.size();
            prev_mels_emitted += (int)(chunk_samples / 480);
        }

        result.sample_rate   = 24000;
        result.t3_tokens     = (int) speech_tokens.size();
        result.audio_samples = (int) result.pcm.size();
        result.s3gen_ms      = s3gen_ms_total;
        return result;
    }

    SynthesisResult synthesize(const std::string & text,
                               const StreamCallback & on_chunk) {
        if (text.empty()) {
            throw std::runtime_error("Engine: text is empty");
        }
        cancel_flag.store(false, std::memory_order_relaxed);

        const auto t3_t0 = std::chrono::steady_clock::now();
        std::vector<int32_t> speech_tokens = run_t3(text);
        const auto t3_t1 = std::chrono::steady_clock::now();

        wait_for_preload(s3gen_preload_thread);

        SynthesisResult partial;
        partial.t3_ms = std::chrono::duration<double, std::milli>(t3_t1 - t3_t0).count();

        const bool use_streaming = on_chunk && opts.stream_chunk_tokens > 0;
        return use_streaming
            ? synthesize_streaming(speech_tokens, on_chunk, std::move(partial))
            : synthesize_batch(speech_tokens, std::move(partial));
    }

    SynthesisResult synthesize(const std::string & text) {
        return synthesize(text, StreamCallback{});
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
    return pimpl_->synthesize(text, on_chunk);
}

void Engine::cancel() {
    if (pimpl_) pimpl_->cancel_flag.store(true, std::memory_order_relaxed);
}

const EngineOptions & Engine::options() const {
    return pimpl_->opts;
}

std::string Engine::backend_name() const {
    if (!pimpl_->model.backend) {
        return "(unknown)";
    }
    if (const char * name = ggml_backend_name(pimpl_->model.backend)) {
        return std::string(name);
    }
    return "(unknown)";
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

} // namespace tts_cpp::chatterbox
