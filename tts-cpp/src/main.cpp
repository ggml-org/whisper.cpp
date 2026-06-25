#include "backend_selection.h"
#include "backend_util.h"
#include "gguf_stream.h"
#include "gpt2_bpe.h"
#include "mtl_tokenizer.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

// Per-backend `#include "ggml-{cuda,metal,vulkan,opencl}.h"` blocks
// used to sit here gated on `GGML_USE_<X>` so the legacy
// `init_backend` cascade below could call the static
// `ggml_backend_<x>_init` entry points directly. Removed alongside
// the cascade: tts-cpp routes every backend decision through the
// ggml-backend registry (`backend_selection.{h,cpp}`), which reaches
// the same backends in both `GGML_BACKEND_DL=ON` (Android prebuild)
// and `=OFF` (desktop dev) modes without linking those static
// symbols. Mirrors parakeet-cpp's design.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tts-cpp/tts-cpp.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"
#include "chatterbox_t3_internal.h"
#include "npy.h"
#include "voice_features.h"
#include "voice_encoder.h"
#include "campplus.h"
#include "s3tokenizer.h"
#include "gguf.h"

#include <sys/stat.h>

using namespace tts_cpp::chatterbox::detail;


namespace tts_cpp::chatterbox::detail {

bool validate_reference_audio(const std::string & path) {
    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(path, wav, sr)) {
        fprintf(stderr, "error: failed to load --reference-audio: %s\n", path.c_str());
        return false;
    }
    const double secs = (double)wav.size() / (double)sr;
    if (secs <= 5.0) {
        fprintf(stderr,
            "error: --reference-audio is only %.2f s; Chatterbox requires strictly more "
            "than 5 s of clean mono speech.  Shorter references produce undersized "
            "conditioning tensors and the model falls back on the built-in voice.\n"
            "  Recommended length: 10–15 s.\n", secs);
        return false;
    }
    if (secs < 10.0) {
        fprintf(stderr,
            "warning: --reference-audio is %.2f s; 10–15 s is recommended for best "
            "voice similarity.\n", secs);
    }
    return true;
}

// Load REF.wav, resample to 24 kHz if needed, pull the 80-ch mel filterbank out
// of the s3gen GGUF, and compute prompt_feat (log-mel) in C++.  out_rows is the
// number of mel frames (= T_mel in the row-major (T_mel, 80) layout).
bool compute_prompt_feat_native(const std::string & wav_path,
                                       const std::string & s3gen_gguf_path,
                                       std::vector<float> & out_feat,
                                       int & out_rows,
                                       bool verbose)
{
    if (verbose) fprintf(stderr, "voice: loading %s\n", wav_path.c_str());
    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(wav_path, wav, sr)) return false;
    if (verbose) fprintf(stderr, "voice:   sr=%d samples=%zu (%.2f s)\n", sr, wav.size(), (double)wav.size() / sr);
    if (sr != 24000) {
        if (verbose) fprintf(stderr, "voice: resampling %d -> 24000\n", sr);
        wav = resample_sinc(wav, sr, 24000);
    }

    // Loudness-normalise to -27 LUFS (matches tts_turbo.norm_loudness).  A
    // quiet reference wav (e.g. -44 LUFS) would otherwise produce mel values
    // 15–20 dB lower than what S3Gen was trained on, and the conditioning
    // would pull the output towards the default voice.
    double pre  = measure_lufs(wav, 24000);
    normalise_lufs(wav, 24000, -27.0);
    if (verbose) fprintf(stderr, "voice:   loudness %.2f LUFS → -27 LUFS (+%.2f dB)\n", pre, -27.0 - pre);

    // Match Python's tts_turbo.prepare_conditionals:
    //   s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]  # 10 * S3GEN_SR = 240000
    //   s3gen.embed_ref(s3gen_ref_wav, 24000)  # → prompt_feat/embedding/prompt_token
    // Trim to the first 10 s at 24 kHz before computing anything, otherwise
    // prompt_feat comes out (~787, 80) instead of the (500, 80) S3Gen was
    // trained on, and conditioning suffers.
    const int dec_cond_samples = 10 * 24000;
    if ((int)wav.size() > dec_cond_samples) wav.resize(dec_cond_samples);

    // Pull the precomputed mel filterbank out of chatterbox-s3gen.gguf.
    // no_alloc=true + streamed read: this used to stage the whole ~1 GB
    // s3gen data section in host memory just to copy one 80-row matrix.
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ &tmp_ctx };
    gguf_context * g = gguf_init_from_file(s3gen_gguf_path.c_str(), gp);
    if (!g) {
        fprintf(stderr, "voice: failed to open %s\n", s3gen_gguf_path.c_str());
        return false;
    }
    ggml_tensor * fb = ggml_get_tensor(tmp_ctx, "s3gen/mel_fb/24k_80");
    if (!fb) {
        fprintf(stderr, "voice: s3gen/mel_fb/24k_80 missing from GGUF; re-run convert-s3gen-to-gguf.py\n");
        gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
        return false;
    }
    std::vector<float> mel_fb(ggml_nelements(fb));
    {
        ::tts_cpp::detail::gguf_stream_reader reader(g, s3gen_gguf_path);
        if (!reader.ok() || !reader.to_host("s3gen/mel_fb/24k_80", mel_fb.data(), ggml_nbytes(fb))) {
            gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
            return false;
        }
    }
    gguf_free(g);
    if (tmp_ctx) ggml_free(tmp_ctx);

    out_feat = mel_extract_24k_80(wav, mel_fb);
    if (out_feat.empty()) return false;
    out_rows = (int)(out_feat.size() / 80);
    if (verbose) fprintf(stderr, "voice: prompt_feat shape=(%d, 80)\n", out_rows);
    return true;
}

// Compute the 192-d `embedding` tensor natively by running the reference wav
// through the C++ Kaldi fbank → mean-subtract → CAMPPlus pipeline.  Returns
// false silently if the s3gen GGUF predates Phase 2d-a (no CAMPPlus tensors).
bool compute_embedding_native(const std::string & wav_path,
                                     const std::string & s3gen_gguf_path,
                                     std::vector<float> & out_emb,
                                     ggml_backend_t backend,
                                     bool verbose)
{
    campplus_weights w;
    if (!campplus_load(s3gen_gguf_path, w)) {
        fprintf(stderr, "voice: s3gen GGUF has no CAMPPlus weights; cannot synthesise "
                        "embedding natively (re-run convert-s3gen-to-gguf.py)\n");
        return false;
    }

    // Mel filterbank for the Kaldi-style fbank lives alongside CAMPPlus in
    // the s3gen GGUF (baked in by the converter).  Streamed read — no
    // full-file staging (see gguf_stream.h).
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ &tmp_ctx };
    gguf_context * g = gguf_init_from_file(s3gen_gguf_path.c_str(), gp);
    if (!g) return false;
    ggml_tensor * fb_t = ggml_get_tensor(tmp_ctx, "campplus/mel_fb_kaldi_80");
    if (!fb_t) {
        fprintf(stderr, "voice: campplus/mel_fb_kaldi_80 missing; rerun converter\n");
        gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
        return false;
    }
    std::vector<float> mel_fb(ggml_nelements(fb_t));
    {
        ::tts_cpp::detail::gguf_stream_reader reader(g, s3gen_gguf_path);
        if (!reader.ok() || !reader.to_host("campplus/mel_fb_kaldi_80", mel_fb.data(), ggml_nbytes(fb_t))) {
            gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);
            return false;
        }
    }
    gguf_free(g); if (tmp_ctx) ggml_free(tmp_ctx);

    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(wav_path, wav, sr)) return false;
    // Python normalises loudness at the ORIGINAL sample rate, before any
    // resampling.  Do the same so the gain matches exactly.
    normalise_lufs(wav, sr, -27.0);
    if (sr != 16000) wav = resample_sinc(wav, sr, 16000);

    // Match Python's s3gen.embed_ref: the reference wav has been trimmed to
    // DEC_COND_LEN = 10 s @ 24 kHz before being passed in, then internally
    // resampled to 16 kHz.  Trimming to the equivalent 10 s @ 16 kHz after
    // resampling gets us the same slice for CAMPPlus (fbank + speaker encoder).
    const int dec_cond_samples_16k = 10 * 16000;
    if ((int)wav.size() > dec_cond_samples_16k) wav.resize(dec_cond_samples_16k);

    std::vector<float> fbank = fbank_kaldi_80(wav, mel_fb);
    if (fbank.empty()) return false;
    const int T = (int)(fbank.size() / 80);

    // Per-utterance mean subtraction over T (matches extract_feature()).
    std::vector<float> col_mean(80, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) col_mean[c] += fbank[(size_t)t * 80 + c];
    for (int c = 0; c < 80; ++c) col_mean[c] /= (float)T;
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) fbank[(size_t)t * 80 + c] -= col_mean[c];

    // Force the scalar C++ CAMPPlus path for now.  The ggml-graph variant
    // (campplus_embed_ggml) produces an antipodal embedding vs the
    // scalar/Python reference on real voice inputs (cos_sim ~ -0.19 vs
    // Python, while the scalar path matches at ~0.9999).  The bug is in
    // the graph construction and isn't exercised by test-campplus because
    // that harness passes backend=nullptr too.  CAMPPlus only runs once
    // per voice-bake, ~500 ms on CPU, so the ggml speed-up isn't critical
    // for user-visible latency — we pay a small one-time cost in exchange
    // for a correct speaker embedding.
    (void)backend;
    if (!campplus_embed(fbank, T, w, /*backend=*/nullptr, out_emb)) return false;
    if (verbose) fprintf(stderr, "voice: embedding shape=(%zu,) via CAMPPlus (%d fbank frames)\n",
            out_emb.size(), T);
    return true;
}

// Tokenize a reference wav with S3TokenizerV2 (the C++ port of the 25 Hz
// speech-token encoder).  Produces both the S3Gen-side prompt_token stream
// (first 10 s → up to 250 tokens) and the T3-side cond_prompt_speech_tokens
// stream (first 15 s → up to max_cond_tokens tokens).
//
// Returns false if the s3gen GGUF pre-dates Phase 2e (no s3tokv2.* tensors).
bool compute_speech_tokens_native(const std::string & wav_path,
                                         const std::string & s3gen_gguf_path,
                                         int max_cond_tokens,
                                         std::vector<int32_t> & out_prompt_tokens,
                                         std::vector<int32_t> & out_cond_tokens,
                                         int n_threads,
                                         ggml_backend_t backend,
                                         bool verbose = false)
{
    s3tokv2_weights w;
    if (!s3tokv2_load(s3gen_gguf_path, w)) {
        fprintf(stderr, "voice: s3gen GGUF has no S3TokenizerV2 weights; cannot "
                        "synthesise speech tokens natively (re-run converter)\n");
        return false;
    }

    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(wav_path, wav, sr)) return false;
    normalise_lufs(wav, sr, -27.0);
    if (sr != 16000) wav = resample_sinc(wav, sr, 16000);

    // prompt_token: first 10 s of the reference → up to 250 tokens (S3Gen side).
    const int dec_cond_samples = 10 * 16000;
    std::vector<float> prompt_wav(wav.begin(), wav.begin() + std::min((int)wav.size(), dec_cond_samples));
    if (!s3tokv2_tokenize(prompt_wav, w, /*max_tokens=*/-1, out_prompt_tokens, n_threads, backend)) return false;

    // cond_prompt_speech_tokens: first 15 s → up to max_cond_tokens (T3 side).
    const int enc_cond_samples = 15 * 16000;
    std::vector<float> cond_wav(wav.begin(), wav.begin() + std::min((int)wav.size(), enc_cond_samples));
    if (!s3tokv2_tokenize(cond_wav, w, max_cond_tokens, out_cond_tokens, n_threads, backend)) return false;

    if (verbose) fprintf(stderr, "voice: prompt_token=(%zu,) cond_prompt_speech_tokens=(%zu,) via S3TokenizerV2\n",
            out_prompt_tokens.size(), out_cond_tokens.size());
    return true;
}

// --------------------------------------------------------------------------

static int64_t require_key(const gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) throw std::runtime_error(std::string("missing GGUF key: ") + key);
    return id;
}

static ggml_tensor * require_tensor(const chatterbox_model & m, const char * name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end() || !it->second) throw std::runtime_error(std::string("missing tensor: ") + name);
    return it->second;
}

// --------------------------------------------------------------------------
// Backend init
// --------------------------------------------------------------------------

// Verbose flag: set once in main() before any ggml init so helpers
// below (init_backend, load_model_gguf) can gate their startup prints on it.
// 0 = quiet, 1 = --verbose mode.  Declared extern in chatterbox_t3_internal.h
// so chatterbox_engine.cpp can flip it from its Engine ctor without a copy.
int g_log_verbose = 0;

ggml_backend_t init_backend(int n_gpu_layers, bool * out_gpu_unsupported) {
    if (out_gpu_unsupported) *out_gpu_unsupported = false;
    const bool v = g_log_verbose != 0;
    // GPU cascade is centralised in backend_selection.cpp's
    // `init_gpu_backend` (Adreno 700+ -> OpenCL, every other GPU ->
    // Vulkan/Metal/CUDA/Mali, with Adreno 6xx OpenCL force-skipped).
    // The registry walk it does reaches the same set of backends in
    // both `GGML_BACKEND_DL=ON` and `=OFF` modes without linking the
    // per-backend static `ggml_backend_<x>_init` entry points.
    bool gpu_present_but_unused = false;
    if (ggml_backend_t b = ::tts_cpp::detail::init_gpu_backend(
            n_gpu_layers, v, "chatterbox", 0,
            /*allow_arm_mali=*/true, &gpu_present_but_unused)) {
        return b;
    }
    if (out_gpu_unsupported) *out_gpu_unsupported = gpu_present_but_unused;
    if (ggml_backend_t b = ::tts_cpp::detail::init_cpu_backend()) {
        if (v) fprintf(stderr, "chatterbox: using CPU backend\n");
        return b;
    }
    throw std::runtime_error("init_backend: no CPU device registered");
}

// --------------------------------------------------------------------------
// Model loading
// --------------------------------------------------------------------------

ggml_type chatterbox_kv_type_from_str(const std::string & s) {
    if (s.empty() || s == "f32") return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    fprintf(stderr, "chatterbox: unknown kv_cache_type '%s' (expected f32|f16|q8_0); using f32\n",
            s.c_str());
    return GGML_TYPE_F32;
}

// Resolve the effective KV-cache dtype for `backend`: returns `requested`
// when the backend accepts a flash-attention node with K/V of that type at
// the model's head geometry, else falls back to F32 with a stderr warning.
//
// F32 is always supported, so it short-circuits.  For f16/q8_0 we build a
// throwaway, no_alloc flash_attn_ext node shaped like the real T3 attention
// (Q=F32 [HD, 1, n_head], K/V=kv_type [HD, 8, n_kv_head], F16 mask) and ask
// ggml_backend_supports_op.  This stops an f16/q8_0 request from asserting
// deep inside a backend that genuinely rejects quantized K/V (the graph
// would otherwise fail at first compute, long after load).
//
// CAVEAT: a backend that ADVERTISES support via supports_op but faults at
// compute is not caught here — notably ggml-vulkan reports q8_0 K/V FA as
// supported on both its scalar and coopmat2 paths (see ggml-vulkan.cpp
// GGML_OP_FLASH_ATTN_EXT and the supertonic notes in supertonic_internal.h).
// That advertise-vs-actual gap is an upstream ggml concern; this probe is
// the load-time guard for honest backends and future ports.
ggml_type chatterbox_resolve_kv_type(ggml_backend_t backend, ggml_type requested,
                                     int head_dim, int n_head, int n_kv_head) {
    if (requested == GGML_TYPE_F32 || !backend) return GGML_TYPE_F32;

    // ggml-vulkan's supports_op ADVERTISES quantized K/V flash-attention
    // but the NV_coopmat2 kernel faults at compute on quantized K/V:
    // toggle-confirmed in CI (QVAC-19557) — a q8_0 KV cache SIGSEGVs the
    // chatterbox GPU smoke on NVIDIA coopmat2 runners where the identical
    // f32 run passes.  MoltenVK (scalar FA, no coopmat) runs q8_0 fine, so
    // this is specific to the coopmat2 dequant-in-shader path.  The probe
    // below can't catch an advertise-vs-actual gap, so force quantized K/V
    // to f32 on Vulkan until the upstream coopmat2 FA kernel is fixed.
    // f16 (the native FA input type, not dequantized in-shader) is left
    // intact.  Metal / CPU keep quantized K/V (validated byte-identical).
    if (ggml_is_quantized(requested) && ::tts_cpp::detail::backend_is_vulkan(backend)) {
        fprintf(stderr, "chatterbox: quantized (%s) KV cache is not yet reliable on the "
                        "Vulkan backend (coopmat2 flash-attn faults); using f32 KV cache\n",
                ggml_type_name(requested));
        return GGML_TYPE_F32;
    }

    bool supported = false;
    ggml_init_params pp = { ggml_tensor_overhead() * 8, nullptr, /*no_alloc=*/true };
    if (ggml_context * pc = ggml_init(pp)) {
        try {
            const int kv = 8;  // any small length; supports_op checks types + head geometry
            // Null mask mirrors the real single-step (N=1) attention call,
            // which avoids ggml's mask-padding constraints in the probe.
            ggml_tensor * Q = ggml_new_tensor_3d(pc, GGML_TYPE_F32, head_dim, 1,  n_head);
            ggml_tensor * K = ggml_new_tensor_3d(pc, requested,     head_dim, kv, n_kv_head);
            ggml_tensor * V = ggml_new_tensor_3d(pc, requested,     head_dim, kv, n_kv_head);
            ggml_tensor * op = ggml_flash_attn_ext(pc, Q, K, V, /*mask=*/nullptr,
                                                   1.0f / std::sqrt((float) head_dim), 0.0f, 0.0f);
            supported = (op != nullptr) && ggml_backend_supports_op(backend, op);
        } catch (...) {
            supported = false;
        }
        ggml_free(pc);
    }

    if (!supported) {
        fprintf(stderr, "chatterbox: backend does not support %s K/V flash-attention; "
                        "falling back to f32 KV cache\n", ggml_type_name(requested));
        return GGML_TYPE_F32;
    }
    return requested;
}

bool load_model_gguf(const std::string & path, chatterbox_model & model, int requested_ctx, int n_gpu_layers, ggml_type kv_type) {
    {
        gguf_init_params peek_params = { /*.no_alloc=*/ true, /*.ctx=*/ nullptr };
        gguf_context * peek_ctx = gguf_init_from_file(path.c_str(), peek_params);
        if (peek_ctx) {
            std::string variant = "t3_turbo";
            const int64_t vk = gguf_find_key(peek_ctx, KEY_VARIANT);
            if (vk >= 0 && gguf_get_kv_type(peek_ctx, vk) == GGUF_TYPE_STRING) {
                const char * v = gguf_get_val_str(peek_ctx, vk);
                if (v) variant = v;
            } else if (vk >= 0) {
                fprintf(stderr, "%s: %s has unexpected GGUF type %d (expected STRING); refusing to load\n",
                        __func__, KEY_VARIANT, (int) gguf_get_kv_type(peek_ctx, vk));
                gguf_free(peek_ctx);
                return false;
            }
            gguf_free(peek_ctx);
            if (variant == "t3_mtl") {
                return load_model_gguf_mtl(path, model, requested_ctx, n_gpu_layers, kv_type);
            }
        }
    }
    // no_alloc=true: tmp_ctx carries tensor METADATA only; payloads are
    // streamed straight from the file into the backend buffer below so the
    // full data section is never staged in host memory (see gguf_stream.h).
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gguf_params = { /*.no_alloc=*/ true, /*.ctx=*/ &tmp_ctx };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_ctx) { fprintf(stderr, "%s: failed to open '%s'\n", __func__, path.c_str()); return false; }

    try {
        auto & hp = model.hparams;
        hp.variant = CHBX_VARIANT_TURBO;
        hp.n_text_vocab       = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_TEXT_VOCAB_SIZE));
        hp.n_speech_vocab     = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_SPEECH_VOCAB_SIZE));
        hp.start_speech_token = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_START_SPEECH));
        hp.stop_speech_token  = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_STOP_SPEECH));
        hp.speaker_embed_size = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_SPEAKER_EMBED));
        hp.cond_prompt_len    = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_COND_PROMPT_LEN));
        hp.eps                = gguf_get_val_f32(gguf_ctx, require_key(gguf_ctx, KEY_LAYER_NORM_EPS));
        hp.n_ctx   = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_CTX));
        hp.n_embd  = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_EMBD));
        hp.n_head  = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_HEAD));
        hp.n_layer = (int32_t) gguf_get_val_u32(gguf_ctx, require_key(gguf_ctx, KEY_N_LAYER));
        if (requested_ctx > 0) hp.n_ctx = std::min(hp.n_ctx, requested_ctx);

        model.backend = init_backend(n_gpu_layers, &model.gpu_unsupported);

        const int64_t num_tensors = gguf_get_n_tensors(gguf_ctx);
        ggml_init_params params = { ggml_tensor_overhead() * (size_t) num_tensors, nullptr, true };
        model.ctx_w = ggml_init(params);
        if (!model.ctx_w) throw std::runtime_error("ggml_init() failed");

        for (int64_t i = 0; i < num_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
            ggml_tensor * dst = ggml_dup_tensor(model.ctx_w, src);
            ggml_set_name(dst, name);
            model.tensors[name] = dst;
        }
        model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
        {
            ::tts_cpp::detail::gguf_stream_reader reader(gguf_ctx, path);
            if (!reader.ok()) throw std::runtime_error("failed to reopen GGUF for tensor data: " + path);
            for (ggml_tensor * cur = ggml_get_first_tensor(model.ctx_w); cur; cur = ggml_get_next_tensor(model.ctx_w, cur)) {
                if (!reader.to_backend(ggml_get_name(cur), cur)) {
                    throw std::runtime_error(std::string("failed to stream tensor '") +
                                             ggml_get_name(cur) + "' from " + path);
                }
            }
        }

        model.wpe              = require_tensor(model, "model/wpe");
        model.ln_f_g           = require_tensor(model, "model/ln_f/g");
        model.ln_f_b           = require_tensor(model, "model/ln_f/b");
        model.text_emb         = require_tensor(model, "chatterbox/text_emb");
        model.speech_emb       = require_tensor(model, "chatterbox/speech_emb");
        model.speech_head      = require_tensor(model, "chatterbox/speech_head");
        model.speech_head_bias = require_tensor(model, "chatterbox/speech_head_bias");
        model.cond_spkr_w      = require_tensor(model, "chatterbox/cond_spkr/w");
        model.cond_spkr_b      = require_tensor(model, "chatterbox/cond_spkr/b");
        model.builtin_speaker_emb        = require_tensor(model, "chatterbox/builtin/speaker_emb");
        model.builtin_cond_prompt_tokens = require_tensor(model, "chatterbox/builtin/cond_prompt_speech_tokens");

        model.layers.resize(hp.n_layer);
        for (int i = 0; i < hp.n_layer; ++i) {
            auto & l = model.layers[i];
            std::string p = "model/h" + std::to_string(i);
            l.ln_1_g        = require_tensor(model, (p + "/ln_1/g").c_str());
            l.ln_1_b        = require_tensor(model, (p + "/ln_1/b").c_str());
            l.ln_2_g        = require_tensor(model, (p + "/ln_2/g").c_str());
            l.ln_2_b        = require_tensor(model, (p + "/ln_2/b").c_str());
            l.c_attn_attn_w = require_tensor(model, (p + "/attn/c_attn/w").c_str());
            l.c_attn_attn_b = require_tensor(model, (p + "/attn/c_attn/b").c_str());
            l.c_attn_proj_w = require_tensor(model, (p + "/attn/c_proj/w").c_str());
            l.c_attn_proj_b = require_tensor(model, (p + "/attn/c_proj/b").c_str());
            l.c_mlp_fc_w    = require_tensor(model, (p + "/mlp/c_fc/w").c_str());
            l.c_mlp_fc_b    = require_tensor(model, (p + "/mlp/c_fc/b").c_str());
            l.c_mlp_proj_w  = require_tensor(model, (p + "/mlp/c_proj/w").c_str());
            l.c_mlp_proj_b  = require_tensor(model, (p + "/mlp/c_proj/b").c_str());
        }

        // Token-major KV slab: per (layer) the cache is [n_embd, n_ctx]
        // rows (one row per cached position), so the per-step append is
        // a contiguous span — required for quantised kv_type (ggml-cpu's
        // dup→quantized path aborts on a non-contiguous dst) and what
        // lets f16/q8_0 shrink the cache (f16 ½, q8_0 ~27% of f32).
        // Fall back to F32 KV if the resolved backend can't run flash
        // attention with the requested quantized/f16 K/V (turbo uses
        // n_head == n_kv_head; head_dim = n_embd / n_head).
        hp.kv_type = chatterbox_resolve_kv_type(model.backend, kv_type,
                                                hp.n_embd / hp.n_head, hp.n_head, hp.n_head);
        ggml_init_params kv_params = { ggml_tensor_overhead() * 2, nullptr, true };
        model.ctx_kv = ggml_init(kv_params);
        int64_t n_elements = (int64_t) hp.n_embd * hp.n_layer * hp.n_ctx;
        model.memory_k = ggml_new_tensor_1d(model.ctx_kv, hp.kv_type, n_elements);
        model.memory_v = ggml_new_tensor_1d(model.ctx_kv, hp.kv_type, n_elements);
        model.buffer_kv = ggml_backend_alloc_ctx_tensors(model.ctx_kv, model.backend);

        if (g_log_verbose) fprintf(stderr, "%s: ctx=%d embd=%d layers=%d heads=%d text_vocab=%d speech_vocab=%d cond_prompt=%d\n",
                __func__, hp.n_ctx, hp.n_embd, hp.n_layer, hp.n_head,
                hp.n_text_vocab, hp.n_speech_vocab, hp.cond_prompt_len);
        if (g_log_verbose) fprintf(stderr, "%s: weights=%.2f MB  KV=%.2f MB\n", __func__,
                ggml_backend_buffer_get_size(model.buffer_w) / (1024.0*1024.0),
                ggml_backend_buffer_get_size(model.buffer_kv) / (1024.0*1024.0));

        // Read embedded tokenizer arrays if present (added by converter-v2).
        {
            const int64_t tok_kid = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
            const int64_t mer_kid = gguf_find_key(gguf_ctx, "tokenizer.ggml.merges");
            if (tok_kid >= 0 && mer_kid >= 0) {
                const size_t n_tok = gguf_get_arr_n(gguf_ctx, tok_kid);
                const size_t n_mer = gguf_get_arr_n(gguf_ctx, mer_kid);
                model.tok_tokens.reserve(n_tok);
                for (size_t i = 0; i < n_tok; ++i) {
                    model.tok_tokens.emplace_back(gguf_get_arr_str(gguf_ctx, tok_kid, i));
                }
                model.tok_merges.reserve(n_mer);
                for (size_t i = 0; i < n_mer; ++i) {
                    model.tok_merges.emplace_back(gguf_get_arr_str(gguf_ctx, mer_kid, i));
                }
                if (g_log_verbose) fprintf(stderr, "%s: tokenizer embedded (%zu tokens, %zu merges)\n",
                        __func__, n_tok, n_mer);
            } else {
                fprintf(stderr, "%s: no embedded tokenizer; --tokenizer-dir will be required for --text\n",
                        __func__);
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "%s: %s\n", __func__, e.what());
        gguf_free(gguf_ctx); if (tmp_ctx) ggml_free(tmp_ctx);
        return false;
    }
    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
    return true;
}

// --------------------------------------------------------------------------
// GPT-2 transformer core (shared by prompt and step graphs)
// --------------------------------------------------------------------------

static ggml_tensor * build_transformer_core(
    ggml_context * ctx, ggml_cgraph * gf,
    const chatterbox_model & model,
    ggml_tensor * inpL, int n_past, int N) {

    const auto & hp = model.hparams;
    const int n_embd = hp.n_embd, n_head = hp.n_head, n_layer = hp.n_layer, n_ctx = hp.n_ctx;

    const int HD = n_embd / n_head;
    const int64_t L = n_past + N;

    // KV cache layout: token-major — each layer is [n_embd, n_ctx]: one
    // ggml_row_size(kv_type, n_embd) row per cached position, heads
    // packed [HD-head0 ‖ HD-head1 ‖ ...] inside the row (the natural
    // order of the fused-QKV projection output, so the append needs no
    // permute).  Two wins over the older head-major [HD, n_ctx, n_head]
    // slab: the per-step append at position n_past is a CONTIGUOUS span
    // (which is what allows a quantised kv_type — ggml-cpu's
    // dup→quantized path aborts on non-contiguous dst), and
    // ggml_flash_attn_ext still reads the [HD, L, n_head] slice with
    // plain strides (pos stride = one token row, head stride = one
    // HD-row inside it; llama.cpp's KV views have the same shape).
    const ggml_type kvt = model.memory_k->type;
    const size_t kv_tok_row     = ggml_row_size(kvt, n_embd);  // bytes per cached position
    const size_t kv_head_row    = ggml_row_size(kvt, HD);      // bytes per head inside a row
    const size_t kv_layer_bytes = (size_t) n_ctx * kv_tok_row;

    // Causal attention mask for flash_attn_ext.  Shape [n_kv, N] broadcast
    // over heads, F16 (Metal FA requires F16 masks; CPU / CUDA / Vulkan all
    // accept that too).  For the single-step path (N=1) every KV position is
    // allowed, so no mask is needed and we pass nullptr to FA.
    ggml_tensor * kq_mask = nullptr;
    if (N > 1) {
        kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, L, N);
        ggml_set_name(kq_mask, "kq_mask");
        ggml_set_input(kq_mask);
    }

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * cur;
        cur = ggml_norm(ctx, inpL, hp.eps);
        cur = ggml_add(ctx, ggml_mul(ctx, cur, model.layers[il].ln_1_g), model.layers[il].ln_1_b);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, model.layers[il].c_attn_attn_w, cur), model.layers[il].c_attn_attn_b);

        // View Q / K / V directly inside the fused QKV projection output as
        // [HD, N, n_head] with explicit strides.  FA consumes this non-contig
        // view directly: nb0=4 (fast HD axis), nb1=3*n_embd*4 (stride between
        // columns of the original cur), nb2=HD*4 (stride between heads within
        // a column).  No cont_3d + permute + cont sequence per layer.
        const size_t qkv_col_stride  = cur->nb[1];                // 3 * n_embd * sizeof(float)
        const size_t qkv_head_stride = (size_t) HD * sizeof(float);

        ggml_tensor * Q = ggml_view_3d(ctx, cur, HD, N, n_head,
            qkv_col_stride,
            qkv_head_stride,
            0);  // Qcur slot
        // K/V slots viewed as [n_embd, N]: one row per position, heads
        // packed inside — already the token-major cache row layout, so
        // the append below is a single contiguous-destination copy.
        ggml_tensor * Kcur = ggml_view_2d(ctx, cur, n_embd, N,
            qkv_col_stride,
            (size_t) n_embd * sizeof(float));  // Kcur slot
        ggml_tensor * Vcur = ggml_view_2d(ctx, cur, n_embd, N,
            qkv_col_stride,
            (size_t) 2 * n_embd * sizeof(float));  // Vcur slot

        // KV cache append: positions [n_past, n_past+N) are consecutive
        // token rows, so the destination view is contiguous and ggml_cpy
        // converts/quantises f32 → kv_type on write.  One kernel launch
        // per tensor.
        const size_t layer_off = (size_t) il * kv_layer_bytes;
        {
            ggml_tensor * k_dst = ggml_view_2d(ctx, model.memory_k,
                n_embd, N,
                kv_tok_row,
                layer_off + (size_t) n_past * kv_tok_row);
            ggml_tensor * v_dst = ggml_view_2d(ctx, model.memory_v,
                n_embd, N,
                kv_tok_row,
                layer_off + (size_t) n_past * kv_tok_row);

            ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur, k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur, v_dst));
        }

        ggml_tensor * K = ggml_view_3d(ctx, model.memory_k,
            HD, L, n_head,
            kv_tok_row,
            kv_head_row,
            layer_off);
        ggml_tensor * V = ggml_view_3d(ctx, model.memory_v,
            HD, L, n_head,
            kv_tok_row,
            kv_head_row,
            layer_off);

        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K, V, kq_mask,
            1.0f / std::sqrt((float) HD), 0.0f, 0.0f);
        // attn: [HD, n_head, N, 1] contiguous -> [n_embd, N]
        cur = ggml_reshape_2d(ctx, attn, n_embd, N);
        cur = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, model.layers[il].c_attn_proj_w, cur), model.layers[il].c_attn_proj_b), inpL);

        ggml_tensor * inpFF = cur;
        cur = ggml_norm(ctx, inpFF, hp.eps);
        cur = ggml_add(ctx, ggml_mul(ctx, cur, model.layers[il].ln_2_g), model.layers[il].ln_2_b);
        cur = ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, model.layers[il].c_mlp_fc_w, cur), model.layers[il].c_mlp_fc_b));
        cur = ggml_add(ctx, ggml_mul_mat(ctx, model.layers[il].c_mlp_proj_w, cur), model.layers[il].c_mlp_proj_b);

        inpL = ggml_add(ctx, cur, inpFF);
    }

    inpL = ggml_norm(ctx, inpL, hp.eps);
    inpL = ggml_add(ctx, ggml_mul(ctx, inpL, model.ln_f_g), model.ln_f_b);

    ggml_tensor * logits = ggml_add(ctx, ggml_mul_mat(ctx, model.speech_head, inpL), model.speech_head_bias);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return logits;
}

// --------------------------------------------------------------------------
// Graph builders
// --------------------------------------------------------------------------

static ggml_cgraph * build_prompt_graph(const chatterbox_model & model, int n_text_tokens) {
    const int N = 1 + model.hparams.cond_prompt_len + n_text_tokens + 1;
    static size_t buf_size = ggml_tensor_overhead()*CHBX_MAX_NODES + ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(text_tokens, "text_tokens"); ggml_set_input(text_tokens);
    ggml_tensor * start_token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(start_token, "speech_token"); ggml_set_input(start_token);
    ggml_tensor * position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(position, "position"); ggml_set_input(position);

    ggml_tensor * spkr = ggml_add(ctx, ggml_mul_mat(ctx, model.cond_spkr_w, model.builtin_speaker_emb), model.cond_spkr_b);
    ggml_tensor * cond = ggml_get_rows(ctx, model.speech_emb, model.builtin_cond_prompt_tokens);
    ggml_tensor * temb = ggml_get_rows(ctx, model.text_emb, text_tokens);
    ggml_tensor * semb = ggml_get_rows(ctx, model.speech_emb, start_token);

    ggml_tensor * inp = ggml_concat(ctx, spkr, cond, 1);
    inp = ggml_concat(ctx, inp, temb, 1);
    inp = ggml_concat(ctx, inp, semb, 1);
    inp = ggml_add(ctx, inp, ggml_get_rows(ctx, model.wpe, position));

    build_transformer_core(ctx, gf, model, inp, 0, N);
    ggml_free(ctx);
    return gf;
}

static ggml_cgraph * build_step_graph(const chatterbox_model & model, int n_past) {
    static size_t buf_size = ggml_tensor_overhead()*CHBX_MAX_NODES + ggml_graph_overhead_custom(CHBX_MAX_NODES, false);
    thread_local std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, CHBX_MAX_NODES, false);

    ggml_tensor * speech_token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speech_token, "speech_token"); ggml_set_input(speech_token);
    ggml_tensor * position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(position, "position"); ggml_set_input(position);

    ggml_tensor * inp = ggml_add(ctx,
        ggml_get_rows(ctx, model.speech_emb, speech_token),
        ggml_get_rows(ctx, model.wpe, position));

    build_transformer_core(ctx, gf, model, inp, n_past, 1);
    ggml_free(ctx);
    return gf;
}

// --------------------------------------------------------------------------
// Evaluation
// --------------------------------------------------------------------------

bool eval_prompt(
    const chatterbox_model & model, ggml_gallocr_t allocr, int n_threads,
    const std::vector<int32_t> & text_tokens, std::vector<float> & logits_out, int & prompt_len) {

    prompt_len = 1 + model.hparams.cond_prompt_len + (int)text_tokens.size() + 1;
    if (prompt_len > model.hparams.n_ctx) {
        fprintf(stderr, "%s: prompt %d exceeds context %d\n", __func__, prompt_len, model.hparams.n_ctx);
        return false;
    }
    ggml_cgraph * gf = build_prompt_graph(model, (int)text_tokens.size());
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "text_tokens"), text_tokens.data(), 0, text_tokens.size()*sizeof(int32_t));
    int32_t st = model.hparams.start_speech_token;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_token"), &st, 0, sizeof(st));
    std::vector<int32_t> pos(prompt_len);
    for (int i = 0; i < prompt_len; ++i) pos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "position"), pos.data(), 0, pos.size()*sizeof(int32_t));

    {
        const int N = prompt_len;
        ggml_tensor * kq_mask = ggml_graph_get_tensor(gf, "kq_mask");
        if (kq_mask) {
            // Metal FA requires F16 masks; other backends accept F16 too.
            const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t ninf_h = ggml_fp32_to_fp16(-INFINITY);
            std::vector<ggml_fp16_t> mask((size_t)N * N, zero_h);
            for (int q = 0; q < N; ++q) {
                for (int k = 0; k < N; ++k) {
                    if (k > q) mask[(size_t)q * N + k] = ninf_h;
                }
            }
            ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size()*sizeof(ggml_fp16_t));
        }
    }

    // Registry-routed n_threads (no-op on non-CPU backends); see
    // src/t3_mtl.cpp for the GGML_BACKEND_DL=ON unresolvable-symbol
    // rationale.
    ::tts_cpp::detail::backend_set_n_threads(model.backend, n_threads);
    ggml_backend_graph_compute(model.backend, gf);

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    logits_out.resize(model.hparams.n_speech_vocab);
    ggml_backend_tensor_get(logits, logits_out.data(),
        (size_t)model.hparams.n_speech_vocab*(prompt_len-1)*sizeof(float),
        (size_t)model.hparams.n_speech_vocab*sizeof(float));
    return true;
}

bool eval_step(
    const chatterbox_model & model, ggml_gallocr_t allocr, int n_threads,
    int n_past, int32_t token, std::vector<float> & logits_out) {

    ggml_cgraph * gf = build_step_graph(model, n_past);
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "speech_token"), &token, 0, sizeof(token));
    int32_t position = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "position"), &position, 0, sizeof(position));

    // Registry-routed n_threads; see src/t3_mtl.cpp for rationale.
    ::tts_cpp::detail::backend_set_n_threads(model.backend, n_threads);
    ggml_backend_graph_compute(model.backend, gf);

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    logits_out.resize(model.hparams.n_speech_vocab);
    ggml_backend_tensor_get(logits, logits_out.data(), 0, (size_t)model.hparams.n_speech_vocab*sizeof(float));
    return true;
}

// --------------------------------------------------------------------------
// Sampling
// --------------------------------------------------------------------------

// Matches HuggingFace LogitsProcessorList order used in inference_turbo:
//   1. TemperatureLogitsWarper   (if temp > 0 and temp != 1)
//   2. TopKLogitsWarper          (if top_k > 0)
//   3. TopPLogitsWarper          (if top_p < 1)
//   4. RepetitionPenaltyLogitsProcessor (if penalty != 1)
// Then softmax + multinomial.
//
// The chatterbox_sampling_params-taking version is the canonical one and is
// also used from src/chatterbox_engine.cpp.  The cli_params-taking wrapper
// below is preserved so the existing CLI call sites keep working unchanged.
int32_t sample_next_token_ex(
    const std::vector<float> & logits,
    const std::vector<int32_t> & generated,
    const chatterbox_sampling_params & params,
    std::mt19937 & rng) {

    const int n = (int)logits.size();
    std::vector<float> scores(logits.begin(), logits.end());

    if (params.temp > 0.0f && params.temp != 1.0f) {
        float inv_t = 1.0f / params.temp;
        for (float & s : scores) s *= inv_t;
    }

    if (params.top_k > 0 && params.top_k < n) {
        std::vector<float> tmp(scores);
        std::nth_element(tmp.begin(), tmp.begin() + params.top_k, tmp.end(), std::greater<float>());
        float threshold = tmp[params.top_k];
        int kept = 0;
        for (float s : scores) if (s > threshold) ++kept;
        if (kept < params.top_k) threshold -= 1e-10f;
        for (float & s : scores) if (s <= threshold) s = -INFINITY;
    }

    if (params.top_p < 1.0f) {
        struct IS { int idx; float s; };
        std::vector<IS> sorted;
        sorted.reserve(n);
        for (int i = 0; i < n; ++i) if (scores[i] != -INFINITY) sorted.push_back({i, scores[i]});
        std::sort(sorted.begin(), sorted.end(), [](const IS& a, const IS& b){ return a.s > b.s; });

        float mx = sorted.empty() ? 0.0f : sorted[0].s;
        std::vector<float> probs(sorted.size());
        float psum = 0;
        for (size_t i = 0; i < sorted.size(); ++i) { probs[i] = std::exp(sorted[i].s - mx); psum += probs[i]; }
        for (float & p : probs) p /= psum;

        float cum = 0;
        std::set<int> keep_set;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cum += probs[i];
            keep_set.insert(sorted[i].idx);
            if (cum >= params.top_p) break;
        }
        if (keep_set.empty() && !sorted.empty()) keep_set.insert(sorted[0].idx);
        for (int i = 0; i < n; ++i) if (keep_set.find(i) == keep_set.end()) scores[i] = -INFINITY;
    }

    if (params.repeat_penalty != 1.0f && !generated.empty()) {
        std::set<int32_t> seen(generated.begin(), generated.end());
        for (int32_t t : seen) {
            if (t < 0 || t >= n) continue;
            if (scores[t] == -INFINITY) continue;
            scores[t] = scores[t] > 0 ? scores[t] / params.repeat_penalty : scores[t] * params.repeat_penalty;
        }
    }

    float mx = -INFINITY;
    for (float s : scores) if (s != -INFINITY) mx = std::max(mx, s);

    std::vector<float> probs(n);
    float psum = 0;
    for (int i = 0; i < n; ++i) {
        probs[i] = (scores[i] == -INFINITY) ? 0.0f : std::exp(scores[i] - mx);
        psum += probs[i];
    }
    if (psum == 0.0f) return 0;
    for (float & p : probs) p /= psum;

    if (params.temp <= 0.0f) {
        return (int32_t)std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    }

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// Log filter: when --verbose is off, drop everything below ERROR.  This
// silences ggml-metal's per-kernel "compiling pipeline" spam, ggml_vulkan's
// device enumeration, ggml-metal's "tensor API disabled" one-liner, etc. —
// none of which a non-debugging user cares about.  Errors still go through
// so real failures are never hidden.
// (g_log_verbose is declared near init_backend; see above.)
void chatterbox_log_cb(ggml_log_level level, const char * text, void * /*ud*/) {
    if (g_log_verbose || level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

} // namespace tts_cpp::chatterbox::detail

