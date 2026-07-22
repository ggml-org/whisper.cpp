#include "audiogen-cpp/acestep/engine.h"

#include "audiogen-cpp/acestep/vae.h"

#include "acestep/bpe_tokenizer.h"
#include "acestep/cond_ggml.h"
#include "acestep/detok_ggml.h"
#include "acestep/dit_ggml.h"
#include "acestep/lm_ggml.h"
#include "acestep/lm_pipeline.h"
#include "acestep/philox.h"
#include "acestep/textenc_ggml.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <thread>

// ACE-Step end-to-end music engine. Wires the ported stages behind
// tts_cpp::acestep::Engine::generate():
//   caption+lyrics -> LM (audio codes) -> FSQ detok (context latents)
//                  -> text-enc + cond-enc (cross-attn states)
//                  -> DiT flow-matching sample -> VAE decode -> stereo 48 kHz.
//
// Turbo text2music path: single sequence, no CFG, Phase-2 codes only (Phase-1
// CoT/metadata generation + metadata FSM are a follow-up). Matches acestep.cpp's
// synth wiring: because LM audio codes are present, the DiT uses the COVER
// instruction and detokenized codes as the conditioning context.

namespace tts_cpp::acestep {

// --- synth-parity debug hooks (compiled out by default) -----------
// Isolate DiT vs VAE by injecting acestep.cpp --dump tensors and dumping ours.
// Dump .bin format: 3x int32 header [ndim, d0, d1] + float32 payload.
// Build with -DACESTEP_PARITY_DEBUG to enable; otherwise these are absent from
// the production generate() path.
#ifdef ACESTEP_PARITY_DEBUG
namespace {
bool dbg_load_dump(const char * path, std::vector<float> & dst) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[acestep-dbg] cannot open %s\n", path); return false; }
    int32_t hdr[3] = {0, 0, 0};
    if (fread(hdr, sizeof(int32_t), 3, f) != 3) { fclose(f); return false; }
    const int    ndim = hdr[0];
    const size_t n    = (size_t) hdr[1] * (size_t) ((ndim >= 2) ? hdr[2] : 1);
    if (n != dst.size()) {
        fprintf(stderr, "[acestep-dbg] size mismatch %s: dump=%zu ours=%zu\n", path, n, dst.size());
        fclose(f);
        return false;
    }
    const size_t got = fread(dst.data(), sizeof(float), n, f);
    fclose(f);
    fprintf(stderr, "[acestep-dbg] injected %s (%zu floats)\n", path, got);
    return got == n;
}

void dbg_write_dump(const std::string & path, const std::vector<float> & src, int d0, int d1) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "[acestep-dbg] cannot write %s\n", path.c_str()); return; }
    int32_t hdr[3] = {2, d0, d1};
    fwrite(hdr, sizeof(int32_t), 3, f);
    fwrite(src.data(), sizeof(float), src.size(), f);
    fclose(f);
    fprintf(stderr, "[acestep-dbg] wrote %s ([%d, %d], %zu floats)\n", path.c_str(), d0, d1, src.size());
}

const char * dbg_env(const char * k) {
    const char * v = getenv(k);
    return (v && *v) ? v : nullptr;
}
}  // namespace
#endif  // ACESTEP_PARITY_DEBUG

// DiT instruction headers (task-types.h). COVER is used whenever LM codes exist.
static const char * DIT_INSTR_COVER = "Generate audio semantic tokens based on the given conditions:";

namespace fs = std::filesystem;

struct Engine::Impl {
    EngineOptions opts;

    ggml_backend_t backend     = nullptr;  // primary backend (GPU or CPU) for textenc/cond/DiT
    ggml_backend_t backend_cpu = nullptr;  // CPU backend for detok (uses a CPY variant Metal lacks); null when primary is CPU
    ggml_backend_t backend_lm  = nullptr;  // backend the LM loads on (CPU when on GPU; see create)
    ggml_backend_t backend_enc = nullptr;  // backend for textenc + cond (CPU when on GPU; see create)

    TextEncModel * textenc = nullptr;
    LMModel *      lm      = nullptr;
    CondModel *    cond    = nullptr;
    DetokModel *   detok   = nullptr;
    DitModel *     dit     = nullptr;

    std::unique_ptr<Vae> vae;

    BpeTokenizer bpe_lm;    // LM vocab (+ audio codes) — Phase-2 prompt
    BpeTokenizer bpe_text;  // text-encoder vocab — DiT prompt / lyric lookup

    mutable std::atomic<bool> cancel_flag{ false };

    ~Impl() {
        if (dit) dit_model_free(dit);
        if (detok) detok_model_free(detok);
        if (cond) cond_model_free(cond);
        if (lm) lm_model_free(lm);
        if (textenc) textenc_model_free(textenc);
        vae.reset();
        if (backend_cpu && backend_cpu != backend) ggml_backend_free(backend_cpu);
        if (backend) ggml_backend_free(backend);
    }
};

// ------------------------------------------------------------ path resolution
static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

// Classify the four ACE-Step GGUFs in models_dir by anchoring on their known
// filename stems (Qwen3-Embedding / acestep-*-lm / vae / acestep-v15-{turbo,sft}).
// Explicit paths in EngineOptions always win over the scan. The stems are chosen
// so no bare short token (like "lm") can match an unrelated file; the most
// specific stages (embedding, vae) are tested before the shorter lm/dit stems.
static void resolve_paths(EngineOptions & o) {
    if (o.models_dir.empty()) return;
    std::error_code ec;
    for (auto & e : fs::directory_iterator(o.models_dir, ec)) {
        if (!e.is_regular_file()) continue;
        std::string path = e.path().string();
        std::string name = to_lower(e.path().filename().string());
        if (name.size() < 5 || name.substr(name.size() - 5) != ".gguf") continue;
        auto has = [&](const char * s) { return name.find(s) != std::string::npos; };
        if (has("embedding") || has("text-enc") || has("textenc")) {
            if (o.text_enc_model_path.empty()) o.text_enc_model_path = path;
        } else if (has("vae")) {
            if (o.vae_model_path.empty()) o.vae_model_path = path;
        } else if (has("-lm") || has("lm-") || has("_lm") || has("ace-lm") || has("5hz-lm")) {
            if (o.lm_model_path.empty()) o.lm_model_path = path;
        } else if (has("turbo") || has("dit") || has("v15") || has("sft")) {
            if (o.dit_model_path.empty()) o.dit_model_path = path;
        }
    }
}

// ------------------------------------------------------------ construction
Engine::Engine() : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::create(const EngineOptions & opts_in) {
    EngineOptions opts = opts_in;
    resolve_paths(opts);

    auto need = [&](const std::string & p, const char * what) {
        if (p.empty()) throw std::runtime_error(std::string("acestep engine: missing ") + what + " GGUF");
    };
    need(opts.text_enc_model_path, "text-encoder");
    need(opts.lm_model_path, "LM");
    need(opts.dit_model_path, "DiT");
    need(opts.vae_model_path, "VAE");

    std::unique_ptr<Engine> eng(new Engine());
    Impl *                  m = eng->impl_.get();
    m->opts                   = opts;

    const bool v = opts.verbose;

    int nth = opts.n_threads > 0 ? opts.n_threads : (int) std::thread::hardware_concurrency();
    if (nth < 1) nth = 4;

    // Backend for the ggml stages (text-encoder, LM, cond/detok, DiT). These use
    // only standard ggml ops, so a GPU backend (Metal on Apple, CUDA/Vulkan
    // elsewhere) can run them; opts.n_gpu_layers > 0 opts in. The VAE gets its
    // own dedicated backend (see Vae::load) and also follows n_gpu_layers now
    // that its snake / col2im_1d ops have Metal kernels in the ggml-speech fork.
    // Falls back to CPU when no GPU backend is registered/available.
    bool on_gpu = false;
    if (opts.n_gpu_layers > 0) {
        m->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        on_gpu     = (m->backend != nullptr);
        if (!on_gpu && v) fprintf(stderr, "[acestep-engine] GPU requested but no GPU backend available; using CPU\n");
    }
    if (!m->backend) m->backend = ggml_backend_cpu_init();
    if (!m->backend) throw std::runtime_error("acestep engine: backend init failed");
    if (on_gpu) {
        if (v) fprintf(stderr, "[acestep-engine] DiT/VAE on GPU backend: %s\n", ggml_backend_name(m->backend));
        // The FSQ detokenizer emits a CPY variant our Metal backend lacks a
        // kernel for, so it always runs on a dedicated CPU backend.
        m->backend_cpu = ggml_backend_cpu_init();
        if (!m->backend_cpu) throw std::runtime_error("acestep engine: CPU backend init failed");
        ggml_backend_cpu_set_n_threads(m->backend_cpu, nth);
    } else {
        ggml_backend_cpu_set_n_threads(m->backend, nth);
        m->backend_cpu = m->backend;  // single CPU backend serves every stage
    }

    // Backend layout when a GPU is active. Almost everything runs on the GPU, but
    // the autoregressive LM is the exception: on iOS A-series Metal it produces
    // empty/garbage logits -> the sampler yields zero audio codes ("LM produced
    // no audio codes"), while the SAME weights decode correctly on CPU. This is a
    // NUMERICAL issue, not memory (unified RAM doesn't help), confirmed by testing
    // the full-GPU path on device. So the LM defaults to the CPU backend whenever
    // a GPU is active; the one-shot text/cond encoders stay on the GPU with the
    // DiT + VAE. Env escape hatches (no rebuild needed):
    //   ACESTEP_LM_GPU=1        -> force the LM back onto the GPU (desktop bench)
    //   ACESTEP_ENCODERS_CPU=1  -> move the encoders to the CPU (trim wired mem)
    ggml_backend_t enc_backend = m->backend;
    ggml_backend_t lm_backend  = m->backend;
    if (on_gpu) {
        lm_backend = m->backend_cpu;  // A-series Metal LM is numerically broken
        if (std::getenv("ACESTEP_LM_GPU"))       lm_backend  = m->backend;
        if (std::getenv("ACESTEP_ENCODERS_CPU")) enc_backend = m->backend_cpu;
        if (v) fprintf(stderr, "[acestep-engine] backends: enc=%s lm=%s dit/vae=%s\n",
                       ggml_backend_name(enc_backend), ggml_backend_name(lm_backend), ggml_backend_name(m->backend));
    }
    m->backend_enc = enc_backend;
    m->backend_lm  = lm_backend;

    m->textenc = textenc_model_load(opts.text_enc_model_path, enc_backend, v);
    if (!m->textenc) throw std::runtime_error("acestep engine: text-encoder load failed");

    // 2 KV sets: cond + uncond for classifier-free guidance on Phase-2 codes.
    m->lm = lm_model_load(opts.lm_model_path, lm_backend, /*max_seq_len=*/2048, v, /*n_kv_sets=*/2);
    if (!m->lm) throw std::runtime_error("acestep engine: LM load failed");

    m->cond = cond_model_load(opts.dit_model_path, enc_backend, v);
    if (!m->cond) throw std::runtime_error("acestep engine: cond-encoder load failed");

    m->detok = detok_model_load(opts.dit_model_path, m->backend_cpu, v);
    if (!m->detok) throw std::runtime_error("acestep engine: FSQ detokenizer load failed");

    m->dit = dit_model_load(opts.dit_model_path, m->backend, v);
    if (!m->dit) throw std::runtime_error("acestep engine: DiT load failed");

    VaeOptions vo;
    vo.verbose      = v;
    vo.with_encoder = false;  // generation only decodes
    vo.n_threads    = nth;
    vo.n_gpu_layers = opts.n_gpu_layers;  // snake / col2im_1d now have Metal kernels
    // Debug hook: force the VAE backend independently of the other stages so the
    // decode can be compared CPU-vs-GPU on an identical latent (ACESTEP_VAE_GPU=1
    // -> GPU, =0 -> CPU). Leaves the LM/DiT backend untouched (=deterministic).
    if (const char * e = std::getenv("ACESTEP_VAE_GPU")) {
        vo.n_gpu_layers = (e[0] == '1') ? 99 : 0;
    }
    m->vae          = Vae::load(opts.vae_model_path, vo);
    if (!m->vae) throw std::runtime_error("acestep engine: VAE load failed");

    // Tokenizers: LM prompt uses the LM vocab; DiT text prompt + lyric lookup use
    // the text-encoder vocab. Fall back to the LM tokenizer if the text-encoder
    // GGUF has no tokenizer KV (same Qwen text vocab in the shared range).
    if (!bpe_load_from_gguf(m->bpe_lm, opts.lm_model_path))
        throw std::runtime_error("acestep engine: LM tokenizer load failed");
    if (!bpe_load_from_gguf(m->bpe_text, opts.text_enc_model_path)) {
        if (v) fprintf(stderr, "[acestep-engine] text-encoder has no tokenizer KV; reusing LM tokenizer\n");
        m->bpe_text = m->bpe_lm;
    }

    if (v) fprintf(stderr, "[acestep-engine] ready (threads=%d)\n", nth);
    return eng;
}

// ------------------------------------------------------------ generate
static std::string build_metas(int bpm, const std::string & timesig, const std::string & keyscale, float dur) {
    char bpm_b[16] = "N/A";
    if (bpm > 0) snprintf(bpm_b, sizeof(bpm_b), "%d", bpm);
    const char * ts = timesig.empty() ? "N/A" : timesig.c_str();
    const char * ks = keyscale.empty() ? "N/A" : keyscale.c_str();
    char         buf[512];
    snprintf(buf, sizeof(buf), "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n", bpm_b, ts,
             ks, (int) dur);
    return buf;
}

GenerateResult Engine::generate(const GenerateParams & params, const ProgressFn & progress) const {
    Impl *         m = impl_.get();
    GenerateResult result;
    result.sample_rate = m->vae->sample_rate();
    result.channels    = 2;
    m->cancel_flag.store(false);

    auto report = [&](const char * stage, int step, int total) -> bool {
        if (progress && !progress(stage, step, total)) { m->cancel_flag.store(true); return false; }
        return !m->cancel_flag.load();
    };

    // Low-memory mode: free each stage model as soon as it is no longer needed so
    // the peak resident set stays small enough for memory-constrained devices
    // (e.g. iOS, where the whole system shares ~8 GB of unified RAM and jetsam
    // kills the app on a system-wide page shortage — Metal weight buffers are
    // wired/non-pageable). The LM (Phase 1/2) and the text-encoder + cond-encoder
    // are only used up front (codes + DiT conditioning); freeing them before the
    // DiT/VAE run frees ~2 GB. They are lazily reloaded at the top of the next
    // generate(). Opt out with ACESTEP_KEEP_STAGES=1 (e.g. servers that generate
    // back-to-back and prefer to avoid the per-call reload).
    const bool keep_stages = [] {
        const char * e = std::getenv("ACESTEP_KEEP_STAGES");
        return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T' || e[0] == 'y' || e[0] == 'Y');
    }();
    const bool low_mem = !keep_stages;

    // Reload any stage model a previous low-mem generation freed. On the first
    // generate() after create() everything is still resident, so this is a no-op.
    {
        const bool vv = m->opts.verbose;
        if (!m->textenc) {
            if (vv) fprintf(stderr, "[acestep-engine] reloading text-encoder\n");
            m->textenc = textenc_model_load(m->opts.text_enc_model_path, m->backend_enc, vv);
            if (!m->textenc) throw std::runtime_error("acestep engine: text-encoder reload failed");
        }
        if (!m->lm) {
            if (vv) fprintf(stderr, "[acestep-engine] reloading LM\n");
            m->lm = lm_model_load(m->opts.lm_model_path, m->backend_lm, /*max_seq_len=*/2048, vv, /*n_kv_sets=*/2);
            if (!m->lm) throw std::runtime_error("acestep engine: LM reload failed");
        }
        if (!m->cond) {
            if (vv) fprintf(stderr, "[acestep-engine] reloading cond-encoder\n");
            m->cond = cond_model_load(m->opts.dit_model_path, m->backend_enc, vv);
            if (!m->cond) throw std::runtime_error("acestep engine: cond-encoder reload failed");
        }
    }

    long long seed = params.seed;
    if (seed < 0) { std::random_device rd; seed = (long long) rd(); }

    const std::string language = params.vocal_language.empty() ? "en" : params.vocal_language;

    // ---- 1. LM: caption+lyrics(+metas) -> audio semantic codes (Phase 2) ----
    AcePrompt prompt;
    prompt.caption        = params.caption;
    prompt.lyrics         = params.lyrics.empty() ? "[Instrumental]" : params.lyrics;
    prompt.duration       = params.duration;
    prompt.bpm            = params.bpm;
    prompt.keyscale       = params.keyscale;
    prompt.timesignature  = params.timesignature;
    prompt.vocal_language = language;

    if (!report("lm", 0, 1)) return result;
    std::vector<int> codes;
    if (!params.audio_codes.empty()) {
        codes = params.audio_codes;  // parity / cached codes: skip the LM
        if (m->opts.verbose) fprintf(stderr, "[acestep-engine] using %zu pre-supplied codes (LM skipped)\n", codes.size());
    } else {
        LmSampleParams lp;
        lp.temperature = params.lm_temperature;
        lp.top_p       = params.lm_top_p;
        lp.top_k       = params.lm_top_k;
        lp.cfg_scale   = params.lm_cfg_scale;
        // The LM sampler RNG is std::mt19937 (32-bit seed), so truncating here
        // is intentional and lossless for its purpose; only the DiT noise needs
        // the full int64 seed (Philox, torch.randn parity), which it gets below.
        lp.seed        = (uint32_t) seed;
        lp.verbose     = m->opts.verbose;
        // Surface incremental Phase-2 progress (the longest stage) so the UI bar
        // advances while the LM writes audio codes, not only during the DiT.
        lp.on_step     = [&](int cur, int total) { report("lm", cur, total); };

        // Phase 1: fill missing metadata (bpm/keyscale/duration/timesignature)
        // from the caption via the FSM, so a bare caption matches the CLI.
        const bool has_all_metas =
            prompt.bpm > 0 && prompt.duration > 0 && !prompt.keyscale.empty() && !prompt.timesignature.empty();
        if (params.lm_phase1 && !has_all_metas) {
            LmSampleParams p1 = lp;
            p1.max_new_tokens = 0;  // FSM stops at </think>
            if (!lm_generate_phase1(m->lm, m->bpe_lm, prompt, p1))
                fprintf(stderr, "[acestep-engine] Phase 1 failed; falling back to provided/default metas\n");
        }

        if (!lm_generate_codes(m->lm, m->bpe_lm, prompt, lp, codes) || codes.empty())
            throw std::runtime_error("acestep engine: LM produced no audio codes");
    }
    if (!report("lm", 1, 1)) return result;

    // The LM is done (codes in hand). Free it now so its weights + KV (~1.1 GB)
    // are not resident during the DiT/VAE. Reloaded on the next generate().
    if (low_mem && m->lm) {
        if (m->opts.verbose) fprintf(stderr, "[acestep-engine] freeing LM (low-mem)\n");
        lm_model_free(m->lm);
        m->lm = nullptr;
    }

    // ---- 2. FSQ detok: codes -> context latents [64, T_25Hz] ----
    const int          T_5Hz  = (int) codes.size();
    const int          T_25Hz = T_5Hz * 5;
    std::vector<float> detok_latent((size_t) 64 * T_25Hz);
    if (detok_model_decode(m->detok, codes.data(), T_5Hz, detok_latent.data()) != T_25Hz)
        throw std::runtime_error("acestep engine: FSQ detokenizer failed");

    // ---- 3. context [ctx_ch=128, T] = [detok latent[64] | mask[64]=1] ----
    const DitConfig & dc     = dit_model_config(m->dit);
    const int         Oc     = dc.out_channels;            // 64
    const int         ctx_ch = dc.in_channels - Oc;        // 128
    const int         patch  = dc.patch_size;              // 2
    int               T      = ((T_25Hz + patch - 1) / patch) * patch;

    std::vector<float> context((size_t) ctx_ch * T, 0.0f);
    // Padded frames in [T_25Hz, T) are filled with the silence latent (not left
    // at zero) and the chunk mask stays 1.0 for every frame, matching
    // acestep.cpp (pipeline-synth-ops.cpp: "decoded latents then silence,
    // mask = 1.0 (training distribution)"). Zeroing the mask on the tail would
    // diverge from the reference, which only ever saw 0/1 masks in training.
    const std::vector<float> & sil = cond_model_silence_frame(m->cond);
    const float *              silf = sil.empty() ? nullptr : sil.data();
    for (int t = 0; t < T; t++) {
        float * dst = context.data() + (size_t) t * ctx_ch;
        if (t < T_25Hz) {
            memcpy(dst, detok_latent.data() + (size_t) t * Oc, (size_t) Oc * sizeof(float));
        } else if (silf) {
            memcpy(dst, silf, (size_t) Oc * sizeof(float));  // silence-latent pad
        }
        for (int c = 0; c < Oc; c++) dst[Oc + c] = 1.0f;  // chunk mask (training distribution)
    }

    // ---- 4. text-encoder: prompt -> text_hidden; lyric lookup -> lyric_embed ----
    std::string metas    = build_metas(prompt.bpm, prompt.timesignature, prompt.keyscale, prompt.duration);
    std::string text_str = std::string("# Instruction\n") + DIT_INSTR_COVER + "\n\n# Caption\n" + prompt.caption +
                           "\n\n# Metas\n" + metas + "<|endoftext|>\n";
    std::string lyric_str = std::string("# Languages\n") + language + "\n\n# Lyric\n" + prompt.lyrics + "<|endoftext|>";

    std::vector<int> text_ids  = bpe_encode(m->bpe_text, text_str, /*add_eos=*/true);
    std::vector<int> lyric_ids = bpe_encode(m->bpe_text, lyric_str, /*add_eos=*/true);
    const int        S_text    = (int) text_ids.size();
    const int        S_lyric   = (int) lyric_ids.size();

#ifdef ACESTEP_PARITY_DEBUG
    if (dbg_env("ACESTEP_DUMP_DIR")) {
        fprintf(stderr, "[acestep-dbg] S_text=%d S_lyric=%d (enc_S=%d)\n", S_text, S_lyric, S_text + S_lyric);
        fprintf(stderr, "[acestep-dbg] text_ids[0..8]:");
        for (int i = 0; i < 8 && i < S_text; i++) fprintf(stderr, " %d", text_ids[i]);
        fprintf(stderr, " ... tail:");
        for (int i = std::max(0, S_text - 4); i < S_text; i++) fprintf(stderr, " %d", text_ids[i]);
        fprintf(stderr, "\n[acestep-dbg] lyric_ids[0..8]:");
        for (int i = 0; i < 8 && i < S_lyric; i++) fprintf(stderr, " %d", lyric_ids[i]);
        fprintf(stderr, " ... tail:");
        for (int i = std::max(0, S_lyric - 4); i < S_lyric; i++) fprintf(stderr, " %d", lyric_ids[i]);
        fprintf(stderr, "\n");
    }
#endif

    std::vector<int32_t> text_ids32(text_ids.begin(), text_ids.end());
    std::vector<int32_t> lyric_ids32(lyric_ids.begin(), lyric_ids.end());

    std::vector<float> text_hidden, lyric_embed;
    if (!textenc_model_forward(m->textenc, text_ids32.data(), S_text, text_hidden))
        throw std::runtime_error("acestep engine: text-encoder forward failed");
    if (!textenc_model_embed_lookup(m->textenc, lyric_ids32.data(), S_lyric, lyric_embed))
        throw std::runtime_error("acestep engine: lyric embed lookup failed");

    // ---- 5. cond-encoder: -> enc_hidden [2048, S_total] ----
    // text2music feeds one frame of the silence latent to the timbre encoder so
    // the enc sequence carries the timbre token (packed lyric|timbre|text),
    // matching acestep.cpp. Without it we drop a token and misalign cross-attn.
    const std::vector<float> & silence_frame = cond_model_silence_frame(m->cond);
    const float *              timbre_feats  = silence_frame.empty() ? nullptr : silence_frame.data();
    const int                  timbre_S_ref  = silence_frame.empty() ? 0 : 1;

    std::vector<float> enc_hidden;
    int                enc_S = 0;
    if (!cond_model_forward(m->cond, text_hidden.data(), S_text, lyric_embed.data(), S_lyric, timbre_feats,
                            timbre_S_ref, enc_hidden, &enc_S))
        throw std::runtime_error("acestep engine: cond-encoder forward failed");
    const int H_enc = (int) (enc_hidden.size() / (size_t) enc_S);  // 2048

    // Cross-attention conditioning (enc_hidden) is now materialised on the host,
    // so the text-encoder (~742 MB, GPU) and cond-encoder (~352 MB) are no longer
    // needed. Free them before the DiT/VAE so their (wired) buffers don't count
    // against the device memory ceiling. Reloaded on the next generate().
    if (low_mem) {
        if (m->textenc) {
            if (m->opts.verbose) fprintf(stderr, "[acestep-engine] freeing text-encoder (low-mem)\n");
            textenc_model_free(m->textenc);
            m->textenc = nullptr;
        }
        if (m->cond) {
            if (m->opts.verbose) fprintf(stderr, "[acestep-engine] freeing cond-encoder (low-mem)\n");
            cond_model_free(m->cond);
            m->cond = nullptr;
        }
    }

    // ---- 6. noise [64, T] (Philox, torch.randn parity) ----
    std::vector<float> noise((size_t) Oc * T);
    philox_randn(seed, noise.data(), (int) noise.size(), /*bf16_round=*/true);

#ifdef ACESTEP_PARITY_DEBUG
    // Debug (env-gated): inject acestep.cpp --dump inputs to isolate the DiT graph.
    if (const char * p = dbg_env("ACESTEP_INJECT_NOISE"))   dbg_load_dump(p, noise);
    if (const char * p = dbg_env("ACESTEP_INJECT_CONTEXT")) dbg_load_dump(p, context);
    if (const char * p = dbg_env("ACESTEP_INJECT_ENC"))     dbg_load_dump(p, enc_hidden);
#endif

    // ---- 7. DiT flow-matching sample -> latent [64, T] ----
    // Resolve steps/shift from the model type when the caller left them at auto
    // (0): turbo = 8 steps / shift 3.0, base/sft = 50 steps / shift 1.0.
    const int   n_steps = params.inference_steps > 0 ? params.inference_steps : (dc.is_turbo ? 8 : 50);
    const float shift   = params.shift > 0.0f ? params.shift : (dc.is_turbo ? 3.0f : 1.0f);
    if (m->opts.verbose)
        fprintf(stderr, "[acestep-engine] DiT: turbo=%d steps=%d shift=%.2f T=%d\n", (int) dc.is_turbo, n_steps, shift, T);

    std::vector<float> schedule;
    dit_build_schedule(shift, n_steps, schedule);

    if (!report("dit", 0, n_steps)) return result;
    DitSampleParams sp;
    sp.noise           = noise.data();
    sp.context_latents = context.data();
    sp.enc_hidden      = enc_hidden.data();
    sp.enc_S           = enc_S;
    sp.H_enc           = H_enc;
    sp.T               = T;
    sp.N               = 1;
    sp.schedule        = schedule.data();
    sp.num_steps       = n_steps;
    sp.real_enc_S      = &enc_S;
    // Surface per-step diffusion progress (the long pole) to the caller's
    // ProgressFn; returning false here also honours cooperative cancellation.
    sp.on_step         = [&](int step, int total) { return report("dit", step, total); };

    std::vector<float> latent;
    if (!dit_sample(m->dit, sp, latent)) {
        // dit_sample returns false for BOTH a real compute failure and a
        // cooperative cancel (its on_step -> report() returned false). Tell them
        // apart via cancel_flag: on cancel honour the engine contract (empty pcm,
        // no throw) like the LM and VAE stages; only a genuine failure throws, so
        // a cancelling ProgressFn never raises across the addon boundary.
        if (m->cancel_flag.load()) return result;
        throw std::runtime_error("acestep engine: DiT sample failed");
    }
    if (!report("dit", n_steps, n_steps)) return result;

#ifdef ACESTEP_PARITY_DEBUG
    // Debug (env-gated): dump our DiT output + inputs for parity vs acestep.cpp.
    if (const char * dir = dbg_env("ACESTEP_DUMP_DIR")) {
        const std::string d(dir);
        dbg_write_dump(d + "/our_dit_output.bin", latent, T, Oc);
        dbg_write_dump(d + "/our_noise.bin", noise, T, Oc);
        dbg_write_dump(d + "/our_context.bin", context, T, ctx_ch);
        dbg_write_dump(d + "/our_enc_hidden.bin", enc_hidden, enc_S, H_enc);
    }
#endif

    // ---- 8. VAE decode -> stereo 48 kHz PCM ----
    // The decode reports per-node graph progress so the (otherwise opaque) VAE
    // stage advances the bar instead of freezing at the last DiT step.
    if (!report("vae", 0, 1)) return result;
    bool vae_ok = true;
    result.pcm  = m->vae->decode(latent, T, [&](int done, int total) {
        vae_ok = report("vae", done, total);
        return vae_ok;
    });
    if (!vae_ok) return result;  // cancelled mid-decode
    if (result.pcm.empty()) throw std::runtime_error("acestep engine: VAE decode failed");
    report("vae", 1, 1);

    // ---- metadata ----
    // Read back from prompt.*, not params.*: Phase 1 gap-fills the prompt in
    // place (bpm/keyscale/timesignature/vocal_language/caption) from a bare
    // caption, so params.* may still be empty/zero here.
    result.metadata.caption        = prompt.caption;
    result.metadata.lyrics         = prompt.lyrics;
    result.metadata.keyscale       = prompt.keyscale;
    result.metadata.vocal_language = prompt.vocal_language;
    result.metadata.bpm            = prompt.bpm;
    // timesignature is a string on the prompt ("4/4" / "4"); surface the
    // numerator as the int metadata field (0 if unset/unparseable).
    result.metadata.timesignature  = prompt.timesignature.empty() ? 0 : atoi(prompt.timesignature.c_str());
    result.metadata.seed           = seed;
    result.metadata.n_codes        = T_5Hz;
    return result;
}

void        Engine::cancel() const { impl_->cancel_flag.store(true); }
int         Engine::sample_rate() const { return impl_->vae ? impl_->vae->sample_rate() : 48000; }
std::string Engine::backend_name() const { return impl_->backend ? ggml_backend_name(impl_->backend) : "cpu"; }

} // namespace tts_cpp::acestep
