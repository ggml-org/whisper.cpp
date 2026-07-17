#include "tts-cpp/acestep/engine.h"

#include "tts-cpp/acestep/vae.h"

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
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <thread>

// ACE-Step end-to-end music engine (QVAC-21921). Wires the ported stages behind
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

// DiT instruction headers (task-types.h). COVER is used whenever LM codes exist.
static const char * DIT_INSTR_COVER = "Generate audio semantic tokens based on the given conditions:";

namespace fs = std::filesystem;

struct Engine::Impl {
    EngineOptions opts;

    ggml_backend_t backend = nullptr;  // shared CPU backend for the ggml stages

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
        if (backend) ggml_backend_free(backend);
    }
};

// ------------------------------------------------------------ path resolution
static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

// Classify the four GGUFs in models_dir by filename substring. Explicit paths in
// EngineOptions win over the scan.
static void resolve_paths(EngineOptions & o) {
    if (o.models_dir.empty()) return;
    std::error_code ec;
    for (auto & e : fs::directory_iterator(o.models_dir, ec)) {
        if (!e.is_regular_file()) continue;
        std::string path = e.path().string();
        std::string name = to_lower(e.path().filename().string());
        if (name.size() < 5 || name.substr(name.size() - 5) != ".gguf") continue;
        if (name.find("embedding") != std::string::npos) {
            if (o.text_enc_model_path.empty()) o.text_enc_model_path = path;
        } else if (name.find("lm") != std::string::npos) {
            if (o.lm_model_path.empty()) o.lm_model_path = path;
        } else if (name.find("vae") != std::string::npos) {
            if (o.vae_model_path.empty()) o.vae_model_path = path;
        } else if (name.find("turbo") != std::string::npos || name.find("dit") != std::string::npos ||
                   name.find("v15") != std::string::npos) {
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

    m->backend = ggml_backend_cpu_init();
    if (!m->backend) throw std::runtime_error("acestep engine: CPU backend init failed");
    int nth = opts.n_threads > 0 ? opts.n_threads : (int) std::thread::hardware_concurrency();
    if (nth < 1) nth = 4;
    ggml_backend_cpu_set_n_threads(m->backend, nth);

    const bool v = opts.verbose;

    m->textenc = textenc_model_load(opts.text_enc_model_path, m->backend, v);
    if (!m->textenc) throw std::runtime_error("acestep engine: text-encoder load failed");

    // 2 KV sets: cond + uncond for classifier-free guidance on Phase-2 codes.
    m->lm = lm_model_load(opts.lm_model_path, m->backend, /*max_seq_len=*/2048, v, /*n_kv_sets=*/2);
    if (!m->lm) throw std::runtime_error("acestep engine: LM load failed");

    m->cond = cond_model_load(opts.dit_model_path, m->backend, v);
    if (!m->cond) throw std::runtime_error("acestep engine: cond-encoder load failed");

    m->detok = detok_model_load(opts.dit_model_path, m->backend, v);
    if (!m->detok) throw std::runtime_error("acestep engine: FSQ detokenizer load failed");

    m->dit = dit_model_load(opts.dit_model_path, m->backend, v);
    if (!m->dit) throw std::runtime_error("acestep engine: DiT load failed");

    VaeOptions vo;
    vo.verbose      = v;
    vo.with_encoder = false;  // generation only decodes
    vo.n_threads    = nth;
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
        lp.seed        = (uint32_t) seed;

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
    for (int t = 0; t < T; t++) {
        float * dst = context.data() + (size_t) t * ctx_ch;
        if (t < T_25Hz) memcpy(dst, detok_latent.data() + (size_t) t * Oc, (size_t) Oc * sizeof(float));
        for (int c = 0; c < Oc; c++) dst[Oc + c] = 1.0f;  // chunk mask
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

    std::vector<int32_t> text_ids32(text_ids.begin(), text_ids.end());
    std::vector<int32_t> lyric_ids32(lyric_ids.begin(), lyric_ids.end());

    std::vector<float> text_hidden, lyric_embed;
    if (!textenc_model_forward(m->textenc, text_ids32.data(), S_text, text_hidden))
        throw std::runtime_error("acestep engine: text-encoder forward failed");
    if (!textenc_model_embed_lookup(m->textenc, lyric_ids32.data(), S_lyric, lyric_embed))
        throw std::runtime_error("acestep engine: lyric embed lookup failed");

    // ---- 5. cond-encoder: -> enc_hidden [2048, S_total] ----
    std::vector<float> enc_hidden;
    int                enc_S = 0;
    if (!cond_model_forward(m->cond, text_hidden.data(), S_text, lyric_embed.data(), S_lyric, nullptr, 0, enc_hidden,
                            &enc_S))
        throw std::runtime_error("acestep engine: cond-encoder forward failed");
    const int H_enc = (int) (enc_hidden.size() / (size_t) enc_S);  // 2048

    // ---- 6. noise [64, T] (Philox, torch.randn parity) ----
    std::vector<float> noise((size_t) Oc * T);
    philox_randn(seed, noise.data(), (int) noise.size(), /*bf16_round=*/true);

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

    std::vector<float> latent;
    if (!dit_sample(m->dit, sp, latent)) throw std::runtime_error("acestep engine: DiT sample failed");
    if (!report("dit", n_steps, n_steps)) return result;

    // ---- 8. VAE decode -> stereo 48 kHz PCM ----
    if (!report("vae", 0, 1)) return result;
    result.pcm = m->vae->decode(latent, T);
    if (result.pcm.empty()) throw std::runtime_error("acestep engine: VAE decode failed");
    report("vae", 1, 1);

    // ---- metadata ----
    result.metadata.caption        = params.caption;
    result.metadata.lyrics         = prompt.lyrics;
    result.metadata.keyscale       = params.keyscale;
    result.metadata.vocal_language = language;
    result.metadata.bpm            = params.bpm;
    result.metadata.seed           = seed;
    result.metadata.n_codes        = T_5Hz;
    return result;
}

void        Engine::cancel() const { impl_->cancel_flag.store(true); }
int         Engine::sample_rate() const { return impl_->vae ? impl_->vae->sample_rate() : 48000; }
std::string Engine::backend_name() const { return impl_->backend ? ggml_backend_name(impl_->backend) : "cpu"; }

} // namespace tts_cpp::acestep
