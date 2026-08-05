#include "audiogen-cpp/acestep/engine.h"

#include "audiogen-cpp/acestep/vae.h"

#include "acestep/bpe_tokenizer.h"
#include "acestep/cond_ggml.h"
#include "acestep/detok_ggml.h"
#include "acestep/dit_ggml.h"
#include "acestep/dit_gguf.h"  // DitGGUF: read DiT config + validate GGUF headers at create()
#include "acestep/lm_ggml.h"
#include "acestep/lm_pipeline.h"
#include "acestep/philox.h"
#include "acestep/textenc_ggml.h"

#include "acestep/backend_registry.h"
#include "acestep/stage_placement.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>

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

    ggml_backend_t backend       = nullptr;  // primary backend (GPU or CPU) for textenc/cond/DiT
    ggml_backend_t backend_cpu   = nullptr;  // CPU backend for the stages pinned off the GPU; == backend when primary is CPU
    ggml_backend_t backend_lm    = nullptr;  // backend the LM loads on (see create)
    ggml_backend_t backend_enc   = nullptr;  // backend for textenc + cond (see create)
    ggml_backend_t backend_detok = nullptr;  // backend the FSQ detok loads on (CPU off-GPU, GPU on Vulkan; see create)
    TextEncModel * textenc = nullptr;
    LMModel *      lm      = nullptr;
    CondModel *    cond    = nullptr;
    DetokModel *   detok   = nullptr;
    DitModel *     dit     = nullptr;

    std::unique_ptr<Vae> vae;

    BpeTokenizer bpe_lm;    // LM vocab (+ audio codes) — Phase-2 prompt
    BpeTokenizer bpe_text;  // text-encoder vocab — DiT prompt / lyric lookup

    mutable std::atomic<bool> cancel_flag{ false };

    // Sequential-loading state (see create()/generate()). In the default low-mem
    // mode no stage is resident after create(); generate() loads each stage right
    // before its step and frees it right after, so the peak resident set is one
    // stage — not all six at once. dit_cfg is read from the DiT GGUF metadata at
    // create() so the context-build step needs no resident DiT; sr lets
    // sample_rate() answer before the VAE is loaded.
    DitConfig  dit_cfg{};
    int        sr          = 48000;
    int        nth         = 4;
    bool       keep_stages = false;  // ACESTEP_KEEP_STAGES: eager-load + never free
    VaeOptions vae_opts{};           // saved so the VAE can be (re)loaded lazily

    ~Impl() {
        free_dit();
        free_detok();
        free_cond();
        free_lm();
        free_textenc();
        free_vae();
        if (backend_cpu && backend_cpu != backend) ggml_backend_free(backend_cpu);
        if (backend) ggml_backend_free(backend);
    }

    // --- per-stage lazy load / free -----------------------------------------
    // ensure_*() loads a stage if it is not already resident (idempotent, so it
    // serves both the eager keep-stages path and the lazy per-generate reload);
    // free_*() releases it. All throw on a genuine load failure.
    void ensure_textenc() {
        if (textenc) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading text-encoder\n");
        textenc = textenc_model_load(opts.text_enc_model_path, backend_enc, opts.verbose);
        if (!textenc) throw std::runtime_error("acestep engine: text-encoder load failed");
    }
    void ensure_lm() {
        if (lm) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading LM\n");
        lm = lm_model_load(opts.lm_model_path, backend_lm, /*max_seq_len=*/2048, opts.verbose, /*n_kv_sets=*/2);
        if (!lm) throw std::runtime_error("acestep engine: LM load failed");
    }
    void ensure_cond() {
        if (cond) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading cond-encoder\n");
        cond = cond_model_load(opts.dit_model_path, backend_enc, opts.verbose);
        if (!cond) throw std::runtime_error("acestep engine: cond-encoder load failed");
    }
    void ensure_detok() {
        if (detok) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading FSQ detokenizer\n");
        detok = detok_model_load(opts.dit_model_path, backend_detok, opts.verbose);
        if (!detok) throw std::runtime_error("acestep engine: FSQ detokenizer load failed");
    }
    void ensure_dit() {
        if (dit) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading DiT\n");
        dit = dit_model_load(opts.dit_model_path, backend, opts.verbose);
        if (!dit) throw std::runtime_error("acestep engine: DiT load failed");
    }
    void ensure_vae() {
        if (vae) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading VAE\n");
        vae = Vae::load(opts.vae_model_path, vae_opts);
        if (!vae) throw std::runtime_error("acestep engine: VAE load failed");
        sr = vae->sample_rate();
    }
    void free_textenc() { if (textenc) { textenc_model_free(textenc); textenc = nullptr; } }
    void free_lm()      { if (lm)      { lm_model_free(lm);           lm      = nullptr; } }
    void free_cond()    { if (cond)    { cond_model_free(cond);       cond    = nullptr; } }
    void free_detok()   { if (detok)   { detok_model_free(detok);     detok   = nullptr; } }
    void free_dit()     { if (dit)     { dit_model_free(dit);         dit     = nullptr; } }
    void free_vae()     { vae.reset(); }
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

    // Load the dlopen'd ggml backend modules (per-microarch CPU variants on
    // arm64, plus any GPU MODULE .so) the addon staged next to its `.bare`, so
    // the registry-based backend init below can find a CPU/GPU device. No-op on
    // static-linked desktop / Apple builds. Must run before any backend init.
    load_backends(opts.backends_dir);

    int nth = opts.n_threads > 0 ? opts.n_threads : (int) std::thread::hardware_concurrency();
    if (nth < 1) nth = 4;

    // Backend for the ggml stages (text-encoder, LM, cond/detok, DiT). These use
    // only standard ggml ops, so a GPU backend (Metal on Apple, CUDA/Vulkan
    // elsewhere) can run them; opts.n_gpu_layers > 0 opts in. The VAE gets its
    // own dedicated backend (see Vae::load) and also follows n_gpu_layers now
    // that its snake / col2im_1d ops have Metal and Vulkan kernels in the
    // ggml-speech fork.
    // Falls back to CPU when no GPU backend is registered/available.
    bool on_gpu = false;
    if (opts.n_gpu_layers > 0) {
        m->backend = backend_gpu_init();
        on_gpu     = (m->backend != nullptr);
        if (!on_gpu && v) fprintf(stderr, "[acestep-engine] GPU requested but no GPU backend available; using CPU\n");
    }
    if (!m->backend) m->backend = backend_cpu_init();
    if (!m->backend) throw std::runtime_error("acestep engine: backend init failed");
    if (on_gpu) {
        if (v) fprintf(stderr, "[acestep-engine] DiT/VAE on GPU backend: %s\n", ggml_backend_name(m->backend));
        // Dedicated CPU backend for whichever stages are pinned off the GPU below.
        m->backend_cpu = backend_cpu_init();
        if (!m->backend_cpu) throw std::runtime_error("acestep engine: CPU backend init failed");
        backend_set_n_threads(m->backend_cpu, nth);
    } else {
        backend_set_n_threads(m->backend, nth);
        m->backend_cpu = m->backend;  // single CPU backend serves every stage
    }

    // Stage placement when a GPU is active. The DiT, the VAE and the one-shot
    // text/cond encoders always run on it. The autoregressive LM and the FSQ
    // detokenizer are allowlisted per backend, so a backend nobody has measured
    // keeps the CPU placement and cannot silently regress generated audio.
    // ACESTEP_LM_GPU / ACESTEP_DETOK_GPU take that measurement without a rebuild.
    // The policy itself lives in stage_placement.h so it can be unit tested
    // without a GPU (test/test_acestep_units.cpp); this only applies its answer.
    //
    // Env escape hatches (applied after the allowlist; CPU wins if both are set):
    //   ACESTEP_LM_GPU=1        -> LM on the GPU, whatever the backend
    //   ACESTEP_LM_CPU=1        -> LM on the CPU, whatever the backend
    //   ACESTEP_DETOK_GPU=1     -> detokenizer on the GPU, whatever the backend
    //   ACESTEP_DETOK_CPU=1     -> detokenizer on the CPU, whatever the backend
    //   ACESTEP_ENCODERS_CPU=1  -> move the encoders to the CPU (trim wired mem)
    ggml_backend_t enc_backend   = m->backend;
    ggml_backend_t lm_backend    = m->backend;
    ggml_backend_t detok_backend = m->backend;
    if (on_gpu) {
        const StagePlacement place =
            resolve_stage_placement(backend_reg_name(m->backend), placement_overrides_from_env());
        if (!place.enc_on_gpu)   enc_backend   = m->backend_cpu;
        if (!place.lm_on_gpu)    lm_backend    = m->backend_cpu;
        if (!place.detok_on_gpu) detok_backend = m->backend_cpu;
        if (v) fprintf(stderr, "[acestep-engine] backends: enc=%s lm=%s detok=%s dit/vae=%s\n",
                       ggml_backend_name(enc_backend), ggml_backend_name(lm_backend),
                       ggml_backend_name(detok_backend), ggml_backend_name(m->backend));
    }
    m->backend_enc   = enc_backend;
    m->backend_lm    = lm_backend;
    m->backend_detok = detok_backend;  // GPU on the allowlisted backends, CPU otherwise
    m->nth           = nth;

    // ACE-Step loads six weight sets (text-enc, LM, cond, detok, DiT, VAE). Held
    // resident at once they peak well past a non-entitled iOS app's memory ceiling
    // and the OS jetsam-kills the process mid-load. So by default the engine loads
    // each stage lazily inside generate() — right before its step, freed right
    // after — bounding the peak to a single stage (QVAC-22955). Opt out with
    // ACESTEP_KEEP_STAGES=1 (servers that generate back-to-back and prefer to pay
    // the load cost once and keep everything resident).
    m->keep_stages = [] {
        const char * e = std::getenv("ACESTEP_KEEP_STAGES");
        return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T' || e[0] == 'y' || e[0] == 'Y');
    }();

    // VAE load options, saved so generate() can (re)load the VAE lazily. Decode
    // only (no encoder). ACESTEP_VAE_GPU forces the VAE backend independently of
    // the other stages so a decode can be compared CPU-vs-GPU on an identical
    // latent (=1 -> GPU, =0 -> CPU); leaves the LM/DiT backend untouched.
    VaeOptions vo;
    vo.verbose      = v;
    vo.with_encoder = false;
    vo.n_threads    = nth;
    vo.n_gpu_layers = opts.n_gpu_layers;  // snake / col2im_1d have Metal + Vulkan kernels
    if (const char * e = std::getenv("ACESTEP_VAE_GPU")) {
        vo.n_gpu_layers = (e[0] == '1') ? 99 : 0;
    }
    m->vae_opts = vo;

    // Read the DiT config from GGUF metadata up front so the context-build step in
    // generate() needs no resident DiT (the full DiT is loaded lazily just before
    // the diffusion step). This also validates the DiT GGUF header at create().
    {
        DitGGUF g;
        if (!dit_gguf_open(g, opts.dit_model_path))
            throw std::runtime_error("acestep engine: cannot open DiT GGUF: " + opts.dit_model_path);
        const bool cfg_ok = dit_gguf_read_config(g, m->dit_cfg);
        dit_gguf_close(g);
        if (!cfg_ok) throw std::runtime_error("acestep engine: bad DiT config in " + opts.dit_model_path);
    }

    // Tokenizers: LM prompt uses the LM vocab; DiT text prompt + lyric lookup use
    // the text-encoder vocab. Fall back to the LM tokenizer if the text-encoder
    // GGUF has no tokenizer KV (same Qwen text vocab in the shared range). Loading
    // them here also validates the LM + text-encoder GGUF headers at create().
    if (!bpe_load_from_gguf(m->bpe_lm, opts.lm_model_path))
        throw std::runtime_error("acestep engine: LM tokenizer load failed");
    if (!bpe_load_from_gguf(m->bpe_text, opts.text_enc_model_path)) {
        if (v) fprintf(stderr, "[acestep-engine] text-encoder has no tokenizer KV; reusing LM tokenizer\n");
        m->bpe_text = m->bpe_lm;
    }

    // Validate the VAE GGUF header too (a no_alloc metadata parse), so create()
    // still fails fast on a missing/corrupt file instead of only at first
    // generate(). The other three GGUFs were validated above.
    {
        DitGGUF g;
        if (!dit_gguf_open(g, opts.vae_model_path))
            throw std::runtime_error("acestep engine: cannot open VAE GGUF: " + opts.vae_model_path);
        dit_gguf_close(g);
    }

    // ACESTEP_KEEP_STAGES: eager-load every stage now and keep it resident (the
    // pre-QVAC-22955 behaviour). Default (lazy) leaves all stages unloaded here;
    // generate() loads/frees them per step.
    if (m->keep_stages) {
        m->ensure_textenc();
        m->ensure_lm();
        m->ensure_cond();
        m->ensure_detok();
        m->ensure_dit();
        m->ensure_vae();
    }

    if (v) fprintf(stderr, "[acestep-engine] ready (threads=%d, %s)\n", nth,
                   m->keep_stages ? "stages resident" : "stages lazy/low-mem");
    return eng;
}

// ------------------------------------------------------------ stage timing
// The ProgressFn only carries cumulative wall-clock at stage boundaries, which
// leaves detok + text-encoder + cond-encoder inside a single unmeasured gap.
// Recording each stage explicitly is what lets the "speed up each stage" vs
// "overlap the stages" question be answered with measurements. Printed when
// EngineOptions::verbose is set.
namespace {
class StageTimes {
public:
    void mark(const char * name) {
        const auto now = std::chrono::steady_clock::now();
        entries_.emplace_back(name, std::chrono::duration<double, std::milli>(now - last_).count());
        last_ = now;
    }

    void dump(FILE * f) const {
        double total = 0.0;
        for (const auto & e : entries_) total += e.second;
        fprintf(f, "[acestep-timing] per-stage wall clock (total %.0f ms)\n", total);
        for (const auto & e : entries_)
            fprintf(f, "[acestep-timing]   %-12s %9.1f ms  %5.1f%%\n", e.first, e.second,
                    total > 0.0 ? 100.0 * e.second / total : 0.0);
    }

private:
    std::chrono::steady_clock::time_point       last_ = std::chrono::steady_clock::now();
    std::vector<std::pair<const char *, double>> entries_;
};

// Per-stage tensor dumps for cross-backend parity work. Inactive (and free)
// unless EngineOptions::dump_stages_dir is set, so this lives in the normal
// build instead of behind a compile flag: localising a backend divergence
// should not require rebuilding the engine.
class StageDump {
public:
    explicit StageDump(std::string dir, bool verbose) : dir_(std::move(dir)), verbose_(verbose) {}

    bool active() const { return !dir_.empty(); }

    // d0 = row count (slow axis), d1 = row length (fastest-varying axis), which
    // is the convention dbg_write_dump and acestep.cpp's --dump already use.
    void write(const char * name, const float * data, size_t n, int d0, int d1) const {
        if (!active()) return;
        const std::string path = dir_ + "/" + name + ".bin";
        FILE *            f    = fopen(path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "[acestep-dump] cannot write %s\n", path.c_str());
            return;
        }
        const int32_t hdr[3] = {2, d0, d1};
        fwrite(hdr, sizeof(int32_t), 3, f);
        fwrite(data, sizeof(float), n, f);
        fclose(f);
        if (verbose_) fprintf(stderr, "[acestep-dump] %s [%d, %d] %zu floats\n", path.c_str(), d0, d1, n);
    }

    void write(const char * name, const std::vector<float> & v, int d0, int d1) const {
        write(name, v.data(), v.size(), d0, d1);
    }

    // Integer stages (LM codes) are stored as float32 so one reader handles every
    // dump; code ids are well inside the exactly-representable range.
    void write_ints(const char * name, const std::vector<int> & v) const {
        if (!active()) return;
        std::vector<float> f(v.begin(), v.end());
        write(name, f, (int) f.size(), 1);
    }

private:
    std::string dir_;
    bool        verbose_ = false;
};
}  // namespace

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
    result.sample_rate = m->sr;  // VAE may not be resident yet (loaded lazily below)
    result.channels    = 2;
    m->cancel_flag.store(false);

    auto report = [&](const char * stage, int step, int total) -> bool {
        if (progress && !progress(stage, step, total)) { m->cancel_flag.store(true); return false; }
        return !m->cancel_flag.load();
    };

    StageTimes timing;
    StageDump  dump(m->opts.dump_stages_dir, m->opts.verbose);

    // Sequential loading (default): each stage is loaded via m->ensure_*() right
    // before its step and freed via m->free_*() right after, so the peak resident
    // set is one stage, or the cond + text encoders where those two overlap,
    // rather than all six at once — small enough for a non-entitled iOS app not
    // to be jetsam-killed. With ACESTEP_KEEP_STAGES=1 every stage is already
    // resident (create() eager-loaded it), ensure_*() is a no-op, and the free_*()
    // calls are skipped so nothing is released between generate() calls. Each
    // stage's lazy-load cost is attributed to its own timing.mark() below — there
    // is no separate up-front reload phase.
    const bool low_mem = !m->keep_stages;

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
        m->ensure_lm();  // load the LM just before its step (freed right after)
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
    // are not resident during the rest of the pipeline. Reloaded next generate().
    if (low_mem) {
        if (m->lm && m->opts.verbose) fprintf(stderr, "[acestep-engine] freeing LM (low-mem)\n");
        m->free_lm();
    }
    timing.mark("lm");
    dump.write_ints("01_lm_codes", codes);

    // ---- 2. FSQ detok: codes -> context latents [64, T_25Hz] ----
    m->ensure_detok();  // load the detokenizer just before its (only) use
    const int          T_5Hz  = (int) codes.size();
    const int          T_25Hz = T_5Hz * 5;
    std::vector<float> detok_latent((size_t) 64 * T_25Hz);
    if (detok_model_decode(m->detok, codes.data(), T_5Hz, detok_latent.data()) != T_25Hz)
        throw std::runtime_error("acestep engine: FSQ detokenizer failed");
    timing.mark("detok");
    dump.write("02_detok_latent", detok_latent, T_25Hz, 64);

    // Detok latents are in hand; the detokenizer is not needed again. Free it.
    if (low_mem) {
        if (m->detok && m->opts.verbose) fprintf(stderr, "[acestep-engine] freeing FSQ detokenizer (low-mem)\n");
        m->free_detok();
    }

    // ---- 3. context [ctx_ch=128, T] = [detok latent[64] | mask[64]=1] ----
    // dit_cfg was read from GGUF metadata at create() so no resident DiT is needed
    // to size the context (the DiT itself is loaded lazily just before step 7).
    const DitConfig & dc     = m->dit_cfg;
    const int         Oc     = dc.out_channels;            // 64
    const int         ctx_ch = dc.in_channels - Oc;        // 128
    const int         patch  = dc.patch_size;              // 2
    int               T      = ((T_25Hz + patch - 1) / patch) * patch;

    // The cond-encoder supplies the silence frame here and runs its forward at
    // step 5, so it is needed from now through step 5; load it once, up front.
    m->ensure_cond();

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

    m->ensure_textenc();  // load the text-encoder just before its step
    std::vector<float> text_hidden, lyric_embed;
    if (!textenc_model_forward(m->textenc, text_ids32.data(), S_text, text_hidden))
        throw std::runtime_error("acestep engine: text-encoder forward failed");
    if (!textenc_model_embed_lookup(m->textenc, lyric_ids32.data(), S_lyric, lyric_embed))
        throw std::runtime_error("acestep engine: lyric embed lookup failed");
    timing.mark("textenc");
    dump.write("04_text_hidden", text_hidden, S_text, (int) (text_hidden.size() / (size_t) S_text));
    dump.write("05_lyric_embed", lyric_embed, S_lyric, (int) (lyric_embed.size() / (size_t) S_lyric));

    // text_hidden + lyric_embed are on the host now; the text-encoder (~742 MB) is
    // done. Free it before the cond-encoder forward so only one of the two is
    // resident at a time.
    if (low_mem) {
        if (m->textenc && m->opts.verbose) fprintf(stderr, "[acestep-engine] freeing text-encoder (low-mem)\n");
        m->free_textenc();
    }

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
    // so the cond-encoder (~352 MB) is no longer needed. Free it before the
    // DiT/VAE so its (wired) buffers don't count against the device memory
    // ceiling. Reloaded on the next generate().
    if (low_mem) {
        if (m->cond && m->opts.verbose) fprintf(stderr, "[acestep-engine] freeing cond-encoder (low-mem)\n");
        m->free_cond();
    }

    timing.mark("cond");
    dump.write("03_context", context, T, ctx_ch);
    dump.write("06_enc_hidden", enc_hidden, enc_S, H_enc);

    // ---- 6. noise [64, T] (Philox, torch.randn parity) ----
    std::vector<float> noise((size_t) Oc * T);
    philox_randn(seed, noise.data(), (int) noise.size(), /*bf16_round=*/true);
    timing.mark("noise");
    dump.write("07_noise", noise, T, Oc);

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

    m->ensure_dit();  // load the DiT (the largest stage) just before diffusion

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
    timing.mark("dit");
    dump.write("08_dit_latent", latent, T, Oc);

    // The denoised latent is on the host; the DiT (the largest stage) is done.
    // Free it before the VAE decode so only the VAE is resident for the last step.
    if (low_mem) {
        if (m->dit && m->opts.verbose) fprintf(stderr, "[acestep-engine] freeing DiT (low-mem)\n");
        m->free_dit();
    }

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
    m->ensure_vae();  // load the VAE just before decode
    if (!report("vae", 0, 1)) return result;
    bool vae_ok = true;
    result.pcm  = m->vae->decode(latent, T, [&](int done, int total) {
        vae_ok = report("vae", done, total);
        return vae_ok;
    });
    if (!vae_ok) return result;  // cancelled mid-decode
    if (result.pcm.empty()) throw std::runtime_error("acestep engine: VAE decode failed");
    report("vae", 1, 1);
    timing.mark("vae");
    dump.write("09_vae_pcm", result.pcm, (int) (result.pcm.size() / 2), 2);
    if (m->opts.verbose) timing.dump(stderr);

    // The VAE is the last stage; free it so nothing is resident between
    // generate() calls (each stage is reloaded on demand next time).
    if (low_mem) {
        if (m->vae && m->opts.verbose) fprintf(stderr, "[acestep-engine] freeing VAE (low-mem)\n");
        m->free_vae();
    }

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
int         Engine::sample_rate() const { return impl_->sr; }  // cached; VAE loaded lazily
std::string Engine::backend_name() const { return impl_->backend ? ggml_backend_name(impl_->backend) : "cpu"; }

} // namespace tts_cpp::acestep
