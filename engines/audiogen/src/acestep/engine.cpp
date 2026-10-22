#include "audiogen-cpp/acestep/engine.h"

#include "audiogen-cpp/acestep/vae.h"

#include "acestep/bpe_tokenizer.h"
#include "acestep/cond_ggml.h"
#include "acestep/detok_ggml.h"
#include "acestep/dit_ggml.h"
#include "acestep/dit_gguf.h"  // DitGGUF: read DiT config + validate GGUF headers at create()
#include "acestep/lm_ggml.h"
#include "acestep/lm_pipeline.h"
#include "acestep/loudness.h"
#include "acestep/lyrics_alignment.h"
#include "acestep/philox.h"
#include "acestep/quality_score.h"
#include "acestep/tok_ggml.h"
#include "acestep/textenc_ggml.h"

#include "acestep/cancellation_scope.h"
#include "acestep/backend_registry.h"
#include "acestep/engine_backends.h"
#include "acestep/engine_paths.h"
#include "acestep/audio_edit.h"
#include "acestep/cover_noise.h"
#include "acestep/generate_task.h"
#include "acestep/generation_conditioning.h"
#include "acestep/generation_plan.h"
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
#include <unordered_map>
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

static const char * DIT_INSTR_TEXT2MUSIC = "Fill the audio semantic mask based on the given conditions:";
static const char * DIT_INSTR_COVER      = "Generate audio semantic tokens based on the given conditions:";
static const char * DIT_INSTR_REPAINT    = "Repaint the mask area based on the given conditions:";

static std::string uppercase_track_name(const std::string & track) {
    std::string upper = track;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char ch) { return (char) std::toupper(ch); });
    return upper;
}

// Lego instruction, uppercase track per the reference implementation
// (task_utils.py formats TASK_INSTRUCTIONS["lego"] with track_name.upper()).
static std::string make_lego_instruction(const std::string & track) {
    return "Generate the " + uppercase_track_name(track) + " track based on the audio context:";
}

namespace fs = std::filesystem;

struct Engine::Impl {
    EngineOptions opts;

    GpuFallbackReason gpu_fallback_reason = GpuFallbackReason::not_requested;

    ggml_backend_t backend       = nullptr;  // primary backend (GPU or CPU) for textenc/cond/DiT
    ggml_backend_t backend_cpu   = nullptr;  // CPU backend for the stages pinned off the GPU; == backend when primary is CPU
    ggml_backend_t backend_lm    = nullptr;  // backend the LM loads on (see create)
    ggml_backend_t backend_lm_extra = nullptr;
    ggml_backend_t backend_enc   = nullptr;  // backend for textenc + cond (see create)
    ggml_backend_t backend_detok = nullptr;  // backend the FSQ detok loads on (CPU off-GPU, GPU on Vulkan; see create)
    TextEncModel * textenc = nullptr;
    LMModel *      lm      = nullptr;
    CondModel *    cond    = nullptr;
    DetokModel *   detok   = nullptr;
    TokModel *     tok     = nullptr;
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
        free_tok();
        free_cond();
        free_lm();
        free_textenc();
        free_vae();
        if (backend_lm_extra) ggml_backend_free(backend_lm_extra);
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
    void ensure_tok() {
        if (tok) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading FSQ tokenizer\n");
        tok = tok_model_load(opts.dit_model_path, backend_detok, opts.verbose);
        if (!tok) throw std::runtime_error("acestep engine: FSQ tokenizer load failed");
    }
    void ensure_dit() {
        if (dit) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading DiT\n");
        dit = dit_model_load(opts.dit_model_path, backend, opts.verbose);
        if (!dit) throw std::runtime_error("acestep engine: DiT load failed");
    }
    void ensure_vae(bool with_encoder = false) {
        if (vae && (!with_encoder || vae->has_encoder())) return;
        if (vae) free_vae();
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading VAE\n");
        VaeOptions load_opts = vae_opts;
        load_opts.with_encoder = with_encoder;
        vae = Vae::load(opts.vae_model_path, load_opts);
        if (!vae) throw std::runtime_error("acestep engine: VAE load failed");
        sr = vae->sample_rate();
    }
    void free_textenc() { if (textenc) { textenc_model_free(textenc); textenc = nullptr; } }
    void free_lm()      { if (lm)      { lm_model_free(lm);           lm      = nullptr; } }
    void free_cond()    { if (cond)    { cond_model_free(cond);       cond    = nullptr; } }
    void free_detok()   { if (detok)   { detok_model_free(detok);     detok   = nullptr; } }
    void free_tok()     { if (tok)     { tok_model_free(tok);         tok     = nullptr; } }
    void free_dit()     { if (dit)     { dit_model_free(dit);         dit     = nullptr; } }
    void free_vae()     { vae.reset(); }
};

// ------------------------------------------------------------ construction
// (models_dir -> per-stage path classification lives in engine_paths.h, shared
// with the memory-fit preflight.)
Engine::Engine() : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::create(const EngineOptions & opts_in) {
    EngineOptions opts = opts_in;
    resolve_stage_paths(opts);

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

    // Backend for the ggml stages (text-encoder, LM, cond/detok, DiT), the CPU
    // backend for stages the placement policy pins off the GPU, and the
    // per-stage assignments. The VAE gets its own dedicated backend (see
    // Vae::load) and also follows n_gpu_layers now that its snake / col2im_1d
    // ops are validated on Metal, Vulkan, and Adreno OpenCL in the ggml-speech
    // fork. The resolution itself lives in engine_backends.h (shared with the
    // memory-fit preflight so both agree by construction); the placement
    // policy in stage_placement.h (unit tested in test/test_acestep_units.cpp).
    //
    // Env escape hatches (applied after the allowlist; CPU wins if both are set):
    //   ACESTEP_LM_GPU=1        -> LM on the GPU, whatever the backend
    //   ACESTEP_LM_CPU=1        -> LM on the CPU, whatever the backend
    //   ACESTEP_DETOK_GPU=1     -> detokenizer on the GPU, whatever the backend
    //   ACESTEP_DETOK_CPU=1     -> detokenizer on the CPU, whatever the backend
    //   ACESTEP_ENCODERS_CPU=1  -> move the encoders to the CPU (trim wired mem)
    BackendRequest request;
    request.n_gpu_layers = opts.n_gpu_layers;
    request.backend      = opts.backend;
    request.lm_backend   = opts.lm_backend;
    request.n_threads    = opts.n_threads;
    request.verbose      = v;
    AcestepBackends rb;
    if (!resolve_acestep_backends(request, rb)) {
        throw std::runtime_error("acestep engine: backend init failed (backend '" + opts.backend +
                                 "', lm_backend '" + opts.lm_backend + "')");
    }
    m->gpu_fallback_reason = rb.gpu_fallback_reason;
    m->backend       = rb.backend;
    m->backend_cpu   = rb.backend_cpu;
    m->backend_enc   = rb.enc;
    m->backend_lm    = rb.lm;
    m->backend_lm_extra = rb.lm_extra;
    m->backend_detok = rb.detok;  // GPU on the allowlisted backends, CPU otherwise
    m->nth           = rb.nth;
    const int nth    = rb.nth;

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
    // Validated on Metal, Vulkan, and Adreno OpenCL; the ACESTEP_VAE_GPU
    // override lives in engine_backends.h, shared with the fit projection.
    const BackendRequest vae_request = vae_backend_request(request);
    vo.n_gpu_layers = vae_request.n_gpu_layers;
    vo.backend      = vae_request.backend;
    m->vae_opts = vo;

    // Read the DiT config from GGUF metadata up front so the context-build step in
    // generate() needs no resident DiT (the full DiT is loaded lazily just before
    // the diffusion step). This also validates the DiT GGUF header at create().
    {
        DitGGUF g;
        if (!dit_gguf_open(g, opts.dit_model_path))
            throw std::runtime_error("acestep engine: cannot open DiT GGUF: " + opts.dit_model_path);
        dit_gguf_require_hexagon_weights(g, opts.dit_model_path, m->backend, "acestep-engine");
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

static constexpr int AUDIO_CHANNELS         = 2;
static constexpr int AUDIO_SAMPLE_RATE      = 48000;
static constexpr int AUDIO_LATENT_RATE      = 25;
static constexpr int AUDIO_LATENT_CHANNELS  = 64;
static constexpr int AUDIO_CODE_FRAME_RATIO = 5;
static constexpr int TURBO_STEPS            = 8;
static constexpr int STANDARD_STEPS         = 50;
static constexpr float TURBO_SHIFT          = 3.0f;
static constexpr float STANDARD_SHIFT       = 1.0f;
static constexpr int DIT_BATCH_SIZE         = 1;
static constexpr int EDIT_CONTEXT_PLANES    = 2;
static constexpr int EDIT_NO_SOURCE_FRAMES  = 0;
static constexpr int EDIT_PROGRESS_START    = 0;
static constexpr int EDIT_PROGRESS_COMPLETE = 1;
static constexpr int EDIT_PROGRESS_TOTAL    = 1;
static constexpr float EDIT_EMPTY_LATENT    = 0.0f;
static constexpr float REPAINT_MIN_STRENGTH = 0.0f;
static constexpr float REPAINT_MAX_STRENGTH = 1.0f;
static constexpr float REPAINT_MASKED_FRAME = 1.0f;
static constexpr int TEMPO_SLOW_BPM_MAX     = 80;
static constexpr int TEMPO_MODERATE_BPM_MAX = 120;
static constexpr int TEMPO_FAST_BPM_MAX     = 160;

static constexpr const char * EDIT_STAGE_SOURCE    = "source";
static constexpr const char * EDIT_STAGE_REFERENCE = "reference";
static constexpr const char * EDIT_STAGE_REPAINT   = "repaint";
static constexpr const char * EDIT_STAGE_FLOW      = "flow-edit";
static constexpr const char * EDIT_STAGE_VAE       = "vae";

static constexpr const char * EDIT_DUMP_SOURCE_LATENT    = "00_source_latent";
static constexpr const char * EDIT_DUMP_REFERENCE_LATENT = "00_reference_latent";
static constexpr const char * EDIT_DUMP_NOISE            = "07_noise";
static constexpr const char * EDIT_DUMP_DIT_LATENT       = "08_dit_latent";
static constexpr const char * EDIT_DUMP_VAE_AUDIO        = "09_vae_audio";

static constexpr const char * EDIT_ERROR_TASK_TYPE =
    "acestep engine: edit_plan cannot be combined with task_type '";
static constexpr const char * EDIT_ERROR_SOURCE_REQUIRED =
    "acestep engine: edit_plan requires source_audio";
static constexpr const char * EDIT_ERROR_SOURCE_STEREO =
    "acestep engine: source_audio must be interleaved stereo";
static constexpr const char * EDIT_ERROR_AUDIO_CODES =
    "acestep engine: edit_plan bypasses LM and cannot use audio_codes";
static constexpr const char * EDIT_ERROR_INTERMEDIATE_DECODE =
    "acestep engine: intermediate edit VAE decode failed";
static constexpr const char * EDIT_ERROR_INTERMEDIATE_ENCODE =
    "acestep engine: intermediate edit VAE re-encode failed";
static constexpr const char * EDIT_ERROR_REPAINT_STRENGTH =
    "acestep engine: repaint strength must be between 0 and 1";
static constexpr const char * EDIT_ERROR_REPAINT_CONTEXT =
    "acestep engine: repaint expects 64 latent + 64 mask context channels";
static constexpr const char * EDIT_ERROR_REPAINT_SILENCE =
    "acestep engine: repaint silence latent is unavailable";
static constexpr const char * EDIT_ERROR_FLOW_SILENCE =
    "acestep engine: flow-edit silence latent is unavailable";
static constexpr const char * EDIT_ERROR_REPAINT_SAMPLE =
    "acestep engine: repaint DiT sample failed";
static constexpr const char * EDIT_ERROR_FLOW_TURBO =
    "acestep engine: flow-edit v1 is validated for turbo DiT only";
static constexpr const char * EDIT_ERROR_FLOW_SAMPLE =
    "acestep engine: flow-edit DiT sample failed";
static constexpr const char * EDIT_ERROR_FINAL_DECODE =
    "acestep engine: edit plan VAE decode failed";
static constexpr const char * ENGINE_ERROR_PREFIX = "acestep engine: ";
static constexpr const char * EDIT_TASK_TYPE_SUFFIX = "'";

using StageReporter = std::function<bool(const char *, int, int)>;

static long long resolve_seed(long long seed) {
    if (seed >= 0) return seed;
    std::random_device random;
    return (long long) random();
}

static const char * tempo_label(int bpm) {
    if (bpm < TEMPO_SLOW_BPM_MAX) return "slow";
    if (bpm < TEMPO_MODERATE_BPM_MAX) return "moderate";
    if (bpm < TEMPO_FAST_BPM_MAX) return "fast";
    return "very fast";
}

static std::string build_conditioning_caption(const std::string & caption, int bpm,
                                              const std::string & timesignature,
                                              const std::string & keyscale) {
    std::string result = caption;
    if (bpm > 0 && result.find("Target tempo:") == std::string::npos) {
        if (!result.empty() && result.back() != '.' && result.back() != '!' && result.back() != '?') {
            result += '.';
        }
        result += "\nTarget tempo: " + std::to_string(bpm) + " BPM (" + tempo_label(bpm) + ").";
        if (bpm < TEMPO_SLOW_BPM_MAX) {
            result += " Use a clearly perceptible slow, spacious pulse.";
        } else if (bpm < TEMPO_MODERATE_BPM_MAX) {
            result += " Use a clearly perceptible steady mid-tempo pulse.";
        } else if (bpm < TEMPO_FAST_BPM_MAX) {
            result += " Use a clearly perceptible fast, energetic pulse.";
        } else {
            result += " Use a clearly perceptible very-fast full-time pulse; avoid half-time interpretation.";
        }
    }
    if (!timesignature.empty() && result.find("Target time signature:") == std::string::npos) {
        result += "\nTarget time signature: " + timesignature + ".";
    }
    if (!keyscale.empty() && result.find("Target key:") == std::string::npos) {
        result += "\nTarget key: " + keyscale + ".";
    }
    return result;
}

static AcePrompt make_prompt(const GenerateParams & params, const std::string & language) {
    AcePrompt prompt;
    prompt.caption        = params.augment_caption_with_metadata
                                ? build_conditioning_caption(
                                      params.caption, params.bpm, params.timesignature, params.keyscale)
                                : params.caption;
    prompt.lyrics         = resolve_prompt_lyrics(params);
    prompt.duration       = params.duration;
    prompt.bpm            = params.bpm;
    prompt.keyscale       = params.keyscale;
    prompt.timesignature  = params.timesignature;
    prompt.vocal_language = language;
    return prompt;
}

static bool encode_audio(Vae & vae, const std::vector<float> & pcm, const char * stage,
                         const char * dump_name, const StageReporter & report, bool verbose,
                         StageDump & dump, StageTimes & timing, EncodedAudio & output) {
    if (!report(stage, 0, 1)) return false;

    const int pcm_frames = (int) (pcm.size() / AUDIO_CHANNELS);
    output.latent = vae.encode(pcm, pcm_frames, &output.frames);
    if (output.latent.empty() || output.frames <= 0) {
        throw std::runtime_error(std::string("acestep engine: ") + stage + " audio VAE encode failed");
    }

    if (verbose) {
        fprintf(stderr, "[acestep-engine] %s audio: %.2fs -> %d latent frames\n",
                stage, (float) pcm_frames / AUDIO_SAMPLE_RATE, output.frames);
    }
    dump.write(dump_name, output.latent, output.frames, AUDIO_LATENT_CHANNELS);
    timing.mark(stage);
    return report(stage, 1, 1);
}

static const char * resolve_conditioning_dump_name(const char * stage) {
    return strcmp(stage, "source") == 0 ? "00_source_latent" : "00_reference_latent";
}

static void fill_dit_context_mask(float * frame, int output_channels) {
    for (int channel = 0; channel < output_channels; ++channel) {
        frame[output_channels + channel] = 1.0f;
    }
}

static void fill_dit_context_frames(std::vector<float> & context, const std::vector<float> & latent,
                                    const std::vector<float> & silence, int frames, int latent_frames,
                                    int context_channels, int output_channels) {
    const int silence_frames = (int) (silence.size() / (size_t) output_channels);
    for (int frame_index = 0; frame_index < frames; ++frame_index) {
        float * frame = context.data() + (size_t) frame_index * context_channels;
        if (frame_index < latent_frames) {
            memcpy(frame, latent.data() + (size_t) frame_index * output_channels,
                   (size_t) output_channels * sizeof(float));
        } else if (silence_frames > 0) {
            const int silence_index = (frame_index - latent_frames) % silence_frames;
            memcpy(frame, silence.data() + (size_t) silence_index * output_channels,
                   (size_t) output_channels * sizeof(float));
        }
        fill_dit_context_mask(frame, output_channels);
    }
}

static std::vector<float> make_dit_context(const std::vector<float> & latent,
                                           const std::vector<float> & silence, int frames,
                                           int latent_frames, int context_channels,
                                           int output_channels) {
    std::vector<float> context((size_t) context_channels * frames, 0.0f);
    fill_dit_context_frames(context, latent, silence, frames, latent_frames,
                            context_channels, output_channels);
    return context;
}

struct GenerationState {
    GenerateTask task;
    GenerationPlan plan;
    long long seed = 0;
    std::string language;
    std::string original_caption;
    AcePrompt prompt;
    GenerationConditioning conditioning;
    std::vector<float> context_latents;
    int code_frames = 0;
    int latent_frames = 0;
    bool low_memory = false;
    double quality_score = 0.0;
    std::string quality_report;
};

struct PromptEncoding {
    std::vector<float> text_hidden;
    std::vector<float> lyric_embedding;
    int text_tokens = 0;
    int lyric_tokens = 0;
};

struct EncoderConditioning {
    std::vector<float> context;
    std::vector<float> hidden;
    std::vector<float> null_emb;
    std::vector<int32_t> lyric_tokens;
    int frames = 0;
    int context_channels = 0;
    int sequence = 0;
    int hidden_size = 0;

    // Post-switch conditioning (empty unless audio_cover_strength < 1); both
    // hidden buffers share the padded row count, sequence_switch is the real one.
    std::vector<float> context_switch;
    std::vector<float> hidden_switch;
    int sequence_switch = 0;
};

static int padded_conditioning_rows(const EncoderConditioning & conditioning) {
    return std::max(conditioning.sequence, conditioning.sequence_switch);
}

static void pad_hidden_rows(std::vector<float> & hidden, int max_rows, int width) {
    hidden.resize((size_t) max_rows * width, 0.0f);
}

struct NoiseSchedule {
    std::vector<float> noise;
    std::vector<float> schedule;
    int steps = 0;
    float shift = 0.0f;
};

static GenerationState make_generation_state(const GenerateParams & params, bool keep_stages) {
    GenerationState state;
    if (const std::string error = resolve_generate_task(params, state.task); !error.empty()) {
        throw std::invalid_argument(error);
    }
    state.plan = make_generation_plan(params, state.task);
    state.seed = resolve_seed(params.seed);
    state.language = resolve_prompt_language(params);
    if (params.augment_caption_with_metadata) {
        state.original_caption = params.caption;
    }
    state.prompt = make_prompt(params, state.language);
    state.low_memory = !keep_stages;
    return state;
}

template <typename EngineImpl>
static bool prepare_audio_conditioning(EngineImpl & engine, const GenerateParams & params,
                                       GenerationState & state, const StageReporter & report,
                                       StageDump & dump, StageTimes & timing) {
    if (!state.plan.encode_source && !state.plan.encode_reference) return true;

    engine.ensure_vae(true);
    const AudioEncoder encode =
        [&](const std::vector<float> & pcm, const char * stage, EncodedAudio & output) {
            return encode_audio(*engine.vae, pcm, stage, resolve_conditioning_dump_name(stage),
                                report, engine.opts.verbose, dump, timing, output);
        };
    if (!prepare_generation_conditioning(params, state.plan, encode, state.conditioning)) return false;

    if (state.plan.encode_source) {
        state.context_latents = std::move(state.conditioning.source.latent);
        state.latent_frames = state.conditioning.source.frames;
        state.prompt.duration = (float) state.latent_frames / AUDIO_LATENT_RATE;
    }
    if (state.low_memory) engine.free_vae();
    return true;
}

static bool has_complete_metadata(const AcePrompt & prompt) {
    return prompt.bpm > 0 && prompt.duration > 0 &&
           !prompt.keyscale.empty() && !prompt.timesignature.empty();
}

static LmSampleParams make_lm_sample_params(const GenerateParams & params, long long seed,
                                            bool verbose, const StageReporter & report) {
    LmSampleParams sample;
    sample.temperature = params.lm_temperature;
    sample.top_p = params.lm_top_p;
    sample.top_k = params.lm_top_k;
    sample.cfg_scale = params.lm_cfg_scale;
    sample.seed = (uint32_t) seed;
    sample.verbose = verbose;
    sample.on_step = [&](int current, int total) { return report("lm", current, total); };
    return sample;
}

static bool needs_lm_phase_one(const GenerateParams & params, const AcePrompt & prompt) {
    if (params.simple_mode || params.rewrite_query) return true;
    return params.lm_phase1 && !has_complete_metadata(prompt);
}

// Fields the LM expansion (Inspire or Format) left empty must never stay
// empty past Phase 1: downstream prompt building and metadata reporting rely
// on them, and state.language must carry the LM-chosen language into
// tokenize_prompt.
static void finalize_lm_expanded_prompt(GenerationState & state) {
    if (state.prompt.lyrics.empty()) state.prompt.lyrics = INSTRUMENTAL_LYRICS;
    if (state.prompt.vocal_language.empty()) state.prompt.vocal_language = DEFAULT_VOCAL_LANGUAGE;
    state.language = state.prompt.vocal_language;
}

// Returns false only on cancellation. A plain Phase-1 failure falls back to the
// provided/default metadata, except in simple mode, where the LM expansion is
// the whole point of the request.
template <typename EngineImpl>
static bool run_lm_phase_one(EngineImpl & engine, const GenerateParams & params,
                             GenerationState & state, const LmSampleParams & sample) {
    LmSampleParams phase_one = sample;
    phase_one.max_new_tokens = 0;
    const LmPhase1Mode mode = params.simple_mode  ? LmPhase1Mode::Inspire
                              : params.rewrite_query ? LmPhase1Mode::Format
                                                     : LmPhase1Mode::Auto;
    if (lm_generate_phase1(engine.lm, engine.bpe_lm, state.prompt, phase_one, true, true, mode)) {
        if (params.simple_mode || params.rewrite_query) finalize_lm_expanded_prompt(state);
        return true;
    }
    if (engine.cancel_flag.load()) return false;
    if (params.simple_mode) {
        throw std::runtime_error("acestep engine: simple_mode LM expansion failed");
    }
    if (params.rewrite_query) {
        throw std::runtime_error("acestep engine: rewrite_query LM formatting failed");
    }
    fprintf(stderr, "[acestep-engine] Phase 1 failed; falling back to provided/default metas\n");
    return true;
}

template <typename EngineImpl>
static bool generate_audio_codes(EngineImpl & engine, const GenerateParams & params,
                                 GenerationState & state, const StageReporter & report,
                                 std::vector<int> & codes) {
    if (!params.audio_codes.empty()) {
        codes = params.audio_codes;
        if (engine.opts.verbose) {
            fprintf(stderr, "[acestep-engine] using %zu pre-supplied codes (LM skipped)\n", codes.size());
        }
        return true;
    }

    engine.ensure_lm();
    const LmSampleParams sample =
        make_lm_sample_params(params, state.seed, engine.opts.verbose, report);
    if (needs_lm_phase_one(params, state.prompt)) {
        if (!run_lm_phase_one(engine, params, state, sample)) return false;
    }
    if (!lm_generate_codes(engine.lm, engine.bpe_lm, state.prompt, sample, codes) || codes.empty()) {
        if (engine.cancel_flag.load()) return false;
        throw std::runtime_error("acestep engine: LM produced no audio codes");
    }
    return true;
}

template <typename EngineImpl>
static bool run_quality_score_stage(EngineImpl & engine, const GenerateParams & params,
                                    GenerationState & state, const StageReporter & report,
                                    StageTimes & timing, const std::vector<int> & codes) {
    if (!params.compute_quality_score) return true;

    engine.ensure_lm();
    QualityScoreParams score_params;
    score_params.on_step = [&](int cur, int total) { return report("score", cur, total); };

    QualityScoreResult score;
    std::string        error;
    if (!compute_quality_score(engine.lm, engine.bpe_lm, state.prompt, codes, score_params, score,
                               error)) {
        if (engine.cancel_flag.load()) return false;
        throw std::runtime_error("acestep engine: quality scoring failed: " + error);
    }
    state.quality_score  = score.global_score;
    state.quality_report = score.report;
    timing.mark("score");
    if (engine.opts.verbose) {
        fprintf(stderr, "[acestep-engine] quality score %.4f\n%s\n", score.global_score,
                score.report.c_str());
    }
    return true;
}

template <typename EngineImpl>
static bool run_lm_stage(EngineImpl & engine, const GenerateParams & params,
                         GenerationState & state, const StageReporter & report,
                         StageDump & dump, StageTimes & timing, std::vector<int> & codes) {
    if (!report("lm", 0, 1)) return false;
    if (!generate_audio_codes(engine, params, state, report, codes)) return false;
    if (!report("lm", 1, 1)) return false;
    if (!run_quality_score_stage(engine, params, state, report, timing, codes)) return false;

    if (state.low_memory) {
        if (engine.lm && engine.opts.verbose) fprintf(stderr, "[acestep-engine] freeing LM (low-mem)\n");
        engine.free_lm();
    }
    timing.mark("lm");
    dump.write_ints("01_lm_codes", codes);
    return true;
}

template <typename EngineImpl>
static void run_detokenizer_stage(EngineImpl & engine, const std::vector<int> & codes,
                                  GenerationState & state, StageDump & dump, StageTimes & timing) {
    engine.ensure_detok();
    state.code_frames = (int) codes.size();
    state.latent_frames = state.code_frames * AUDIO_CODE_FRAME_RATIO;
    state.context_latents.assign((size_t) AUDIO_LATENT_CHANNELS * state.latent_frames, 0.0f);
    if (detok_model_decode(engine.detok, codes.data(), state.code_frames,
                           state.context_latents.data()) != state.latent_frames) {
        throw std::runtime_error("acestep engine: FSQ detokenizer failed");
    }
    timing.mark("detok");
    dump.write("02_detok_latent", state.context_latents, state.latent_frames, AUDIO_LATENT_CHANNELS);

    if (state.low_memory) {
        if (engine.detok && engine.opts.verbose) {
            fprintf(stderr, "[acestep-engine] freeing FSQ detokenizer (low-mem)\n");
        }
        engine.free_detok();
    }
}

template <typename EngineImpl>
static bool prepare_context_latents(EngineImpl & engine, const GenerateParams & params,
                                    GenerationState & state, const StageReporter & report,
                                    StageDump & dump, StageTimes & timing) {
    if (!state.plan.run_lm || !state.plan.run_detokenizer) {
        timing.mark("lm");
        timing.mark("detok");
        return true;
    }

    std::vector<int> codes;
    if (!run_lm_stage(engine, params, state, report, dump, timing, codes)) return false;
    run_detokenizer_stage(engine, codes, state, dump, timing);
    return true;
}

struct PromptTokens {
    std::vector<int32_t> text;
    std::vector<int32_t> lyrics;
};

#ifdef ACESTEP_PARITY_DEBUG
static void print_token_range(const std::vector<int> & tokens, int first, int last) {
    for (int index = first; index < last; ++index) fprintf(stderr, " %d", tokens[(size_t) index]);
}

static void print_prompt_tokens(const std::vector<int> & text, const std::vector<int> & lyrics) {
    fprintf(stderr, "[acestep-dbg] S_text=%zu S_lyric=%zu (enc_S=%zu)\n",
            text.size(), lyrics.size(), text.size() + lyrics.size());
    fprintf(stderr, "[acestep-dbg] text_ids[0..8]:");
    print_token_range(text, 0, std::min(8, (int) text.size()));
    fprintf(stderr, " ... tail:");
    print_token_range(text, std::max(0, (int) text.size() - 4), (int) text.size());
    fprintf(stderr, "\n[acestep-dbg] lyric_ids[0..8]:");
    print_token_range(lyrics, 0, std::min(8, (int) lyrics.size()));
    fprintf(stderr, " ... tail:");
    print_token_range(lyrics, std::max(0, (int) lyrics.size() - 4), (int) lyrics.size());
    fprintf(stderr, "\n");
}
#endif

static PromptTokens tokenize_prompt(const BpeTokenizer & tokenizer, const AcePrompt & prompt,
                                    const std::string & language, const char * instruction) {
    const std::string metadata =
        build_metas(prompt.bpm, prompt.timesignature, prompt.keyscale, prompt.duration);
    const std::string text = std::string("# Instruction\n") + instruction +
                             "\n\n# Caption\n" + prompt.caption + "\n\n# Metas\n" +
                             metadata + "<|endoftext|>\n";
    const std::string lyrics = std::string("# Languages\n") + language +
                               "\n\n# Lyric\n" + prompt.lyrics + "<|endoftext|>";
    const std::vector<int> text_ids = bpe_encode(tokenizer, text, true);
    const std::vector<int> lyric_ids = bpe_encode(tokenizer, lyrics, true);

#ifdef ACESTEP_PARITY_DEBUG
    if (dbg_env("ACESTEP_DUMP_DIR")) print_prompt_tokens(text_ids, lyric_ids);
#endif

    return {
        std::vector<int32_t>(text_ids.begin(), text_ids.end()),
        std::vector<int32_t>(lyric_ids.begin(), lyric_ids.end())
    };
}

template <typename EngineImpl>
static PromptEncoding encode_prompt(EngineImpl & engine, const PromptTokens & tokens,
                                    GenerationState & state, StageDump & dump, StageTimes & timing,
                                    bool release_stage = true) {
    engine.ensure_textenc();
    PromptEncoding encoding;
    encoding.text_tokens = (int) tokens.text.size();
    encoding.lyric_tokens = (int) tokens.lyrics.size();
    if (!textenc_model_forward(engine.textenc, tokens.text.data(), encoding.text_tokens,
                               encoding.text_hidden)) {
        throw std::runtime_error("acestep engine: text-encoder forward failed");
    }
    if (!textenc_model_embed_lookup(engine.textenc, tokens.lyrics.data(), encoding.lyric_tokens,
                                    encoding.lyric_embedding)) {
        throw std::runtime_error("acestep engine: lyric embed lookup failed");
    }
    timing.mark("textenc");
    dump.write("04_text_hidden", encoding.text_hidden, encoding.text_tokens,
               (int) (encoding.text_hidden.size() / (size_t) encoding.text_tokens));
    dump.write("05_lyric_embed", encoding.lyric_embedding, encoding.lyric_tokens,
               (int) (encoding.lyric_embedding.size() / (size_t) encoding.lyric_tokens));

    if (state.low_memory && release_stage) {
        if (engine.textenc && engine.opts.verbose) {
            fprintf(stderr, "[acestep-engine] freeing text-encoder (low-mem)\n");
        }
        engine.free_textenc();
    }
    return encoding;
}

template <typename EngineImpl>
static void encode_cross_attention(EngineImpl & engine, const PromptEncoding & prompt,
                                   const GenerationState & state, EncoderConditioning & output,
                                   StageDump & dump, StageTimes & timing,
                                   bool release_stage = true) {
    const TimbreInput timbre = resolve_timbre_input(
        state.plan, state.conditioning.reference, state.context_latents,
        state.latent_frames, cond_model_silence_frame(engine.cond));
    if (!cond_model_forward(engine.cond, prompt.text_hidden.data(), prompt.text_tokens,
                            prompt.lyric_embedding.data(), prompt.lyric_tokens,
                            timbre.data, timbre.frames, output.hidden, &output.sequence)) {
        throw std::runtime_error("acestep engine: cond-encoder forward failed");
    }
    output.hidden_size = (int) (output.hidden.size() / (size_t) output.sequence);

    if (state.low_memory && release_stage) {
        if (engine.cond && engine.opts.verbose) {
            fprintf(stderr, "[acestep-engine] freeing cond-encoder (low-mem)\n");
        }
        engine.free_cond();
    }
    timing.mark("cond");
    dump.write("03_context", output.context, output.frames, output.context_channels);
    dump.write("06_enc_hidden", output.hidden, output.sequence, output.hidden_size);
}

static void validate_lego_model(const DitConfig & config, const GenerateTask & task) {
    if (!is_lego_task(task.type)) return;
    if (const std::string error = lego_model_error(config.is_turbo, config.is_sft); !error.empty()) {
        throw std::invalid_argument(error);
    }
}

static std::string resolve_dit_instruction(const GenerateTask & task) {
    if (is_lego_task(task.type)) return make_lego_instruction(task.track);
    return DIT_INSTR_COVER;
}

template <typename EngineImpl>
static void encode_switch_hidden(EngineImpl & engine, const PromptEncoding & prompt,
                                 const GenerationState & state, EncoderConditioning & output) {
    const TimbreInput timbre = resolve_timbre_input(
        state.plan, state.conditioning.reference, state.context_latents,
        state.latent_frames, cond_model_silence_frame(engine.cond));
    if (!cond_model_forward(engine.cond, prompt.text_hidden.data(), prompt.text_tokens,
                            prompt.lyric_embedding.data(), prompt.lyric_tokens,
                            timbre.data, timbre.frames, output.hidden_switch,
                            &output.sequence_switch)) {
        throw std::runtime_error("acestep engine: cond-encoder switch forward failed");
    }
    if (state.low_memory) {
        if (engine.cond && engine.opts.verbose) {
            fprintf(stderr, "[acestep-engine] freeing cond-encoder (low-mem)\n");
        }
        engine.free_cond();
    }
}

template <typename EngineImpl>
static EncoderConditioning prepare_encoder_conditioning(EngineImpl & engine,
                                                         GenerationState & state,
                                                         StageDump & dump, StageTimes & timing) {
    const DitConfig & config = engine.dit_cfg;
    validate_lego_model(config, state.task);
    const int patch = config.patch_size;
    EncoderConditioning output;
    output.frames = ((state.latent_frames + patch - 1) / patch) * patch;
    output.context_channels = config.in_channels - config.out_channels;

    engine.ensure_cond();
    output.context = make_dit_context(
        state.context_latents, cond_model_silence_latent(engine.cond), output.frames,
        state.latent_frames, output.context_channels, config.out_channels);
    output.null_emb = cond_model_null_emb(engine.cond);
    const bool needs_switch = needs_cover_conditioning_switch(state.task);
    if (needs_switch) {
        output.context_switch = make_dit_context(
            state.context_latents, cond_model_silence_latent(engine.cond), output.frames,
            0, output.context_channels, config.out_channels);
    }
    const std::string instruction = resolve_dit_instruction(state.task);
    const PromptTokens tokens = tokenize_prompt(
        engine.bpe_text, state.prompt, state.language, instruction.c_str());
    output.lyric_tokens = tokens.lyrics;
    const PromptEncoding prompt = encode_prompt(engine, tokens, state, dump, timing, !needs_switch);
    encode_cross_attention(engine, prompt, state, output, dump, timing, !needs_switch);
    if (needs_switch) {
        const PromptTokens tokens_switch = tokenize_prompt(
            engine.bpe_text, state.prompt, state.language, DIT_INSTR_TEXT2MUSIC);
        const PromptEncoding prompt_switch = encode_prompt(engine, tokens_switch, state, dump, timing);
        encode_switch_hidden(engine, prompt_switch, state, output);
        const int rows = padded_conditioning_rows(output);
        pad_hidden_rows(output.hidden, rows, output.hidden_size);
        pad_hidden_rows(output.hidden_switch, rows, output.hidden_size);
    }
    return output;
}

static NoiseSchedule prepare_noise(const GenerateParams & params, const GenerationState & state,
                                   const DitConfig & config, const EncoderConditioning & conditioning,
                                   bool verbose, StageDump & dump, StageTimes & timing) {
    NoiseSchedule output;
    output.noise.resize((size_t) config.out_channels * conditioning.frames);
    philox_randn(state.seed, output.noise.data(), (int) output.noise.size(), true);
    output.steps = params.inference_steps > 0
                       ? params.inference_steps
                       : (config.is_turbo ? TURBO_STEPS : STANDARD_STEPS);
    output.shift = params.shift > 0.0f
                       ? params.shift
                       : (config.is_turbo ? TURBO_SHIFT : STANDARD_SHIFT);
    dit_build_schedule(output.shift, output.steps, output.schedule);

    if (state.plan.blend_cover_noise) {
        const CoverNoiseResult adjustment = apply_cover_noise(
            output.noise, state.context_latents, conditioning.frames, state.latent_frames,
            config.out_channels, state.task.cover_noise_strength, output.schedule);
        output.steps = adjustment.remaining_steps;
        if (verbose) {
            fprintf(stderr,
                    "[acestep-engine] cover_noise_strength=%.2f -> nearest_t=%.4f remaining_steps=%d\n",
                    state.task.cover_noise_strength, adjustment.nearest_time, output.steps);
        }
    }
    timing.mark("noise");
    dump.write("07_noise", output.noise, conditioning.frames, config.out_channels);
    return output;
}

#ifdef ACESTEP_PARITY_DEBUG
static void inject_parity_inputs(NoiseSchedule & noise, EncoderConditioning & conditioning) {
    if (const char * path = dbg_env("ACESTEP_INJECT_NOISE")) dbg_load_dump(path, noise.noise);
    if (const char * path = dbg_env("ACESTEP_INJECT_CONTEXT")) dbg_load_dump(path, conditioning.context);
    if (const char * path = dbg_env("ACESTEP_INJECT_ENC")) dbg_load_dump(path, conditioning.hidden);
}

static void dump_parity_inputs(const std::vector<float> & latent, const NoiseSchedule & noise,
                               const EncoderConditioning & conditioning, int output_channels) {
    const char * directory = dbg_env("ACESTEP_DUMP_DIR");
    if (!directory) return;
    const std::string path(directory);
    dbg_write_dump(path + "/our_dit_output.bin", latent, conditioning.frames, output_channels);
    dbg_write_dump(path + "/our_noise.bin", noise.noise, conditioning.frames, output_channels);
    dbg_write_dump(path + "/our_context.bin", conditioning.context, conditioning.frames,
                   conditioning.context_channels);
    dbg_write_dump(path + "/our_enc_hidden.bin", conditioning.hidden, conditioning.sequence,
                   conditioning.hidden_size);
}
#endif

// Cross-attention heads whose lyric alignment the reference validated for the
// official 24-layer / 16-head ACE-Step v15 DiT.
static constexpr DitAttentionHead LRC_ALIGNMENT_HEADS[] = {
    { 2, 6 }, { 3, 10 }, { 3, 11 }, { 4, 3 }, { 5, 8 }, { 5, 9 }, { 6, 8 },
};
static constexpr int   LRC_HEADS_N_LAYERS       = 24;
static constexpr int   LRC_HEADS_N_HEADS        = 16;
static constexpr float LRC_VIOLENCE_LEVEL       = 2.0f;
static constexpr int   LRC_MEDIAN_FILTER_WIDTH  = 1;

// Raw byte-level decode over arbitrary token prefixes. The alignment's
// incremental prefix decoding needs the literal text of every piece, which
// bpe_decode's tag expansion and code skipping would distort.
static std::string decode_lyric_token_ids(const BpeTokenizer & bpe, const std::vector<int> & ids) {
    std::unordered_map<std::string, char> encoded_bytes;
    encoded_bytes.reserve(256);
    for (int byte = 0; byte < 256; byte++) {
        encoded_bytes[bpe.byte2str[byte]] = (char) byte;
    }
    std::string result;
    for (int id : ids) {
        if (id == bpe.eos_id) {
            result += "<|endoftext|>";
            continue;
        }
        if (id < 0 || id >= (int) bpe.id_to_str.size()) continue;
        const std::string & piece  = bpe.id_to_str[(size_t) id];
        size_t              offset = 0;
        while (offset < piece.size()) {
            int advance = 0;
            bpe_utf8_codepoint(piece.c_str() + offset, &advance);
            if (advance <= 0 || offset + (size_t) advance > piece.size()) break;
            const std::string symbol = piece.substr(offset, (size_t) advance);
            const auto        found  = encoded_bytes.find(symbol);
            if (found != encoded_bytes.end()) {
                result.push_back(found->second);
            } else {
                result += symbol;
            }
            offset += (size_t) advance;
        }
    }
    return result;
}

struct LyricSegment {
    int start = 0;
    int end   = 0;
};

// The lyric branch opens the encoder sequence, so its token indices map
// directly onto attention rows: the segment spans from just past the encoded
// "# Languages/<lang>/# Lyric" header to the first end-of-text token.
static LyricSegment resolve_lyric_segment(const BpeTokenizer & bpe,
                                          const std::vector<int32_t> & lyric_tokens,
                                          const std::string & language, int enc_S) {
    const std::string header =
        std::string("# Languages\n") + language + "\n\n# Lyric\n";
    LyricSegment segment;
    segment.start = (int) bpe_encode(bpe, header, false).size();
    segment.end   = (int) lyric_tokens.size();
    for (int i = segment.start; i < (int) lyric_tokens.size(); i++) {
        if (lyric_tokens[(size_t) i] == bpe.eos_id) {
            segment.end = i;
            break;
        }
    }
    if (segment.start >= segment.end || segment.end > enc_S) {
        throw std::runtime_error("acestep engine: encoded lyric segment is empty or outside the attention matrix");
    }
    return segment;
}

static std::vector<lyrics::Matrix> slice_lyric_rows(const std::vector<std::vector<float>> & captured,
                                                    const LyricSegment & segment, int enc_S, int S) {
    std::vector<lyrics::Matrix> heads;
    heads.reserve(captured.size());
    for (const std::vector<float> & raw : captured) {
        const lyrics::Matrix matrix =
            lyrics::matrix_from_column_major(raw.data(), (size_t) enc_S, (size_t) S);
        lyrics::Matrix sliced((size_t) (segment.end - segment.start), matrix.cols);
        for (int token = segment.start; token < segment.end; token++) {
            memcpy(sliced.values.data() + (size_t) (token - segment.start) * matrix.cols,
                   matrix.values.data() + (size_t) token * matrix.cols,
                   matrix.cols * sizeof(float));
        }
        heads.push_back(std::move(sliced));
    }
    return heads;
}

static std::vector<DitAttentionHead> resolve_lrc_heads(const DitConfig & config) {
    if (config.n_layers != LRC_HEADS_N_LAYERS || config.n_heads != LRC_HEADS_N_HEADS) {
        throw std::runtime_error(
            "acestep engine: generate_lrc requires the official 24-layer/16-head DiT");
    }
    return std::vector<DitAttentionHead>(
        LRC_ALIGNMENT_HEADS, LRC_ALIGNMENT_HEADS + sizeof(LRC_ALIGNMENT_HEADS) / sizeof(LRC_ALIGNMENT_HEADS[0]));
}

static void align_lyric_heads(const BpeTokenizer & bpe, const std::vector<lyrics::Matrix> & heads,
                              const std::vector<int> & pure_ids, float duration,
                              GenerateResult & result) {
    const auto processed = lyrics::preprocess_alignment(heads, LRC_VIOLENCE_LEVEL, LRC_MEDIAN_FILTER_WIDTH);
    const auto decoder   = [&bpe](const std::vector<int> & ids) { return decode_lyric_token_ids(bpe, ids); };
    const std::vector<std::string> decoded = lyrics::decode_tokens_incrementally(pure_ids, decoder);
    const auto token_stamps =
        lyrics::token_timestamps(processed.calc_matrix, pure_ids, decoded, duration);
    const auto sentences = lyrics::sentence_timestamps(token_stamps, decoder);
    result.metadata.lrc  = lyrics::format_lrc(sentences);

    const auto     scoring = lyrics::preprocess_scoring(heads, LRC_MEDIAN_FILTER_WIDTH);
    lyrics::Matrix costs   = scoring.calc_matrix;
    for (float & value : costs.values) value = -value;
    const auto path       = lyrics::dtw(costs);
    const auto type_mask  = lyrics::token_type_mask(decoded);
    const auto metrics    = lyrics::compute_alignment_metrics(scoring.energy_matrix, path, type_mask);
    result.metadata.lyrics_score = lyrics::calculate_lyrics_score(metrics);
}

// One extra DiT forward at the final timestep captures the lyric
// cross-attention, then DTW aligns the lyric lines with the audio timeline.
template <typename EngineImpl>
static void run_lrc_stage(EngineImpl & engine, const GenerateParams & params,
                          const GenerationState & state, const EncoderConditioning & conditioning,
                          int num_steps, const std::vector<float> & latent,
                          StageTimes & timing, GenerateResult & result) {
    if (!params.generate_lrc) return;

    const std::vector<DitAttentionHead> heads = resolve_lrc_heads(engine.dit_cfg);
    const LyricSegment segment = resolve_lyric_segment(
        engine.bpe_text, conditioning.lyric_tokens, state.language, conditioning.sequence);

    DitAttentionProbeInputs probe;
    probe.context    = conditioning.context.data();
    probe.latent     = latent.data();
    probe.enc_hidden = conditioning.hidden.data();
    probe.T          = conditioning.frames;
    probe.enc_S      = conditioning.sequence;
    probe.H_enc      = conditioning.hidden_size;
    probe.real_enc_S = conditioning.sequence;
    probe.num_steps  = num_steps;
    probe.seed       = state.seed;

    std::vector<std::vector<float>> captured;
    if (!dit_probe_cross_attention(engine.dit, probe, heads, captured)) {
        throw std::runtime_error("acestep engine: lyric alignment probe failed");
    }
    if (state.low_memory) {
        if (engine.dit && engine.opts.verbose) fprintf(stderr, "[acestep-engine] freeing DiT (low-mem)\n");
        engine.free_dit();
    }

    const std::vector<lyrics::Matrix> lyric_heads =
        slice_lyric_rows(captured, segment, conditioning.sequence, conditioning.frames / engine.dit_cfg.patch_size);
    const std::vector<int> pure_ids(conditioning.lyric_tokens.begin() + segment.start,
                                    conditioning.lyric_tokens.begin() + segment.end);
    const float duration = (float) conditioning.frames / AUDIO_LATENT_RATE;
    align_lyric_heads(engine.bpe_text, lyric_heads, pure_ids, duration, result);
    if (result.metadata.lrc.empty()) {
        throw std::runtime_error("acestep engine: lyric alignment produced no LRC lines");
    }
    timing.mark("lrc");
}

template <typename EngineImpl>
static bool sample_dit_latent(EngineImpl & engine, const GenerateParams & params,
                              const GenerationState & state, EncoderConditioning & conditioning,
                              NoiseSchedule & noise, const StageReporter & report,
                              StageDump & dump, StageTimes & timing, std::vector<float> & latent) {
    const DitConfig & config = engine.dit_cfg;
    const float guidance = resolve_guidance_scale(params.guidance_scale, config.is_turbo);
    const int cover_switch_step = resolve_cover_switch_step(state.task, noise.steps);
    if (engine.opts.verbose) {
        fprintf(stderr,
                "[acestep-engine] DiT: turbo=%d steps=%d shift=%.2f guidance=%.2f T=%d task=%s cover_switch=%d\n",
                (int) config.is_turbo, noise.steps, noise.shift, guidance,
                conditioning.frames, state.task.type.c_str(), cover_switch_step);
    }
    engine.ensure_dit();
    if (!report("dit", 0, noise.steps)) return false;

    const bool has_switch = cover_switch_step >= 0 && !conditioning.context_switch.empty();
    DitSampleParams sample;
    sample.noise = noise.noise.data();
    sample.context_latents = conditioning.context.data();
    sample.enc_hidden = conditioning.hidden.data();
    sample.enc_S = has_switch ? padded_conditioning_rows(conditioning) : conditioning.sequence;
    sample.H_enc = conditioning.hidden_size;
    sample.T = conditioning.frames;
    sample.N = 1;
    sample.schedule = noise.schedule.data();
    sample.num_steps = noise.steps;
    sample.real_enc_S = &conditioning.sequence;
    sample.guidance_scale = guidance;
    sample.null_cond_emb = conditioning.null_emb.size() >= (size_t) conditioning.hidden_size
                               ? conditioning.null_emb.data()
                               : nullptr;
    if (has_switch) {
        sample.context_switch = conditioning.context_switch.data();
        sample.enc_hidden_switch = conditioning.hidden_switch.data();
        sample.real_enc_S_switch = &conditioning.sequence_switch;
        sample.cover_switch_step = cover_switch_step;
    }
    sample.dcw_enabled = resolve_dcw_enabled(params.dcw_enabled, config.is_turbo);
    sample.dcw_scaler = params.dcw_scaler;
    sample.dcw_high_scaler = params.dcw_high_scaler;
    sample.on_step = [&](int step, int total) { return report("dit", step, total); };

    if (!dit_sample(engine.dit, sample, latent)) {
        if (engine.cancel_flag.load()) return false;
        throw std::runtime_error("acestep engine: DiT sample failed");
    }
    if (!report("dit", noise.steps, noise.steps)) return false;
    timing.mark("dit");
    dump.write("08_dit_latent", latent, conditioning.frames, config.out_channels);
    if (state.low_memory && !params.generate_lrc) {
        if (engine.dit && engine.opts.verbose) fprintf(stderr, "[acestep-engine] freeing DiT (low-mem)\n");
        engine.free_dit();
    }

#ifdef ACESTEP_PARITY_DEBUG
    dump_parity_inputs(latent, noise, conditioning, config.out_channels);
#endif
    return true;
}

template <typename EngineImpl>
static bool decode_audio(EngineImpl & engine, const std::vector<float> & latent,
                         const EncoderConditioning & conditioning, bool low_memory,
                         const StageReporter & report, StageDump & dump,
                         StageTimes & timing, GenerateResult & result) {
    engine.ensure_vae();
    if (!report("vae", 0, 1)) return false;

    bool completed = true;
    result.pcm = engine.vae->decode(latent, conditioning.frames, [&](int done, int total) {
        completed = report("vae", done, total);
        return completed;
    });
    if (!completed) {
        result.pcm.clear();
        return false;
    }
    if (result.pcm.empty()) throw std::runtime_error("acestep engine: VAE decode failed");
    report("vae", 1, 1);
    timing.mark("vae");
    dump.write("09_vae_pcm", result.pcm, (int) (result.pcm.size() / AUDIO_CHANNELS), AUDIO_CHANNELS);
    if (engine.opts.verbose) timing.dump(stderr);

    if (low_memory) {
        if (engine.vae && engine.opts.verbose) fprintf(stderr, "[acestep-engine] freeing VAE (low-mem)\n");
        engine.free_vae();
    }
    return true;
}

// A lego stem must mix sample-for-sample over its source. The decode length is
// a whole multiple of the VAE hop rounded up to the DiT patch size, so it can
// overshoot the source (trim) or undershoot it by up to one hop (pad silence).
static void match_stem_to_source_length(const GenerateParams & params, const GenerationState & state,
                                        GenerateResult & result) {
    if (!is_lego_task(state.task.type)) return;
    result.pcm.resize(params.source_audio.size(), 0.0f);
}

static AcePrompt make_edit_prompt(const GenerateParams & params, const std::string & caption,
                                  const std::string & lyrics, float duration,
                                  const std::string & language) {
    AcePrompt prompt = make_prompt(params, language);
    if (!caption.empty() && params.augment_caption_with_metadata) {
        prompt.caption =
            build_conditioning_caption(caption, prompt.bpm, prompt.timesignature, prompt.keyscale);
    } else if (!caption.empty()) {
        prompt.caption = caption;
    }
    prompt.lyrics = lyrics.empty() ? params.lyrics : lyrics;
    if (prompt.lyrics.empty()) prompt.lyrics = INSTRUMENTAL_LYRICS;
    prompt.duration = duration;
    return prompt;
}

static float audio_duration_seconds(int samples) {
    return (float) samples / AUDIO_SAMPLE_RATE;
}

static int round_edit_frames(int latent_frames, int patch_size) {
    return ((latent_frames + patch_size - 1) / patch_size) * patch_size;
}

static void fill_padded_edit_source_frames(
    std::vector<float> & padded, const std::vector<float> & latent,
    int source_frames, int target_frames, const std::vector<float> & silence,
    int silence_frames) {
    for (int frame = 0; frame < target_frames; ++frame) {
        const float * source = nullptr;
        if (frame < source_frames) {
            source = latent.data() + (size_t) frame * AUDIO_LATENT_CHANNELS;
        } else if (silence_frames > 0) {
            const int silence_index = (frame - source_frames) % silence_frames;
            source = silence.data() + (size_t) silence_index * AUDIO_LATENT_CHANNELS;
        }
        if (source) {
            memcpy(padded.data() + (size_t) frame * AUDIO_LATENT_CHANNELS,
                   source, AUDIO_LATENT_CHANNELS * sizeof(float));
        }
    }
}

static std::vector<float> pad_edit_source(const std::vector<float> & latent,
                                          int source_frames, int target_frames,
                                          const std::vector<float> & silence) {
    std::vector<float> padded(
        (size_t) target_frames * AUDIO_LATENT_CHANNELS, EDIT_EMPTY_LATENT);
    const int silence_frames =
        (int) (silence.size() / (size_t) AUDIO_LATENT_CHANNELS);
    fill_padded_edit_source_frames(
        padded, latent, source_frames, target_frames, silence, silence_frames);
    return padded;
}

static void fill_repaint_context_frames(std::vector<float> & context,
                                        const std::vector<float> & clean_source,
                                        const std::vector<float> & silence,
                                        const std::vector<float> & mask,
                                        int context_channels, int silence_frames) {
    for (size_t frame = 0; frame < mask.size(); ++frame) {
        float * destination = context.data() + frame * context_channels;
        const float * acoustic =
            mask[frame] == REPAINT_MASKED_FRAME
                ? silence.data() + (frame % (size_t) silence_frames) * AUDIO_LATENT_CHANNELS
                : clean_source.data() + frame * AUDIO_LATENT_CHANNELS;
        memcpy(destination, acoustic, AUDIO_LATENT_CHANNELS * sizeof(float));
        std::fill(destination + AUDIO_LATENT_CHANNELS,
                  destination + context_channels, mask[frame]);
    }
}

static std::vector<float> make_repaint_context(const std::vector<float> & clean_source,
                                               const std::vector<float> & silence,
                                               const std::vector<float> & mask,
                                               int context_channels) {
    if (context_channels != AUDIO_LATENT_CHANNELS * EDIT_CONTEXT_PLANES) {
        throw std::runtime_error(EDIT_ERROR_REPAINT_CONTEXT);
    }
    const int silence_frames =
        (int) (silence.size() / (size_t) AUDIO_LATENT_CHANNELS);
    if (silence_frames <= 0) {
        throw std::runtime_error(EDIT_ERROR_REPAINT_SILENCE);
    }
    std::vector<float> context(mask.size() * (size_t) context_channels);
    fill_repaint_context_frames(
        context, clean_source, silence, mask, context_channels, silence_frames);
    return context;
}

template <typename EngineImpl>
static EncoderConditioning prepare_edit_encoder_conditioning(
    EngineImpl & engine, GenerationState & state, std::vector<float> context,
    int frames, const char * instruction, StageDump & dump, StageTimes & timing) {
    EncoderConditioning output;
    output.frames = frames;
    output.context_channels = engine.dit_cfg.in_channels - engine.dit_cfg.out_channels;
    output.context = std::move(context);

    engine.ensure_cond();
    const PromptTokens tokens =
        tokenize_prompt(engine.bpe_text, state.prompt, state.language, instruction);
    const PromptEncoding prompt = encode_prompt(engine, tokens, state, dump, timing);
    encode_cross_attention(engine, prompt, state, output, dump, timing);
    return output;
}

static NoiseSchedule make_edit_schedule(const GenerateParams & params,
                                        const DitConfig & config) {
    NoiseSchedule noise;
    noise.steps = params.inference_steps > 0
                      ? params.inference_steps
                      : (config.is_turbo ? TURBO_STEPS : STANDARD_STEPS);
    noise.shift = params.shift > 0.0f
                      ? params.shift
                      : (config.is_turbo ? TURBO_SHIFT : STANDARD_SHIFT);
    dit_build_schedule(noise.shift, noise.steps, noise.schedule);
    return noise;
}

static NoiseSchedule make_edit_noise(const GenerateParams & params, long long seed,
                                     const DitConfig & config, int frames) {
    NoiseSchedule noise = make_edit_schedule(params, config);
    noise.noise.resize((size_t) config.out_channels * frames);
    philox_randn(seed, noise.noise.data(), (int) noise.noise.size(), true);
    return noise;
}

struct EditPlanState {
    GenerationState generation;
    int output_samples = 0;
    long long operation_seed = 0;
};

struct RepaintOperationState {
    RepaintRange range;
    RepaintConfig config;
    int frames = 0;
    std::vector<float> clean_source;
    std::vector<float> mask;
    EncoderConditioning conditioning;
    NoiseSchedule noise;
};

struct FlowOperationState {
    int frames = 0;
    uint64_t seed = 0;
    std::vector<float> source;
    EncoderConditioning source_conditioning;
    EncoderConditioning target_conditioning;
    NoiseSchedule schedule;
};

static void populate_metadata(const GenerationState & state, GenerateResult & result);

static void validate_edit_plan_request(const GenerateParams & params) {
    if (params.task_type != TASK_TEXT2MUSIC) {
        throw std::invalid_argument(
            std::string(EDIT_ERROR_TASK_TYPE) + params.task_type +
            EDIT_TASK_TYPE_SUFFIX);
    }
    if (params.source_audio.empty()) {
        throw std::invalid_argument(EDIT_ERROR_SOURCE_REQUIRED);
    }
    if (params.source_audio.size() % AUDIO_CHANNELS != 0) {
        throw std::invalid_argument(EDIT_ERROR_SOURCE_STEREO);
    }
    if (!params.audio_codes.empty()) {
        throw std::invalid_argument(EDIT_ERROR_AUDIO_CODES);
    }
}

static EditPlanState make_edit_plan_state(const GenerateParams & params, bool keep_stages) {
    EditPlanState state;
    state.generation = make_generation_state(params, keep_stages);
    state.output_samples = (int) (params.source_audio.size() / AUDIO_CHANNELS);
    state.operation_seed = state.generation.seed;
    state.generation.prompt.duration = audio_duration_seconds(state.output_samples);
    state.generation.plan.run_lm = false;
    state.generation.plan.run_detokenizer = false;
    return state;
}

template <typename EngineImpl>
static bool encode_edit_plan_inputs(EngineImpl & engine, const GenerateParams & params,
                                    EditPlanState & state, const StageReporter & report,
                                    StageDump & dump, StageTimes & timing,
                                    AudioEditArtifact & artifact) {
    engine.ensure_vae(true);
    EncodedAudio source;
    if (!encode_audio(*engine.vae, params.source_audio, EDIT_STAGE_SOURCE,
                      EDIT_DUMP_SOURCE_LATENT, report, engine.opts.verbose,
                      dump, timing, source)) {
        return false;
    }
    if (!params.reference_audio.empty() &&
        !encode_audio(*engine.vae, params.reference_audio, EDIT_STAGE_REFERENCE,
                      EDIT_DUMP_REFERENCE_LATENT, report, engine.opts.verbose,
                      dump, timing, state.generation.conditioning.reference)) {
        return false;
    }
    if (state.generation.low_memory) engine.free_vae();
    artifact.latent = std::move(source.latent);
    artifact.latent_frames = source.frames;
    artifact.pcm = params.source_audio;
    artifact.pcm_is_current = true;
    return true;
}

template <typename EngineImpl>
static void materialize_repaint_source(EngineImpl & engine, const EditPlanState & state,
                                       AudioEditArtifact & artifact,
                                       const StageReporter & report) {
    engine.ensure_vae(true);
    bool completed = true;
    std::vector<float> materialized = engine.vae->decode(
        artifact.latent, artifact.latent_frames, [&](int done, int total) {
            completed = report(EDIT_STAGE_VAE, done, total);
            return completed;
        });
    if (!completed) return;
    if (materialized.empty()) {
        throw std::runtime_error(EDIT_ERROR_INTERMEDIATE_DECODE);
    }
    materialized.resize(std::min(
        materialized.size(), (size_t) state.output_samples * AUDIO_CHANNELS));
    if (artifact.pending_waveform_splice) {
        repaint_splice_waveform(
            materialized, artifact.pcm, artifact.pending_range.sample_start,
            artifact.pending_range.sample_end, artifact.pending_crossfade_samples);
    }
    artifact.pcm = std::move(materialized);
    artifact.pcm_is_current = true;
    artifact.pending_waveform_splice = false;
    artifact.latent = engine.vae->encode(
        artifact.pcm, (int) artifact.pcm.size() / AUDIO_CHANNELS,
        &artifact.latent_frames, [&](int done, int total) {
            completed = report(EDIT_STAGE_VAE, done, total);
            return completed;
        });
    if (!completed) return;
    if (artifact.latent.empty() || artifact.latent_frames <= 0) {
        throw std::runtime_error(EDIT_ERROR_INTERMEDIATE_ENCODE);
    }
    if (state.generation.low_memory) engine.free_vae();
}

static RepaintRange resolve_repaint_operation_range(const RepaintParams & edit,
                                                    const AudioEditArtifact & artifact) {
    if (edit.mode == RepaintMode::Balanced &&
        (!std::isfinite(edit.strength) ||
         edit.strength < REPAINT_MIN_STRENGTH ||
         edit.strength > REPAINT_MAX_STRENGTH)) {
        throw std::invalid_argument(EDIT_ERROR_REPAINT_STRENGTH);
    }
    RepaintRange range;
    const std::string error = resolve_repaint_range(
        edit.start_seconds, edit.end_seconds,
        (int) artifact.pcm.size() / AUDIO_CHANNELS,
        artifact.latent_frames, range);
    if (!error.empty()) {
        throw std::invalid_argument(std::string(ENGINE_ERROR_PREFIX) + error);
    }
    return range;
}

static GenerationState make_edit_generation_state(
    const EditPlanState & base, const GenerateParams & params,
    const std::string & caption, const std::string & lyrics, float duration,
    const std::vector<float> & context_latents, int frames) {
    GenerationState state = base.generation;
    state.prompt = make_edit_prompt(
        params, caption, lyrics, duration, base.generation.language);
    state.context_latents = context_latents;
    state.latent_frames = frames;
    state.plan.reuse_source_reference = false;
    return state;
}

template <typename EngineImpl>
static RepaintOperationState prepare_repaint_operation(
    EngineImpl & engine, const GenerateParams & params, EditPlanState & base,
    const RepaintParams & edit, const AudioEditArtifact & artifact,
    StageDump & dump, StageTimes & timing) {
    RepaintOperationState operation;
    operation.range = resolve_repaint_operation_range(edit, artifact);
    operation.config = resolve_repaint_config(edit.mode, edit.strength);
    operation.frames =
        round_edit_frames(artifact.latent_frames, engine.dit_cfg.patch_size);
    engine.ensure_cond();
    const std::vector<float> & silence = cond_model_silence_latent(engine.cond);
    operation.clean_source = pad_edit_source(
        artifact.latent, artifact.latent_frames, operation.frames, silence);
    operation.mask = make_repaint_mask(
        operation.frames, operation.range.latent_start, operation.range.latent_end);
    std::vector<float> context = make_repaint_context(
        operation.clean_source, silence, operation.mask,
        engine.dit_cfg.in_channels - engine.dit_cfg.out_channels);
    GenerationState state = make_edit_generation_state(
        base, params, edit.caption, edit.lyrics,
        audio_duration_seconds((int) artifact.pcm.size() / AUDIO_CHANNELS),
        operation.clean_source, operation.frames);
    operation.conditioning = prepare_edit_encoder_conditioning(
        engine, state, std::move(context), operation.frames,
        DIT_INSTR_REPAINT, dump, timing);
    operation.noise = make_edit_noise(
        params, base.operation_seed++, engine.dit_cfg, operation.frames);
    dump.write(EDIT_DUMP_NOISE, operation.noise.noise,
               operation.frames, engine.dit_cfg.out_channels);
#ifdef ACESTEP_PARITY_DEBUG
    inject_parity_inputs(operation.noise, operation.conditioning);
#endif
    return operation;
}

static DitSampleParams make_repaint_sample(
    const GenerateParams & params, RepaintOperationState & operation,
    const StageReporter & report) {
    DitSampleParams sample;
    sample.noise = operation.noise.noise.data();
    sample.context_latents = operation.conditioning.context.data();
    sample.enc_hidden = operation.conditioning.hidden.data();
    sample.enc_S = operation.conditioning.sequence;
    sample.H_enc = operation.conditioning.hidden_size;
    sample.T = operation.frames;
    sample.N = DIT_BATCH_SIZE;
    sample.schedule = operation.noise.schedule.data();
    sample.num_steps = operation.noise.steps;
    sample.real_enc_S = &operation.conditioning.sequence;
    sample.dcw_enabled = params.dcw_enabled;
    sample.dcw_scaler = params.dcw_scaler;
    sample.dcw_high_scaler = params.dcw_high_scaler;
    sample.repaint_mask = operation.mask.data();
    sample.clean_source_latents = operation.clean_source.data();
    sample.repaint_injection_ratio = operation.config.injection_ratio;
    sample.repaint_crossfade_frames = operation.config.latent_blend_frames;
    sample.repaint_preserve_latent = operation.config.preserve_waveform;
    sample.on_step = [&report](int step, int total) {
        return report(EDIT_STAGE_REPAINT, step, total);
    };
    return sample;
}

template <typename EngineImpl>
static bool sample_repaint_operation(
    EngineImpl & engine, const GenerateParams & params,
    RepaintOperationState & operation, const StageReporter & report,
    StageDump & dump, std::vector<float> & generated) {
    engine.ensure_dit();
    if (!report(EDIT_STAGE_REPAINT, EDIT_PROGRESS_START, operation.noise.steps)) {
        return false;
    }
    DitSampleParams sample = make_repaint_sample(params, operation, report);
    if (!dit_sample(engine.dit, sample, generated)) {
        if (engine.cancel_flag.load()) return false;
        throw std::runtime_error(EDIT_ERROR_REPAINT_SAMPLE);
    }
    dump.write(EDIT_DUMP_DIT_LATENT, generated,
               operation.frames, engine.dit_cfg.out_channels);
#ifdef ACESTEP_PARITY_DEBUG
    dump_parity_inputs(generated, operation.noise, operation.conditioning,
                       engine.dit_cfg.out_channels);
#endif
    report(EDIT_STAGE_REPAINT, operation.noise.steps, operation.noise.steps);
    return true;
}

static void apply_repaint_result(AudioEditArtifact & artifact,
                                 const RepaintOperationState & operation,
                                 std::vector<float> generated) {
    artifact.latent = std::move(generated);
    artifact.latent_frames = operation.frames;
    artifact.pcm_is_current = false;
    artifact.pending_waveform_splice = operation.config.preserve_waveform;
    artifact.pending_range = operation.range;
    artifact.pending_crossfade_samples =
        (int) (operation.config.waveform_fade_sec * AUDIO_SAMPLE_RATE);
}

template <typename EngineImpl>
static void run_repaint_operation(
    EngineImpl & engine, const GenerateParams & params, EditPlanState & state,
    const RepaintParams & edit, AudioEditArtifact & artifact,
    const StageReporter & report, StageDump & dump, StageTimes & timing) {
    RepaintOperationState operation = prepare_repaint_operation(
        engine, params, state, edit, artifact, dump, timing);
    std::vector<float> generated;
    if (!sample_repaint_operation(
            engine, params, operation, report, dump, generated)) {
        return;
    }
    if (state.generation.low_memory) engine.free_dit();
    apply_repaint_result(artifact, operation, std::move(generated));
}

static void validate_flow_operation(const FlowEditParams & edit,
                                    const DitConfig & config) {
    const std::string error = validate_flow_edit_params(edit);
    if (!error.empty()) {
        throw std::invalid_argument(std::string(ENGINE_ERROR_PREFIX) + error);
    }
    if (!config.is_turbo) {
        throw std::invalid_argument(EDIT_ERROR_FLOW_TURBO);
    }
}

template <typename EngineImpl>
static EncoderConditioning prepare_flow_conditioning(
    EngineImpl & engine, const GenerateParams & params,
    const EditPlanState & base, const std::string & caption,
    const std::string & lyrics, const std::vector<float> & silence_source,
    std::vector<float> context, int frames, StageDump & dump,
    StageTimes & timing) {
    GenerationState state = make_edit_generation_state(
        base, params, caption, lyrics,
        audio_duration_seconds(base.output_samples), silence_source, frames);
    return prepare_edit_encoder_conditioning(
        engine, state, std::move(context), frames,
        DIT_INSTR_TEXT2MUSIC, dump, timing);
}

template <typename EngineImpl>
static FlowOperationState prepare_flow_operation(
    EngineImpl & engine, const GenerateParams & params, EditPlanState & base,
    const FlowEditParams & edit, const AudioEditArtifact & artifact,
    StageDump & dump, StageTimes & timing) {
    validate_flow_operation(edit, engine.dit_cfg);
    FlowOperationState operation;
    operation.frames =
        round_edit_frames(artifact.latent_frames, engine.dit_cfg.patch_size);
    engine.ensure_cond();
    const std::vector<float> & silence = cond_model_silence_latent(engine.cond);
    if (silence.empty()) {
        throw std::runtime_error(EDIT_ERROR_FLOW_SILENCE);
    }
    operation.source = pad_edit_source(
        artifact.latent, artifact.latent_frames, operation.frames, silence);
    std::vector<float> silence_source =
        pad_edit_source({}, EDIT_NO_SOURCE_FRAMES, operation.frames, silence);
    std::vector<float> context = make_dit_context(
        silence_source, silence, operation.frames, operation.frames,
        engine.dit_cfg.in_channels - engine.dit_cfg.out_channels,
        engine.dit_cfg.out_channels);
    operation.source_conditioning = prepare_flow_conditioning(
        engine, params, base, edit.source_caption, edit.source_lyrics,
        silence_source, context, operation.frames, dump, timing);
    operation.target_conditioning = prepare_flow_conditioning(
        engine, params, base, edit.target_caption, edit.target_lyrics,
        silence_source, std::move(context), operation.frames, dump, timing);
    operation.seed = (uint64_t) base.operation_seed++;
    operation.schedule = make_edit_schedule(params, engine.dit_cfg);
    return operation;
}

static DitFlowEditParams make_flow_sample(
    const FlowEditParams & edit, FlowOperationState & operation,
    const StageReporter & report) {
    DitFlowEditParams sample;
    sample.source_latents = operation.source.data();
    sample.T = operation.frames;
    sample.schedule = operation.schedule.schedule.data();
    sample.num_steps = operation.schedule.steps;
    sample.n_min = edit.n_min;
    sample.n_max = edit.n_max;
    sample.n_avg = edit.n_avg;
    sample.seed = operation.seed;
    sample.source = {
        operation.source_conditioning.context.data(),
        operation.source_conditioning.hidden.data(),
        operation.source_conditioning.sequence,
        operation.source_conditioning.hidden_size,
        operation.source_conditioning.sequence,
    };
    sample.target = {
        operation.target_conditioning.context.data(),
        operation.target_conditioning.hidden.data(),
        operation.target_conditioning.sequence,
        operation.target_conditioning.hidden_size,
        operation.target_conditioning.sequence,
    };
    sample.on_step = [&report](int step, int total) {
        return report(EDIT_STAGE_FLOW, step, total);
    };
    return sample;
}

template <typename EngineImpl>
static bool sample_flow_operation(
    EngineImpl & engine, const FlowEditParams & edit,
    FlowOperationState & operation, const StageReporter & report,
    std::vector<float> & generated) {
    engine.ensure_dit();
    DitFlowEditParams sample = make_flow_sample(edit, operation, report);
    if (dit_flow_edit(engine.dit, sample, generated)) return true;
    if (engine.cancel_flag.load()) return false;
    throw std::runtime_error(EDIT_ERROR_FLOW_SAMPLE);
}

static void apply_flow_result(AudioEditArtifact & artifact,
                              const FlowOperationState & operation,
                              std::vector<float> generated) {
    artifact.latent = std::move(generated);
    artifact.latent_frames = operation.frames;
    artifact.pcm_is_current = false;
    artifact.pending_waveform_splice = false;
}

template <typename EngineImpl>
static void run_flow_operation(
    EngineImpl & engine, const GenerateParams & params, EditPlanState & state,
    const FlowEditParams & edit, AudioEditArtifact & artifact,
    const StageReporter & report, StageDump & dump, StageTimes & timing) {
    FlowOperationState operation = prepare_flow_operation(
        engine, params, state, edit, artifact, dump, timing);
    std::vector<float> generated;
    if (!sample_flow_operation(engine, edit, operation, report, generated)) return;
    if (state.generation.low_memory) engine.free_dit();
    apply_flow_result(artifact, operation, std::move(generated));
}

template <typename EngineImpl>
static AudioEditCapabilities make_edit_capabilities(
    EngineImpl & engine, const GenerateParams & params, EditPlanState & state,
    const StageReporter & report, StageDump & dump, StageTimes & timing) {
    EngineImpl * engine_ptr = &engine;
    const GenerateParams * params_ptr = &params;
    EditPlanState * state_ptr = &state;
    const StageReporter * report_ptr = &report;
    StageDump * dump_ptr = &dump;
    StageTimes * timing_ptr = &timing;
    AudioEditCapabilities capabilities;
    capabilities.cancelled = [engine_ptr] {
        return engine_ptr->cancel_flag.load();
    };
    capabilities.prepare_repaint_source = [engine_ptr, state_ptr, report_ptr](
                                                AudioEditArtifact & artifact) {
        materialize_repaint_source(
            *engine_ptr, *state_ptr, artifact, *report_ptr);
    };
    capabilities.repaint = [engine_ptr, params_ptr, state_ptr, report_ptr,
                            dump_ptr, timing_ptr](
                               const RepaintParams & edit,
                               AudioEditArtifact & artifact) {
        run_repaint_operation(
            *engine_ptr, *params_ptr, *state_ptr, edit, artifact,
            *report_ptr, *dump_ptr, *timing_ptr);
    };
    capabilities.flow_edit = [engine_ptr, params_ptr, state_ptr, report_ptr,
                              dump_ptr, timing_ptr](
                                 const FlowEditParams & edit,
                                 AudioEditArtifact & artifact) {
        run_flow_operation(
            *engine_ptr, *params_ptr, *state_ptr, edit, artifact,
            *report_ptr, *dump_ptr, *timing_ptr);
    };
    return capabilities;
}

static void apply_pending_waveform_splice(const AudioEditArtifact & artifact,
                                          std::vector<float> & pcm) {
    if (!artifact.pending_waveform_splice) return;
    repaint_splice_waveform(
        pcm, artifact.pcm, artifact.pending_range.sample_start,
        artifact.pending_range.sample_end, artifact.pending_crossfade_samples);
}

template <typename EngineImpl>
static GenerateResult decode_edit_plan_result(
    EngineImpl & engine, const EditPlanState & state,
    const AudioEditArtifact & artifact, const StageReporter & report,
    StageDump & dump) {
    GenerateResult result;
    result.sample_rate = engine.sr;
    result.channels = AUDIO_CHANNELS;
    engine.ensure_vae();
    if (!report(EDIT_STAGE_VAE, EDIT_PROGRESS_START, EDIT_PROGRESS_TOTAL)) {
        return result;
    }
    bool completed = true;
    result.pcm = engine.vae->decode(
        artifact.latent, artifact.latent_frames, [&](int done, int total) {
          completed = report(EDIT_STAGE_VAE, done, total);
          return completed;
        });
    if (!completed) {
        result.pcm.clear();
        return result;
    }
    if (result.pcm.empty()) {
        throw std::runtime_error(EDIT_ERROR_FINAL_DECODE);
    }
    dump.write(EDIT_DUMP_VAE_AUDIO, result.pcm,
               (int) (result.pcm.size() / AUDIO_CHANNELS), AUDIO_CHANNELS);
    result.pcm.resize(std::min(
        result.pcm.size(), (size_t) state.output_samples * AUDIO_CHANNELS));
    apply_pending_waveform_splice(artifact, result.pcm);
    report(EDIT_STAGE_VAE, EDIT_PROGRESS_COMPLETE, EDIT_PROGRESS_TOTAL);
    if (state.generation.low_memory) engine.free_vae();
    populate_metadata(state.generation, result);
    return result;
}

template <typename EngineImpl>
static GenerateResult run_audio_edit_plan(EngineImpl & engine,
                                          const GenerateParams & params,
                                          const StageReporter & report,
                                          StageDump & dump, StageTimes & timing) {
    GenerateResult result;
    result.sample_rate = engine.sr;
    result.channels = AUDIO_CHANNELS;
    validate_edit_plan_request(params);
    EditPlanState state = make_edit_plan_state(params, engine.keep_stages);
    AudioEditArtifact artifact;
    if (!encode_edit_plan_inputs(
            engine, params, state, report, dump, timing, artifact)) {
        return result;
    }
    const AudioEditPipeline pipeline = make_audio_edit_pipeline(params.edit_plan);
    const AudioEditCapabilities capabilities = make_edit_capabilities(
        engine, params, state, report, dump, timing);
    pipeline.execute(artifact, capabilities);
    if (engine.cancel_flag.load()) return result;
    return decode_edit_plan_result(engine, state, artifact, report, dump);
}

static void populate_metadata(const GenerationState & state, GenerateResult & result) {
    result.metadata.caption =
        state.original_caption.empty() ? state.prompt.caption : state.original_caption;
    result.metadata.lyrics = state.prompt.lyrics;
    result.metadata.keyscale = state.prompt.keyscale;
    result.metadata.vocal_language = state.prompt.vocal_language;
    result.metadata.bpm = state.prompt.bpm;
    result.metadata.timesignature =
        state.prompt.timesignature.empty() ? 0 : atoi(state.prompt.timesignature.c_str());
    result.metadata.seed = state.seed;
    result.metadata.n_codes = state.code_frames;
    result.metadata.quality_score = state.quality_score;
    result.metadata.quality_report = state.quality_report;
}

GenerateResult Engine::generate(const GenerateParams & params, const ProgressFn & progress) const {
    Impl & engine = *impl_;
    CancellationScope cancellation_scope(engine.cancel_flag);

    GenerateResult result;
    result.sample_rate = engine.sr;
    result.channels = AUDIO_CHANNELS;

    if (engine.cancel_flag.load()) {
        return result;
    }

    const StageReporter report = [&](const char * stage, int step, int total) {
        if (progress && !progress(stage, step, total)) {
            engine.cancel_flag.store(true);
            return false;
        }
        return !engine.cancel_flag.load();
    };
    StageTimes timing;
    StageDump dump(engine.opts.dump_stages_dir, engine.opts.verbose);
    if (!params.edit_plan.empty()) {
        return run_audio_edit_plan(engine, params, report, dump, timing);
    }

    GenerationState state = make_generation_state(params, engine.keep_stages);
    if (!prepare_audio_conditioning(engine, params, state, report, dump, timing)) return result;
    if (!prepare_context_latents(engine, params, state, report, dump, timing)) return result;

    EncoderConditioning conditioning = prepare_encoder_conditioning(engine, state, dump, timing);
    NoiseSchedule noise = prepare_noise(
        params, state, engine.dit_cfg, conditioning, engine.opts.verbose, dump, timing);

#ifdef ACESTEP_PARITY_DEBUG
    inject_parity_inputs(noise, conditioning);
#endif

    std::vector<float> latent;
    if (!sample_dit_latent(engine, params, state, conditioning, noise,
                           report, dump, timing, latent)) {
        return result;
    }
    run_lrc_stage(engine, params, state, conditioning, noise.steps, latent, timing, result);
    if (!decode_audio(engine, latent, conditioning, state.low_memory,
                      report, dump, timing, result)) {
        return result;
    }
    match_stem_to_source_length(params, state, result);
    if (params.normalize_loudness && !is_lego_task(state.task.type)) {
        normalize_loudness(result.pcm);
    }
    populate_metadata(state, result);
    return result;
}

static void validate_understand_request(const UnderstandParams & params) {
    if (params.audio.empty()) {
        throw std::invalid_argument("acestep engine: understand requires audio");
    }
    if ((params.audio.size() & 1u) != 0) {
        throw std::invalid_argument("acestep engine: understand audio must be interleaved stereo");
    }
}

UnderstandResult Engine::understand(const UnderstandParams & params, const ProgressFn & progress) const {
    Impl & engine = *impl_;
    CancellationScope cancellation_scope(engine.cancel_flag);

    UnderstandResult result;
    if (engine.cancel_flag.load()) {
        return result;
    }
    validate_understand_request(params);
    result.seed = resolve_seed(params.seed);

    const StageReporter report = [&](const char * stage, int step, int total) {
        if (progress && !progress(stage, step, total)) {
            engine.cancel_flag.store(true);
            return false;
        }
        return !engine.cancel_flag.load();
    };
    const bool low_memory = !engine.keep_stages;
    StageTimes timing;
    StageDump  dump(engine.opts.dump_stages_dir, engine.opts.verbose);

    engine.ensure_vae(true);
    EncodedAudio source;
    if (!encode_audio(*engine.vae, params.audio, "source", "00_source_latent", report,
                      engine.opts.verbose, dump, timing, source)) {
        return result;
    }
    if (low_memory) engine.free_vae();

    if (!report("tok", 0, 1)) return result;
    engine.ensure_tok();
    std::vector<int> codes;
    if (tok_model_encode(engine.tok, source.latent.data(), source.frames, codes) <= 0) {
        throw std::runtime_error("acestep engine: FSQ tokenize failed");
    }
    if (low_memory) engine.free_tok();
    timing.mark("tok");
    dump.write_ints("01_tok_codes", codes);
    if (!report("tok", 1, 1)) return result;

    engine.ensure_lm();
    LmSampleParams sample;
    sample.temperature = params.lm_temperature;
    sample.top_p       = params.lm_top_p;
    sample.top_k       = params.lm_top_k;
    sample.seed        = (uint32_t) result.seed;
    sample.verbose     = engine.opts.verbose;
    sample.on_step     = [&](int current, int total) { return report("understand", current, total); };

    AcePrompt parsed;
    if (!lm_understand(engine.lm, engine.bpe_lm, codes, sample, params.vocal_language, parsed)) {
        if (engine.cancel_flag.load()) return result;
        throw std::runtime_error("acestep engine: LM understand failed");
    }
    if (low_memory) engine.free_lm();
    timing.mark("understand");

    result.caption        = parsed.caption;
    result.bpm            = parsed.bpm;
    result.duration       = parsed.duration;
    result.keyscale       = parsed.keyscale;
    result.timesignature  = parsed.timesignature;
    result.vocal_language = parsed.vocal_language;
    result.audio_codes    = std::move(codes);
    return result;
}

void        Engine::cancel() const { impl_->cancel_flag.store(true); }
int         Engine::sample_rate() const { return impl_->sr; }  // cached; VAE loaded lazily
std::string Engine::backend_name() const { return impl_->backend ? ggml_backend_name(impl_->backend) : "cpu"; }

GpuFallbackReason Engine::gpu_fallback_reason() const { return impl_->gpu_fallback_reason; }

} // namespace tts_cpp::acestep
