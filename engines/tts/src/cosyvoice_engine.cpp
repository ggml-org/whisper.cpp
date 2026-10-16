// CosyVoice3 engine — native CPU pipeline (Qwen2 LM -> DiT flow -> CausalHiFT).
//
// Wires the public API (construction, option plumbing, batch + streaming
// synthesis, cancellation, backend reporting) to the validated ggml pipeline
// in cosyvoice_pipeline.cpp.  synthesize() now returns real 24 kHz speech, not
// placeholder audio.
//
// Model directory layout (resolve_component discovers these under model_dir):
//   cosyvoice3-llm*.gguf      Qwen2.5-0.5B speech LM
//   cosyvoice3-flow*.gguf     DiT conditional-flow-matching estimator
//   cosyvoice3-hift*.gguf     CausalHiFT vocoder
//   voice.gguf                baked default voice (prompt tensors)
//   vocab.json / merges.txt   Qwen2 byte-level BPE frontend
//   cosyvoice3-s3tok*.gguf    speech_tokenizer_v3 (only with reference_audio)
//   cosyvoice3-campplus*.gguf CAM++ speaker encoder (only with reference_audio)
//
// Zero-shot / cross-lingual cloning: with reference_audio set, the native
// front-end (cosyvoice_frontend.cpp) bakes the four prompt tensors from the
// recording at construction, replacing voice.gguf's; prompt_text selects the
// mode (transcript given = zero-shot, empty = cross-lingual).

#include "tts-cpp/cosyvoice/engine.h"

#include "backend_selection.h"
#include "backend_util.h"
#include "cosyvoice_frontend.h"
#include "cosyvoice_instruct.h"
#include "cosyvoice_pipeline.h"
#include "qwen_tokenizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts_cpp::cosyvoice {

// The public native rate (engine.h) must match the vocoder's internal rate
// (cosyvoice_pipeline.h). This is the one place both are visible; assert they
// agree so a change to either side fails the build instead of drifting.
static_assert(kNativeSampleRate == kCosyvoiceNativeSampleRate,
              "SynthesisResult sample rate must match the vocoder native rate");

namespace {

// CosyVoice3 speech tokens run at 25 Hz -> 960 samples/token at 24 kHz.  Used
// to size streaming chunks so the streaming contract keeps realistic cadence.
constexpr int kSamplesPerToken = 960;

// Fallback transcript when neither the voice.gguf metadata nor an explicit
// EngineOptions.prompt_text is set (matches the stock Chinese prompt).
constexpr const char * kDefaultTranscript = "希望你以后能够做的比我还好呦。";

std::string resolve_component(const std::string & explicit_path,
                              const std::string & dir,
                              const std::string & exact_or_prefix,
                              bool exact) {
    if (!explicit_path.empty()) return explicit_path;
    if (dir.empty()) return {};
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return {};
    if (exact) {
        std::filesystem::path p = std::filesystem::path(dir) / exact_or_prefix;
        return std::filesystem::exists(p, ec) ? p.string() : std::string{};
    }
    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind(exact_or_prefix, 0) == 0 &&
            name.size() >= 5 && name.substr(name.size() - 5) == ".gguf") {
            return entry.path().string();
        }
    }
    return {};
}

// Read an i32 GGUF tensor into a vector<int>.
std::vector<int> read_i32(const model_ctx & m, const std::string & name) {
    ggml_tensor * t = cosyvoice_get(m, name);
    int n = (int)ggml_nelements(t);
    std::vector<int> out(n);
    ggml_backend_tensor_get(t, out.data(), 0, (size_t)n * sizeof(int32_t));
    return out;
}
// Read an f32 GGUF tensor into a vector<float>.
std::vector<float> read_f32(const model_ctx & m, const std::string & name) {
    ggml_tensor * t = cosyvoice_get(m, name);
    int n = (int)ggml_nelements(t);
    std::vector<float> out(n);
    ggml_backend_tensor_get(t, out.data(), 0, (size_t)n * sizeof(float));
    return out;
}

// Frees a model_ctx on scope exit (normal or exception) so stages load one at a
// time; peak resident weight memory is the largest single stage, not the sum.
struct ModelGuard {
    model_ctx & m;
    explicit ModelGuard(model_ctx & mm) : m(mm) {}
    ~ModelGuard() { cosyvoice_free(m); }
    ModelGuard(const ModelGuard &) = delete;
    ModelGuard & operator=(const ModelGuard &) = delete;
};

} // namespace

struct Engine::Impl {
    EngineOptions opts;

    // Owned by Impl and SHARED by each stage's model_ctx (which borrow it).
    // Freed in ~Impl after the last stage; the stages are loaded/freed inside
    // run(), so nothing borrows it between syntheses.
    ggml_backend_t backend = nullptr;
    // True when a GPU was present but this engine declined it (unvalidated
    // vendor, or a non-OpenCL backend), so a caller can tell "CPU by policy"
    // from "CPU because there is no GPU".
    bool gpu_present_but_unused = false;

    // llm/flow/hift are loaded on demand in run(); only their paths are kept.
    std::string llm_path_, flow_path_, hift_path_;
    std::shared_ptr<cosyvoice_coreml> coreml_lm, coreml_flow, coreml_frontend, coreml_hift, coreml_f0;
    QwenTokenizer tokenizer;

    // baked voice (from voice.gguf)
    std::vector<int>   voice_prompt_stok;   // LM prompt speech tokens
    std::vector<int>   voice_prompt_token;  // flow prompt speech tokens
    std::vector<float> voice_prompt_feat;   // prompt mel [mel_len1*80] mel-fastest
    int                voice_mel_len1 = 0;
    std::vector<float> voice_embedding;     // 192-d CAM++ embedding
    std::string        voice_prompt_text;   // baked reference transcript (voice.gguf)
    // True when the voice_* tensors were baked from opts.reference_audio at
    // construction instead of loaded from voice.gguf.  Gates the zero-shot /
    // cross-lingual branch in run(): a cloned voice must never fall back to
    // the baked transcript or kDefaultTranscript, which describe the baked
    // clip rather than the reference recording.
    bool               cloned = false;

    std::atomic_bool cancel_flag{false};

    explicit Impl(EngineOptions o) : opts(std::move(o)) {
        detail::validate_controls(opts.default_controls);
        if (!opts.model_dir.empty()) {
            std::error_code ec;
            if (!std::filesystem::is_directory(opts.model_dir, ec)) {
                throw std::runtime_error("cosyvoice: model_dir does not exist: " + opts.model_dir);
            }
        }
        const std::string llm_path   = resolve_component(opts.llm_gguf_path,  opts.model_dir, "cosyvoice3-llm",  false);
        const std::string flow_path  = resolve_component(opts.flow_gguf_path, opts.model_dir, "cosyvoice3-flow", false);
        const std::string hift_path  = resolve_component(opts.hift_gguf_path, opts.model_dir, "cosyvoice3-hift", false);
        const std::string voice_path = resolve_component(opts.voice_gguf_path, opts.model_dir, "voice.gguf",  true);
        const std::string vocab_path = resolve_component(opts.vocab_path,      opts.model_dir, "vocab.json",  true);
        const std::string merges_path= resolve_component(opts.merges_path,     opts.model_dir, "merges.txt",  true);

        auto require = [](const std::string & p, const char * what) {
            if (p.empty()) throw std::runtime_error(
                std::string("cosyvoice: missing required model component: ") + what);
        };
        require(llm_path, "LM gguf (cosyvoice3-llm*.gguf)");
        require(flow_path, "flow gguf (cosyvoice3-flow*.gguf)");
        require(hift_path, "hift gguf (cosyvoice3-hift*.gguf)");
        require(voice_path, "voice.gguf");
        require(vocab_path, "vocab.json");
        require(merges_path, "merges.txt");

        // ---- backend selection -------------------------------------------
        namespace det = ::tts_cpp::detail;
        if (!opts.backends_dir.empty())     det::set_backends_directory(opts.backends_dir);
        if (!opts.opencl_cache_dir.empty()) det::set_opencl_cache_dir(opts.opencl_cache_dir);

        // Automatic selection is constrained to the backends the CosyVoice3
        // graphs are parity-gated against (test-cosyvoice-{flow,llm,hift}-gpu):
        // OpenCL on Adreno, Metal on Apple, and Vulkan on desktop hosts.  The
        // requirement filters at the device walk, so on a multi-backend host
        // an off-policy device cannot shadow an admitted one; the per-platform
        // split (Android keeps Vulkan out) lives in cosyvoice_gpu_requirement().
        // An explicit opts.backend skips the walk and has no fallback.
        backend = cosyvoice_init_backend(opts.backend, opts.n_gpu_layers, opts.vulkan_device,
                                         &gpu_present_but_unused);
        if (!backend && !det::backend_request_is_auto(opts.backend)) {
            throw std::runtime_error("cosyvoice: requested backend '" + opts.backend + "' is not available");
        }
        if (!backend) throw std::runtime_error("cosyvoice: failed to init a compute backend");

        // ~Impl does not run when the constructor throws, so every throw
        // below this point (tokenizer/voice.gguf load, cloning validation)
        // would leak the backend without this guard; it is disarmed once
        // construction succeeds.
        struct BackendGuard {
            ggml_backend_t & b;
            bool armed = true;
            ~BackendGuard() {
                if (armed && b) { ggml_backend_free(b); b = nullptr; }
            }
        } backend_guard{backend};

        llm_path_  = llm_path;
        flow_path_ = flow_path;
        hift_path_ = hift_path;

        if (!tokenizer.load(vocab_path, merges_path)) {
            throw std::runtime_error("cosyvoice: failed to load tokenizer (" +
                                     vocab_path + ", " + merges_path + ")");
        }

        // voice.gguf: read its four tensors into vectors, then free it.
        {
            model_ctx voice_m = cosyvoice_load_gguf(
                voice_path, det::backend_is_cpu(backend) ? backend : nullptr);
            ModelGuard voice_guard(voice_m);
            voice_prompt_stok  = read_i32(voice_m, "voice/prompt_stok");
            voice_prompt_token = read_i32(voice_m, "voice/prompt_token");
            voice_prompt_feat  = read_f32(voice_m, "voice/prompt_feat");
            voice_embedding    = read_f32(voice_m, "voice/embedding");
        }
        voice_mel_len1     = (int)(voice_prompt_feat.size() / 80);
        voice_prompt_text  = cosyvoice_gguf_meta_str(voice_path, "voice.prompt_text", "");

        // ---- zero-shot voice cloning bake (optional) ---------------------
        // Runs the native front-end once and overwrites the baked-voice
        // tensors above.  Every failure throws: silently keeping the baked
        // voice would synthesise with the wrong speaker, which the caller has
        // no way to detect.
        if (!opts.reference_audio.empty()) {
            const std::string s3tok_path = resolve_component(
                opts.s3tok_gguf_path, opts.model_dir, "cosyvoice3-s3tok", false);
            const std::string campplus_path = resolve_component(
                opts.campplus_gguf_path, opts.model_dir, "cosyvoice3-campplus", false);
            require(s3tok_path,
                    "speech tokenizer gguf (cosyvoice3-s3tok*.gguf; required when reference_audio is set)");
            require(campplus_path,
                    "CAM++ gguf (cosyvoice3-campplus*.gguf; required when reference_audio is set)");

            cosyvoice_prompt ref;
            std::string err;
            if (!cosyvoice_frontend_run(opts.reference_audio, s3tok_path, campplus_path,
                                        cosyvoice_frontend_backend(backend), opts.n_threads, ref, err)) {
                throw std::runtime_error("cosyvoice: voice cloning failed: " + err);
            }
            voice_prompt_stok.assign(ref.prompt_stok.begin(), ref.prompt_stok.end());
            voice_prompt_token = voice_prompt_stok;
            voice_prompt_feat  = std::move(ref.prompt_feat);
            voice_mel_len1     = ref.mel_len1;
            voice_embedding    = std::move(ref.embedding);
            // The baked transcript describes the baked clip, not this
            // reference; without opts.prompt_text run() goes cross-lingual.
            voice_prompt_text.clear();
            cloned = true;
            fprintf(stderr, "cosyvoice: cloned voice from %s (%zu prompt tokens, %s)\n",
                    opts.reference_audio.c_str(), voice_prompt_stok.size(),
                    opts.prompt_text.empty() ? "no transcript: cross-lingual mode"
                                             : "transcript given: zero-shot mode");
        }

        coreml_lm = cosyvoice_coreml::load(llm_path_);
        coreml_flow = cosyvoice_coreml::load(flow_path_);
        coreml_frontend = cosyvoice_coreml::load(flow_path_, "-frontend");
        coreml_hift = cosyvoice_coreml::load(hift_path_);
        coreml_f0 = cosyvoice_coreml::load(hift_path_, "-f0");
        backend_guard.armed = false;
    }

    ~Impl() {
        // Stages are freed inside run(); nothing borrows `backend` between calls.
        if (backend) { ggml_backend_free(backend); backend = nullptr; }
    }

    int sample_rate() const {
        // output_sample_rate is reserved (no resampling yet) — always report the
        // native rate we actually emit.
        return kNativeSampleRate;
    }

    // Full text -> 24 kHz PCM pipeline.  `instruct` is the already-resolved
    // bare instruction ("" = zero-shot).  `tmg` / `out_tokens` are optional.
    std::vector<float> run(const std::string & text,
                           const std::string & instruct,
                           cosyvoice_timings * tmg = nullptr,
                           std::vector<int> * out_tokens = nullptr) {
        for (const auto & stage : {coreml_lm, coreml_flow, coreml_frontend, coreml_hift, coreml_f0})
            if (stage) stage->clear_predictions();
        std::string prompt_text;
        std::vector<int> lm_prompt_stok;
        bool cross_lingual = false;
        if (!instruct.empty()) {
            // Instruct mode (dialects / accents / emotion / pace / volume):
            // the natural-language instruction sits in the LM prompt *before*
            // <|endofprompt|>, and the LM receives NO prompt speech tokens.
            // This mirrors CosyVoice3 frontend_instruct2 == frontend_zero_shot
            // minus llm_prompt_speech_token.  The baked voice still supplies the
            // *timbre* via the flow tensors below; the instruction drives the
            // dialect/style.
            prompt_text = detail::build_lm_prompt_instruct(instruct);
            lm_prompt_stok = {};  // dropped in instruct mode
        } else if (cloned && opts.prompt_text.empty()) {
            // Cross-lingual cloning: no transcript for the reference, so the
            // LM gets the bare template (which still carries the
            // <|endofprompt|> marker the CosyVoice3 LM requires) and NO
            // prompt speech tokens — upstream frontend_cross_lingual ==
            // frontend_zero_shot minus prompt_text minus
            // llm_prompt_speech_token.  The flow tensors keep the cloned
            // timbre; the LM is free to speak the target language natively
            // instead of continuing the reference clip's language.
            prompt_text = detail::build_lm_prompt_zero_shot("");
            lm_prompt_stok = {};
            cross_lingual = true;
        } else {
            // Zero-shot: transcript precedence explicit > baked voice > fallback.
            // (`cloned` implies opts.prompt_text is non-empty here, so a cloned
            // voice never picks up the baked metadata or kDefaultTranscript.)
            const std::string transcript =
                !opts.prompt_text.empty()    ? opts.prompt_text
                : !voice_prompt_text.empty() ? voice_prompt_text
                                             : std::string(kDefaultTranscript);
            prompt_text = detail::build_lm_prompt_zero_shot(transcript);
            lm_prompt_stok = voice_prompt_stok;
        }

        std::vector<int> tts_ids  = tokenizer.encode(text);
        std::vector<int> text_ids = tokenizer.encode(prompt_text);
        text_ids.insert(text_ids.end(), tts_ids.begin(), tts_ids.end());

        // Length-guard basis mirrors upstream, which scales min/max on the
        // tokens that are NOT LM prompt text.  Upstream cross-lingual deletes
        // prompt_text and carries the <|endofprompt|> template inside
        // tts_text, so the template tokens count toward its basis; our port
        // keeps the template in prompt_text, so add them back explicitly or
        // short cross-lingual text could reach EOS earlier than upstream.
        const int lm_basis  = cross_lingual ? (int)text_ids.size() : (int)tts_ids.size();
        const int min_len   = 2 * lm_basis;       // min_token_text_ratio
        const int max_steps = 50 * (lm_basis + 1); // generous cap
        const bool greedy   = opts.greedy;

        if (tmg) {
            tmg->n_text_ids             = (int) text_ids.size();
            tmg->n_prompt_speech_tokens = (int) lm_prompt_stok.size();
        }

        // Stage 1 — LM: text ids -> speech tokens (freed at scope end). A pinned
        // trajectory skips the LM entirely -- sampling is chaotic in the logits,
        // so pinning is the only way two backends run the flow and vocoder over
        // identical input -- so the LM is not even loaded on that path.
        std::vector<int> speech_tokens;
        if (opts.force_speech_tokens.empty()) {
            model_ctx llm_m = cosyvoice_load_gguf(llm_path_, backend);
            ModelGuard llm_guard(llm_m);
            llm_m.n_threads = opts.n_threads;
            llm_m.coreml = coreml_lm;
            qwen_hp hp = cosyvoice_qwen_hp(llm_m);
            speech_tokens = cosyvoice_llm_generate(llm_m, hp, text_ids, lm_prompt_stok,
                                                   max_steps, greedy, opts.seed, min_len, tmg);
        } else {
            speech_tokens = opts.force_speech_tokens;
        }
        if (speech_tokens.empty()) {
            throw std::runtime_error("cosyvoice: LM produced no speech tokens");
        }
        if (tmg && !opts.force_speech_tokens.empty()) {
            tmg->n_speech_tokens = (int) speech_tokens.size();
        }
        if (out_tokens) *out_tokens = speech_tokens;

        // Stage 2 — DiT flow: speech tokens -> mel.
        int mel_len = 0;
        std::vector<float> mel;
        {
            model_ctx flow_m = cosyvoice_load_gguf(flow_path_, backend);
            ModelGuard flow_guard(flow_m);
            flow_m.n_threads = opts.n_threads;
            flow_m.coreml = coreml_flow;
            flow_m.coreml_frontend = coreml_frontend;
            mel = cosyvoice_flow_run(
                flow_m, voice_prompt_token, speech_tokens,
                voice_prompt_feat, voice_mel_len1, voice_embedding, mel_len, tmg,
                opts.flow_cut_prompt ? voice_mel_len1 : 0);
        }

        // Stage 3 — CausalHiFT vocoder: mel -> 24 kHz PCM.
        {
            model_ctx hift_m = cosyvoice_load_gguf(hift_path_, backend);
            ModelGuard hift_guard(hift_m);
            hift_m.n_threads = opts.n_threads;
            hift_m.coreml = coreml_hift;
            hift_m.coreml_f0 = coreml_f0;
            return cosyvoice_hift_synth(hift_m, mel, mel_len, opts.seed, tmg);
        }
    }
};

Engine::Engine(const EngineOptions & opts)
    : pimpl_(std::make_unique<Impl>(opts)) {}

Engine::~Engine() = default;
Engine::Engine(Engine &&) noexcept = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

SynthesisResult Engine::synthesize(const std::string & text) {
    return synthesize(text, pimpl_->opts.default_controls, StreamCallback{});
}

SynthesisResult Engine::synthesize(const std::string & text,
                                   const StreamCallback & on_chunk) {
    return synthesize(text, pimpl_->opts.default_controls, on_chunk);
}

SynthesisResult Engine::synthesize(const std::string & text,
                                   const VoiceControls & controls) {
    return synthesize(text, controls, StreamCallback{});
}

SynthesisResult Engine::synthesize(const std::string & text,
                                   const VoiceControls & controls,
                                   const StreamCallback & on_chunk) {
    if (text.empty()) {
        throw std::runtime_error("cosyvoice: empty text");
    }
    const std::string instruct = detail::resolve_instruct(controls);
    pimpl_->cancel_flag.store(false, std::memory_order_relaxed);

    cosyvoice_timings tmg;
    std::vector<int>  tokens;
    const auto t_all = std::chrono::steady_clock::now();
    std::vector<float> pcm = pimpl_->run(text, instruct, &tmg, &tokens);
    const double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_all).count();

    SynthesisResult result;
    result.sample_rate = kNativeSampleRate;
    result.duration_s  = static_cast<float>(pcm.size()) / static_cast<float>(kNativeSampleRate);
    result.speech_tokens = std::move(tokens);
    result.timings.lm_prefill_ms    = tmg.lm_prefill_ms;
    result.timings.lm_decode_ms     = tmg.lm_decode_ms;
    result.timings.flow_frontend_ms = tmg.flow_frontend_ms;
    result.timings.dit_euler_ms     = tmg.dit_euler_ms;
    result.timings.hift_f0_ms       = tmg.hift_f0_ms;
    result.timings.hift_source_ms   = tmg.hift_source_ms;
    result.timings.hift_stft_ms     = tmg.hift_stft_ms;
    result.timings.hift_decode_ms   = tmg.hift_decode_ms;
    result.timings.total_ms         = total_ms;
    result.timings.n_decode_steps   = tmg.n_decode_steps;
    result.timings.n_speech_tokens  = tmg.n_speech_tokens;
    result.timings.tm               = tmg.tm;
    result.timings.mel_len          = tmg.mel_len;
    result.timings.n_text_ids       = tmg.n_text_ids;
    result.timings.n_prompt_speech_tokens = tmg.n_prompt_speech_tokens;

    const bool streaming = pimpl_->opts.stream_chunk_tokens > 0 && static_cast<bool>(on_chunk);
    if (!streaming) {
        if (pimpl_->cancel_flag.load(std::memory_order_relaxed)) {
            throw std::runtime_error("cosyvoice: synthesis cancelled");
        }
        result.pcm = std::move(pcm);
        return result;
    }

    // Streaming: slice the completed waveform into token-cadence chunks.  (True
    // incremental token2wav hops are a later optimisation; the concatenation
    // invariant result.pcm == concat(chunks) still holds.)
    const int first_tokens = pimpl_->opts.stream_first_chunk_tokens > 0
        ? pimpl_->opts.stream_first_chunk_tokens
        : pimpl_->opts.stream_chunk_tokens;
    const std::size_t chunk_samples =
        static_cast<std::size_t>(pimpl_->opts.stream_chunk_tokens) * kSamplesPerToken;
    const std::size_t first_chunk_samples =
        static_cast<std::size_t>(first_tokens) * kSamplesPerToken;

    std::size_t pos = 0;
    int chunk_index = 0;
    while (pos < pcm.size()) {
        if (pimpl_->cancel_flag.load(std::memory_order_relaxed)) {
            throw std::runtime_error("cosyvoice: synthesis cancelled");
        }
        const std::size_t take = chunk_index == 0
            ? std::min(first_chunk_samples, pcm.size() - pos)
            : std::min(chunk_samples, pcm.size() - pos);
        const std::size_t end = pos + std::max<std::size_t>(take, 1);
        const bool is_last = end >= pcm.size();
        on_chunk(pcm.data() + pos, end - pos, chunk_index, is_last);
        pos = end;
        ++chunk_index;
    }
    if (chunk_index == 0) {
        on_chunk(pcm.data(), 0, 0, true);
    }

    result.pcm = std::move(pcm);
    return result;
}

void Engine::cancel() {
    pimpl_->cancel_flag.store(true, std::memory_order_relaxed);
}

const EngineOptions & Engine::options() const { return pimpl_->opts; }

std::string Engine::backend_name() const {
    ggml_backend_t b = pimpl_ ? pimpl_->backend : nullptr;
    const char * n = b ? ggml_backend_name(b) : nullptr;
    std::string label = n ? std::string(n) : std::string("(none)");
    if (pimpl_) {
        for (const auto & stage : {pimpl_->coreml_lm, pimpl_->coreml_flow, pimpl_->coreml_frontend,
                                  pimpl_->coreml_hift, pimpl_->coreml_f0}) {
            if (stage && stage->predictions() && label.find(stage->label()) == std::string::npos)
                label += "+" + stage->label();
        }
    }
    return label;
}

BackendDevice Engine::backend_device() const {
    return ::tts_cpp::detail::backend_is_cpu(pimpl_ ? pimpl_->backend : nullptr)
        ? BackendDevice::CPU : BackendDevice::GPU;
}

bool Engine::gpu_unsupported() const { return pimpl_ && pimpl_->gpu_present_but_unused; }

SynthesisResult synthesize(const EngineOptions & opts, const std::string & text) {
    Engine e(opts);
    return e.synthesize(text);
}

SynthesisResult synthesize(const EngineOptions & opts, const std::string & text,
                           const VoiceControls & controls) {
    Engine e(opts);
    return e.synthesize(text, controls);
}

} // namespace tts_cpp::cosyvoice
