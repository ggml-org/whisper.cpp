#include "tts-cpp/parler/engine.h"

#include "internal.h"
#include "tokenizer.h"
#include "bpe_tokenizer.h"
#include "text_norm.h"
#include "delay.h"
#include "sampler.h"
#include "backend_selection.h"
#include "backend_util.h"

#include <atomic>
#include <random>
#include <stdexcept>
#include <thread>

namespace tts_cpp {
namespace parler {

using namespace ::tts_cpp::parler::detail;

// DAC right receptive field ~10 frames; the extra margin also covers the
// near-EOS drop region so streamed chunks match the whole-sequence decode.
static constexpr int kDacStreamRfMargin = 16;

struct Engine::Impl {
    EngineOptions        opts;
    parler_model         model;
    parler_tokenizer     tokenizer;         // descriptions (and prompts when shared)
    parler_bpe_tokenizer prompt_tokenizer;  // prompts, iff the GGUF ships one
    ggml_gallocr_t       allocr = nullptr;
    std::string        cached_description;
    bool               has_cached_description = false;
    std::atomic<bool>  cancel_requested{false};

    ~Impl() {
        if (allocr) ggml_gallocr_free(allocr);
        parler_free_model(model);
    }

    int resolve_threads() const {
        if (opts.n_threads > 0) return opts.n_threads;
        const unsigned hw = std::thread::hardware_concurrency();
        return (int) std::min(hw ? hw : 4u, 4u);
    }

    void check_cancel() const {
        if (cancel_requested.load(std::memory_order_relaxed)) {
            throw std::runtime_error("parler: synthesis cancelled");
        }
    }

    SynthesisResult run(const std::string & prompt, const std::string & description,
                        const StreamCallback * on_chunk = nullptr) {
        if (prompt.empty()) {
            throw std::runtime_error("parler: prompt must not be empty");
        }
        if (description.empty()) {
            throw std::runtime_error("parler: description must not be empty");
        }
        cancel_requested.store(false, std::memory_order_relaxed);
        const parler_hparams & hp = model.hparams;
        if (opts.max_frames > 0 && opts.max_frames <= hp.n_codebooks) {
            throw std::runtime_error("parler: max_frames must be 0 or > " +
                                     std::to_string(hp.n_codebooks) +
                                     " (fewer delayed steps cannot yield one audio frame)");
        }
        const int n_threads = resolve_threads();

        const std::string spoken_prompt = !opts.normalize_numbers ? prompt
            : model.has_prompt_tok ? normalize_numbers_indic(prompt)
                                   : normalize_numbers_en(prompt);
        const std::vector<int32_t> prompt_ids = model.has_prompt_tok
            ? prompt_tokenizer.encode(spoken_prompt)
            : tokenizer.encode(spoken_prompt);
        if (prompt_ids.empty()) {
            throw std::runtime_error("parler: prompt tokenized to zero tokens");
        }

        if (!has_cached_description || cached_description != description) {
            const std::vector<int32_t> desc_ids = tokenizer.encode(description);
            if (desc_ids.empty()) {
                throw std::runtime_error("parler: description tokenized to zero tokens");
            }
            // re-encoding destroys the previous cross-KV state up front, so the
            // cache must be invalidated even if the encode below fails
            has_cached_description = false;
            if (!parler_encode_description(model, desc_ids, n_threads, nullptr)) {
                throw std::runtime_error("parler: description encoding failed");
            }
            cached_description = description;
            has_cached_description = true;
        }
        check_cancel();

        delay_config dcfg;
        dcfg.n_codebooks = hp.n_codebooks;
        dcfg.bos_id = hp.bos_id;
        dcfg.eos_id = hp.eos_id;
        dcfg.pad_id = hp.pad_id;
        dcfg.max_length = hp.gen_max_length;
        if (opts.max_frames > 0 && opts.max_frames < dcfg.max_length) {
            dcfg.max_length = opts.max_frames;
        }
        dcfg.min_new_tokens = opts.min_new_tokens >= 0 ? opts.min_new_tokens
                                                       : hp.gen_min_new_tokens;
        delay_state st(dcfg);

        parler_sampling_params sp;
        sp.greedy      = opts.greedy || !hp.gen_do_sample;
        sp.temperature = opts.temperature > 0.0f ? opts.temperature : hp.gen_temperature;
        sp.top_k       = opts.top_k > 0 ? opts.top_k : hp.gen_top_k;
        sp.top_p       = opts.top_p;
        std::mt19937 rng((uint32_t) opts.seed);

        std::vector<float> logits;
        int n_past = 0;
        if (!parler_dec_prefill(model, prompt_ids, st.input_frame(), allocr, n_threads,
                                logits, n_past)) {
            throw std::runtime_error("parler: decoder prefill failed");
        }

        SynthesisResult res;
        res.sample_rate = hp.dac_sample_rate;

        // Streaming re-decodes the growing code prefix every stream_chunk_frames
        // steps and emits the newly-stable samples (holding back the DAC right
        // receptive field so each chunk matches the whole-sequence decode).
        const bool streaming = on_chunk != nullptr && opts.stream_chunk_frames > 0;
        int emitted_frames = 0;
        int chunk_index    = 0;
        auto emit = [&](bool final_flush) {
            int nf = 0;
            const std::vector<int32_t> c = st.undelay(hp.dac_codebook_size, &nf);
            const int emit_end = final_flush ? nf : nf - kDacStreamRfMargin;
            if (emit_end <= emitted_frames) return;
            std::vector<float> pcm;
            if (!parler_dac_decode(model, c.data(), nf, n_threads, pcm)) {
                throw std::runtime_error("parler: DAC decode failed");
            }
            const std::size_t a = (std::size_t) emitted_frames * hp.dac_hop;
            const std::size_t b = (std::size_t) emit_end       * hp.dac_hop;
            (*on_chunk)(pcm.data() + a, b - a, chunk_index++, final_flush);
            res.pcm.insert(res.pcm.end(), pcm.begin() + a, pcm.begin() + b);
            emitted_frames = emit_end;
        };

        int step = 0;
        int next_emit_at = opts.stream_first_chunk_frames > 0
                         ? opts.stream_first_chunk_frames : opts.stream_chunk_frames;
        while (true) {
            check_cancel();
            st.process_logits(logits.data(), hp.dec_vocab);
            st.append(parler_sample_frame(logits.data(), hp.n_codebooks, hp.dec_vocab, sp, rng));
            if (st.finished()) break;
            if (!parler_dec_step(model, st.input_frame(), n_past, allocr, n_threads, logits)) {
                throw std::runtime_error("parler: decoder step failed");
            }
            n_past++;
            if (streaming && ++step >= next_emit_at) {
                emit(false);
                next_emit_at = step + opts.stream_chunk_frames;
            }
        }

        int n_frames = 0;
        const std::vector<int32_t> codes = st.undelay(hp.dac_codebook_size, &n_frames);
        if (n_frames <= 0) {
            throw std::runtime_error("parler: generation produced no audio frames");
        }
        check_cancel();

        if (streaming) {
            emit(true);
            res.duration_s = res.pcm.empty() ? 0.0f
                           : (float) res.pcm.size() / (float) hp.dac_sample_rate;
            return res;
        }

        if (!parler_dac_decode(model, codes.data(), n_frames, n_threads, res.pcm)) {
            throw std::runtime_error("parler: DAC decode failed");
        }
        res.duration_s  = res.pcm.empty() ? 0.0f
                        : (float) res.pcm.size() / (float) hp.dac_sample_rate;
        return res;
    }
};

Engine::Engine(const EngineOptions & opts) : pimpl_(new Impl()) {
    pimpl_->opts = opts;
    if (!opts.backends_dir.empty()) {
        ::tts_cpp::detail::set_backends_directory(opts.backends_dir);
    }
    std::string err;
    if (!parler_load_gguf(opts.model_gguf_path, pimpl_->model, opts.n_gpu_layers, &err)) {
        throw std::runtime_error("parler: " + err);
    }
    if (!pimpl_->tokenizer.load(pimpl_->model.tok_pieces, pimpl_->model.tok_scores,
                                pimpl_->model.tok_charsmap, pimpl_->model.tok_unk_id,
                                pimpl_->model.tok_eos_id, pimpl_->model.tok_add_eos)) {
        throw std::runtime_error("parler: tokenizer load failed");
    }
    if (pimpl_->model.has_prompt_tok &&
        !pimpl_->prompt_tokenizer.load(pimpl_->model.ptok_pieces, pimpl_->model.ptok_merges,
                                       pimpl_->model.ptok_unk_id, pimpl_->model.ptok_bos_id,
                                       pimpl_->model.ptok_add_bos)) {
        throw std::runtime_error("parler: prompt tokenizer load failed");
    }
    pimpl_->allocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(pimpl_->model.backend));
    if (!pimpl_->allocr) {
        throw std::runtime_error("parler: graph allocator creation failed");
    }
}

Engine::~Engine() = default;
Engine::Engine(Engine &&) noexcept = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

SynthesisResult Engine::synthesize(const std::string & prompt) {
    if (pimpl_->opts.default_description.empty()) {
        throw std::runtime_error("parler: no default_description configured; "
                                 "use synthesize(prompt, description)");
    }
    return pimpl_->run(prompt, pimpl_->opts.default_description);
}

SynthesisResult Engine::synthesize(const std::string & prompt,
                                   const std::string & description) {
    return pimpl_->run(prompt, description);
}

SynthesisResult Engine::synthesize(const std::string & prompt,
                                   const std::string & description,
                                   const StreamCallback & on_chunk) {
    return pimpl_->run(prompt, description, &on_chunk);
}

void Engine::cancel() {
    pimpl_->cancel_requested.store(true, std::memory_order_relaxed);
}

const EngineOptions & Engine::options() const { return pimpl_->opts; }

std::string Engine::backend_name() const {
    if (!pimpl_->model.backend) return "(unknown)";
    return ggml_backend_name(pimpl_->model.backend);
}

BackendDevice Engine::backend_device() const {
    return ::tts_cpp::detail::backend_is_cpu(pimpl_->model.backend)
        ? BackendDevice::CPU : BackendDevice::GPU;
}

SynthesisResult synthesize(const EngineOptions & opts, const std::string & prompt,
                           const std::string & description) {
    Engine engine(opts);
    return engine.synthesize(prompt, description);
}

} // namespace parler
} // namespace tts_cpp
