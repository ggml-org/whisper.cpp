#include "decoder_runtime.h"
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace tts_cpp { namespace parler { namespace detail {
namespace {

bool read_values(const ggml_tensor * tensor, size_t offset, int64_t count,
                 std::vector<unsigned char> & bytes, float * output) {
    if (!tensor || count <= 0 || count % ggml_blck_size(tensor->type)) return false;
    const size_t size = ggml_row_size(tensor->type, count);
    if (offset > ggml_nbytes(tensor) || size > ggml_nbytes(tensor) - offset) return false;
    bytes.resize(size);
    ggml_backend_tensor_get(tensor, bytes.data(), offset, size);
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(output, bytes.data(), size);
        return true;
    }
    const auto * traits = ggml_get_type_traits(tensor->type);
    if (!traits->to_float) return false;
    traits->to_float(bytes.data(), output, count);
    return true;
}

} // namespace

parler_decoder_runtime::parler_decoder_runtime(const parler_model & model) : model_(model) {}
parler_decoder_runtime::~parler_decoder_runtime() = default;

void parler_decoder_runtime::attach(std::unique_ptr<parler_decoder_step_model> model, bool strict) {
    sidecar_ = std::move(model);
    strict_ = strict;
    failed_ = false;
    ready_ = false;
}

void parler_decoder_runtime::set_cancel_check(std::function<bool()> check) { cancelled_ = std::move(check); }

bool parler_decoder_runtime::active() const { return sidecar_ != nullptr; }
const std::string & parler_decoder_runtime::backend() const { return backend_; }

bool parler_decoder_runtime::seed(int past, std::string & error) {
    const auto & hp = model_.hparams;
    const int d = hp.dec_d_model, heads = hp.dec_n_head, hd = d / heads;
    const int layers = hp.dec_n_layer, cross = model_.cross_len;
    if (past > sidecar_->context_capacity() || cross > sidecar_->cross_capacity()) {
        error = "decoder sidecar context is too small for the prompt or description";
        return false;
    }
    std::vector<float> keys((size_t) layers * past * d), values(keys.size());
    std::vector<float> ck((size_t) layers * cross * d), cv(ck.size());
    row_.resize((size_t) std::max(past, cross) * d);
    const size_t token_bytes = ggml_row_size(model_.kv_type, d);
    for (int l = 0; l < layers; ++l) {
        for (int v = 0; v < 2; ++v) {
            const auto * tensor = v ? model_.memory_v : model_.memory_k;
            if (!read_values(tensor, (size_t) l * hp.n_ctx * token_bytes,
                             (int64_t) past * d, row_bytes_, row_.data())) return false;
            auto & dst = v ? values : keys;
            for (int h = 0; h < heads; ++h) for (int t = 0; t < past; ++t) {
                std::copy_n(row_.data() + (size_t) t * d + h * hd, hd,
                            dst.data() + ((size_t) l * heads * past + h * past + t) * hd);
            }
        }
        for (int v = 0; v < 2; ++v) {
            const auto * tensor = v ? model_.cross_v_t[l] : model_.cross_k[l];
            if (!read_values(tensor, 0, (int64_t) cross * d, row_bytes_, row_.data())) return false;
            auto & dst = v ? cv : ck;
            for (int h = 0; h < heads; ++h) for (int t = 0; t < cross; ++t) for (int i = 0; i < hd; ++i) {
                const size_t src = v && !model_.use_fa ? (size_t) (h * hd + i) * cross + t
                                                                     : (size_t) t * d + h * hd + i;
                dst[((size_t) l * heads * cross + h * cross + t) * hd + i] = row_[src];
            }
        }
    }
    return sidecar_->seed(keys, values, past, ck, cv, cross, error);
}

bool parler_decoder_runtime::embed(const std::vector<int32_t> & frame, int past) {
    const auto & hp = model_.hparams;
    if ((int) frame.size() != hp.n_codebooks || past < 0 || past >= hp.max_position) return false;
    const int d = hp.dec_d_model;
    hidden_.assign(d, 0.0f);
    row_.resize(d);
    for (int q = 0; q < hp.n_codebooks; ++q) {
        const auto * table = model_.dec_embed[q];
        if (!table || frame[q] < 0 || frame[q] >= table->ne[1] ||
            !read_values(table, (size_t) frame[q] * table->nb[1], d, row_bytes_, row_.data())) return false;
        for (int i = 0; i < d; ++i) hidden_[i] += row_[i];
    }
    const auto * positions = model_.embed_positions;
    if (!read_values(positions, (size_t) past * positions->nb[1], d, row_bytes_, row_.data())) return false;
    for (int i = 0; i < d; ++i) hidden_[i] += row_[i];
    return true;
}

bool parler_decoder_runtime::prefill(const std::vector<int32_t> & prompt,
                                    const std::vector<int32_t> & start,
                                    ggml_gallocr_t allocator, int threads,
                                    parler_step_logits & logits, int & past, int max_frames) {
    ready_ = false;
    use_sidecar_ = false;
    backend_ = "ggml";
    history_.clear();
    prompt_ = prompt;
    start_ = start;
    if (failed_ && strict_) return false;
    if (!parler_dec_prefill(model_, prompt, start, allocator, threads, logits, past)) return false;
    if (sidecar_) {
        const int cap = max_frames > 0 ? max_frames : model_.hparams.gen_max_length;
        // max_length includes BOS and the final sampled frame, for which no
        // further prediction is needed. Prefill itself always consumes one row.
        const int64_t required = (int64_t) prompt.size() + std::max(1, cap - 1);
        if (required > sidecar_->context_capacity() || model_.cross_len > sidecar_->cross_capacity()) {
            fprintf(stderr, "parler: request exceeds decoder sidecar capacity; %s\n",
                    strict_ ? "strict request rejected (sidecar retained)" : "this call stays on ggml");
            ready_ = !strict_;
            return ready_;
        }
        std::string error;
        if (!seed(past, error)) {
            fprintf(stderr, "parler: Core ML decoder initialization failed: %s\n", error.c_str());
            sidecar_.reset();
            failed_ = true;
            if (strict_) return false;
        } else {
            use_sidecar_ = true;
        }
    }
    ready_ = true;
    return true;
}

bool parler_decoder_runtime::recover(ggml_gallocr_t allocator, int threads,
                                    parler_step_logits & logits) {
    int past = 0;
    if (cancelled_ && cancelled_()) return false;
    if (!parler_dec_prefill(model_, prompt_, start_, allocator, threads, logits, past)) return false;
    for (const auto & frame : history_) {
        if (cancelled_ && cancelled_()) return false;
        if (!parler_dec_step(model_, frame, past++, allocator, threads, logits)) return false;
    }
    return true;
}

bool parler_decoder_runtime::step(const std::vector<int32_t> & frame, int past,
                                 ggml_gallocr_t allocator, int threads,
                                 parler_step_logits & logits, parler_decoder_step_profile * profile) {
    using clock = std::chrono::steady_clock;
    using milliseconds = std::chrono::duration<double, std::milli>;
    if (profile) *profile = {};
    const auto step_start = profile ? clock::now() : clock::time_point{};
    const auto abort = [&]() { ready_ = false; return false; };
    if (!ready_ || (failed_ && strict_) || past != (int) prompt_.size() + 1 + (int) history_.size() ||
        (cancelled_ && cancelled_())) return abort();
    // Retain only token IDs for replay; no duplicate floating-point KV cache.
    if (past < 0 || past >= model_.hparams.n_ctx || past >= model_.hparams.max_position ||
        (int) frame.size() != model_.hparams.n_codebooks ||
        std::any_of(frame.begin(), frame.end(), [&](int32_t id) {
            return id < 0 || id >= model_.hparams.dec_vocab;
        })) return abort();
    history_.push_back(frame);
    if (!use_sidecar_) {
        if (!parler_dec_step(model_, frame, past, allocator, threads, logits)) return abort();
        return true;
    }
    std::string error;
    auto & output = logits.copy;
    const size_t expected = (size_t) model_.hparams.n_codebooks * model_.hparams.dec_vocab;
    parler_decoder_step_profile measured;
    const auto embedding_start = profile ? clock::now() : clock::time_point{};
    const bool embedded = past < sidecar_->context_capacity() && embed(frame, past);
    const auto embedding_end = profile ? clock::now() : clock::time_point{};
    bool ok = embedded && sidecar_->step(hidden_, past, output, error, profile ? &measured : nullptr) &&
              output.size() == expected;
    if (ok) ok = std::all_of(output.begin(), output.end(), [](float v) { return std::isfinite(v); });
    if (ok) {
        logits.view = logits.copy.data();
        backend_ = sidecar_->label();
        if (profile && measured.native_timing_available) {
            measured.valid = true;
            measured.embedding_ms = milliseconds(embedding_end - embedding_start).count();
            measured.step_wall_ms = milliseconds(clock::now() - step_start).count();
            *profile = measured;
        }
        return true;
    }
    if (error.empty()) error = "invalid embedding, logits or cache capacity";
    fprintf(stderr, "parler: Core ML decoder failed at position %d: %s; retiring sidecar%s\n",
            past, error.c_str(), strict_ ? "" : " and replaying token history on ggml");
    const bool used_sidecar = backend_ != "ggml";
    sidecar_.reset();
    use_sidecar_ = false;
    failed_ = true;
    if (strict_) return abort();
    backend_ = used_sidecar ? "mixed" : "ggml";
    if (!recover(allocator, threads, logits)) return abort();
    return true;
}

}}} // namespace tts_cpp::parler::detail
