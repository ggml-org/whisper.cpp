#include "decoder_runtime.h"
#include "internal.h"
#include "../coreml_sidecar_path.h"
#include "../json.hpp"

#ifdef TTS_CPP_USE_COREML
#include "decoder_coreml.h"
#include "decoder_update_plan.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace tts_cpp { namespace parler { namespace detail {
bool parler_validate_decoder_contract(const std::string & metadata, const parler_hparams & hp,
                                     int context, int cross, int per_state, std::string & error) {
    error.clear();
    try {
        const auto m = nlohmann::json::parse(metadata);
        if (m.at("contract") != 2 || m.at("kind") != "parler-decoder-step") {
            error = "unsupported decoder sidecar contract; re-export with the current exporter (contract 2 required)";
        } else if (m.at("normalization") != "last-axis") {
            error = "decoder sidecar requires last-axis normalization; re-export with the current exporter";
        } else if (m.at("states").at("k_layout") != "rows") {
            error = "decoder sidecar requires k_layout=rows; cols is experimental and unsupported by this runtime";
        } else if (m.at("mask_off") != -30000.0) {
            error = "decoder sidecar mask_off must be -30000";
        } else if (m.at("n_layer") != hp.dec_n_layer || m.at("d_model") != hp.dec_d_model ||
                   m.at("n_head") != hp.dec_n_head || m.at("head_dim") != hp.dec_d_model / hp.dec_n_head ||
                   m.at("n_q") != hp.n_codebooks || m.at("vocab") != hp.dec_vocab ||
                   m.at("ctx") != context || m.at("cross_context") != cross ||
                   m.at("states").at("layers_per_state") != per_state) {
            error = "decoder sidecar contract geometry does not match its state buffers and GGUF";
        } else if (!hp.variant.empty() && m.at("variant") != hp.variant) {
            error = "decoder sidecar checkpoint variant does not match the GGUF";
        }
    } catch (const std::exception & e) {
        error = std::string("missing or invalid parler.decoder.contract metadata; re-export with the current exporter: ") + e.what();
    }
    return error.empty();
}
#ifdef TTS_CPP_USE_COREML
namespace {

std::string state_name(const char * kind, int layer) {
    char suffix[24];
    std::snprintf(suffix, sizeof(suffix), "_%02d", layer);
    return std::string(kind) + suffix;
}

const decoder_coreml_tensor * find(const std::vector<decoder_coreml_tensor> & tensors,
                                   const std::string & name) {
    const auto it = std::find_if(tensors.begin(), tensors.end(), [&](const auto & t) { return t.name == name; });
    return it == tensors.end() ? nullptr : &*it;
}

bool shape_is(const std::vector<decoder_coreml_tensor> & tensors, const std::string & name,
              const std::vector<int64_t> & shape) {
    const auto * t = find(tensors, name);
    return t && t->shape == shape && t->dtype != decoder_coreml_dtype::int32;
}

class coreml_step final : public parler_decoder_step_model {
public:
    coreml_step(std::unique_ptr<decoder_coreml_session> session, const parler_hparams & hp,
                int context, int cross, int per_state, std::string label)
        : session_(std::move(session)), layers_(hp.dec_n_layer), heads_(hp.dec_n_head),
          hd_(hp.dec_d_model / hp.dec_n_head), context_(context), cross_(cross), per_state_(per_state),
          logit_count_((size_t) hp.n_codebooks * hp.dec_vocab), label_(std::move(label)),
          updates_(parler_decoder_update_plan(hp.dec_n_layer, per_state)) {}

    int context_capacity() const override { return context_; }
    int cross_capacity() const override { return cross_; }
    const char * label() const override { return label_.c_str(); }

    bool seed(const std::vector<float> & keys, const std::vector<float> & values, int past,
              const std::vector<float> & cross_keys, const std::vector<float> & cross_values,
              int cross_length, std::string & error) override {
        if (past <= 0 || past > context_ || cross_length <= 0 || cross_length > cross_ ||
            keys.size() != (size_t) layers_ * heads_ * past * hd_ || values.size() != keys.size() ||
            cross_keys.size() != (size_t) layers_ * heads_ * cross_length * hd_ ||
            cross_values.size() != cross_keys.size()) {
            error = "invalid decoder cache seed dimensions";
            return false;
        }
        if (!session_->reset(error)) return false;
        for (int g = 0; g < layers_ / per_state_; ++g) {
            const size_t offset = (size_t) g * per_state_ * heads_ * past * hd_;
            const size_t co = (size_t) g * per_state_ * heads_ * cross_length * hd_;
            for (int v = 0; v < 2; ++v) {
                if (!session_->write_state(state_name(v ? "self_v" : "self_k", g), {0, 0, 0, 0},
                        {per_state_, heads_, past, hd_}, (v ? values : keys).data() + offset,
                        (size_t) per_state_ * heads_ * past * hd_, error) ||
                    !session_->write_state(state_name(v ? "cross_v" : "cross_k", g), {0, 0, 0, 0},
                        {per_state_, heads_, cross_length, hd_}, (v ? cross_values : cross_keys).data() + co,
                        (size_t) per_state_ * heads_ * cross_length * hd_, error)) return false;
            }
        }
        self_mask_.assign(context_, -30000.0f);
        cross_mask_.assign(cross_, -30000.0f);
        std::fill_n(cross_mask_.begin(), cross_length, 0.0f);
        next_position_ = past;
        return session_->set_input("cross_mask", cross_mask_.data(), cross_mask_.size(), error);
    }

    bool step(const std::vector<float> & hidden, int position, std::vector<float> & logits,
              std::string & error, parler_decoder_step_profile * profile) override {
        using clock = std::chrono::steady_clock;
        using milliseconds = std::chrono::duration<double, std::milli>;
        if (profile) *profile = {};
        const auto prepare_start = profile ? clock::now() : clock::time_point{};
        if (position != next_position_ || position < 0 || position >= context_) {
            error = "decoder cache position is out of order or outside capacity";
            return false;
        }
        std::fill_n(self_mask_.begin(), position, 0.0f);
        for (auto & update : updates_) update.row = position;
        if (!session_->set_input("hidden", hidden.data(), hidden.size(), error) ||
            !session_->set_input("self_mask", self_mask_.data(), self_mask_.size(), error)) return false;
        const auto prepare_end = profile ? clock::now() : clock::time_point{};
        decoder_coreml_timing timing;
        if (!session_->predict(updates_, profile ? &timing : nullptr, error)) return false;
        const auto readback_start = profile ? clock::now() : clock::time_point{};
        logits.resize(logit_count_);
        if (!session_->copy_output("logits", logits.data(), logits.size(), error)) return false;
        if (profile) {
            // Runtime publishes validity only after checking finite logits and
            // expected output geometry; this local result is never aggregated.
            profile->native_timing_available = true;
            profile->input_prepare_ms = milliseconds(prepare_end - prepare_start).count();
            profile->predict_ms = timing.predict_ms;
            profile->row_write_ms = timing.row_write_ms;
            profile->session_total_ms = timing.total_ms;
            profile->row_bytes_written = timing.row_bytes_written;
            profile->logit_readback_ms = milliseconds(clock::now() - readback_start).count();
        }
        ++next_position_;
        return true;
    }

private:
    std::unique_ptr<decoder_coreml_session> session_;
    int layers_, heads_, hd_, context_, cross_, per_state_, next_position_ = -1;
    size_t logit_count_;
    std::string label_;
    std::vector<float> self_mask_, cross_mask_;
    std::vector<decoder_coreml_row_update> updates_;
};

} // namespace
#endif

std::unique_ptr<parler_decoder_step_model> parler_open_decoder_sidecar(
    const parler_model & model, const std::string & gguf_path, std::string & error) {
#ifdef TTS_CPP_USE_COREML
    if (parler_coreml_disabled()) { error = "Core ML is disabled"; return nullptr; }
    const auto path = ::tts_cpp::detail::coreml_sidecar_path(gguf_path, "-decoder");
    if (!::tts_cpp::detail::coreml_sidecar_exists(path)) {
        error = "decoder sidecar not found: " + path;
        return nullptr;
    }
    const char * env = std::getenv(PARLER_COREML_UNITS_ENV);
    const std::string units = env && *env ? env : "all";
    decoder_coreml_units selection;
    std::string label;
    if (units == "all") { selection = decoder_coreml_units::all; label = "coreml-all"; }
    else if (units == "cpu_and_ane") { selection = decoder_coreml_units::cpu_and_ane; label = "coreml-ane"; }
    else if (units == "cpu_and_gpu") { selection = decoder_coreml_units::cpu_and_gpu; label = "coreml-gpu"; }
    else if (units == "cpu_only") { selection = decoder_coreml_units::cpu_only; label = "coreml-cpu"; }
    else { error = "invalid PARLER_COREML_COMPUTE_UNITS for decoder"; return nullptr; }
    auto session = decoder_coreml_session::open(path, selection, error);
    if (!session) return nullptr;
    const auto & hp = model.hparams;
    const int h = hp.dec_n_head, hd = hp.dec_d_model / h;
    const auto * sk = find(session->states(), "self_k_00");
    const auto * ck = find(session->states(), "cross_k_00");
    if (!sk || !ck || sk->shape.size() != 4 || ck->shape.size() != 4 ||
        sk->shape[0] <= 0 || sk->shape[0] > hp.dec_n_layer || hp.dec_n_layer % sk->shape[0] != 0 ||
        sk->shape[2] <= 0 || sk->shape[2] > hp.max_position ||
        ck->shape[2] <= 0 || ck->shape[2] > 32768) {
        error = "decoder sidecar has invalid cache dimensions";
        return nullptr;
    }
    const int ctx = (int) sk->shape[2], cross = (int) ck->shape[2], per_state = (int) sk->shape[0];
    if (!parler_validate_decoder_contract(session->metadata("parler.decoder.contract"), hp,
                                         ctx, cross, per_state, error)) return nullptr;
    bool valid = session->inputs().size() == 3 && session->outputs().size() == 3 &&
        session->states().size() == (size_t) 4 * (hp.dec_n_layer / per_state) &&
        shape_is(session->inputs(), "hidden", {1, hp.dec_d_model, 1, 1}) &&
        shape_is(session->inputs(), "self_mask", {1, 1, 1, ctx}) &&
        shape_is(session->inputs(), "cross_mask", {1, 1, 1, cross}) &&
        shape_is(session->outputs(), "logits", {hp.n_codebooks, hp.dec_vocab}) &&
        shape_is(session->outputs(), "new_k", {hp.dec_n_layer, h, 1, hd}) &&
        shape_is(session->outputs(), "new_v", {hp.dec_n_layer, h, 1, hd});
    for (int g = 0; g < hp.dec_n_layer / per_state; ++g) {
        for (const auto * kind : {"self_k", "self_v", "cross_k", "cross_v"}) {
            const int rows = kind[0] == 's' ? ctx : cross;
            valid = valid && shape_is(session->states(), state_name(kind, g), {per_state, h, rows, hd});
        }
    }
    if (!valid) { error = "decoder sidecar feature shapes do not match the GGUF"; return nullptr; }
    return std::make_unique<coreml_step>(std::move(session), hp, ctx, cross, per_state, label);
#else
    (void) model;
    (void) gguf_path;
    error = "this build has no Core ML decoder support";
    return nullptr;
#endif
}

}}} // namespace tts_cpp::parler::detail
