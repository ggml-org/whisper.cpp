#pragma once

#include "ggml-alloc.h"
#include "decoder_profile.h"
#include "coreml_policy.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp { namespace parler { namespace detail {

struct parler_model;
struct parler_hparams;
struct parler_step_logits;

// Validate contract 2, last-axis normalization, layout, mask and geometry
// before interpreting native state. Older channel-axis exports need re-export.
bool parler_validate_decoder_contract(const std::string & metadata, const parler_hparams & hp,
                                     int context, int cross, int per_state, std::string & error);

// Host-facing step contract. All arrays are float32; cache/cross arrays are
// packed [layer, head, token, head_dim]. The implementation owns MLState and
// converts/copies only a new cache row after each successful prediction.
class parler_decoder_step_model {
public:
    virtual ~parler_decoder_step_model() = default;
    virtual int context_capacity() const = 0;
    virtual int cross_capacity() const = 0;
    virtual const char * label() const = 0;
    virtual bool seed(const std::vector<float> & keys, const std::vector<float> & values,
                      int past, const std::vector<float> & cross_keys,
                      const std::vector<float> & cross_values, int cross_length,
                      std::string & error) = 0;
    virtual bool step(const std::vector<float> & hidden, int position,
                      std::vector<float> & logits, std::string & error,
                      parler_decoder_step_profile * profile = nullptr) = 0;
};

// Open a decoder sidecar next to the GGUF. Null with an error on unsupported
// builds, absent or incompatible sidecars. Caller controls fallback/strictness.
std::unique_ptr<parler_decoder_step_model> parler_open_decoder_sidecar(
    const parler_model & model, const std::string & gguf_path, std::string & error);
std::unique_ptr<parler_decoder_step_model> parler_open_decoder_sidecar(
    const parler_model & model, const std::string & gguf_path,
    const parler_coreml_policy & policy, std::string & error);

// Fast ggml prefill followed by Core ML single-token steps. If a sidecar
// fails, replay the exact consumed token history into ggml before resuming;
// the sampler is never rerun and previously streamed code frames stay valid.
class parler_decoder_runtime {
public:
    explicit parler_decoder_runtime(const parler_model & model);
    ~parler_decoder_runtime();
    void attach(std::unique_ptr<parler_decoder_step_model> model, bool strict = false);
    void set_cancel_check(std::function<bool()> check);
    bool active() const;
    const std::string & backend() const;
    bool prefill(const std::vector<int32_t> & prompt, const std::vector<int32_t> & start,
                 ggml_gallocr_t allocator, int threads,
                 parler_step_logits & logits, int & past, int max_frames = 0);
    bool step(const std::vector<int32_t> & frame, int past,
              ggml_gallocr_t allocator, int threads, parler_step_logits & logits,
              parler_decoder_step_profile * profile = nullptr);

private:
    const parler_model & model_;
    std::unique_ptr<parler_decoder_step_model> sidecar_;
    std::vector<int32_t> prompt_, start_;
    std::vector<std::vector<int32_t>> history_;
    std::vector<float> hidden_, row_;
    std::vector<unsigned char> row_bytes_;
    std::string backend_ = "ggml";
    std::function<bool()> cancelled_;
    bool strict_ = false;
    bool failed_ = false;
    bool ready_ = false;
    bool use_sidecar_ = false;
    bool seed(int past, std::string & error);
    bool embed(const std::vector<int32_t> & frame, int past);
    bool recover(ggml_gallocr_t allocator, int threads, parler_step_logits & logits);
};

}}} // namespace tts_cpp::parler::detail
