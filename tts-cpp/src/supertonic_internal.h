#pragma once

#include <cstdint>
#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace tts_cpp::supertonic::detail {

struct supertonic_hparams {
    std::string arch = "supertonic2";
    std::string ftype = "f32";
    int sample_rate = 44100;
    int base_chunk_size = 512;
    int ttl_chunk_compress_factor = 6;
    int latent_dim = 24;
    int latent_channels = 144;
    int default_steps = 5;
    float default_speed = 1.05f;
    std::string language_wrap_mode = "open_close";
    std::string default_voice = "F1";
};

struct supertonic_voice_style {
    std::string name;
    ggml_tensor * ttl = nullptr; // (256, 50, 1) in ggml axis order for JSON (1, 50, 256)
    ggml_tensor * dp  = nullptr; // (16, 8, 1) in ggml axis order for JSON (1, 8, 16)
};

struct supertonic_vocoder_convnext_weights {
    ggml_tensor * dw_w = nullptr;
    ggml_tensor * dw_b = nullptr;
    ggml_tensor * norm_g = nullptr;
    ggml_tensor * norm_b = nullptr;
    ggml_tensor * pw1_w = nullptr;
    ggml_tensor * pw1_b = nullptr;
    ggml_tensor * pw2_w = nullptr;
    ggml_tensor * pw2_b = nullptr;
    ggml_tensor * gamma = nullptr;
};

struct supertonic_vocoder_weights {
    ggml_tensor * normalizer_scale = nullptr;
    ggml_tensor * latent_mean = nullptr;
    ggml_tensor * latent_std = nullptr;
    ggml_tensor * embed_w = nullptr;
    ggml_tensor * embed_b = nullptr;
    std::array<supertonic_vocoder_convnext_weights, 10> convnext{};
    ggml_tensor * final_norm_g = nullptr;
    ggml_tensor * final_norm_b = nullptr;
    ggml_tensor * final_norm_running_mean = nullptr;
    ggml_tensor * final_norm_running_var = nullptr;
    ggml_tensor * head1_w = nullptr;
    ggml_tensor * head1_b = nullptr;
    ggml_tensor * head_prelu = nullptr;
    ggml_tensor * head2_w = nullptr;
};

struct supertonic_trace_tensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct supertonic_model {
    supertonic_hparams hparams;
    supertonic_vocoder_weights vocoder;

    uint64_t generation_id = 0;
    int n_threads = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;

    std::map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, ggml_tensor *> source_tensors;
    std::unordered_map<std::string, supertonic_voice_style> voices;

    std::vector<int32_t> unicode_indexer;
    std::vector<std::string> languages;
    std::string tts_json;
};

bool load_supertonic_gguf(const std::string & path,
                          supertonic_model & model,
                          int n_gpu_layers = 0,
                          bool verbose = false);
void free_supertonic_model(supertonic_model & model);
void supertonic_set_n_threads(supertonic_model & model, int n_threads);
void supertonic_graph_compute(const supertonic_model & model, ggml_cgraph * graph);

ggml_tensor * require_tensor(const supertonic_model & model, const std::string & name);
ggml_tensor * require_source_tensor(const supertonic_model & model, const std::string & source_name);

std::string supertonic_preprocess_text(const std::string & text,
                                       const std::string & language,
                                       const std::string & language_wrap_mode);
bool supertonic_text_to_ids(const supertonic_model & model,
                            const std::string & text,
                            const std::string & language,
                            std::vector<int32_t> & ids,
                            std::string * normalized_text = nullptr,
                            std::string * error = nullptr);

bool supertonic_vocoder_forward_cpu(const supertonic_model & model,
                                    const float * latent,
                                    int latent_len,
                                    std::vector<float> & wav_out,
                                    std::string * error = nullptr);

bool supertonic_vocoder_forward_ggml(const supertonic_model & model,
                                     const float * latent,
                                     int latent_len,
                                     std::vector<float> & wav_out,
                                     std::string * error = nullptr);

bool supertonic_vocoder_trace_scalar(const supertonic_model & model,
                                     const float * latent,
                                     int latent_len,
                                     std::vector<supertonic_trace_tensor> & trace_out,
                                     std::string * error = nullptr);

bool supertonic_vocoder_trace_ggml(const supertonic_model & model,
                                   const float * latent,
                                   int latent_len,
                                   std::vector<supertonic_trace_tensor> & trace_out,
                                   std::string * error = nullptr);

bool supertonic_duration_forward_cpu(const supertonic_model & model,
                                     const int64_t * text_ids,
                                     int text_len,
                                     const float * style_dp,
                                     float & duration_out,
                                     std::string * error = nullptr);

bool supertonic_duration_forward_ggml(const supertonic_model & model,
                                      const int64_t * text_ids,
                                      int text_len,
                                      const float * style_dp,
                                      float & duration_out,
                                      std::string * error = nullptr);

bool supertonic_duration_trace_ggml(const supertonic_model & model,
                                    const int64_t * text_ids,
                                    int text_len,
                                    std::vector<supertonic_trace_tensor> & scalar_trace,
                                    std::vector<supertonic_trace_tensor> & ggml_trace,
                                    std::string * error = nullptr,
                                    bool include_scalar_trace = true,
                                    bool include_ggml_trace = true,
                                    std::vector<float> * sentence_proj_out = nullptr);

bool supertonic_text_encoder_forward_cpu(const supertonic_model & model,
                                         const int64_t * text_ids,
                                         int text_len,
                                         const float * style_ttl,
                                         std::vector<float> & text_emb_out,
                                         std::string * error = nullptr);

bool supertonic_text_encoder_forward_ggml(const supertonic_model & model,
                                          const int64_t * text_ids,
                                          int text_len,
                                          const float * style_ttl,
                                          std::vector<float> & text_emb_out,
                                          std::string * error = nullptr);

bool supertonic_text_encoder_trace_ggml(const supertonic_model & model,
                                        const int64_t * text_ids,
                                        int text_len,
                                        std::vector<supertonic_trace_tensor> & scalar_trace,
                                        std::vector<supertonic_trace_tensor> & ggml_trace,
                                        std::string * error = nullptr);

bool supertonic_vector_step_cpu(const supertonic_model & model,
                                const float * noisy_latent,
                                int latent_len,
                                const float * text_emb,
                                int text_len,
                                const float * style_ttl,
                                const float * latent_mask,
                                int current_step,
                                int total_steps,
                                std::vector<float> & next_latent_out,
                                std::string * error = nullptr);

bool supertonic_vector_step_ggml(const supertonic_model & model,
                                 const float * noisy_latent,
                                 int latent_len,
                                 const float * text_emb,
                                 int text_len,
                                 const float * style_ttl,
                                 const float * latent_mask,
                                 int current_step,
                                 int total_steps,
                                 std::vector<float> & next_latent_out,
                                 std::string * error = nullptr);

bool supertonic_vector_trace_proj_ggml(const supertonic_model & model,
                                       const float * noisy_latent,
                                       const float * text_emb,
                                       int text_len,
                                       const float * style_ttl,
                                       const float * latent_mask,
                                       int latent_len,
                                       int current_step,
                                       int total_steps,
                                       std::vector<supertonic_trace_tensor> & scalar_trace,
                                       std::vector<supertonic_trace_tensor> & ggml_trace,
                                       std::string * error = nullptr,
                                       bool include_scalar_trace = true,
                                       bool include_ggml_trace = true,
                                       std::vector<float> * next_latent_tc_out = nullptr);

// Process-wide alive registry: each loaded supertonic_model registers
// its generation_id with this set on success and unregisters at the
// start of free_supertonic_model.  The thread_local graph caches in
// supertonic_vocoder.cpp / supertonic_text_encoder.cpp /
// supertonic_vector_estimator.cpp own ggml_gallocr_t handles allocated
// against a specific model's ggml_backend_t; on a cache miss the
// existing teardown code calls ggml_gallocr_free(cache.allocr).  When
// the model that backed the cache has already been destroyed, that
// free path asserts inside the GPU-backend dylib finaliser.  The
// is_supertonic_alive() check at every free_*_cache() site lets the
// teardown skip the gallocr_free call for a generation that's no
// longer alive (the underlying GPU buffers were freed when the
// model's backend was freed; we only leak the gallocr bookkeeping
// struct itself, ~80 bytes per cache type).
//
// Thread-safe: backed by a single std::mutex.  Lookup is on the
// hot per-call path inside free_*_cache(), but the lock is held only
// for an unordered_set::find() so contention is minimal.
void register_supertonic_alive(uint64_t generation_id);
void unregister_supertonic_alive(uint64_t generation_id);
bool is_supertonic_alive(uint64_t generation_id);

// Helper consumed by every per-stage free_*_cache().  Skips the
// ggml_gallocr_free call when the allocr's backend has already been
// torn down (model.generation_id no longer in the alive registry).
inline void supertonic_safe_gallocr_free(ggml_gallocr_t & allocr, uint64_t generation_id) {
    if (allocr && is_supertonic_alive(generation_id)) {
        ggml_gallocr_free(allocr);
    }
    allocr = nullptr;
}

} // namespace tts_cpp::supertonic::detail
