#pragma once

#include "moss/backend_request.h"
#include "tts-cpp/fit.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace tts_cpp::moss::detail {

inline constexpr const char * SFX_TOKEN_TABLE = "text.token_embd.weight";

struct SfxTextConfig {
    int n_layers   = 0;
    int n_embd     = 0;
    int n_ff       = 0;
    int n_heads    = 0;
    int n_kv_heads = 0;
    int head_dim   = 0;
    int max_tokens = 0;
    float rope_base = 1000000.0f;
    float rms_eps   = 1e-6f;
};

struct SfxDitConfig {
    int n_layers     = 0;
    int n_embd       = 0;
    int n_ff         = 0;
    int n_heads      = 0;
    int in_channels  = 0;
    int out_channels = 0;
    int text_dim     = 0;
    int freq_dim     = 0;
    float eps        = 1e-6f;
};

struct SfxVaeConfig {
    int latent_dim  = 0;
    int decoder_dim = 0;
    int sample_rate = 0;
    std::vector<int> rates;
    int hop_length() const;
};

struct SfxConfig {
    SfxTextConfig text;
    SfxDitConfig dit;
    SfxVaeConfig vae;
    float max_seconds = 0.0f;
    float sigma_shift = 0.0f;
    int train_timesteps = 0;
    int default_steps = 0;
    float default_guidance = 0.0f;
    int latent_frames() const;
};

void require_shape(const ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2 = 1);
void require_matrix(const ggml_tensor * tensor, int64_t ne0, int64_t ne1);
void require_vector(const ggml_tensor * tensor, int64_t ne0, int64_t ne1);

struct SfxLayout {
    TablePlacement table = TablePlacement::Automatic;
    AttentionLayout attention = AttentionLayout::Automatic;
};

struct SfxMemory {
    uint64_t device_bytes = 0;
    uint64_t host_compute_bytes = 0;
    uint64_t host_bytes = 0;
};

class SfxGraph {
public:
    explicit SfxGraph(int max_nodes);
    ~SfxGraph();
    SfxGraph(const SfxGraph &) = delete;
    SfxGraph & operator=(const SfxGraph &) = delete;

    ggml_context * ctx() const;
    ggml_cgraph * graph() const;
    ggml_tensor * input_f32(int64_t ne0, int64_t ne1);
    ggml_tensor * input_i32(int64_t ne0);

private:
    ggml_context * ctx_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
};

class SfxModel {
public:
    SfxModel(const std::string & path, const BackendRequest & backend, int n_threads, bool measure_only = false,
             const SfxLayout & layout = {});
    ~SfxModel();
    SfxModel(const SfxModel &) = delete;
    SfxModel & operator=(const SfxModel &) = delete;

    const SfxConfig & config() const;
    const char * backend_name() const;
    bool table_on_host() const;
    bool fused_attention() const;
    ggml_tensor * tensor(const std::string & name) const;
    std::vector<std::string> tokenizer_tokens() const;
    std::vector<std::string> tokenizer_merges() const;

    void allocate(SfxGraph & graph);
    void compute(SfxGraph & graph);
    SfxMemory measure(SfxGraph & graph);
    FitResult measure_weights();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss::detail
