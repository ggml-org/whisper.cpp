#include "dit_gguf.h"

#include "backend_registry.h"
#include "ggml-backend.h"

#include <cstdio>
#include <stdexcept>

namespace tts_cpp::acestep {

static constexpr size_t ACE_PROBE_ALIGNMENT = 64;

bool dit_gguf_open(DitGGUF & g, const std::string & path) {
    if (!mapped_file_open(g.file, path, "acestep-dit")) return false;
    struct gguf_init_params p = { /*no_alloc=*/true, /*ctx=*/&g.meta };
    g.ctx                     = gguf_init_from_file(path.c_str(), p);
    if (!g.ctx) {
        fprintf(stderr, "[acestep-dit] failed to parse %s\n", path.c_str());
        return false;
    }
    g.data_off = gguf_get_data_offset(g.ctx);
    return true;
}

void dit_gguf_close(DitGGUF & g) {
    if (g.ctx) gguf_free(g.ctx);
    if (g.meta) ggml_free(g.meta);
    mapped_file_close(g.file);
    g.ctx  = nullptr;
    g.meta = nullptr;
    g.data_off = 0;
}

const void * dit_gdata(const DitGGUF & g, const std::string & name) {
    int64_t idx = gguf_find_tensor(g.ctx, name.c_str());
    if (idx < 0) return nullptr;
    return g.file.data + g.data_off + gguf_get_tensor_offset(g.ctx, idx);
}

ggml_tensor * dit_gmeta(const DitGGUF & g, const std::string & name) {
    return ggml_get_tensor(g.meta, name.c_str());
}

ggml_backend_buffer_t dit_gguf_cpu_map_buffer(const DitGGUF & g) {
    // One CPU buffer wrapping the whole mmap; every mapped tensor's data lives
    // at an offset inside [g.file.data, g.file.data + g.file.size). The buffer
    // is read-only in practice (weights are never written); freeing it does not
    // munmap (that is dit_gguf_close's job at model teardown).
    return ggml_backend_cpu_buffer_from_ptr((void *) g.file.data, g.file.size);
}

bool dit_gguf_backend_maps_weights(ggml_backend_t backend) {
    if (!ggml_backend_buft_is_host(ggml_backend_get_default_buffer_type(backend))) return false;
    alignas(ACE_PROBE_ALIGNMENT) uint8_t probe[ACE_PROBE_ALIGNMENT];
    ggml_backend_buffer_t host = ggml_backend_cpu_buffer_from_ptr(probe, sizeof(probe));
    if (!host) return false;
    const bool computes_on_host = ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(host));
    ggml_backend_buffer_free(host);
    return computes_on_host;
}

ggml_backend_buffer_type_t dit_gguf_weight_buffer_type(ggml_backend_t backend) {
    ggml_backend_buffer_type_t fallback = ggml_backend_get_default_buffer_type(backend);
    if (!backend_name_is_hexagon(backend_reg_name(backend))) return fallback;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto get_extra = reg ? (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
                               reg, "ggml_backend_dev_get_extra_bufts")
                         : nullptr;
    ggml_backend_buffer_type_t * extra = get_extra ? get_extra(dev) : nullptr;
    return extra && extra[0] ? extra[0] : fallback;
}

static std::string hexagon_weight_type_names() {
    std::string names;
    for (const ggml_type type : HEXAGON_WEIGHT_TYPES) {
        if (!names.empty()) names += ", ";
        names += ggml_type_name(type);
    }
    return names;
}

std::string hexagon_weight_refusal(const gguf_context * weights, int64_t index, const std::string & path,
                                   const char * tag) {
    return std::string("[") + tag + "] " + path + ": " + gguf_get_tensor_name(weights, index) + " is " +
           ggml_type_name(gguf_get_tensor_type(weights, index)) + "; the Hexagon NPU runs " +
           hexagon_weight_type_names() + " weights: use a GGUF in one of those types or another backend";
}

std::string dit_gguf_hexagon_refusal(const gguf_context * weights, const std::string & path,
                                     ggml_backend_t backend, const char * tag) {
    if (!backend_name_is_hexagon(backend_reg_name(backend))) return "";
    const int64_t refused = first_tensor_hexagon_cannot_run(weights);
    if (refused < 0) return "";
    return hexagon_weight_refusal(weights, refused, path, tag);
}

void dit_gguf_require_hexagon_weights(DitGGUF & g, const std::string & path, ggml_backend_t backend,
                                      const char * tag) {
    const std::string refusal = dit_gguf_hexagon_refusal(g.ctx, path, backend, tag);
    if (refusal.empty()) return;
    dit_gguf_close(g);
    throw std::runtime_error(refusal);
}

// GGUF tensor data is aligned to general.alignment (default 32); ggml's CPU quant
// kernels rely on that. Refuse to map (fall back to copy) if a tensor's pointer
// is under-aligned, so we never hand the CPU backend a misaligned quant block.
static constexpr uintptr_t ACE_TENSOR_ALIGNMENT = 32;

bool dit_gguf_map_tensor(ggml_tensor * dst, const DitGGUF & g, const std::string & name,
                         ggml_backend_buffer_t map_buf) {
    if (!dst) return false;
    const void * src = dit_gdata(g, name);
    if (!src) {
        fprintf(stderr, "[acestep] cannot map tensor (absent): %s\n", name.c_str());
        return false;
    }
    // Bounds + alignment guard against a truncated / corrupt GGUF: the tensor's
    // bytes must lie wholly inside the mapping (else graph_compute would SIGBUS
    // reading past the end) and be aligned for the quant kernels. On failure
    // leave ->data NULL so the caller allocates + copies instead of mapping.
    const size_t nb  = ggml_nbytes(dst);
    const size_t off = (const uint8_t *) src - g.file.data;
    if (off > g.file.size || nb > g.file.size - off || ((uintptr_t) src % ACE_TENSOR_ALIGNMENT) != 0) {
        fprintf(stderr, "[acestep] refusing to map %s: out-of-bounds or misaligned in %zu-byte mapping\n",
                name.c_str(), g.file.size);
        return false;
    }
    // Point the tensor at the mmap and attach the shared CPU buffer. With ->data
    // set, ggml_backend_alloc_ctx_tensors skips this tensor (no dirty RAM), and
    // the direct single-backend graph_compute reads the weight straight from the
    // mapped page.
    dst->data   = (void *) src;
    dst->buffer = map_buf;
    return true;
}

bool dit_gguf_has(const DitGGUF & g, const std::string & name) {
    return gguf_find_tensor(g.ctx, name.c_str()) >= 0;
}

uint32_t dit_gguf_u32(const DitGGUF & g, const std::string & key) {
    int64_t id = gguf_find_key(g.ctx, key.c_str());
    if (id < 0) throw std::runtime_error("acestep-dit: missing GGUF key: " + key);
    return gguf_get_val_u32(g.ctx, id);
}

float dit_gguf_f32(const DitGGUF & g, const std::string & key) {
    int64_t id = gguf_find_key(g.ctx, key.c_str());
    if (id < 0) throw std::runtime_error("acestep-dit: missing GGUF key: " + key);
    return gguf_get_val_f32(g.ctx, id);
}

bool dit_gguf_bool(const DitGGUF & g, const std::string & key, bool def) {
    int64_t id = gguf_find_key(g.ctx, key.c_str());
    if (id < 0) return def;
    return gguf_get_val_bool(g.ctx, id);
}

std::string dit_gguf_str(const DitGGUF & g, const std::string & key, const std::string & def) {
    int64_t id = gguf_find_key(g.ctx, key.c_str());
    if (id < 0) return def;
    return gguf_get_val_str(g.ctx, id);
}

bool dit_gguf_read_config(const DitGGUF & g, DitConfig & cfg) {
    try {
        cfg.n_layers          = (int) dit_gguf_u32(g, "acestep-dit.block_count");
        cfg.hidden_size       = (int) dit_gguf_u32(g, "acestep-dit.embedding_length");
        cfg.intermediate_size = (int) dit_gguf_u32(g, "acestep-dit.feed_forward_length");
        cfg.n_heads           = (int) dit_gguf_u32(g, "acestep-dit.attention.head_count");
        cfg.n_kv_heads        = (int) dit_gguf_u32(g, "acestep-dit.attention.head_count_kv");
        cfg.head_dim          = (int) dit_gguf_u32(g, "acestep-dit.attention.key_length");
        cfg.in_channels       = (int) dit_gguf_u32(g, "acestep.in_channels");
        cfg.out_channels      = (int) dit_gguf_u32(g, "acestep.audio_acoustic_hidden_dim");
        cfg.patch_size        = (int) dit_gguf_u32(g, "acestep.patch_size");
        cfg.sliding_window    = (int) dit_gguf_u32(g, "acestep.sliding_window");
        cfg.rope_theta        = dit_gguf_f32(g, "acestep-dit.rope.freq_base");
        cfg.rms_norm_eps      = dit_gguf_f32(g, "acestep-dit.attention.layer_norm_rms_epsilon");
        // convert.py only writes acestep.is_turbo when true; absent => base/sft.
        cfg.is_turbo          = dit_gguf_bool(g, "acestep.is_turbo", false);
        cfg.model_name        = dit_gguf_str(g, "general.name", "");
        cfg.is_sft            = is_sft_model_name(cfg.model_name);
    } catch (const std::exception & e) {
        fprintf(stderr, "[acestep-dit] %s\n", e.what());
        return false;
    }
    const bool ok = cfg.n_layers && cfg.hidden_size && cfg.intermediate_size && cfg.n_heads &&
                    cfg.n_kv_heads && cfg.head_dim && cfg.in_channels && cfg.out_channels &&
                    cfg.patch_size && cfg.sliding_window && cfg.rope_theta > 0.0f &&
                    cfg.rms_norm_eps > 0.0f;
    if (!ok) fprintf(stderr, "[acestep-dit] incomplete DiT config in GGUF\n");
    return ok;
}

} // namespace tts_cpp::acestep
