#include "dit_gguf.h"

#include "ggml-backend.h"

#include <cstdio>
#include <stdexcept>

namespace tts_cpp::acestep {

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
