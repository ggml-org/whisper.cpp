#include "dit_gguf.h"

#include "ggml-backend.h"

#include <cstdio>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tts_cpp::acestep {

bool dit_gguf_open(DitGGUF & g, const std::string & path) {
    g.fd = open(path.c_str(), O_RDONLY);
    if (g.fd < 0) {
        fprintf(stderr, "[acestep-dit] cannot open %s\n", path.c_str());
        return false;
    }
    struct stat sb;
    if (fstat(g.fd, &sb) != 0) {
        fprintf(stderr, "[acestep-dit] fstat failed\n");
        close(g.fd);
        g.fd = -1;
        return false;
    }
    g.fsize = (size_t) sb.st_size;
    g.map   = (uint8_t *) mmap(nullptr, g.fsize, PROT_READ, MAP_PRIVATE, g.fd, 0);
    if (g.map == MAP_FAILED) {
        fprintf(stderr, "[acestep-dit] mmap failed\n");
        close(g.fd);
        g.fd  = -1;
        g.map = nullptr;
        return false;
    }
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
    if (g.map) munmap(g.map, g.fsize);
    if (g.fd >= 0) close(g.fd);
    g = {};
}

const void * dit_gdata(const DitGGUF & g, const std::string & name) {
    int64_t idx = gguf_find_tensor(g.ctx, name.c_str());
    if (idx < 0) return nullptr;
    return g.map + g.data_off + gguf_get_tensor_offset(g.ctx, idx);
}

ggml_tensor * dit_gmeta(const DitGGUF & g, const std::string & name) {
    return ggml_get_tensor(g.meta, name.c_str());
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
