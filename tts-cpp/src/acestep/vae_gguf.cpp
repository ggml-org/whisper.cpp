#include "vae_gguf.h"

#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tts_cpp::acestep {

bool vae_gguf_open(VaeGGUF & g, const std::string & path) {
    g.fd = open(path.c_str(), O_RDONLY);
    if (g.fd < 0) { fprintf(stderr, "[acestep-vae] cannot open %s\n", path.c_str()); return false; }
    struct stat sb;
    if (fstat(g.fd, &sb) != 0) { fprintf(stderr, "[acestep-vae] fstat failed\n"); close(g.fd); g.fd = -1; return false; }
    g.fsize = (size_t) sb.st_size;
    g.map   = (uint8_t *) mmap(nullptr, g.fsize, PROT_READ, MAP_PRIVATE, g.fd, 0);
    if (g.map == MAP_FAILED) { fprintf(stderr, "[acestep-vae] mmap failed\n"); close(g.fd); g.fd = -1; g.map = nullptr; return false; }
    struct gguf_init_params p = { /*no_alloc=*/true, /*ctx=*/&g.meta };
    g.ctx = gguf_init_from_file(path.c_str(), p);
    if (!g.ctx) { fprintf(stderr, "[acestep-vae] failed to parse %s\n", path.c_str()); return false; }
    g.data_off = gguf_get_data_offset(g.ctx);
    return true;
}

void vae_gguf_close(VaeGGUF & g) {
    if (g.ctx) gguf_free(g.ctx);
    if (g.meta) ggml_free(g.meta);
    if (g.map) munmap(g.map, g.fsize);
    if (g.fd >= 0) close(g.fd);
    g = {};
}

const void * vae_gdata(const VaeGGUF & g, const std::string & name) {
    int64_t idx = gguf_find_tensor(g.ctx, name.c_str());
    if (idx < 0) return nullptr;
    return g.map + g.data_off + gguf_get_tensor_offset(g.ctx, idx);
}

ggml_tensor * vae_gmeta(const VaeGGUF & g, const std::string & name) {
    return ggml_get_tensor(g.meta, name.c_str());
}

bool vae_gguf_has(const VaeGGUF & g, const std::string & name) {
    return gguf_find_tensor(g.ctx, name.c_str()) >= 0;
}

static float bf16_to_f32(uint16_t v) {
    ggml_bf16_t b; b.bits = v; return ggml_bf16_to_fp32(b);
}

static void upload_f32_as(ggml_tensor * dst, const std::vector<float> & w) {
    if (dst->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(w.size());
        ggml_fp32_to_fp16_row(w.data(), h.data(), (int) w.size());
        ggml_backend_tensor_set(dst, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(dst, w.data(), 0, w.size() * sizeof(float));
    }
}

void vae_fuse_wn(ggml_tensor * dst, const VaeGGUF & g, const std::string & pfx) {
    ggml_tensor *    mv   = vae_gmeta(g, pfx + ".weight_v");
    const uint16_t * gp   = (const uint16_t *) vae_gdata(g, pfx + ".weight_g");
    const uint16_t * vp   = (const uint16_t *) vae_gdata(g, pfx + ".weight_v");
    const int        nd   = ggml_n_dims(mv);
    const int        dim0 = (int) mv->ne[nd - 1];
    const int        fan  = (int) (ggml_nelements(mv) / dim0);
    std::vector<float> w((size_t) dim0 * fan);
    for (int d = 0; d < dim0; d++) {
        float gv = bf16_to_f32(gp[d]), nsq = 0.0f;
        for (int i = 0; i < fan; i++) { float vv = bf16_to_f32(vp[(size_t) d * fan + i]); nsq += vv * vv; }
        float s = gv / (sqrtf(nsq) + 1e-12f);
        for (int i = 0; i < fan; i++) w[(size_t) d * fan + i] = bf16_to_f32(vp[(size_t) d * fan + i]) * s;
    }
    upload_f32_as(dst, w);
}

void vae_fuse_wn_ct(ggml_tensor * dst, const VaeGGUF & g, const std::string & pfx) {
    ggml_tensor *    mv   = vae_gmeta(g, pfx + ".weight_v");
    const uint16_t * gp   = (const uint16_t *) vae_gdata(g, pfx + ".weight_g");
    const uint16_t * vp   = (const uint16_t *) vae_gdata(g, pfx + ".weight_v");
    const int        nd   = ggml_n_dims(mv);
    const int        dim0 = (int) mv->ne[nd - 1];              // IC
    const int        fan  = (int) (ggml_nelements(mv) / dim0); // K*OC
    std::vector<float> w((size_t) dim0 * fan);
    for (int d = 0; d < dim0; d++) {
        float gv = bf16_to_f32(gp[d]), nsq = 0.0f;
        for (int i = 0; i < fan; i++) { float vv = bf16_to_f32(vp[(size_t) d * fan + i]); nsq += vv * vv; }
        float s = gv / (sqrtf(nsq) + 1e-12f);
        for (int i = 0; i < fan; i++) w[(size_t) i * dim0 + d] = bf16_to_f32(vp[(size_t) d * fan + i]) * s;  // transpose
    }
    upload_f32_as(dst, w);
}

void vae_load_snake(ggml_tensor * dst, const VaeGGUF & g, const std::string & name, bool inv) {
    ggml_tensor *    mt  = vae_gmeta(g, name);
    const int        C   = (int) mt->ne[1];
    const uint16_t * raw = (const uint16_t *) vae_gdata(g, name);
    std::vector<float> d(C);
    for (int i = 0; i < C; i++) { float e = expf(bf16_to_f32(raw[i])); d[i] = inv ? 1.0f / e : e; }
    ggml_backend_tensor_set(dst, d.data(), 0, C * sizeof(float));
}

void vae_load_bias(ggml_tensor * dst, const VaeGGUF & g, const std::string & name) {
    ggml_tensor *    mt  = vae_gmeta(g, name);
    const int        C   = (int) mt->ne[0];
    const uint16_t * raw = (const uint16_t *) vae_gdata(g, name);
    std::vector<float> d(C);
    for (int i = 0; i < C; i++) d[i] = bf16_to_f32(raw[i]);
    ggml_backend_tensor_set(dst, d.data(), 0, C * sizeof(float));
}

} // namespace tts_cpp::acestep
