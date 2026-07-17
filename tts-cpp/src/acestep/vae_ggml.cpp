#include "vae_ggml.h"

#include "vae_gguf.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace tts_cpp::acestep {

static const int UPSAMPLE = 10 * 6 * 4 * 4 * 2;  // 1920

// ----------------------------------------------------------------- structs
struct ResUnit { ggml_tensor *s1a, *s1b, *c1w, *c1b, *s2a, *s2b, *c2w, *c2b; int dilation; };
struct DecBlock { ggml_tensor *sa, *sb, *ctw, *ctb; int in_ch, out_ch, stride, kernel; ResUnit ru[3]; };
struct Decoder  { ggml_tensor *c1w, *c1b; DecBlock blk[5]; ggml_tensor *sa, *sb, *c2w; };

struct EncBlock { ResUnit ru[3]; ggml_tensor *sa, *sb, *dw, *db; int in_ch, out_ch, stride, kernel, padding; };
struct Encoder  { ggml_tensor *c1w, *c1b; EncBlock blk[5]; ggml_tensor *sa, *sb, *c2w, *c2b; };

struct VaeModel {
    ggml_backend_t        backend    = nullptr;  // borrowed
    ggml_context *        weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    Decoder               dec        = {};
    Encoder               enc        = {};
    bool                  has_enc    = false;
};

// ----------------------------------------------------------------- ops
static ggml_tensor * op_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * a, ggml_tensor * inv_b) {
    return ggml_snake(ctx, x, a, inv_b);
}

static ggml_tensor * op_conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                               ggml_tensor * x, int stride, int pad, int dil) {
    ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, pad, dil);
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return y;
}

static ggml_tensor * op_conv_t1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                                 ggml_tensor * x, int stride, int pad, int oc) {
    ggml_tensor * xt  = ggml_cont(ctx, ggml_transpose(ctx, x));  // [IC, T_in]
    ggml_tensor * col = ggml_mul_mat(ctx, w, xt);                // [K*OC, T_in]
    ggml_tensor * y   = ggml_col2im_1d(ctx, col, stride, oc, pad);
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return y;
}

static ggml_tensor * op_res_unit(ggml_context * ctx, ResUnit * ru, ggml_tensor * x) {
    ggml_tensor * skip = x;
    const int pad = 3 * ru->dilation;  // (k-1)*dil/2 with k=7
    x = op_snake(ctx, x, ru->s1a, ru->s1b);
    x = op_conv1d(ctx, ru->c1w, ru->c1b, x, 1, pad, ru->dilation);
    x = op_snake(ctx, x, ru->s2a, ru->s2b);
    x = op_conv1d(ctx, ru->c2w, ru->c2b, x, 1, 0, 1);
    return ggml_add(ctx, skip, x);
}

// ----------------------------------------------------------------- tensor creation
static void decoder_create(Decoder & m, ggml_context * ctx) {
    static const int STR[5] = { 10, 6, 4, 4, 2 };
    static const int IC[5]  = { 2048, 1024, 512, 256, 128 };
    static const int OC[5]  = { 1024, 512, 256, 128, 128 };
    static const int DIL[3] = { 1, 3, 9 };
    m.c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 64, 2048);
    m.c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    for (int i = 0; i < 5; ++i) {
        DecBlock & b = m.blk[i];
        b.in_ch = IC[i]; b.out_ch = OC[i]; b.stride = STR[i]; b.kernel = STR[i] * 2;
        const int C = b.out_ch;
        b.sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, b.in_ch);
        b.sb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, b.in_ch);
        b.ctw = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, b.in_ch, b.kernel * b.out_ch);
        b.ctb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, b.out_ch);
        for (int r = 0; r < 3; ++r) {
            ResUnit & ru = b.ru[r];
            ru.dilation = DIL[r];
            ru.s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s1b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, C, C);
            ru.c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            ru.s2a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s2b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, C, C);
            ru.c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        }
    }
    m.sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 128);
    m.sb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 128);
    m.c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 128, 2);
}

static void encoder_create(Encoder & m, ggml_context * ctx) {
    static const int IC[5]  = { 128, 128, 256, 512, 1024 };
    static const int OC[5]  = { 128, 256, 512, 1024, 2048 };
    static const int STR[5] = { 2, 4, 4, 6, 10 };
    static const int DIL[3] = { 1, 3, 9 };
    m.c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 2, 128);
    m.c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    for (int i = 0; i < 5; ++i) {
        EncBlock & b = m.blk[i];
        b.in_ch = IC[i]; b.out_ch = OC[i]; b.stride = STR[i];
        b.kernel = STR[i] * 2; b.padding = (STR[i] + 1) / 2;
        const int C = b.in_ch;
        for (int r = 0; r < 3; ++r) {
            ResUnit & ru = b.ru[r];
            ru.dilation = DIL[r];
            ru.s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s1b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, C, C);
            ru.c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            ru.s2a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s2b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, C, C);
            ru.c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        }
        b.sa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
        b.sb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
        b.dw = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, b.kernel, b.in_ch, b.out_ch);
        b.db = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, b.out_ch);
    }
    m.sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2048);
    m.sb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2048);
    m.c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, 2048, 128);
    m.c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
}

// ----------------------------------------------------------------- weight load
static void decoder_load(Decoder & m, const VaeGGUF & g) {
    vae_fuse_wn(m.c1w, g, "decoder.conv1");
    vae_load_bias(m.c1b, g, "decoder.conv1.bias");
    for (int i = 0; i < 5; ++i) {
        DecBlock &  b  = m.blk[i];
        std::string bp = "decoder.block." + std::to_string(i);
        vae_load_snake(b.sa, g, bp + ".snake1.alpha", false);
        vae_load_snake(b.sb, g, bp + ".snake1.beta",  true);
        vae_fuse_wn_ct(b.ctw, g, bp + ".conv_t1");
        vae_load_bias(b.ctb, g, bp + ".conv_t1.bias");
        for (int r = 0; r < 3; ++r) {
            ResUnit &   ru = b.ru[r];
            std::string rp = bp + ".res_unit" + std::to_string(r + 1);
            vae_load_snake(ru.s1a, g, rp + ".snake1.alpha", false);
            vae_load_snake(ru.s1b, g, rp + ".snake1.beta",  true);
            vae_fuse_wn(ru.c1w, g, rp + ".conv1");
            vae_load_bias(ru.c1b, g, rp + ".conv1.bias");
            vae_load_snake(ru.s2a, g, rp + ".snake2.alpha", false);
            vae_load_snake(ru.s2b, g, rp + ".snake2.beta",  true);
            vae_fuse_wn(ru.c2w, g, rp + ".conv2");
            vae_load_bias(ru.c2b, g, rp + ".conv2.bias");
        }
    }
    vae_load_snake(m.sa, g, "decoder.snake1.alpha", false);
    vae_load_snake(m.sb, g, "decoder.snake1.beta",  true);
    vae_fuse_wn(m.c2w, g, "decoder.conv2");
}

static void encoder_load(Encoder & m, const VaeGGUF & g) {
    vae_fuse_wn(m.c1w, g, "encoder.conv1");
    vae_load_bias(m.c1b, g, "encoder.conv1.bias");
    for (int i = 0; i < 5; ++i) {
        EncBlock &  b  = m.blk[i];
        std::string bp = "encoder.block." + std::to_string(i);
        for (int r = 0; r < 3; ++r) {
            ResUnit &   ru = b.ru[r];
            std::string rp = bp + ".res_unit" + std::to_string(r + 1);
            vae_load_snake(ru.s1a, g, rp + ".snake1.alpha", false);
            vae_load_snake(ru.s1b, g, rp + ".snake1.beta",  true);
            vae_fuse_wn(ru.c1w, g, rp + ".conv1");
            vae_load_bias(ru.c1b, g, rp + ".conv1.bias");
            vae_load_snake(ru.s2a, g, rp + ".snake2.alpha", false);
            vae_load_snake(ru.s2b, g, rp + ".snake2.beta",  true);
            vae_fuse_wn(ru.c2w, g, rp + ".conv2");
            vae_load_bias(ru.c2b, g, rp + ".conv2.bias");
        }
        vae_load_snake(b.sa, g, bp + ".snake1.alpha", false);
        vae_load_snake(b.sb, g, bp + ".snake1.beta",  true);
        vae_fuse_wn(b.dw, g, bp + ".conv1");
        vae_load_bias(b.db, g, bp + ".conv1.bias");
    }
    vae_load_snake(m.sa, g, "encoder.snake1.alpha", false);
    vae_load_snake(m.sb, g, "encoder.snake1.beta",  true);
    vae_fuse_wn(m.c2w, g, "encoder.conv2");
    vae_load_bias(m.c2b, g, "encoder.conv2.bias");
}

// ----------------------------------------------------------------- graphs
static ggml_tensor * build_decode(ggml_context * ctx, Decoder * m, ggml_tensor * latent) {
    ggml_tensor * x = op_conv1d(ctx, m->c1w, m->c1b, latent, 1, 3, 1);  // [T, 2048]
    for (int i = 0; i < 5; ++i) {
        DecBlock & b = m->blk[i];
        x = op_snake(ctx, x, b.sa, b.sb);
        const int pad = (b.kernel - b.stride) / 2;
        x = op_conv_t1d(ctx, b.ctw, b.ctb, x, b.stride, pad, b.out_ch);
        for (int r = 0; r < 3; ++r) x = op_res_unit(ctx, &b.ru[r], x);
    }
    x = op_snake(ctx, x, m->sa, m->sb);
    x = op_conv1d(ctx, m->c2w, nullptr, x, 1, 3, 1);  // [T_audio, 2]
    return x;
}

static ggml_tensor * build_encode(ggml_context * ctx, Encoder * m, ggml_tensor * audio) {
    ggml_tensor * x = op_conv1d(ctx, m->c1w, m->c1b, audio, 1, 3, 1);  // [T, 128]
    for (int i = 0; i < 5; ++i) {
        EncBlock & b = m->blk[i];
        for (int r = 0; r < 3; ++r) x = op_res_unit(ctx, &b.ru[r], x);
        x = op_snake(ctx, x, b.sa, b.sb);
        x = op_conv1d(ctx, b.dw, b.db, x, b.stride, b.padding, 1);
    }
    x = op_snake(ctx, x, m->sa, m->sb);
    x = op_conv1d(ctx, m->c2w, m->c2b, x, 1, 1, 1);  // [T_latent, 128]
    return x;
}

// ----------------------------------------------------------------- public
VaeModel * vae_model_load(const std::string & path, ggml_backend_t backend, bool with_encoder, bool verbose) {
    VaeGGUF g;
    if (!vae_gguf_open(g, path)) return nullptr;

    // encoder tensors only exist in the file if it's a full VAE
    if (with_encoder && !vae_gguf_has(g, "encoder.conv1.weight_v")) {
        if (verbose) fprintf(stderr, "[acestep-vae] GGUF has no encoder tensors; loading decoder only\n");
        with_encoder = false;
    }

    VaeModel * m = new VaeModel();
    m->backend = backend;
    m->has_enc = with_encoder;

    ggml_init_params ip{ ggml_tensor_overhead() * 1024, nullptr, /*no_alloc=*/true };
    m->weight_ctx = ggml_init(ip);

    decoder_create(m->dec, m->weight_ctx);
    if (with_encoder) encoder_create(m->enc, m->weight_ctx);

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->weight_ctx, backend);
    if (!m->weight_buf) {
        fprintf(stderr, "[acestep-vae] failed to allocate weight buffer\n");
        ggml_free(m->weight_ctx);
        vae_gguf_close(g);
        delete m;
        return nullptr;
    }

    decoder_load(m->dec, g);
    if (with_encoder) encoder_load(m->enc, g);
    vae_gguf_close(g);

    if (verbose) {
        fprintf(stderr, "[acestep-vae] loaded %s: %.1f MB, %s, upsample=1920x\n", path.c_str(),
                (float) ggml_backend_buffer_get_size(m->weight_buf) / (1024 * 1024),
                with_encoder ? "encoder+decoder" : "decoder");
    }
    return m;
}

void vae_model_free(VaeModel * m) {
    if (!m) return;
    if (m->weight_buf) ggml_backend_buffer_free(m->weight_buf);
    if (m->weight_ctx) ggml_free(m->weight_ctx);
    delete m;
}

bool   vae_model_has_encoder(const VaeModel * m) { return m && m->has_enc; }
size_t vae_model_weight_bytes(const VaeModel * m) { return (m && m->weight_buf) ? ggml_backend_buffer_get_size(m->weight_buf) : 0; }

int vae_model_decode(VaeModel * m, const float * latent, int T_latent, std::vector<float> & pcm_out) {
    ggml_init_params gp{ ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false), nullptr, true };
    ggml_context * ctx = ggml_init(gp);

    ggml_tensor * lat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_latent, 64);
    ggml_set_input(lat);
    ggml_tensor * out = build_decode(ctx, &m->dec, lat);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "[acestep-vae] decode alloc failed (T_latent=%d)\n", T_latent);
        if (ga) ggml_gallocr_free(ga);
        ggml_free(ctx);
        return -1;
    }

    // input ggml [T_latent(ne0), 64(ne1)] channel-major: idx = c*T_latent + t
    std::vector<float> lin((size_t) T_latent * 64);
    for (int c = 0; c < 64; ++c)
        for (int t = 0; t < T_latent; ++t) lin[(size_t) c * T_latent + t] = latent[(size_t) t * 64 + c];
    ggml_backend_tensor_set(lat, lin.data(), 0, lin.size() * sizeof(float));

    int rc = ggml_backend_graph_compute(m->backend, gf);
    if (rc != GGML_STATUS_SUCCESS) { ggml_gallocr_free(ga); ggml_free(ctx); return -1; }

    const int T_audio = (int) out->ne[0];
    std::vector<float> planar((size_t) T_audio * 2);
    ggml_backend_tensor_get(out, planar.data(), 0, ggml_nbytes(out));  // [ch0..][ch1..]

    pcm_out.resize((size_t) T_audio * 2);
    for (int t = 0; t < T_audio; ++t) {
        pcm_out[(size_t) t * 2 + 0] = planar[t];
        pcm_out[(size_t) t * 2 + 1] = planar[(size_t) T_audio + t];
    }

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return T_audio;
}

int vae_model_encode(VaeModel * m, const float * pcm, int frames, std::vector<float> & latent_out) {
    if (!m->has_enc) { fprintf(stderr, "[acestep-vae] encode called but encoder not loaded\n"); return -1; }

    ggml_init_params gp{ ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false), nullptr, true };
    ggml_context * ctx = ggml_init(gp);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frames, 2);
    ggml_set_input(a);
    ggml_tensor * z = build_encode(ctx, &m->enc, a);  // [T_latent, 128]
    ggml_set_output(z);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, z);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "[acestep-vae] encode alloc failed (frames=%d)\n", frames);
        if (ga) ggml_gallocr_free(ga);
        ggml_free(ctx);
        return -1;
    }

    // input ggml [frames(ne0), 2(ne1)] channel-major: idx = c*frames + t
    std::vector<float> ain((size_t) frames * 2);
    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < frames; ++t) ain[(size_t) c * frames + t] = pcm[(size_t) t * 2 + c];
    ggml_backend_tensor_set(a, ain.data(), 0, ain.size() * sizeof(float));

    int rc = ggml_backend_graph_compute(m->backend, gf);
    if (rc != GGML_STATUS_SUCCESS) { ggml_gallocr_free(ga); ggml_free(ctx); return -1; }

    const int T_lat = (int) z->ne[0];
    const int ZC    = (int) z->ne[1];
    std::vector<float> raw((size_t) T_lat * ZC);
    ggml_backend_tensor_get(z, raw.data(), 0, ggml_nbytes(z));

    // extract mean (channels 0..63), store time-major
    latent_out.resize((size_t) T_lat * 64);
    for (int t = 0; t < T_lat; ++t)
        for (int c = 0; c < 64; ++c) latent_out[(size_t) t * 64 + c] = raw[(size_t) c * T_lat + t];

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return T_lat;
}

} // namespace tts_cpp::acestep
