// ANEForge encoder backend (see whisper-aneforge.h). dlopen's the ANEForge dispatch
// dylib (libane_e5rt_dispatch.dylib) so no extra link configuration is needed; the
// path comes from ANEFORGE_DYLIB. Compiles the persisted encoder MIL once at init
// (compile-with-cache; the on-device program is not cold-loadable across processes),
// then dispatches it per encode.
#include "whisper-aneforge.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <dlfcn.h>

typedef struct ane_e5rt_program ane_e5rt_program_t;
typedef ane_e5rt_program_t * (*compile_fn)(const char *, const char *, uint64_t,
    const char * const *, const size_t *, size_t, const char * const *, const size_t *, size_t);
typedef int  (*set_in_fn)(ane_e5rt_program_t *, const char *, const uint16_t *, size_t);
typedef int  (*get_out_fn)(ane_e5rt_program_t *, const char *, uint16_t *, size_t);
typedef int  (*exec_fn)(ane_e5rt_program_t *);
typedef void (*release_fn)(ane_e5rt_program_t *);

struct whisper_aneforge_context {
    void * dl = nullptr;
    ane_e5rt_program_t * prog = nullptr;
    set_in_fn set_input = nullptr;
    get_out_fn get_output = nullptr;
    exec_fn execute = nullptr;
    release_fn release = nullptr;
    std::string mel_port, pos_port, out_port;
    size_t mel_n = 0, pos_n = 0, out_n = 0;
    std::vector<uint16_t> mel16;
    std::vector<uint16_t> out16;
};

// arm64 has a native IEEE binary16 type; these casts compile to NEON fcvt and
// auto-vectorize, unlike branchy scalar bit-twiddling. ANEForge's fp16 ports are
// IEEE binary16, which __fp16 matches bit-for-bit.
static inline void f32_to_f16(const float * src, uint16_t * dst, size_t n) {
    __fp16 * d = (__fp16 *) dst;
    for (size_t i = 0; i < n; i++) d[i] = (__fp16) src[i];
}
static inline void f16_to_f32(const uint16_t * src, float * dst, size_t n) {
    const __fp16 * s = (const __fp16 *) src;
    for (size_t i = 0; i < n; i++) dst[i] = (float) s[i];
}

struct whisper_aneforge_context * whisper_aneforge_init(const char * bundle_dir) {
    const char * dylib = getenv("ANEFORGE_DYLIB");
    if (!dylib) { fprintf(stderr, "aneforge: set ANEFORGE_DYLIB to libane_e5rt_dispatch.dylib\n"); return nullptr; }
    void * dl = dlopen(dylib, RTLD_NOW | RTLD_LOCAL);
    if (!dl) { fprintf(stderr, "aneforge: dlopen(%s) failed: %s\n", dylib, dlerror()); return nullptr; }

    auto ctx = new whisper_aneforge_context();
    ctx->dl = dl;
    auto compile  = (compile_fn) dlsym(dl, "ane_e5rt_program_compile");
    ctx->set_input  = (set_in_fn)  dlsym(dl, "ane_e5rt_program_set_input_fp16");
    ctx->get_output = (get_out_fn) dlsym(dl, "ane_e5rt_program_get_output_fp16");
    ctx->execute    = (exec_fn)    dlsym(dl, "ane_e5rt_program_execute");
    ctx->release    = (release_fn) dlsym(dl, "ane_e5rt_program_release");
    if (!compile || !ctx->set_input || !ctx->get_output || !ctx->execute || !ctx->release) {
        fprintf(stderr, "aneforge: missing dispatch symbols\n"); whisper_aneforge_free(ctx); return nullptr;
    }

    // ports.txt: three lines "name nelems" for mel, pos, output (in that order).
    std::string dir = bundle_dir;
    FILE * pf = fopen((dir + "/ports.txt").c_str(), "r");
    if (!pf) { fprintf(stderr, "aneforge: no ports.txt in %s\n", bundle_dir); whisper_aneforge_free(ctx); return nullptr; }
    char nm[256]; size_t ne;
    std::string * ports[3] = {&ctx->mel_port, &ctx->pos_port, &ctx->out_port};
    size_t * nes[3] = {&ctx->mel_n, &ctx->pos_n, &ctx->out_n};
    for (int i = 0; i < 3; i++) { if (fscanf(pf, "%255s %zu", nm, &ne) != 2) { fclose(pf); whisper_aneforge_free(ctx); return nullptr; } *ports[i] = nm; *nes[i] = ne; }
    fclose(pf);

    std::string mil = dir + "/model.mil", cache = dir + "/cache";
    const char * in_names[2] = {ctx->mel_port.c_str(), ctx->pos_port.c_str()};
    size_t in_bytes[2] = {ctx->mel_n * 2, ctx->pos_n * 2};
    const char * out_name = ctx->out_port.c_str();
    size_t out_bytes = ctx->out_n * 2;
    ctx->prog = compile(mil.c_str(), cache.c_str(), 0x4 /*ANE*/, in_names, in_bytes, 2, &out_name, &out_bytes, 1);
    if (!ctx->prog) { fprintf(stderr, "aneforge: compile failed\n"); whisper_aneforge_free(ctx); return nullptr; }

    // The positional-embedding port is a constant; set it once from pos.f16.
    std::vector<uint16_t> pos(ctx->pos_n);
    FILE * pp = fopen((dir + "/pos.f16").c_str(), "rb");
    if (!pp || fread(pos.data(), 2, ctx->pos_n, pp) != ctx->pos_n) { fprintf(stderr, "aneforge: pos.f16 read failed\n"); if (pp) fclose(pp); whisper_aneforge_free(ctx); return nullptr; }
    fclose(pp);
    ctx->set_input(ctx->prog, ctx->pos_port.c_str(), pos.data(), ctx->pos_n);

    ctx->mel16.resize(ctx->mel_n);
    ctx->out16.resize(ctx->out_n);
    fprintf(stderr, "aneforge: encoder ready (mel=%s pos=%s out=%s)\n",
            ctx->mel_port.c_str(), ctx->pos_port.c_str(), ctx->out_port.c_str());
    return ctx;
}

void whisper_aneforge_encode(struct whisper_aneforge_context * ctx,
                             int64_t n_mel, int64_t n_len, const float * mel, float * out) {
    size_t n = (size_t) n_mel * (size_t) n_len;
    if (n != ctx->mel_n) { fprintf(stderr, "aneforge: mel size %zu != %zu\n", n, ctx->mel_n); return; }
    f32_to_f16(mel, ctx->mel16.data(), n);
    ctx->set_input(ctx->prog, ctx->mel_port.c_str(), ctx->mel16.data(), ctx->mel_n);
    ctx->execute(ctx->prog);
    ctx->get_output(ctx->prog, ctx->out_port.c_str(), ctx->out16.data(), ctx->out_n);
    // The encoder output is [S, d_model] row-major, which is whisper.cpp's embd_enc layout.
    f16_to_f32(ctx->out16.data(), out, ctx->out_n);
}

void whisper_aneforge_free(struct whisper_aneforge_context * ctx) {
    if (!ctx) return;
    if (ctx->prog && ctx->release) ctx->release(ctx->prog);
    if (ctx->dl) dlclose(ctx->dl);
    delete ctx;
}
