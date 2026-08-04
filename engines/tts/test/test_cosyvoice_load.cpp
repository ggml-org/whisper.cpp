// Unit test for cosyvoice_load_gguf() map-in-place loading (src/cosyvoice_pipeline.cpp).
//
// No model fixture: writes its own synthetic GGUF (mixed dtypes, plus a tensor
// larger than the stream reader's chunk), loads it on the CPU backend and
// asserts every tensor's bytes round-trip exactly through the mmap-backed path.
// Then truncates the data section and asserts the load fails cleanly (throws)
// instead of reading past the mapping.

#include "test_env_portable.h"
#include "cosyvoice_pipeline.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...) do {                                  \
    if (!(cond)) {                                             \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);   \
        fprintf(stderr, __VA_ARGS__);                          \
        fprintf(stderr, "\n");                                 \
        ++g_failures;                                          \
    }                                                          \
} while (0)

struct spec { const char * name; ggml_type type; int64_t ne0; int64_t ne1; };
static const spec SPECS[] = {
    { "w/small_f32", GGML_TYPE_F32, 7,    1 },
    { "w/mat_f32",   GGML_TYPE_F32, 129, 65 },
    { "w/mat_f16",   GGML_TYPE_F16, 256, 33 },
    { "w/q8",        GGML_TYPE_Q8_0, 512, 40 },
    { "w/big_f32",   GGML_TYPE_F32, 1031, 2503 },  // > 8 MiB stream chunk
};
static const size_t N_SPECS = sizeof(SPECS) / sizeof(SPECS[0]);

static std::map<std::string, std::vector<uint8_t>> g_ref;

static std::string write_fixture() {
    const std::string path = test_tmpdir() + "/test-cosyvoice-load-fixture.gguf";

    size_t total = 0;
    for (size_t i = 0; i < N_SPECS; ++i) {
        total += ggml_row_size(SPECS[i].type, SPECS[i].ne0) * (size_t) SPECS[i].ne1;
    }
    ggml_init_params p = { total + (N_SPECS + 1) * ggml_tensor_overhead(), nullptr, false };
    ggml_context * ctx = ggml_init(p);

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "cosyvoice3");

    for (size_t i = 0; i < N_SPECS; ++i) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, SPECS[i].type, SPECS[i].ne0, SPECS[i].ne1);
        ggml_set_name(t, SPECS[i].name);
        uint8_t * d = (uint8_t *) t->data;
        const size_t nb = ggml_nbytes(t);
        for (size_t j = 0; j < nb; ++j) d[j] = (uint8_t) ((j * 131 + i * 31 + 7) & 0xff);
        g_ref[SPECS[i].name] = std::vector<uint8_t>(d, d + nb);
        gguf_add_tensor(g, t);
    }

    if (!gguf_write_to_file(g, path.c_str(), /*only_meta=*/ false)) {
        fprintf(stderr, "FATAL: cannot write fixture %s\n", path.c_str());
        exit(1);
    }
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

static void check_roundtrip(const std::string & path) {
    model_ctx m = cosyvoice_load_gguf(path);
    for (size_t i = 0; i < N_SPECS; ++i) {
        ggml_tensor * t = cosyvoice_get(m, SPECS[i].name);
        CHECK(t != nullptr, "tensor '%s' missing after load", SPECS[i].name);
        if (!t) continue;
        std::vector<uint8_t> got(ggml_nbytes(t));
        ggml_backend_tensor_get(t, got.data(), 0, got.size());
        const auto & want = g_ref.at(SPECS[i].name);
        CHECK(got.size() == want.size(), "tensor '%s' size mismatch", SPECS[i].name);
        CHECK(memcmp(got.data(), want.data(), got.size()) == 0,
              "tensor '%s' bytes differ after map-in-place load", SPECS[i].name);
    }
    cosyvoice_free(m);
}

// Copy the fixture but drop the tail so the last tensor's data runs past EOF.
static std::string write_truncated(const std::string & src) {
    FILE * in = fopen(src.c_str(), "rb");
    if (!in) { fprintf(stderr, "FATAL: cannot reopen %s\n", src.c_str()); exit(1); }
    fseek(in, 0, SEEK_END);
    const long size = ftell(in);
    fseek(in, 0, SEEK_SET);
    std::vector<uint8_t> bytes((size_t) size);
    if (fread(bytes.data(), 1, bytes.size(), in) != bytes.size()) { fclose(in); exit(1); }
    fclose(in);

    const std::string path = test_tmpdir() + "/test-cosyvoice-load-truncated.gguf";
    FILE * out = fopen(path.c_str(), "wb");
    if (!out) { fprintf(stderr, "FATAL: cannot write %s\n", path.c_str()); exit(1); }
    const size_t keep = bytes.size() - (bytes.size() / 4);  // lose ~25% of the data section
    fwrite(bytes.data(), 1, keep, out);
    fclose(out);
    return path;
}

int main() {
    const std::string path = write_fixture();
    check_roundtrip(path);

    const std::string trunc = write_truncated(path);
    bool threw = false;
    try {
        model_ctx m = cosyvoice_load_gguf(trunc);
        cosyvoice_free(m);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw, "loading a truncated GGUF must throw, not read past the mapping");

    remove(path.c_str());
    remove(trunc.c_str());

    if (g_failures) {
        fprintf(stderr, "test-cosyvoice-load: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("test-cosyvoice-load: all checks passed\n");
    return 0;
}
