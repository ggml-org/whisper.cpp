// Unit test for src/gguf_stream.h.
//
// Verifies that gguf_stream_reader produces byte-identical tensor payloads
// to the legacy no_alloc=false staging path, without materialising the
// GGUF data section in host memory.  No model fixture needed: the test
// writes its own synthetic GGUF (including one tensor larger than the
// reader's CHUNK so the chunked to_backend loop is exercised across a
// chunk boundary) into a temp file, loads it through both paths, and
// compares.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "gguf_stream.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

// Tensor roster.  "big" must exceed gguf_stream_reader's 8 MiB CHUNK so
// to_backend takes more than one fread/tensor_set round trip, with a
// non-multiple-of-chunk tail.  Sizes in ELEMENTS.
struct spec { const char * name; ggml_type type; int64_t ne0; int64_t ne1; };
static const spec SPECS[] = {
    { "stream/small_f32",  GGML_TYPE_F32, 7,             1 },
    { "stream/mat_f32",    GGML_TYPE_F32, 129,          65 },
    { "stream/mat_f16",    GGML_TYPE_F16, 256,          33 },
    { "stream/big_f32",    GGML_TYPE_F32, 1031,       2503 },  // ~10.3 MB > 8 MiB CHUNK
};
static const size_t N_SPECS = sizeof(SPECS) / sizeof(SPECS[0]);

static std::string write_fixture() {
    const char * tmpdir = getenv("TMPDIR");
    std::string path = std::string(tmpdir ? tmpdir : "/tmp") + "/test-gguf-stream-fixture.gguf";

    size_t total = 0;
    for (size_t i = 0; i < N_SPECS; ++i) {
        total += ggml_row_size(SPECS[i].type, SPECS[i].ne0) * (size_t) SPECS[i].ne1;
    }
    ggml_init_params p = { total + (N_SPECS + 1) * ggml_tensor_overhead(), nullptr, false };
    ggml_context * ctx = ggml_init(p);

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "test-gguf-stream");
    gguf_set_val_u32(g, "test.n_tensors", (uint32_t) N_SPECS);

    for (size_t i = 0; i < N_SPECS; ++i) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, SPECS[i].type, SPECS[i].ne0, SPECS[i].ne1);
        ggml_set_name(t, SPECS[i].name);
        // Deterministic per-tensor byte pattern; good enough to catch
        // offset/ordering mistakes (every tensor's payload differs at
        // every position).
        uint8_t * d = (uint8_t *) t->data;
        const size_t nb = ggml_nbytes(t);
        for (size_t j = 0; j < nb; ++j) {
            d[j] = (uint8_t) ((j * 131 + i * 31 + 7) & 0xff);
        }
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

// Reference load: legacy staging path (no_alloc=false), payload memcpy'd
// out of the staged blob.
static std::map<std::string, std::vector<uint8_t>> load_reference(const std::string & path) {
    std::map<std::string, std::vector<uint8_t>> out;
    ggml_context * tmp = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    CHECK(g != nullptr, "reference: gguf_init_from_file failed");
    if (!g) return out;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(tmp, name);
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        memcpy(bytes.data(), ggml_get_data(t), bytes.size());
        out[name] = std::move(bytes);
    }
    gguf_free(g);
    ggml_free(tmp);
    return out;
}

int main() {
    const std::string path = write_fixture();
    const auto ref = load_reference(path);
    CHECK(ref.size() == N_SPECS, "reference loaded %zu tensors, want %zu", ref.size(), N_SPECS);

    // ---- streaming load: metadata only, then to_backend per tensor ----
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ &meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    CHECK(g != nullptr, "stream: gguf_init_from_file(no_alloc=true) failed");
    if (!g) return 1;

    ggml_backend_t backend = ggml_backend_cpu_init();

    const int64_t n_tensors = gguf_get_n_tensors(g);
    ggml_init_params wp = { ggml_tensor_overhead() * (size_t) n_tensors, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    std::map<std::string, ggml_tensor *> tensors;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * src = ggml_get_tensor(meta, name);
        CHECK(src != nullptr && src->data == nullptr,
              "no_alloc metadata tensor '%s' should have NULL data", name);
        ggml_tensor * dst = ggml_dup_tensor(ctx_w, src);
        ggml_set_name(dst, name);
        tensors[name] = dst;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    CHECK(buf != nullptr, "backend alloc failed");

    {
        tts_cpp::detail::gguf_stream_reader reader(g, path);
        CHECK(reader.ok(), "reader failed to open %s", path.c_str());
        for (auto & kv : tensors) {
            CHECK(reader.to_backend(kv.first.c_str(), kv.second),
                  "to_backend('%s') failed", kv.first.c_str());
        }

        // Byte-exact parity with the staging path.
        for (auto & kv : tensors) {
            const auto it = ref.find(kv.first);
            CHECK(it != ref.end(), "tensor '%s' missing from reference", kv.first.c_str());
            if (it == ref.end()) continue;
            std::vector<uint8_t> got(ggml_nbytes(kv.second));
            ggml_backend_tensor_get(kv.second, got.data(), 0, got.size());
            CHECK(got.size() == it->second.size(), "tensor '%s' size mismatch", kv.first.c_str());
            CHECK(memcmp(got.data(), it->second.data(), got.size()) == 0,
                  "tensor '%s' bytes differ between streaming and staging load", kv.first.c_str());
        }

        // to_host parity (the voice_encoder / campplus / s3tokenizer path).
        for (size_t i = 0; i < N_SPECS; ++i) {
            ggml_tensor * t = ggml_get_tensor(meta, SPECS[i].name);
            std::vector<uint8_t> got(ggml_nbytes(t));
            CHECK(reader.to_host(SPECS[i].name, got.data(), got.size()),
                  "to_host('%s') failed", SPECS[i].name);
            const auto & want = ref.at(SPECS[i].name);
            CHECK(memcmp(got.data(), want.data(), got.size()) == 0,
                  "to_host('%s') bytes differ from reference", SPECS[i].name);
        }

        // Failure modes: unknown name, size mismatch.  Both must fail
        // loudly instead of corrupting the destination.
        {
            std::vector<uint8_t> sink(16);
            CHECK(!reader.to_host("stream/does_not_exist", sink.data(), sink.size()),
                  "to_host on a missing tensor must fail");
            ggml_tensor * t = ggml_get_tensor(meta, "stream/small_f32");
            CHECK(!reader.to_host("stream/small_f32", sink.data(), ggml_nbytes(t) + 1),
                  "to_host with a wrong destination size must fail");
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    gguf_free(g);
    ggml_free(meta);
    remove(path.c_str());

    if (g_failures) {
        fprintf(stderr, "test-gguf-stream: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("test-gguf-stream: all checks passed\n");
    return 0;
}
