// Pins unfused_attention (T3 attention on backends that decline
// FLASH_ATTN_EXT, e.g. ggml-opencl on Adreno) to ggml_flash_attn_ext on the
// CPU.  K and V are strided views of a token-major cache, as T3 reads them.
// With a backend name (e.g. "opencl"), attention_for_backend also runs on that
// device against the CPU's flash attention.  Synthetic inputs; no fixtures.

#include "attention_fallback.h"
#include "backend_selection.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

namespace {

constexpr double kCpuTolerance    = 1e-5;
constexpr double kDeviceTolerance = 2e-3;
constexpr size_t kArenaBytes      = 64 * 1024 * 1024;

struct attention_case {
    const char * name;
    int  head_dim;
    int  heads;
    int  kv_heads;
    int  queries;
    int  keys;
    int  batch;
    bool causal;
};

// Turbo prompt and decode step, the multilingual CFG pair, and grouped heads.
constexpr attention_case kCases[] = {
    { "prompt",  64, 16, 16, 12, 12, 1, true  },
    { "step",    64, 16, 16,  1, 40, 1, false },
    { "batch-2", 64, 16, 16, 10, 10, 2, true  },
    { "grouped", 64,  8,  2,  5, 21, 1, true  },
};

struct attention_inputs {
    ggml_tensor * queries = nullptr;
    ggml_tensor * key_cache = nullptr;
    ggml_tensor * value_cache = nullptr;
    ggml_tensor * mask = nullptr;
};

using attention_builder = std::function<ggml_tensor *(ggml_context *, ggml_tensor *, ggml_tensor *,
                                                      ggml_tensor *, ggml_tensor *, float)>;

std::vector<float> make_values(size_t n, float phase, float scale) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = scale * std::sin(phase * (float) i + 0.3f);
    return v;
}

std::vector<ggml_fp16_t> make_causal_mask(const attention_case & c) {
    const int past = c.keys - c.queries;
    std::vector<ggml_fp16_t> mask((size_t) c.keys * c.queries);
    for (int i = 0; i < c.queries; ++i) {
        for (int j = 0; j < c.keys; ++j) {
            mask[(size_t) i * c.keys + j] = ggml_fp32_to_fp16(j <= past + i ? 0.0f : -INFINITY);
        }
    }
    return mask;
}

attention_inputs declare_inputs(ggml_context * ctx, const attention_case & c) {
    attention_inputs in;
    in.queries     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.head_dim, c.queries, c.heads, c.batch);
    in.key_cache   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, (int64_t) c.head_dim * c.kv_heads, c.keys, c.batch);
    in.value_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, (int64_t) c.head_dim * c.kv_heads, c.keys, c.batch);
    ggml_set_input(in.queries);
    ggml_set_input(in.key_cache);
    ggml_set_input(in.value_cache);
    if (c.causal) {
        in.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.keys, c.queries);
        ggml_set_input(in.mask);
    }
    return in;
}

ggml_tensor * cache_heads(ggml_context * ctx, ggml_tensor * cache, const attention_case & c) {
    const size_t row = cache->nb[1];
    return ggml_view_4d(ctx, cache, c.head_dim, c.keys, c.kv_heads, c.batch,
                        row, (size_t) c.head_dim * sizeof(float), cache->nb[2], 0);
}

void upload_inputs(const attention_inputs & in, const attention_case & c) {
    const auto queries = make_values(ggml_nelements(in.queries), 0.13f, 1.0f);
    const auto keys    = make_values(ggml_nelements(in.key_cache), 0.29f, 1.0f);
    const auto values  = make_values(ggml_nelements(in.value_cache), 0.07f, 0.5f);
    ggml_backend_tensor_set(in.queries, queries.data(), 0, ggml_nbytes(in.queries));
    ggml_backend_tensor_set(in.key_cache, keys.data(), 0, ggml_nbytes(in.key_cache));
    ggml_backend_tensor_set(in.value_cache, values.data(), 0, ggml_nbytes(in.value_cache));
    if (in.mask) {
        const auto mask = make_causal_mask(c);
        ggml_backend_tensor_set(in.mask, mask.data(), 0, ggml_nbytes(in.mask));
    }
}

std::vector<float> run(ggml_backend_t backend, const attention_case & c, const attention_builder & build) {
    ggml_init_params p = { kArenaBytes, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph *  gf  = ggml_new_graph(ctx);
    const attention_inputs in = declare_inputs(ctx, c);
    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    ggml_tensor * out = build(ctx, in.queries, cache_heads(ctx, in.key_cache, c),
                              cache_heads(ctx, in.value_cache, c), in.mask, scale);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);
    upload_inputs(in, c);
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return res;
}

double max_abs(const std::vector<float> & v) {
    double m = 0.0;
    for (float x : v) m = std::max(m, (double) std::fabs(x));
    return m;
}

bool matches(const std::vector<float> & got, const std::vector<float> & ref, double tolerance,
             const char * what, const char * name) {
    if (got.size() != ref.size()) {
        fprintf(stderr, "FAIL: %s %s size %zu vs %zu\n", what, name, got.size(), ref.size());
        return false;
    }
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, (double) std::fabs(got[i] - ref[i]));
    const double scale = std::max(max_abs(ref), 1e-12);
    fprintf(stderr, "  %s %-8s max_abs_err=%.3e rel=%.3e\n", what, name, worst, worst / scale);
    if (!(worst <= tolerance * scale)) {
        fprintf(stderr, "FAIL: %s %s exceeds %.1e\n", what, name, tolerance);
        return false;
    }
    return true;
}

ggml_tensor * build_fused(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                          ggml_tensor * mask, float scale) {
    return ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
}

bool check_case(ggml_backend_t cpu, ggml_backend_t device, const attention_case & c) {
    const auto reference = run(cpu, c, build_fused);
    bool ok = matches(run(cpu, c, ::tts_cpp::detail::unfused_attention), reference, kCpuTolerance, "cpu", c.name);
    if (device) {
        const attention_builder on_device = [device](ggml_context * ctx, ggml_tensor * q, ggml_tensor * k,
                                                     ggml_tensor * v, ggml_tensor * mask, float scale) {
            return ::tts_cpp::detail::attention_for_backend(ctx, device, q, k, v, mask, scale);
        };
        ok = matches(run(device, c, on_device), reference, kDeviceTolerance, "device", c.name) && ok;
    }
    return ok;
}

bool check_all_cases(ggml_backend_t cpu, ggml_backend_t device) {
    bool ok = true;
    for (const attention_case & c : kCases) ok = check_case(cpu, device, c) && ok;
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    ggml_backend_t cpu = ::tts_cpp::detail::init_cpu_backend();
    if (!cpu) {
        fprintf(stderr, "FAIL: no CPU backend\n");
        return 1;
    }
    ggml_backend_t device = nullptr;
    if (argc > 1) {
        device = ::tts_cpp::detail::init_requested_backend(argv[1], /*verbose=*/true, "test-attention-fallback");
        if (!device) {
            fprintf(stderr, "FAIL: backend '%s' is not available\n", argv[1]);
            ggml_backend_free(cpu);
            return 1;
        }
    }
    const bool ok = check_all_cases(cpu, device);
    if (device) ggml_backend_free(device);
    ggml_backend_free(cpu);
    fprintf(stderr, ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
