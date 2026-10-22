// Pins conv_transpose_1d_columns (the Hexagon graph form of the S3Gen HiFT
// upsamplers and iSTFT) to ggml_conv_transpose_1d on the CPU, and, when a
// backend name is passed (e.g. "hexagon"), the same form on that device to the
// CPU reference.  Synthetic inputs; no fixtures.

#include "backend_selection.h"
#include "conv_columns.h"

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
constexpr double kDeviceTolerance = 5e-3;
constexpr size_t kArenaBytes      = 16 * 1024 * 1024;

struct transposed_conv_case {
    const char * name;
    int taps;
    int out_channels;
    int in_channels;
    int stride;
    int padding;
    int frames;
};

// The three HiFT upsamplers and the iSTFT, at short frame counts.
constexpr transposed_conv_case kCases[] = {
    { "hift ups 0", 16, 256, 512, 8, 4, 13 },
    { "hift ups 1", 11, 128, 256, 5, 3, 29 },
    { "hift ups 2",  7,  64, 128, 3, 2, 41 },
    { "istft",      16,   1,  18, 4, 0, 57 },
};

using graph_builder = std::function<ggml_tensor *(ggml_context *, ggml_tensor *, ggml_tensor *)>;

std::vector<float> make_values(size_t n, float phase, float scale) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = scale * std::sin(phase * (float) i + 0.3f) + 0.01f * (float) (i % 7);
    }
    return v;
}

std::vector<float> run(ggml_backend_t backend, const transposed_conv_case & c, const graph_builder & build) {
    ggml_init_params p = { kArenaBytes, nullptr, /*no_alloc=*/true };
    ggml_context * ctx    = ggml_init(p);
    ggml_cgraph *  gf     = ggml_new_graph(ctx);
    ggml_tensor *  kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.taps, c.out_channels, c.in_channels);
    ggml_tensor *  signal = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.frames, c.in_channels);
    ggml_set_input(kernel);
    ggml_set_input(signal);
    ggml_tensor * out = build(ctx, kernel, signal);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);
    const std::vector<float> kernel_data = make_values(ggml_nelements(kernel), 0.37f, 0.2f);
    const std::vector<float> signal_data = make_values(ggml_nelements(signal), 0.11f, 1.0f);
    ggml_backend_tensor_set(kernel, kernel_data.data(), 0, ggml_nbytes(kernel));
    ggml_backend_tensor_set(signal, signal_data.data(), 0, ggml_nbytes(signal));
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return res;
}

ggml_tensor * build_reference(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * signal,
                              const transposed_conv_case & c) {
    ggml_tensor * full = ggml_conv_transpose_1d(ctx, kernel, signal, c.stride, 0, 1);
    const int64_t kept = full->ne[0] - 2 * c.padding;
    ggml_tensor * cropped = ggml_view_2d(ctx, full, kept, full->ne[1], full->nb[1],
                                         (size_t) c.padding * full->nb[0]);
    return ggml_cont(ctx, cropped);
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
    for (size_t i = 0; i < ref.size(); ++i) {
        worst = std::max(worst, (double) std::fabs(got[i] - ref[i]));
    }
    const double scale = std::max(max_abs(ref), 1e-12);
    fprintf(stderr, "  %s %-10s max_abs_err=%.3e rel=%.3e\n", what, name, worst, worst / scale);
    if (!(worst <= tolerance * scale)) {
        fprintf(stderr, "FAIL: %s %s exceeds %.1e\n", what, name, tolerance);
        return false;
    }
    return true;
}

bool check_case(ggml_backend_t cpu, ggml_backend_t device, const transposed_conv_case & c) {
    const auto reference = run(cpu, c, [&](ggml_context * ctx, ggml_tensor * k, ggml_tensor * x) {
        return build_reference(ctx, k, x, c);
    });
    const graph_builder columns = [&](ggml_context * ctx, ggml_tensor * k, ggml_tensor * x) {
        return ::tts_cpp::detail::conv_transpose_1d_columns(ctx, k, x, c.stride, c.padding);
    };
    bool ok = matches(run(cpu, c, columns), reference, kCpuTolerance, "cpu", c.name);
    if (device) {
        ok = matches(run(device, c, columns), reference, kDeviceTolerance, "device", c.name) && ok;
    }
    return ok;
}

bool check_all_cases(ggml_backend_t cpu, ggml_backend_t device) {
    bool ok = true;
    for (const transposed_conv_case & c : kCases) {
        ok = check_case(cpu, device, c) && ok;
    }
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
        device = ::tts_cpp::detail::init_requested_backend(argv[1], /*verbose=*/true, "test-conv-columns");
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
