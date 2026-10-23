// A quantized matmul whose result exceeds the F16 range must come back finite
// and correct with the headroom the engine applies on that device: Hexagon's
// HMX writes matmul results through F16, so 2560 terms of 0.5 x 60 (76800)
// would read back as inf there. On the CPU the headroom is 1 and the guarded
// graph is the plain matmul itself.
//   test-matmul-headroom [--device NAME]   (default CPU; e.g. HTP0)
// Exit 0 on success, 3 when the device is not present, 1 on failure.
#include "backend_util.h"
#include "packed_affine.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int   kInputs        = 2560;  // TranscribePsy's feed-forward width
constexpr int   kOutputs       = 64;
constexpr int   kFrames        = 96;
constexpr float kWeight        = 0.5f;
constexpr float kActivation    = 60.0f;
constexpr float kF16Max        = 65504.0f;
constexpr float kMaxRelError   = 1e-2f;
constexpr int   kSkipReturnCode = 3;

struct Device {
    ggml_backend_t backend = nullptr;
    std::string    name    = "CPU";
};

int parse_device(int argc, char ** argv, std::string & name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            name = argv[++i];
            continue;
        }
        std::fprintf(stderr, "usage: %s [--device NAME]\n", argv[0]);
        return 2;
    }
    return 0;
}

std::vector<uint8_t> quantized_weight() {
    const std::vector<float> dense((size_t) kInputs * kOutputs, kWeight);
    std::vector<uint8_t> blocks(ggml_row_size(GGML_TYPE_Q8_0, kInputs) * kOutputs);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, dense.data(), blocks.data(), 0, kOutputs, kInputs, nullptr);
    return blocks;
}

struct Weight {
    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *         tensor = nullptr;
    ~Weight() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

bool upload_weight(ggml_backend_t backend, Weight & w) {
    ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
    w.ctx = ggml_init(ip);
    if (!w.ctx) return false;
    w.tensor = ggml_new_tensor_2d(w.ctx, GGML_TYPE_Q8_0, kInputs, kOutputs);
    w.buffer = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, parakeet::weight_buffer_type(backend));
    if (!w.buffer) return false;
    const std::vector<uint8_t> blocks = quantized_weight();
    ggml_backend_tensor_set(w.tensor, blocks.data(), 0, blocks.size());
    return true;
}

std::vector<float> read(ggml_tensor * t) {
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

int count_off_target(const std::vector<float> & v, float expected) {
    int bad = 0;
    for (float x : v) {
        if (!std::isfinite(x) || std::fabs(x - expected) > kMaxRelError * expected) ++bad;
    }
    return bad;
}

int count_bit_differences(const std::vector<float> & a, const std::vector<float> & b) {
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++bad;
    }
    return bad;
}

int check(bool ok, const char * what) {
    std::printf("[matmul-headroom] %s: %s\n", what, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

int run(const Device & dev) {
    Weight w;
    if (!upload_weight(dev.backend, w)) return check(false, "weight uploads to the device's weight buffer");

    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kInputs, kFrames);
    ggml_set_input(x);
    const float headroom = parakeet::matmul_output_headroom(dev.backend);
    ggml_tensor * plain   = ggml_mul_mat(ctx, w.tensor, x);
    ggml_tensor * guarded = parakeet::affine_mul_mat_with_headroom(ctx, w.tensor, x, headroom);
    ggml_set_output(plain);
    ggml_set_output(guarded);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, plain);
    ggml_build_forward_expand(gf, guarded);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dev.backend));
    int failures = check(ga && ggml_gallocr_alloc_graph(ga, gf), "graph allocates");
    if (failures == 0) {
        const std::vector<float> input((size_t) kInputs * kFrames, kActivation);
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        failures += check(ggml_backend_graph_compute(dev.backend, gf) == GGML_STATUS_SUCCESS, "graph computes");
    }
    if (failures == 0) {
        const float expected = kInputs * kWeight * kActivation;
        const std::vector<float> p = read(plain);
        const std::vector<float> g = read(guarded);
        std::printf("[matmul-headroom] device %s, headroom %.0f, expected %.0f (F16 max %.0f), "
                    "plain matmul off target in %d of %zu\n",
                    dev.name.c_str(), headroom, expected, kF16Max, count_off_target(p, expected), p.size());
        failures += check(count_off_target(g, expected) == 0, "guarded matmul is finite and on target");
        if (headroom == 1.0f) {
            failures += check(guarded->op == GGML_OP_MUL_MAT && guarded->src[1] == x, "headroom 1 adds no nodes");
            failures += check(count_bit_differences(p, g) == 0, "headroom 1 is bit-identical");
        }
        failures += check(parakeet::backend_is_hexagon(dev.backend) == (headroom > 1.0f),
                          "only Hexagon takes a headroom");
    }
    if (ga) ggml_gallocr_free(ga);
    ggml_free(ctx);
    return failures;
}

} // namespace

int main(int argc, char ** argv) {
    Device dev;
    if (int rc = parse_device(argc, argv, dev.name); rc != 0) return rc;
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name(dev.name.c_str());
    if (!device) {
        std::printf("[matmul-headroom] SKIP: no device %s\n", dev.name.c_str());
        return kSkipReturnCode;
    }
    dev.backend = ggml_backend_dev_init(device, nullptr);
    if (!dev.backend) {
        std::printf("[matmul-headroom] SKIP: device %s does not initialise\n", dev.name.c_str());
        return kSkipReturnCode;
    }
    const int failures = run(dev);
    ggml_backend_free(dev.backend);
    std::printf("[matmul-headroom] %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
