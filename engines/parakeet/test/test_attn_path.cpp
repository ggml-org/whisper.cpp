// Fused attention is a build option that only CUDA, Metal and Vulkan take; every other
// backend, CPU and Hexagon included, keeps the unfused graph so its output does not
// depend on which GPU backend the binary carries. On Hexagon the unfused matmuls run
// at a 32-aligned head size, chosen from the model's own head size at load.
//   test-attn-path                                  policy checks on the live backends
//   test-attn-path <tdt.gguf> [--n-gpu-layers N] [--backend NAME]
//                                                   the built encoder graph agrees with the policy
#include "backend_util.h"
#include "parakeet_ctc.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

constexpr int kUnalignedHeadDim  = 80;   // TranscribePsy
constexpr int kAlignedHeadDim    = 128;  // Parakeet 0.6B
constexpr int kHmxHeadDim        = 96;
constexpr int kBackendUnavailable = 11;
constexpr int kSkipReturnCode     = 3;

int check(bool ok, const char * what) {
    std::printf("[attn-path] %s: %s\n", what, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

ggml_backend_t init_hexagon() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        if (reg && std::strcmp(reg, "HTP") == 0) return ggml_backend_dev_init(dev, nullptr);
    }
    return nullptr;
}

int check_hexagon_policy() {
    ggml_backend_t htp = init_hexagon();
    if (!htp) {
        std::printf("[attn-path] no Hexagon backend on this host, Hexagon cases skipped\n");
        return 0;
    }
    const int padded = parakeet::f16_activations_compiled() ? kHmxHeadDim : 0;
    int failures = check(!parakeet::flash_attn_allowed(true, htp), "Hexagon keeps the unfused graph");
    failures += check(parakeet::attention_matmul_head_dim(htp, kUnalignedHeadDim) == padded,
                      "Hexagon pads a head size HMX cannot take to a multiple of 32");
    failures += check(parakeet::attention_matmul_head_dim(htp, kAlignedHeadDim) == 0,
                      "Hexagon keeps a 32-aligned head size");
    ggml_backend_free(htp);
    return failures;
}

ggml_backend_t init_first_gpu() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        if (ggml_backend_t b = ggml_backend_dev_init(dev, nullptr)) return b;
    }
    return nullptr;
}

// The policy keys on registry names; a backend that renames itself would make the
// engine and this test agree on "unfused" silently, so every GPU name must be known.
bool known_gpu_backend_name(const char * name) {
    static const char * const known[] = {"CUDA", "MTL", "Metal", "Vulkan", "OpenCL", "WebGPU", "HIP", "SYCL", "HTP"};
    for (const char * k : known) {
        if (std::strcmp(name, k) == 0) return true;
    }
    return false;
}

int check_gpu_backend_names() {
    int failures = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        const char * name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        std::printf("[attn-path] GPU device %zu registry name %s\n", i, name ? name : "(null)");
        failures += check(name && known_gpu_backend_name(name), "GPU registry name is one the attention policy knows");
    }
    return failures;
}

int check_policy() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        std::printf("[attn-path] no CPU backend registered\n");
        return 1;
    }
    int failures = check_gpu_backend_names();
    failures += check(!parakeet::flash_attn_allowed(true, cpu),     "CPU keeps the unfused graph when flash-attn is compiled in");
    failures += check(!parakeet::flash_attn_allowed(false, cpu),    "CPU keeps the unfused graph when flash-attn is compiled out");
    failures += check(!parakeet::flash_attn_allowed(true, nullptr), "no backend means no fused graph");
    failures += check(parakeet::attention_matmul_head_dim(cpu, kUnalignedHeadDim) == 0,
                      "CPU runs the attention matmuls at the model's head size");
    ggml_backend_free(cpu);
    failures += check_hexagon_policy();

    if (ggml_backend_t gpu = init_first_gpu()) {
        const bool fused_backend = parakeet::backend_is_cuda(gpu) || parakeet::backend_is_metal(gpu) ||
                                   parakeet::backend_is_vulkan(gpu);
        std::printf("[attn-path] GPU backend %s, fused attention backend: %s\n",
                    parakeet::backend_reg_name(gpu), fused_backend ? "yes" : "no");
        failures += check(parakeet::flash_attn_allowed(true, gpu) == fused_backend, "GPU takes the fused graph only on CUDA, Metal and Vulkan");
        failures += check(!parakeet::flash_attn_allowed(false, gpu),               "GPU keeps the unfused graph when compiled out");
        ggml_backend_free(gpu);
    } else {
        std::printf("[attn-path] no GPU backend on this host, GPU cases skipped\n");
    }
    return failures;
}

// The loader probes the attention path once it knows the model's dimensions, so
// the head size it chose is the one the policy gives for this model's head size.
int check_model_head_dim(const parakeet::ParakeetCtcModel & model, ggml_backend_t active) {
    const int expected = parakeet::attention_matmul_head_dim(active, model.encoder_cfg.head_dim);
    const int chosen   = parakeet::model_attention_matmul_head_dim(model);
    std::printf("[attn-path] model head size %d, attention matmul head size %d (policy %d)\n",
                model.encoder_cfg.head_dim, chosen, expected);
    return check(chosen == expected, "loader picks the attention head size from the model's own head size");
}

int check_graph(const char * gguf, int n_gpu_layers, const std::string & requested) {
    using namespace parakeet;
    ParakeetCtcModel model;
    if (int rc = load_from_gguf(gguf, model, /*n_threads=*/0, n_gpu_layers, /*verbose=*/false, requested); rc != 0) {
        if (rc == kBackendUnavailable) {
            std::printf("[attn-path] SKIP: backend %s unavailable\n", requested.c_str());
            return -1;
        }
        std::fprintf(stderr, "[attn-path] load_from_gguf rc=%d\n", rc);
        return 1;
    }
    const std::string backend = model_encoder_backend_name(model);
    ggml_backend_t    active  = model_active_backend(model);
    const bool fused_backend  = active && (backend_is_cuda(active) || backend_is_metal(active) || backend_is_vulkan(active));
    const bool uses_fa        = encoder_graph_uses_op(model, GGML_OP_FLASH_ATTN_EXT);
    std::printf("[attn-path] encoder backend %s, flash-attn compiled %d, graph uses it %d\n",
                backend.c_str(), (int) flash_attn_compiled(), (int) uses_fa);
    int failures = check_model_head_dim(model, active);
    if (!fused_backend) {
        return failures + check(!uses_fa, "encoder graph on this backend has no fused attention node");
    }
    return failures + check(uses_fa == flash_attn_compiled(), "encoder graph on CUDA/Metal/Vulkan follows the build option");
}

} // namespace

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    int failures = 0;
    if (argc < 2) {
        failures = check_policy();
    } else {
        int n_gpu_layers = 0;
        std::string backend = "auto";
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::strcmp(argv[i], "--n-gpu-layers") == 0) n_gpu_layers = std::atoi(argv[i + 1]);
            if (std::strcmp(argv[i], "--backend") == 0) backend = argv[i + 1];
            if (std::strcmp(argv[i], "--backends-dir") == 0) parakeet::set_backends_directory(argv[i + 1]);
        }
        failures = check_graph(argv[1], n_gpu_layers, backend);
        if (failures < 0) return kSkipReturnCode;
    }
    std::printf("[attn-path] %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
