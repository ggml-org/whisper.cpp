// Fused attention is a build option that only CUDA, Metal and Vulkan take; every other
// backend, CPU included, keeps the unfused graph so its output does not depend
// on which GPU backend the binary carries.
//   test-attn-path                                  policy checks on the live backends
//   test-attn-path <tdt.gguf> [--n-gpu-layers N]    the built encoder graph agrees with the policy
#include "backend_util.h"
#include "parakeet_ctc.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int check(bool ok, const char * what) {
    std::printf("[attn-path] %s: %s\n", what, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
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
    ggml_backend_free(cpu);

    if (ggml_backend_t gpu = init_first_gpu()) {
        const bool fused_backend = parakeet::backend_is_cuda(gpu) || parakeet::backend_is_metal(gpu) ||
                                   parakeet::backend_is_vulkan(gpu) || parakeet::backend_is_hexagon(gpu);
        std::printf("[attn-path] GPU backend %s, fused attention backend: %s\n",
                    parakeet::backend_reg_name(gpu), fused_backend ? "yes" : "no");
        failures += check(parakeet::flash_attn_allowed(true, gpu) == fused_backend, "GPU takes the fused graph only on CUDA, Metal, Vulkan and Hexagon");
        failures += check(!parakeet::flash_attn_allowed(false, gpu),               "GPU keeps the unfused graph when compiled out");
        ggml_backend_free(gpu);
    } else {
        std::printf("[attn-path] no GPU backend on this host, GPU cases skipped\n");
    }
    return failures;
}

int check_graph(const char * gguf, int n_gpu_layers) {
    using namespace parakeet;
    ParakeetCtcModel model;
    if (int rc = load_from_gguf(gguf, model, /*n_threads=*/0, n_gpu_layers, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "[attn-path] load_from_gguf rc=%d\n", rc);
        return 1;
    }
    const std::string backend = model_encoder_backend_name(model);
    ggml_backend_t    active  = model_active_backend(model);
    const bool fused_backend  = active && (backend_is_cuda(active) || backend_is_metal(active) || backend_is_vulkan(active) || backend_is_hexagon(active));
    const bool uses_fa        = encoder_graph_uses_op(model, GGML_OP_FLASH_ATTN_EXT);
    std::printf("[attn-path] encoder backend %s, flash-attn compiled %d, graph uses it %d\n",
                backend.c_str(), (int) flash_attn_compiled(), (int) uses_fa);
    if (n_gpu_layers <= 0 || !fused_backend) {
        return check(!uses_fa, "encoder graph on this backend has no fused attention node");
    }
    return check(uses_fa == flash_attn_compiled(), "encoder graph on CUDA/Metal/Vulkan follows the build option");
}

} // namespace

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    int failures = 0;
    if (argc < 2) {
        failures = check_policy();
    } else {
        int n_gpu_layers = 0;
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::strcmp(argv[i], "--n-gpu-layers") == 0) n_gpu_layers = std::atoi(argv[i + 1]);
            if (std::strcmp(argv[i], "--backends-dir") == 0) parakeet::set_backends_directory(argv[i + 1]);
        }
        failures = check_graph(argv[1], n_gpu_layers);
    }
    std::printf("[attn-path] %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
