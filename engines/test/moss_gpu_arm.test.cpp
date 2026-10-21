// Model-free unit test for moss_gpu_arm: the accelerator arms the MOSS GPU
// suites accept on their command line, and the ggml instance names each one
// recognises as "selected".

#include "moss_gpu_arm.h"

#include <cstdio>
#include <stdexcept>

namespace {

int g_failures = 0;

void expect(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

void expect_arm(const char * request, const char * prefix, const char * instance) {
    const MossGpuArm arm = moss_gpu_arm(request);
    expect(arm.request == request, "arm keeps its request name");
    expect(arm.prefix == prefix, "arm maps to the registry prefix");
    expect(arm.selected(instance), "arm recognises its own instance");
    expect(!arm.selected("CPU"), "arm rejects the CPU instance");
}

bool unknown_arm_throws(const char * request) {
    try {
        moss_gpu_arm(request);
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    expect_arm("cuda", "CUDA", "CUDA0");
    expect_arm("vulkan", "Vulkan", "Vulkan0");
    expect_arm("hexagon", "HTP", "HTP0");
    expect_arm("opencl", "OpenCL", "OpenCL");
    expect_arm("webgpu", "WebGPU", "WebGPU: WebGPU");
    expect(unknown_arm_throws("metal"), "an arm nobody configured is rejected");
    expect(unknown_arm_throws(""), "an empty arm is rejected");
    if (g_failures) {
        std::fprintf(stderr, "%d moss arm checks failed\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
