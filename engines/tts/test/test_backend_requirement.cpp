// Policy test for GpuBackendRequirement filtering in the backend-selection
// walk. Runs devicelessly: gpu_backend_satisfies_requirement() is the exact
// predicate init_gpu_backend() applies per registered device, so pinning it
// pins which devices each requirement may consider. The multi-backend case
// this guards: with Metal|OpenCL, a Vulkan device that sorts first must be
// skipped at the walk (leaving Metal selectable) instead of winning selection
// and being rejected afterwards, which would drop a Metal-capable host to CPU.

#include "audio8/backend_requirement.h"
#include "backend_selection.h"
#include "cosyvoice_pipeline.h"
#include "lavasr/backend_requirement.h"
#include "parler/backend_requirement.h"

#include <cstdio>
#include <string>

using tts_cpp::detail::GpuBackendRequirement;
using tts_cpp::detail::gpu_backend_satisfies_requirement;

namespace {

int g_failures = 0;

void expect(bool cond, const char * what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

void expect_accepts(GpuBackendRequirement set, const char * engine, const char * backend) {
    const std::string what = std::string(engine) + " accepts " + backend;
    expect(gpu_backend_satisfies_requirement(backend, set), what.c_str());
}

void expect_desktop_and_webgpu_set(GpuBackendRequirement set, const char * engine) {
    for (const char * backend : { "Vulkan", "MTL", "OpenCL", "CUDA", "WebGPU" }) {
        expect_accepts(set, engine, backend);
    }
    const std::string what = std::string(engine) + " rejects unnamed, so adding WebGPU did not widen it to Any";
    expect(!gpu_backend_satisfies_requirement(nullptr, set), what.c_str());
}

} // namespace

int main() {
    const auto any     = GpuBackendRequirement::Any;
    const auto vulkan  = GpuBackendRequirement::Vulkan;
    const auto opencl  = GpuBackendRequirement::OpenCL;
    const auto vkmtl   = GpuBackendRequirement::Vulkan | GpuBackendRequirement::Metal;
    const auto mtlcl   = GpuBackendRequirement::Metal  | GpuBackendRequirement::OpenCL;
    const auto vkmtlcl = GpuBackendRequirement::Vulkan | GpuBackendRequirement::Metal |
                         GpuBackendRequirement::OpenCL;

    expect(gpu_backend_satisfies_requirement("Metal", any), "Any accepts Metal");
    expect(gpu_backend_satisfies_requirement("Vulkan", any), "Any accepts Vulkan");
    expect(gpu_backend_satisfies_requirement("CUDA", any), "Any accepts CUDA");
    expect(gpu_backend_satisfies_requirement(nullptr, any), "Any accepts unnamed");

    expect(gpu_backend_satisfies_requirement("Vulkan", vulkan), "Vulkan accepts Vulkan");
    expect(!gpu_backend_satisfies_requirement("Metal", vulkan), "Vulkan rejects Metal");
    expect(!gpu_backend_satisfies_requirement(nullptr, vulkan), "Vulkan rejects unnamed");

    expect(gpu_backend_satisfies_requirement("OpenCL", opencl), "OpenCL accepts OpenCL");
    expect(!gpu_backend_satisfies_requirement("Vulkan", opencl), "OpenCL rejects Vulkan");
    expect(!gpu_backend_satisfies_requirement("MTL", opencl), "OpenCL rejects MTL");
    expect(!gpu_backend_satisfies_requirement(nullptr, opencl), "OpenCL rejects unnamed");

    expect(gpu_backend_satisfies_requirement("Vulkan", vkmtl), "Vulkan|Metal accepts Vulkan");
    expect(gpu_backend_satisfies_requirement("MTL", vkmtl), "Vulkan|Metal accepts MTL");
    expect(gpu_backend_satisfies_requirement("Metal", vkmtl), "Vulkan|Metal accepts Metal under its former name");
    expect(!gpu_backend_satisfies_requirement("OpenCL", vkmtl), "Vulkan|Metal rejects OpenCL");
    expect(!gpu_backend_satisfies_requirement("CUDA", vkmtl), "Vulkan|Metal rejects CUDA");
    expect(!gpu_backend_satisfies_requirement(nullptr, vkmtl), "Vulkan|Metal rejects unnamed");

    expect(gpu_backend_satisfies_requirement("Metal", mtlcl), "Metal|OpenCL accepts Metal");
    expect(gpu_backend_satisfies_requirement("MTL", mtlcl), "Metal|OpenCL accepts MTL");
    expect(gpu_backend_satisfies_requirement("OpenCL", mtlcl), "Metal|OpenCL accepts OpenCL");
    expect(!gpu_backend_satisfies_requirement("Vulkan", mtlcl), "Metal|OpenCL rejects Vulkan");
    expect(!gpu_backend_satisfies_requirement("CUDA", mtlcl), "Metal|OpenCL rejects CUDA");
    expect(!gpu_backend_satisfies_requirement(nullptr, mtlcl), "Metal|OpenCL rejects unnamed");

    // The three-backend set admits everything it names and still excludes CUDA,
    // so widening it for OpenCL did not silently turn into Any.
    expect(gpu_backend_satisfies_requirement("Vulkan", vkmtlcl), "Vulkan|Metal|OpenCL accepts Vulkan");
    expect(gpu_backend_satisfies_requirement("MTL", vkmtlcl), "Vulkan|Metal|OpenCL accepts MTL");
    expect(gpu_backend_satisfies_requirement("Metal", vkmtlcl), "Vulkan|Metal|OpenCL accepts Metal under its former name");
    expect(gpu_backend_satisfies_requirement("OpenCL", vkmtlcl), "Vulkan|Metal|OpenCL accepts OpenCL");
    expect(!gpu_backend_satisfies_requirement("CUDA", vkmtlcl), "Vulkan|Metal|OpenCL rejects CUDA");
    expect(!gpu_backend_satisfies_requirement(nullptr, vkmtlcl), "Vulkan|Metal|OpenCL rejects unnamed");

    // CUDA is a named bit rather than a backend reachable only through Any, so
    // an engine that has not validated it keeps excluding it (asserted above)
    // while one that has can name it without widening to every other backend.
    const auto cuda    = GpuBackendRequirement::CUDA;
    const auto vkcuda  = GpuBackendRequirement::Vulkan | GpuBackendRequirement::CUDA;
    // The four backends LavaSR names (lavasr::gpu_requirement()).
    const auto desktop = GpuBackendRequirement::Vulkan | GpuBackendRequirement::Metal |
                         GpuBackendRequirement::OpenCL | GpuBackendRequirement::CUDA;

    expect(gpu_backend_satisfies_requirement("CUDA", cuda), "CUDA accepts CUDA");
    expect(!gpu_backend_satisfies_requirement("Vulkan", cuda), "CUDA rejects Vulkan");
    expect(!gpu_backend_satisfies_requirement("MTL", cuda), "CUDA rejects MTL");
    expect(!gpu_backend_satisfies_requirement("OpenCL", cuda), "CUDA rejects OpenCL");
    expect(!gpu_backend_satisfies_requirement(nullptr, cuda), "CUDA rejects unnamed");

    expect(gpu_backend_satisfies_requirement("CUDA", vkcuda), "Vulkan|CUDA accepts CUDA");
    expect(gpu_backend_satisfies_requirement("Vulkan", vkcuda), "Vulkan|CUDA accepts Vulkan");
    expect(!gpu_backend_satisfies_requirement("MTL", vkcuda), "Vulkan|CUDA rejects MTL");
    expect(!gpu_backend_satisfies_requirement("OpenCL", vkcuda), "Vulkan|CUDA rejects OpenCL");

    expect(gpu_backend_satisfies_requirement("CUDA", desktop), "four-backend set accepts CUDA");
    expect(gpu_backend_satisfies_requirement("Vulkan", desktop), "four-backend set accepts Vulkan");
    expect(gpu_backend_satisfies_requirement("MTL", desktop), "four-backend set accepts MTL");
    expect(gpu_backend_satisfies_requirement("OpenCL", desktop), "four-backend set accepts OpenCL");
    expect(!gpu_backend_satisfies_requirement(nullptr, desktop),
           "four-backend set still rejects unnamed, so naming every backend is not Any");

    // WebGPU is its own bit: the four-backend desktop set keeps excluding it,
    // Any still admits it, and naming it does not admit anything else.
    const auto webgpu   = GpuBackendRequirement::WebGPU;
    const auto vkwebgpu = GpuBackendRequirement::Vulkan | GpuBackendRequirement::WebGPU;
    expect(gpu_backend_satisfies_requirement("WebGPU", any), "Any accepts WebGPU");
    expect(!gpu_backend_satisfies_requirement("WebGPU", desktop), "four-backend set rejects WebGPU");
    expect(gpu_backend_satisfies_requirement("WebGPU", webgpu), "WebGPU accepts WebGPU");
    expect(!gpu_backend_satisfies_requirement("Vulkan", webgpu), "WebGPU rejects Vulkan");
    expect(!gpu_backend_satisfies_requirement("CUDA", webgpu), "WebGPU rejects CUDA");
    expect(!gpu_backend_satisfies_requirement(nullptr, webgpu), "WebGPU rejects unnamed");
    expect(gpu_backend_satisfies_requirement("WebGPU", vkwebgpu), "Vulkan|WebGPU accepts WebGPU");
    expect(gpu_backend_satisfies_requirement("Vulkan", vkwebgpu), "Vulkan|WebGPU accepts Vulkan");
    expect(!gpu_backend_satisfies_requirement("MTL", vkwebgpu), "Vulkan|WebGPU rejects MTL");

    // LavaSR runs without per-op CPU fallback, so its set names the four
    // backends that implement its whole graph and keeps WebGPU out.
    const auto lavasr = tts_cpp::lavasr::gpu_requirement();
    expect(gpu_backend_satisfies_requirement("Vulkan", lavasr), "LavaSR accepts Vulkan");
    expect(gpu_backend_satisfies_requirement("MTL", lavasr), "LavaSR accepts MTL");
    expect(gpu_backend_satisfies_requirement("OpenCL", lavasr), "LavaSR accepts OpenCL");
    expect(gpu_backend_satisfies_requirement("CUDA", lavasr), "LavaSR accepts CUDA");
    expect(!gpu_backend_satisfies_requirement("WebGPU", lavasr), "LavaSR rejects WebGPU");

    // Parler, Audio8 and CosyVoice3 (desktop) admit WebGPU since ggml-webgpu
    // honours GGML_PREC_F32; LavaSR still keeps it out (asserted above).
    expect_desktop_and_webgpu_set(tts_cpp::parler::gpu_requirement(), "Parler");
    expect_desktop_and_webgpu_set(tts_cpp::audio8::gpu_requirement(), "Audio8");
#if defined(_WIN32) || (defined(__linux__) && !defined(__ANDROID__))
    expect_desktop_and_webgpu_set(cosyvoice_gpu_requirement(), "CosyVoice3");
#else
    expect(!gpu_backend_satisfies_requirement("WebGPU", cosyvoice_gpu_requirement()),
           "CosyVoice3 keeps WebGPU out with Vulkan off-desktop");
#endif

    if (g_failures) {
        fprintf(stderr, "%d requirement checks failed\n", g_failures);
        return 1;
    }
    fprintf(stderr, "PASS\n");
    return 0;
}
