// Model-free unit tests for parakeet's GPU-tier classification
// (parakeet::gpu_tier_for + the GpuTier enum ordering).
//
// parakeet's init_gpu_backend must prefer CUDA over Vulkan on NVIDIA, and
// prefer discrete GPUs over integrated ones, matching the existing
// llm-llamacpp / diffusion behavior. The tier ranking is a pure function so
// a synthesised device topology can be scored without touching the live
// ggml-backend registry (which needs real drivers to populate).
//
// SCOPE: these tests cover the classifier (what tier a device lands in) and
// the enum-constant ordering (which tier outranks which). They do NOT
// exercise init_gpu_backend's actual bucket walk in parakeet_ctc.cpp: that
// function has its own parallel `try_init(...)` sequence and does not call
// gpu_tier_for today, so a reordering of its try_init calls would silently
// pass these tests. Keeping the two in sync is a code-review contract; the
// deeper fix (drive init_gpu_backend off gpu_tier_for as the single source
// of truth) is tracked separately.
//
// Exit 0 on success; non-zero with FAIL lines otherwise.

#include "backend_util.h"

#include "ggml-backend.h"

#include <cstdio>
#include <string>

using parakeet::GpuTier;
using parakeet::gpu_tier_for;

namespace {

int g_checks   = 0;
int g_failures = 0;

void expect(bool cond, const char * what) {
    ++g_checks;
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

#define CHECK(cond) expect((cond), #cond)

void test_gpu_tier_policy() {
    // Adreno 700+ OpenCL wins the walk — Snapdragon 8 Gen 2+ ships validated
    // OpenCL kernels that outperform Vulkan.
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_GPU,  740) == GpuTier::AdrenoOpenCL700Plus);
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_IGPU, 830) == GpuTier::AdrenoOpenCL700Plus);
    // Adreno 6xx / non-Adreno OpenCL lands in the last-resort tier — its
    // Adreno-6xx-broken skip lives inside init_gpu_backend (enumeration
    // side, not ranking side).
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_GPU,  640) == GpuTier::OpenCLOther);
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_GPU,  -1)  == GpuTier::OpenCLOther);

    // CUDA outranks Vulkan on the same NVIDIA card.
    CHECK(gpu_tier_for("CUDA",   GGML_BACKEND_DEVICE_TYPE_GPU,  -1)  == GpuTier::CudaDiscrete);
    CHECK(gpu_tier_for("Vulkan", GGML_BACKEND_DEVICE_TYPE_GPU,  -1)  == GpuTier::OtherDiscrete);
    CHECK(static_cast<int>(GpuTier::CudaDiscrete) <
          static_cast<int>(GpuTier::OtherDiscrete));

    // Tegra / Jetson report CUDA as IGPU on some drivers — CUDA-first still
    // wins over a discrete Vulkan adapter.
    CHECK(gpu_tier_for("CUDA", GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::CudaIntegrated);
    CHECK(static_cast<int>(GpuTier::CudaIntegrated) <
          static_cast<int>(GpuTier::OtherDiscrete));

    // Discrete outranks integrated at every tier that has both — matches the
    // llm-llamacpp / diffusion dGPU-first preference.
    CHECK(static_cast<int>(GpuTier::CudaDiscrete)  < static_cast<int>(GpuTier::CudaIntegrated));
    CHECK(static_cast<int>(GpuTier::OtherDiscrete) < static_cast<int>(GpuTier::OtherIntegrated));

    // Metal is on the same tier as Vulkan (both non-OpenCL, non-CUDA GPUs).
    CHECK(gpu_tier_for("Metal", GGML_BACKEND_DEVICE_TYPE_GPU,  -1) == GpuTier::OtherDiscrete);
    CHECK(gpu_tier_for("MTL",   GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::OtherIntegrated);

    // Non-GPU / non-IGPU devices are not selectable — CPU, ACCEL, unknown
    // types must never be picked into a GPU bucket.
    CHECK(gpu_tier_for("Vulkan", GGML_BACKEND_DEVICE_TYPE_CPU,   -1) == GpuTier::NotSelectable);
    CHECK(gpu_tier_for("BLAS",   GGML_BACKEND_DEVICE_TYPE_ACCEL, -1) == GpuTier::NotSelectable);
    CHECK(gpu_tier_for(nullptr,  GGML_BACKEND_DEVICE_TYPE_CPU,   -1) == GpuTier::NotSelectable);

    // A null / empty registry name on a GPU lands in the OtherDiscrete/Integrated
    // bucket (not OpenCL, not CUDA) — same fallback used for un-recognised drivers.
    CHECK(gpu_tier_for(nullptr, GGML_BACKEND_DEVICE_TYPE_GPU,  -1) == GpuTier::OtherDiscrete);
    CHECK(gpu_tier_for("",      GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::OtherIntegrated);
}

void test_explicit_selection() {
    using parakeet::backend_selection_is_auto;
    using parakeet::backend_selection_matches;
    CHECK(backend_selection_is_auto("auto"));
    CHECK(backend_selection_is_auto(""));
    CHECK(!backend_selection_is_auto("hexagon"));
    // Both devices advertise GPU. An explicit NPU request must still select
    // HTP0 when OpenCL is registered first, regardless of normal GPU tiering.
    CHECK(!backend_selection_matches("hexagon", "OpenCL", "OpenCL", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_selection_matches("hexagon", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_selection_matches("hexagon", "HTP", "HTP1", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_selection_matches("HTP1", "HTP", "HTP1", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_selection_matches("HTP0", "HTP", "HTP1", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_selection_matches("opencl", "OpenCL", "OpenCL0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_selection_matches("opencl", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_selection_matches("vulkan", "Vulkan", "Vulkan0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_selection_matches("vulkan", "Vulkan", "Vulkan1", GGML_BACKEND_DEVICE_TYPE_IGPU));
    CHECK(!backend_selection_matches("vulkan", "CUDA", "CUDA0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_selection_matches("vulkan", "Metal", "Metal", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_selection_matches("Vulkan1", "Vulkan", "Vulkan1", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_selection_matches("Vulkan1", "Vulkan", "Vulkan0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_selection_matches("cuda", "CUDA", "CUDA0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_selection_matches("cuda", "CUDA", "CUDA1", GGML_BACKEND_DEVICE_TYPE_IGPU));
    CHECK(!backend_selection_matches("cuda", "Vulkan", "Vulkan0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_selection_matches("CUDA1", "CUDA", "CUDA1", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_selection_matches("cpu", "CPU", "CPU", GGML_BACKEND_DEVICE_TYPE_CPU));
    CHECK(!backend_selection_matches("cpu", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_selection_matches("auto", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_selection_matches("typo", "HTP", "HTP0", GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(!backend_selection_matches("hexagon", nullptr, nullptr, GGML_BACKEND_DEVICE_TYPE_GPU));

    using parakeet::prepend_dsp_library_directory;
    CHECK(prepend_dsp_library_directory("/app/lib", "") == "/app/lib");
    CHECK(prepend_dsp_library_directory("/app/lib", "/custom;/vendor/dsp") == "/app/lib;/custom;/vendor/dsp");
    CHECK(prepend_dsp_library_directory("/app/lib", "/custom;/app/lib") == "/custom;/app/lib");
    CHECK(prepend_dsp_library_directory("/app/lib", "/app/library") == "/app/lib;/app/library");
    CHECK(prepend_dsp_library_directory("", "/custom") == "/custom");
}

void test_adreno_descriptions() {
    using parakeet::description_names_adreno;
    CHECK(description_names_adreno("QUALCOMM Adreno(TM) 830 (OpenCL 3.0 Adreno(TM) 830)"));
    CHECK(description_names_adreno("Adreno (TM) 830"));
    CHECK(description_names_adreno("adreno 740"));
    CHECK(!description_names_adreno("NVIDIA GeForce RTX 5090"));
    CHECK(!description_names_adreno("Mali-G715"));
    CHECK(!description_names_adreno(""));
    CHECK(!description_names_adreno(nullptr));
}

}  // namespace

int main() {
    test_gpu_tier_policy();
    test_explicit_selection();
    test_adreno_descriptions();

    std::fprintf(stderr, "[test-parakeet-backend-selection] %d/%d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
