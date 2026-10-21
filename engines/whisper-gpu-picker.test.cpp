// Model-free unit tests for qvac::speech::pick_whisper_gpu_device_from —
// the pure tier-policy function backing engines/whisper-gpu-picker.h.
//
// whisper.cpp's own whisper_backend_init_gpu picks the Nth GPU-or-IGPU
// device (N = params.gpu_device, default 0) in registry order, which loses
// on:
//   1. NVIDIA hosts where Vulkan enumerates before CUDA (the same card is
//      exposed through both; CUDA is vendor-native and faster).
//   2. Hybrid dGPU + iGPU hosts where the iGPU shows up first in ggml's
//      registry order.
// The picker maps a device topology to the whisper-side index the tier
// policy prefers. This test exercises that mapping against synthesised
// topologies so the tier order is guarded independently of any host GPU.
//
// Exit 0 on success; non-zero with FAIL lines otherwise.

#include "whisper-gpu-picker.h"

#include <cstdio>
#include <initializer_list>

using qvac::speech::PickerDevice;
using qvac::speech::pick_whisper_gpu_device_from;

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

// Helper: whisper.cpp counts GPU-or-IGPU devices in enumeration order, so
// the returned index is the position of the chosen device in the input array.
int pick(std::initializer_list<PickerDevice> devs) {
    return pick_whisper_gpu_device_from(devs.begin(), devs.size());
}

void test_cuda_over_vulkan_same_nvidia_card() {
    // Same NVIDIA card exposed through both Vulkan and CUDA — ggml
    // usually enumerates Vulkan first, so whisper's own default
    // (gpu_device=0) would pick the Vulkan adapter. The picker prefers
    // CUDA at index 1.
    CHECK(pick({
        {"Vulkan", false, -1},  // index 0 — dGPU Vulkan on NVIDIA
        {"CUDA",   false, -1},  // index 1 — CUDA on the same card
    }) == 1);
    // Order-independent: CUDA first in the enumeration also wins.
    CHECK(pick({
        {"CUDA",   false, -1},
        {"Vulkan", false, -1},
    }) == 0);
}

void test_discrete_over_integrated() {
    // Vulkan on Intel iGPU + Vulkan on NVIDIA dGPU — the picker prefers
    // the discrete adapter over the integrated one, matching llm-llamacpp
    // / diffusion / tts behavior.
    CHECK(pick({
        {"Vulkan", true,  -1},   // index 0 — Intel iGPU
        {"Vulkan", false, -1},   // index 1 — NVIDIA dGPU (or AMD)
    }) == 1);
    CHECK(pick({
        {"Vulkan", false, -1},   // dGPU first — same result
        {"Vulkan", true,  -1},
    }) == 0);
}

void test_cuda_wins_full_hybrid_topology() {
    // Full realistic topology on an NVIDIA + Intel-iGPU laptop: Vulkan on
    // the iGPU (enum 0), Vulkan on the dGPU (enum 1), CUDA on the dGPU
    // (enum 2). Whisper's default picks the iGPU; the picker picks CUDA.
    CHECK(pick({
        {"Vulkan", true,  -1},
        {"Vulkan", false, -1},
        {"CUDA",   false, -1},
    }) == 2);
}

void test_cuda_integrated_still_over_other_discrete() {
    // Tegra / Jetson expose CUDA as IGPU on some drivers; with a Vulkan
    // dGPU also present the picker still prefers CUDA (vendor-native path).
    CHECK(pick({
        {"Vulkan", false, -1},   // some external Vulkan GPU
        {"CUDA",   true,  -1},   // Tegra CUDA reported as IGPU
    }) == 1);
}

void test_adreno_700plus_opencl_wins_top_tier() {
    // Snapdragon 8 Gen 2/3/4: same Adreno GPU exposed as both OpenCL (IGPU)
    // and Vulkan (IGPU). Whisper's own default (gpu_device=0) picks whichever
    // ggml enumerates first — typically Vulkan. The picker prefers the
    // OpenCL adapter because ggml-opencl kernels on Adreno 7xx+ are
    // validated and faster than the same card's Vulkan path. Matches the
    // Adreno-first tier used by parakeet / audiogen / tts.
    CHECK(pick({
        {"Vulkan", true, -1},    // index 0 — Adreno 750 via Vulkan
        {"OpenCL", true, 750},   // index 1 — Adreno 750 via OpenCL (WIN)
    }) == 1);
    CHECK(pick({
        {"OpenCL", true, 830},   // Adreno 830 via OpenCL first — still wins
        {"Vulkan", true, -1},
    }) == 0);
    // Snapdragon-X naming: parse_adreno_version maps "Adreno X1-85" to 800,
    // so the OpenCL adapter still takes the top tier.
    CHECK(pick({
        {"Vulkan", true, -1},
        {"OpenCL", true, 800},
    }) == 1);
}

void test_adreno_6xx_opencl_not_promoted() {
    // Older Adreno 6xx OpenCL is known-broken (miscomputes on some kernels),
    // so it does NOT get the top-tier bump and is routed to the last-resort
    // OpenCL bucket. A Vulkan sibling on the same host must win regardless of
    // enumeration order — before the last-resort bucket existed, 6xx OpenCL
    // enumerated first grabbed the integrated tier and beat Vulkan.
    CHECK(pick({
        {"Vulkan", true, -1},    // index 0 — Adreno 660 Vulkan
        {"OpenCL", true, 660},   // index 1 — Adreno 660 OpenCL (last resort)
    }) == 0);
    CHECK(pick({
        {"OpenCL", true, 660},   // index 0 — Adreno 660 OpenCL first
        {"Vulkan", true, -1},    // index 1 — Vulkan still wins
    }) == 1);
    CHECK(pick({
        {"OpenCL", true, 660},   // OpenCL-only — the last-resort bucket is
    }) == 0);                    // still selectable, just deprioritised.
}

void test_desktop_opencl_deprioritized_below_vulkan() {
    // Non-Adreno desktop OpenCL (adreno_version == -1) is treated like
    // 6xx: routed to the last-resort tier, so a Vulkan/Metal dGPU on the same
    // host wins. Matches the `opencl_other` bucket in parakeet / audiogen /
    // tts so the four speech engines reach the same verdict on the same host.
    CHECK(pick({
        {"OpenCL", false, -1},   // index 0 — desktop OpenCL first
        {"Vulkan", false, -1},   // index 1 — Vulkan wins
    }) == 1);
    CHECK(pick({
        {"Vulkan", false, -1},
        {"OpenCL", false, -1},
    }) == 0);
    // Desktop OpenCL is still selectable when it is the only GPU present.
    CHECK(pick({
        {"OpenCL", false, -1},
    }) == 0);
}

void test_webgpu_ranks_with_vulkan() {
    // ggml-webgpu (Dawn) exposes one discrete adapter named "WebGPU". It sits
    // in the generic discrete tier: CUDA on the same NVIDIA card still wins,
    // an integrated adapter loses to it, and alone it is index 0.
    CHECK(pick({
        {"WebGPU", false, -1},
        {"CUDA",   false, -1},
    }) == 1);
    CHECK(pick({
        {"Vulkan", true,  -1},
        {"WebGPU", false, -1},
    }) == 1);
    CHECK(pick({
        {"WebGPU", false, -1},
    }) == 0);
}

void test_only_integrated_available() {
    // Only iGPU visible — the picker returns the first (only) one, which
    // is what whisper's own default already does. This is the no-op case
    // that must stay a no-op so we don't regress iGPU-only hosts.
    CHECK(pick({
        {"Vulkan", true, -1},
    }) == 0);
}

void test_only_discrete_available() {
    // Only a single dGPU — index 0.
    CHECK(pick({
        {"Vulkan", false, -1},
    }) == 0);
    CHECK(pick({
        {"Metal", false, -1},
    }) == 0);
}

void test_empty_device_list() {
    // No GPUs — the picker returns 0 so whisper_backend_init_gpu falls
    // through its own "no GPU found" path and hands back nullptr; the
    // caller then falls back to CPU as before.
    CHECK(pick_whisper_gpu_device_from(nullptr, 0) == 0);
}

void test_null_reg_name_treated_as_non_cuda() {
    // A device with a null registry name (test-only quirk / broken driver)
    // is still selectable via the discrete-first rule, but never wins the
    // CUDA-first pass.
    CHECK(pick({
        {nullptr,  false, -1},   // discrete, non-CUDA — becomes the fallback pick
        {"CUDA",   false, -1},   // and CUDA still wins over it
    }) == 1);
    CHECK(pick({
        {nullptr,  false, -1},
    }) == 0);
}

}  // namespace

int main() {
    test_cuda_over_vulkan_same_nvidia_card();
    test_discrete_over_integrated();
    test_cuda_wins_full_hybrid_topology();
    test_cuda_integrated_still_over_other_discrete();
    test_adreno_700plus_opencl_wins_top_tier();
    test_adreno_6xx_opencl_not_promoted();
    test_desktop_opencl_deprioritized_below_vulkan();
    test_webgpu_ranks_with_vulkan();
    test_only_integrated_available();
    test_only_discrete_available();
    test_empty_device_list();
    test_null_reg_name_treated_as_non_cuda();

    std::fprintf(stderr, "[test-whisper-gpu-picker] %d/%d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
