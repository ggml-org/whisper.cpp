#pragma once

// Registry-only GPU backend selection for tts-cpp.
//
// Replaces the three legacy `init_backend` / `s3gen_init_backend` /
// `init_supertonic_backend` `#ifdef GGML_USE_<X>` ladders that called
// `ggml_backend_{cuda,metal,vk,opencl}_init` directly. Under the
// dynamic-loader build mode embedded host applications ship with
// (`GGML_BACKEND_DL=ON`) those static entry points live in separate
// `.so` files that are dlopen()'d at runtime and are not linkable
// from libtts-cpp; the ggml-backend registry walk reaches the same
// backends in both `GGML_BACKEND_DL=ON` and `=OFF` modes, mirroring
// parakeet-cpp's design.
//
// Selection follows the same Adreno tier policy as parakeet-cpp's
// `init_gpu_backend` and the qvac llm-llamacpp `BackendSelection.cpp::
// chooseBackend`: Adreno 700+ devices take the OpenCL branch
// (validated, faster than Vulkan on Snapdragon 8 Gen 2/3/4 and on the
// Snapdragon X Elite parts that report as `Adreno X<n>`); every other
// GPU (Vulkan on all non-Adreno Android, Metal on Apple, CUDA on
// Linux/Windows desktop, Mali iGPU via Vulkan, ...) goes through the
// non-OpenCL preference. Adreno 6xx OpenCL is force-skipped (known
// broken kernels) unless the caller opts in via
// `TTS_CPP_ALLOW_ADRENO_6XX=1`.

#include "ggml-backend.h"

#include <string>

namespace tts_cpp::detail {

// First-Engine-wins override for the directory `ggml_backend_load_all*()`
// scans on the first `ensure_backends_loaded()` call. Call before
// constructing the first Engine; later calls log a one-shot warn and
// are ignored (the ggml-backend registry is a process-wide singleton).
void set_backends_directory(const std::string & dir);

// First-Engine-wins override for `$GGML_OPENCL_CACHE_DIR`. Honoured
// only on `__ANDROID__` builds; ignored elsewhere (desktop OpenCL
// platforms don't ship the program-binary-cache patch that reads this
// env var). Call before constructing the first Engine.
void set_opencl_cache_dir(const std::string & dir);

// Idempotent process-wide load of every registered ggml backend.
// Routed through a function-static guard so callers can invoke it
// from every init helper without paying the directory walk cost
// more than once.
void ensure_backends_loaded();

// Pick a GPU backend using the Adreno tier policy described above.
// Returns nullptr when no GPU was requested (`n_gpu_layers <= 0`),
// when no GPU device is registered, or when every candidate device
// refused `ggml_backend_dev_init`. `log_prefix` controls the
// per-call log line tag (e.g. "s3gen", "supertonic", "chatterbox")
// so the existing user-visible logs in the three init sites stay
// distinguishable; verbose=false suppresses everything except hard
// errors.
//
// `vulkan_device` selects which Vulkan adapter to prefer when more
// than one is visible in the registry (QVAC-18605 round 3 / 12):
//   - 0 (default): first Vulkan adapter in registry order.
//   - N > 0      : the Nth Vulkan adapter (0-indexed); throws on out
//                  of range so a CLI typo fails loud instead of
//                  silently falling through to CPU.
//   - -1         : auto-pick: argmax(free VRAM), with a UMA bias
//                  that excludes integrated-GPU adapters whenever at
//                  least one discrete adapter is also visible (avoids
//                  the iGPU's UMA-reported system RAM dwarfing the
//                  discrete's true VRAM and silently stealing the
//                  pick on hybrid desktops/laptops).
// No effect when zero / one Vulkan adapters are visible, or when the
// chosen backend is non-Vulkan (CUDA / Metal / OpenCL).
// `out_gpu_present_but_unused` (optional): set true only when CPU fallback is by
// policy (GPU present but off-allowlist, e.g. Mali / Adreno 6xx); NOT on init failure.
// `allow_arm_mali` (default false): opt-in to ARM Mali/Immortalis (Valhall) Vulkan.
// Only Supertonic sets it — its st_mul_mat output-pad dodges the Valhall mul_mat bug.
ggml_backend_t init_gpu_backend(int n_gpu_layers,
                                bool verbose,
                                const char * log_prefix,
                                int vulkan_device = 0,
                                bool allow_arm_mali = false,
                                bool * out_gpu_present_but_unused = nullptr);

// Convenience wrapper that picks up the registered CPU device and
// returns its init handle. Mirrors parakeet-cpp's
// `init_cpu_backend()`. Never throws; returns nullptr when the
// ggml-cpu backend isn't available (no .so on disk and not
// statically linked).
ggml_backend_t init_cpu_backend();

// Returns the first registered BLAS accel backend (if any) or
// nullptr. Mirrors parakeet-cpp's `init_blas_backend()`. Today no
// tts-cpp call site uses this but it is exposed for parity with
// the parakeet helper API so callers that want to mirror parakeet's
// (cpu + blas accel + gpu) cascade can.
ggml_backend_t init_blas_backend();

// Adreno-generation parser. Returns:
//   - a 3-or-4-digit generation number ("Adreno (TM) 750" -> 750,
//     "Adreno 830" -> 830, "Adreno 660" -> 660)
//   - a synthetic 800 for the "Adreno X<n>" naming used by
//     Snapdragon X Elite parts (X1-85 / X1-45 etc.)
//   - -1 when no Adreno marker is present (Mali, desktop GPUs, ...)
//
// Exposed for the tier-policy implementation; safe to call on
// nullptr / empty strings.
int parse_adreno_version(const char * s);

bool is_adreno_6xx(const char * s);
bool is_adreno_700plus(const char * s);

// Vendor check (name OR description, ASCII case-insensitive): true for a
// Qualcomm Adreno GPU. Unlike parse_adreno_version it does not require a
// model number, so it also matches the bare OpenCL "QUALCOMM Adreno(TM)"
// string. Used to gate Android GPU selection to the only validated vendor.
bool is_qualcomm_adreno(const char * name, const char * desc);

} // namespace tts_cpp::detail
