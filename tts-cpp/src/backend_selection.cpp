#include "backend_selection.h"
#include "backend_util.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts_cpp::detail {
namespace {

// Backends-dir / OpenCL-cache-dir override + warning state. The
// setters are intended to be called by the first Engine
// construction; both are consumed once and then frozen for the rest
// of the process lifetime (the ggml-backend registry and
// $GGML_OPENCL_CACHE_DIR are both process-singleton state).
//
// `g_backends_loaded` is the canonical "registry already populated"
// flag, set inside `ensure_backends_loaded()` *before* the load-all
// call returns AND under the mutex so concurrent `set_*` calls
// either land their write (and have it picked up by the in-flight
// load) or atomically observe the flag and warn. We track it
// separately from `g_recorded_backends_dir` because the first
// Engine may have legitimately constructed with an empty
// `backends_dir` (default ggml search path), in which case
// `g_recorded_backends_dir` stays empty and is no longer a reliable
// "have we loaded?" sentinel -- a subsequent setter would otherwise
// silently write to `g_backends_dir`, never get re-scanned, and
// surface zero diagnostic to the caller.
//
// Mirrors parakeet-cpp/src/parakeet_ctc.cpp 1:1 (same Engine ctor /
// process-singleton-registry interaction). Kept in a tts-cpp-local
// anon namespace so the two libraries can be vendored side-by-side
// without ODR collisions on the static state.
std::mutex     g_backends_dir_mutex;
std::string    g_backends_dir;
std::string    g_recorded_backends_dir;
std::string    g_recorded_opencl_cache_dir;
std::atomic<bool> g_backends_loaded{false};
std::atomic<bool> g_backends_dir_warned{false};
std::atomic<bool> g_opencl_cache_dir_warned{false};

const char * dev_reg_name(ggml_backend_dev_t dev) {
    if (!dev) return "";
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg ? ggml_backend_reg_name(reg) : "";
}

// QVAC-18605 — Vulkan multi-adapter pick. Pure logic on the two
// per-device vectors so the policy stays unit-testable (a richer
// copy lives in `tts_cpp::supertonic::detail::resolve_vulkan_device_index`
// with its own DocTest harness; this in-process copy is kept lean so
// the shared GPU-init helper doesn't introduce a back-edge into the
// supertonic translation unit).
//
//   requested == -1 → auto-pick: argmax(free_vram), but if any
//                     discrete adapter exists, restrict the argmax
//                     to the discrete subset (excludes UMA iGPUs
//                     reporting system RAM as free VRAM).
//   requested ==  0 → first adapter in registry order.
//   requested >   0 → that adapter index (0-based against the
//                     Vulkan-only subset).
//   requested <  -1 → reserved; throws.
// Out-of-range positive index throws too. Vectors must be the same
// length; mismatched non-empty UMA list throws.
int pick_vulkan_device_index(int requested,
                             const std::vector<size_t> & free_vram_per_device,
                             const std::vector<bool> &   is_uma_per_device) {
    const int dev_count = (int) free_vram_per_device.size();
    if (dev_count <= 0) {
        throw std::runtime_error(
            "tts-cpp: cannot resolve --vulkan-device against an empty "
            "device list (no Vulkan adapter visible)");
    }
    if (!is_uma_per_device.empty() &&
        is_uma_per_device.size() != free_vram_per_device.size()) {
        throw std::runtime_error("tts-cpp: is_uma_per_device length mismatch");
    }
    if (requested < -1) {
        throw std::runtime_error(
            "tts-cpp: --vulkan-device " + std::to_string(requested) +
            " is reserved (only -1 means auto-pick)");
    }
    if (requested == -1) {
        bool any_discrete = false;
        if (!is_uma_per_device.empty()) {
            for (bool u : is_uma_per_device) {
                if (!u) { any_discrete = true; break; }
            }
        }
        int    best_idx  = 0;
        size_t best_vram = 0;
        bool   first     = true;
        for (int i = 0; i < dev_count; ++i) {
            if (any_discrete && is_uma_per_device[(size_t) i]) continue;
            if (first || free_vram_per_device[(size_t) i] > best_vram) {
                best_idx  = i;
                best_vram = free_vram_per_device[(size_t) i];
                first     = false;
            }
        }
        return best_idx;
    }
    if (requested >= dev_count) {
        throw std::runtime_error(
            "tts-cpp: --vulkan-device " + std::to_string(requested) +
            " out of range (visible Vulkan adapters: " +
            std::to_string(dev_count) + ")");
    }
    return requested;
}

} // namespace

void set_backends_directory(const std::string & dir) {
    std::lock_guard<std::mutex> lock(g_backends_dir_mutex);
    if (g_backends_loaded.load(std::memory_order_acquire)) {
        // Registry already populated for this process. We can't
        // re-scan a different directory mid-flight (ggml's registry
        // is a process-wide singleton), so log the conflict at most
        // once and otherwise stay silent on subsequent identical
        // sets (the common case when a host instantiates several
        // Engines back-to-back from the same backends folder, or
        // when the second value happens to match the recorded one).
        if (dir != g_recorded_backends_dir &&
            !g_backends_dir_warned.exchange(true)) {
            if (g_recorded_backends_dir.empty()) {
                // First Engine constructed without an explicit
                // backends_dir, so ggml's compile-time default
                // search path was used. The current caller wanted
                // a specific dir but missed the window.
                fprintf(stderr,
                    "tts-cpp: set_backends_directory('%s') ignored -- the "
                    "ggml-backend registry was already populated against "
                    "ggml's default search path (no explicit backends_dir on "
                    "the first Engine). Call set_backends_directory() (or "
                    "construct an Engine with backends_dir set) before the "
                    "first Engine to influence which directory is scanned.\n",
                    dir.c_str());
            } else {
                fprintf(stderr,
                    "tts-cpp: set_backends_directory('%s') ignored -- backends "
                    "already loaded from '%s' earlier in this process.\n",
                    dir.c_str(), g_recorded_backends_dir.c_str());
            }
        }
        return;
    }
    g_backends_dir = dir;
}

void set_opencl_cache_dir(const std::string & dir) {
#if defined(__ANDROID__)
    // Same "first Engine wins" contract as set_backends_directory:
    // ggml-opencl reads $GGML_OPENCL_CACHE_DIR once per process at
    // backend init (before the first kernel build), so a setenv
    // after init is effectively a no-op on the cache binding. Gate
    // on the shared g_backends_loaded flag because the OpenCL
    // backend is registered at the same `ggml_backend_load_all*`
    // call that flips the flag -- conservative because it might
    // still take effect when the host hasn't yet instantiated a
    // GPU device, but matches what the engine-ctor documentation
    // promises and avoids the same silent-failure mode as
    // set_backends_directory's previous gate.
    std::lock_guard<std::mutex> lock(g_backends_dir_mutex);
    if (g_backends_loaded.load(std::memory_order_acquire)) {
        if (!dir.empty() && dir != g_recorded_opencl_cache_dir &&
            !g_opencl_cache_dir_warned.exchange(true)) {
            if (g_recorded_opencl_cache_dir.empty()) {
                fprintf(stderr,
                    "tts-cpp: set_opencl_cache_dir('%s') ignored -- backends "
                    "were already loaded with no explicit OpenCL cache dir "
                    "earlier in this process ($GGML_OPENCL_CACHE_DIR either "
                    "unset or set by another consumer). Call "
                    "set_opencl_cache_dir() before the first Engine to take "
                    "effect.\n",
                    dir.c_str());
            } else {
                fprintf(stderr,
                    "tts-cpp: set_opencl_cache_dir('%s') ignored -- "
                    "$GGML_OPENCL_CACHE_DIR already pinned to '%s' earlier in "
                    "this process.\n",
                    dir.c_str(), g_recorded_opencl_cache_dir.c_str());
            }
        }
        return;
    }
    if (dir.empty()) return;
    // ggml-opencl's program-binary-cache patch reads this once per
    // process at backend init (before the first kernel build). Set
    // it before constructing the first Engine; later calls don't
    // re-bind the cache but cost nothing.
    ::setenv("GGML_OPENCL_CACHE_DIR", dir.c_str(), /*overwrite=*/1);
    g_recorded_opencl_cache_dir = dir;
#else
    (void) dir;
#endif
}

// Trigger one-time discovery + load of every available ggml backend.
// Idempotent: repeated calls inside the same process are no-ops once
// the registry is populated. Routed through a static guard so we
// don't pay the directory-walk cost on every model load.
//
// Why this instead of the per-backend ggml_backend_<x>_init() entry
// points the cascade used to call directly: with GGML_BACKEND_DL=ON
// (the dynamic-loader mode embedded host applications typically
// ship with) the CUDA / Metal / Vulkan / OpenCL / BLAS / ggml-cpu
// backends live in separate shared libraries that are dlopened at
// runtime; their concrete init symbols are not linkable from
// libtts-cpp, and the only supported entry point is the registry.
// With GGML_BACKEND_DL=OFF the backends are statically linked into
// libggml, registered at constructor time, and
// ggml_backend_load_all() is a cheap no-op. Both modes therefore
// reach the same registry walk below, matching the convention used
// by llama.cpp / parakeet-cpp / other ggml-based libraries.
//
// The optional backends dir comes from `set_backends_directory()`
// (typically wired from `EngineOptions::backends_dir`). When set and
// non-empty, the loader walks that single directory instead of the
// compile-time defaults so embedded host apps can ship the
// `lib<prefix>ggml-{vulkan,opencl,cpu-*}.so` files in their own
// per-module folder rather than relying on `LD_LIBRARY_PATH` /
// `dlopen()` heuristics.
void ensure_backends_loaded() {
    static const bool loaded = []() {
        std::string dir;
        {
            std::lock_guard<std::mutex> lock(g_backends_dir_mutex);
            dir = g_backends_dir;
            g_recorded_backends_dir = g_backends_dir;
            // Flip the loaded sentinel under the mutex (and *before*
            // we release it for the load-all call below) so any
            // concurrent setter that's about to acquire the mutex
            // sees the registry as already-claimed and falls into
            // its warn-once branch. Without this, a setter racing
            // a first Engine construction would land its value
            // *after* we already captured `dir` into the local --
            // the registry would scan against the wrong directory
            // (or the default), and the second Engine would have
            // no idea its override was lost.
            g_backends_loaded.store(true, std::memory_order_release);
        }
        if (!dir.empty()) {
            ggml_backend_load_all_from_path(dir.c_str());
        } else {
            ggml_backend_load_all();
        }
        return true;
    }();
    (void) loaded;
}

// Parse the Adreno generation number from a device name /
// description string. Returns:
//   - a 3-or-4-digit generation number ("Adreno (TM) 750" -> 750,
//     "Adreno 830" -> 830, "Adreno 660" -> 660)
//   - a synthetic 800 for the "Adreno X<n>" naming used by
//     Snapdragon X Elite parts (X1-85 / X1-45 etc.). These are
//     7xx/8xx-tier silicon with kernels that ggml-opencl supports
//     and outperform Vulkan on. Mapped to 800 here so they take
//     the OpenCL branch in the tier policy.
//   - -1 when no Adreno marker is present (Mali, desktop GPUs, ...)
//
// Used to drive the OpenCL vs Vulkan tier policy below: Adreno
// 7xx/8xx/X<n> ship OpenCL kernels that outperform Vulkan on those
// parts, while Adreno 6xx ggml-opencl is known broken (incorrect
// results). Mirrors parakeet-cpp's `parse_adreno_version` and the
// equivalent helper in llm-llamacpp's
// BackendSelection.cpp::parseAdrenoVersion so the three stacks
// reach the same decision on the same hardware.
int parse_adreno_version(const char * s) {
    if (!s) return -1;
    std::string lowered(s);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    // After an "adreno" marker (skipping "(tm)", spaces, punctuation), the model
    // is a 3-4 digit generation ("740"/"830") or the Snapdragon-X "x<n>" token
    // ("x1-85" -> 800-tier). Scan every marker and keep the highest; requiring
    // 3-4 digits skips the "opencl 3.0" noise in a combined OpenCL description
    // like "QUALCOMM Adreno(TM) (OpenCL 3.0 Adreno(TM) 740)" -> 740, not 3.
    static const std::regex re(R"(dreno\D*?(\d{3,4}|x\d))", std::regex::optimize);
    int best = -1;
    for (std::sregex_iterator it(lowered.begin(), lowered.end(), re), end; it != end; ++it) {
        const std::string tok = (*it)[1].str();
        const int v = (tok[0] == 'x') ? 800 : std::stoi(tok);
        if (v > best) best = v;
    }
    return best;
}

bool is_adreno_6xx(const char * s) {
    const int v = parse_adreno_version(s);
    return v >= 600 && v < 700;
}

bool is_adreno_700plus(const char * s) {
    const int v = parse_adreno_version(s);
    return v >= 700;
}

// True if the device name/description identifies a Qualcomm Adreno GPU.
// Unlike parse_adreno_version (which needs a 3-digit model number and so
// returns -1 for the bare OpenCL "QUALCOMM Adreno(TM)" string), this is a
// vendor check used to gate Android GPU selection. ASCII case-insensitive
// because the strings vary in capitalisation: ggml-opencl reports
// CL_DEVICE_NAME ("QUALCOMM Adreno(TM)") and ggml-vulkan reports the Vulkan
// deviceName ("Adreno (TM) 740").
bool is_qualcomm_adreno(const char * name, const char * desc) {
    return str_contains_ci(name, "adreno")   || str_contains_ci(desc, "adreno") ||
           str_contains_ci(name, "qualcomm") || str_contains_ci(desc, "qualcomm");
}

// Samsung Xclipse (AMD RDNA2) detector. Validated Android GPU vendor: the graphs
// compute correctly on its Vulkan driver (ggml-opencl is not loaded for non-Adreno).
bool is_samsung_xclipse(const char * name, const char * desc) {
    return str_contains_ci(name, "xclipse") || str_contains_ci(desc, "xclipse");
}

// ARM Mali / Immortalis (Valhall) detector. Shares the backend_util.h matcher with
// the st_mul_mat pad gate so the allowlist and that gate use ONE predicate.
bool is_arm_mali(const char * name, const char * desc) {
    return desc_or_name_is_arm_mali(name, desc);
}

// Pick a GPU backend using the same tier policy as parakeet-cpp's
// `init_gpu_backend` / llm-llamacpp's BackendSelection: ggml-opencl
// is only used when an Adreno 700+ device is present (where its
// kernels are validated and faster than Vulkan); every other GPU
// (Vulkan, Metal, CUDA, Intel iGPU, ...) goes through the non-OpenCL
// preference. Adreno 6xx OpenCL is known broken (incorrect outputs)
// and is force-skipped unless the caller opts in via
// `TTS_CPP_ALLOW_ADRENO_6XX=1`.
//
// On Android the device walk is additionally gated to Qualcomm Adreno
// only: other Android GPU vendors are not validated and at least one
// (ARM Mali / Tensor) aborts the host process from inside graph
// compute, so they are skipped and the engine falls back to CPU.
// Desktop GPU vendors are unaffected.
//
// Routed exclusively through the ggml-backend registry
// (`ggml_backend_load_all` + `ggml_backend_dev_*`). No direct calls
// to `ggml_backend_vulkan_init` / `ggml_backend_opencl_init` /
// `ggml_backend_metal_init` are made anywhere in tts-cpp -- under
// the GGML_BACKEND_DL=ON build mode embedded host applications ship
// with, those entry points live in separate shared libraries that
// are dlopen()'d at runtime and are not linkable from libtts-cpp.
// The registry walk reaches the same backends in both modes.
ggml_backend_t init_gpu_backend(int n_gpu_layers,
                                bool verbose,
                                const char * log_prefix,
                                int vulkan_device,
                                bool allow_arm_mali,
                                bool * out_gpu_present_but_unused) {
    if (out_gpu_present_but_unused) *out_gpu_present_but_unused = false;
    if (n_gpu_layers <= 0) return nullptr;
    if (!log_prefix) log_prefix = "tts-cpp";

    ensure_backends_loaded();

    struct Cand {
        ggml_backend_dev_t dev;
        const char *       name;
        const char *       desc;
        const char *       reg_name;
    };
    std::vector<Cand> opencl_adreno_700plus;
    std::vector<Cand> other_gpu;    // Vulkan / Metal / CUDA / Mali / Intel / ...
    std::vector<Cand> opencl_other; // Non-Adreno OpenCL (e.g. desktop)
    int max_adreno_version = -1;

    // QVAC-18605 — track every visible Vulkan adapter so we can apply
    // the round-12 device-selection policy (vulkan_device index +
    // free-VRAM auto-pick with UMA bias) before draining the bucket.
    std::vector<Cand>   vulkan_devs;
    std::vector<size_t> vulkan_free_vram;
    std::vector<bool>   vulkan_is_uma;

    // Set when a GPU was present but declined BY POLICY (non-allowlisted Android vendor,
    // or Adreno 6xx) so the caller accepts CPU fallback; NOT set when a validated GPU's init failed.
    bool gpu_present_but_unvalidated = false;

    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        const char * name     = ggml_backend_dev_name(dev);
        const char * desc     = ggml_backend_dev_description(dev);
        const char * reg_name = dev_reg_name(dev);
        const bool   is_opencl = reg_name && std::strcmp(reg_name, "OpenCL") == 0;
        const bool   is_vulkan = reg_name && std::strcmp(reg_name, "Vulkan") == 0;

        if (is_vulkan) {
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            vulkan_devs.push_back({dev, name, desc, reg_name});
            vulkan_free_vram.push_back(free);
            vulkan_is_uma.push_back(type == GGML_BACKEND_DEVICE_TYPE_IGPU);
            if (verbose && vulkan_device == -1) {
                fprintf(stderr,
                        "%s: vulkan device %d: %s — free %.0f MB / total %.0f MB%s\n",
                        log_prefix,
                        (int) (vulkan_devs.size() - 1),
                        desc && *desc ? desc : (name && *name ? name : "unknown"),
                        (double) free  / (1024.0 * 1024.0),
                        (double) total / (1024.0 * 1024.0),
                        type == GGML_BACKEND_DEVICE_TYPE_IGPU
                            ? " [UMA — biased against on hybrid machines]" : "");
            }
        }

#if defined(__ANDROID__)
        // Android allowlist: Adreno + Xclipse always; Mali only with allow_arm_mali (Supertonic).
        // Others fall through to CPU: unvalidated drivers can miscompute or ggml_abort() (uncatchable).
        if (!is_qualcomm_adreno(name, desc) && !is_samsung_xclipse(name, desc) &&
            !(is_arm_mali(name, desc) && allow_arm_mali)) {
            gpu_present_but_unvalidated = true;
            if (verbose) {
                fprintf(stderr,
                    "%s: Android GPU '%s' (%s) is not a validated vendor for this "
                    "engine (Qualcomm Adreno / Samsung Xclipse; ARM Mali is "
                    "Supertonic-only); skipping (falling through to CPU)\n",
                    log_prefix,
                    name ? name : "?",
                    desc ? desc : "?");
            }
            continue;
        }
#endif

        const int adreno_v = std::max(parse_adreno_version(name),
                                      parse_adreno_version(desc));
        if (adreno_v > max_adreno_version) max_adreno_version = adreno_v;

        if (is_opencl) {
            if (adreno_v >= 700) {
                opencl_adreno_700plus.push_back({dev, name, desc, reg_name});
            } else if (adreno_v >= 600 && adreno_v < 700) {
                const char * reported = name ? name : (desc ? desc : "unknown");
                const char * override_env = std::getenv("TTS_CPP_ALLOW_ADRENO_6XX");
                if (!override_env || override_env[0] != '1') {
                    gpu_present_but_unvalidated = true;
                    if (verbose) {
                        fprintf(stderr,
                            "%s: OpenCL device '%s' is Adreno 6xx; "
                            "skipping (7xx/8xx/X1E supported, set "
                            "TTS_CPP_ALLOW_ADRENO_6XX=1 to override)\n",
                            log_prefix, reported);
                    }
                    continue;
                }
                if (verbose) {
                    fprintf(stderr,
                        "%s: TTS_CPP_ALLOW_ADRENO_6XX=1 set; "
                        "keeping OpenCL backend on '%s' anyway\n",
                        log_prefix, reported);
                }
                opencl_other.push_back({dev, name, desc, reg_name});
            } else {
                opencl_other.push_back({dev, name, desc, reg_name});
            }
        } else {
            other_gpu.push_back({dev, name, desc, reg_name});
        }
    }

    // Tier policy:
    //   1. Adreno 700+: prefer OpenCL (validated, faster than Vulkan
    //      on Snapdragon 8 Gen 2/3/4 etc.).
    //   2. Anything else with a non-OpenCL GPU: prefer that
    //      (Adreno Vulkan on Android — non-Adreno is filtered out
    //      above; Metal on Apple; CUDA / Vulkan on Linux/Windows
    //      desktop).
    //   3. Last resort: any other OpenCL device (e.g. desktop OpenCL,
    //      or Adreno OpenCL whose version string lacked a model number).
    auto try_init = [&](const std::vector<Cand> & bucket) -> ggml_backend_t {
        for (const Cand & c : bucket) {
            ggml_backend_t b = ggml_backend_dev_init(c.dev, nullptr);
            if (!b) continue;
            if (verbose) {
                fprintf(stderr,
                    "%s: using %s backend (%s)\n",
                    log_prefix,
                    c.reg_name && *c.reg_name ? c.reg_name : "GPU",
                    c.name ? c.name : (c.desc ? c.desc : "unknown"));
            }
            return b;
        }
        return nullptr;
    };

    // QVAC-18605 — when a `vulkan_device` override or auto-pick is
    // requested AND at least one Vulkan adapter is visible, resolve
    // the chosen Vulkan adapter and move it to the front of
    // `other_gpu` so `try_init` picks it first.
    //
    //   `vulkan_device == 0` (default): tier policy unchanged.
    //   `vulkan_device == -1`         : auto-pick across Vulkan
    //                                   adapters; tier policy
    //                                   unchanged (user asked for
    //                                   "best Vulkan device", not
    //                                   "must be Vulkan over OpenCL").
    //   `vulkan_device  >  0`         : explicit override.  User
    //                                   asked for Vulkan device N
    //                                   specifically, so honour it
    //                                   by trying `other_gpu`
    //                                   BEFORE `opencl_adreno_700plus`
    //                                   below — otherwise on a
    //                                   Snapdragon device that
    //                                   exposes both backends, the
    //                                   OpenCL-Adreno tier would
    //                                   silently shadow the override.
    //
    // PR #31 review comment 3355973146: guard on `!vulkan_devs.empty()`
    // so a `vulkan_device != 0` config doesn't abort `init_gpu_backend`
    // on a no-Vulkan machine (Metal-only Mac, CUDA-only Linux,
    // Adreno-OpenCL-only Snapdragon) — without the guard,
    // `pick_vulkan_device_index` would throw on the empty device list
    // and prevent the tier policy from falling through to the
    // available non-Vulkan backend.
    bool vulkan_override_wins_tier_policy = false;
    if (vulkan_device != 0 && !vulkan_devs.empty()) {
        const int chosen = pick_vulkan_device_index(vulkan_device,
                                                    vulkan_free_vram,
                                                    vulkan_is_uma);
        const ggml_backend_dev_t chosen_dev = vulkan_devs[(size_t) chosen].dev;
        auto it = std::find_if(other_gpu.begin(), other_gpu.end(),
                                [&](const Cand & c) { return c.dev == chosen_dev; });
        if (it != other_gpu.end()) {
            Cand c = *it;
            other_gpu.erase(it);
            other_gpu.insert(other_gpu.begin(), c);
        }
        // Explicit non-auto override (`vulkan_device > 0`) means the
        // operator deliberately selected Vulkan; surface that to the
        // tier dispatch below so the OpenCL-Adreno preference doesn't
        // silently win on Snapdragon-class devices.
        if (vulkan_device > 0) vulkan_override_wins_tier_policy = true;
        if (verbose) {
            const Cand & c = vulkan_devs[(size_t) chosen];
            const char * label = c.desc && *c.desc ? c.desc :
                                 (c.name && *c.name ? c.name : "unknown");
            if (vulkan_device == -1) {
                bool any_discrete = false;
                for (bool u : vulkan_is_uma) {
                    if (!u) { any_discrete = true; break; }
                }
                fprintf(stderr,
                    "%s: auto-picked Vulkan device %d (%s) — most free VRAM of %d adapter(s)%s\n",
                    log_prefix, chosen, label,
                    (int) vulkan_devs.size(),
                    any_discrete ? " (round-12 UMA bias)" : "");
            } else {
                fprintf(stderr,
                    "%s: using Vulkan device %d (%s) per --vulkan-device override\n",
                    log_prefix, chosen, label);
            }
        }
    } else if (vulkan_device != 0 && vulkan_devs.empty() && verbose) {
        // Override requested but no Vulkan adapter present — log and
        // fall through to the tier policy so the available GPU
        // (CUDA / Metal / Adreno-OpenCL) still gets used.
        fprintf(stderr,
            "%s: vulkan_device=%d requested but no Vulkan adapter visible; "
            "falling through to the tier policy\n",
            log_prefix, vulkan_device);
    }

    // Tier dispatch.  When the operator pinned a specific Vulkan
    // adapter via `vulkan_device > 0`, that explicit choice outranks
    // the OpenCL-Adreno tier preference (review comment 3355995666):
    // the user wants Vulkan, give them Vulkan.  Otherwise the tier
    // policy is unchanged.
    if (vulkan_override_wins_tier_policy) {
        if (ggml_backend_t b = try_init(other_gpu)) return b;
    }
    if (!opencl_adreno_700plus.empty()) {
        if (ggml_backend_t b = try_init(opencl_adreno_700plus)) return b;
    }
    if (!vulkan_override_wins_tier_policy) {
        if (ggml_backend_t b = try_init(other_gpu)) return b;
    }
    if (ggml_backend_t b = try_init(opencl_other)) return b;

    if (verbose) {
        if (max_adreno_version >= 600 && max_adreno_version < 700) {
            fprintf(stderr,
                "%s: only Adreno 6xx OpenCL detected (broken); "
                "falling back to CPU\n",
                log_prefix);
        } else {
            fprintf(stderr,
                "%s: no GPU backend available, falling back to CPU\n",
                log_prefix);
        }
    }
    // Report a GPU declined by policy so the caller accepts CPU fallback as correct
    // (not a regression); not set when a validated GPU was tried and failed to init.
    if (out_gpu_present_but_unused) *out_gpu_present_but_unused = gpu_present_but_unvalidated;
    return nullptr;
}

ggml_backend_t init_cpu_backend() {
    ensure_backends_loaded();
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

ggml_backend_t init_blas_backend() {
    ensure_backends_loaded();
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_ACCEL) continue;
        const char * reg_name = dev_reg_name(dev);
        if (!reg_name || std::strcmp(reg_name, "BLAS") != 0) continue;
        return ggml_backend_dev_init(dev, nullptr);
    }
    return nullptr;
}

} // namespace tts_cpp::detail
