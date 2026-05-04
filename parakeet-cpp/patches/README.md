# ggml patches for parakeet.cpp

`ggml` is vendored as a pristine upstream clone (see the top-level
[`README.md`](../README.md) and [`scripts/setup-ggml.sh`](../scripts/setup-ggml.sh)),
so the local fixes parakeet.cpp depends on live here as standalone
patches and are applied after the clone.

Three patches ship today:

1. [`ggml-backend-reg-filename-prefix.patch`](#ggml-backend-reg-filename-prefixpatch)
   — teaches `ggml_backend_load_best()` to honour a compile-time
   `GGML_BACKEND_DL_PROJECT_PREFIX` macro, so renaming the bundled
   backend .so/.dll files (parakeet does this to avoid colliding with
   another consumer's `libggml-*` files in the same host process) does
   not break runtime backend discovery under `GGML_BACKEND_DL=ON`.
   No-op when the macro is undefined.
2. [`ggml-opencl-allow-non-adreno.patch`](#ggml-opencl-allow-non-adrenopatch)
   — lets the OpenCL backend bring up on commodity desktop GPUs
   (NVIDIA, AMD, Apple) so `parakeet.cpp` can be built and parity-
   tested with `-DGGML_OPENCL=ON` outside Adreno-only environments.
   No-op on real Adreno targets (the patch only relaxes the rejection
   of unknown GPU vendors and the assertion in
   `ggml_backend_opencl_init()` when no devices were found).
3. [`ggml-opencl-program-binary-cache.patch`](#ggml-opencl-program-binary-cachepatch)
   — adds a persistent on-disk cache for compiled OpenCL kernel
   binaries, removing the multi-second `clBuildProgram` wave at every
   cold start. Honours `$GGML_OPENCL_CACHE_DIR`, with
   `$XDG_CACHE_HOME/ggml/opencl` → `$HOME/.cache/ggml/opencl`
   fallbacks. Opt-out via `GGML_OPENCL_CACHE_DIR=""`.

`scripts/setup-ggml.sh` applies every `patches/ggml-*.patch` in
lexicographic order; the script is idempotent and resets the ggml
worktree to the pinned commit before applying.

## Apply

The top-level [`scripts/setup-ggml.sh`](../scripts/setup-ggml.sh) does
everything for you:

```bash
# From the repo root.  Clones ggml if needed, checks out the pinned
# commit, and applies every patch under patches/.  Idempotent --
# re-running is a no-op.
./scripts/setup-ggml.sh
```

Then configure + build as usual. Pick the backend flags for your
platform; OpenCL pulls in the patch automatically:

```bash
# Apple Silicon
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON

# NVIDIA / desktop
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON

# Vulkan (anything else)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON

# OpenCL: Adreno (Android) target
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON

# OpenCL: NVIDIA / AMD / Apple desktop (dev / CI parity testing) --
# Adreno-tuned matmul kernels OFF, generic OpenCL paths only:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF
```

If you'd rather run the steps by hand (e.g. to pin a different
upstream commit), the script is effectively:

```bash
git clone https://github.com/ggml-org/ggml.git ggml
cd ggml && git checkout $GGML_COMMIT
git apply ../patches/ggml-backend-reg-filename-prefix.patch
git apply ../patches/ggml-opencl-allow-non-adreno.patch
git apply ../patches/ggml-opencl-program-binary-cache.patch
```

`GGML_COMMIT` lives at the top of `scripts/setup-ggml.sh` as the
single source of truth -- bump it when re-generating the patches
against a newer upstream ggml. To confirm everything applied
cleanly:

```bash
(cd ggml && git status --short)
# Expected: 2 modified files
#   ggml/src/ggml-backend-reg.cpp     (filename-prefix patch)
#   ggml/src/ggml-opencl/ggml-opencl.cpp  (both OpenCL patches stack on this file)
```

CPU / CUDA / Metal / Vulkan builds get the pinned commit and the
filename-prefix patch (which is a strict no-op when the host
project does not define `GGML_BACKEND_DL_PROJECT_PREFIX`); the
OpenCL changes are no-op for every other backend.

## `ggml-backend-reg-filename-prefix.patch`

Base commit: `58c38058` (`sync : llama.cpp`, 2026-04-09).

Adds a single compile-time switch
`GGML_BACKEND_DL_PROJECT_PREFIX` to `ggml_backend_load_best()` so
the runtime backend-discovery walk can be retargeted at the
filename prefix used by a host project that renames the bundled
`libggml-*` files to avoid colliding with another consumer's
`libggml-*` files in the same host process.

Background: parakeet ships its bundled ggml backends as
`libspeech-ggml-*.{so,dll}` (CMake option
`PARAKEET_GGML_LIB_PREFIX=ON`, default) so a host process that
loads two consumers each vendoring its own ggml does not see a
name clash on `libggml-vulkan.so` / `libggml-cuda.so` / etc. The
`speech-` prefix is shared with the rest of the QVAC speech stack
(whisper, parakeet, chatterbox, supertonic, ...) so the family
co-vendors a single ggml file set.
Without this patch, the rename works at link time but
`ggml_backend_load_best()` still searches for `libggml-*.so` /
`ggml-*.dll`, so under `GGML_BACKEND_DL=ON` the renamed files are
on disk but never discovered and Vulkan/OpenCL/CUDA backends
silently fail to load.

| Symptom | Root cause | What this patch does |
|---------|-----------|----------------------|
| `speech-ggml-vulkan.so` (etc.) is on disk but ggml's loader never picks it up under `GGML_BACKEND_DL=ON` | `backend_filename_prefix()` hard-codes `libggml-` / `ggml-` and `ggml_backend_load_best` filters directory entries by that fixed prefix | Honour an optional compile-time `GGML_BACKEND_DL_PROJECT_PREFIX` string literal (e.g. `"speech-"`); when defined, the loader searches for `lib<prefix>ggml-*` / `<prefix>ggml-*` instead. Macro undefined ⇒ behaviour byte-equal to upstream. |

The CMake side wires the macro from `PARAKEET_GGML_LIB_PREFIX`:
when that option is on (the default), parakeet's top-level
`CMakeLists.txt` does
`target_compile_definitions(ggml PRIVATE GGML_BACKEND_DL_PROJECT_PREFIX="speech-")`
on the `ggml` target (which is what compiles
`ggml-backend-reg.cpp`). Consumers that prefer the upstream
filenames (system ggml, single-consumer hosts) configure with
`-DPARAKEET_GGML_LIB_PREFIX=OFF` and the macro stays undefined,
so the loader behaviour matches stock ggml exactly.

## `ggml-opencl-allow-non-adreno.patch`

Base commit: `58c38058` (`sync : llama.cpp`, 2026-04-09).

Fixes two gaps in `ggml-opencl` that make `-DGGML_OPENCL=ON` builds of
`parakeet.cpp` impossible to bring up outside an Adreno-only
environment:

| Symptom                                                                                                | Root cause in `ggml-opencl`                                                                                                                                                                                                                                                                                            | What this patch does                                                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Every NVIDIA / AMD / Apple OpenCL device is dropped at init with `Unsupported GPU: <device-name>`      | `ggml_cl2_init()` whitelists `Adreno` / `Qualcomm` / `Intel` and returns `nullptr` for everything else. Even with `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF`, a non-Adreno GPU never reaches the generic kernels.                                                                                                           | Default behaviour is byte-equal to upstream (still returns `nullptr`). Set `GGML_OPENCL_ALLOW_UNKNOWN_GPU=1` to opt the device through with `GPU_FAMILY::UNKNOWN`; we additionally require `cl_intel_required_subgroup_size` *or* `cl_qcom_reqd_sub_group_size` (the matmul-vec kernels need one to define `N_DST`/`N_SIMDGROUP`/`N_SIMDWIDTH`), so AMD/NVIDIA still fall back to host instead of crashing in `clBuildProgram`. |
| `parakeet --n-gpu-layers 1` aborts with `GGML_ASSERT(index < ggml_backend_opencl_reg_device_count(reg))` when zero usable devices were found | `ggml_backend_opencl_init()` calls `ggml_backend_reg_dev_get(reg, 0)` unconditionally. When the device discovery cleared the list (e.g. only an unsupported GPU was present), `dev_get(0)` asserts and the host process aborts. parakeet's `init_gpu_backend()` cascade expects a nullable result so it can fall back. | Check `ggml_backend_reg_dev_count(reg) == 0` before `dev_get` and return `nullptr` on empty. Also propagate `nullptr` when `ggml_cl2_init()` rejects the device, so the host-side fallback path actually runs.                |

The patch is **strictly additive** for real Adreno targets:
`gpu_family == ADRENO` is computed exactly as before, the Adreno
shuffle / large-buffer paths still trigger when (and only when) the
device is Adreno, and without `GGML_OPENCL_ALLOW_UNKNOWN_GPU=1` the
non-Adreno reject path is byte-equal to upstream so production Android
builds get the same compile-time guarantees as before.

The intended audience for the patch is:

  * `parakeet.cpp` developers running CI on Intel iGPU desktop
    hardware (the matmul-vec kernels gate on
    `cl_intel_required_subgroup_size`, so Intel iGPU is the only
    desktop class that can actually execute the OpenCL kernels;
    AMD/NVIDIA users get a clean CPU fallback instead of crashing
    inside `clBuildProgram`).
  * Anyone who wants to reproduce the OpenCL backend's mel/encoder
    parity numbers without an Adreno device.

Opt-in is gated behind `GGML_OPENCL_ALLOW_UNKNOWN_GPU=1` so misconfigured
production builds still get the same explicit `Unsupported GPU` error
upstream returned, instead of a silent "running with an untested GPU".

It is **not** intended to ship a fast OpenCL path on NVIDIA / AMD /
Apple desktops (CUDA / Vulkan / Metal are far better suited there);
its only purpose is bring-up + parity testing.

## `ggml-opencl-program-binary-cache.patch`

Base commit: `58c38058` (`sync : llama.cpp`, 2026-04-09).

Adds a persistent on-disk cache for compiled OpenCL kernel binaries
to `ggml-opencl`. Upstream `build_program_from_source()` calls
`clCreateProgramWithSource` + `clBuildProgram` on every cold start,
re-paying the driver's shader-compile wave (multiple seconds on
Adreno / Mesa / Mali; tens of ms on most desktop drivers). This
patch drops the call to `clCreateProgramWithBinary` against a
device-specific cache blob whenever one exists, and persists every
freshly-compiled program back to disk on miss.

| Symptom                                                                                | Root cause                                                                              | What this patch does                                                                                              |
|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| Every cold-start `parakeet --n-gpu-layers 1` re-compiles all 88 OpenCL kernels    | `build_program_from_source` always calls `clCreateProgramWithSource` + `clBuildProgram` | Look up `<cache_dir>/<key>.bin` first via `clCreateProgramWithBinary`; only fall through to source compile on miss |
| Hosts already `setenv` `GGML_OPENCL_CACHE_DIR` for the same goal, but ggml-opencl ignores it | The env var is read **nowhere** in upstream ggml-opencl at this commit  | Resolves cache dir from `$GGML_OPENCL_CACHE_DIR` → `$XDG_CACHE_HOME/ggml/opencl` → `$HOME/.cache/ggml/opencl`, so the env-var contract takes effect. |

### Cache key

`<src_hash>_<opts_hash>_<driver_hash>_<dev_name_hash>_<dev_ver_hash>.bin`,
where each component is FNV-1a-64. Each kernel's `program_buffer`
hashes independently (88 different cache files per device); a
driver upgrade or moving to a different device silently invalidates
the cache because either `driver_hash` or `dev_*_hash` changes.
There is no manual invalidation step.

### Atomic writes

The cache writer dumps `getProgramInfo(CL_PROGRAM_BINARIES)` to
`<path>.tmp` then `rename(2)`s into place. POSIX rename is atomic,
so concurrent processes can't read a half-written file; the
last-writer-wins result is fine because each blob is independently
valid for the same `(src, opts, driver, dev)` combination.

### Footprint

Each kernel binary lands at ~10-200 KB on Adreno (driver-dependent);
88 kernels × ~50 KB average ≈ 4-5 MB on disk per device per process
family. No size cap on disk today -- if it ever becomes a concern
on tightly-budgeted mobile installs, wrap the writer with a
ceiling.

### Opt-out / disable

`GGML_OPENCL_CACHE_DIR=""` (literal empty string) short-circuits
both the read and the write paths and runs the original
source-compile route. Useful for benchmarking the cold-start cost,
or in a CI runner that wants every run to re-compile.

When the cache dir resolves but `mkdir -p` fails (read-only
filesystem, permissions, ...), the writer logs nothing and falls
through to source compile silently -- no behavioural difference
versus running with the patch absent.

### Stale-cache handling

`clCreateProgramWithBinary` can return `CL_INVALID_BINARY` (or the
subsequent `clBuildProgram` can fail) when the on-disk blob is
stale (driver upgrade, different shader IR version, mismatched
device). The patch handles every such failure by releasing the
program and falling through to source compile. The next run then
overwrites the bad blob.

### Measured impact

This patch is **not yet benchmarked on a real Adreno device**: the
benchmark hosts the patch was developed on are NVIDIA-only, and
NVIDIA's OpenCL driver lacks the fp16 / OpenCL C 2.0 features
ggml-opencl mandates -- the kernels never compile at all there, so
there is nothing to cache. Expected impact:

  * **Cold start (no cache)**: same as upstream -- multi-second
    shader compile wave on Adreno.
  * **Warm cache** (any subsequent invocation): saves the entire
    `clBuildProgram` wave; typical Adreno saving is multiple
    seconds per process.

Once Adreno hardware is available for follow-up benchmarking, the
expected bench shape is the standard pipeline-cache curve:
cold ≫ ggml-warm ≈ both-warm.

## Dropping the patches

If upstream ggml-opencl decides to relax the GPU-vendor whitelist
itself, or ships its own kernel binary cache, delete the patch
file(s) and remove the corresponding entry from the `PATCHES=(…)`
glob in `scripts/setup-ggml.sh`. The C++ side of parakeet uses
only ops that ggml-opencl already supports natively (per the
op-coverage audit), so nothing else needs to change.
