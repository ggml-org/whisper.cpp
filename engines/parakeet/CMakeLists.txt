cmake_minimum_required(VERSION 3.20)
project(parakeet-cpp LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(PARAKEET_STANDALONE_DEFAULT ON)
else()
    set(PARAKEET_STANDALONE_DEFAULT OFF)
endif()

option(PARAKEET_BUILD_LIBRARY     "parakeet: build the parakeet library (linkage follows BUILD_SHARED_LIBS; defaults to STATIC when unset)" ON)
option(PARAKEET_BUILD_EXECUTABLES "parakeet: build the parakeet CLI"                            ${PARAKEET_STANDALONE_DEFAULT})
option(PARAKEET_BUILD_TESTS       "parakeet: build per-stage numerical validation harnesses"         ${PARAKEET_STANDALONE_DEFAULT})
option(PARAKEET_BUILD_EXAMPLES    "parakeet: build standalone examples (live-mic, ...)"              ${PARAKEET_STANDALONE_DEFAULT})
option(PARAKEET_INSTALL           "parakeet: generate install rules"                                 ON)
option(PARAKEET_USE_SYSTEM_GGML   "parakeet: use system-installed GGML library"                      OFF)
option(PARAKEET_CCACHE            "parakeet: use ccache as compiler launcher if available"           ON)
option(PARAKEET_GGML_LIB_PREFIX   "parakeet: rename bundled ggml libraries to libspeech-ggml-* (filename only; CMake target names and C symbols are unchanged). Prevents shared-library filename collisions with other ggml consumers in the same host process; the `speech-` prefix is shared with the rest of the QVAC speech stack (whisper, parakeet, chatterbox, supertonic, ...) so they vendor a single ggml file set. Has no effect when PARAKEET_USE_SYSTEM_GGML=ON." ON)
option(PARAKEET_COREML            "parakeet: enable the Apple Core ML (Neural Engine) encoder sidecar (Apple-only, additive; the FastConformer encoder runs on a <model>-encoder.mlmodelc when present, else falls back to the ggml encoder)." OFF)

# Enable Objective-C and Objective-C++ up front (before add_subdirectory(ggml))
# for the Apple Core ML encoder sidecar. Both must happen here:
#   - enabling OBJCXX only after ggml pulls in the Metal toolchain leaves
#     CMAKE_OBJCXX_COMPILE_OBJECT unset under CMake 4.x (the .mm wrapper then
#     gets no compile rule at generate time);
#   - ggml-metal ships Objective-C (.m) sources; enabling OBJCXX without OBJC
#     makes CMake compile those .m files as Objective-C++, whose C++ semantics
#     reject ggml-metal's implicit void* conversions. Enabling OBJC keeps the
#     .m sources on Objective-C (C) semantics.
if (PARAKEET_COREML)
    if (NOT APPLE)
        message(FATAL_ERROR "PARAKEET_COREML is Apple-only (requires the CoreML framework); leave it OFF on ${CMAKE_SYSTEM_NAME}.")
    endif()
    enable_language(OBJC)
    enable_language(OBJCXX)
endif()

# ccache shaves ~30 % off iterative rebuild time on developer machines.
# Opt-out via -DPARAKEET_CCACHE=OFF for clean reproducibility runs.
# Scoped to parakeet's own targets via the helper below; ggml has
# its own GGML_CCACHE option which we leave alone so the two launchers
# don't double-up on the ggml subdirectory.
if (PARAKEET_CCACHE)
    find_program(PARAKEET_CCACHE_PROGRAM ccache)
    if (PARAKEET_CCACHE_PROGRAM)
        message(STATUS "parakeet: using ccache at ${PARAKEET_CCACHE_PROGRAM} (parakeet targets only; ggml uses GGML_CCACHE)")
    endif()
endif()

function(parakeet_apply_ccache target)
    if (PARAKEET_CCACHE_PROGRAM)
        set_target_properties(${target} PROPERTIES
            C_COMPILER_LAUNCHER   "${PARAKEET_CCACHE_PROGRAM}"
            CXX_COMPILER_LAUNCHER "${PARAKEET_CCACHE_PROGRAM}"
        )
    endif()
endfunction()

if (CMAKE_CROSSCOMPILING)
    if (NOT DEFINED GGML_NATIVE OR GGML_NATIVE)
        message(STATUS "parakeet: cross-compiling to ${CMAKE_SYSTEM_NAME} -- "
                       "forcing GGML_NATIVE=OFF (ggml's ARM feature probes use try_run() "
                       "which CMake refuses in cross-compile mode). The EOU q8_0 precision "
                       "caveat from native macOS/Linux builds does not apply: mobile prebuilds "
                       "are per-arch and use the armv8-a baseline.")
        set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    endif()
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    if (NOT DEFINED GGML_BLAS OR GGML_BLAS)
        set(GGML_BLAS OFF CACHE BOOL "" FORCE)
    endif()
    if (NOT DEFINED GGML_ACCELERATE OR GGML_ACCELERATE)
        set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
    endif()
endif()

# Android default backend stack: dynamic loading of Vulkan + OpenCL +
# per-arch CPU variants. Mirrors the qvac llm-llamacpp Android config
# (see qvac-registry-vcpkg/ports/llama-cpp/portfile.cmake) so the
# parakeet prebuilds drop into the same `qvac__transcription-parakeet/`
# folder shape as the llamacpp ones: a `.bare` module + sibling
# `lib<prefix>ggml-{vulkan,opencl,cpu-android_armv*_*}.so` files that
# `ggml_backend_load_all_from_path()` discovers at runtime.
#
# Selection at runtime is centralised in `init_gpu_backend()`
# (src/parakeet_ctc.cpp): OpenCL when an Adreno 700+ device is
# present, Vulkan for every other GPU (non-Adreno, Adreno < 700,
# Mali, Xclipse, ...). No static GPU backend entry points are linked
# anywhere in libqvac-parakeet; the registry walk reaches the right
# backend in both GGML_BACKEND_DL=ON (Android prebuild) and
# GGML_BACKEND_DL=OFF (desktop dev) modes.
#
# Callers that have specific reasons to deviate (e.g. a desktop bring-
# up build that wants Vulkan only) can still override any of these
# at the cmake command line; we only set defaults that haven't already
# been provided.
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    if (NOT DEFINED CACHE{GGML_BACKEND_DL})
        set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    endif()
    if (NOT DEFINED CACHE{GGML_CPU_ALL_VARIANTS})
        set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
    endif()
    if (NOT DEFINED CACHE{GGML_CPU_REPACK})
        set(GGML_CPU_REPACK ON CACHE BOOL "" FORCE)
    endif()
    if (NOT DEFINED CACHE{GGML_VULKAN})
        set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    endif()
    if (NOT DEFINED CACHE{GGML_OPENCL})
        set(GGML_OPENCL ON CACHE BOOL "" FORCE)
    endif()
    # ggml-vulkan's coopmat / coopmat2 shader compile pulls in extensions
    # that most Android Vulkan drivers don't expose; the upstream llama
    # Android build disables both for the same reason.
    if (NOT DEFINED CACHE{GGML_VULKAN_DISABLE_COOPMAT})
        set(GGML_VULKAN_DISABLE_COOPMAT ON CACHE BOOL "" FORCE)
    endif()
    if (NOT DEFINED CACHE{GGML_VULKAN_DISABLE_COOPMAT2})
        set(GGML_VULKAN_DISABLE_COOPMAT2 ON CACHE BOOL "" FORCE)
    endif()
endif()

# Two related workarounds for clang-cl / MSVC builds on Windows. Both
# come from msys2 sneaking GCC-flavoured libraries onto CMake's search
# paths and then being mismatched against clang-cl-compiled translation
# units. Both are no-ops on MinGW (excluded) and on non-Windows.
#
# (1) libm. ggml/src/CMakeLists.txt does
#       find_library(MATH_LIBRARY m)
#       if (MATH_LIBRARY AND (NOT WIN32 OR NOT DEFINED ENV{ONEAPI_ROOT}))
#           target_link_libraries(ggml-base PRIVATE m)
#     and unconditionally links libm on Windows unless ONEAPI_ROOT is in
#     the environment. With clang-cl / lld-link libm isn't a separate
#     library (math is in the MSVC C runtime); find_library picks up a
#     stray `libm.dll.a` from msys2 and lld-link fails with
#     `could not open 'm.lib'`. Piggyback on ggml's existing escape hatch
#     by setting ONEAPI_ROOT in the configure-time environment.
#     `set(ENV{...})` affects only the running CMake configure process,
#     which is the same scope ggml's check runs in.
#
# (2) OpenMP. CMake's FindOpenMP picks `-fopenmp=libomp` (LLVM-style)
#     for clang-cl as the compile flag, but its `OpenMP_*_LIBRARIES`
#     resolves to msys2's `libgomp.dll.a` if msys2 is on PATH. The
#     resulting object files reference LLVM `__kmpc_*` symbols that GOMP
#     doesn't provide and lld-link errors out with `undefined symbol:
#     __kmpc_global_thread_num` etc. Setting up clang-cl + libomp end
#     to end on Windows requires per-environment knobs (LLVM's own
#     `libomp.lib` location, OMP_PATH, runtime DLL on PATH, ...) that
#     don't belong in the listfile. Disable OpenMP for this toolchain;
#     the perf hit on the CPU encoder is bounded and only affects this
#     specific build environment.
if (WIN32 AND NOT MINGW)
    if (NOT DEFINED ENV{ONEAPI_ROOT})
        set(ENV{ONEAPI_ROOT} "1")
    endif()
    if (NOT DEFINED GGML_OPENMP AND NOT DEFINED CACHE{GGML_OPENMP})
        # Normal (non-cache) variable on purpose: scoped to this listfile +
        # its add_subdirectory(ggml) below, doesn't poison parent-project
        # CMakeCache when parakeet-cpp is consumed via add_subdirectory.
        # ggml's `option(GGML_OPENMP ... ON)` respects an existing normal
        # variable under CMP0077 (CMake 3.13+), so this still takes effect.
        # Consumers can override at any time with `-DGGML_OPENMP=ON` (cache)
        # or by pre-`set()`ing the variable in the parent listfile.
        set(GGML_OPENMP OFF)
    endif()
endif()

# Bundled-ggml library filename prefix. qvac-ext-ggml's `speech` branch
# exposes `GGML_LIB_OUTPUT_PREFIX` (commit 4cec2d3a) which handles both
# the OUTPUT_NAME rename for every ggml target (core + per-backend
# .so/.dll/.a) AND the runtime loader's filename prefix
# (`GGML_BACKEND_DL_PROJECT_PREFIX` compile define on ggml-base), so
# the renamed `libspeech-ggml-{vulkan,opencl,cpu-*}.so` files are
# actually discovered by `ggml_backend_load_all_from_path()` at
# runtime.
#
# Setting `GGML_LIB_OUTPUT_PREFIX` here (as a cache variable, before
# `add_subdirectory(ggml)`) is the supported way to override the
# branch default (`qvac-speech-`) on a per-consumer basis without
# editing the ggml subtree. The `speech-` prefix is shared across the
# QVAC speech stack (whisper, parakeet, chatterbox, supertonic, ...)
# so they can vendor a single ggml file set side-by-side without
# colliding with the `qvac-` prefix used by the llm fork.
if (PARAKEET_GGML_LIB_PREFIX AND NOT PARAKEET_USE_SYSTEM_GGML)
    if (NOT DEFINED CACHE{GGML_LIB_OUTPUT_PREFIX})
        set(GGML_LIB_OUTPUT_PREFIX "speech-" CACHE STRING
            "ggml: prefix for built ggml library filenames (parakeet default)" FORCE)
    endif()
    message(STATUS "parakeet: bundled ggml libraries will be emitted with prefix '${GGML_LIB_OUTPUT_PREFIX}' (set PARAKEET_GGML_LIB_PREFIX=OFF to use the qvac-ext-ggml@speech default, or override -DGGML_LIB_OUTPUT_PREFIX=<other>)")
endif()

if (NOT TARGET ggml)
    if (PARAKEET_USE_SYSTEM_GGML)
        find_package(ggml CONFIG REQUIRED)
        if (NOT ggml_FOUND)
            message(FATAL_ERROR "System-installed GGML library not found.")
        endif()
        add_library(ggml ALIAS ggml::ggml)
    else()
        add_subdirectory(ggml)
    endif()
endif()

# Same OpenMP avoidance as for ggml above: on Windows non-MinGW builds
# CMake's FindOpenMP picks LLVM's `-fopenmp=libomp` compile flag but
# resolves OpenMP_*_LIBRARIES to msys2 libgomp -> link-time mismatch.
# Skip find_package() entirely so OpenMP_*_FOUND stays falsy and the
# parakeet target falls back to its non-OpenMP code paths. Pass
# `-DPARAKEET_OPENMP=ON` to override after wiring up LLVM libomp.
option(PARAKEET_OPENMP "parakeet: enable OpenMP for the parakeet target" ON)
if (WIN32 AND NOT MINGW AND PARAKEET_OPENMP AND NOT DEFINED CACHE{PARAKEET_OPENMP_USER_OVERRIDE})
    set(PARAKEET_OPENMP OFF CACHE BOOL "" FORCE)
    message(STATUS "parakeet: OpenMP disabled on Windows non-MinGW (clang-cl / MSVC + msys2 libgomp mismatch). Set -DPARAKEET_OPENMP_USER_OVERRIDE=ON and -DPARAKEET_OPENMP=ON to force-enable.")
endif()

if (PARAKEET_OPENMP)
    find_package(OpenMP)
endif()

# Legacy interface library kept for export-set compatibility (it is
# still part of `install(EXPORT qvac-parakeet-targets)` below and
# downstream `find_package(qvac-parakeet)` consumers list it as a link
# dep). Body intentionally empty: parakeet routes every backend
# decision through the ggml-backend registry
# (`ggml_backend_load_all` + `ggml_backend_dev_*`, see
# `init_gpu_backend()` / `init_cpu_backend()` / `init_blas_backend()`
# in src/parakeet_ctc.cpp) and does NOT call any
# `ggml_backend_<backend>_init` / `ggml_backend_is_<backend>` entry
# point directly. The `GGML_USE_VULKAN` / `GGML_USE_OPENCL` /
# `GGML_USE_METAL` / `GGML_USE_CUDA` / `GGML_USE_BLAS` compile defines
# that used to live here were only consumed by `#ifdef` cascades that
# called those static entry points; with the registry-only design
# they're dead, and shipping them would falsely advertise a static
# backend dependency that the GGML_BACKEND_DL=ON Android/Linux builds
# explicitly do not have (their backends live in separately-loadable
# `.so` files that are dlopen()'d by `ggml_backend_load_all_from_path`
# at runtime).
add_library(parakeet-backend-defs INTERFACE)

set(PARAKEET_MOSS_SOURCES
    src/moss/gguf_metadata.cpp
    src/moss/transcribe_runtime.cpp
    src/moss/transcribe_bpe.cpp
    src/moss/transcribe_model.cpp
    src/moss/transcribe_tensors.cpp
    src/moss/transcribe_audio.cpp
    src/moss/transcribe_encoder.cpp
    src/moss/transcribe_coreml.cpp
    src/moss/transcribe_decoder.cpp
    src/moss/transcribe_text.cpp
    src/moss/transcribe_fit.cpp
    src/moss/transcribe_engine.cpp
    src/moss/transcribe_cli.cpp
)

set_source_files_properties(${PARAKEET_MOSS_SOURCES} PROPERTIES
    COMPILE_OPTIONS "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-fast-math>")

set(PARAKEET_LIB_SOURCES
    ${PARAKEET_MOSS_SOURCES}
    src/parakeet_ctc.cpp
    src/parakeet_coreml_path.cpp
    src/coreml/parakeet_coreml_shape.cpp
    src/parakeet_engine.cpp
    src/parakeet_nemotron.cpp
    src/parakeet_unified.cpp
    src/parakeet_tdt.cpp
    src/parakeet_eou.cpp
    src/parakeet_sortformer.cpp
    src/parakeet_diarization_v3.cpp
    src/parakeet_log.cpp
    src/mel_preprocess.cpp
    src/sentencepiece_bpe.cpp
    src/energy_vad.cpp
    # Memory-fit preflight (include/parakeet/fit.h + parakeet-fit-params).
    src/parakeet_fit.cpp
    # Lives in the library so `parakeet_cli_main` (declared in
    # the umbrella header `parakeet.h`) is actually a library
    # symbol that consumers / host applications can link against.
    # The CLI executable below is a thin shim that just calls into it.
    src/main.cpp
    # Same shape for `parakeet_fit_cli_main` (the parakeet-fit-params tool).
    src/fit_main.cpp
)

if (PARAKEET_BUILD_LIBRARY)
    # Honour BUILD_SHARED_LIBS when set explicitly; otherwise default to
    # STATIC. Same shape as the standard llama-style library contract
    # (BUILD_SHARED_LIBS-driven), so consumers that build other ggml-
    # based libraries shared see a consistent linkage selector.
    if (DEFINED BUILD_SHARED_LIBS)
        add_library(qvac-parakeet ${PARAKEET_LIB_SOURCES})
    else()
        add_library(qvac-parakeet STATIC ${PARAKEET_LIB_SOURCES})
    endif()
    add_library(qvac::parakeet ALIAS qvac-parakeet)
    parakeet_apply_ccache(qvac-parakeet)

    target_include_directories(qvac-parakeet
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    )
    target_link_libraries(qvac-parakeet PUBLIC ggml)

    # Export the language requirement so it reaches consumers through the
    # installed target. The public headers use C++11 constructs (defaulted
    # /deleted members, `noexcept`, alias templates in streaming.h), and
    # the library itself is compiled at C++17 -- but CMAKE_CXX_STANDARD is
    # a *build-local* variable and is NOT part of the export, so before
    # this a `find_package(qvac-parakeet)` consumer that did not set its
    # own standard compiled the headers at its compiler's default dialect
    # and failed on those constructs. In-repo consumers (and the qvac
    # addon, which sets CMAKE_CXX_STANDARD 20) masked the gap; the
    # package-consumption test deliberately sets no standard so it stays
    # caught. cxx_std_17 rather than cxx_std_11: 17 is what the library is
    # built and tested at, so advertising 11 would claim a configuration
    # nothing exercises.
    target_compile_features(qvac-parakeet PUBLIC cxx_std_17)

    if (OpenMP_CXX_FOUND)
        target_link_libraries(qvac-parakeet PRIVATE OpenMP::OpenMP_CXX)
    endif()

    target_link_libraries(qvac-parakeet PRIVATE parakeet-backend-defs)

    # Single source of truth for the version string the CLI prints
    # under `--version`. Kept PRIVATE because it only drives the in-
    # library `main.cpp` impl; consumers that need a runtime version
    # can read it via the public API surface or the CMake package
    # version file (`qvac-parakeet-config-version.cmake`).
    target_compile_definitions(qvac-parakeet PRIVATE
        PARAKEET_VERSION="${PROJECT_VERSION}"
    )

    # `PARAKEET_EXPERIMENTAL_FLASH_ATTN` gates the rel-pos MHA
    # fused-attention path in `parakeet_ctc.cpp`. Per-backend defaults:
    #
    #   - Metal:  ON. Validated on M3 Ultra (5 invocations x 15 runs,
    #             sample-16k.wav, q8_0): encoder 67.35 -> 66.96 ms
    #             (-0.39 ms / -0.6 %), inference 119.24 -> 118.84 ms
    #             (-0.40 ms / -0.34 %), parity gates byte-exact (TDT
    #             decoder 95 tokens; encoder-capture 258048 + 258300
    #             floats bit-equal).
    #             `kernel_flash_attn_ext_f32_dk128_dv128` is the kernel
    #             actually loaded; head_dim 128 covers parakeet-{ctc,
    #             tdt,eou} d_model=1024/n_heads=8 and parakeet-tdt-1.1b
    #             d_model=2048/n_heads=16. Sortformer's transformer
    #             head_dim=24 falls below ggml_flash_attn_ext's
    #             supported set so the encoder path there is untouched
    #             (Sortformer uses `parakeet_sortformer.cpp` not
    #             `parakeet_ctc.cpp::rel_pos_mha_graph`).
    #   - CUDA:   ON. The rel-pos attention feeds the flash-attn kernel
    #             head-major views with the bias folded into the mask;
    #             per-head masks are rejected there, so heads fold into
    #             the batch dim (probed at load). RTX 3080, TDT q8_0:
    #             encoder 30.0 -> 25.8 ms on 30 s audio, transcripts
    #             unchanged, WER subset 2.1741 -> 2.1839 %.
    #   - CPU:    OFF. Cast-to-f16 of the BD mask before softmax
    #             measurably changed accumulator order and showed
    #             +3.1 % encoder regression on the CPU parity sweep;
    #             risks the bit-equal-NeMo gate.
    #   - Others: OFF until backend-specific A/B. ggml-opencl supports
    #             `flash_attn_ext` for head_dim in
    #             {40,64,80,96,112,128,192,256}; first-Adreno bring-up
    #             should flip and bench. Vulkan validation harness
    #             (test/test_vk_vs_cpu.cpp) is the natural home for
    #             that A/B once a non-Tegra Vulkan device is available.
    if (GGML_METAL OR GGML_CUDA OR GGML_VULKAN)
        set(PARAKEET_FLASH_ATTN_DEFAULT ON)
    else()
        set(PARAKEET_FLASH_ATTN_DEFAULT OFF)
    endif()
    option(PARAKEET_FLASH_ATTN "parakeet: enable fused flash-attn in MHA (default ON for Metal, CUDA and Vulkan; OFF elsewhere pending per-backend A/B)" ${PARAKEET_FLASH_ATTN_DEFAULT})
    if (PARAKEET_FLASH_ATTN)
        target_compile_definitions(qvac-parakeet PRIVATE PARAKEET_EXPERIMENTAL_FLASH_ATTN)
    endif()

    # `PARAKEET_F16_ACTIVATIONS` narrows the F32 encoder residual stream at
    # exactly one site: the three batched attention MULMATs in
    # `rel_pos_mha_unfused_graph`. `k_perm` / `p_perm` / `v_for_mm` are
    # zero-padded on their matmul N/K axis to a 32-aligned tail and cast to
    # F16, and the softmax output is zero-padded to match; the MULMATs
    # (`ac`, `bd`, `attn_v`) then land on the Hexagon HMX F16-batched kernel
    # (`hmx_mm_f16_f32_batched`) instead of `hvx-batched-f32`. Everything
    # else — subsampling, FF/Conv projections, layer norms, residual adds,
    # the `attn_out` Q8_0 matmul, the CTC head — stays F32 exactly as
    # before, so downstream decoders and every other backend are unaffected.
    # When OFF the helpers early-return with the input tensor unmodified:
    # no PAD, no CAST, no CONT nodes added, byte-identical to the
    # pre-refactor graph on every backend (CPU / OpenCL / Metal / Vulkan /
    # Hexagon).
    #
    # Default flipped ON after full correctness validation (2026-09-30):
    #   * All four Parakeet variants (CTC / TDT / EOU / Sortformer)
    #     produce byte-identical transcripts on CPU, Hexagon HTP0 and
    #     OpenCL Adreno with the flag ON vs OFF
    #   * LibriSpeech dev-clean 100-clip WER on Hexagon: delta = 0.000%
    #     (2.349% both configs; 100/100 hypotheses byte-identical)
    #   * Long-form 191 s clip: byte-identical CPU vs HTP0 vs OpenCL
    #   * Per-stage encoder cosine vs CPU >= 0.9994 at every stage
    #   * Perf: -47% encoder-time on Hexagon CTC (4790 -> 2541 ms median
    #     on speech-60, RTF 12.5x -> 23.6x realtime)
    # Set `-DPARAKEET_F16_ACTIVATIONS=OFF` to opt out (e.g. for a
    # backend-specific build where the extra PAD/CAST/CONT nodes have no
    # perf upside and the byte-identical fallback is desired).
    #
    # Motivation: HMX v79 is F16-only. A prior attempt to route these
    # matmuls to HMX by casting src0 without alignment padding failed the
    # kernel's `n%32==0 && k%32==0` gate (matmul-ops.c:3081); a full F16
    # residual-stream refactor failed on-device because Hexagon carries
    # only `quantize_f32_q8_0_*` — Q8_0xF16 FF/Conv matmuls return
    # DSP NO_SUPPORT. The narrow boundary cast + N/K pad matches
    # whisper.cpp's convention and preserves transcript byte-parity.
    option(PARAKEET_F16_ACTIVATIONS "parakeet: cast attention K/V-perm and softmax output to F16 (with 32-aligned pad) at the batched attention MULMAT boundary so Hexagon HMX F16-batched engages; default ON (validated 2026-09-30)." ON)
    if (PARAKEET_F16_ACTIVATIONS)
        target_compile_definitions(qvac-parakeet PRIVATE PARAKEET_F16_ACTIVATIONS=1)
    endif()

    # -----------------------------------------------------------------------
    # Apple Core ML encoder sidecar.
    #
    # Optional, Apple-only, additive. Compiles the Objective-C++ Core ML wrapper
    # (src/coreml/parakeet-encoder.mm) into the parakeet library and defines
    # PARAKEET_USE_COREML so run_encoder can route the offline FastConformer
    # forward through a `<model>-encoder.mlmodelc` on the Apple Neural Engine.
    # Mel preprocessing, TDT decode and the tokenizer stay on ggml, and any
    # non-Apple build or missing/incompatible sidecar falls back to the ggml
    # encoder. Modelled on the whisper.cpp WHISPER_COREML pattern; the encoder
    # I/O contract the exported .mlmodelc must honour is documented in
    # src/coreml/parakeet-encoder.h.
    if (PARAKEET_COREML)
        # OBJCXX is enabled up front (see the top-level PARAKEET_COREML block);
        # here we only bind the frameworks and fold the wrapper into libqvac-parakeet.
        find_library(PARAKEET_FOUNDATION_FRAMEWORK Foundation REQUIRED)
        find_library(PARAKEET_COREML_FRAMEWORK     CoreML     REQUIRED)

        # OBJECT library so the ARC-compiled Core ML objects fold into libqvac-parakeet
        # (a single archive for the static vcpkg build; no extra install/export target).
        add_library(parakeet.coreml OBJECT src/coreml/parakeet-encoder.mm)
        parakeet_apply_ccache(parakeet.coreml)
        target_include_directories(parakeet.coreml PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
        )
        set_target_properties(parakeet.coreml PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            COMPILE_FLAGS "-fobjc-arc"
            XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
        )

        target_sources(qvac-parakeet PRIVATE $<TARGET_OBJECTS:parakeet.coreml>)
        target_compile_definitions(qvac-parakeet PRIVATE PARAKEET_USE_COREML)
        target_link_libraries(qvac-parakeet PUBLIC
            ${PARAKEET_FOUNDATION_FRAMEWORK}
            ${PARAKEET_COREML_FRAMEWORK}
        )
        message(STATUS "parakeet: Core ML encoder sidecar enabled (PARAKEET_USE_COREML)")
    endif()

    set_target_properties(qvac-parakeet PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        # EXPORT_NAME + the qvac:: install namespace below make the
        # imported target `qvac::parakeet`, matching the in-tree ALIAS.
        EXPORT_NAME parakeet
        CXX_VISIBILITY_PRESET    hidden
        VISIBILITY_INLINES_HIDDEN ON
        # VERSION drives the full filename suffix on Linux/macOS shared
        # libs (libqvac-parakeet.so.0.1.0); SOVERSION drives the symlink
        # the dynamic loader actually opens (libqvac-parakeet.so.0). Pinning
        # SOVERSION to 0 (rather than ${PROJECT_VERSION_MAJOR}) keeps
        # the ABI tag stable across pre-1.0 PROJECT_VERSION bumps so
        # downstream consumers do not need to re-link on every patch
        # release. Both properties are no-ops for STATIC builds.
        VERSION   ${PROJECT_VERSION}
        SOVERSION 0
    )

    # Symbol-export contract for shared builds. Standard llama-style
    # `*_SHARED` / `*_BUILD` pair: PUBLIC define so consumers see the
    # import side of PARAKEET_API, PRIVATE define inside the library's
    # TUs to flip Windows from `dllimport` to `dllexport`. Both no-ops
    # in static builds (the default), so static consumers see exactly
    # the same surface as before.
    get_target_property(_parakeet_type qvac-parakeet TYPE)
    if (_parakeet_type STREQUAL "SHARED_LIBRARY")
        target_compile_definitions(qvac-parakeet PUBLIC  PARAKEET_SHARED)
        target_compile_definitions(qvac-parakeet PRIVATE PARAKEET_BUILD)
    endif()

    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
        target_compile_options(qvac-parakeet PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:C>:-O3>
            $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
            $<$<COMPILE_LANGUAGE:C>:-ffast-math>
            $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>
            $<$<COMPILE_LANGUAGE:C>:-funroll-loops>
        )
    endif()
endif()

if (PARAKEET_BUILD_EXECUTABLES)
    if (NOT PARAKEET_BUILD_LIBRARY)
        message(FATAL_ERROR "PARAKEET_BUILD_EXECUTABLES requires PARAKEET_BUILD_LIBRARY=ON")
    endif()

    add_executable(parakeet-cli src/cli_main.cpp)
    target_link_libraries(parakeet-cli PRIVATE qvac-parakeet)
    set_target_properties(parakeet-cli PROPERTIES OUTPUT_NAME parakeet)
    parakeet_apply_ccache(parakeet-cli)

    add_executable(parakeet-fit-params src/fit_cli_main.cpp)
    target_link_libraries(parakeet-fit-params PRIVATE qvac-parakeet)
    parakeet_apply_ccache(parakeet-fit-params)

    # Backend parity diagnostic. Loads the same GGUF twice on two backends
    # (default cpu vs hexagon) and prints per-stage encoder cosine to
    # localize the first divergent Conformer stage on HTP0, and compares
    # Sortformer head probabilities on identical encoder input. Kept in the
    # executables block (not PARAKEET_BUILD_TESTS) so the Android
    # dynamic-loader build produces it without pulling in the whole
    # fixture harness.
    add_executable(test-encoder-backend-parity
        test/test_encoder_backend_parity.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_ctc.cpp
        src/parakeet_log.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-encoder-backend-parity PRIVATE ggml)
    target_include_directories(test-encoder-backend-parity
        PRIVATE ggml/include src include)
    if (PARAKEET_FLASH_ATTN)
        target_compile_definitions(test-encoder-backend-parity
            PRIVATE PARAKEET_EXPERIMENTAL_FLASH_ATTN)
    endif()
    if (PARAKEET_F16_ACTIVATIONS)
        target_compile_definitions(test-encoder-backend-parity
            PRIVATE PARAKEET_F16_ACTIVATIONS=1)
    endif()
    parakeet_apply_ccache(test-encoder-backend-parity)

    add_executable(moss-transcribe src/moss/transcribe_cli_main.cpp)
    target_link_libraries(moss-transcribe PRIVATE qvac-parakeet)
    parakeet_apply_ccache(moss-transcribe)
endif()

if (PARAKEET_BUILD_TESTS)
    enable_testing()
    include(CTest)

    # Roots that the test registrations below resolve fixture paths against.
    # All three are CACHE PATHs so CI / package builds can point them at
    # pre-staged mirrors (e.g. a tarball of `.gguf`s + reference `.npy`s)
    # without editing the listfile.
    #
    #   MODEL_DIR  - GGUF checkpoints (produced by scripts/convert-nemo-to-gguf.py)
    #   AUDIO_DIR  - 16 kHz mono WAV fixtures committed to test/samples/
    #   REF_DIR    - NeMo reference dumps (`.npy`) produced by
    #                scripts/dump-{ctc,tdt,eou,sortformer}-reference.py;
    #                expected layout is REF_DIR/{ctc,tdt,eou,sortformer}-ref/.
    #                Tests that need .npy refs auto-disable when the file
    #                isn't present, so a fresh checkout still gives a green
    #                ctest run on the model+wav harnesses.
    set(PARAKEET_TEST_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models"        CACHE PATH "parakeet: directory containing .gguf fixtures used by the test suite")
    set(PARAKEET_TEST_AUDIO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test/samples"  CACHE PATH "parakeet: directory containing .wav fixtures used by the test suite")
    set(PARAKEET_TEST_REF_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/artifacts"     CACHE PATH "parakeet: directory containing NeMo .npy reference dumps (scripts/dump-*-reference.py)")

    # Helper: register a built test target with CTest. Same shape as
    # the standard llama-style `llama_test()` macro: every harness
    # ends up surfaced via `ctest` against the build dir, with labels
    # that allow filtering (e.g. `ctest -L cpu` to skip GPU-only tests
    # on CI fleets without the matching backend). The optional REQUIRES
    # list names paths that must exist at configure time; if any is
    # missing the test is still registered (so it shows up in
    # `ctest -N`) but marked DISABLED so a missing reference dump
    # shows as "Not Run" rather than a failure.
    function(parakeet_register_test target)
        cmake_parse_arguments(QPRT "" "LABEL;EXE" "ARGS;REQUIRES" ${ARGN})
        if (NOT DEFINED QPRT_LABEL)
            set(QPRT_LABEL "cpu")
        endif()
        # EXE defaults to the test name when the test has its own
        # executable target. Pass an explicit EXE when several tests
        # share one binary and differ only in CLI args / fixtures
        # (e.g. test-encoder-capture-parity-{tdt,eou,sortformer} all
        # invoke test-encoder-capture-parity.exe).
        if (NOT DEFINED QPRT_EXE)
            set(QPRT_EXE ${target})
        endif()
        add_test(NAME ${target}
                 COMMAND $<TARGET_FILE:${QPRT_EXE}> ${QPRT_ARGS})
        set_property(TEST ${target} PROPERTY LABELS ${QPRT_LABEL})
        # A test that finds its backend without the feature under test exits 3.
        set_property(TEST ${target} PROPERTY SKIP_RETURN_CODE 3)
        if (WIN32)
            set(_qprt_test_path "${CMAKE_BINARY_DIR}/bin;$ENV{PATH}")
            if (PARAKEET_USE_SYSTEM_GGML AND GGML_LIBRARY)
                get_filename_component(_qprt_ggml_lib_dir "${GGML_LIBRARY}" DIRECTORY)
                get_filename_component(_qprt_ggml_prefix "${_qprt_ggml_lib_dir}" DIRECTORY)
                set(_qprt_test_path "${_qprt_ggml_prefix}/bin;${_qprt_test_path}")
            endif()
            string(REPLACE ";" "\\;" _qprt_test_path "${_qprt_test_path}")
            set_property(TEST ${target} PROPERTY ENVIRONMENT "PATH=${_qprt_test_path}")
        endif()
        set(_qprt_missing "")
        foreach (_qprt_req IN LISTS QPRT_REQUIRES)
            if (NOT EXISTS "${_qprt_req}")
                list(APPEND _qprt_missing "${_qprt_req}")
            endif()
        endforeach()
        if (_qprt_missing)
            set_property(TEST ${target} PROPERTY DISABLED TRUE)
            list(JOIN _qprt_missing "\n        " _qprt_missing_pretty)
            message(STATUS "parakeet: test ${target} disabled (missing fixture(s):\n        ${_qprt_missing_pretty})")
        endif()
    endfunction()

    # Helper: keep PARAKEET_EXPERIMENTAL_FLASH_ATTN consistent across the
    # parakeet library and any test executable that recompiles
    # parakeet_ctc.cpp from source. Backend selection itself goes
    # through the ggml-backend registry (no per-backend `GGML_USE_*`
    # #ifdef cascade in parakeet_ctc.cpp anymore -- see the comment on
    # `parakeet-backend-defs` above), so this helper only carries the
    # flash-attn gate plus the shared ccache launcher.
    function(parakeet_apply_backend_defs target)
        if (PARAKEET_FLASH_ATTN)
            target_compile_definitions(${target} PRIVATE PARAKEET_EXPERIMENTAL_FLASH_ATTN)
        endif()
        parakeet_apply_ccache(${target})
    endfunction()

    # Stable shorthands for the fixtures referenced by the registrations
    # below. Resolved at configure time; missing files turn into DISABLED
    # tests via parakeet_register_test(REQUIRES ...).
    #
    # q8_0 GGUFs are the runtime fixtures (smoke, streaming, perf): smaller
    # download, exercises the same code paths as production. f16 GGUFs are
    # the *parity* fixtures: their per-stage NeMo `.npy` references were
    # dumped from FP32 PyTorch, and even q8_0's well-behaved 0.5-1 % drift
    # swamps the per-block tolerance gates (3e-3 ... 5e-3) the parity
    # harnesses use to catch real operator bugs. So we point the parity
    # tests at the f16 GGUFs and leave q8_0 in place for everything else.
    set(_qvp_ctc_q8_gguf   "${PARAKEET_TEST_MODEL_DIR}/parakeet-ctc-0.6b.q8_0.gguf")
    set(_qvp_ctc_f16_gguf  "${PARAKEET_TEST_MODEL_DIR}/parakeet-ctc-0.6b.f16.gguf")
    set(_qvp_rnnt_q8_gguf  "${PARAKEET_TEST_MODEL_DIR}/parakeet-unified-en-0.6b.q8_0.gguf")
    set(_qvp_rnnt_coreml   "${PARAKEET_TEST_MODEL_DIR}/parakeet-unified-en-0.6b-encoder.mlmodelc")
    set(_qvp_rnnt_hybrid_q8_gguf "${PARAKEET_TEST_MODEL_DIR}/stt_ka_fastconformer_hybrid_large_pc.q8_0.gguf")
    set(_qvp_tdt_q8_gguf   "${PARAKEET_TEST_MODEL_DIR}/parakeet-tdt-0.6b-v3.q8_0.gguf")
    set(_qvp_tdt_f16_gguf  "${PARAKEET_TEST_MODEL_DIR}/parakeet-tdt-0.6b-v3.f16.gguf")
    set(_qvp_tdt_coreml     "${PARAKEET_TEST_MODEL_DIR}/parakeet-tdt-0.6b-v3-encoder.mlmodelc")
    set(_qvp_eou_q8_gguf   "${PARAKEET_TEST_MODEL_DIR}/parakeet_realtime_eou_120m-v1.q8_0.gguf")
    set(_qvp_eou_coreml    "${PARAKEET_TEST_MODEL_DIR}/parakeet_realtime_eou_120m-v1-encoder.mlmodelc")
    set(_qvp_nemotron_f16_gguf "${PARAKEET_TEST_MODEL_DIR}/nemotron-3.5-asr-streaming-0.6b.f16.gguf")
    set(_qvp_nemotron_coreml "${PARAKEET_TEST_MODEL_DIR}/nemotron-3.5-asr-streaming-0.6b-encoder.mlmodelc")
    set(_qvp_sf_q8_gguf    "${PARAKEET_TEST_MODEL_DIR}/diar_sortformer_4spk-v1.q8_0.gguf")
    set(_qvp_sf_f16_gguf   "${PARAKEET_TEST_MODEL_DIR}/diar_sortformer_4spk-v1.f16.gguf")
    set(_qvp_sfs_q8_gguf   "${PARAKEET_TEST_MODEL_DIR}/diar_streaming_sortformer_4spk-v2.q8_0.gguf")
    set(_qvp_sfsv21_q8_gguf "${PARAKEET_TEST_MODEL_DIR}/diar_streaming_sortformer_4spk-v2.1.q8_0.gguf")
    set(_qvp_sfsv21_f16_gguf "${PARAKEET_TEST_MODEL_DIR}/diar_streaming_sortformer_4spk-v2.1.f16.gguf")
    set(_qvp_sfsv21_coreml "${PARAKEET_TEST_MODEL_DIR}/diar_streaming_sortformer_4spk-v2.1-encoder.mlmodelc")
    set(_qvp_sfsv21_coreml_bypass "${PARAKEET_TEST_MODEL_DIR}/diar_streaming_sortformer_4spk-v2.1-encoder-bypass-pre-encode.mlmodelc")
    set(_qvp_indic_q8_gguf "${PARAKEET_TEST_MODEL_DIR}/indic-conformer-600m-multilingual.q8_0.gguf")
    set(_qvp_indic_q4_gguf "${PARAKEET_TEST_MODEL_DIR}/indic-conformer-600m-multilingual.q4_0.gguf")
    set(_qvp_indic_f16_gguf "${PARAKEET_TEST_MODEL_DIR}/indic-conformer-600m-multilingual.f16.gguf")
    set(_qvp_indic_coreml "${PARAKEET_TEST_MODEL_DIR}/indic-conformer-600m-multilingual-encoder.mlmodelc")
    set(_qvp_jfk_wav       "${PARAKEET_TEST_AUDIO_DIR}/jfk.wav")
    set(_qvp_diar_wav      "${PARAKEET_TEST_AUDIO_DIR}/diarization-sample-16k.wav")
    set(_qvp_fr_multipiece_wav "${PARAKEET_TEST_AUDIO_DIR}/fr-multipiece-16k.wav")
    set(_qvp_hi_wav        "${PARAKEET_TEST_AUDIO_DIR}/hi-16k.wav")
    set(_qvp_rnnt_hybrid_wav "${PARAKEET_TEST_AUDIO_DIR}/rnnt-ka-16k.wav")
    set(_qvp_hi_expected   "${PARAKEET_TEST_AUDIO_DIR}/hi-16k.expected.txt")
    set(_qvp_abcba_wav     "${PARAKEET_TEST_AUDIO_DIR}/abcba.wav")
    set(_qvp_abcba_rttm    "${PARAKEET_TEST_AUDIO_DIR}/abcba.rttm")
    set(_qvp_abcdba_wav    "${PARAKEET_TEST_AUDIO_DIR}/abcdba.wav")
    set(_qvp_abcdba_rttm   "${PARAKEET_TEST_AUDIO_DIR}/abcdba.rttm")
    # 6-minute mono 16 kHz PCM WAV, > k_enc_proj_T_max = 4096 encoder frames
    # (frame stride 80 ms => 4096 * 0.08 = 327.68 s ~= 5.5 min). Staged
    # optionally so tests that need to cross the window-split boundary
    # register but stay DISABLED on hosts without it.
    set(_qvp_speech_6min_wav "${PARAKEET_TEST_AUDIO_DIR}/speech-6min.wav")
    set(_qvp_ctc_ref       "${PARAKEET_TEST_REF_DIR}/ctc-ref")
    set(_qvp_tdt_ref       "${PARAKEET_TEST_REF_DIR}/tdt-ref")
    set(_qvp_rnnt_hybrid_ref "${PARAKEET_TEST_REF_DIR}/rnnt-ref")
    set(_qvp_sf_ref        "${PARAKEET_TEST_REF_DIR}/sortformer-ref")
    set(_qvp_nemotron_en_ref
        "${PARAKEET_TEST_REF_DIR}/nemotron-ref-en-us-320/offline")
    set(_qvp_nemotron_auto_ref
        "${PARAKEET_TEST_REF_DIR}/nemotron-ref-auto-320/offline")
    set(_qvp_nemotron_hi_ref
        "${PARAKEET_TEST_REF_DIR}/nemotron-ref-hi-in-320/offline")
    set(_qvp_nemotron_stream_ref
        "${PARAKEET_TEST_REF_DIR}/nemotron-ref-en-us-320")
    set(_qvp_nemotron_stream_80_ref
        "${PARAKEET_TEST_REF_DIR}/nemotron-ref-en-us-80")
    set(_qvp_nemotron_stream_160_ref
        "${PARAKEET_TEST_REF_DIR}/nemotron-ref-en-us-160")
    set(_qvp_nemotron_stream_560_ref
        "${PARAKEET_TEST_REF_DIR}/nemotron-ref-en-us-560")
    set(_qvp_nemotron_stream_1120_ref
        "${PARAKEET_TEST_REF_DIR}/nemotron-ref-en-us-1120")
    set(_qvp_unified_ref "${PARAKEET_TEST_REF_DIR}/unified-ref-jfk-70-7-7")

    add_executable(test-mel
        test/test_mel.cpp
        src/mel_preprocess.cpp
        src/parakeet_ctc.cpp
        src/parakeet_log.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-mel PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-mel PRIVATE ggml/include src include)
    parakeet_apply_backend_defs(test-mel)
    parakeet_register_test(test-mel
        LABEL    "fixture"
        ARGS     "${_qvp_ctc_f16_gguf}" "${_qvp_jfk_wav}" "${_qvp_ctc_ref}/mel.npy"
        REQUIRES "${_qvp_ctc_f16_gguf}" "${_qvp_jfk_wav}" "${_qvp_ctc_ref}/mel.npy")

    add_executable(test-encoder
        test/test_encoder.cpp
        src/parakeet_ctc.cpp
        src/parakeet_log.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-encoder PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-encoder PRIVATE ggml/include src include)
    parakeet_apply_backend_defs(test-encoder)
    parakeet_register_test(test-encoder
        LABEL    "fixture"
        ARGS     "${_qvp_ctc_f16_gguf}" "${_qvp_ctc_ref}"
        REQUIRES "${_qvp_ctc_f16_gguf}" "${_qvp_ctc_ref}/mel.npy" "${_qvp_ctc_ref}/encoder_out.npy")

    add_executable(test-ctc
        test/test_ctc.cpp
        src/parakeet_ctc.cpp
        src/parakeet_log.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-ctc PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-ctc PRIVATE ggml/include src include)
    parakeet_apply_backend_defs(test-ctc)
    parakeet_register_test(test-ctc
        LABEL    "fixture"
        ARGS     "${_qvp_ctc_q8_gguf}" "${_qvp_ctc_ref}/logits.npy" "${_qvp_ctc_ref}/decoded.txt"
        REQUIRES "${_qvp_ctc_q8_gguf}" "${_qvp_ctc_ref}/logits.npy" "${_qvp_ctc_ref}/decoded.txt")

    # Multilingual CTC language-mask unit tests (synthetic logits; no GGUF).
    add_executable(test-ctc-language-mask
        test/test_ctc_language_mask.cpp
        src/parakeet_ctc.cpp
        src/parakeet_log.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-ctc-language-mask PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-ctc-language-mask PRIVATE ggml/include src include)
    parakeet_apply_backend_defs(test-ctc-language-mask)
    parakeet_register_test(test-ctc-language-mask LABEL "unit")

    # GPU-tier selection unit tests. Pure ranking function
    # (parakeet::gpu_tier_for) against synthesised device tuples — no ggml
    # backends actually loaded, no GGUF fixtures, no CUDA/Vulkan drivers
    # required; runs on every host in the matrix.
    #
    # Target name is `test-parakeet-backend-selection` (not
    # `test-backend-selection`) because engines/tts registers its own
    # `test-backend-selection` and the two collide when the umbrella
    # build pulls both engines into a single CMake tree.
    add_executable(test-parakeet-backend-selection test/test_backend_selection.cpp)
    target_link_libraries(test-parakeet-backend-selection PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-parakeet-backend-selection PRIVATE ggml/include src include)
    parakeet_apply_ccache(test-parakeet-backend-selection)
    parakeet_register_test(test-parakeet-backend-selection LABEL "unit")

    add_executable(test-parakeet-backend-request test/test_backend_request.cpp)
    target_link_libraries(test-parakeet-backend-request PRIVATE qvac-parakeet)
    parakeet_apply_ccache(test-parakeet-backend-request)
    parakeet_register_test(test-parakeet-backend-request LABEL "unit")

    add_executable(test-parakeet-streaming test/test_streaming.cpp)
    target_link_libraries(test-parakeet-streaming PRIVATE qvac-parakeet)
    target_include_directories(test-parakeet-streaming PRIVATE include src ggml/include)
    parakeet_apply_ccache(test-parakeet-streaming)
    parakeet_register_test(test-parakeet-streaming
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_ctc_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_ctc_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-streaming-rnnt
        LABEL    "fixture"
        EXE      test-parakeet-streaming
        ARGS     "--model" "${_qvp_rnnt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}")

    # Quantized TDT and EOU decoded on Hexagon HTP0. Hexagon GET_ROWS accepts
    # F32 only, so backend_runs_embedding_lookup() rejects the Q8_0
    # predict_embed and tdt_prepare_runtime / eou_prepare_runtime allocate an
    # F32 copy (see parakeet_tdt.cpp:1282, parakeet_eou.cpp:732). These tests
    # exercise that F32 fallback end to end: transcribe (Mode 1) must produce
    # non-empty text and match the streaming (Mode 2) tokenization across the
    # standard chunk_ms sweep. Auto-skip via SKIP_RETURN_CODE on hosts
    # without HTP0.
    parakeet_register_test(test-streaming-tdt-hexagon
        LABEL    "fixture"
        EXE      test-parakeet-streaming
        ARGS     "--model" "${_qvp_tdt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
                 "--backend" "hexagon"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-streaming-eou-hexagon
        LABEL    "fixture"
        EXE      test-parakeet-streaming
        ARGS     "--model" "${_qvp_eou_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
                 "--backend" "hexagon"
        REQUIRES "${_qvp_eou_q8_gguf}" "${_qvp_jfk_wav}")

    # Long-form offline TDT and EOU that cross k_enc_proj_T_max = 4096
    # encoder frames. tdt_decode_graph and eou_decode_graph split the encoded
    # recording into bounded projection windows while preserving predictor
    # state and (for TDT) the duration carry across each split; without a
    # test above 4096 frames, a state-carry bug at the boundary would go
    # unnoticed. speech-6min.wav is ~5.5 min => ~4500 encoder frames, which
    # forces one window transition. Auto-DISABLED when the fixture is not
    # staged. See parakeet_tdt.cpp:1878 and parakeet_eou.cpp for the split.
    parakeet_register_test(test-streaming-tdt-long-form
        LABEL    "fixture"
        EXE      test-parakeet-streaming
        ARGS     "--model" "${_qvp_tdt_q8_gguf}" "--wav" "${_qvp_speech_6min_wav}"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_speech_6min_wav}")
    parakeet_register_test(test-streaming-eou-long-form
        LABEL    "fixture"
        EXE      test-parakeet-streaming
        ARGS     "--model" "${_qvp_eou_q8_gguf}" "--wav" "${_qvp_speech_6min_wav}"
        REQUIRES "${_qvp_eou_q8_gguf}" "${_qvp_speech_6min_wav}")

    add_executable(test-unified-streaming test/test_unified_streaming.cpp)
    target_link_libraries(test-unified-streaming PRIVATE qvac-parakeet)
    target_include_directories(test-unified-streaming PRIVATE include src ggml/include)
    parakeet_apply_ccache(test-unified-streaming)
    parakeet_register_test(test-unified-streaming
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_rnnt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}")

    add_executable(test-unified-stream-step
        test/test_unified_stream_step.cpp
        src/parakeet_ctc.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-unified-stream-step PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-unified-stream-step PRIVATE src include ggml/include)
    parakeet_apply_backend_defs(test-unified-stream-step)

    function(parakeet_register_unified_stream_case test_name)
        set(_qvp_unified_stream_label "fixture")
        if (ARGN)
            set(_qvp_unified_stream_label "gpu")
        endif()
        parakeet_register_test(${test_name}
            EXE test-unified-stream-step
            LABEL "${_qvp_unified_stream_label}"
            ARGS     "${_qvp_rnnt_q8_gguf}" "${_qvp_unified_ref}" "${_qvp_jfk_wav}" ${ARGN}
            REQUIRES "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}"
                     "${_qvp_unified_ref}/buffered/step-000/encoder_chunk.npy"
                     "${_qvp_unified_ref}/buffered/transcript.txt")
    endfunction()

    parakeet_register_unified_stream_case(test-unified-stream-step)
    if (GGML_METAL)
        parakeet_register_unified_stream_case(test-unified-stream-step-metal --n-gpu-layers 1)
    endif()
    if (GGML_CUDA)
        parakeet_register_unified_stream_case(test-unified-stream-step-cuda --n-gpu-layers 1)
    endif()
    if (GGML_VULKAN)
        parakeet_register_unified_stream_case(test-unified-stream-step-vulkan --n-gpu-layers 1)
    endif()

    add_executable(test-eou-streaming test/test_eou_streaming.cpp)
    target_link_libraries(test-eou-streaming PRIVATE qvac-parakeet)
    target_include_directories(test-eou-streaming PRIVATE include src ggml/include)
    parakeet_apply_ccache(test-eou-streaming)
    parakeet_register_test(test-eou-streaming
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_eou_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_eou_q8_gguf}" "${_qvp_jfk_wav}")

    # Model-independent finalization state regression. This runs on every
    # checkout even when the streaming Sortformer v2.1 fixture is not staged.
    add_executable(test-sortformer-finalize test/test_sortformer_finalize.cpp)
    target_include_directories(test-sortformer-finalize PRIVATE src)
    parakeet_apply_ccache(test-sortformer-finalize)
    parakeet_register_test(test-sortformer-finalize LABEL "unit")

    # Probability-thresholding logic shared by the offline and AOSC
    # diarization paths (header-only, no GGUF).
    add_executable(test-sortformer-segments test/test_sortformer_segments.cpp)
    target_include_directories(test-sortformer-segments PRIVATE src)
    parakeet_apply_ccache(test-sortformer-segments)
    parakeet_register_test(test-sortformer-segments LABEL "unit")

    # RMS energy VAD state machine used by the CTC/TDT streaming sessions.
    add_executable(test-energy-vad
        test/test_energy_vad.cpp
        src/energy_vad.cpp)
    target_include_directories(test-energy-vad PRIVATE src)
    parakeet_apply_ccache(test-energy-vad)
    parakeet_register_test(test-energy-vad LABEL "unit")

    # SentencePiece detokenizer + starts_word stamping (synthetic vocab).
    add_executable(test-sentencepiece-bpe
        test/test_sentencepiece_bpe.cpp
        src/sentencepiece_bpe.cpp)
    target_include_directories(test-sentencepiece-bpe PRIVATE src)
    parakeet_apply_ccache(test-sentencepiece-bpe)
    parakeet_register_test(test-sentencepiece-bpe LABEL "unit")

    add_executable(test-sortformer-streaming test/test_sortformer_streaming.cpp)
    target_link_libraries(test-sortformer-streaming PRIVATE qvac-parakeet)
    target_include_directories(test-sortformer-streaming PRIVATE include src ggml/include)
    parakeet_apply_ccache(test-sortformer-streaming)
    parakeet_register_test(test-sortformer-streaming
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_sfsv21_q8_gguf}" "--wav" "${_qvp_diar_wav}"
        REQUIRES "${_qvp_sfsv21_q8_gguf}" "${_qvp_diar_wav}")

    # v2.1 AOSC speaker-correctness regression. Asserts speaker coverage,
    # re-entry slot continuity (the AOSC contract), and frame-level DER
    # ceiling against the RTTM ground truth. One binary, two ctest
    # registrations (one per LIFO re-entry fixture).
    add_executable(test-sortformer-aosc-speakers test/test_sortformer_aosc_speakers.cpp)
    target_link_libraries(test-sortformer-aosc-speakers PRIVATE qvac-parakeet)
    target_include_directories(test-sortformer-aosc-speakers PRIVATE include src ggml/include)
    parakeet_apply_ccache(test-sortformer-aosc-speakers)
    parakeet_register_test(test-sortformer-aosc-speakers-abcba
        LABEL    "fixture"
        EXE      test-sortformer-aosc-speakers
        ARGS     "--model" "${_qvp_sfsv21_q8_gguf}"
                 "--wav"   "${_qvp_abcba_wav}"
                 "--ref-rttm" "${_qvp_abcba_rttm}"
        REQUIRES "${_qvp_sfsv21_q8_gguf}" "${_qvp_abcba_wav}" "${_qvp_abcba_rttm}")
    parakeet_register_test(test-sortformer-aosc-speakers-abcdba
        LABEL    "fixture"
        EXE      test-sortformer-aosc-speakers
        ARGS     "--model" "${_qvp_sfsv21_q8_gguf}"
                 "--wav"   "${_qvp_abcdba_wav}"
                 "--ref-rttm" "${_qvp_abcdba_rttm}"
        REQUIRES "${_qvp_sfsv21_q8_gguf}" "${_qvp_abcdba_wav}" "${_qvp_abcdba_rttm}")

    # test-encoder-backend-parity (built above, outside PARAKEET_BUILD_TESTS)
    # is registered here so ctest exercises the per-stage CPU-vs-Hexagon
    # encoder cosine gate on every Parakeet family. The harness auto-skips
    # (returns 3, treated as SKIPPED via SKIP_RETURN_CODE) on builds without
    # the requested backend, so a CPU-only host build shows these as "Not
    # Run" rather than failing.
    #
    # Coverage of PARAKEET_F16_ACTIVATIONS: the compile-time flag drives the
    # F16 pad+cast helpers in parakeet_ctc.cpp. When the flag is ON (default
    # after f409940a) these tests exercise the padded attention boundary
    # against CPU. When the flag is OFF the helpers become no-ops and the
    # same tests structurally validate the F32 fast path. To cover BOTH
    # settings in one CI matrix, build twice: once with
    # `-DPARAKEET_F16_ACTIVATIONS=ON` and once with `OFF`, then run
    # `ctest -L parity`.
    if (TARGET test-encoder-backend-parity)
        parakeet_register_test(test-encoder-backend-parity-ctc
            LABEL    "parity"
            EXE      test-encoder-backend-parity
            ARGS     "--model" "${_qvp_ctc_f16_gguf}" "--wav" "${_qvp_jfk_wav}"
                     "--backend-a" "cpu" "--backend-b" "hexagon" "--min-cos" "0.999"
            REQUIRES "${_qvp_ctc_f16_gguf}" "${_qvp_jfk_wav}")
        parakeet_register_test(test-encoder-backend-parity-tdt
            LABEL    "parity"
            EXE      test-encoder-backend-parity
            ARGS     "--model" "${_qvp_tdt_f16_gguf}" "--wav" "${_qvp_jfk_wav}"
                     "--backend-a" "cpu" "--backend-b" "hexagon" "--min-cos" "0.999"
            REQUIRES "${_qvp_tdt_f16_gguf}" "${_qvp_jfk_wav}")
        parakeet_register_test(test-encoder-backend-parity-eou
            LABEL    "parity"
            EXE      test-encoder-backend-parity
            ARGS     "--model" "${_qvp_eou_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
                     "--backend-a" "cpu" "--backend-b" "hexagon" "--min-cos" "0.995"
            REQUIRES "${_qvp_eou_q8_gguf}" "${_qvp_jfk_wav}")
        parakeet_register_test(test-encoder-backend-parity-sortformer
            LABEL    "parity"
            EXE      test-encoder-backend-parity
            ARGS     "--model" "${_qvp_sf_f16_gguf}" "--wav" "${_qvp_diar_wav}"
                     "--backend-a" "cpu" "--backend-b" "hexagon" "--min-cos" "0.999"
            REQUIRES "${_qvp_sf_f16_gguf}" "${_qvp_diar_wav}")
    endif()

    add_executable(test-perf-regression test/test_perf_regression.cpp)
    target_link_libraries(test-perf-regression PRIVATE qvac-parakeet)
    target_include_directories(test-perf-regression PRIVATE include src ggml/include)
    parakeet_apply_ccache(test-perf-regression)
    parakeet_register_test(test-perf-regression
        LABEL    "perf"
        ARGS     "--model" "${_qvp_ctc_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_ctc_q8_gguf}" "${_qvp_jfk_wav}")
    # Sortformer dispatch path for the perf-regression
    # harness. Same gates (cache-hit ratio, cold-overhead, mel-ms ceiling)
    # as transcribe-path; the only difference is that
    # `engine.diarize(wav)` produces a `DiarizationResult` instead of an
    # `EngineResult`, and the deterministic comparison runs over a
    # segment-list fingerprint rather than the transcript text.
    parakeet_register_test(test-perf-regression-sortformer
        LABEL    "perf"
        EXE      test-perf-regression
        ARGS     "--model" "${_qvp_sf_q8_gguf}" "--wav" "${_qvp_diar_wav}"
        REQUIRES "${_qvp_sf_q8_gguf}" "${_qvp_diar_wav}")

    # Internal regression gate for the round-2 mel preprocess sweep
    # (real-FFT, MelState reuse). Pure unit test -- no GGUF needed.
    add_executable(test-mel-fft-parity
        test/test_mel_fft_parity.cpp
        src/mel_preprocess.cpp
        src/parakeet_log.cpp)
    target_link_libraries(test-mel-fft-parity PRIVATE ggml)
    target_include_directories(test-mel-fft-parity PRIVATE ggml/include src include)
    parakeet_apply_backend_defs(test-mel-fft-parity)
    parakeet_register_test(test-mel-fft-parity LABEL "unit")

    # Threaded mel front-end must be byte-equal to the single-thread result.
    add_executable(test-mel-threads
        test/test_mel_threads.cpp
        src/mel_preprocess.cpp
        src/parakeet_log.cpp)
    target_link_libraries(test-mel-threads PRIVATE ggml)
    target_include_directories(test-mel-threads PRIVATE ggml/include src include)
    parakeet_apply_backend_defs(test-mel-threads)
    parakeet_register_test(test-mel-threads LABEL "unit")

    add_executable(test-mel-bands
        test/test_mel_bands.cpp
        src/mel_preprocess.cpp
        src/parakeet_log.cpp)
    target_link_libraries(test-mel-bands PRIVATE ggml)
    target_include_directories(test-mel-bands PRIVATE ggml/include src include)
    parakeet_apply_backend_defs(test-mel-bands)
    parakeet_register_test(test-mel-bands LABEL "unit")

    # test_rnnt.cpp exercises internal decoder entry points
    # (rnnt_decode_window, build_transducer_argmax_outputs) that are not
    # exported from the shared library (CXX_VISIBILITY_PRESET hidden), so
    # compile the sources in directly like test-tdt-decoder-parity.
    add_executable(test-rnnt
        test/test_rnnt.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-rnnt PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-rnnt PRIVATE src include ggml/include)
    parakeet_apply_backend_defs(test-rnnt)
    parakeet_register_test(test-rnnt LABEL "unit")

    add_executable(test-unified-loader
        test/test_unified_loader.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-unified-loader PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-unified-loader PRIVATE src include ggml/include)
    parakeet_apply_backend_defs(test-unified-loader)
    parakeet_register_test(test-unified-loader
        LABEL    "fixture"
        ARGS     "${_qvp_rnnt_q8_gguf}"
        REQUIRES "${_qvp_rnnt_q8_gguf}")

    add_executable(test-nemotron-loader
        test/test_nemotron_loader.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-nemotron-loader PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-nemotron-loader PRIVATE src include ggml/include)
    parakeet_apply_backend_defs(test-nemotron-loader)
    parakeet_register_test(test-nemotron-loader
        LABEL    "fixture"
        ARGS     "${_qvp_nemotron_f16_gguf}"
        REQUIRES "${_qvp_nemotron_f16_gguf}")

    set(PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF
        "${PARAKEET_TEST_MODEL_DIR}/Nemotron-3-Diarization.q8_0.gguf"
        CACHE FILEPATH "Nemotron 3 Diarization GGUF fixture")
    add_executable(test-nemotron-diarization-loader
        test/test_nemotron_diarization_loader.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-nemotron-diarization-loader PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-nemotron-diarization-loader PRIVATE src include ggml/include)
    parakeet_apply_backend_defs(test-nemotron-diarization-loader)
    parakeet_register_test(test-nemotron-diarization-loader
        LABEL "fixture"
        ARGS "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}"
        REQUIRES "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}")

    add_executable(test-nemotron-diarization
        test/test_nemotron_diarization.cpp)
    target_link_libraries(test-nemotron-diarization PRIVATE qvac-parakeet)
    target_include_directories(test-nemotron-diarization PRIVATE src include)
    parakeet_apply_ccache(test-nemotron-diarization)
    parakeet_register_test(test-nemotron-diarization
        LABEL "fixture"
        ARGS "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}"
             "${CMAKE_CURRENT_SOURCE_DIR}/test/samples/diarization-sample-16k.wav"
             "${CMAKE_CURRENT_SOURCE_DIR}/test/reference/nemotron-3-diarization-sample-f32.bin"
        REQUIRES "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/samples/diarization-sample-16k.wav"
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/reference/nemotron-3-diarization-sample-f32.bin")

    set(PARAKEET_TEST_GPU_LAYERS 99)
    set(PARAKEET_TEST_LONG_FORM_DER_MAX 15.0)
    parakeet_register_test(test-nemotron-diarization-long-form-abcba
        LABEL    "fixture"
        EXE      test-sortformer-aosc-speakers
        ARGS     "--model" "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}"
                 "--wav"   "${_qvp_abcba_wav}"
                 "--ref-rttm" "${_qvp_abcba_rttm}"
                 "--offline" "--der-max" "${PARAKEET_TEST_LONG_FORM_DER_MAX}"
        REQUIRES "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}" "${_qvp_abcba_wav}" "${_qvp_abcba_rttm}")
    parakeet_register_test(test-nemotron-diarization-long-form-abcdba
        LABEL    "fixture"
        EXE      test-sortformer-aosc-speakers
        ARGS     "--model" "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}"
                 "--wav"   "${_qvp_abcdba_wav}"
                 "--ref-rttm" "${_qvp_abcdba_rttm}"
                 "--offline" "--der-max" "${PARAKEET_TEST_LONG_FORM_DER_MAX}"
        REQUIRES "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}" "${_qvp_abcdba_wav}" "${_qvp_abcdba_rttm}")

    parakeet_register_test(test-nemotron-diarization-gpu
        EXE test-nemotron-diarization
        LABEL "gpu"
        ARGS "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}"
             "${CMAKE_CURRENT_SOURCE_DIR}/test/samples/diarization-sample-16k.wav"
             "${CMAKE_CURRENT_SOURCE_DIR}/test/reference/nemotron-3-diarization-sample-f32.bin"
             "${PARAKEET_TEST_GPU_LAYERS}"
        REQUIRES "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/samples/diarization-sample-16k.wav"
                 "${CMAKE_CURRENT_SOURCE_DIR}/test/reference/nemotron-3-diarization-sample-f32.bin")
    parakeet_register_test(test-nemotron-diarization-long-form-abcdba-gpu
        LABEL    "gpu"
        EXE      test-sortformer-aosc-speakers
        ARGS     "--model" "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}"
                 "--wav"   "${_qvp_abcdba_wav}"
                 "--ref-rttm" "${_qvp_abcdba_rttm}"
                 "--offline" "--der-max" "${PARAKEET_TEST_LONG_FORM_DER_MAX}"
                 "--n-gpu-layers" "${PARAKEET_TEST_GPU_LAYERS}"
        REQUIRES "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}" "${_qvp_abcdba_wav}" "${_qvp_abcdba_rttm}")

    add_executable(test-nemotron
        test/test_nemotron.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-nemotron PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-nemotron PRIVATE src include ggml/include)
    parakeet_apply_backend_defs(test-nemotron)
    parakeet_register_test(test-nemotron
        LABEL "fixture"
        ARGS
            "${_qvp_nemotron_f16_gguf}"
            "${_qvp_jfk_wav}"
            "${_qvp_nemotron_en_ref}"
            "${_qvp_nemotron_auto_ref}"
            "${_qvp_hi_wav}"
            "${_qvp_nemotron_hi_ref}"
        REQUIRES
            "${_qvp_nemotron_f16_gguf}"
            "${_qvp_jfk_wav}"
            "${_qvp_nemotron_en_ref}/mel.npy"
            "${_qvp_nemotron_en_ref}/encoder_raw.npy"
            "${_qvp_nemotron_en_ref}/prompt_output.npy"
            "${_qvp_nemotron_en_ref}/token_ids.npy"
            "${_qvp_nemotron_en_ref}/transcript.txt"
            "${_qvp_nemotron_auto_ref}/token_ids.npy"
            "${_qvp_nemotron_auto_ref}/transcript.txt"
            "${_qvp_hi_wav}"
            "${_qvp_nemotron_hi_ref}/token_ids.npy"
            "${_qvp_nemotron_hi_ref}/transcript.txt")

    add_executable(test-nemotron-stream-step
        test/test_nemotron_stream_step.cpp
        src/parakeet_ctc.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-nemotron-stream-step PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-nemotron-stream-step PRIVATE src include ggml/include)
    parakeet_apply_backend_defs(test-nemotron-stream-step)
    function(parakeet_register_nemotron_stream_case test_name reference_dir right_context)
        set(_qvp_nemotron_stream_label "fixture")
        if (ARGN)
            set(_qvp_nemotron_stream_label "gpu")
        endif()
        parakeet_register_test(${test_name}
            EXE test-nemotron-stream-step
            LABEL "${_qvp_nemotron_stream_label}"
            ARGS
                "${_qvp_nemotron_f16_gguf}"
                "${reference_dir}"
                "${right_context}"
                "${_qvp_jfk_wav}"
                ${ARGN}
            REQUIRES
                "${_qvp_nemotron_f16_gguf}"
                "${reference_dir}/step-000/processed_signal.npy"
                "${reference_dir}/step-000/encoder_raw.npy"
                "${reference_dir}/step-000/cache_channel_after.npy"
                "${reference_dir}/step-000/cache_time_after.npy"
                "${reference_dir}/step-001/processed_signal.npy"
                "${reference_dir}/step-001/encoder_raw.npy"
                "${reference_dir}/step-001/cache_channel_after.npy"
                "${reference_dir}/step-001/cache_time_after.npy"
                "${reference_dir}/offline/mel.npy"
                "${reference_dir}/transcript.txt"
                "${_qvp_jfk_wav}")
    endfunction()
    parakeet_register_nemotron_stream_case(
        test-nemotron-stream-step-80 "${_qvp_nemotron_stream_80_ref}" 0)
    parakeet_register_nemotron_stream_case(
        test-nemotron-stream-step-160 "${_qvp_nemotron_stream_160_ref}" 1)
    parakeet_register_nemotron_stream_case(
        test-nemotron-stream-step "${_qvp_nemotron_stream_ref}" 3)
    parakeet_register_nemotron_stream_case(
        test-nemotron-stream-step-560 "${_qvp_nemotron_stream_560_ref}" 6)
    parakeet_register_nemotron_stream_case(
        test-nemotron-stream-step-1120 "${_qvp_nemotron_stream_1120_ref}" 13)

    if (GGML_METAL)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-metal
            "${_qvp_nemotron_stream_ref}" 3 --n-gpu-layers 1)
    endif()

    if (GGML_CUDA)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-cuda-80
            "${_qvp_nemotron_stream_80_ref}" 0 --n-gpu-layers 1)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-cuda-160
            "${_qvp_nemotron_stream_160_ref}" 1 --n-gpu-layers 1)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-cuda
            "${_qvp_nemotron_stream_ref}" 3 --n-gpu-layers 1)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-cuda-560
            "${_qvp_nemotron_stream_560_ref}" 6 --n-gpu-layers 1)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-cuda-1120
            "${_qvp_nemotron_stream_1120_ref}" 13 --n-gpu-layers 1)
    endif()

    if (GGML_VULKAN)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-vulkan-80
            "${_qvp_nemotron_stream_80_ref}" 0 --n-gpu-layers 1)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-vulkan-160
            "${_qvp_nemotron_stream_160_ref}" 1 --n-gpu-layers 1)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-vulkan
            "${_qvp_nemotron_stream_ref}" 3 --n-gpu-layers 1)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-vulkan-560
            "${_qvp_nemotron_stream_560_ref}" 6 --n-gpu-layers 1)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-vulkan-1120
            "${_qvp_nemotron_stream_1120_ref}" 13 --n-gpu-layers 1)
    endif()

    if (GGML_OPENCL)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-opencl-80
            "${_qvp_nemotron_stream_80_ref}" 0 --n-gpu-layers 1)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-opencl-160
            "${_qvp_nemotron_stream_160_ref}" 1 --n-gpu-layers 1)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-opencl
            "${_qvp_nemotron_stream_ref}" 3 --n-gpu-layers 1)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-opencl-560
            "${_qvp_nemotron_stream_560_ref}" 6 --n-gpu-layers 1)
        parakeet_register_nemotron_stream_case(
            test-nemotron-stream-step-opencl-1120
            "${_qvp_nemotron_stream_1120_ref}" 13 --n-gpu-layers 1)
    endif()

    set(PARAKEET_MOSS_TEST_SOURCES
        ${PARAKEET_MOSS_SOURCES}
        src/parakeet_coreml_path.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)

    set(_qvp_moss_arms "")
    if (GGML_CUDA)
        list(APPEND _qvp_moss_arms cuda)
    endif()
    if (GGML_VULKAN)
        list(APPEND _qvp_moss_arms vulkan)
    endif()
    if (_qvp_moss_arms)
        add_executable(test-moss-transcribe-gpu test/moss/test_transcribe_gpu.cpp ${PARAKEET_MOSS_TEST_SOURCES})
        target_link_libraries(test-moss-transcribe-gpu PRIVATE ggml parakeet-backend-defs)
        target_include_directories(test-moss-transcribe-gpu PRIVATE src include ggml/include)
        parakeet_apply_backend_defs(test-moss-transcribe-gpu)
        parakeet_apply_ccache(test-moss-transcribe-gpu)
        foreach(_qvp_moss_arm IN LISTS _qvp_moss_arms)
            parakeet_register_test(test-moss-transcribe-${_qvp_moss_arm} LABEL "gpu"
                EXE test-moss-transcribe-gpu ARGS ${_qvp_moss_arm})
        endforeach()
    endif()

    add_executable(test-moss-transcribe test/moss/test_transcribe.cpp ${PARAKEET_MOSS_TEST_SOURCES})
    target_link_libraries(test-moss-transcribe PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-moss-transcribe PRIVATE src include ggml/include)
    target_compile_options(test-moss-transcribe PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
    parakeet_apply_backend_defs(test-moss-transcribe)
    parakeet_apply_ccache(test-moss-transcribe)
    parakeet_register_test(test-moss-transcribe LABEL "unit")

    add_executable(test-moss-transcribe-fit test/moss/test_transcribe_fit.cpp ${PARAKEET_MOSS_TEST_SOURCES})
    target_link_libraries(test-moss-transcribe-fit PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-moss-transcribe-fit PRIVATE src include ggml/include)
    target_compile_options(test-moss-transcribe-fit PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
    parakeet_apply_backend_defs(test-moss-transcribe-fit)
    parakeet_apply_ccache(test-moss-transcribe-fit)
    parakeet_register_test(test-moss-transcribe-fit LABEL "unit")

    add_executable(test-moss-transcribe-parity test/moss/test_transcribe_parity.cpp ${PARAKEET_MOSS_TEST_SOURCES})
    target_link_libraries(test-moss-transcribe-parity PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-moss-transcribe-parity PRIVATE src include ggml/include)
    parakeet_apply_backend_defs(test-moss-transcribe-parity)
    parakeet_apply_ccache(test-moss-transcribe-parity)
    parakeet_register_test(test-moss-transcribe-parity LABEL "fixture")
    set_property(TEST test-moss-transcribe-parity PROPERTY SKIP_RETURN_CODE 77)

    # Self-skips (77) without PARAKEET_COREML or MOSS_TRANSCRIBE_COREML_MODEL
    # (a GGUF with its -encoder.mlmodelc staged beside it).
    add_executable(test-moss-transcribe-coreml-parity test/moss/test_transcribe_coreml_parity.cpp
        ${PARAKEET_MOSS_TEST_SOURCES} src/coreml/parakeet_coreml_shape.cpp)
    target_link_libraries(test-moss-transcribe-coreml-parity PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-moss-transcribe-coreml-parity PRIVATE src include ggml/include)
    parakeet_apply_backend_defs(test-moss-transcribe-coreml-parity)
    parakeet_apply_ccache(test-moss-transcribe-coreml-parity)
    if (PARAKEET_COREML)
        find_library(PARAKEET_FOUNDATION_FRAMEWORK Foundation REQUIRED)
        find_library(PARAKEET_COREML_FRAMEWORK     CoreML     REQUIRED)
        target_sources(test-moss-transcribe-coreml-parity PRIVATE src/coreml/parakeet-encoder.mm)
        set_source_files_properties(src/coreml/parakeet-encoder.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
        target_compile_definitions(test-moss-transcribe-coreml-parity PRIVATE PARAKEET_USE_COREML)
        target_link_libraries(test-moss-transcribe-coreml-parity PRIVATE
            ${PARAKEET_FOUNDATION_FRAMEWORK} ${PARAKEET_COREML_FRAMEWORK})
    endif()
    parakeet_register_test(test-moss-transcribe-coreml-parity
        LABEL "fixture;coreml"
        ARGS  "${CMAKE_CURRENT_SOURCE_DIR}/test/samples/diarization-sample-16k.wav")
    set_property(TEST test-moss-transcribe-coreml-parity PROPERTY SKIP_RETURN_CODE 77)

    # Not a test: times one 30 s window on the staged sidecar against the
    # ggml GPU encoder (median of 5 after a warm-up) as a markdown row.
    add_executable(bench-moss-transcribe-coreml test/moss/bench_transcribe_coreml.cpp
        ${PARAKEET_MOSS_TEST_SOURCES} src/coreml/parakeet_coreml_shape.cpp)
    target_link_libraries(bench-moss-transcribe-coreml PRIVATE ggml parakeet-backend-defs)
    target_include_directories(bench-moss-transcribe-coreml PRIVATE src include ggml/include)
    parakeet_apply_backend_defs(bench-moss-transcribe-coreml)
    parakeet_apply_ccache(bench-moss-transcribe-coreml)
    if (PARAKEET_COREML)
        target_sources(bench-moss-transcribe-coreml PRIVATE src/coreml/parakeet-encoder.mm)
        target_compile_definitions(bench-moss-transcribe-coreml PRIVATE PARAKEET_USE_COREML)
        target_link_libraries(bench-moss-transcribe-coreml PRIVATE
            ${PARAKEET_FOUNDATION_FRAMEWORK} ${PARAKEET_COREML_FRAMEWORK})
    endif()

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        add_test(
            NAME test-converter-rnnt
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_convert_nemo_to_gguf.py")
        set_property(TEST test-converter-rnnt PROPERTY LABELS unit)
        add_test(
            NAME test-converter-moss-transcribe
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_convert_moss_transcribe.py")
        set_property(TEST test-converter-moss-transcribe PROPERTY LABELS unit)
        set_property(TEST test-converter-moss-transcribe PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
        set_property(TEST test-converter-moss-transcribe PROPERTY SKIP_REGULAR_EXPRESSION "SKIP: ")
    endif()

    find_program(PARAKEET_BASH_EXECUTABLE bash)
    if (PARAKEET_BASH_EXECUTABLE)
        add_test(
            NAME test-download-all-models
            COMMAND ${PARAKEET_BASH_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_download_all_models.sh")
        set_property(TEST test-download-all-models PROPERTY LABELS unit)
    endif()

    # Regression gate for the round-1 capture_intermediates flag.
    # Asserts run_encoder(capture=true) and (capture=false) produce
    # bit-equal encoder_out + logits on a CTC GGUF.
    add_executable(test-encoder-capture-parity
        test/test_encoder_capture_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_log.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-encoder-capture-parity PRIVATE ggml)
    target_include_directories(test-encoder-capture-parity PRIVATE ggml/include src include)
    parakeet_apply_backend_defs(test-encoder-capture-parity)
    parakeet_register_test(test-encoder-capture-parity
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_ctc_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_ctc_q8_gguf}" "${_qvp_jfk_wav}")
    # Same capture-parity contract on TDT/EOU/Sortformer. Each shares
    # the test-encoder-capture-parity exe and gets auto-disabled by
    # parakeet_register_test() when the relevant GGUF isn't present,
    # so the CTC variant stays enabled even when the others aren't.
    parakeet_register_test(test-encoder-capture-parity-tdt
        LABEL    "fixture"
        EXE      test-encoder-capture-parity
        ARGS     "--model" "${_qvp_tdt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-encoder-capture-parity-rnnt
        LABEL    "fixture"
        EXE      test-encoder-capture-parity
        ARGS     "--model" "${_qvp_rnnt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-encoder-capture-parity-eou
        LABEL    "fixture"
        EXE      test-encoder-capture-parity
        ARGS     "--model" "${_qvp_eou_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_eou_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-encoder-capture-parity-sortformer
        LABEL    "fixture"
        EXE      test-encoder-capture-parity
        ARGS     "--model" "${_qvp_sf_q8_gguf}" "--wav" "${_qvp_diar_wav}"
        REQUIRES "${_qvp_sf_q8_gguf}" "${_qvp_diar_wav}")

    # Core ML encoder parity gate. Compiles run_encoder directly (internal symbol)
    # and compares the Apple Neural Engine encoder against the ggml encoder via
    # per-frame cosine similarity. Skips at runtime when the Core ML sidecar isn't
    # active, so it is a no-op on non-Apple / no-sidecar CI and a real gate on Apple.
    # When PARAKEET_COREML is on, the ObjC++ wrapper + CoreML/Foundation frameworks
    # are compiled/linked into the harness so the directly-compiled run_encoder gets
    # its Core ML branch.
    add_executable(test-encoder-coreml-parity
        test/test_encoder_coreml_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_coreml_path.cpp
        src/coreml/parakeet_coreml_shape.cpp
        src/parakeet_log.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-encoder-coreml-parity PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-encoder-coreml-parity PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-encoder-coreml-parity)
    if (PARAKEET_COREML)
        find_library(PARAKEET_FOUNDATION_FRAMEWORK Foundation REQUIRED)
        find_library(PARAKEET_COREML_FRAMEWORK     CoreML     REQUIRED)
        target_sources(test-encoder-coreml-parity PRIVATE src/coreml/parakeet-encoder.mm)
        set_source_files_properties(src/coreml/parakeet-encoder.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
        target_compile_definitions(test-encoder-coreml-parity PRIVATE PARAKEET_USE_COREML)
        target_link_libraries(test-encoder-coreml-parity PRIVATE
            ${PARAKEET_FOUNDATION_FRAMEWORK} ${PARAKEET_COREML_FRAMEWORK})
    endif()
    parakeet_register_test(test-encoder-coreml-parity
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_tdt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-encoder-coreml-parity-rnnt
        LABEL    "fixture"
        EXE      test-encoder-coreml-parity
        ARGS     "--model" "${_qvp_rnnt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}" "${_qvp_rnnt_coreml}")
    parakeet_register_test(test-encoder-coreml-parity-eou
        LABEL    "fixture"
        EXE      test-encoder-coreml-parity
        ARGS     "--model" "${_qvp_eou_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_eou_q8_gguf}" "${_qvp_jfk_wav}" "${_qvp_eou_coreml}")
    parakeet_register_test(test-encoder-coreml-parity-nemotron
        LABEL    "fixture;coreml"
        EXE      test-encoder-coreml-parity
        ARGS     --model "${_qvp_nemotron_f16_gguf}" --wav "${_qvp_jfk_wav}" --min-cos 0.995
        REQUIRES "${_qvp_nemotron_f16_gguf}" "${_qvp_jfk_wav}" "${_qvp_nemotron_coreml}")
    parakeet_register_test(test-encoder-coreml-parity-indic
        LABEL    "fixture;coreml"
        EXE      test-encoder-coreml-parity
        ARGS     --model "${_qvp_indic_f16_gguf}" --wav "${_qvp_hi_wav}" --min-cos 0.995
        REQUIRES "${_qvp_indic_f16_gguf}" "${_qvp_hi_wav}" "${_qvp_indic_coreml}")
    add_executable(test-encoder-coreml-bypass-parity
        test/test_encoder_coreml_bypass_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_coreml_path.cpp
        src/coreml/parakeet_coreml_shape.cpp
        src/parakeet_log.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-encoder-coreml-bypass-parity PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-encoder-coreml-bypass-parity PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-encoder-coreml-bypass-parity)
    if (PARAKEET_COREML)
        target_sources(test-encoder-coreml-bypass-parity PRIVATE src/coreml/parakeet-encoder.mm)
        set_source_files_properties(src/coreml/parakeet-encoder.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
        target_compile_definitions(test-encoder-coreml-bypass-parity PRIVATE PARAKEET_USE_COREML)
        target_link_libraries(test-encoder-coreml-bypass-parity PRIVATE
            ${PARAKEET_FOUNDATION_FRAMEWORK} ${PARAKEET_COREML_FRAMEWORK})
    endif()
    parakeet_register_test(test-encoder-coreml-bypass-parity
        LABEL    "fixture"
        ARGS     "${_qvp_sfsv21_f16_gguf}"
        REQUIRES "${_qvp_sfsv21_f16_gguf}" "${_qvp_sfsv21_coreml_bypass}")

    # Portable unit test for the Core ML sidecar path derivation
    # (coreml_encoder_sidecar_path). Pure string logic -- no ggml, model, or
    # Apple frameworks -- so it runs on every CI regardless of PARAKEET_COREML
    # and locks the `<model>-encoder.mlmodelc` naming (quant tag stripped).
    add_executable(test-coreml-paths
        test/test_coreml_paths.cpp
        src/parakeet_coreml_path.cpp)
    target_include_directories(test-coreml-paths PRIVATE src)
    parakeet_apply_ccache(test-coreml-paths)
    parakeet_register_test(test-coreml-paths LABEL "cpu")

    # Portable unit test for the Core ML input-shape resolution
    # (coreml_resolve_input_dims). Pure integer logic -- no ggml, model, or Apple
    # frameworks -- so it runs on every CI regardless of PARAKEET_COREML and locks
    # the wrapper's variable-length behaviour (fixed verbatim / flexible rebuild /
    # features-major default).
    add_executable(test-coreml-shapes
        test/test_coreml_shapes.cpp
        src/coreml/parakeet_coreml_shape.cpp)
    target_include_directories(test-coreml-shapes PRIVATE src)
    parakeet_apply_ccache(test-coreml-shapes)
    parakeet_register_test(test-coreml-shapes LABEL "cpu")

    if (Python3_Interpreter_FOUND)
        add_test(
            NAME test-export-encoder-coreml
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_export_encoder_coreml.py")
        set_property(TEST test-export-encoder-coreml PROPERTY LABELS "unit")
        add_test(
            NAME test-export-moss-transcribe-coreml
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_export_moss_transcribe_coreml.py")
        set_property(TEST test-export-moss-transcribe-coreml PROPERTY LABELS "unit")
        set_property(TEST test-export-moss-transcribe-coreml PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
        set_property(TEST test-export-moss-transcribe-coreml PROPERTY SKIP_REGULAR_EXPRESSION "SKIP: ")
    endif()

    # End-to-end transcription parity: full Engine::transcribe() with the Core ML
    # encoder vs the ggml encoder (forced via PARAKEET_COREML_DISABLE) must yield
    # identical text. Links libqvac-parakeet for the public Engine API. Skips at runtime
    # when the Core ML sidecar is inactive, so it is a no-op on non-Apple /
    # no-sidecar CI and a real gate on Apple with a sidecar present.
    if (TARGET qvac-parakeet)
        add_executable(test-transcribe-coreml-parity
            test/test_transcribe_coreml_parity.cpp)
        target_link_libraries(test-transcribe-coreml-parity PRIVATE qvac-parakeet)
        target_include_directories(test-transcribe-coreml-parity PRIVATE include)
        parakeet_apply_ccache(test-transcribe-coreml-parity)
        parakeet_register_test(test-transcribe-coreml-parity
            LABEL    "fixture"
            ARGS     "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}"
            REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}")
        parakeet_register_test(test-transcribe-coreml-parity-rnnt
            LABEL    "fixture"
            EXE      test-transcribe-coreml-parity
            ARGS     "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}"
            REQUIRES "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}" "${_qvp_rnnt_coreml}")

        # Drive Unified RNN-T beyond the 1501-mel-frame sidecar capacity. The
        # test proves the fixed-capacity long-form plan is selected, every
        # encoder window stays on Core ML, and its token stream remains close
        # to the forced-ggml reference.
        add_executable(test-rnnt-coreml-long-form
            test/test_rnnt_coreml_long_form.cpp)
        target_link_libraries(test-rnnt-coreml-long-form PRIVATE qvac-parakeet)
        target_include_directories(test-rnnt-coreml-long-form PRIVATE include)
        parakeet_apply_ccache(test-rnnt-coreml-long-form)
        parakeet_register_test(test-rnnt-coreml-long-form
            LABEL    "fixture"
            ARGS     "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}"
                     "1501" "16" "0.95" "windowed-coreml"
            REQUIRES "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}" "${_qvp_rnnt_coreml}")
        parakeet_register_test(test-nemotron-cache-aware-offline
            LABEL    "fixture;coreml"
            EXE      test-rnnt-coreml-long-form
            ARGS     "${_qvp_nemotron_f16_gguf}" "${_qvp_jfk_wav}"
                     "1101" "16" "0.85" "cache-aware"
            REQUIRES "${_qvp_nemotron_f16_gguf}" "${_qvp_jfk_wav}" "${_qvp_nemotron_coreml}")
        parakeet_register_test(test-transcribe-coreml-parity-eou
            LABEL    "fixture"
            EXE      test-transcribe-coreml-parity
            ARGS     "${_qvp_eou_q8_gguf}" "${_qvp_jfk_wav}"
            REQUIRES "${_qvp_eou_q8_gguf}" "${_qvp_jfk_wav}" "${_qvp_eou_coreml}")
        parakeet_register_test(test-transcribe-coreml-parity-indic
            LABEL    "fixture;coreml"
            EXE      test-transcribe-coreml-parity
            ARGS     "${_qvp_indic_f16_gguf}" "${_qvp_hi_wav}" "hi"
            REQUIRES "${_qvp_indic_f16_gguf}" "${_qvp_hi_wav}" "${_qvp_indic_coreml}")

        add_executable(test-diarize-coreml-short-parity
            test/test_diarize_coreml_short_parity.cpp)
        target_link_libraries(test-diarize-coreml-short-parity PRIVATE qvac-parakeet)
        target_include_directories(test-diarize-coreml-short-parity PRIVATE include)
        parakeet_apply_ccache(test-diarize-coreml-short-parity)
        parakeet_register_test(test-diarize-coreml-short-parity
            LABEL    "fixture"
            ARGS     "${_qvp_sfsv21_f16_gguf}"
            REQUIRES "${_qvp_sfsv21_f16_gguf}" "${_qvp_sfsv21_coreml}")
    endif()

    # CLI-level Core ML gate: --require-coreml must prove every measured encoder
    # invocation used the expected ASR sidecar while the decoder remains on ggml.
    if (PARAKEET_COREML AND Python3_Interpreter_FOUND AND TARGET parakeet-cli)
        add_test(
            NAME test-cli-coreml-bench
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_cli_coreml_bench.py"
                    "--binary" "$<TARGET_FILE:parakeet-cli>"
                    "--model" "${_qvp_tdt_q8_gguf}"
                    "--wav" "${_qvp_jfk_wav}")
        set_property(TEST test-cli-coreml-bench PROPERTY LABELS "fixture;coreml")
        if (NOT EXISTS "${_qvp_tdt_q8_gguf}" OR
            NOT EXISTS "${_qvp_jfk_wav}" OR
            NOT EXISTS "${_qvp_tdt_coreml}")
            set_property(TEST test-cli-coreml-bench PROPERTY DISABLED TRUE)
        endif()
        add_test(
            NAME test-cli-coreml-bench-eou
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_cli_coreml_bench.py"
                    "--binary" "$<TARGET_FILE:parakeet-cli>"
                    "--model" "${_qvp_eou_q8_gguf}"
                    "--wav" "${_qvp_jfk_wav}")
        set_property(TEST test-cli-coreml-bench-eou PROPERTY LABELS "fixture;coreml")
        if (NOT EXISTS "${_qvp_eou_q8_gguf}" OR
            NOT EXISTS "${_qvp_jfk_wav}" OR
            NOT EXISTS "${_qvp_eou_coreml}")
            set_property(TEST test-cli-coreml-bench-eou PROPERTY DISABLED TRUE)
        endif()
        add_test(
            NAME test-cli-coreml-bench-rnnt
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_cli_coreml_bench.py"
                    "--binary" "$<TARGET_FILE:parakeet-cli>"
                    "--model" "${_qvp_rnnt_q8_gguf}"
                    "--wav" "${_qvp_jfk_wav}")
        set_property(TEST test-cli-coreml-bench-rnnt PROPERTY LABELS "fixture;coreml")
        if (NOT EXISTS "${_qvp_rnnt_q8_gguf}" OR
            NOT EXISTS "${_qvp_jfk_wav}" OR
            NOT EXISTS "${_qvp_rnnt_coreml}")
            set_property(TEST test-cli-coreml-bench-rnnt PROPERTY DISABLED TRUE)
        endif()
        add_test(
            NAME test-cli-coreml-bench-nemotron
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_cli_coreml_bench.py"
                    "--binary" "$<TARGET_FILE:parakeet-cli>"
                    "--model" "${_qvp_nemotron_f16_gguf}"
                    "--wav" "${_qvp_jfk_wav}")
        set_property(TEST test-cli-coreml-bench-nemotron PROPERTY LABELS "fixture;coreml")
        if (NOT EXISTS "${_qvp_nemotron_f16_gguf}" OR
            NOT EXISTS "${_qvp_jfk_wav}" OR
            NOT EXISTS "${_qvp_nemotron_coreml}")
            set_property(TEST test-cli-coreml-bench-nemotron PROPERTY DISABLED TRUE)
        endif()
        add_test(
            NAME test-cli-coreml-bench-indic
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/test/test_cli_coreml_bench.py"
                    "--binary" "$<TARGET_FILE:parakeet-cli>"
                    "--model" "${_qvp_indic_f16_gguf}"
                    "--wav" "${_qvp_hi_wav}"
                    "--language" "hi")
        set_property(TEST test-cli-coreml-bench-indic PROPERTY LABELS "fixture;coreml")
        if (NOT EXISTS "${_qvp_indic_f16_gguf}" OR
            NOT EXISTS "${_qvp_hi_wav}" OR
            NOT EXISTS "${_qvp_indic_coreml}")
            set_property(TEST test-cli-coreml-bench-indic PROPERTY DISABLED TRUE)
        endif()
    endif()

    # Regression gate for cached-encoder-graph reuse determinism. Compiles the
    # sources directly (run_encoder is internal, not exported). Runs run_encoder 20x
    # on the same cached graph and asserts the output stays byte-identical across
    # every reuse; the -gpu variant locks the Adreno OpenCL/Vulkan reuse path that
    # regressed under the shared scheduler and is fixed by the persistent gallocr.
    add_executable(test-sched-encoder-determinism
        test/test_sched_encoder_determinism.cpp
        src/parakeet_ctc.cpp
        src/parakeet_log.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-sched-encoder-determinism PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-sched-encoder-determinism PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-sched-encoder-determinism)
    parakeet_register_test(test-sched-encoder-determinism
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_ctc_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_ctc_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-sched-encoder-determinism-gpu
        LABEL    "gpu"
        EXE      test-sched-encoder-determinism
        ARGS     "--model" "${_qvp_ctc_q8_gguf}" "--wav" "${_qvp_jfk_wav}" "--n-gpu-layers" "1"
        REQUIRES "${_qvp_ctc_q8_gguf}" "${_qvp_jfk_wav}")

    # Regression gate for the shared-scheduler lifecycle: subsampling -> encoder ->
    # Sortformer head through one sched must match a clean encoder -> head reference,
    # byte-identical, across reuse. Locks reset-at-head / download-before-reset order.
    add_executable(test-sched-diarize-lifecycle
        test/test_sched_diarize_lifecycle.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-sched-diarize-lifecycle PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-sched-diarize-lifecycle PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-sched-diarize-lifecycle)
    parakeet_register_test(test-sched-diarize-lifecycle
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_sf_q8_gguf}" "--wav" "${_qvp_diar_wav}"
        REQUIRES "${_qvp_sf_q8_gguf}" "${_qvp_diar_wav}")
    parakeet_register_test(test-sched-diarize-lifecycle-gpu
        LABEL    "gpu"
        EXE      test-sched-diarize-lifecycle
        ARGS     "--model" "${_qvp_sf_q8_gguf}" "--wav" "${_qvp_diar_wav}" "--n-gpu-layers" "1"
        REQUIRES "${_qvp_sf_q8_gguf}" "${_qvp_diar_wav}")

    # The subsampling runs on the valid mel prefix and zero-pads its output: a zero-tailed mel
    # must match the truncated mel on the valid frames and hold the projection bias after them.
    add_executable(test-subsampling-tail
        test/test_subsampling_tail.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-subsampling-tail PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-subsampling-tail PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-subsampling-tail)
    parakeet_register_test(test-subsampling-tail
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_tdt_q8_gguf}"
        REQUIRES "${_qvp_tdt_q8_gguf}")
    parakeet_register_test(test-subsampling-tail-gpu
        LABEL    "gpu"
        EXE      test-subsampling-tail
        ARGS     "--model" "${_qvp_tdt_q8_gguf}" "--n-gpu-layers" "1"
        REQUIRES "${_qvp_tdt_q8_gguf}")

    add_executable(test-tdt-encoder-parity
        test/test_tdt_encoder_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-tdt-encoder-parity PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-tdt-encoder-parity PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-tdt-encoder-parity)
    parakeet_register_test(test-tdt-encoder-parity
        LABEL    "fixture"
        ARGS     "${_qvp_tdt_f16_gguf}" "${_qvp_jfk_wav}" "${_qvp_tdt_ref}"
        REQUIRES "${_qvp_tdt_f16_gguf}" "${_qvp_jfk_wav}" "${_qvp_tdt_ref}/encoder_out.npy")

    add_executable(test-tdt-decoder-parity
        test/test_tdt_decoder_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-tdt-decoder-parity PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-tdt-decoder-parity PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-tdt-decoder-parity)
    parakeet_register_test(test-tdt-decoder-parity
        LABEL    "fixture"
        ARGS     "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}" "${_qvp_tdt_ref}"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-rnnt-decoder-parity
        LABEL    "fixture"
        EXE      test-tdt-decoder-parity
        ARGS     "${_qvp_rnnt_hybrid_q8_gguf}" "${_qvp_rnnt_hybrid_wav}"
                 "${_qvp_rnnt_hybrid_ref}" "--require-reference"
        REQUIRES "${_qvp_rnnt_hybrid_q8_gguf}" "${_qvp_rnnt_hybrid_wav}"
                 "${_qvp_rnnt_hybrid_ref}/token_ids.npy")
    # Cached positional projections must not change the encoder output.
    add_executable(test-pos-proj-cache-parity
        test/test_pos_proj_cache_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-pos-proj-cache-parity PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-pos-proj-cache-parity PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-pos-proj-cache-parity)
    parakeet_register_test(test-pos-proj-cache-parity
        LABEL    "fixture"
        ARGS     "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-pos-proj-cache-parity-gpu
        LABEL    "gpu"
        EXE      test-pos-proj-cache-parity
        ARGS     "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}" "1"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}")
    # Load-time [w_ih | w_hh] row stacking, independent of any GGUF fixture.
    add_executable(test-lstm-input-concat test/test_lstm_input_concat.cpp)
    target_link_libraries(test-lstm-input-concat PRIVATE ggml)
    target_include_directories(test-lstm-input-concat PRIVATE src ggml/include)
    parakeet_apply_ccache(test-lstm-input-concat)
    parakeet_register_test(test-lstm-input-concat LABEL "unit")

    # Fused attention is a build option; only CUDA and Metal take it, and the
    # encoder graph built on CPU must never contain the fused node.
    add_executable(test-attn-path
        test/test_attn_path.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-attn-path PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-attn-path PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-attn-path)
    parakeet_register_test(test-attn-path LABEL "unit")
    parakeet_register_test(test-attn-path-graph
        LABEL    "fixture"
        EXE      test-attn-path
        ARGS     "${_qvp_tdt_q8_gguf}"
        REQUIRES "${_qvp_tdt_q8_gguf}")
    parakeet_register_test(test-attn-path-graph-gpu
        LABEL    "gpu"
        EXE      test-attn-path
        ARGS     "${_qvp_tdt_q8_gguf}" "--n-gpu-layers" "1"
        REQUIRES "${_qvp_tdt_q8_gguf}")

    # The channel-contiguous depthwise-conv view needs two or more frames.
    add_executable(test-conv-channels-view test/test_conv_channels_view.cpp)
    target_link_libraries(test-conv-channels-view PRIVATE ggml)
    target_include_directories(test-conv-channels-view PRIVATE src ggml/include)
    parakeet_apply_ccache(test-conv-channels-view)
    parakeet_register_test(test-conv-channels-view LABEL "unit")

    # The fused LSTM cell op and the concatenated-input GEMV must both reproduce
    # the per-gate decoder graph token for token.
    add_executable(test-tdt-lstm-parity
        test/test_tdt_lstm_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-tdt-lstm-parity PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-tdt-lstm-parity PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-tdt-lstm-parity)
    parakeet_register_test(test-tdt-lstm-parity-gpu
        LABEL    "gpu"
        EXE      test-tdt-lstm-parity
        ARGS     "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}" "--n-gpu-layers" "1"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}")
    # The unrolled decode graph (K greedy steps per commit) must reproduce the
    # sequential per-step loop token for token, at K=1 and K=8.
    add_executable(test-tdt-unroll-parity
        test/test_tdt_unroll_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-tdt-unroll-parity PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-tdt-unroll-parity PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-tdt-unroll-parity)
    parakeet_register_test(test-tdt-unroll-parity-gpu
        LABEL    "gpu"
        EXE      test-tdt-unroll-parity
        ARGS     "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}" "--n-gpu-layers" "1" "--require-unrolled"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}")
    # CPU decode has no graph path, so this one guards that the unroll setter
    # leaves the sequential loop alone.
    parakeet_register_test(test-tdt-unroll-parity
        LABEL    "fixture"
        EXE      test-tdt-unroll-parity
        ARGS     "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}")
    # Multi-piece French guard for the on-device predictor: the gpu0
    # scalar path and the gpu1 graph path must produce identical token IDs on a
    # word with multiple word-final sub-words (e.g. "analyses"). English jfk does
    # not exercise this, and the divergence only surfaces when the graph path runs
    # on a real GPU -> gpu tier. No ref-dir: this is a CPU-vs-graph cross-check.
    parakeet_register_test(test-tdt-decoder-parity-fr-gpu
        LABEL    "gpu"
        EXE      test-tdt-decoder-parity
        ARGS     "${_qvp_tdt_q8_gguf}" "${_qvp_fr_multipiece_wav}"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_fr_multipiece_wav}")

    # Fit-projection parity (include/parakeet/fit.h): metadata-only projection
    # vs the buffers a real load/encode actually allocates, on the CPU backend
    # (plus the GPU backend for Nemotron 3 Diarization). One binary,
    # registered once per model family fixture present.
    add_executable(test-fit-params
        test/test_fit_params.cpp
        src/parakeet_ctc.cpp
        src/parakeet_fit.cpp
        src/parakeet_log.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-fit-params PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-fit-params PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-fit-params)
    parakeet_register_test(test-fit-params
        LABEL    "fixture"
        ARGS     "${_qvp_rnnt_q8_gguf}"
        REQUIRES "${_qvp_rnnt_q8_gguf}")
    parakeet_register_test(test-fit-params-ctc
        LABEL    "fixture"
        EXE      test-fit-params
        ARGS     "${_qvp_ctc_q8_gguf}"
        REQUIRES "${_qvp_ctc_q8_gguf}")
    parakeet_register_test(test-fit-params-tdt
        LABEL    "fixture"
        EXE      test-fit-params
        ARGS     "${_qvp_tdt_q8_gguf}"
        REQUIRES "${_qvp_tdt_q8_gguf}")
    parakeet_register_test(test-fit-params-eou
        LABEL    "fixture"
        EXE      test-fit-params
        ARGS     "${_qvp_eou_q8_gguf}"
        REQUIRES "${_qvp_eou_q8_gguf}")
    parakeet_register_test(test-fit-params-sortformer
        LABEL    "fixture"
        EXE      test-fit-params
        ARGS     "${_qvp_sf_q8_gguf}"
        REQUIRES "${_qvp_sf_q8_gguf}")
    parakeet_register_test(test-fit-params-nemotron
        LABEL    "fixture"
        EXE      test-fit-params
        ARGS     "${_qvp_nemotron_f16_gguf}"
        REQUIRES "${_qvp_nemotron_f16_gguf}")
    parakeet_register_test(test-fit-params-nemotron-diarization
        LABEL    "fixture"
        EXE      test-fit-params
        ARGS     "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}"
        REQUIRES "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}")
    parakeet_register_test(test-fit-params-nemotron-diarization-gpu
        LABEL    "gpu"
        EXE      test-fit-params
        ARGS     "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}" "${PARAKEET_TEST_GPU_LAYERS}"
        REQUIRES "${PARAKEET_TEST_NEMOTRON_DIARIZATION_GGUF}")

    # Model-free CLI-layer unit tests: strict parsers, saturating arithmetic,
    # JSON escaping, and the parakeet-fit-params exit-code contract.
    add_executable(test-fit-cli
        test/test_fit_cli.cpp
        src/fit_main.cpp
        src/parakeet_ctc.cpp
        src/parakeet_fit.cpp
        src/parakeet_log.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp)
    target_link_libraries(test-fit-cli PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-fit-cli PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-fit-cli)
    parakeet_register_test(test-fit-cli LABEL "unit")

    add_executable(test-eou-decoder-parity
        test/test_eou_decoder_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-eou-decoder-parity PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-eou-decoder-parity PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-eou-decoder-parity)
    parakeet_register_test(test-eou-decoder-parity
        LABEL    "fixture"
        ARGS     "${_qvp_eou_q8_gguf}" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_eou_q8_gguf}" "${_qvp_jfk_wav}")
    # Multi-utterance guard for the on-device EOU decode (span-batched joint +
    # <EOU> state reset): abcba is long multi-speaker audio that emits several
    # mid-file <EOU> boundaries, so the gpu1 graph path must reproduce the gpu0
    # host path's token IDs AND segment splits exactly. The divergence risk only
    # surfaces when the graph path runs on a real GPU -> gpu tier.
    parakeet_register_test(test-eou-decoder-parity-multiutt-gpu
        LABEL    "gpu"
        EXE      test-eou-decoder-parity
        ARGS     "${_qvp_eou_q8_gguf}" "${_qvp_abcba_wav}"
        REQUIRES "${_qvp_eou_q8_gguf}" "${_qvp_abcba_wav}")

    add_executable(test-sortformer-parity
        test/test_sortformer_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_nemotron.cpp
        src/parakeet_unified.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
        src/parakeet_diarization_v3.cpp
        src/mel_preprocess.cpp
        src/sentencepiece_bpe.cpp
        src/energy_vad.cpp)
    target_link_libraries(test-sortformer-parity PRIVATE ggml parakeet-backend-defs)
    target_include_directories(test-sortformer-parity PRIVATE include src ggml/include)
    parakeet_apply_backend_defs(test-sortformer-parity)
    parakeet_register_test(test-sortformer-parity
        LABEL    "fixture"
        ARGS     "${_qvp_sf_f16_gguf}" "${_qvp_diar_wav}" "${_qvp_sf_ref}"
        REQUIRES "${_qvp_sf_f16_gguf}" "${_qvp_diar_wav}" "${_qvp_sf_ref}/encoder_out.npy")

    # Decoder determinism gate. Runs each decoder N times
    # against the same encoder output and asserts byte-equal results
    # across all repetitions. Catches buffer-reuse correctness bugs
    # (residual scratch state leaking into the next call) that the
    # existing per-stage parity tests (which load+decode once) wouldn't
    # see. Runs against any model type; switches to diarize() when the
    # GGUF declares Sortformer.
    add_executable(test-decoder-determinism test/test_decoder_determinism.cpp)
    target_link_libraries(test-decoder-determinism PRIVATE qvac-parakeet)
    target_include_directories(test-decoder-determinism PRIVATE include src ggml/include)
    parakeet_apply_ccache(test-decoder-determinism)
    parakeet_register_test(test-decoder-determinism
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_ctc_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_ctc_q8_gguf}" "${_qvp_jfk_wav}")
    # Per-model-type instances so ctest exercises each path. Each is
    # individually DISABLED via parakeet_register_test() when its GGUF
    # isn't present, so a fresh checkout with only ctc.q8_0.gguf cached
    # still gives a green ctest.
    parakeet_register_test(test-decoder-determinism-tdt
        LABEL    "fixture"
        EXE      test-decoder-determinism
        ARGS     "--model" "${_qvp_tdt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-decoder-determinism-rnnt
        LABEL    "fixture"
        EXE      test-decoder-determinism
        ARGS     "--model" "${_qvp_rnnt_q8_gguf}" "--wav" "${_qvp_jfk_wav}" "--threads" "4"
        REQUIRES "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-decoder-determinism-eou
        LABEL    "fixture"
        EXE      test-decoder-determinism
        ARGS     "--model" "${_qvp_eou_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_eou_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-decoder-determinism-sortformer
        LABEL    "fixture"
        EXE      test-decoder-determinism
        ARGS     "--model" "${_qvp_sf_q8_gguf}" "--wav" "${_qvp_diar_wav}"
        REQUIRES "${_qvp_sf_q8_gguf}" "${_qvp_diar_wav}")
    parakeet_register_test(test-decoder-determinism-indic
        LABEL    "fixture"
        EXE      test-decoder-determinism
        ARGS     "--model" "${_qvp_indic_q8_gguf}" "--wav" "${_qvp_hi_wav}"
                 "--language" "hi" "--expect-text-file" "${_qvp_hi_expected}"
        REQUIRES "${_qvp_indic_q8_gguf}" "${_qvp_hi_wav}" "${_qvp_hi_expected}")
    parakeet_register_test(test-decoder-determinism-indic-q4
        LABEL    "fixture"
        EXE      test-decoder-determinism
        ARGS     "--model" "${_qvp_indic_q4_gguf}" "--wav" "${_qvp_hi_wav}"
                 "--language" "hi" "--expect-text-file" "${_qvp_hi_expected}"
        REQUIRES "${_qvp_indic_q4_gguf}" "${_qvp_hi_wav}" "${_qvp_hi_expected}")
    parakeet_register_test(test-decoder-determinism-indic-f16
        LABEL    "fixture"
        EXE      test-decoder-determinism
        ARGS     "--model" "${_qvp_indic_f16_gguf}" "--wav" "${_qvp_hi_wav}"
                 "--language" "hi" "--expect-text-file" "${_qvp_hi_expected}"
        REQUIRES "${_qvp_indic_f16_gguf}" "${_qvp_hi_wav}" "${_qvp_hi_expected}")

    # Long-form encoder windowing. The model-free unit test (planner + resolver
    # + seam trim/append) runs on every checkout; the parity test is
    # fixture-gated and auto-disables when the GGUF/wav aren't staged.
    add_executable(test-long-form-windows test/test_long_form_windows.cpp)
    target_include_directories(test-long-form-windows PRIVATE src include)
    parakeet_apply_ccache(test-long-form-windows)
    parakeet_register_test(test-long-form-windows LABEL "unit")

    add_executable(test-long-form test/test_long_form.cpp)
    target_link_libraries(test-long-form PRIVATE qvac-parakeet)
    target_include_directories(test-long-form PRIVATE include src ggml/include)
    parakeet_apply_ccache(test-long-form)
    parakeet_register_test(test-long-form
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_ctc_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_ctc_q8_gguf}" "${_qvp_jfk_wav}")
    parakeet_register_test(test-long-form-tdt
        LABEL    "fixture"
        EXE      test-long-form
        ARGS     "--model" "${_qvp_tdt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_tdt_q8_gguf}" "${_qvp_jfk_wav}")

    if (GGML_VULKAN OR GGML_METAL OR GGML_CUDA)
        add_executable(test-gpu-vs-cpu
            test/test_gpu_vs_cpu.cpp
            src/parakeet_ctc.cpp
            src/parakeet_log.cpp
            src/mel_preprocess.cpp
            src/sentencepiece_bpe.cpp)
        target_link_libraries(test-gpu-vs-cpu PRIVATE ggml parakeet-backend-defs)
        target_include_directories(test-gpu-vs-cpu PRIVATE ggml/include src include)
        parakeet_apply_backend_defs(test-gpu-vs-cpu)
        parakeet_register_test(test-gpu-vs-cpu
            LABEL    "gpu"
            ARGS     "${_qvp_ctc_q8_gguf}" "${_qvp_jfk_wav}"
            REQUIRES "${_qvp_ctc_q8_gguf}" "${_qvp_jfk_wav}")
        parakeet_register_test(test-gpu-vs-cpu-indic
            LABEL    "gpu"
            EXE      test-gpu-vs-cpu
            ARGS     "${_qvp_indic_q8_gguf}" "${_qvp_hi_wav}" "hi"
            REQUIRES "${_qvp_indic_q8_gguf}" "${_qvp_hi_wav}")
        parakeet_register_test(test-gpu-vs-cpu-indic-f16
            LABEL    "gpu"
            EXE      test-gpu-vs-cpu
            ARGS     "${_qvp_indic_f16_gguf}" "${_qvp_hi_wav}" "hi"
            REQUIRES "${_qvp_indic_f16_gguf}" "${_qvp_hi_wav}")
        parakeet_register_test(test-decoder-determinism-indic-gpu
            LABEL    "gpu"
            EXE      test-decoder-determinism
            ARGS     "--model" "${_qvp_indic_q8_gguf}" "--wav" "${_qvp_hi_wav}"
                     "--language" "hi" "--n-gpu-layers" "1" "--require-gpu"
                     "--expect-text-file" "${_qvp_hi_expected}"
            REQUIRES "${_qvp_indic_q8_gguf}" "${_qvp_hi_wav}" "${_qvp_hi_expected}")
        parakeet_register_test(test-decoder-determinism-indic-f16-gpu
            LABEL    "gpu"
            EXE      test-decoder-determinism
            ARGS     "--model" "${_qvp_indic_f16_gguf}" "--wav" "${_qvp_hi_wav}"
                     "--language" "hi" "--n-gpu-layers" "1" "--require-gpu"
                     "--expect-text-file" "${_qvp_hi_expected}"
            REQUIRES "${_qvp_indic_f16_gguf}" "${_qvp_hi_wav}" "${_qvp_hi_expected}")
        parakeet_register_test(test-decoder-determinism-indic-q4-gpu
            LABEL    "gpu"
            EXE      test-decoder-determinism
            ARGS     "--model" "${_qvp_indic_q4_gguf}" "--wav" "${_qvp_hi_wav}"
                     "--language" "hi" "--n-gpu-layers" "1" "--require-gpu"
                     "--expect-text-file" "${_qvp_hi_expected}"
            REQUIRES "${_qvp_indic_q4_gguf}" "${_qvp_hi_wav}" "${_qvp_hi_expected}")
    endif()

    if (GGML_VULKAN)
        parakeet_register_test(test-streaming-rnnt-vulkan
            LABEL    "gpu"
            EXE      test-parakeet-streaming
            ARGS     "--model" "${_qvp_rnnt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
                     "--threads" "4" "--n-gpu-layers" "1"
            REQUIRES "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}")
        parakeet_register_test(test-decoder-determinism-rnnt-vulkan
            LABEL    "gpu"
            EXE      test-decoder-determinism
            ARGS     "--model" "${_qvp_rnnt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
                     "--threads" "4" "--n-gpu-layers" "1" "--require-gpu"
            REQUIRES "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}")
    endif()

    # Metal GPU-path instances of the Unified RNN-T fixtures (mirrors the
    # GGML_VULKAN block above; --n-gpu-layers 1 moves the encoder + graph
    # decoder to the registered GPU, which is Metal on Apple builds).
    if (GGML_METAL)
        parakeet_register_test(test-streaming-rnnt-metal
            LABEL    "gpu"
            EXE      test-parakeet-streaming
            ARGS     "--model" "${_qvp_rnnt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
                     "--threads" "4" "--n-gpu-layers" "1"
            REQUIRES "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}")
        parakeet_register_test(test-decoder-determinism-rnnt-metal
            LABEL    "gpu"
            EXE      test-decoder-determinism
            ARGS     "--model" "${_qvp_rnnt_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
                     "--threads" "4" "--n-gpu-layers" "1" "--require-gpu"
            REQUIRES "${_qvp_rnnt_q8_gguf}" "${_qvp_jfk_wav}")
    endif()
endif()

if (PARAKEET_BUILD_EXAMPLES)
    if (NOT PARAKEET_BUILD_LIBRARY)
        message(FATAL_ERROR "PARAKEET_BUILD_EXAMPLES requires PARAKEET_BUILD_LIBRARY=ON")
    endif()

    add_executable(live-mic
        examples/live-mic.cpp
        examples/miniaudio_impl.cpp)
    target_link_libraries(live-mic PRIVATE qvac-parakeet)
    target_include_directories(live-mic PRIVATE include examples)
    target_compile_options(live-mic PRIVATE -Wno-deprecated-declarations)
    parakeet_apply_ccache(live-mic)

    add_executable(live-mic-attributed
        examples/live-mic-attributed.cpp
        examples/miniaudio_impl.cpp)
    target_link_libraries(live-mic-attributed PRIVATE qvac-parakeet)
    target_include_directories(live-mic-attributed PRIVATE include examples)
    target_compile_options(live-mic-attributed PRIVATE -Wno-deprecated-declarations)
    parakeet_apply_ccache(live-mic-attributed)

    if (APPLE)
        find_library(COREAUDIO_LIBRARY CoreAudio REQUIRED)
        find_library(COREFOUNDATION_LIBRARY CoreFoundation REQUIRED)
        find_library(AUDIOUNIT_LIBRARY AudioUnit REQUIRED)
        find_library(AUDIOTOOLBOX_LIBRARY AudioToolbox REQUIRED)
        foreach (tgt live-mic live-mic-attributed)
            target_link_libraries(${tgt} PRIVATE
                ${COREAUDIO_LIBRARY}
                ${COREFOUNDATION_LIBRARY}
                ${AUDIOUNIT_LIBRARY}
                ${AUDIOTOOLBOX_LIBRARY})
        endforeach()
    elseif (UNIX)
        find_package(Threads REQUIRED)
        foreach (tgt live-mic live-mic-attributed)
            target_link_libraries(${tgt} PRIVATE Threads::Threads dl m)
        endforeach()
    elseif (WIN32)
        foreach (tgt live-mic live-mic-attributed)
            target_link_libraries(${tgt} PRIVATE winmm ole32)
        endforeach()
    endif()
endif()

if (PARAKEET_INSTALL AND PARAKEET_BUILD_LIBRARY)
    # Re-install the bundled ggml targets into a `ggml-targets` export
    # set so `install(EXPORT qvac-parakeet-targets)` can resolve the
    # `qvac-parakeet PUBLIC ggml` link as the namespaced `ggml::ggml`.
    # Without this, CMake refuses to generate qvac-parakeet-targets.cmake
    # because the in-tree `ggml` target is not part of any export set.
    # Replicates the standard
    # `install(TARGETS ggml ggml-base EXPORT ggml-targets)` pattern
    # that downstream ggml-based projects ship in their own CMake
    # (e.g. llama.cpp); we add it here at the parakeet level because
    # the upstream ggml at the pinned commit installs ggml without
    # an EXPORT clause. The bundled ggml's hand-written
    # ggml-config.cmake (find_library + UNKNOWN IMPORTED) still
    # ships and is what's loaded via find_dependency(ggml) -- we
    # don't replace it; the new ggml-targets.cmake just carries the
    # export so CMake's cross-package reference check passes here.
    if (NOT PARAKEET_USE_SYSTEM_GGML)
        set(_parakeet_ggml_export_targets ggml ggml-base)
        if (DEFINED GGML_AVAILABLE_BACKENDS)
            list(APPEND _parakeet_ggml_export_targets ${GGML_AVAILABLE_BACKENDS})
        endif()
        # No PUBLIC_HEADER DESTINATION here: the bundled ggml's own
        # `install(TARGETS ggml LIBRARY PUBLIC_HEADER)` already copies
        # the ggml/include/*.h headers into the install tree (relative
        # paths resolve against ggml/CMakeLists.txt's source dir). If
        # we duplicated the PUBLIC_HEADER install here, the same
        # relative paths would be re-resolved against the parakeet
        # source dir and `cmake --install` would fail with
        # "file INSTALL cannot find <root>/include/ggml.h: File exists".
        # CMake emits a developer warning ("PUBLIC_HEADER files but no
        # PUBLIC_HEADER DESTINATION") about the gap, which is benign
        # here -- ggml's own rule still publishes the headers. Keep
        # the rule minimal: we only need EXPORT + library destinations
        # so `qvac-parakeet-targets` can resolve the namespaced
        # `ggml::ggml` link without rebuilding the headers install.
        install(TARGETS ${_parakeet_ggml_export_targets}
            EXPORT ggml-targets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
        install(EXPORT ggml-targets
            FILE ggml-targets.cmake
            NAMESPACE ggml::
            DESTINATION ${CMAKE_INSTALL_DATADIR}/ggml
        )
        unset(_parakeet_ggml_export_targets)
    endif()

    # Export the parakeet target via an install(EXPORT ...) set so the
    # generated qvac-parakeet-targets.cmake carries the correct shape
    # for whatever linkage the build produced (STATIC vs SHARED, with
    # IMPORTED_IMPLIB on Windows shared builds, etc.). Hand-writing
    # a STATIC IMPORTED stub here would silently mis-resolve the
    # downstream target on every shared / Windows install and is the
    # bug this replaces. Same pattern as the standard llama-style
    # `install(EXPORT <project>-targets NAMESPACE <project>::)` flow.
    # parakeet-backend-defs is an INTERFACE library that carries the
    # GGML_USE_* compile defines; for STATIC parakeet builds the
    # PRIVATE link bleeds into the interface (static archives don't
    # carry transitive PRIVATE deps the way shared libs do), so the
    # exporter would otherwise refuse with
    # "requires target parakeet-backend-defs that is not in any export set".
    # Every installable artifact lives under the `qvac-parakeet` name:
    # upstream whisper.cpp (v1.9.1+) builds and installs its own
    # `parakeet` (lib/libparakeet.*, lib/cmake/parakeet/, parakeet.pc),
    # so a vcpkg tree holding both the whisper-cpp port and this engine
    # would otherwise fail with a file-ownership conflict on the shared
    # filenames. Public headers stay at include/parakeet/ (a directory,
    # no clash with upstream's single include/parakeet.h) and the C++
    # `parakeet::` namespace is untouched — the collision is file
    # ownership, not symbols.
    install(TARGETS qvac-parakeet parakeet-backend-defs
        EXPORT qvac-parakeet-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY include/parakeet
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )

    # lib/cmake/qvac-parakeet mirrors the whisper subtree's
    # lib/cmake/whisper convention; the vcpkg port relocates it with
    # vcpkg_cmake_config_fixup(PACKAGE_NAME qvac-parakeet
    #                          CONFIG_PATH lib/cmake/qvac-parakeet)
    # and must also call vcpkg_fixup_pkgconfig() for the .pc below
    # (its prefix= bakes in the absolute packages dir otherwise).
    set(PARAKEET_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/qvac-parakeet)

    install(EXPORT qvac-parakeet-targets
        FILE qvac-parakeet-targets.cmake
        NAMESPACE qvac::
        DESTINATION ${PARAKEET_CONFIG_INSTALL_DIR}
    )

    # Whether this build linked OpenMP, baked into the package config so a
    # consumer re-resolves the same dependency set. See the comment in
    # cmake/qvac-parakeet-config.cmake.in: the PRIVATE OpenMP link is
    # exported as $<LINK_ONLY:OpenMP::OpenMP_CXX> for the STATIC build, so
    # the imported target must exist in the consumer's scope.
    if (OpenMP_CXX_FOUND)
        set(QVAC_PARAKEET_WITH_OPENMP TRUE)
    else()
        set(QVAC_PARAKEET_WITH_OPENMP FALSE)
    endif()

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/qvac-parakeet-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/qvac-parakeet-config.cmake
        INSTALL_DESTINATION ${PARAKEET_CONFIG_INSTALL_DIR}
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/qvac-parakeet-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/qvac-parakeet-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/qvac-parakeet-config-version.cmake
        DESTINATION ${PARAKEET_CONFIG_INSTALL_DIR}
    )

    # ---------------------------------------------------------------
    # pkg-config linkage.
    #
    # ggml: derived from the actual ggml targets rather than from a
    # hardcoded name list, because every QVAC ggml build renames its
    # libraries via GGML_LIB_OUTPUT_PREFIX (the ggml-speech vcpkg port
    # uses `qvac-speech-`, the bundled build above sets `speech-`) and
    # a system ggml may live in a different prefix than ours. For an
    # imported (system) ggml we read IMPORTED_LOCATION and emit both
    # the containing -L and the real library stem; for the bundled
    # in-tree build we read OUTPUT_NAME and rely on our own ${libdir}
    # (the re-install rule above puts ggml there). Deliberately NOT
    # `Requires: ggml`: the qvac-ext-ggml@speech ggml.pc advertises an
    # unprefixed `-lggml` that no install actually produces and omits
    # ggml-base entirely, so inheriting it would under-link. Revisit
    # once that .pc is fixed upstream in the fork.
    set(_pk_pc_ggml_libs "")
    set(_pk_pc_ggml_libdirs "")
    foreach (_pk_ggml_tgt ggml::ggml ggml::ggml-base ggml ggml-base)
        if (NOT TARGET ${_pk_ggml_tgt})
            continue()
        endif()
        get_target_property(_pk_ggml_imported ${_pk_ggml_tgt} IMPORTED)
        set(_pk_ggml_libname "")
        if (_pk_ggml_imported)
            set(_pk_ggml_loc "")
            get_target_property(_pk_ggml_loc ${_pk_ggml_tgt} IMPORTED_LOCATION)
            if (NOT _pk_ggml_loc)
                foreach (_pk_cfg RELEASE NOCONFIG DEBUG "")
                    if (_pk_cfg)
                        get_target_property(_pk_ggml_loc ${_pk_ggml_tgt} IMPORTED_LOCATION_${_pk_cfg})
                    endif()
                    if (_pk_ggml_loc)
                        break()
                    endif()
                endforeach()
            endif()
            if (_pk_ggml_loc)
                get_filename_component(_pk_ggml_dir  "${_pk_ggml_loc}" DIRECTORY)
                get_filename_component(_pk_ggml_stem "${_pk_ggml_loc}" NAME_WE)
                string(REGEX REPLACE "^lib" "" _pk_ggml_libname "${_pk_ggml_stem}")
                if (NOT _pk_ggml_dir STREQUAL "${CMAKE_INSTALL_FULL_LIBDIR}")
                    list(APPEND _pk_pc_ggml_libdirs "-L${_pk_ggml_dir}")
                endif()
            endif()
        else()
            get_target_property(_pk_ggml_libname ${_pk_ggml_tgt} OUTPUT_NAME)
            if (NOT _pk_ggml_libname)
                set(_pk_ggml_libname "${_pk_ggml_tgt}")
            endif()
        endif()
        # Namespaced and plain spellings of the same target both appear in
        # the loop (only one exists per acquisition path, but the bundled
        # build aliases ggml::ggml -> ggml); de-duplicate on the resolved
        # library name so the .pc never lists a library twice.
        if (_pk_ggml_libname AND NOT "-l${_pk_ggml_libname}" IN_LIST _pk_pc_ggml_libs)
            list(APPEND _pk_pc_ggml_libs "-l${_pk_ggml_libname}")
        endif()
    endforeach()
    if (NOT _pk_pc_ggml_libs)
        # Nothing resolvable (e.g. a hand-rolled ggml import with no
        # location): fall back to the prefix-derived names so the .pc is
        # still populated rather than silently empty.
        set(_pk_pc_ggml_libs "-l${GGML_LIB_OUTPUT_PREFIX}ggml" "-l${GGML_LIB_OUTPUT_PREFIX}ggml-base")
        message(WARNING "parakeet: could not resolve ggml library files for qvac-parakeet.pc; "
                        "falling back to prefix-derived names (${_pk_pc_ggml_libs})")
    endif()
    list(REMOVE_DUPLICATES _pk_pc_ggml_libdirs)
    string(JOIN " " PARAKEET_PC_GGML_LIBS ${_pk_pc_ggml_libdirs} ${_pk_pc_ggml_libs})

    # ggml include dir, same reasoning as the -L above: the installed
    # <parakeet/log.h> includes "ggml.h", so a pkg-config consumer needs
    # ggml's headers on the search path. They are only co-located with
    # ours in the bundled build; a system ggml lives in its own prefix.
    set(_pk_pc_ggml_incs "")
    foreach (_pk_ggml_tgt ggml::ggml ggml)
        if (NOT TARGET ${_pk_ggml_tgt})
            continue()
        endif()
        get_target_property(_pk_ggml_ifaces ${_pk_ggml_tgt} INTERFACE_INCLUDE_DIRECTORIES)
        if (_pk_ggml_ifaces)
            foreach (_pk_ggml_inc IN LISTS _pk_ggml_ifaces)
                # Skip generator expressions ($<BUILD_INTERFACE:...> on the
                # in-tree target) and our own includedir.
                if (_pk_ggml_inc MATCHES "\\$<" OR NOT IS_DIRECTORY "${_pk_ggml_inc}")
                    continue()
                endif()
                if (NOT _pk_ggml_inc STREQUAL "${CMAKE_INSTALL_FULL_INCLUDEDIR}")
                    list(APPEND _pk_pc_ggml_incs "-I${_pk_ggml_inc}")
                endif()
            endforeach()
        endif()
        break()
    endforeach()
    list(REMOVE_DUPLICATES _pk_pc_ggml_incs)
    string(JOIN " " PARAKEET_PC_GGML_CFLAGS ${_pk_pc_ggml_incs})

    # Libs.private: deps that are PRIVATE/interface-only on the target and
    # therefore NOT contained in the static archive. pkg-config consumers
    # of a static libqvac-parakeet must pass these themselves or the link
    # fails on unresolved OpenMP / Core ML symbols; `pkg-config --static`
    # is what pulls them in. Harmless for shared consumption (the .so
    # already carries its own dependencies).
    set(_pk_pc_private "")
    if (OpenMP_CXX_FOUND)
        # OpenMP_CXX_FLAGS carries the compile/link driver flag (-fopenmp,
        # or `-Xclang -fopenmp` on AppleClang, which needs the explicit
        # runtime below). OpenMP_CXX_LIBRARIES holds absolute paths;
        # pkg-config accepts those verbatim, so pass them through rather
        # than guessing between -lgomp / -lomp / -liomp5.
        if (OpenMP_CXX_FLAGS)
            list(APPEND _pk_pc_private ${OpenMP_CXX_FLAGS})
        endif()
        foreach (_pk_omp_lib IN LISTS OpenMP_CXX_LIBRARIES)
            if (IS_ABSOLUTE "${_pk_omp_lib}")
                list(APPEND _pk_pc_private "${_pk_omp_lib}")
            else()
                list(APPEND _pk_pc_private "-l${_pk_omp_lib}")
            endif()
        endforeach()
    endif()
    if (PARAKEET_COREML)
        list(APPEND _pk_pc_private "-framework CoreML" "-framework Foundation")
    endif()
    if (_pk_pc_private)
        list(REMOVE_DUPLICATES _pk_pc_private)
        string(JOIN " " _pk_pc_private_str ${_pk_pc_private})
        set(PARAKEET_PC_LIBS_PRIVATE "Libs.private: ${_pk_pc_private_str}")
    else()
        set(PARAKEET_PC_LIBS_PRIVATE "")
    endif()

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/qvac-parakeet.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/qvac-parakeet.pc
        @ONLY
    )

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/qvac-parakeet.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )
endif()
