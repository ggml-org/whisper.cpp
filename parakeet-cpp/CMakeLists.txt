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

function(parakeet_apply_ggml_prefix target)
    if (NOT TARGET ${target})
        return()
    endif()
    get_target_property(_qpgp_type ${target} TYPE)
    if (_qpgp_type STREQUAL "INTERFACE_LIBRARY" OR _qpgp_type STREQUAL "OBJECT_LIBRARY")
        return()
    endif()
    get_target_property(_qpgp_old_name ${target} OUTPUT_NAME)
    if (NOT _qpgp_old_name OR _qpgp_old_name STREQUAL "_qpgp_old_name-NOTFOUND")
        set(_qpgp_old_name ${target})
    endif()
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "speech-${_qpgp_old_name}"
    )
endfunction()

if (PARAKEET_GGML_LIB_PREFIX AND NOT PARAKEET_USE_SYSTEM_GGML)
    foreach (_qpgp_target ggml ggml-base)
        parakeet_apply_ggml_prefix(${_qpgp_target})
    endforeach()
    if (DEFINED GGML_AVAILABLE_BACKENDS)
        foreach (_qpgp_target ${GGML_AVAILABLE_BACKENDS})
            parakeet_apply_ggml_prefix(${_qpgp_target})
        endforeach()
    endif()
    # Renaming the bundled backend .so/.dll files alone is not enough:
    # ggml's runtime loader (`ggml_backend_load_best`) hard-codes the
    # `libggml-` / `ggml-` filename prefix when scanning for backends
    # under `GGML_BACKEND_DL=ON`. The companion patch
    # `patches/ggml-backend-reg-filename-prefix.patch` adds a
    # `GGML_BACKEND_DL_PROJECT_PREFIX` macro to that loader; defining
    # it here teaches the runtime to look for our prefixed filenames
    # instead. Otherwise the renamed .so/.dll files exist on disk but
    # are never discovered, and Vulkan/OpenCL/CUDA backends silently
    # fail to load.
    if (TARGET ggml)
        target_compile_definitions(ggml PRIVATE
            GGML_BACKEND_DL_PROJECT_PREFIX="speech-"
        )
    endif()
    message(STATUS "parakeet: bundled ggml libraries will be emitted as libspeech-ggml-* (set PARAKEET_GGML_LIB_PREFIX=OFF to keep upstream filenames)")
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

# Centralised GGML_USE_* backend defines. Anything that compiles
# parakeet_ctc.cpp (the library target plus the standalone test
# executables that recompile it) must link against this so the
# `init_gpu_backend` / BLAS / CUDA / Metal / Vulkan code paths get
# selected consistently. Without this, e.g. test-encoder would silently
# build with the GPU branch compiled out and `--n-gpu-layers 1` would
# be a no-op.
add_library(parakeet-backend-defs INTERFACE)
if (GGML_CUDA)
    target_compile_definitions(parakeet-backend-defs INTERFACE GGML_USE_CUDA)
endif()
if (GGML_METAL)
    target_compile_definitions(parakeet-backend-defs INTERFACE GGML_USE_METAL)
endif()
if (GGML_VULKAN)
    target_compile_definitions(parakeet-backend-defs INTERFACE GGML_USE_VULKAN)
endif()
if (GGML_BLAS)
    target_compile_definitions(parakeet-backend-defs INTERFACE GGML_USE_BLAS)
endif()
if (GGML_OPENCL)
    target_compile_definitions(parakeet-backend-defs INTERFACE GGML_USE_OPENCL)
endif()

set(PARAKEET_LIB_SOURCES
    src/parakeet_ctc.cpp
    src/parakeet_engine.cpp
    src/parakeet_tdt.cpp
    src/parakeet_eou.cpp
    src/parakeet_sortformer.cpp
    src/parakeet_log.cpp
    src/mel_preprocess.cpp
    src/sentencepiece_bpe.cpp
    src/energy_vad.cpp
    # Lives in the library so `parakeet_cli_main` (declared in
    # the umbrella header `parakeet.h`) is actually a library
    # symbol that consumers / host applications can link against.
    # The CLI executable below is a thin shim that just calls into it.
    src/main.cpp
)

if (PARAKEET_BUILD_LIBRARY)
    # Honour BUILD_SHARED_LIBS when set explicitly; otherwise default to
    # STATIC. Same shape as the standard llama-style library contract
    # (BUILD_SHARED_LIBS-driven), so consumers that build other ggml-
    # based libraries shared see a consistent linkage selector.
    if (DEFINED BUILD_SHARED_LIBS)
        add_library(parakeet ${PARAKEET_LIB_SOURCES})
    else()
        add_library(parakeet STATIC ${PARAKEET_LIB_SOURCES})
    endif()
    add_library(parakeet::parakeet ALIAS parakeet)
    parakeet_apply_ccache(parakeet)

    target_include_directories(parakeet
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    )
    target_link_libraries(parakeet PUBLIC ggml)

    if (OpenMP_CXX_FOUND)
        target_link_libraries(parakeet PRIVATE OpenMP::OpenMP_CXX)
    endif()

    target_link_libraries(parakeet PRIVATE parakeet-backend-defs)

    # Single source of truth for the version string the CLI prints
    # under `--version`. Kept PRIVATE because it only drives the in-
    # library `main.cpp` impl; consumers that need a runtime version
    # can read it via the public API surface or the CMake package
    # version file (`parakeet-cppConfigVersion.cmake`).
    target_compile_definitions(parakeet PRIVATE
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
    if (GGML_METAL)
        set(PARAKEET_FLASH_ATTN_DEFAULT ON)
    else()
        set(PARAKEET_FLASH_ATTN_DEFAULT OFF)
    endif()
    option(PARAKEET_FLASH_ATTN "parakeet: enable fused flash-attn in MHA (default ON for Metal; OFF elsewhere pending per-backend A/B)" ${PARAKEET_FLASH_ATTN_DEFAULT})
    if (PARAKEET_FLASH_ATTN)
        target_compile_definitions(parakeet PRIVATE PARAKEET_EXPERIMENTAL_FLASH_ATTN)
    endif()

    set_target_properties(parakeet PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME parakeet
        CXX_VISIBILITY_PRESET    hidden
        VISIBILITY_INLINES_HIDDEN ON
        # VERSION drives the full filename suffix on Linux/macOS shared
        # libs (libparakeet.so.0.1.0); SOVERSION drives the symlink the
        # dynamic loader actually opens (libparakeet.so.0). Pinning
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
    get_target_property(_parakeet_type parakeet TYPE)
    if (_parakeet_type STREQUAL "SHARED_LIBRARY")
        target_compile_definitions(parakeet PUBLIC  PARAKEET_SHARED)
        target_compile_definitions(parakeet PRIVATE PARAKEET_BUILD)
    endif()

    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
        target_compile_options(parakeet PRIVATE
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
    target_link_libraries(parakeet-cli PRIVATE parakeet)
    set_target_properties(parakeet-cli PROPERTIES OUTPUT_NAME parakeet)
    parakeet_apply_ccache(parakeet-cli)
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
        if (WIN32)
            # When ggml builds as shared (the clang-cl / MSVC default with
            # GGML_BACKEND_DL on), test exes link against import libs and
            # need to find `speech-ggml-*.dll` at runtime. CMake places
            # those at `${CMAKE_BINARY_DIR}/bin`, but Windows has no rpath
            # and the test exes live next to the binary root, not in bin/.
            # Without this, the OS loader fails with STATUS_DLL_NOT_FOUND
            # (0xc0000135) before main() runs. Inject the bin/ dir into
            # the test's PATH; configure-time $ENV{PATH} is appended so
            # any compiler-runtime DLLs the user needs stay reachable.
            # Static-ggml builds (msys2 .a archives) are unaffected --
            # the prepended dir simply doesn't exist and is ignored.
            set_property(TEST ${target} PROPERTY ENVIRONMENT
                "PATH=${CMAKE_BINARY_DIR}/bin\;$ENV{PATH}")
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

    # Helper: keep the parakeet_ctc.cpp #ifdefs (BLAS / CUDA / Metal / Vulkan /
    # OpenCL backend init) consistent across the parakeet library and any
    # test executable that compiles parakeet_ctc.cpp from source. Without this,
    # tests that don't link the library would always evaluate the #ifdefs as
    # "no backend defined", producing link errors against ggml-blas / ggml-vk
    # / ggml-opencl when the parent build did enable them.
    function(parakeet_apply_backend_defs target)
        if (GGML_BLAS)
            target_compile_definitions(${target} PRIVATE GGML_USE_BLAS)
        endif()
        if (GGML_CUDA)
            target_compile_definitions(${target} PRIVATE GGML_USE_CUDA)
        endif()
        if (GGML_METAL)
            target_compile_definitions(${target} PRIVATE GGML_USE_METAL)
        endif()
        if (GGML_VULKAN)
            target_compile_definitions(${target} PRIVATE GGML_USE_VULKAN)
        endif()
        if (GGML_OPENCL)
            target_compile_definitions(${target} PRIVATE GGML_USE_OPENCL)
        endif()
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
    set(_qvp_tdt_q8_gguf   "${PARAKEET_TEST_MODEL_DIR}/parakeet-tdt-0.6b-v3.q8_0.gguf")
    set(_qvp_tdt_f16_gguf  "${PARAKEET_TEST_MODEL_DIR}/parakeet-tdt-0.6b-v3.f16.gguf")
    set(_qvp_eou_q8_gguf   "${PARAKEET_TEST_MODEL_DIR}/parakeet_realtime_eou_120m-v1.q8_0.gguf")
    set(_qvp_sf_q8_gguf    "${PARAKEET_TEST_MODEL_DIR}/diar_sortformer_4spk-v1.q8_0.gguf")
    set(_qvp_sf_f16_gguf   "${PARAKEET_TEST_MODEL_DIR}/diar_sortformer_4spk-v1.f16.gguf")
    set(_qvp_sfs_q8_gguf   "${PARAKEET_TEST_MODEL_DIR}/diar_streaming_sortformer_4spk-v2.q8_0.gguf")
    set(_qvp_jfk_wav       "${PARAKEET_TEST_AUDIO_DIR}/jfk.wav")
    set(_qvp_diar_wav      "${PARAKEET_TEST_AUDIO_DIR}/diarization-sample-16k.wav")
    set(_qvp_ctc_ref       "${PARAKEET_TEST_REF_DIR}/ctc-ref")
    set(_qvp_tdt_ref       "${PARAKEET_TEST_REF_DIR}/tdt-ref")
    set(_qvp_sf_ref        "${PARAKEET_TEST_REF_DIR}/sortformer-ref")

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

    add_executable(test-streaming test/test_streaming.cpp)
    target_link_libraries(test-streaming PRIVATE parakeet)
    target_include_directories(test-streaming PRIVATE include src ggml/include)
    parakeet_apply_ccache(test-streaming)
    parakeet_register_test(test-streaming
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_ctc_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_ctc_q8_gguf}" "${_qvp_jfk_wav}")

    add_executable(test-eou-streaming test/test_eou_streaming.cpp)
    target_link_libraries(test-eou-streaming PRIVATE parakeet)
    target_include_directories(test-eou-streaming PRIVATE include src ggml/include)
    parakeet_apply_ccache(test-eou-streaming)
    parakeet_register_test(test-eou-streaming
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_eou_q8_gguf}" "--wav" "${_qvp_jfk_wav}"
        REQUIRES "${_qvp_eou_q8_gguf}" "${_qvp_jfk_wav}")

    add_executable(test-sortformer-streaming test/test_sortformer_streaming.cpp)
    target_link_libraries(test-sortformer-streaming PRIVATE parakeet)
    target_include_directories(test-sortformer-streaming PRIVATE include src ggml/include)
    parakeet_apply_ccache(test-sortformer-streaming)
    parakeet_register_test(test-sortformer-streaming
        LABEL    "fixture"
        ARGS     "--model" "${_qvp_sfs_q8_gguf}" "--wav" "${_qvp_diar_wav}"
        REQUIRES "${_qvp_sfs_q8_gguf}" "${_qvp_diar_wav}")

    add_executable(test-perf-regression test/test_perf_regression.cpp)
    target_link_libraries(test-perf-regression PRIVATE parakeet)
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

    add_executable(test-tdt-encoder-parity
        test/test_tdt_encoder_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
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
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
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

    add_executable(test-sortformer-parity
        test/test_sortformer_parity.cpp
        src/parakeet_ctc.cpp
        src/parakeet_engine.cpp
        src/parakeet_log.cpp
        src/parakeet_tdt.cpp
        src/parakeet_eou.cpp
        src/parakeet_sortformer.cpp
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
    target_link_libraries(test-decoder-determinism PRIVATE parakeet)
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

    if (GGML_VULKAN)
        add_executable(test-vk-vs-cpu
            test/test_vk_vs_cpu.cpp
            src/parakeet_ctc.cpp
            src/parakeet_log.cpp
            src/mel_preprocess.cpp
            src/sentencepiece_bpe.cpp)
        target_link_libraries(test-vk-vs-cpu PRIVATE ggml parakeet-backend-defs)
        target_include_directories(test-vk-vs-cpu PRIVATE ggml/include src include)
        parakeet_apply_backend_defs(test-vk-vs-cpu)
        parakeet_register_test(test-vk-vs-cpu
            LABEL    "gpu"
            ARGS     "${_qvp_ctc_q8_gguf}" "${_qvp_jfk_wav}"
            REQUIRES "${_qvp_ctc_q8_gguf}" "${_qvp_jfk_wav}")
    endif()
endif()

if (PARAKEET_BUILD_EXAMPLES)
    if (NOT PARAKEET_BUILD_LIBRARY)
        message(FATAL_ERROR "PARAKEET_BUILD_EXAMPLES requires PARAKEET_BUILD_LIBRARY=ON")
    endif()

    add_executable(live-mic
        examples/live-mic.cpp
        examples/miniaudio_impl.cpp)
    target_link_libraries(live-mic PRIVATE parakeet)
    target_include_directories(live-mic PRIVATE include examples)
    target_compile_options(live-mic PRIVATE -Wno-deprecated-declarations)
    parakeet_apply_ccache(live-mic)

    add_executable(live-mic-attributed
        examples/live-mic-attributed.cpp
        examples/miniaudio_impl.cpp)
    target_link_libraries(live-mic-attributed PRIVATE parakeet)
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
    # set so `install(EXPORT parakeet-cpp-targets)` can resolve the
    # `parakeet PUBLIC ggml` link as the namespaced `ggml::ggml`.
    # Without this, CMake refuses to generate parakeet-cpp-targets.cmake
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
        # so `parakeet-cpp-targets` can resolve the namespaced
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
    # generated parakeet-cpp-targets.cmake carries the correct shape
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
    install(TARGETS parakeet parakeet-backend-defs
        EXPORT parakeet-cpp-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY include/parakeet
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )

    set(PARAKEET_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_DATADIR}/parakeet-cpp)

    install(EXPORT parakeet-cpp-targets
        FILE parakeet-cpp-targets.cmake
        NAMESPACE parakeet::
        DESTINATION ${PARAKEET_CONFIG_INSTALL_DIR}
    )

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/parakeet-cppConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/parakeet-cppConfig.cmake
        INSTALL_DESTINATION ${PARAKEET_CONFIG_INSTALL_DIR}
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/parakeet-cppConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/parakeet-cppConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/parakeet-cppConfigVersion.cmake
        DESTINATION ${PARAKEET_CONFIG_INSTALL_DIR}
    )
endif()
