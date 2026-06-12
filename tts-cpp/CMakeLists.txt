cmake_minimum_required(VERSION 3.20)
project(tts-cpp LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CheckIncludeFileCXX)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(TTS_CPP_STANDALONE_DEFAULT ON)
else()
    set(TTS_CPP_STANDALONE_DEFAULT OFF)
endif()

option(TTS_CPP_BUILD_LIBRARY     "tts-cpp: build the tts-cpp library"                        ON)
option(TTS_CPP_BUILD_SHARED      "tts-cpp: build tts-cpp as a SHARED library (otherwise STATIC). Defaults to OFF; the test harnesses (test-mtl-tokenizer, test-supertonic-*) link against tts-cpp directly and use detail-namespaced symbols outside the TTS_CPP_API public surface, so SHARED hides them and disables those targets. Decoupled from BUILD_SHARED_LIBS because ggml's own CMake declares its own option(BUILD_SHARED_LIBS) which pollutes the cache with a platform-dependent default once any configure has run; using a project-namespaced option keeps the two independent and prevents a second configure from silently flipping linkage." OFF)
option(TTS_CPP_BUILD_EXECUTABLES "tts-cpp: build the tts-cli executable + mel2wav demo"     ${TTS_CPP_STANDALONE_DEFAULT})
option(TTS_CPP_BUILD_TESTS       "tts-cpp: build per-stage numerical validation harnesses"   ${TTS_CPP_STANDALONE_DEFAULT})
option(TTS_CPP_INSTALL           "tts-cpp: generate install rules"                           ON)
option(TTS_CPP_USE_SYSTEM_GGML   "tts-cpp: use system-installed GGML library (find_package(ggml) instead of add_subdirectory(ggml)). Defaults to ON in this in-tree subtree of qvac-ext-lib-whisper.cpp because the standalone setup-ggml.sh + patches/ tooling has been dropped here; the QVAC speech-stack vcpkg port `ggml-speech` ships a pre-patched ggml from the qvac-ext-ggml/speech branch and is the canonical source-of-truth for builds out of this tree.  Flipping OFF in this tree is rejected at configure time (no patches/ to apply); use the standalone github.com/gianni-cor/chatterbox.cpp repo if you need a bundled-ggml dev build." ON)
# Note: the standalone chatterbox.cpp repo carries a TTS_CPP_GGML_LIB_PREFIX
# option here that controls renaming of bundled libggml-*.so to
# libspeech-ggml-*.so when add_subdirectory(ggml) builds it locally.  In
# this in-tree subtree that path is rejected at configure time (the
# patches/ folder is intentionally absent; see TTS_CPP_USE_SYSTEM_GGML
# below), so the option, the helper function, and the foreach prefix
# loop that used to live here would all be unreachable code.  Dropped
# entirely; the `ggml-speech` vcpkg port emits the renamed filenames
# itself, so consumers picking up tts-cpp through find_package never
# see the standalone bundled-rename path.
option(TTS_CPP_CCACHE            "tts-cpp: use ccache as compiler launcher if available"     ON)

# ccache shaves ~30 % off iterative rebuild time on developer machines.
# Opt-out via -DTTS_CPP_CCACHE=OFF for clean reproducibility runs.
# Scoped to tts-cpp's own targets via the helper below; ggml has its own
# GGML_CCACHE option which we leave alone so the two launchers don't
# double-up on the ggml subdirectory.
if (TTS_CPP_CCACHE)
    find_program(TTS_CPP_CCACHE_PROGRAM ccache)
    if (TTS_CPP_CCACHE_PROGRAM)
        message(STATUS "tts-cpp: using ccache at ${TTS_CPP_CCACHE_PROGRAM} (tts-cpp targets only; ggml uses GGML_CCACHE)")
    endif()
endif()

function(tts_cpp_apply_ccache target)
    if (TTS_CPP_CCACHE_PROGRAM)
        set_target_properties(${target} PROPERTIES
            C_COMPILER_LAUNCHER   "${TTS_CPP_CCACHE_PROGRAM}"
            CXX_COMPILER_LAUNCHER "${TTS_CPP_CCACHE_PROGRAM}"
        )
    endif()
endfunction()

if (CMAKE_CROSSCOMPILING)
    if (NOT DEFINED GGML_NATIVE OR GGML_NATIVE)
        message(STATUS "tts-cpp: cross-compiling to ${CMAKE_SYSTEM_NAME} -- "
                       "forcing GGML_NATIVE=OFF (ggml's ARM feature probes use try_run() "
                       "which CMake refuses in cross-compile mode).")
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
# per-arch CPU variants. Mirrors parakeet-cpp's same-repo sibling and
# the qvac llm-llamacpp Android config (see
# qvac-registry-vcpkg/ports/llama-cpp/portfile.cmake) so the tts-cpp
# Android prebuilds drop into the same `qvac__tts-ggml/` folder shape
# as the parakeet / llamacpp ones: a `.bare` module + sibling
# `lib<prefix>ggml-{vulkan,opencl,cpu-android_armv*_*}.so` files that
# `ggml_backend_load_all_from_path()` discovers at runtime.
#
# Selection at runtime is centralised in
# `tts_cpp::detail::init_gpu_backend()` (src/backend_selection.cpp):
# OpenCL when an Adreno 700+ device is present, Vulkan for every
# other GPU (non-Adreno, Adreno < 700, Mali, Xclipse, ...). No
# static GPU backend entry points are linked anywhere in libtts-cpp;
# the registry walk reaches the right backend in both
# GGML_BACKEND_DL=ON (Android prebuild) and GGML_BACKEND_DL=OFF
# (desktop dev) modes.
#
# Callers that have specific reasons to deviate (e.g. a desktop
# bring-up build that wants Vulkan only) can still override any of
# these at the cmake command line; we only set defaults that haven't
# already been provided.
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
    # ggml-vulkan's coopmat / coopmat2 shader compile pulls in
    # extensions that most Android Vulkan drivers don't expose; the
    # upstream llama Android build disables both for the same reason.
    if (NOT DEFINED CACHE{GGML_VULKAN_DISABLE_COOPMAT})
        set(GGML_VULKAN_DISABLE_COOPMAT ON CACHE BOOL "" FORCE)
    endif()
    if (NOT DEFINED CACHE{GGML_VULKAN_DISABLE_COOPMAT2})
        set(GGML_VULKAN_DISABLE_COOPMAT2 ON CACHE BOOL "" FORCE)
    endif()
endif()

# Two related workarounds for clang-cl / MSVC builds on Windows.  Both
# come from msys2 sneaking GCC-flavoured libraries onto CMake's search
# paths and being mismatched against MSVC-compiled translation units.
# Both are no-ops on MinGW (excluded) and on non-Windows.
#
# (1) libm. ggml/src/CMakeLists.txt unconditionally links libm on Windows
#     unless ONEAPI_ROOT is set, which fails when find_library picks up a
#     stray libm.dll.a from msys2 (lld-link errors with `could not open
#     'm.lib'`). Set ONEAPI_ROOT in the configure-time environment so
#     ggml's existing escape hatch fires.  set(ENV{...}) only affects
#     the running CMake configure process — the same scope ggml's check
#     runs in.
#
# (2) OpenMP. CMake's FindOpenMP picks `-fopenmp=libomp` (LLVM-style) for
#     MSVC, but resolves OpenMP_*_LIBRARIES to msys2's libgomp.dll.a if
#     msys2 is on PATH.  The resulting object files reference LLVM
#     `__kmpc_*` symbols that GOMP doesn't provide and lld-link errors
#     out with `undefined symbol: __kmpc_global_thread_num`.  Disable
#     OpenMP for this toolchain; the perf hit on the CPU encoder is
#     bounded and only affects this specific build environment.
if (WIN32 AND NOT MINGW)
    if (NOT DEFINED ENV{ONEAPI_ROOT})
        set(ENV{ONEAPI_ROOT} "1")
    endif()
    if (NOT DEFINED GGML_OPENMP AND NOT DEFINED CACHE{GGML_OPENMP})
        # Normal (non-cache) variable on purpose: scoped to this
        # listfile + its add_subdirectory(ggml) below, doesn't poison
        # parent-project CMakeCache when tts-cpp is consumed via
        # add_subdirectory.  ggml's `option(GGML_OPENMP ... ON)` respects
        # an existing normal variable under CMP0077 (CMake 3.13+).
        set(GGML_OPENMP OFF)
    endif()
endif()

if (NOT TARGET ggml)
    if (TTS_CPP_USE_SYSTEM_GGML)
        find_package(ggml CONFIG REQUIRED)
        if (NOT ggml_FOUND)
            message(FATAL_ERROR
                "tts-cpp: system-installed GGML library not found.  Install the "
                "qvac speech-stack `ggml-speech` vcpkg port (qvac-ext-ggml/speech "
                "branch) and re-run cmake.")
        endif()
        add_library(ggml ALIAS ggml::ggml)
    else()
        # Bundled-ggml dev build path (TTS_CPP_USE_SYSTEM_GGML=OFF).
        # Expects `tts-cpp/ggml/` to be a checkout of the
        # tetherto/qvac-ext-ggml repo on the `speech` branch — the QVAC
        # fork carrying every infrastructure patch + the Supertonic 2
        # fused custom op family as commits (not as a patches/ overlay).
        #
        # Run `bash tts-cpp/scripts/setup-ggml.sh` first to clone +
        # check out the pinned commit.  No patches/ directory is
        # consulted: the speech branch is already pre-patched at the
        # commit level.
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
            message(FATAL_ERROR
                "tts-cpp: bundled-ggml build requires tts-cpp/ggml/ to be "
                "a checkout of tetherto/qvac-ext-ggml@speech.  Run "
                "`bash tts-cpp/scripts/setup-ggml.sh` first, or pass "
                "-DTTS_CPP_USE_SYSTEM_GGML=ON to consume the QVAC "
                "speech-stack `ggml-speech` vcpkg port.")
        endif()
        add_subdirectory(ggml)
    endif()
endif()

# The bundled-ggml filename-prefix block (helper function +
# tts_cpp_apply_ggml_prefix loop + GGML_BACKEND_DL_PROJECT_PREFIX
# define) used to live here in the standalone chatterbox.cpp source.
# Removed in this in-tree subtree: the bundled add_subdirectory(ggml)
# path is rejected above when patches/ is absent, and find_package
# (TTS_CPP_USE_SYSTEM_GGML=ON, the default here) consumes a ggml
# whose filenames + macro are already set by the ggml-speech vcpkg
# port at its build time.  See the standalone chatterbox.cpp repo if
# you need the locally-rename flow.

option(TTS_CPP_OPENMP "tts-cpp: enable OpenMP for the tts-cpp target" ON)
if (WIN32 AND NOT MINGW AND TTS_CPP_OPENMP AND NOT DEFINED CACHE{TTS_CPP_OPENMP_USER_OVERRIDE})
    set(TTS_CPP_OPENMP OFF CACHE BOOL "" FORCE)
    message(STATUS "tts-cpp: OpenMP disabled on Windows non-MinGW (clang-cl / MSVC + msys2 libgomp mismatch, plus consumer-toolchain probe leaks via static-archive transitive link interface). Set -DTTS_CPP_OPENMP_USER_OVERRIDE=ON and -DTTS_CPP_OPENMP=ON to force-enable.")
endif()

if (TTS_CPP_OPENMP)
    find_package(OpenMP)
endif()

if (MSVC)
    add_compile_definitions(_USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
endif()

# Legacy interface library kept for export-set compatibility (it is
# still part of `install(EXPORT tts-cppTargets)` below and downstream
# `find_package(tts-cpp)` consumers list it as a link dep). Body is
# intentionally empty: tts-cpp routes every backend SELECTION and
# capability query through the ggml-backend registry
# (`init_gpu_backend()` / `init_cpu_backend()` / `init_blas_backend()`
# in src/backend_selection.cpp) and does NOT call any
# `ggml_backend_<backend>_init` / `ggml_backend_is_<backend>` /
# `ggml_backend_vk_*` entry point directly — the registry walk +
# `ggml_backend_get_device` / `ggml_backend_dev_*` calls reach the
# right backend in both `GGML_BACKEND_DL=ON` (Android / Linux .so
# prebuild) and `GGML_BACKEND_DL=OFF` (static-link desktop) modes.
# Mirrors parakeet-cpp's `parakeet-backend-defs`.
add_library(tts-cpp-backend-defs INTERFACE)

set(TTS_CPP_LIB_SOURCES
    src/backend_selection.cpp
    src/main.cpp
    src/chatterbox_cli.cpp
    src/gpt2_bpe.cpp
    src/chatterbox_tts.cpp
    src/chatterbox_engine.cpp
    src/log.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp
    src/voice_encoder.cpp
    src/campplus.cpp
    src/s3tokenizer.cpp
    src/t3_mtl.cpp
    src/supertonic_gguf.cpp
    src/supertonic_preprocess.cpp
    src/supertonic_vocoder.cpp
    src/supertonic_duration.cpp
    src/supertonic_text_encoder.cpp
    src/supertonic_vector_estimator.cpp
    src/supertonic_engine.cpp
    src/supertonic_chunker.cpp
    src/mtl_tokenizer.cpp
    src/text_preprocess.cpp
)

if (TTS_CPP_BUILD_LIBRARY)
    # Decoupled from BUILD_SHARED_LIBS by design.  parakeet.cpp's same-
    # repo sibling re-couples (`if (DEFINED BUILD_SHARED_LIBS)`), but
    # that pattern silently breaks on the second configure: ggml's own
    # add_subdirectory(ggml) declares option(BUILD_SHARED_LIBS ...) which
    # populates the cache with a platform-dependent default (Windows
    # non-MinGW: ON), and any subsequent DEFINED check returns TRUE
    # regardless of whether the user explicitly asked for shared linkage.
    # Empirically: cmake -B build (1st run, no -D) -> STATIC; cmake -B
    # build (2nd run, no -D) -> SHARED.  Project-namespaced option keeps
    # the two strictly independent.  See PR review item #15.
    #
    # SHARED hides the detail-namespaced symbols that the test harnesses
    # link against directly (test-mtl-tokenizer, test-supertonic-*); the
    # CXX_VISIBILITY_PRESET=hidden setting added below makes those
    # symbols invisible at SHARED-link time.  Disable the tests
    # (TTS_CPP_BUILD_TESTS=OFF) when building shared.
    if (TTS_CPP_BUILD_SHARED)
        add_library(tts-cpp SHARED ${TTS_CPP_LIB_SOURCES})
    else()
        add_library(tts-cpp STATIC ${TTS_CPP_LIB_SOURCES})
    endif()
    add_library(tts-cpp::tts-cpp ALIAS tts-cpp)
    tts_cpp_apply_ccache(tts-cpp)

    # Hide every symbol in the .so/.dll by default; only those tagged with
    # TTS_CPP_API (see include/tts-cpp/export.h) end up exported.  Mirrors
    # parakeet.cpp's pattern and shrinks the runtime dynamic-symbol table to
    # just the public API surface, which is also what lets a host that
    # vendors libtts-cpp.a inside its own .so see only one tts_cpp::* surface
    # instead of every detail-namespaced helper.
    set_target_properties(tts-cpp PROPERTIES
        CXX_VISIBILITY_PRESET     hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    target_include_directories(tts-cpp
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
            ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src
    )
    target_compile_features(tts-cpp PUBLIC cxx_std_17)
    target_link_libraries(tts-cpp PUBLIC ggml)

    # ------------------------------------------------------------------
    # Optional native preprocessors for the MTL non-Latin tier:
    #   * `ja` -- libmecab (dictionary path passed at runtime)
    #   * `zh` -- Cangjie5_TC mapping (TSV path passed at runtime)
    #
    # libmecab is autodetected via find_library / find_path. The
    # dictionary path is no longer baked in at compile time; callers
    # pass it at runtime via --mecab-dict (CLI) or
    # EngineOptions::mecab_dict_path (API). Use
    # scripts/build_mecab_dict.py to materialize the IPAdic dictionary
    # from the `ipadic` PyPI package.
    #
    # Cangjie no longer requires a compile-time TSV path; callers
    # pass it at runtime via --cangjie-tsv (CLI) or
    # EngineOptions::cangjie_tsv_path (API).
    # ------------------------------------------------------------------
    find_library(MECAB_LIBRARY    NAMES mecab)
    find_path   (MECAB_INCLUDE_DIR NAMES mecab.h)
    if (MECAB_LIBRARY AND MECAB_INCLUDE_DIR)
        message(STATUS "MeCab (Japanese preprocess): ENABLED")
        message(STATUS "  library : ${MECAB_LIBRARY}")
        message(STATUS "  include : ${MECAB_INCLUDE_DIR}")
        # MeCab is an implementation detail of src/text_preprocess.cpp; no
        # public header under include/ exposes MeCab types or symbols. Keep
        # the define, include dir and link all PRIVATE so consumers doing
        # find_package(tts-cpp) don't inherit a libmecab dependency (and so
        # the MinGW mecab.h headers never leak into a consumer's TU).
        target_compile_definitions(tts-cpp PRIVATE TTS_CPP_HAS_MECAB)
        target_include_directories(tts-cpp PRIVATE ${MECAB_INCLUDE_DIR})
        target_link_libraries     (tts-cpp PRIVATE ${MECAB_LIBRARY})
    else()
        message(STATUS "MeCab (Japanese preprocess): DISABLED (libmecab / mecab.h not found)")
    endif()

    # Symbol-export contract for shared builds.  Standard llama-style
    # PAIR: PUBLIC define so consumers see the import side of TTS_CPP_API
    # (declspec(dllimport) on Windows / visibility("default") elsewhere),
    # PRIVATE define inside the library's TUs to flip Windows from
    # dllimport to dllexport.  Both no-ops in static builds (the
    # default), so static consumers see exactly the same surface.
    get_target_property(_tts_cpp_type tts-cpp TYPE)
    if (_tts_cpp_type STREQUAL "SHARED_LIBRARY")
        target_compile_definitions(tts-cpp PUBLIC  TTS_CPP_SHARED)
        target_compile_definitions(tts-cpp PRIVATE TTS_CPP_BUILD)
    endif()

    # Versioning: VERSION drives the SONAME suffix on Linux/macOS shared
    # builds; SOVERSION 0 keeps the load-time name (libtts-cpp.so.0)
    # stable across patch releases so consumers don't relink on every
    # bump.  No effect on static / Windows builds.  Mirrors parakeet.cpp.
    set_target_properties(tts-cpp PROPERTIES
        VERSION   ${PROJECT_VERSION}
        SOVERSION 0
    )

    # Release-mode optimisation flags.  GCC/clang use -O3 in CMake's
    # default Release config already, but ffast-math + funroll-loops
    # buy ~5-15% on the T3-dominated CPU path (~80% of wall time on
    # M3 CPU per README's benchmarks).  Conditioned on non-MSVC; the
    # MSVC default /O2 is good enough and these are GCC-syntax flags.
    if (NOT MSVC AND (CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE))
        target_compile_options(tts-cpp PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:C>:-O3>
            $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
            $<$<COMPILE_LANGUAGE:C>:-ffast-math>
            $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>
            $<$<COMPILE_LANGUAGE:C>:-funroll-loops>
        )
    endif()

    if (OpenMP_CXX_FOUND)
        target_link_libraries(tts-cpp PRIVATE OpenMP::OpenMP_CXX)
    endif()

    target_link_libraries(tts-cpp PRIVATE tts-cpp-backend-defs)

    set_target_properties(tts-cpp PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME tts-cpp
    )
endif()

if (TTS_CPP_BUILD_EXECUTABLES)
    if (NOT TTS_CPP_BUILD_LIBRARY)
        message(FATAL_ERROR "TTS_CPP_BUILD_EXECUTABLES requires TTS_CPP_BUILD_LIBRARY=ON")
    endif()

    add_executable(tts-cli src/cli_main.cpp)
    target_link_libraries(tts-cli PRIVATE tts-cpp)
    tts_cpp_apply_ccache(tts-cli)

    add_executable(mel2wav src/mel2wav.cpp)
    target_link_libraries(mel2wav PRIVATE ggml)
    target_include_directories(mel2wav PRIVATE ggml/include src)
    tts_cpp_apply_ccache(mel2wav)

    add_executable(supertonic-cli src/supertonic_cli.cpp)
    target_link_libraries(supertonic-cli PRIVATE tts-cpp)
    tts_cpp_apply_ccache(supertonic-cli)
endif()

if (TTS_CPP_BUILD_TESTS)
    enable_testing()
    include(CTest)

    # Roots that the test registrations below resolve fixture paths against.
    # All three are CACHE PATHs so CI / package builds can point them at
    # pre-staged mirrors (e.g. a tarball of `.gguf`s + reference `.npy`s)
    # without editing this listfile.
    #
    #   MODEL_DIR  - GGUF checkpoints (produced by scripts/convert-*-to-gguf.py
    #                or `bash scripts/setup-supertonic2.sh`)
    #   AUDIO_DIR  - reference WAV fixtures (test/reference-audio/jfk.wav, ...)
    #   REF_DIR    - PyTorch / ONNX reference dumps (`.npy`) produced by
    #                scripts/dump-{s3gen,supertonic,t3-mtl}-reference.py;
    #                expected layout is REF_DIR/{s3gen-ref,supertonic-ref-quick,t3-mtl-ref}/.
    #                Tests that need a ref dump auto-disable when the dump
    #                isn't present, so a fresh checkout still gives a green
    #                ctest run on the harnesses whose fixtures it does have.
    set(TTS_CPP_TEST_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models"               CACHE PATH "tts-cpp: directory containing .gguf fixtures used by the test suite")
    set(TTS_CPP_TEST_AUDIO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test/reference-audio" CACHE PATH "tts-cpp: directory containing .wav fixtures used by the test suite")
    set(TTS_CPP_TEST_REF_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/artifacts"            CACHE PATH "tts-cpp: directory containing PyTorch / ONNX .npy reference dumps")

    # Helper: register a built test target with CTest.  Mirrors the
    # standard llama-style `<project>_test()` macro: every harness ends
    # up surfaced via `ctest` against the build dir, with labels that
    # allow filtering (e.g. `ctest -L cpu` to skip GPU-only tests on CI
    # fleets without the matching backend).  The optional REQUIRES list
    # names paths that must exist at configure time; if any is missing
    # the test is still registered (so it shows up in `ctest -N`) but
    # marked DISABLED so a missing reference dump shows as "Not Run"
    # rather than a hard failure.
    function(tts_cpp_register_test target)
        cmake_parse_arguments(TCRT "" "LABEL;EXE" "ARGS;REQUIRES" ${ARGN})
        if (NOT DEFINED TCRT_LABEL)
            set(TCRT_LABEL "cpu")
        endif()
        # EXE defaults to the test name when the test has its own
        # executable target.  Pass an explicit EXE when several tests
        # share one binary and differ only in CLI args / fixtures
        # (e.g. test-cpu-caches-{turbo,mtl} all invoke test-cpu-caches.exe).
        if (NOT DEFINED TCRT_EXE)
            set(TCRT_EXE ${target})
        endif()
        add_test(NAME ${target}
                 COMMAND $<TARGET_FILE:${TCRT_EXE}> ${TCRT_ARGS})
        set_property(TEST ${target} PROPERTY LABELS ${TCRT_LABEL})
        if (WIN32)
            # When ggml builds as shared (the MSVC default), test exes
            # link against import libs and need to find speech-ggml-*.dll
            # at runtime.  Windows has no rpath, so we prepend the
            # build-tree DLL output dirs to PATH.  Both `bin/` (Ninja /
            # single-config) and `bin/<Config>/` (Visual Studio /
            # multi-config) are listed because the generator type isn't
            # known at configure time on a fresh checkout; the
            # generator-expression $<CONFIG> resolves to the active
            # config at test time.  Without this, ctest sees
            # STATUS_DLL_NOT_FOUND (0xc0000135) before main() runs.
            set_property(TEST ${target} PROPERTY ENVIRONMENT
                "PATH=${CMAKE_BINARY_DIR}/bin/$<CONFIG>\;${CMAKE_BINARY_DIR}/bin\;$ENV{PATH}")
        endif()
        set(_tcrt_missing "")
        foreach (_tcrt_req IN LISTS TCRT_REQUIRES)
            if (NOT EXISTS "${_tcrt_req}")
                list(APPEND _tcrt_missing "${_tcrt_req}")
            endif()
        endforeach()
        if (_tcrt_missing)
            set_property(TEST ${target} PROPERTY DISABLED TRUE)
            list(JOIN _tcrt_missing "\n        " _tcrt_missing_pretty)
            message(STATUS "tts-cpp: test ${target} disabled (missing fixture(s):\n        ${_tcrt_missing_pretty})")
        endif()
    endfunction()

    # Stable shorthands for the fixtures referenced by the registrations
    # below.  Resolved at configure time; missing files turn into
    # DISABLED tests via tts_cpp_register_test(REQUIRES ...).
    set(_tcb_t3_turbo_gguf      "${TTS_CPP_TEST_MODEL_DIR}/chatterbox-t3-turbo.gguf")
    set(_tcb_t3_mtl_gguf        "${TTS_CPP_TEST_MODEL_DIR}/chatterbox-t3-mtl.gguf")
    set(_tcb_s3gen_gguf         "${TTS_CPP_TEST_MODEL_DIR}/chatterbox-s3gen.gguf")
    set(_tcb_s3gen_mtl_gguf     "${TTS_CPP_TEST_MODEL_DIR}/chatterbox-s3gen-mtl.gguf")
    set(_tcb_super_gguf         "${TTS_CPP_TEST_MODEL_DIR}/supertonic.gguf")
    set(_tcb_super_ref          "${TTS_CPP_TEST_REF_DIR}/supertonic-ref-quick")
    set(_tcb_super_ref_noise    "${_tcb_super_ref}/noise.npy")
    set(_tcb_s3gen_ref          "${TTS_CPP_TEST_REF_DIR}/s3gen-ref")
    set(_tcb_s3gen_ref_logits   "${_tcb_s3gen_ref}/logits.npy")
    set(_tcb_t3_mtl_ref         "${TTS_CPP_TEST_REF_DIR}/t3-mtl-ref")
    set(_tcb_streaming_ref      "${TTS_CPP_TEST_REF_DIR}/streaming-ref")
    set(_tcb_jfk_wav            "${TTS_CPP_TEST_AUDIO_DIR}/jfk.wav")

    add_executable(test-s3gen test/test_s3gen.cpp)
    target_link_libraries(test-s3gen PRIVATE ggml)
    target_include_directories(test-s3gen PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-s3gen)
    tts_cpp_register_test(test-s3gen
        LABEL    "fixture"
        ARGS     "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref}" "ALL"
        REQUIRES "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref_logits}")

    add_executable(test-voice-features
        test/test_voice_features.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-voice-features PRIVATE ggml)
    target_include_directories(test-voice-features PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-voice-features)
    tts_cpp_register_test(test-voice-features
        LABEL    "fixture"
        ARGS     "${_tcb_s3gen_gguf}" "${_tcb_jfk_wav}" "${_tcb_s3gen_ref}/prompt_feat.npy"
        REQUIRES "${_tcb_s3gen_gguf}" "${_tcb_jfk_wav}" "${_tcb_s3gen_ref}/prompt_feat.npy")

    add_executable(test-resample
        test/test_resample.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-resample PRIVATE ggml)
    target_include_directories(test-resample PRIVATE src)
    tts_cpp_apply_ccache(test-resample)
    tts_cpp_register_test(test-resample LABEL "unit")

    add_executable(test-voice-encoder
        test/test_voice_encoder.cpp
        src/voice_encoder.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-voice-encoder PRIVATE ggml)
    target_include_directories(test-voice-encoder PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-voice-encoder)
    tts_cpp_register_test(test-voice-encoder
        LABEL    "fixture"
        ARGS     "${_tcb_t3_turbo_gguf}" "${_tcb_jfk_wav}" "${_tcb_s3gen_ref}/speaker_emb.npy"
        REQUIRES "${_tcb_t3_turbo_gguf}" "${_tcb_jfk_wav}" "${_tcb_s3gen_ref}/speaker_emb.npy")

    add_executable(test-campplus
        test/test_campplus.cpp
        src/campplus.cpp)
    target_link_libraries(test-campplus PRIVATE ggml)
    target_include_directories(test-campplus PRIVATE ggml/include src)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(test-campplus PRIVATE OpenMP::OpenMP_CXX)
    endif()
    tts_cpp_apply_ccache(test-campplus)
    tts_cpp_register_test(test-campplus
        LABEL    "fixture"
        ARGS     "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref}/fbank.npy" "${_tcb_s3gen_ref}/embedding.npy"
        REQUIRES "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref}/fbank.npy" "${_tcb_s3gen_ref}/embedding.npy")

    add_executable(test-fbank
        test/test_fbank.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-fbank PRIVATE ggml)
    target_include_directories(test-fbank PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-fbank)
    tts_cpp_register_test(test-fbank
        LABEL    "fixture"
        ARGS     "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref}/wav_16k.npy" "${_tcb_s3gen_ref}/fbank.npy"
        REQUIRES "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref}/wav_16k.npy" "${_tcb_s3gen_ref}/fbank.npy")

    add_executable(test-voice-embedding
        test/test_voice_embedding.cpp
        src/campplus.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-voice-embedding PRIVATE ggml)
    target_include_directories(test-voice-embedding PRIVATE ggml/include src)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(test-voice-embedding PRIVATE OpenMP::OpenMP_CXX)
    endif()
    tts_cpp_apply_ccache(test-voice-embedding)
    tts_cpp_register_test(test-voice-embedding
        LABEL    "fixture"
        ARGS     "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref}/wav_16k.npy" "${_tcb_s3gen_ref}/embedding.npy"
        REQUIRES "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref}/wav_16k.npy" "${_tcb_s3gen_ref}/embedding.npy")

    add_executable(test-s3tokenizer
        test/test_s3tokenizer.cpp
        src/s3tokenizer.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-s3tokenizer PRIVATE ggml)
    target_include_directories(test-s3tokenizer PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-s3tokenizer)
    tts_cpp_register_test(test-s3tokenizer
        LABEL    "fixture"
        ARGS     "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref}/wav_16k.npy" "${_tcb_s3gen_ref}/log_mel.npy" "${_tcb_s3gen_ref}/speech_tokens.npy"
        REQUIRES "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref}/wav_16k.npy" "${_tcb_s3gen_ref}/log_mel.npy" "${_tcb_s3gen_ref}/speech_tokens.npy")

    add_executable(test-mtl-tokenizer test/test_mtl_tokenizer.cpp)
    target_link_libraries(test-mtl-tokenizer PRIVATE tts-cpp)
    target_include_directories(test-mtl-tokenizer PRIVATE src)
    tts_cpp_apply_ccache(test-mtl-tokenizer)
    tts_cpp_register_test(test-mtl-tokenizer
        LABEL    "fixture"
        ARGS     "${TTS_CPP_TEST_REF_DIR}/mtl-tokenizer/tokenizer.json" "${TTS_CPP_TEST_REF_DIR}/mtl-tokenizer/golden.json"
        REQUIRES "${TTS_CPP_TEST_REF_DIR}/mtl-tokenizer/tokenizer.json" "${TTS_CPP_TEST_REF_DIR}/mtl-tokenizer/golden.json")

    add_executable(test-t3-mtl test/test_t3_mtl.cpp)
    target_link_libraries(test-t3-mtl PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-t3-mtl PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-t3-mtl)
    tts_cpp_register_test(test-t3-mtl
        LABEL    "fixture"
        ARGS     "${_tcb_t3_mtl_gguf}" "${_tcb_t3_mtl_ref}"
        REQUIRES "${_tcb_t3_mtl_gguf}" "${_tcb_t3_mtl_ref}")

    add_executable(test-t3-mtl-stages test/test_t3_mtl_stages.cpp)
    target_link_libraries(test-t3-mtl-stages PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-t3-mtl-stages PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-t3-mtl-stages)
    tts_cpp_register_test(test-t3-mtl-stages
        LABEL    "fixture"
        ARGS     "${_tcb_t3_mtl_gguf}" "${_tcb_t3_mtl_ref}" "ALL"
        REQUIRES "${_tcb_t3_mtl_gguf}" "${_tcb_t3_mtl_ref}")

    # ------------------------------------------------------------------
    # Multilingual integration tests (23 languages).
    # ------------------------------------------------------------------
    add_executable(test-multilingual-synth test/test_multilingual_synth.cpp)
    target_link_libraries(test-multilingual-synth PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-multilingual-synth PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-multilingual-synth)

    set(_mtl_phrases
        "en|Hello, this is a multilingual text-to-speech test.|tier1"
        "es|Hola, esta es una prueba multilingüe de síntesis de voz.|tier1"
        "fr|Bonjour, ceci est un test multilingue de synthèse vocale.|tier1"
        "de|Hallo, dies ist ein mehrsprachiger Sprachsynthese-Test.|tier1"
        "it|Ciao, questo è un test multilingue di sintesi vocale.|tier1"
        "pt|Olá, este é um teste multilíngue de síntese de voz.|tier1"
        "nl|Hallo, dit is een meertalige spraaksynthese-test.|tier1"
        "pl|Cześć, to jest wielojęzyczny test syntezy mowy.|tier1"
        "tr|Merhaba, bu çok dilli bir konuşma sentezi testidir.|tier1"
        "sv|Hej, detta är ett flerspråkigt talsyntestest.|tier1"
        "da|Hej, dette er en flersproget talesyntese-test.|tier1"
        "fi|Hei, tämä on monikielinen puhesynteesitesti.|tier1"
        "no|Hei, dette er en flerspråklig talesyntese-test.|tier1"
        "el|Γεια σας, αυτή είναι μια πολύγλωσση δοκιμή σύνθεσης ομιλίας.|tier1"
        "ms|Helo, ini ialah ujian sintesis pertuturan berbilang bahasa.|tier1"
        "sw|Habari, huu ni mtihani wa usanisi wa sauti wa lugha nyingi.|tier1"
        "ar|مرحبًا، هذا اختبار متعدد اللغات لتحويل النص إلى كلام.|tier1"
        "ko|안녕하세요, 이것은 다국어 음성 합성 테스트입니다.|tier1"
        "ja|こんにちは、これは多言語の音声合成テストです。|tier2"
        "he|שלום, זוהי בדיקת סינתזת דיבור רב-לשונית.|tier2"
        "ru|Привет, это многоязычный тест синтеза речи.|tier2"
        # zh intentionally omitted: Cangjie preprocessing yields ~97% CER and
        # is excluded from mtl_tokenizer::supported_languages() until reworked.
        "hi|नमस्ते, यह एक बहुभाषी वाक् संश्लेषण परीक्षण है।|tier2"
    )
    set(_mtl_out_dir "${CMAKE_BINARY_DIR}/test-multilingual-out")
    file(MAKE_DIRECTORY "${_mtl_out_dir}")

    # Optional L3 (ASR) verification via whisper-cli.
    find_program(WHISPER_CLI NAMES whisper-cli)
    set(WHISPER_MODEL "${CMAKE_SOURCE_DIR}/models/ggml-large-v3-turbo-q5_0.bin"
        CACHE FILEPATH "Path to a multilingual Whisper ggml model used by mtl-asr-* tests")
    if (WHISPER_CLI AND EXISTS "${WHISPER_MODEL}")
        set(_mtl_asr_enabled ON)
        message(STATUS "Multilingual L3 (ASR) tests: ENABLED")
        message(STATUS "  whisper-cli : ${WHISPER_CLI}")
        message(STATUS "  model       : ${WHISPER_MODEL}")
        add_executable(test-multilingual-asr test/test_multilingual_asr.cpp)
        target_include_directories(test-multilingual-asr PRIVATE src include)
        tts_cpp_apply_ccache(test-multilingual-asr)
    else()
        set(_mtl_asr_enabled OFF)
        message(STATUS "Multilingual L3 (ASR) tests: SKIPPED")
        if (NOT WHISPER_CLI)
            message(STATUS "  reason: whisper-cli not found on PATH")
        endif()
        if (NOT EXISTS "${WHISPER_MODEL}")
            message(STATUS "  reason: model missing at ${WHISPER_MODEL}")
        endif()
    endif()

    foreach(_entry IN LISTS _mtl_phrases)
        string(REPLACE "|" ";" _parts "${_entry}")
        list(GET _parts 0 _lang)
        list(GET _parts 1 _text)
        list(GET _parts 2 _tier)
        add_test(
            NAME mtl-synth-${_lang}
            COMMAND $<TARGET_FILE:test-multilingual-synth>
                --lang  "${_lang}"
                --text  "${_text}"
                --out   "${_mtl_out_dir}/${_lang}.wav"
        )
        set_tests_properties(mtl-synth-${_lang} PROPERTIES
            LABELS "multilingual;mtl-${_tier}"
            TIMEOUT 180
        )
        if (_mtl_asr_enabled)
            add_test(
                NAME mtl-asr-${_lang}
                COMMAND $<TARGET_FILE:test-multilingual-asr>
                    --whisper-cli "${WHISPER_CLI}"
                    --model       "${WHISPER_MODEL}"
                    --lang        "${_lang}"
                    --wav         "${_mtl_out_dir}/${_lang}.wav"
                    --expected    "${_text}"
                    --txt-prefix  "${_mtl_out_dir}/${_lang}.asr"
            )
            set_tests_properties(mtl-asr-${_lang} PROPERTIES
                LABELS  "multilingual;asr;mtl-${_tier}-asr"
                DEPENDS mtl-synth-${_lang}
                TIMEOUT 120
            )
        endif()
    endforeach()

    add_executable(test-streaming
        test/test_streaming.cpp
        src/chatterbox_tts.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-streaming PRIVATE ggml tts-cpp-backend-defs)
    target_include_directories(test-streaming PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-streaming)
    tts_cpp_register_test(test-streaming
        LABEL    "fixture"
        ARGS     "${_tcb_s3gen_gguf}" "${_tcb_streaming_ref}"
        REQUIRES "${_tcb_s3gen_gguf}" "${_tcb_streaming_ref}")

    # CPU-side persistent-cache validation.
    # Exercises the time_mlp / time_emb / cfm_estimator / weight_mirror
    # caches that amortise per-synth overhead on the multilingual CPU
    # path.  Links chatterbox_tts.cpp directly so it can reach the
    # internal test-hook entrypoints.
    add_executable(test-cpu-caches
        test/test_cpu_caches.cpp
        src/chatterbox_tts.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-cpu-caches PRIVATE ggml tts-cpp-backend-defs)
    target_include_directories(test-cpu-caches PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-cpu-caches)
    # No-model invocation: bit-cast cache-key + initial-state checks.
    tts_cpp_register_test(test-cpu-caches LABEL "unit")
    # Per-variant warm-cache + bit-exact pipeline checks.  Each shares
    # the same exe but takes a different GGUF; auto-disabled when its
    # GGUF isn't present.
    tts_cpp_register_test(test-cpu-caches-turbo
        LABEL    "fixture"
        EXE      test-cpu-caches
        ARGS     "${_tcb_s3gen_gguf}"
        REQUIRES "${_tcb_s3gen_gguf}")
    tts_cpp_register_test(test-cpu-caches-mtl
        LABEL    "fixture"
        EXE      test-cpu-caches
        ARGS     "${_tcb_s3gen_mtl_gguf}"
        REQUIRES "${_tcb_s3gen_mtl_gguf}")

    # T3 step-graph cache validation.
    add_executable(test-t3-caches test/test_t3_caches.cpp)
    target_link_libraries(test-t3-caches PRIVATE tts-cpp ggml)
    target_include_directories(test-t3-caches PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-t3-caches)
    tts_cpp_register_test(test-t3-caches LABEL "unit")
    tts_cpp_register_test(test-t3-caches-mtl
        LABEL    "fixture"
        EXE      test-t3-caches
        ARGS     "${_tcb_t3_mtl_gguf}"
        REQUIRES "${_tcb_t3_mtl_gguf}")

    add_executable(test-metal-ops test/test_metal_ops.cpp)
    target_link_libraries(test-metal-ops PRIVATE ggml tts-cpp-backend-defs)
    target_include_directories(test-metal-ops PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-metal-ops)
    # Metal-only kernel parity check.  Useful only when built with
    # -DGGML_METAL=ON; skipped on CI fleets without Metal via `ctest -LE gpu`.
    # GGML_USE_METAL is supplied locally here (rather than via
    # tts-cpp-backend-defs) because the library itself no longer
    # consumes the macro -- every #ifdef GGML_USE_<X> in src/ was
    # removed alongside the registry-only refactor. The test still
    # uses the macro to gate its direct ggml_backend_metal_init()
    # call site (it's exercising the Metal-backend implementation
    # directly, not going through tts-cpp's backend selection).
    if (GGML_METAL)
        target_compile_definitions(test-metal-ops PRIVATE GGML_USE_METAL)
        tts_cpp_register_test(test-metal-ops LABEL "gpu")
    endif()

    if (NOT TTS_CPP_BUILD_LIBRARY)
        message(FATAL_ERROR "Supertonic tests require TTS_CPP_BUILD_LIBRARY=ON")
    endif()

    function(add_supertonic_harness target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE tts-cpp)
        target_include_directories(${target} PRIVATE ggml/include src include)
        tts_cpp_apply_ccache(${target})
        # Every Supertonic harness takes (MODEL.gguf REF_DIR) and is
        # auto-disabled when either is absent.  REQUIRES checks the dir
        # plus a sentinel file inside it (noise.npy is produced by every
        # `dump-supertonic-reference.py` run) so the test isn't merely
        # gated on an empty dir existing.
        tts_cpp_register_test(${target}
            LABEL    "fixture"
            ARGS     "${_tcb_super_gguf}" "${_tcb_super_ref}"
            REQUIRES "${_tcb_super_gguf}" "${_tcb_super_ref_noise}")
    endfunction()

    add_supertonic_harness(test-supertonic-preprocess test/test_supertonic_preprocess.cpp)
    add_supertonic_harness(test-supertonic-vocoder test/test_supertonic_vocoder.cpp)
    add_supertonic_harness(test-supertonic-vocoder-trace test/test_supertonic_vocoder_trace.cpp)
    add_supertonic_harness(test-supertonic-vocoder-pointwise test/test_supertonic_vocoder_pointwise.cpp)
    add_supertonic_harness(test-supertonic-duration test/test_supertonic_duration.cpp)
    add_supertonic_harness(test-supertonic-duration-trace test/test_supertonic_duration_trace.cpp)
    add_supertonic_harness(test-supertonic-text-encoder test/test_supertonic_text_encoder.cpp)
    add_supertonic_harness(test-supertonic-text-encoder-trace test/test_supertonic_text_encoder_trace.cpp)
    add_supertonic_harness(test-supertonic-vector test/test_supertonic_vector.cpp)
    add_supertonic_harness(test-supertonic-vector-trace test/test_supertonic_vector_trace.cpp)
    add_supertonic_harness(test-supertonic-pipeline test/test_supertonic_pipeline.cpp)
    add_supertonic_harness(test-supertonic-engine-cycle test/test_supertonic_engine_cycle.cpp)

    # ------------------------------------------------------------------
    # QVAC-19305 — Supertonic 3 fixture suite.
    #
    # The harnesses above are arch-agnostic: they take (MODEL.gguf REF_DIR)
    # and don't care which Supertonic generation produced them.  We re-use
    # those same binaries (via `EXE`) to also run them against a Supertonic
    # 3 GGUF plus per-language ONNX-Runtime references.  Stage the data with
    #     scripts/fetch-supertonic3-testdata.sh
    # (or, once published, the QVAC S3 model registry) and point the build
    # at it:
    #     -DTTS_CPP_TEST_MODEL_DIR=<staging>/models
    #     -DTTS_CPP_TEST_REF_DIR=<staging>/artifacts
    # Each registration auto-DISABLEs when its GGUF / reference dump is
    # missing, exactly like the v2 fixtures, so a checkout without the data
    # still configures cleanly.
    #
    # We deliberately run the *Supertonic 2* language set (en/ko/es/pt/fr)
    # through the v3 model — the v3-only languages are covered by the
    # model-free test-supertonic-languages unit test — so that both the old
    # and the new languages get exercised end-to-end on Supertonic 3.
    set(_tcb_super3_gguf "${TTS_CPP_TEST_MODEL_DIR}/supertonic3.gguf")
    set(SUPERTONIC3_REF_LANGS en ko es pt fr
        CACHE STRING "tts-cpp: languages exercised end-to-end against the Supertonic 3 fixture")

    # Per-stage parity on English (the default reference language).  The
    # full pipeline is covered per-language below, so it is omitted here.
    foreach (_st3 IN ITEMS
            test-supertonic-preprocess
            test-supertonic-text-encoder
            test-supertonic-vector
            test-supertonic-vocoder
            test-supertonic-duration)
        set(_st3_ref "${TTS_CPP_TEST_REF_DIR}/supertonic3-ref-en")
        tts_cpp_register_test(${_st3}-v3
            EXE      ${_st3}
            LABEL    "fixture"
            ARGS     "${_tcb_super3_gguf}" "${_st3_ref}"
            REQUIRES "${_tcb_super3_gguf}" "${_st3_ref}/noise.npy")
    endforeach()

    # Full-pipeline parity per language — old + new languages on Supertonic 3.
    foreach (_st3lang IN LISTS SUPERTONIC3_REF_LANGS)
        set(_st3_ref "${TTS_CPP_TEST_REF_DIR}/supertonic3-ref-${_st3lang}")
        tts_cpp_register_test(test-supertonic-pipeline-v3-${_st3lang}
            EXE      test-supertonic-pipeline
            LABEL    "fixture"
            ARGS     "${_tcb_super3_gguf}" "${_st3_ref}"
            REQUIRES "${_tcb_super3_gguf}" "${_st3_ref}/noise.npy")
    endforeach()

    # OpenCL optimization audit follow-up harnesses (F1–F11).
    add_supertonic_harness(test-supertonic-load-caches    test/test_supertonic_load_caches.cpp)
    add_supertonic_harness(test-supertonic-graph-rewrites test/test_supertonic_graph_rewrites.cpp)
    # OpenCL audit follow-up #2 — text-encoder caches (F13, F16),
    # Phase 2A F16-weight roster (predicate-level), Phase 2D
    # profile-CSV emitter (unit-only).
    add_supertonic_harness(test-supertonic-text-encoder-caches
        test/test_supertonic_text_encoder_caches.cpp)
    add_supertonic_harness(test-supertonic-f16-weights
        test/test_supertonic_f16_weights.cpp)
    # Phase 2D profile-CSV emitter — unit-level, no GGUF needed.
    add_executable(test-supertonic-profile-csv
        test/test_supertonic_profile_csv.cpp)
    target_link_libraries(test-supertonic-profile-csv PRIVATE tts-cpp)
    target_include_directories(test-supertonic-profile-csv PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-profile-csv)
    tts_cpp_register_test(test-supertonic-profile-csv LABEL "unit")
    # OpenCL audit follow-up #3 — F17 duration scalar-weight
    # cache + F18 text-encoder convnext-front graph cache +
    # F19 vector-estimator front-block graph cache.
    add_supertonic_harness(test-supertonic-audit3-caches
        test/test_supertonic_audit3_caches.cpp)
    # OpenCL audit follow-up #4 — F20 partial / Phase 2H RoPE-in-
    # graph helper (parity vs scalar apply_rope on CPU backend with
    # synthetic input).  Unit-level — no GGUF, no fixture.
    add_executable(test-supertonic-rope-in-graph
        test/test_supertonic_rope_in_graph.cpp)
    target_link_libraries(test-supertonic-rope-in-graph PRIVATE ggml)
    target_include_directories(test-supertonic-rope-in-graph PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-supertonic-rope-in-graph)
    tts_cpp_register_test(test-supertonic-rope-in-graph LABEL "unit")

    # Audit follow-up #5 — packed-QK RoPE adapter parity test for
    # `apply_rope_to_packed_qk` (F23 = F20 integration shim).  The
    # helper bridges the `[head_dim, n_heads, L]` layout consumed
    # by `apply_rope_in_graph` with the `[H*D, L]` packed layout
    # produced by `dense_matmul_time_ggml` — see
    # `aiDocs/AUDIT_SUPERTONIC_OPENCL.md` finding F23.  Unit-level:
    # CPU-only parity, no GGUF, no fixture; runs in <50 ms.
    add_executable(test-supertonic-rope-packed-qk
        test/test_supertonic_rope_packed_qk.cpp)
    target_link_libraries(test-supertonic-rope-packed-qk PRIVATE ggml)
    target_include_directories(test-supertonic-rope-packed-qk PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-supertonic-rope-packed-qk)
    tts_cpp_register_test(test-supertonic-rope-packed-qk LABEL "unit")

    # Audit follow-up #6 (F7) — fused ConvNeXt block builder.  The
    # helper rewires the vocoder's per-block LN + pw1 + gelu + pw2 +
    # gamma + residual chain to skip the layer-norm back-permute and
    # to lower K=1 pointwise convs to direct `ggml_mul_mat` against
    # the `[C, T0]` LN-output layout, eliminating two redundant
    # `[T0, C]` copies per block (~16.8 MiB / vocoder pass).  Unit-
    # level: CPU-only parity vs scalar reference on synthetic
    # weights; no GGUF, no fixture; runs in <50 ms.
    add_executable(test-supertonic-convnext-block-fused
        test/test_supertonic_convnext_block_fused.cpp)
    target_link_libraries(test-supertonic-convnext-block-fused PRIVATE ggml)
    target_include_directories(test-supertonic-convnext-block-fused PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-supertonic-convnext-block-fused)
    tts_cpp_register_test(test-supertonic-convnext-block-fused LABEL "unit")

    # Audit follow-up #6 (F12) — in-graph time/channel transpose
    # helper to kill the per-call `pack_time_channel_for_ggml`
    # CPU loops at every vector / text / duration estimator cache
    # ingestion point.  The helper exposes `cache.x_in` as
    # `ne=[C, L]` so callers upload CPU-native `x_tc` directly,
    # and the graph immediately does `ggml_cont(ggml_transpose(x))`
    # to recover the `[L, C]` view downstream ops expect.  Unit-
    # level: CPU-only parity vs the reference `pack_time_channel_for_ggml`
    # on three shapes (group_graph, tail noise, vocoder-realistic)
    # + an L=1 trip-wire.  No GGUF needed; runs in <50 ms.
    add_executable(test-supertonic-in-graph-transpose
        test/test_supertonic_in_graph_transpose.cpp)
    target_link_libraries(test-supertonic-in-graph-transpose PRIVATE ggml)
    target_include_directories(test-supertonic-in-graph-transpose PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-supertonic-in-graph-transpose)
    tts_cpp_register_test(test-supertonic-in-graph-transpose LABEL "unit")

    # Audit follow-up #6 (2C-lite) — same-backend `ggml_backend_
    # tensor_copy` regression test.  Locks in the contract the
    # `run_text_attention_cache_gpu` fast path depends on: a
    # device→device blit between two cached graphs that share a
    # backend produces bit-exact output equivalent to the
    # `tensor_get` + `tensor_set` host round-trip the slow path
    # used to perform.  Five shapes including an L=1 trip-wire
    # and both attn / style head configurations.  Pure-CPU; no
    # GGUF; runs in <50 ms.
    add_executable(test-supertonic-graph-to-graph-blit
        test/test_supertonic_graph_to_graph_blit.cpp)
    target_link_libraries(test-supertonic-graph-to-graph-blit PRIVATE ggml)
    target_include_directories(test-supertonic-graph-to-graph-blit PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-supertonic-graph-to-graph-blit)
    tts_cpp_register_test(test-supertonic-graph-to-graph-blit LABEL "unit")

    # OpenCL bring-up unit tests (QVAC-18607).  Three CPU-only
    # parity / structural tests for the dispatch + portable-op
    # primitives.  No GGUF needed; register as "unit" label so a
    # fresh checkout's ctest exercises them.  Links against tts-cpp
    # (STATIC) so the detail-namespace symbols are reachable, same
    # pattern as test-mtl-tokenizer / test-t3-mtl / test-streaming.
    add_executable(test-supertonic-backend-dispatch
        test/test_supertonic_backend_dispatch.cpp)
    target_link_libraries(test-supertonic-backend-dispatch PRIVATE tts-cpp)
    target_include_directories(test-supertonic-backend-dispatch PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-backend-dispatch)
    tts_cpp_register_test(test-supertonic-backend-dispatch LABEL "unit")

    add_executable(test-supertonic-portable-ops
        test/test_supertonic_portable_ops.cpp)
    target_link_libraries(test-supertonic-portable-ops PRIVATE tts-cpp)
    target_include_directories(test-supertonic-portable-ops PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-portable-ops)
    tts_cpp_register_test(test-supertonic-portable-ops LABEL "unit")

    # QVAC-18605 — CPU-only unit test for the Vulkan-specific
    # dispatch additions: `backend_is_vk`, `use_native_leaky_relu`,
    # the `supertonic_op_dispatch_scope` mirror for the new flag,
    # and the `supertonic_backend_supports_f16_kv_flash_attn`
    # backend probe.  No GGUF / model fixture required — runs on a
    # fresh checkout under `ctest -L unit`.  See the file header
    # for the full coverage matrix.
    add_executable(test-supertonic-vulkan-dispatch
        test/test_supertonic_vulkan_dispatch.cpp)
    target_link_libraries(test-supertonic-vulkan-dispatch PRIVATE tts-cpp)
    target_include_directories(test-supertonic-vulkan-dispatch PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-vulkan-dispatch)
    tts_cpp_register_test(test-supertonic-vulkan-dispatch LABEL "unit")

    # QVAC-18605 follow-up — process-wide capability-probe cache +
    # F16 mul_mat probe + Q8_0 K/V flash-attn probe regression test.
    # CPU-only; runs on a fresh checkout under `ctest -L unit`.
    add_executable(test-supertonic-capability-cache
        test/test_supertonic_capability_cache.cpp)
    target_link_libraries(test-supertonic-capability-cache PRIVATE tts-cpp)
    target_include_directories(test-supertonic-capability-cache PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-capability-cache)
    tts_cpp_register_test(test-supertonic-capability-cache LABEL "unit")

    # QVAC-18605 follow-up — Engine::warm_up + EngineOptions::prewarm_text
    # API-surface lockdown.  CPU-only compile-time + runtime contract test;
    # the Vulkan-side first-synth-latency reduction is exercised by the
    # fixture-bound integration tests on a Vulkan-capable host.
    add_executable(test-supertonic-warm-up-api
        test/test_supertonic_warm_up_api.cpp)
    target_link_libraries(test-supertonic-warm-up-api PRIVATE tts-cpp)
    target_include_directories(test-supertonic-warm-up-api PRIVATE include)
    tts_cpp_apply_ccache(test-supertonic-warm-up-api)
    tts_cpp_register_test(test-supertonic-warm-up-api LABEL "unit")

    # QVAC-18605 round 3 — multi-device Vulkan auto-pick policy
    # (--vulkan-device -1 → pick device with most free VRAM).
    # CPU-only TDD test for the pure-logic helper; the Vulkan-only
    # plumbing that calls ggml_backend_vk_get_device_memory() per
    # device + dispatches into the helper is exercised by the
    # fixture-bound integration tests on a multi-GPU Vulkan host.
    add_executable(test-supertonic-vulkan-device-select
        test/test_supertonic_vulkan_device_select.cpp)
    target_link_libraries(test-supertonic-vulkan-device-select PRIVATE tts-cpp)
    target_include_directories(test-supertonic-vulkan-device-select PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-vulkan-device-select)
    tts_cpp_register_test(test-supertonic-vulkan-device-select LABEL "unit")

    # QVAC-18605 round 6 — F16-weights deny-list API surface
    # (EngineOptions::f16_weights_deny_list + load_supertonic_gguf
    # 7th parameter + 2-arg should_materialise_f16_weight overload).
    # CPU-only compile-time SFINAE + runtime defaults check; the
    # predicate-level behaviour is covered by the existing
    # test-supertonic-f16-weights TU.  The fixture-level shape /
    # dtype check (loads model with deny-list, verifies a denied
    # tensor stays F32) runs under the same fixture as the
    # baseline F16-weights test on hosts with the GGUF available.
    add_executable(test-supertonic-f16-deny-list-api
        test/test_supertonic_f16_deny_list_api.cpp)
    target_link_libraries(test-supertonic-f16-deny-list-api PRIVATE tts-cpp)
    target_include_directories(test-supertonic-f16-deny-list-api PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-f16-deny-list-api)
    tts_cpp_register_test(test-supertonic-f16-deny-list-api LABEL "unit")

    # QVAC-18605 round 4 — multi-dtype K/V flash-attention dispatch
    # resolver (`resolve_kv_attn_type`) — pure-logic policy split
    # from the Vulkan-only dispatch site so the behaviour matrix
    # is testable on CPU with synthetic probe inputs.
    add_executable(test-supertonic-kv-attn-type
        test/test_supertonic_kv_attn_type.cpp)
    target_link_libraries(test-supertonic-kv-attn-type PRIVATE tts-cpp)
    target_include_directories(test-supertonic-kv-attn-type PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-kv-attn-type)
    tts_cpp_register_test(test-supertonic-kv-attn-type LABEL "unit")

    # QVAC-18605 round 4 — API-surface lockdown for the new
    # EngineOptions::kv_attn_type field, supertonic_model field,
    # supertonic_kv_attn_type() thread-local accessor, and the
    # dispatch-scope `prev_kv_attn_type` for RAII teardown.
    add_executable(test-supertonic-kv-attn-type-api
        test/test_supertonic_kv_attn_type_api.cpp)
    target_link_libraries(test-supertonic-kv-attn-type-api PRIVATE tts-cpp)
    target_include_directories(test-supertonic-kv-attn-type-api PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-kv-attn-type-api)
    tts_cpp_register_test(test-supertonic-kv-attn-type-api LABEL "unit")

    # QVAC-19305 — Supertonic v3 language-coverage contract.
    # Pins the runtime language validator against the v3 bundle
    # (31 languages + the language-agnostic `na`) shipped in the
    # GGUF `supertonic.languages` array; asserts the built-in
    # fallback still only accepts the legacy v1/v2 5-language set.
    # CPU-only unit test (no model); an optional fixture sub-test
    # cross-checks a loaded model's advertised languages when a
    # GGUF path is passed as argv[1].
    add_executable(test-supertonic-languages
        test/test_supertonic_languages.cpp)
    target_link_libraries(test-supertonic-languages PRIVATE tts-cpp)
    target_include_directories(test-supertonic-languages PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-languages)
    tts_cpp_register_test(test-supertonic-languages LABEL "unit")

    # QVAC-18605 round 7 — Vulkan env-var passthrough mechanism
    # (EngineOptions::vulkan_env_overrides + apply_vulkan_env_overrides
    # public helper).  Tests cover: SFINAE field existence, empty-
    # map noop, single-entry-sets-env, operator-env-wins (set_env_if_unset
    # semantics), invalid-key-throws (loud-failure for typos), and
    # all-or-nothing-on-mixed-validity (no partial application).
    add_executable(test-supertonic-vulkan-env-overrides
        test/test_supertonic_vulkan_env_overrides.cpp)
    target_link_libraries(test-supertonic-vulkan-env-overrides PRIVATE tts-cpp)
    target_include_directories(test-supertonic-vulkan-env-overrides PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-vulkan-env-overrides)
    tts_cpp_register_test(test-supertonic-vulkan-env-overrides LABEL "unit")

    # QVAC-18605 round 7 — voice ttl/dp host cache
    # (`tts_cpp::supertonic::detail::voice_host_cache`).  Standalone
    # helper extracted from Engine::Impl::synthesize() so the
    # lookup-or-load semantics are testable on CPU without
    # instantiating a full Engine.  Tests cover: empty / first-load-
    # populates / second-load-hits-cache (null-tensor passthrough
    # proves the cache hit) / multi-voice / clear / null-on-miss
    # throws.
    add_executable(test-supertonic-voice-host-cache
        test/test_supertonic_voice_host_cache.cpp)
    target_link_libraries(test-supertonic-voice-host-cache PRIVATE tts-cpp)
    target_include_directories(test-supertonic-voice-host-cache PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-voice-host-cache)
    tts_cpp_register_test(test-supertonic-voice-host-cache LABEL "unit")

    # QVAC-18605 round 10 — pointer-compare upload-skip tracker
    # (`tts_cpp::supertonic::detail::upload_skip_tracker`).
    # Generalises the F4 pattern from `vector_res_style_qkv_cache`
    # (style_v_in / kctx_in upload-skip) to the front-block /
    # group-graph `text_in` uploads, which receive the same
    # `text_emb` pointer 5 times per synth.  Tests cover: default
    # state, upload + skip happy path, pointer-change forces
    # upload, reset() invalidation (synth-boundary contract),
    # interleaved-instance independence, cross-synth pointer-
    # reuse hazard simulation (the bug the synth-boundary reset
    # exists to prevent), and reset-on-empty no-op.
    add_executable(test-supertonic-upload-skip-tracker
        test/test_supertonic_upload_skip_tracker.cpp)
    target_link_libraries(test-supertonic-upload-skip-tracker PRIVATE tts-cpp)
    target_include_directories(test-supertonic-upload-skip-tracker PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-upload-skip-tracker)
    tts_cpp_register_test(test-supertonic-upload-skip-tracker LABEL "unit")

    # QVAC-18605 round 12 #6 — text-encoder speech-prompted-attention
    # GPU bridge.  Master's Metal-port branch built
    # `speech_prompted_merged_cache` (one merged graph for QKV proj +
    # head-split + flash-attn + out-proj) but never wired its run path
    # into the production text-encoder loop.  Round 12 adds
    # `run_speech_prompted_merged_cache` + dispatches to it on non-CPU
    # backends, eliminating 10 sync points / synth (2 layers × 5
    # download+pack+reupload steps each) at the text encoder.  This
    # test pins the new symbol's existence + the merged-cache struct's
    # field contract via SFINAE; equivalence vs. the scalar reference
    # is verified end-to-end by the model-fixture tests
    # `test-supertonic-text-encoder-trace` + `test-supertonic-pipeline`.
    add_executable(test-supertonic-text-encoder-gpu-bridge
        test/test_supertonic_text_encoder_gpu_bridge.cpp)
    target_link_libraries(test-supertonic-text-encoder-gpu-bridge PRIVATE tts-cpp)
    target_include_directories(test-supertonic-text-encoder-gpu-bridge PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-text-encoder-gpu-bridge)
    tts_cpp_register_test(test-supertonic-text-encoder-gpu-bridge LABEL "unit")

    # QVAC-18605 round 12 #5 — pinned-host-buffer input allocation
    # helper.  Round 3 shipped the capability probe but deferred the
    # per-engine input-scratchpad refactor that actually USES the
    # host-pinned buffer to skip ggml-vulkan's internal staging-
    # buffer hop.  Round 12 #5 lands `try_alloc_inputs_in_pinned_host_buffer`
    # and applies it at the hot per-step input sites
    # (vector_group_graph_cache + ve_front_block_graph_cache).
    # The CPU-only test pins the symbol's existence + the
    # `nullptr` return contract on CPU backend + idempotent
    # repeat calls + null-pointer safety on null backend / null
    # ctx (defensive failure modes in error-handler paths).
    add_executable(test-supertonic-pinned-host-buffer
        test/test_supertonic_pinned_host_buffer.cpp)
    target_link_libraries(test-supertonic-pinned-host-buffer PRIVATE tts-cpp)
    target_include_directories(test-supertonic-pinned-host-buffer PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-pinned-host-buffer)
    tts_cpp_register_test(test-supertonic-pinned-host-buffer LABEL "unit")

    # QVAC-18605 round 13 #1 — input-scratchpad allocator helper
    # that consolidates the pinned-host + default-backend fallback
    # boilerplate round 12 #5 manually inlined at 4 cache sites.
    # Round 13 needs to extend the pattern to 5+ more caches
    # (vector_loop_one_graph, vocoder, style residual + QKV, merged
    # speech-prompted) — without this helper that's 5x copy-paste.
    # CPU-only test pins the symbol + CPU-fallback contract + null-
    # argument throws (defensive failure modes in error paths).
    add_executable(test-supertonic-input-scratchpad
        test/test_supertonic_input_scratchpad.cpp)
    target_link_libraries(test-supertonic-input-scratchpad PRIVATE tts-cpp)
    target_include_directories(test-supertonic-input-scratchpad PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-input-scratchpad)
    tts_cpp_register_test(test-supertonic-input-scratchpad LABEL "unit")

    add_executable(test-supertonic-f16-attn-parity
        test/test_supertonic_f16_attn_parity.cpp)
    target_link_libraries(test-supertonic-f16-attn-parity PRIVATE ggml)
    target_include_directories(test-supertonic-f16-attn-parity PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-supertonic-f16-attn-parity)
    tts_cpp_register_test(test-supertonic-f16-attn-parity LABEL "unit")

    # supertonic-bench is a benchmark CLI (takes --text / --out / --runs),
    # not a parity test, so it doesn't go through add_supertonic_harness
    # which auto-registers everything as a positional-args (MODEL, REF_DIR)
    # ctest target.  Built unconditionally with the rest of the test set
    # so `cmake --build --target supertonic-bench` works the same way.
    add_executable(supertonic-bench src/supertonic_bench.cpp)
    target_link_libraries(supertonic-bench PRIVATE tts-cpp)
    target_include_directories(supertonic-bench PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(supertonic-bench)
endif()

set(TTS_CPP_POINTWISE_ACCEL_TARGETS
    tts-cpp
    test-supertonic-vector
    test-supertonic-vector-trace
    test-supertonic-vocoder
    test-supertonic-vocoder-trace
    test-supertonic-vocoder-pointwise
    test-supertonic-pipeline
    supertonic-bench)

if (APPLE)
    find_library(TTS_CPP_ACCELERATE_FRAMEWORK Accelerate)
    if (TTS_CPP_ACCELERATE_FRAMEWORK)
        foreach(_tts_accel_target ${TTS_CPP_POINTWISE_ACCEL_TARGETS})
            if (TARGET ${_tts_accel_target})
                target_link_libraries(${_tts_accel_target} PRIVATE ${TTS_CPP_ACCELERATE_FRAMEWORK})
                target_compile_definitions(${_tts_accel_target} PRIVATE TTS_CPP_USE_ACCELERATE)
            endif()
        endforeach()
    endif()
else()
    check_include_file_cxx(cblas.h TTS_CPP_HAS_CBLAS_H)
    find_package(BLAS)
    if (TTS_CPP_HAS_CBLAS_H AND BLAS_FOUND)
        foreach(_tts_cblas_target ${TTS_CPP_POINTWISE_ACCEL_TARGETS})
            if (TARGET ${_tts_cblas_target})
                target_link_libraries(${_tts_cblas_target} PRIVATE ${BLAS_LIBRARIES})
                target_compile_definitions(${_tts_cblas_target} PRIVATE TTS_CPP_USE_CBLAS)
            endif()
        endforeach()
    endif()
endif()

if (TTS_CPP_INSTALL AND TTS_CPP_BUILD_LIBRARY)
    include(CMakePackageConfigHelpers)

    # Re-install the bundled ggml targets into a `ggml-targets` export
    # set so install(EXPORT tts-cppTargets) can resolve the
    # `tts-cpp PUBLIC ggml` link as the namespaced `ggml::ggml`.  Without
    # this, CMake refuses to generate tts-cppTargets.cmake because the
    # in-tree `ggml` target is not part of any export set.  Replicates
    # the standard install(TARGETS ggml ggml-base EXPORT ggml-targets)
    # pattern that downstream ggml-based projects ship in their own
    # CMake (e.g. llama.cpp); we add it here at the tts-cpp level
    # because the upstream ggml at the pinned commit installs ggml
    # without an EXPORT clause.  The bundled ggml's hand-written
    # ggml-config.cmake (find_library + UNKNOWN IMPORTED) still ships
    # and is what's loaded via find_dependency(ggml) — we don't replace
    # it; the new ggml-targets.cmake just carries the export so
    # CMake's cross-package reference check passes here.
    if (NOT TTS_CPP_USE_SYSTEM_GGML)
        set(_tts_cpp_ggml_export_targets ggml ggml-base)
        if (DEFINED GGML_AVAILABLE_BACKENDS)
            list(APPEND _tts_cpp_ggml_export_targets ${GGML_AVAILABLE_BACKENDS})
        endif()
        # No PUBLIC_HEADER DESTINATION here: the bundled ggml's own
        # install(TARGETS ggml LIBRARY PUBLIC_HEADER) already copies the
        # ggml/include/*.h headers into the install tree.  Duplicating
        # the rule here re-resolves the same relative paths against the
        # tts-cpp source dir and `cmake --install` fails with
        # "file INSTALL cannot find <root>/include/ggml.h: File exists".
        # Keep this rule minimal: EXPORT + library destinations only,
        # so tts-cppTargets can resolve the namespaced ggml::ggml link
        # without rebuilding the headers install.
        install(TARGETS ${_tts_cpp_ggml_export_targets}
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
        unset(_tts_cpp_ggml_export_targets)
    endif()

    # Export the tts-cpp target via an install(EXPORT) set so the
    # generated tts-cppTargets.cmake carries the correct shape for
    # whatever linkage the build produced (STATIC vs SHARED, with
    # IMPORTED_IMPLIB on Windows shared builds, etc.).
    # tts-cpp-backend-defs is an INTERFACE library that ships compile
    # defines only (no archives / no runtime artefacts).  It is part of
    # tts-cpp's PRIVATE link line for static archives, so the exporter
    # requires it to be in the same export set even though there's
    # nothing to install for it on disk.  Adding it here keeps the
    # generated tts-cppTargets.cmake self-contained.
    install(TARGETS tts-cpp tts-cpp-backend-defs
        EXPORT tts-cppTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY include/tts-cpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )
    install(EXPORT tts-cppTargets
        FILE tts-cppTargets.cmake
        NAMESPACE tts-cpp::
        DESTINATION ${CMAKE_INSTALL_DATADIR}/tts-cpp
    )

    set(TTS_CPP_OPTIONAL_DEPS "")
    if (OpenMP_CXX_FOUND)
        string(APPEND TTS_CPP_OPTIONAL_DEPS "find_dependency(OpenMP COMPONENTS CXX)\n")
    endif()
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tts-cppConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/tts-cppConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/tts-cpp
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/tts-cppConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMinorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/tts-cppConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/tts-cppConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_DATADIR}/tts-cpp
    )
endif()
