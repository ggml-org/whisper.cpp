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
        # In-tree subtree of qvac-ext-lib-whisper.cpp: the standalone
        # patches/ folder + scripts/setup-ggml.sh tooling is intentionally
        # absent here.  Without them, an add_subdirectory(ggml) build
        # would silently miss the ggml-backend-reg-filename-prefix patch
        # that GGML_BACKEND_DL_PROJECT_PREFIX="speech-" depends on, so
        # libspeech-ggml-*.so files would exist on disk but the runtime
        # loader would still search for libggml-*.so under
        # GGML_BACKEND_DL=ON.  Reject up front with a pointer at the
        # right consumption path.
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches")
            message(FATAL_ERROR
                "tts-cpp: this in-tree subtree does not ship the patches/ "
                "directory.  Pass -DTTS_CPP_USE_SYSTEM_GGML=ON to consume "
                "the QVAC speech-stack `ggml-speech` vcpkg port (which "
                "carries the pre-applied patches), or use the standalone "
                "github.com/gianni-cor/chatterbox.cpp repo for a "
                "bundled-ggml dev build with patches/ present.")
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

find_package(OpenMP)

if (MSVC)
    add_compile_definitions(_USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
endif()

# INTERFACE library that holds the GGML_USE_<BACKEND> compile defines
# every TU that includes ggml.h needs to dispatch correctly on the
# enabled backend.  The tts-cpp library AND any test executable that
# recompiles src/chatterbox_tts.cpp / src/main.cpp from source (i.e.
# bypasses the tts-cpp link) must link against this; otherwise the
# #ifdef GGML_USE_<BACKEND> branches inside those TUs evaluate as
# undefined and the GPU code paths get silently compiled out of the
# test executable, even when the parent build did enable the backend.
# Mirrors parakeet-cpp's parakeet-backend-defs INTERFACE lib.
add_library(tts-cpp-backend-defs INTERFACE)
if (GGML_CUDA)
    target_compile_definitions(tts-cpp-backend-defs INTERFACE GGML_USE_CUDA)
endif()
if (GGML_METAL)
    target_compile_definitions(tts-cpp-backend-defs INTERFACE GGML_USE_METAL)
endif()
if (GGML_VULKAN)
    target_compile_definitions(tts-cpp-backend-defs INTERFACE GGML_USE_VULKAN)
endif()
if (GGML_BLAS)
    target_compile_definitions(tts-cpp-backend-defs INTERFACE GGML_USE_BLAS)
endif()
if (GGML_OPENCL)
    target_compile_definitions(tts-cpp-backend-defs INTERFACE GGML_USE_OPENCL)
endif()

set(TTS_CPP_LIB_SOURCES
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
    src/mtl_tokenizer.cpp
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
        src/mel_extract_stft.cpp)
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
        src/mel_extract_stft.cpp)
    target_link_libraries(test-resample PRIVATE ggml)
    target_include_directories(test-resample PRIVATE src)
    tts_cpp_apply_ccache(test-resample)
    tts_cpp_register_test(test-resample LABEL "unit")

    add_executable(test-voice-encoder
        test/test_voice_encoder.cpp
        src/voice_encoder.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp)
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
        src/mel_extract_stft.cpp)
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
        src/mel_extract_stft.cpp)
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
        src/s3tokenizer.cpp)
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

    add_executable(test-streaming
        test/test_streaming.cpp
        src/chatterbox_tts.cpp)
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
        src/chatterbox_tts.cpp)
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
    if (GGML_METAL)
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
