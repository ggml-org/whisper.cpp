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
option(TTS_CPP_BUILD_EXECUTABLES "tts-cpp: build tts-cli, mel2wav, CosyVoice, Supertonic, Parler, LavaSR, and Audio8 command-line tools" ${TTS_CPP_STANDALONE_DEFAULT})
option(TTS_CPP_BUILD_TESTS       "tts-cpp: build per-stage numerical validation harnesses"   ${TTS_CPP_STANDALONE_DEFAULT})
option(TTS_CPP_INSTALL           "tts-cpp: generate install rules"                           ON)
option(TTS_CPP_USE_SYSTEM_GGML   "tts-cpp: use find_package(ggml) instead of add_subdirectory(ggml). Defaults to ON for qvac-fabric-speech.cpp and vcpkg builds. OFF requires an engines/tts/ggml checkout of qvac-ext-ggml@speech, staged via scripts/setup-ggml.sh (or by hand); no patches overlay — the speech branch is patched at the commit level." ON)
# Note: the standalone chatterbox.cpp repo carries a TTS_CPP_GGML_LIB_PREFIX
# option here that controls renaming of bundled libggml-*.so to
# libspeech-ggml-*.so when add_subdirectory(ggml) builds it locally. This
# in-tree package can consume a locally staged speech-branch checkout but
# does not expose the standalone filename-prefix option; the speech branch or
# `ggml-speech` vcpkg port already emits the integrated filenames.
option(TTS_CPP_CCACHE            "tts-cpp: use ccache as compiler launcher if available"     ON)
option(TTS_CPP_COREML            "tts-cpp: enable the Apple Core ML (Neural Engine) sidecars for the Audio8 codec synthesis stack, the Supertonic vocoder and the MOSS stages (Apple-only, additive; each stage runs on its .mlmodelc next to the GGUF when present, else on ggml)." OFF)

# OBJC + OBJCXX up front, before any add_subdirectory(ggml); see engines/parakeet.
if (TTS_CPP_COREML)
    if (NOT APPLE)
        message(FATAL_ERROR "TTS_CPP_COREML is Apple-only (requires the CoreML framework); leave it OFF on ${CMAKE_SYSTEM_NAME}.")
    endif()
    enable_language(OBJC)
    enable_language(OBJCXX)
endif()

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
        # Stage the checkout with `scripts/setup-ggml.sh`, which clones the
        # pinned speech ref into `engines/tts/ggml/` and is idempotent on
        # re-run. No patches/ directory is consulted: the speech branch is
        # already patched at the commit level.
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
            message(FATAL_ERROR
                "tts-cpp: bundled-ggml build requires engines/tts/ggml/ to be "
                "a checkout of tetherto/qvac-ext-ggml@speech. Run "
                "scripts/setup-ggml.sh to stage it, or pass "
                "-DTTS_CPP_USE_SYSTEM_GGML=ON to consume the QVAC "
                "speech-stack `ggml-speech` vcpkg port.")
        endif()
        # tinyBLAS rationale and measurements: docs/build.md.
        if (NOT DEFINED GGML_LLAMAFILE_DEFAULT)
            set(GGML_LLAMAFILE_DEFAULT ON)
        endif()
        add_subdirectory(ggml)
    endif()
endif()

# GPU-labelled test registration must see which ggml backends exist in both
# integration modes. The add_subdirectory build exposes the GGML_<BACKEND>
# options directly and the speech-branch package config re-exports them
# (GGML_VARIABLES_EXPANDED); GGML_AVAILABLE_BACKENDS is the fallback for a
# system ggml whose config predates that expansion. Each engine's gpu test
# variants register only for backends that engine's production allowlist
# accepts, so an off-allowlist build neither runs unvalidated backends nor
# silently skips validated ones.
foreach(_tts_ggml_backend metal vulkan cuda opencl)
    string(TOUPPER "${_tts_ggml_backend}" _tts_ggml_backend_uc)
    set(TTS_CPP_GGML_${_tts_ggml_backend_uc} FALSE)
    if (GGML_${_tts_ggml_backend_uc} OR
        "ggml-${_tts_ggml_backend}" IN_LIST GGML_AVAILABLE_BACKENDS)
        set(TTS_CPP_GGML_${_tts_ggml_backend_uc} TRUE)
    endif()
endforeach()
# Parler allowlist: Metal / Vulkan / OpenCL / CUDA (parler/gguf.cpp).
set(TTS_CPP_PARLER_GPU FALSE)
if (TTS_CPP_GGML_METAL OR TTS_CPP_GGML_VULKAN OR TTS_CPP_GGML_OPENCL OR
    TTS_CPP_GGML_CUDA)
    set(TTS_CPP_PARLER_GPU TRUE)
endif()
# CosyVoice3 allowlist: Metal / OpenCL everywhere, Vulkan and CUDA on
# Windows and non-Android Linux only (cosyvoice_gpu_requirement() in
# cosyvoice_pipeline.h -- keep the two conditions in lockstep). The
# platform guard mirrors the runtime split: a build whose only GPU
# backend the runtime requirement would refuse (Android or Apple
# Vulkan-only) must not register gpu test variants that would hard-fail
# instead of skipping.
set(TTS_CPP_COSYVOICE_GPU FALSE)
set(_tts_cosy_desktop_platform FALSE)
if (WIN32 OR (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT ANDROID))
    set(_tts_cosy_desktop_platform TRUE)
endif()
if (TTS_CPP_GGML_METAL OR TTS_CPP_GGML_OPENCL OR
    ((TTS_CPP_GGML_VULKAN OR TTS_CPP_GGML_CUDA) AND _tts_cosy_desktop_platform))
    set(TTS_CPP_COSYVOICE_GPU TRUE)
endif()
set(_tcb_cosy_arms "")
if (TTS_CPP_GGML_CUDA AND _tts_cosy_desktop_platform)
    list(APPEND _tcb_cosy_arms cuda)
endif()
if (TTS_CPP_GGML_VULKAN AND _tts_cosy_desktop_platform)
    list(APPEND _tcb_cosy_arms vulkan)
endif()

# The standalone chatterbox.cpp source carries a bundled-ggml filename-prefix
# block (helper function, tts_cpp_apply_ggml_prefix loop, and
# GGML_BACKEND_DL_PROJECT_PREFIX define). This in-tree package does not expose
# that rename surface. System builds consume filenames and macros installed by
# the ggml-speech port; bundled builds use the filenames produced by the
# locally staged speech-branch checkout (scripts/setup-ggml.sh).

option(TTS_CPP_OPENMP "tts-cpp: enable OpenMP for the tts-cpp target" ON)
if (WIN32 AND NOT MINGW AND TTS_CPP_OPENMP AND NOT DEFINED CACHE{TTS_CPP_OPENMP_USER_OVERRIDE})
    set(TTS_CPP_OPENMP OFF CACHE BOOL "" FORCE)
    message(STATUS "tts-cpp: OpenMP disabled on Windows non-MinGW (clang-cl / MSVC + msys2 libgomp mismatch, plus consumer-toolchain probe leaks via static-archive transitive link interface). Set -DTTS_CPP_OPENMP_USER_OVERRIDE=ON and -DTTS_CPP_OPENMP=ON to force-enable.")
endif()

if (TTS_CPP_OPENMP)
    find_package(OpenMP)
endif()

if (MSVC)
    add_compile_definitions(_USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS NOMINMAX)
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
    src/voice_controls.cpp
    src/backend_selection.cpp
    src/sched_dispatch.cpp
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
    src/t3_stop_controller.cpp
    src/t3_alignment_analyzer.cpp
    src/supertonic_gguf.cpp
    src/supertonic_preprocess.cpp
    src/supertonic_vocoder.cpp
    src/supertonic_duration.cpp
    src/supertonic_text_encoder.cpp
    src/supertonic_vector_estimator.cpp
    src/supertonic_engine.cpp
    src/supertonic_pace.cpp
    src/supertonic_voice_json.cpp
    src/supertonic_chunker.cpp
    src/supertonic_coreml_path.cpp
    src/coreml_sidecar_path.cpp
    # CosyVoice3 (Fun-CosyVoice3-0.5B / 1.5B) — native CPU pipeline.
    # cosyvoice_pipeline.cpp is the shared graph core (LM / DiT flow / HiFT),
    # also used by the cosyvoice-{llm,flow,hift} parity CLIs below.
    src/cosyvoice_pipeline.cpp
    src/cosyvoice_instruct.cpp
    src/cosyvoice_frontend.cpp
    src/cosyvoice_engine.cpp
    src/mtl_tokenizer.cpp
    src/text_preprocess.cpp
    # LavaSR neural speech enhancement — CPU/GGML post-process.
    src/lavasr/dsp/resampler.cpp
    src/lavasr/dsp/stft_processor.cpp
    src/lavasr/dsp/mel_filterbank.cpp
    src/lavasr/dsp/fastlr_merge.cpp
    src/lavasr/dsp/overlap_add.cpp
    src/lavasr/enhancer_core.cpp
    src/lavasr/enhancer.cpp
    src/lavasr/enhancer_ggml.cpp
    src/lavasr/enhancer_gguf.cpp
    src/lavasr/enhancer_api.cpp
    # LavaSR denoiser stage (follow-up) — UL-UNAS neural denoiser
    # (runs BEFORE the enhancer, rate-preserving).  See src/lavasr/denoiser*.{h,cpp}.
    src/lavasr/denoiser_core.cpp
    src/lavasr/denoiser.cpp
    src/lavasr/denoiser_gguf.cpp
    src/lavasr/denoiser_api.cpp
    src/lavasr/denoiser_ggml.cpp
    # Parler-TTS engine (T5 encoder + delay-pattern decoder LM + DAC codec).
    src/parler/gguf.cpp
    src/parler/description.cpp
    src/parler/text_norm.cpp
    src/parler/tokenizer.cpp
    src/parler/bpe_tokenizer.cpp
    src/parler/t5.cpp
    src/parler/decoder.cpp
    src/parler/delay.cpp
    src/parler/sampler.cpp
    src/parler/dac.cpp
    src/parler/dac_accel.cpp
    src/parler/engine.cpp
    # Audio8 engine (DualAR semantic LM + DAC-style codec with in-process cloning).
    src/audio8/unicode.cpp
    src/audio8/tokenizer.cpp
    src/audio8/gguf.cpp
    src/audio8/graph.cpp
    src/audio8/lm.cpp
    src/audio8/codec_ops.cpp
    src/audio8/codec_decode.cpp
    src/audio8/codec_encode.cpp
    src/audio8/window_stream.cpp
    src/audio8/coreml_path.cpp
    src/audio8/sampling.cpp
    src/audio8/engine.cpp
    # Pocket TTS numerical core; Mimi synthesis is not exposed yet.
    src/pocket/flow_lm.cpp
    src/pocket/mimi.cpp
    src/pocket/frontend.cpp
    src/pocket/engine.cpp
    src/pocket/fit.cpp
    src/pocket/reference_audio.cpp
    src/moss/codec.cpp
    src/moss/delay_lm.cpp
    src/moss/generation.cpp
    src/moss/frontend.cpp
    src/moss/engine.cpp
    src/moss/fit.cpp
    src/moss/reference_wav.cpp
    src/moss/sfx_model.cpp
    src/moss/sfx_text_encoder.cpp
    src/moss/sfx_dit.cpp
    src/moss/sfx_vae.cpp
    src/moss/sfx_prompt.cpp
    src/moss/sfx_sampler.cpp
    src/moss/sfx_tokenizer.cpp
    src/moss/sfx_engine.cpp
    src/moss/sfx_request.cpp
    src/moss/sfx_fit.cpp
    src/moss/gguf_metadata.cpp
    src/moss/speech_lm.cpp
    src/moss/speech_tokenizer.cpp
    src/moss/speech_codec.cpp
    src/moss/speech_prompt.cpp
    src/moss/speech_generation.cpp
    src/moss/speech_engine.cpp
    src/moss/coreml_sidecar.cpp
    src/moss/sfx_coreml.cpp
    src/moss/codec_coreml.cpp
    # Memory-fit preflight (include/tts-cpp/fit.h + the *-fit-params tools).
    src/fit_common.cpp
    src/audio8/fit.cpp
    src/audio8/fit_main.cpp
    src/chatterbox_fit.cpp
    src/chatterbox_fit_main.cpp
    src/parler/fit.cpp
    src/parler/fit_main.cpp
    src/cosyvoice_fit.cpp
    src/cosyvoice_fit_main.cpp
    src/supertonic_fit.cpp
    src/supertonic_fit_main.cpp
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

    # Core ML sidecars (Audio8 codec, Supertonic vocoder, MOSS stages): ARC objects fold
    # into libtts-cpp; frameworks are PUBLIC for static consumers.
    if (TTS_CPP_COREML)
        find_library(TTS_CPP_FOUNDATION_FRAMEWORK Foundation REQUIRED)
        find_library(TTS_CPP_COREML_FRAMEWORK     CoreML     REQUIRED)

        add_library(tts-cpp.coreml OBJECT
            src/audio8/coreml/codec-synth.mm
            src/supertonic_coreml_vocoder.mm
            src/moss/coreml/runner.mm)
        tts_cpp_apply_ccache(tts-cpp.coreml)
        target_include_directories(tts-cpp.coreml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        set_target_properties(tts-cpp.coreml PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            COMPILE_FLAGS "-fobjc-arc"
            XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
        )

        target_sources(tts-cpp PRIVATE $<TARGET_OBJECTS:tts-cpp.coreml>)
        target_compile_definitions(tts-cpp PRIVATE TTS_CPP_USE_COREML)
        target_link_libraries(tts-cpp PUBLIC
            ${TTS_CPP_FOUNDATION_FRAMEWORK}
            ${TTS_CPP_COREML_FRAMEWORK}
        )
        message(STATUS "tts-cpp: Core ML sidecars enabled (TTS_CPP_USE_COREML): Audio8 codec, Supertonic vocoder, MOSS stages")
    endif()
    target_link_libraries(tts-cpp PUBLIC ggml)
    find_package(Threads REQUIRED)
    target_link_libraries(tts-cpp PRIVATE Threads::Threads)

    # ------------------------------------------------------------------
    # Optional native preprocessors for the MTL non-Latin tier:
    #   * `ja` -- libmecab (dictionary path passed at runtime)
    #   * `zh` -- Cangjie5_TC mapping (TSV path passed at runtime)
    #
    # libmecab is autodetected via find_package / find_library / find_path.
    # The dictionary path is no longer baked in at compile time; callers
    # pass it at runtime via --mecab-dict (CLI) or EngineOptions::mecab_dict_path
    # (API). Use scripts/build_mecab_dict.py to materialize the IPAdic dictionary
    # from the `ipadic` PyPI package.
    #
    # Cangjie no longer requires a compile-time TSV path; callers
    # pass it at runtime via --cangjie-tsv (CLI) or
    # EngineOptions::cangjie_tsv_path (API).
    # ------------------------------------------------------------------
    find_package(mecab CONFIG QUIET)
    find_library(MECAB_LIBRARY    NAMES mecab)
    find_path   (MECAB_INCLUDE_DIR NAMES mecab.h PATH_SUFFIXES mecab)
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
        if (TARGET mecab::mecab)
            # The vcpkg mecab target carries the Windows static-library
            # compile definitions needed to avoid dllimport references.
            target_link_libraries(tts-cpp PRIVATE mecab::mecab)
        else()
            target_link_libraries(tts-cpp PRIVATE ${MECAB_LIBRARY})
        endif()
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
            # -ffast-math implies -ffinite-math-only, which lets the compiler
            # assume no NaN/Inf and optimise away the LavaSR DSP's numerical
            # guards (e.g. the sinc `x != 0` and `weight_sum > 0` guards in the
            # resampler). On the denoiser -> enhancer chain that produced NaNs in
            # the denoiser output which the FastLR FFT merge then spread across
            # the whole enhancer result -> silent output. Keep the fast-math
            # speedups but restore IEEE NaN/Inf handling so those guards hold.
            $<$<COMPILE_LANGUAGE:CXX>:-fno-finite-math-only>
            $<$<COMPILE_LANGUAGE:C>:-fno-finite-math-only>
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

# cosyvoice_pipeline.cpp reaches into tts_cpp::detail for the registry-routed
# CPU backend init (ggml_backend_cpu_init is a libggml-cpu symbol, absent under
# GGML_BACKEND_DL=ON) and for the sched dual-path dispatch.  Declared outside the
# executables/tests blocks because both consume it.  Mirrors _tcb_parler_core_srcs.
set(_tcb_cosy_core_srcs
    src/cosyvoice_pipeline.cpp
    src/backend_selection.cpp
    src/sched_dispatch.cpp)

if (TTS_CPP_BUILD_EXECUTABLES)
    if (NOT TTS_CPP_BUILD_LIBRARY)
        message(FATAL_ERROR "TTS_CPP_BUILD_EXECUTABLES requires TTS_CPP_BUILD_LIBRARY=ON")
    endif()

    add_executable(tts-cli src/cli_main.cpp)
    add_executable(pocket-cli src/pocket/cli.cpp)
    target_link_libraries(pocket-cli PRIVATE tts-cpp)
    target_include_directories(pocket-cli PRIVATE src)
    tts_cpp_apply_ccache(pocket-cli)

    target_link_libraries(tts-cli PRIVATE tts-cpp)
    tts_cpp_apply_ccache(tts-cli)

    add_executable(moss-cli src/moss/cli_main.cpp src/moss/cli.cpp)
    target_link_libraries(moss-cli PRIVATE tts-cpp)
    target_include_directories(moss-cli PRIVATE src)
    tts_cpp_apply_ccache(moss-cli)

    add_executable(mel2wav src/mel2wav.cpp)
    target_link_libraries(mel2wav PRIVATE ggml)
    target_include_directories(mel2wav PRIVATE ggml/include src)
    tts_cpp_apply_ccache(mel2wav)

    # CosyVoice3 CausalHiFTGenerator vocoder (mel -> 24 kHz wav). Same
    # HiFTGenerator lineage as Chatterbox's mel2wav with CosyVoice3's causal
    # semantics + SineGen2. Validated at 0.989 log-mel corr vs the PyTorch
    # reference (0.996 waveform corr after the source-excitation phase fix).
    add_executable(cosyvoice-hift src/cosyvoice_hift.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(cosyvoice-hift PRIVATE ggml)
    target_include_directories(cosyvoice-hift PRIVATE ggml/include src)
    tts_cpp_apply_ccache(cosyvoice-hift)

    # CosyVoice3 flow (stage 4): 22-layer DiT + CFM Euler loop (speech tokens ->
    # mel). Parity-gated against the PyTorch reference (flow_mel cosine
    # 1.000000). Uses only standard ggml ops.
    add_executable(cosyvoice-flow src/cosyvoice_flow.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(cosyvoice-flow PRIVATE ggml)
    target_include_directories(cosyvoice-flow PRIVATE ggml/include src)
    tts_cpp_apply_ccache(cosyvoice-flow)

    # CosyVoice3 LLM (stage 5): Qwen2.5-0.5B speech LM (text -> speech tokens),
    # autoregressive decode + RAS sampling + Qwen2 BPE tokenizer (qwen_tokenizer.h).
    # Prefill logits cosine 1.000000 vs the PyTorch reference.
    add_executable(cosyvoice-llm src/cosyvoice_llm.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(cosyvoice-llm PRIVATE ggml)
    target_include_directories(cosyvoice-llm PRIVATE ggml/include src)
    tts_cpp_apply_ccache(cosyvoice-llm)

    add_executable(supertonic-cli src/supertonic_cli.cpp)
    target_link_libraries(supertonic-cli PRIVATE tts-cpp)
    tts_cpp_apply_ccache(supertonic-cli)

    add_executable(parler-cli src/parler/cli.cpp)
    target_link_libraries(parler-cli PRIVATE tts-cpp)
    target_include_directories(parler-cli PRIVATE src)
    tts_cpp_apply_ccache(parler-cli)

    add_executable(parler-bench src/parler/bench.cpp)
    target_link_libraries(parler-bench PRIVATE tts-cpp)
    target_include_directories(parler-bench PRIVATE ggml/include src src/parler include)
    tts_cpp_apply_ccache(parler-bench)

    # LavaSR two-stage (denoiser -> enhancer) bench + by-ear harness.
    add_executable(lavasr-bench src/lavasr/lavasr_bench.cpp)
    target_link_libraries(lavasr-bench PRIVATE tts-cpp)
    target_include_directories(lavasr-bench PRIVATE include ggml/include src)
    tts_cpp_apply_ccache(lavasr-bench)

    # Supertonic single-pass bench (--text / --runs / --json-out). Sits with
    # the other -bench CLIs rather than the TESTS block so the standalone
    # benchmark workflow can build it with SPEECH_BUILD_TESTS=OFF.
    add_executable(supertonic-bench src/supertonic_bench.cpp)
    target_link_libraries(supertonic-bench PRIVATE tts-cpp)
    target_include_directories(supertonic-bench PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(supertonic-bench)

    # CosyVoice3 end-to-end CLI: text -> 24 kHz WAV via the public Engine (the
    # same API the JS addon / @qvac/sdk drive).  See src/cosyvoice_cli.cpp.
    add_executable(cosyvoice-cli src/cosyvoice_cli.cpp)
    target_link_libraries(cosyvoice-cli PRIVATE tts-cpp)
    tts_cpp_apply_ccache(cosyvoice-cli)

    # CosyVoice3 per-stage benchmark.  Same --json-out schema as parler-bench so
    # the .diag summarisers read it unchanged; --tokens-out/--tokens-in pin the
    # LM trajectory so a CPU leg and a GPU leg do provably equal work.
    add_executable(cosyvoice-bench src/cosyvoice_bench.cpp)
    target_link_libraries(cosyvoice-bench PRIVATE tts-cpp)
    tts_cpp_apply_ccache(cosyvoice-bench)

    # Audio8 end-to-end CLI: text -> 44.1 kHz WAV, with in-process cloning when
    # the codec encoder GGUF is supplied.  See src/audio8_cli.cpp.
    add_executable(audio8-cli src/audio8_cli.cpp src/audio8/cli.cpp)
    target_link_libraries(audio8-cli PRIVATE tts-cpp)
    target_include_directories(audio8-cli PRIVATE src)
    tts_cpp_apply_ccache(audio8-cli)

    # Codec-only reconstruction accesses internal codec symbols, which are
    # intentionally hidden in shared-library builds (as with codec unit tests).
    if (NOT TTS_CPP_BUILD_SHARED)
        add_executable(audio8-codec-bench src/audio8_codec_bench.cpp)
        target_link_libraries(audio8-codec-bench PRIVATE tts-cpp ggml)
        target_include_directories(audio8-codec-bench PRIVATE src)
        tts_cpp_apply_ccache(audio8-codec-bench)
    endif()

    # Memory-fit preflight CLIs: project a model set against the free device
    # memory without loading weights.  See src/audio8/fit_main.cpp and
    # src/chatterbox_fit_main.cpp.
    add_executable(audio8-fit-params src/audio8_fit_params.cpp)
    target_link_libraries(audio8-fit-params PRIVATE tts-cpp)
    tts_cpp_apply_ccache(audio8-fit-params)

    add_executable(chatterbox-fit-params src/chatterbox_fit_params.cpp)
    target_link_libraries(chatterbox-fit-params PRIVATE tts-cpp)
    tts_cpp_apply_ccache(chatterbox-fit-params)

    add_executable(parler-fit-params src/parler_fit_params.cpp)
    target_link_libraries(parler-fit-params PRIVATE tts-cpp)
    tts_cpp_apply_ccache(parler-fit-params)

    add_executable(cosyvoice-fit-params src/cosyvoice_fit_params.cpp)
    target_link_libraries(cosyvoice-fit-params PRIVATE tts-cpp)
    tts_cpp_apply_ccache(cosyvoice-fit-params)

    add_executable(supertonic-fit-params src/supertonic_fit_params.cpp)
    target_link_libraries(supertonic-fit-params PRIVATE tts-cpp)
    tts_cpp_apply_ccache(supertonic-fit-params)
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
    if (DEFINED ENV{CHATTERBOX_CANGJIE_TSV} AND NOT "$ENV{CHATTERBOX_CANGJIE_TSV}" STREQUAL "")
        set(_tcb_default_cangjie_tsv "$ENV{CHATTERBOX_CANGJIE_TSV}")
    else()
        set(_tcb_default_cangjie_tsv "${TTS_CPP_TEST_MODEL_DIR}/Cangjie5_TC.tsv")
    endif()
    set(TTS_CPP_TEST_CANGJIE_TSV "${_tcb_default_cangjie_tsv}"                    CACHE FILEPATH "tts-cpp: Cangjie5_TC TSV used by zh multilingual tests")

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
        cmake_parse_arguments(TCRT "" "LABEL;EXE;PASS_REGEX" "ARGS;REQUIRES" ${ARGN})
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
        # For tests whose binary is not a harness: the pass criterion is what
        # it prints, not just a zero exit code.
        if (DEFINED TCRT_PASS_REGEX)
            set_property(TEST ${target} PROPERTY PASS_REGULAR_EXPRESSION "${TCRT_PASS_REGEX}")
        endif()
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

    # Model-free: the CPU registry's feature list and the tinyBLAS detection the
    # Parler CPU parity tests key on. GGML_LLAMAFILE comes from the ggml package
    # config (or the bundled build), so the expectation follows the ggml build.
    add_executable(test-backend-features test/test_backend_features.cpp)
    target_link_libraries(test-backend-features PRIVATE ggml)
    target_include_directories(test-backend-features PRIVATE src)
    tts_cpp_apply_ccache(test-backend-features)
    tts_cpp_register_test(test-backend-features LABEL "unit")
    if (DEFINED GGML_LLAMAFILE)
        if (GGML_LLAMAFILE)
            set(_tts_expect_llamafile 1)
        else()
            set(_tts_expect_llamafile 0)
        endif()
        set_property(TEST test-backend-features APPEND PROPERTY ENVIRONMENT
            "TTS_CPP_EXPECT_LLAMAFILE=${_tts_expect_llamafile}")
    endif()

    add_executable(test-bench-wav test/test_bench_wav.cpp)
    target_include_directories(test-bench-wav PRIVATE src)
    tts_cpp_apply_ccache(test-bench-wav)
    tts_cpp_register_test(test-bench-wav LABEL "unit"
        ARGS "${CMAKE_CURRENT_BINARY_DIR}/bench wav test.wav")


    # Stable shorthands for the fixtures referenced by the registrations
    # below.  Resolved at configure time; missing files turn into
    # DISABLED tests via tts_cpp_register_test(REQUIRES ...).
    set(_tcb_t3_turbo_gguf      "${TTS_CPP_TEST_MODEL_DIR}/chatterbox-t3-turbo.gguf")
    set(_tcb_t3_mtl_gguf        "${TTS_CPP_TEST_MODEL_DIR}/chatterbox-t3-mtl.gguf")
    set(_tcb_t3_mtl_q4_gguf     "${TTS_CPP_TEST_MODEL_DIR}/chatterbox-t3-mtl-q4_0.gguf")
    set(_tcb_s3gen_gguf         "${TTS_CPP_TEST_MODEL_DIR}/chatterbox-s3gen.gguf")
    set(_tcb_s3gen_q4_gguf      "${TTS_CPP_TEST_MODEL_DIR}/chatterbox-s3gen-q4_0.gguf")
    set(_tcb_s3gen_mtl_gguf     "${TTS_CPP_TEST_MODEL_DIR}/chatterbox-s3gen-mtl.gguf")
    set(_tcb_super_gguf         "${TTS_CPP_TEST_MODEL_DIR}/supertonic.gguf")
    set(_tcb_super2_gguf        "${TTS_CPP_TEST_MODEL_DIR}/supertonic2.gguf")
    set(_tcb_super3_gguf        "${TTS_CPP_TEST_MODEL_DIR}/supertonic3.gguf")
    set(_tcb_super_ref          "${TTS_CPP_TEST_REF_DIR}/supertonic-ref-quick")
    set(_tcb_super_ref_noise    "${_tcb_super_ref}/noise.npy")
    set(_tcb_s3gen_ref          "${TTS_CPP_TEST_REF_DIR}/s3gen-ref")
    # Written last by dump-s3gen-reference.py, so it gates on a dump that ran
    # to completion rather than one that died partway through.
    set(_tcb_s3gen_ref_sentinel "${_tcb_s3gen_ref}/waveform.npy")
    set(_tcb_t3_mtl_ref         "${TTS_CPP_TEST_REF_DIR}/t3-mtl-ref")
    set(_tcb_streaming_ref      "${TTS_CPP_TEST_REF_DIR}/streaming-ref")
    set(_tcb_jfk_wav            "${TTS_CPP_TEST_AUDIO_DIR}/jfk.wav")
    set(_tcb_parler_gguf        "${TTS_CPP_TEST_MODEL_DIR}/parler-mini-v1-f32.gguf")
    set(_tcb_parler_ref         "${TTS_CPP_TEST_REF_DIR}/parler-ref")
    set(_tcb_audio8_lm          "${TTS_CPP_TEST_MODEL_DIR}/audio8-lm-f32.gguf")
    set(_tcb_audio8_decoder     "${TTS_CPP_TEST_MODEL_DIR}/audio8-codec-decoder-f32.gguf")
    set(_tcb_audio8_encoder     "${TTS_CPP_TEST_MODEL_DIR}/audio8-codec-encoder-f32.gguf")
    set(_tcb_audio8_lm_q8       "${TTS_CPP_TEST_MODEL_DIR}/audio8-lm-q8_0.gguf")
    set(_tcb_audio8_decoder_q8  "${TTS_CPP_TEST_MODEL_DIR}/audio8-codec-decoder-q8_0.gguf")
    set(_tcb_audio8_ref         "${TTS_CPP_TEST_REF_DIR}/audio8-ref")
    set(_tcb_audio8_ref_clone   "${TTS_CPP_TEST_REF_DIR}/audio8-ref-clone")
    set(_tcb_audio8_gpu_layers  99)
    # The recording the cloning fixtures were enrolled from, which the engine
    # test hands to the codec encoder instead of pre-computed codes.
    set(_tcb_audio8_ref_wav     "${TTS_CPP_TEST_AUDIO_DIR}/audio8-reference-en.wav")
    set(_tcb_parler_indic_gguf  "${TTS_CPP_TEST_MODEL_DIR}/parler-indic-f32.gguf")
    set(_tcb_parler_indic_ref   "${TTS_CPP_TEST_REF_DIR}/parler-indic-ref")

    # ---- CosyVoice3 per-stage parity tests --------------------------------
    # Fixtures (f32 GGUFs + PyTorch reference .npy + HF tokenizer_ref.txt) are
    # produced by scripts/dump-cosyvoice3-{reference,flow-reference,
    # llm-reference,tokenizer-reference}.py; the tests auto-disable when a
    # fixture is missing (fresh checkout stays green).
    set(_tcb_cosy_llm_gguf   "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-llm-f32.gguf")
    set(_tcb_cosy_flow_gguf  "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-flow-f32.gguf")
    set(_tcb_cosy_hift_gguf  "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-hift-f32.gguf")
    set(_tcb_cosy_ref        "${TTS_CPP_TEST_REF_DIR}/cosyvoice3-ref")

    add_executable(test-cosyvoice-flow test/test_cosyvoice_flow.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(test-cosyvoice-flow PRIVATE ggml)
    target_include_directories(test-cosyvoice-flow PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-cosyvoice-flow)
    tts_cpp_register_test(test-cosyvoice-flow
        LABEL    "fixture"
        ARGS     "--flow-gguf" "${_tcb_cosy_flow_gguf}" "--in-dir" "${_tcb_cosy_ref}"
        REQUIRES "${_tcb_cosy_flow_gguf}" "${_tcb_cosy_ref}/flow_mel.npy")

    # The opt-in prompt-cut path deviates from the reference by design, so it
    # gets its own leg with a looser reference bound.
    tts_cpp_register_test(test-cosyvoice-flow-cut
        EXE      test-cosyvoice-flow
        LABEL    "fixture"
        ARGS     "--flow-gguf" "${_tcb_cosy_flow_gguf}" "--in-dir" "${_tcb_cosy_ref}"
                 "--cut-min-cosine" "0.95"
        REQUIRES "${_tcb_cosy_flow_gguf}" "${_tcb_cosy_ref}/flow_mel.npy")

    # Same binaries on the GPU backend — the per-stage reference legs behind
    # the cosyvoice_engine.cpp allowlist.  Skipped via `ctest -LE gpu`.
    if (TTS_CPP_COSYVOICE_GPU)
        tts_cpp_register_test(test-cosyvoice-flow-gpu
            EXE      test-cosyvoice-flow
            LABEL    "gpu"
            ARGS     "--flow-gguf" "${_tcb_cosy_flow_gguf}" "--in-dir" "${_tcb_cosy_ref}"
            REQUIRES "${_tcb_cosy_flow_gguf}" "${_tcb_cosy_ref}/flow_mel.npy")
        set_property(TEST test-cosyvoice-flow-gpu APPEND PROPERTY ENVIRONMENT "COSYVOICE_TEST_GPU=1")
        # Per-backend pins of the same harness, so a host carrying several GPU
        # backends covers each by name; TTS_CPP_GPU_BACKEND fails loud when the
        # named backend is absent, so an arm cannot pass on the CPU.
        foreach(_tcb_cosy_arm IN LISTS _tcb_cosy_arms)
            tts_cpp_register_test(test-cosyvoice-flow-${_tcb_cosy_arm}
                EXE      test-cosyvoice-flow
                LABEL    "gpu"
                ARGS     "--flow-gguf" "${_tcb_cosy_flow_gguf}" "--in-dir" "${_tcb_cosy_ref}"
                REQUIRES "${_tcb_cosy_flow_gguf}" "${_tcb_cosy_ref}/flow_mel.npy")
            set_property(TEST test-cosyvoice-flow-${_tcb_cosy_arm} APPEND PROPERTY ENVIRONMENT
                         "COSYVOICE_TEST_GPU=1" "TTS_CPP_GPU_BACKEND=${_tcb_cosy_arm}")
        endforeach()
    endif()

    add_executable(test-cosyvoice-llm test/test_cosyvoice_llm.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(test-cosyvoice-llm PRIVATE ggml)
    target_include_directories(test-cosyvoice-llm PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-cosyvoice-llm)
    tts_cpp_register_test(test-cosyvoice-llm
        LABEL    "fixture"
        ARGS     "--llm-gguf" "${_tcb_cosy_llm_gguf}" "--in-dir" "${_tcb_cosy_ref}"
        REQUIRES "${_tcb_cosy_llm_gguf}" "${_tcb_cosy_ref}/gen_tokens.npy")

    if (TTS_CPP_COSYVOICE_GPU)
        tts_cpp_register_test(test-cosyvoice-llm-gpu
            EXE      test-cosyvoice-llm
            LABEL    "gpu"
            ARGS     "--llm-gguf" "${_tcb_cosy_llm_gguf}" "--in-dir" "${_tcb_cosy_ref}"
            REQUIRES "${_tcb_cosy_llm_gguf}" "${_tcb_cosy_ref}/gen_tokens.npy")
        set_property(TEST test-cosyvoice-llm-gpu APPEND PROPERTY ENVIRONMENT "COSYVOICE_TEST_GPU=1")
        # Per-backend pins of the same harness, so a host carrying several GPU
        # backends covers each by name; TTS_CPP_GPU_BACKEND fails loud when the
        # named backend is absent, so an arm cannot pass on the CPU.
        foreach(_tcb_cosy_arm IN LISTS _tcb_cosy_arms)
            tts_cpp_register_test(test-cosyvoice-llm-${_tcb_cosy_arm}
                EXE      test-cosyvoice-llm
                LABEL    "gpu"
                ARGS     "--llm-gguf" "${_tcb_cosy_llm_gguf}" "--in-dir" "${_tcb_cosy_ref}"
                REQUIRES "${_tcb_cosy_llm_gguf}" "${_tcb_cosy_ref}/gen_tokens.npy")
            set_property(TEST test-cosyvoice-llm-${_tcb_cosy_arm} APPEND PROPERTY ENVIRONMENT
                         "COSYVOICE_TEST_GPU=1" "TTS_CPP_GPU_BACKEND=${_tcb_cosy_arm}")
        endforeach()
        # Quantized-weight variants: mul_mat_f32acc specifically changes the
        # f16/quantized accumulation paths (ggml-vulkan reduces those in f16
        # under PREC_DEFAULT), which the f32 fixture cannot exercise. Same
        # binary and gates; the cross-backend leg compares CPU and GPU on the
        # same quantized GGUF, so the f32-derived reference prefix still
        # applies. Auto-disabled when the quantized fixture is not staged.
        foreach(_tcb_cosy_quant q8_0 q4_0)
            set(_tcb_cosy_llm_quant_gguf "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-llm-${_tcb_cosy_quant}.gguf")
            tts_cpp_register_test(test-cosyvoice-llm-gpu-${_tcb_cosy_quant}
                EXE      test-cosyvoice-llm
                LABEL    "gpu"
                ARGS     "--llm-gguf" "${_tcb_cosy_llm_quant_gguf}" "--in-dir" "${_tcb_cosy_ref}"
                REQUIRES "${_tcb_cosy_llm_quant_gguf}" "${_tcb_cosy_ref}/gen_tokens.npy")
            set_property(TEST test-cosyvoice-llm-gpu-${_tcb_cosy_quant} APPEND PROPERTY ENVIRONMENT "COSYVOICE_TEST_GPU=1")
            foreach(_tcb_cosy_arm IN LISTS _tcb_cosy_arms)
                tts_cpp_register_test(test-cosyvoice-llm-${_tcb_cosy_arm}-${_tcb_cosy_quant}
                    EXE      test-cosyvoice-llm
                    LABEL    "gpu"
                    ARGS     "--llm-gguf" "${_tcb_cosy_llm_quant_gguf}" "--in-dir" "${_tcb_cosy_ref}"
                    REQUIRES "${_tcb_cosy_llm_quant_gguf}" "${_tcb_cosy_ref}/gen_tokens.npy")
                set_property(TEST test-cosyvoice-llm-${_tcb_cosy_arm}-${_tcb_cosy_quant} APPEND PROPERTY ENVIRONMENT
                             "COSYVOICE_TEST_GPU=1" "TTS_CPP_GPU_BACKEND=${_tcb_cosy_arm}")
            endforeach()
        endforeach()
    endif()

    add_executable(test-cosyvoice-hift test/test_cosyvoice_hift.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(test-cosyvoice-hift PRIVATE ggml)
    target_include_directories(test-cosyvoice-hift PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-cosyvoice-hift)
    tts_cpp_register_test(test-cosyvoice-hift
        LABEL    "fixture"
        ARGS     "--hift-gguf" "${_tcb_cosy_hift_gguf}" "--in-dir" "${_tcb_cosy_ref}"
        REQUIRES "${_tcb_cosy_hift_gguf}" "${_tcb_cosy_ref}/hift_wav.npy")

    if (TTS_CPP_COSYVOICE_GPU)
        tts_cpp_register_test(test-cosyvoice-hift-gpu
            EXE      test-cosyvoice-hift
            LABEL    "gpu"
            ARGS     "--hift-gguf" "${_tcb_cosy_hift_gguf}" "--in-dir" "${_tcb_cosy_ref}"
            REQUIRES "${_tcb_cosy_hift_gguf}" "${_tcb_cosy_ref}/hift_wav.npy")
        set_property(TEST test-cosyvoice-hift-gpu APPEND PROPERTY ENVIRONMENT "COSYVOICE_TEST_GPU=1")
        # Per-backend pins of the same harness, so a host carrying several GPU
        # backends covers each by name; TTS_CPP_GPU_BACKEND fails loud when the
        # named backend is absent, so an arm cannot pass on the CPU.
        foreach(_tcb_cosy_arm IN LISTS _tcb_cosy_arms)
            tts_cpp_register_test(test-cosyvoice-hift-${_tcb_cosy_arm}
                EXE      test-cosyvoice-hift
                LABEL    "gpu"
                ARGS     "--hift-gguf" "${_tcb_cosy_hift_gguf}" "--in-dir" "${_tcb_cosy_ref}"
                REQUIRES "${_tcb_cosy_hift_gguf}" "${_tcb_cosy_ref}/hift_wav.npy")
            set_property(TEST test-cosyvoice-hift-${_tcb_cosy_arm} APPEND PROPERTY ENVIRONMENT
                         "COSYVOICE_TEST_GPU=1" "TTS_CPP_GPU_BACKEND=${_tcb_cosy_arm}")
        endforeach()
    endif()

    # ---- flow / HiFT weight-tier parity ------------------------------------
    # Thresholds are the measured tier deviations with margin: docs/cosyvoice3.md.
    add_executable(test-cosyvoice-tier test/test_cosyvoice_tier.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(test-cosyvoice-tier PRIVATE ggml)
    target_include_directories(test-cosyvoice-tier PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-cosyvoice-tier)
    # The flow gates are per-arch because both metrics measure the host CPU
    # backend's matmul rounding as much as they measure the tier GGUF: the
    # harness drives the DiT with deliberately synthetic off-distribution
    # inputs, so the x86 and ARM quantized paths land on measurably different
    # deviations from the same weights. A broken tier instead collapses the
    # cosine, far past either column.
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        set(_tcb_cosy_flow_tier_gates
            "f16|0.998|2.0"
            "bf16|0.997|4.0"
            "q8_0|0.997|4.0"
            "q4_0|0.96|8.0")
    else()
        set(_tcb_cosy_flow_tier_gates
            "f16|0.998|2.0"
            "bf16|0.998|2.0"
            "q8_0|0.998|2.0"
            "q4_0|0.98|8.0")
    endif()
    foreach(_tcb_cosy_tier_gate IN LISTS _tcb_cosy_flow_tier_gates)
        string(REPLACE "|" ";" _tcb_cosy_tier_gate "${_tcb_cosy_tier_gate}")
        list(GET _tcb_cosy_tier_gate 0 _tcb_cosy_tier)
        list(GET _tcb_cosy_tier_gate 1 _tcb_cosy_tier_min_cosine)
        list(GET _tcb_cosy_tier_gate 2 _tcb_cosy_tier_max_abs)
        set(_tcb_cosy_flow_tier_gguf "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-flow-${_tcb_cosy_tier}.gguf")
        tts_cpp_register_test(test-cosyvoice-flow-tier-${_tcb_cosy_tier}
            EXE      test-cosyvoice-tier
            LABEL    "fixture"
            ARGS     "--stage" "flow" "--ref-gguf" "${_tcb_cosy_flow_gguf}"
                     "--tier-gguf" "${_tcb_cosy_flow_tier_gguf}"
                     "--min-cosine" "${_tcb_cosy_tier_min_cosine}"
                     "--max-abs" "${_tcb_cosy_tier_max_abs}"
            REQUIRES "${_tcb_cosy_flow_gguf}" "${_tcb_cosy_flow_tier_gguf}")
    endforeach()
    set(_tcb_cosy_hift_f16_gguf "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-hift-f16.gguf")
    tts_cpp_register_test(test-cosyvoice-hift-tier-f16
        EXE      test-cosyvoice-tier
        LABEL    "fixture"
        ARGS     "--stage" "hift" "--ref-gguf" "${_tcb_cosy_hift_gguf}"
                 "--tier-gguf" "${_tcb_cosy_hift_f16_gguf}"
                 "--min-cosine" "0.999" "--max-abs" "0.01"
        REQUIRES "${_tcb_cosy_hift_gguf}" "${_tcb_cosy_hift_f16_gguf}")

    # ---- speech_tokenizer_v3 (CosyVoice3 zero-shot cloning front-end) -----
    # Fixtures come from scripts/dump-s3tokenizer-v3-reference.py; the GGUFs
    # from scripts/convert-s3tokenizer-v3-to-gguf.py.  Asserting harness (the
    # chatterbox test-s3tokenizer only reports): exact token count, minimum
    # exact-match ratio, elementwise log-mel bound.  The f16/q8_0 variants
    # exercise the type-adopting weight loader; q8_0's looser bound is the
    # measured FSQ near-boundary flip rate with margin.
    set(_tcb_cosy_s3tok_f32  "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-s3tok-f32.gguf")
    set(_tcb_cosy_s3tok_ref  "${TTS_CPP_TEST_REF_DIR}/s3tok-v3-ref")

    add_executable(test-s3tokenizer-v3
        test/test_s3tokenizer_v3.cpp
        src/s3tokenizer.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-s3tokenizer-v3 PRIVATE ggml)
    target_include_directories(test-s3tokenizer-v3 PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-s3tokenizer-v3)
    tts_cpp_register_test(test-s3tokenizer-v3
        LABEL    "fixture"
        ARGS     "--s3tok-gguf" "${_tcb_cosy_s3tok_f32}" "--in-dir" "${_tcb_cosy_s3tok_ref}"
        REQUIRES "${_tcb_cosy_s3tok_f32}" "${_tcb_cosy_s3tok_ref}/tokens.npy")
    foreach(_tcb_s3tok_quant f16 q8_0)
        set(_tcb_cosy_s3tok_quant_gguf "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-s3tok-${_tcb_s3tok_quant}.gguf")
        if (_tcb_s3tok_quant STREQUAL "q8_0")
            set(_tcb_s3tok_match "--min-token-match" "0.93")
        else()
            set(_tcb_s3tok_match "")
        endif()
        tts_cpp_register_test(test-s3tokenizer-v3-${_tcb_s3tok_quant}
            EXE      test-s3tokenizer-v3
            LABEL    "fixture"
            ARGS     "--s3tok-gguf" "${_tcb_cosy_s3tok_quant_gguf}" "--in-dir" "${_tcb_cosy_s3tok_ref}" ${_tcb_s3tok_match}
            REQUIRES "${_tcb_cosy_s3tok_quant_gguf}" "${_tcb_cosy_s3tok_ref}/tokens.npy")
    endforeach()
    if (TTS_CPP_COSYVOICE_GPU)
        tts_cpp_register_test(test-s3tokenizer-v3-gpu
            EXE      test-s3tokenizer-v3
            LABEL    "gpu"
            ARGS     "--s3tok-gguf" "${_tcb_cosy_s3tok_f32}" "--in-dir" "${_tcb_cosy_s3tok_ref}"
            REQUIRES "${_tcb_cosy_s3tok_f32}" "${_tcb_cosy_s3tok_ref}/tokens.npy")
        set_property(TEST test-s3tokenizer-v3-gpu APPEND PROPERTY ENVIRONMENT "COSYVOICE_TEST_GPU=1")
        foreach(_tcb_cosy_arm IN LISTS _tcb_cosy_arms)
            tts_cpp_register_test(test-s3tokenizer-v3-${_tcb_cosy_arm}
                EXE      test-s3tokenizer-v3
                LABEL    "gpu"
                ARGS     "--s3tok-gguf" "${_tcb_cosy_s3tok_f32}" "--in-dir" "${_tcb_cosy_s3tok_ref}"
                REQUIRES "${_tcb_cosy_s3tok_f32}" "${_tcb_cosy_s3tok_ref}/tokens.npy")
            set_property(TEST test-s3tokenizer-v3-${_tcb_cosy_arm} APPEND PROPERTY ENVIRONMENT
                         "COSYVOICE_TEST_GPU=1" "TTS_CPP_GPU_BACKEND=${_tcb_cosy_arm}")
        endforeach()
        set(_tcb_cosy_s3tok_q8_gguf "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-s3tok-q8_0.gguf")
        tts_cpp_register_test(test-s3tokenizer-v3-gpu-q8_0
            EXE      test-s3tokenizer-v3
            LABEL    "gpu"
            ARGS     "--s3tok-gguf" "${_tcb_cosy_s3tok_q8_gguf}" "--in-dir" "${_tcb_cosy_s3tok_ref}" "--min-token-match" "0.93"
            REQUIRES "${_tcb_cosy_s3tok_q8_gguf}" "${_tcb_cosy_s3tok_ref}/tokens.npy")
        set_property(TEST test-s3tokenizer-v3-gpu-q8_0 APPEND PROPERTY ENVIRONMENT "COSYVOICE_TEST_GPU=1")
        foreach(_tcb_cosy_arm IN LISTS _tcb_cosy_arms)
            tts_cpp_register_test(test-s3tokenizer-v3-${_tcb_cosy_arm}-q8_0
                EXE      test-s3tokenizer-v3
                LABEL    "gpu"
                ARGS     "--s3tok-gguf" "${_tcb_cosy_s3tok_q8_gguf}" "--in-dir" "${_tcb_cosy_s3tok_ref}" "--min-token-match" "0.93"
                REQUIRES "${_tcb_cosy_s3tok_q8_gguf}" "${_tcb_cosy_s3tok_ref}/tokens.npy")
            set_property(TEST test-s3tokenizer-v3-${_tcb_cosy_arm}-q8_0 APPEND PROPERTY ENVIRONMENT
                         "COSYVOICE_TEST_GPU=1" "TTS_CPP_GPU_BACKEND=${_tcb_cosy_arm}")
        endforeach()
    endif()

    # ---- CosyVoice3 cloning front-end round-trip -------------------------
    # Runs the native reference-audio front-end on the exact wav the baked
    # voice.gguf was enrolled from and asserts it reproduces the upstream
    # frontend_zero_shot tensors in cosyvoice3-ref (tokens / prompt mel /
    # CAM++ embedding + the 2:1 alignment invariant).  The guards variant is
    # fixture-free and pins the fail-closed paths (duration bounds, missing
    # GGUF/wav) on every checkout.
    set(_tcb_cosy_campplus_gguf "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-campplus-f32.gguf")
    set(_tcb_cosy_prompt_wav    "${_tcb_cosy_ref}/zero_shot_prompt.wav")

    add_executable(test-cosyvoice-frontend
        test/test_cosyvoice_frontend.cpp
        src/cosyvoice_frontend.cpp
        src/s3tokenizer.cpp
        src/campplus.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-cosyvoice-frontend PRIVATE ggml)
    target_include_directories(test-cosyvoice-frontend PRIVATE ggml/include src)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(test-cosyvoice-frontend PRIVATE OpenMP::OpenMP_CXX)
    endif()
    tts_cpp_apply_ccache(test-cosyvoice-frontend)
    tts_cpp_register_test(test-cosyvoice-frontend-guards
        EXE   test-cosyvoice-frontend
        LABEL "unit"
        ARGS  "--guards-only")
    tts_cpp_register_test(test-cosyvoice-frontend
        LABEL    "fixture"
        ARGS     "--s3tok-gguf" "${_tcb_cosy_s3tok_f32}" "--campplus-gguf" "${_tcb_cosy_campplus_gguf}"
                 "--wav" "${_tcb_cosy_prompt_wav}" "--in-dir" "${_tcb_cosy_ref}"
        REQUIRES "${_tcb_cosy_s3tok_f32}" "${_tcb_cosy_campplus_gguf}"
                 "${_tcb_cosy_prompt_wav}" "${_tcb_cosy_ref}/prompt_stok.npy")
    if (TTS_CPP_COSYVOICE_GPU)
        tts_cpp_register_test(test-cosyvoice-frontend-gpu
            EXE      test-cosyvoice-frontend
            LABEL    "gpu"
            ARGS     "--s3tok-gguf" "${_tcb_cosy_s3tok_f32}" "--campplus-gguf" "${_tcb_cosy_campplus_gguf}"
                     "--wav" "${_tcb_cosy_prompt_wav}" "--in-dir" "${_tcb_cosy_ref}"
            REQUIRES "${_tcb_cosy_s3tok_f32}" "${_tcb_cosy_campplus_gguf}"
                     "${_tcb_cosy_prompt_wav}" "${_tcb_cosy_ref}/prompt_stok.npy")
        set_property(TEST test-cosyvoice-frontend-gpu APPEND PROPERTY ENVIRONMENT "COSYVOICE_TEST_GPU=1")
        foreach(_tcb_cosy_arm IN LISTS _tcb_cosy_arms)
            tts_cpp_register_test(test-cosyvoice-frontend-${_tcb_cosy_arm}
                EXE      test-cosyvoice-frontend
                LABEL    "gpu"
                ARGS     "--s3tok-gguf" "${_tcb_cosy_s3tok_f32}" "--campplus-gguf" "${_tcb_cosy_campplus_gguf}"
                         "--wav" "${_tcb_cosy_prompt_wav}" "--in-dir" "${_tcb_cosy_ref}"
                REQUIRES "${_tcb_cosy_s3tok_f32}" "${_tcb_cosy_campplus_gguf}"
                         "${_tcb_cosy_prompt_wav}" "${_tcb_cosy_ref}/prompt_stok.npy")
            set_property(TEST test-cosyvoice-frontend-${_tcb_cosy_arm} APPEND PROPERTY ENVIRONMENT
                         "COSYVOICE_TEST_GPU=1" "TTS_CPP_GPU_BACKEND=${_tcb_cosy_arm}")
        endforeach()
    endif()

    # Thread-safety guard for the shared mel/fbank extraction (concurrent
    # engine constructions bake voice conditioning in parallel).  Synthetic
    # inputs; no fixtures.
    add_executable(test-mel-thread-safety
        test/test_mel_thread_safety.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-mel-thread-safety PRIVATE ggml)
    target_include_directories(test-mel-thread-safety PRIVATE ggml/include src)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(test-mel-thread-safety PRIVATE OpenMP::OpenMP_CXX)
    endif()
    tts_cpp_apply_ccache(test-mel-thread-safety)
    tts_cpp_register_test(test-mel-thread-safety LABEL "unit")

    # Branch coverage for assemble-cosyvoice3-model.py (add-on name
    # validation, inputs already at their destination, copy + symlink modes).
    # Registered only when a Python interpreter is available.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        if (TARGET cosyvoice-bench)
            add_test(NAME test-cosyvoice-bench-overrides
                     COMMAND "${Python3_EXECUTABLE}"
                             "${CMAKE_CURRENT_SOURCE_DIR}/test/test_cosyvoice_bench.py"
                             $<TARGET_FILE:cosyvoice-bench> "${TTS_CPP_TEST_MODEL_DIR}"
                             "${_tcb_cosy_llm_gguf}" "${_tcb_cosy_flow_gguf}" "${_tcb_cosy_hift_gguf}")
            set_tests_properties(test-cosyvoice-bench-overrides PROPERTIES
                LABELS "fixture"
                SKIP_RETURN_CODE 77)
            if (WIN32)
                set_property(TEST test-cosyvoice-bench-overrides PROPERTY ENVIRONMENT
                    "PATH=${CMAKE_BINARY_DIR}/bin/$<CONFIG>\;${CMAKE_BINARY_DIR}/bin\;$ENV{PATH}")
            endif()
        endif()
        add_test(NAME test-assemble-cosyvoice3-model
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_assemble_cosyvoice3_model.py")
        set_property(TEST test-assemble-cosyvoice3-model PROPERTY LABELS "unit")
        add_test(NAME test-convert-cosyvoice3-flow
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_convert_cosyvoice3_flow.py")
        set_property(TEST test-convert-cosyvoice3-flow PROPERTY LABELS "unit")
        add_test(NAME test-convert-cosyvoice3-llm
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_convert_cosyvoice3_llm.py")
        set_property(TEST test-convert-cosyvoice3-llm PROPERTY LABELS "unit")

        add_test(NAME test-convert-moss
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_convert_moss.py")
        set_property(TEST test-convert-moss PROPERTY LABELS "unit")
        set_property(TEST test-convert-moss PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
        set_property(TEST test-convert-moss PROPERTY SKIP_REGULAR_EXPRESSION "SKIP: ")

        # Skips without torch; the TTS CI macOS lane runs it in the export venv.
        add_test(NAME test-audio8-codec-coreml-exporter
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_CURRENT_SOURCE_DIR}/test/test_export_audio8_codec_coreml.py")
        set_property(TEST test-audio8-codec-coreml-exporter PROPERTY LABELS "unit")
        set_property(TEST test-audio8-codec-coreml-exporter PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

        # The GGUF helper tests (aliases, quantized counter, stem rule) need
        # only numpy; the Vocoder rebuild tests additionally need torch and
        # run in the Core ML export venv (the tts-ci macOS fixture step). A
        # numpy-less interpreter reports Skipped, never a vacuous pass.
        add_test(NAME test-supertonic-coreml-exporter
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_CURRENT_SOURCE_DIR}/test/test_export_supertonic_coreml.py")
        set_property(TEST test-supertonic-coreml-exporter PROPERTY LABELS "unit")
        set_property(TEST test-supertonic-coreml-exporter PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
        set_property(TEST test-supertonic-coreml-exporter PROPERTY SKIP_REGULAR_EXPRESSION "SKIP: numpy")

        foreach(_moss_exporter IN ITEMS speech_tokenizer sfx codec)
            string(REPLACE "_" "-" _moss_exporter_test "${_moss_exporter}")
            add_test(NAME test-moss-${_moss_exporter_test}-coreml-exporter
                     COMMAND "${Python3_EXECUTABLE}"
                             "${CMAKE_CURRENT_SOURCE_DIR}/test/test_export_moss_${_moss_exporter}_coreml.py")
            set_property(TEST test-moss-${_moss_exporter_test}-coreml-exporter PROPERTY LABELS "unit")
            set_property(TEST test-moss-${_moss_exporter_test}-coreml-exporter
                         PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
            set_property(TEST test-moss-${_moss_exporter_test}-coreml-exporter
                         PROPERTY SKIP_REGULAR_EXPRESSION "SKIP: ")
        endforeach()
    endif()

    # CLI-level coverage for the cloning flags: the engine harnesses bypass
    # argument parsing, so a mapping regression would not fail them.  The
    # synthesis legs are fixture-gated like the other cosyvoice tests and pin
    # the engine's mode log; FAIL_REGULAR_EXPRESSION keeps a failed synthesis
    # from passing on the log line alone.  The missing-value leg is
    # always-on: PASS_REGEX requires the usage text.
    if (TARGET cosyvoice-cli)
        tts_cpp_register_test(test-cosyvoice-cli-clone-zero-shot
            EXE        cosyvoice-cli
            LABEL      "fixture"
            PASS_REGEX "zero-shot mode"
            ARGS       "--model-dir" "${TTS_CPP_TEST_MODEL_DIR}"
                       "--reference-audio" "${_tcb_cosy_prompt_wav}"
                       "--prompt-text" "reference transcript" "--text" "Hi." "--greedy"
                       "--out" "cosyvoice-cli-clone-zs.wav"
            REQUIRES   "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-llm-f32.gguf"
                       "${_tcb_cosy_s3tok_f32}" "${_tcb_cosy_campplus_gguf}"
                       "${_tcb_cosy_prompt_wav}")
        set_property(TEST test-cosyvoice-cli-clone-zero-shot PROPERTY
            FAIL_REGULAR_EXPRESSION "cosyvoice-cli: error")
        tts_cpp_register_test(test-cosyvoice-cli-clone-cross-lingual
            EXE        cosyvoice-cli
            LABEL      "fixture"
            PASS_REGEX "cross-lingual mode"
            ARGS       "--model-dir" "${TTS_CPP_TEST_MODEL_DIR}"
                       "--reference-audio" "${_tcb_cosy_prompt_wav}"
                       "--text" "Hi." "--greedy"
                       "--out" "cosyvoice-cli-clone-xl.wav"
            REQUIRES   "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-llm-f32.gguf"
                       "${_tcb_cosy_s3tok_f32}" "${_tcb_cosy_campplus_gguf}"
                       "${_tcb_cosy_prompt_wav}")
        set_property(TEST test-cosyvoice-cli-clone-cross-lingual PROPERTY
            FAIL_REGULAR_EXPRESSION "cosyvoice-cli: error")
        tts_cpp_register_test(test-cosyvoice-cli-clone-missing-value
            EXE        cosyvoice-cli
            LABEL      "unit"
            PASS_REGEX "usage:"
            ARGS       "--reference-audio")
    endif()

    # ---- CosyVoice3 engine-level cloning ---------------------------------
    # Full-engine legs with the LM trajectory pinned: cloned-vs-baked
    # waveform equivalence (voice.gguf was enrolled from the same wav),
    # cross-lingual-vs-zero-shot bit-equality (the mode may only change the
    # LM prompt), and the fail-closed constructor paths.  Needs the full
    # model dir (llm/flow/hift/voice + s3tok/campplus).
    add_executable(test-cosyvoice-clone test/test_cosyvoice_clone.cpp)
    target_link_libraries(test-cosyvoice-clone PRIVATE tts-cpp ggml)
    target_include_directories(test-cosyvoice-clone PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-cosyvoice-clone)
    tts_cpp_register_test(test-cosyvoice-clone
        LABEL    "fixture"
        ARGS     "--model-dir" "${TTS_CPP_TEST_MODEL_DIR}" "--wav" "${_tcb_cosy_prompt_wav}"
                 "--tokens-npy" "${_tcb_cosy_ref}/gen_tokens.npy" "--s3tok-gguf" "${_tcb_cosy_s3tok_f32}" "--campplus-gguf" "${_tcb_cosy_campplus_gguf}"
        REQUIRES "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-llm-f32.gguf"
                 "${TTS_CPP_TEST_MODEL_DIR}/voice.gguf"
                 "${_tcb_cosy_s3tok_f32}" "${_tcb_cosy_campplus_gguf}"
                 "${_tcb_cosy_prompt_wav}" "${_tcb_cosy_ref}/gen_tokens.npy")
    if (TTS_CPP_COSYVOICE_GPU)
        tts_cpp_register_test(test-cosyvoice-clone-gpu
            EXE      test-cosyvoice-clone
            LABEL    "gpu"
            ARGS     "--model-dir" "${TTS_CPP_TEST_MODEL_DIR}" "--wav" "${_tcb_cosy_prompt_wav}"
                     "--tokens-npy" "${_tcb_cosy_ref}/gen_tokens.npy" "--s3tok-gguf" "${_tcb_cosy_s3tok_f32}" "--campplus-gguf" "${_tcb_cosy_campplus_gguf}"
            REQUIRES "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-llm-f32.gguf"
                     "${TTS_CPP_TEST_MODEL_DIR}/voice.gguf"
                     "${_tcb_cosy_s3tok_f32}" "${_tcb_cosy_campplus_gguf}"
                     "${_tcb_cosy_prompt_wav}" "${_tcb_cosy_ref}/gen_tokens.npy")
        set_property(TEST test-cosyvoice-clone-gpu APPEND PROPERTY ENVIRONMENT "COSYVOICE_TEST_GPU=1")
        foreach(_tcb_cosy_arm IN LISTS _tcb_cosy_arms)
            tts_cpp_register_test(test-cosyvoice-clone-${_tcb_cosy_arm}
                EXE      test-cosyvoice-clone
                LABEL    "gpu"
                ARGS     "--model-dir" "${TTS_CPP_TEST_MODEL_DIR}" "--wav" "${_tcb_cosy_prompt_wav}"
                         "--tokens-npy" "${_tcb_cosy_ref}/gen_tokens.npy" "--s3tok-gguf" "${_tcb_cosy_s3tok_f32}" "--campplus-gguf" "${_tcb_cosy_campplus_gguf}"
                REQUIRES "${TTS_CPP_TEST_MODEL_DIR}/cosyvoice3-llm-f32.gguf"
                         "${TTS_CPP_TEST_MODEL_DIR}/voice.gguf"
                         "${_tcb_cosy_s3tok_f32}" "${_tcb_cosy_campplus_gguf}"
                         "${_tcb_cosy_prompt_wav}" "${_tcb_cosy_ref}/gen_tokens.npy")
            set_property(TEST test-cosyvoice-clone-${_tcb_cosy_arm} APPEND PROPERTY ENVIRONMENT
                         "COSYVOICE_TEST_GPU=1" "TTS_CPP_GPU_BACKEND=${_tcb_cosy_arm}")
        endforeach()
    endif()

    # Deviceless policy test for GpuBackendRequirement filtering (the per-device
    # predicate of init_gpu_backend); guards the multi-backend selection order.
    add_executable(test-backend-requirement test/test_backend_requirement.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-backend-requirement PRIVATE ggml)
    target_include_directories(test-backend-requirement PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-backend-requirement)
    tts_cpp_register_test(test-backend-requirement LABEL "unit")

    # Serial-vs-parallel equality of the SineGen2 source excitation
    # (cosyvoice_sinegen2_source): every multi-threaded run must be
    # byte-identical to the 1-thread reference for the same f0 pattern and
    # seed. Host-side code only; synthetic inputs, no fixtures.
    add_executable(test-cosyvoice-sinegen test/test_cosyvoice_sinegen.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(test-cosyvoice-sinegen PRIVATE ggml)
    target_include_directories(test-cosyvoice-sinegen PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-cosyvoice-sinegen)
    tts_cpp_register_test(test-cosyvoice-sinegen LABEL "unit")

    # Non-contiguous-input guard in cosyvoice_conv1d_f32: a strided view must
    # match the pre-contiguous reference bit-for-bit on CPU, and (gpu variant)
    # the built graph must stay fully supported on the selected GPU backend
    # instead of demoting to the sched-fallback path. Synthetic inputs; no
    # fixtures.
    add_executable(test-cosyvoice-hexagon-graphs test/test_cosyvoice_hexagon_graphs.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(test-cosyvoice-hexagon-graphs PRIVATE ggml)
    target_include_directories(test-cosyvoice-hexagon-graphs PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-cosyvoice-hexagon-graphs)
    tts_cpp_register_test(test-cosyvoice-hexagon-graphs LABEL "unit")

    add_executable(test-cosyvoice-conv1d test/test_cosyvoice_conv1d.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(test-cosyvoice-conv1d PRIVATE ggml)
    target_include_directories(test-cosyvoice-conv1d PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-cosyvoice-conv1d)
    tts_cpp_register_test(test-cosyvoice-conv1d LABEL "unit")
    if (TTS_CPP_COSYVOICE_GPU)
        tts_cpp_register_test(test-cosyvoice-conv1d-gpu
            EXE   test-cosyvoice-conv1d
            LABEL "gpu")
        set_property(TEST test-cosyvoice-conv1d-gpu APPEND PROPERTY ENVIRONMENT "COSYVOICE_TEST_GPU=1")
        # Per-backend pins of the same harness, so a host carrying several GPU
        # backends covers each by name; TTS_CPP_GPU_BACKEND fails loud when the
        # named backend is absent, so an arm cannot pass on the CPU.
        foreach(_tcb_cosy_arm IN LISTS _tcb_cosy_arms)
            tts_cpp_register_test(test-cosyvoice-conv1d-${_tcb_cosy_arm}
                EXE   test-cosyvoice-conv1d
                LABEL "gpu")
            set_property(TEST test-cosyvoice-conv1d-${_tcb_cosy_arm} APPEND PROPERTY ENVIRONMENT
                         "COSYVOICE_TEST_GPU=1" "TTS_CPP_GPU_BACKEND=${_tcb_cosy_arm}")
        endforeach()
    endif()

    add_executable(test-cosyvoice-lm-attention test/test_cosyvoice_lm_attention.cpp)
    target_link_libraries(test-cosyvoice-lm-attention PRIVATE tts-cpp ggml)
    target_include_directories(test-cosyvoice-lm-attention PRIVATE src)
    tts_cpp_apply_ccache(test-cosyvoice-lm-attention)
    tts_cpp_register_test(test-cosyvoice-lm-attention LABEL "unit")
    if (TTS_CPP_GGML_CUDA AND _tts_cosy_desktop_platform)
        tts_cpp_register_test(test-cosyvoice-lm-attention-cuda
            EXE test-cosyvoice-lm-attention LABEL "gpu" ARGS "CUDA0")
    endif()

    add_executable(test-cosyvoice-xb test/test_cosyvoice_xb.cpp)
    target_link_libraries(test-cosyvoice-xb PRIVATE tts-cpp ggml)
    target_include_directories(test-cosyvoice-xb PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-cosyvoice-xb)
    if (TTS_CPP_COSYVOICE_GPU)
        tts_cpp_register_test(test-cosyvoice-xb
            LABEL    "gpu"
            ARGS     "--model-dir" "${TTS_CPP_TEST_MODEL_DIR}"
                     "--llm-gguf" "${_tcb_cosy_llm_gguf}"
                     "--flow-gguf" "${_tcb_cosy_flow_gguf}"
                     "--hift-gguf" "${_tcb_cosy_hift_gguf}"
            REQUIRES "${_tcb_cosy_llm_gguf}" "${_tcb_cosy_flow_gguf}" "${_tcb_cosy_hift_gguf}"
                     "${TTS_CPP_TEST_MODEL_DIR}/voice.gguf")
        set_property(TEST test-cosyvoice-xb APPEND PROPERTY ENVIRONMENT "COSYVOICE_TEST_GPU=1")
        foreach(_tcb_cosy_arm IN LISTS _tcb_cosy_arms)
            tts_cpp_register_test(test-cosyvoice-xb-${_tcb_cosy_arm}
                EXE      test-cosyvoice-xb
                LABEL    "gpu"
                ARGS     "--model-dir" "${TTS_CPP_TEST_MODEL_DIR}"
                         "--llm-gguf" "${_tcb_cosy_llm_gguf}"
                         "--flow-gguf" "${_tcb_cosy_flow_gguf}"
                         "--hift-gguf" "${_tcb_cosy_hift_gguf}"
                REQUIRES "${_tcb_cosy_llm_gguf}" "${_tcb_cosy_flow_gguf}" "${_tcb_cosy_hift_gguf}"
                         "${TTS_CPP_TEST_MODEL_DIR}/voice.gguf")
            set_property(TEST test-cosyvoice-xb-${_tcb_cosy_arm} APPEND PROPERTY ENVIRONMENT
                         "COSYVOICE_TEST_GPU=1" "TTS_CPP_GPU_BACKEND=${_tcb_cosy_arm}")
        endforeach()
    endif()

    # Map-in-place loader: byte-exact round-trip + clean failure on a truncated
    # GGUF. Self-contained (writes its own synthetic GGUF), so it always runs.
    add_executable(test-cosyvoice-load test/test_cosyvoice_load.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(test-cosyvoice-load PRIVATE ggml)
    target_include_directories(test-cosyvoice-load PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-cosyvoice-load)
    tts_cpp_register_test(test-cosyvoice-load LABEL "unit")

    add_executable(test-cosyvoice-bf16 test/test_cosyvoice_bf16.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(test-cosyvoice-bf16 PRIVATE ggml)
    target_include_directories(test-cosyvoice-bf16 PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-cosyvoice-bf16)
    tts_cpp_register_test(test-cosyvoice-bf16 LABEL "unit")

    # Sinusoidal timestep embedding for the flow DiT. Pure function on the CFM
    # step path, so it needs no GGUF.
    add_executable(test-cosyvoice-time-emb
        test/test_cosyvoice_time_emb.cpp ${_tcb_cosy_core_srcs})
    target_link_libraries(test-cosyvoice-time-emb PRIVATE ggml)
    target_include_directories(test-cosyvoice-time-emb PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-cosyvoice-time-emb)
    tts_cpp_register_test(test-cosyvoice-time-emb LABEL "unit")

    # Qwen2 BPE tokenizer parity (header-only; no GGUF weights, just the Qwen2
    # vocab.json/merges.txt bundled with the model + HF-reference ids from
    # scripts/dump-cosyvoice3-tokenizer-reference.py).
    set(_tcb_cosy_vocab  "${TTS_CPP_TEST_MODEL_DIR}/vocab.json")
    set(_tcb_cosy_merges "${TTS_CPP_TEST_MODEL_DIR}/merges.txt")
    add_executable(test-cosyvoice-tokenizer test/test_cosyvoice_tokenizer.cpp)
    target_include_directories(test-cosyvoice-tokenizer PRIVATE src)
    tts_cpp_apply_ccache(test-cosyvoice-tokenizer)
    tts_cpp_register_test(test-cosyvoice-tokenizer
        LABEL    "fixture"
        ARGS     "--vocab" "${_tcb_cosy_vocab}" "--merges" "${_tcb_cosy_merges}"
                 "--ref" "${_tcb_cosy_ref}/tokenizer_ref.txt"
        REQUIRES "${_tcb_cosy_vocab}" "${_tcb_cosy_merges}" "${_tcb_cosy_ref}/tokenizer_ref.txt")

    add_executable(test-moss-generation test/moss/test_generation.cpp)
    target_link_libraries(test-moss-generation PRIVATE tts-cpp)
    target_include_directories(test-moss-generation PRIVATE src)
    tts_cpp_apply_ccache(test-moss-generation)
    tts_cpp_register_test(test-moss-generation)

    add_executable(test-moss-fit-models test/moss/test_fit_models.cpp)
    target_link_libraries(test-moss-fit-models PRIVATE tts-cpp)
    tts_cpp_apply_ccache(test-moss-fit-models)
    tts_cpp_register_test(test-moss-fit-models LABEL "fixture")
    set_property(TEST test-moss-fit-models PROPERTY SKIP_RETURN_CODE 77)

    add_executable(test-moss-fit test/moss/test_fit.cpp)
    target_link_libraries(test-moss-fit PRIVATE tts-cpp)
    target_include_directories(test-moss-fit PRIVATE src)
    tts_cpp_apply_ccache(test-moss-fit)
    tts_cpp_register_test(test-moss-fit)

    add_executable(test-moss-load test/moss/test_load.cpp)
    target_link_libraries(test-moss-load PRIVATE tts-cpp)
    target_include_directories(test-moss-load PRIVATE src)
    tts_cpp_apply_ccache(test-moss-load)
    tts_cpp_register_test(test-moss-load)

    add_executable(test-moss-stream test/moss/test_stream.cpp)
    target_link_libraries(test-moss-stream PRIVATE tts-cpp)
    target_include_directories(test-moss-stream PRIVATE src)
    tts_cpp_apply_ccache(test-moss-stream)
    tts_cpp_register_test(test-moss-stream)

    add_executable(test-moss-cancel test/moss/test_cancel.cpp)
    target_link_libraries(test-moss-cancel PRIVATE tts-cpp)
    target_include_directories(test-moss-cancel PRIVATE src)
    tts_cpp_apply_ccache(test-moss-cancel)
    tts_cpp_register_test(test-moss-cancel)

    add_executable(test-moss-codec-stream test/moss/test_codec_stream.cpp)
    target_link_libraries(test-moss-codec-stream PRIVATE tts-cpp)
    target_include_directories(test-moss-codec-stream PRIVATE src)
    tts_cpp_apply_ccache(test-moss-codec-stream)
    tts_cpp_register_test(test-moss-codec-stream)

    add_executable(test-moss-codec-coreml test/moss/test_codec_coreml.cpp)
    target_link_libraries(test-moss-codec-coreml PRIVATE tts-cpp)
    target_include_directories(test-moss-codec-coreml PRIVATE src)
    tts_cpp_apply_ccache(test-moss-codec-coreml)
    tts_cpp_register_test(test-moss-codec-coreml)

    add_executable(test-moss-cli test/moss/test_cli.cpp src/moss/cli.cpp)
    target_link_libraries(test-moss-cli PRIVATE tts-cpp)
    target_include_directories(test-moss-cli PRIVATE src)
    tts_cpp_apply_ccache(test-moss-cli)
    tts_cpp_register_test(test-moss-cli)

    add_executable(test-moss-dialogue test/moss/test_dialogue.cpp)
    target_link_libraries(test-moss-dialogue PRIVATE tts-cpp)
    target_include_directories(test-moss-dialogue PRIVATE src)
    tts_cpp_apply_ccache(test-moss-dialogue)
    tts_cpp_register_test(test-moss-dialogue)

    add_executable(test-moss-sfx test/moss/test_sfx.cpp src/moss/cli.cpp)
    target_link_libraries(test-moss-sfx PRIVATE tts-cpp)
    target_include_directories(test-moss-sfx PRIVATE src)
    target_compile_options(test-moss-sfx PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
    tts_cpp_apply_ccache(test-moss-sfx)
    tts_cpp_register_test(test-moss-sfx)

    add_executable(test-moss-sfx-fit test/moss/test_sfx_fit.cpp)
    target_link_libraries(test-moss-sfx-fit PRIVATE tts-cpp)
    target_include_directories(test-moss-sfx-fit PRIVATE src)
    tts_cpp_apply_ccache(test-moss-sfx-fit)
    tts_cpp_register_test(test-moss-sfx-fit)

    add_executable(test-moss-speech test/moss/test_speech.cpp src/moss/cli.cpp)
    target_link_libraries(test-moss-speech PRIVATE tts-cpp)
    target_include_directories(test-moss-speech PRIVATE src)
    tts_cpp_apply_ccache(test-moss-speech)
    tts_cpp_register_test(test-moss-speech)

    add_executable(test-moss-sfx-parity test/moss/test_sfx_parity.cpp)
    target_link_libraries(test-moss-sfx-parity PRIVATE tts-cpp)
    target_include_directories(test-moss-sfx-parity PRIVATE src)
    tts_cpp_apply_ccache(test-moss-sfx-parity)
    tts_cpp_register_test(test-moss-sfx-parity LABEL "fixture")
    set_property(TEST test-moss-sfx-parity PROPERTY SKIP_RETURN_CODE 77)

    add_executable(test-moss-speech-parity test/moss/test_speech_parity.cpp)
    target_link_libraries(test-moss-speech-parity PRIVATE tts-cpp)
    target_include_directories(test-moss-speech-parity PRIVATE src)
    tts_cpp_apply_ccache(test-moss-speech-parity)
    tts_cpp_register_test(test-moss-speech-parity LABEL "fixture")
    set_property(TEST test-moss-speech-parity PROPERTY SKIP_RETURN_CODE 77)

    # Self-skips (77) without TTS_CPP_COREML or a staged MOSS sidecar
    # (MOSS_SPEECH_COREML_CODEC, MOSS_SFX_COREML_MODEL, MOSS_CODEC_COREML_DECODER).
    add_executable(test-moss-coreml-parity test/moss/test_coreml_parity.cpp)
    target_link_libraries(test-moss-coreml-parity PRIVATE tts-cpp)
    target_include_directories(test-moss-coreml-parity PRIVATE src)
    if (TTS_CPP_COREML)
        target_compile_definitions(test-moss-coreml-parity PRIVATE TTS_CPP_USE_COREML)
    endif()
    tts_cpp_apply_ccache(test-moss-coreml-parity)
    tts_cpp_register_test(test-moss-coreml-parity LABEL "fixture"
        ARGS "${CMAKE_CURRENT_SOURCE_DIR}/test/reference-audio/jfk.wav")
    set_property(TEST test-moss-coreml-parity PROPERTY SKIP_RETURN_CODE 77)

    # Not a test: times each staged MOSS sidecar against the ggml GPU stage
    # (median of 5 after a warm-up) and prints a markdown table.
    add_executable(bench-moss-coreml test/moss/bench_coreml.cpp)
    target_link_libraries(bench-moss-coreml PRIVATE tts-cpp)
    target_include_directories(bench-moss-coreml PRIVATE src)
    if (TTS_CPP_COREML)
        target_compile_definitions(bench-moss-coreml PRIVATE TTS_CPP_USE_COREML)
    endif()
    tts_cpp_apply_ccache(bench-moss-coreml)

    add_executable(test-backend-selection test/test_backend_selection.cpp)
    target_link_libraries(test-backend-selection PRIVATE tts-cpp)
    target_include_directories(test-backend-selection PRIVATE src)
    tts_cpp_apply_ccache(test-backend-selection)
    tts_cpp_register_test(test-backend-selection LABEL "unit")

    # ---- Pocket numerical core (fixtures from dump-pocket-flow-reference.py) ----
    add_executable(test-pocket-graph-memory test/pocket/test_graph_memory.cpp)
    target_include_directories(test-pocket-graph-memory PRIVATE src)
    target_link_libraries(test-pocket-graph-memory PRIVATE ggml)
    tts_cpp_register_test(test-pocket-graph-memory LABEL "unit")
    add_executable(test-pocket-generation test/pocket/test_generation.cpp)
    target_include_directories(test-pocket-generation PRIVATE src)
    tts_cpp_register_test(test-pocket-generation LABEL "unit")
    add_executable(test-pocket-reference-audio test/pocket/test_reference_audio.cpp src/pocket/reference_audio.cpp)
    target_include_directories(test-pocket-reference-audio PRIVATE src)
    tts_cpp_register_test(test-pocket-reference-audio LABEL "unit")
    set(TTS_CPP_POCKET_MODEL_DIR "${TTS_CPP_TEST_MODEL_DIR}/pocket"
        CACHE PATH "Complete Pocket bundle for frontend and engine fixtures")
    add_executable(test-pocket-frontend test/pocket/test_frontend.cpp src/pocket/frontend.cpp)
    target_include_directories(test-pocket-frontend PRIVATE src)
    tts_cpp_register_test(test-pocket-frontend LABEL "fixture"
        ARGS "${TTS_CPP_POCKET_MODEL_DIR}/frontend.json" "${TTS_CPP_POCKET_MODEL_DIR}/frontend-corpus.json"
        REQUIRES "${TTS_CPP_POCKET_MODEL_DIR}/frontend.json" "${TTS_CPP_POCKET_MODEL_DIR}/frontend-corpus.json")
    if (TARGET tts-cpp)
        add_executable(test-pocket-fit test/pocket/test_fit.cpp src/pocket/frontend.cpp)
        target_include_directories(test-pocket-fit PRIVATE src)
        target_link_libraries(test-pocket-fit PRIVATE tts-cpp ggml)
        tts_cpp_register_test(test-pocket-fit LABEL "fixture"
            ARGS "${TTS_CPP_POCKET_MODEL_DIR}"
            REQUIRES "${TTS_CPP_POCKET_MODEL_DIR}/flow-lm.gguf" "${TTS_CPP_POCKET_MODEL_DIR}/mimi.gguf"
                     "${TTS_CPP_POCKET_MODEL_DIR}/frontend.json" "${TTS_CPP_POCKET_MODEL_DIR}/voice.gguf")
        add_executable(test-pocket-engine test/pocket/test_engine.cpp src/pocket/frontend.cpp)
        target_include_directories(test-pocket-engine PRIVATE src)
        target_link_libraries(test-pocket-engine PRIVATE tts-cpp ggml)
        tts_cpp_register_test(test-pocket-engine LABEL "fixture"
            ARGS "${TTS_CPP_POCKET_MODEL_DIR}" "${TTS_CPP_POCKET_MODEL_DIR}/flow-lm.gguf"
            REQUIRES "${TTS_CPP_POCKET_MODEL_DIR}/flow-lm.gguf" "${TTS_CPP_POCKET_MODEL_DIR}/mimi.gguf"
                     "${TTS_CPP_POCKET_MODEL_DIR}/frontend.json" "${TTS_CPP_POCKET_MODEL_DIR}/voice.gguf")
    endif()
    set(TTS_CPP_POCKET_REFERENCE_DIR "${TTS_CPP_TEST_MODEL_DIR}/pocket-reference"
        CACHE PATH "Pocket FlowLM fixtures from dump-pocket-flow-reference.py")
    # Exercise the shipped objects (including their optimization flags) for
    # static builds. Shared builds intentionally hide this internal stage API.
    set(_pocket_test_sources src/pocket/flow_lm.cpp src/pocket/mimi.cpp src/backend_selection.cpp src/log.cpp)
    set(_pocket_test_library ggml)
    if (TARGET tts-cpp AND NOT TTS_CPP_BUILD_SHARED)
        set(_pocket_test_sources)
        set(_pocket_test_library tts-cpp)
    endif()
    add_executable(test-pocket-load test/pocket/test_load.cpp
        ${_pocket_test_sources})
    target_link_libraries(test-pocket-load PRIVATE ${_pocket_test_library})
    target_include_directories(test-pocket-load PRIVATE include src)
    tts_cpp_apply_ccache(test-pocket-load)
    tts_cpp_register_test(test-pocket-load LABEL "unit")
    add_executable(test-pocket-flow-lm test/pocket/test_flow_lm.cpp
        ${_pocket_test_sources})
    target_link_libraries(test-pocket-flow-lm PRIVATE ${_pocket_test_library})
    target_include_directories(test-pocket-flow-lm PRIVATE include src)
    tts_cpp_apply_ccache(test-pocket-flow-lm)
    tts_cpp_register_test(test-pocket-flow-lm LABEL "fixture"
        ARGS "${TTS_CPP_POCKET_REFERENCE_DIR}/flow-lm-f32.gguf" "${TTS_CPP_POCKET_REFERENCE_DIR}" "0.0001"
        REQUIRES "${TTS_CPP_POCKET_REFERENCE_DIR}/flow-lm-f32.gguf" "${TTS_CPP_POCKET_REFERENCE_DIR}/provenance.json")
    tts_cpp_register_test(test-pocket-flow-lm-f16 LABEL "fixture" EXE test-pocket-flow-lm
        ARGS "${TTS_CPP_POCKET_REFERENCE_DIR}/flow-lm-f16.gguf" "${TTS_CPP_POCKET_REFERENCE_DIR}" "0.01"
        REQUIRES "${TTS_CPP_POCKET_REFERENCE_DIR}/flow-lm-f16.gguf" "${TTS_CPP_POCKET_REFERENCE_DIR}/provenance.json")

    add_executable(test-pocket-mimi test/pocket/test_mimi.cpp ${_pocket_test_sources})
    target_link_libraries(test-pocket-mimi PRIVATE ${_pocket_test_library})
    target_include_directories(test-pocket-mimi PRIVATE include src)
    tts_cpp_apply_ccache(test-pocket-mimi)
    set(TTS_CPP_POCKET_MIMI_REFERENCE_DIR "${TTS_CPP_TEST_MODEL_DIR}/pocket-mimi-reference"
        CACHE PATH "Pocket Mimi fixtures from dump-pocket-mimi-reference.py")
    tts_cpp_register_test(test-pocket-mimi LABEL "fixture"
        ARGS "${TTS_CPP_POCKET_MIMI_REFERENCE_DIR}/mimi.gguf" "${TTS_CPP_POCKET_MIMI_REFERENCE_DIR}"
        REQUIRES "${TTS_CPP_POCKET_MIMI_REFERENCE_DIR}/mimi.gguf"
                 "${TTS_CPP_POCKET_MIMI_REFERENCE_DIR}/provenance.json"
                 "${TTS_CPP_POCKET_MIMI_REFERENCE_DIR}/latents.npy"
                 "${TTS_CPP_POCKET_MIMI_REFERENCE_DIR}/pcm.npy"
                 "${TTS_CPP_POCKET_MIMI_REFERENCE_DIR}/encoded.npy")

    # ---- Audio8 tests (fixtures from scripts/dump-audio8-*-reference.py) ----
    set(_tcb_audio8_core_srcs
        src/audio8/gguf.cpp
        src/audio8/graph.cpp
        src/backend_selection.cpp
        src/sched_dispatch.cpp
        src/log.cpp)

    # Model-backed, fixed-input CPU/backend comparison; no production changes.
    add_executable(test-audio8-backend-parity test/audio8/test_backend_parity.cpp
        src/audio8/lm.cpp src/audio8/tokenizer.cpp src/audio8/unicode.cpp
        src/audio8/codec_ops.cpp src/audio8/codec_decode.cpp src/audio8/window_stream.cpp
        ${_tcb_audio8_core_srcs})
    target_link_libraries(test-audio8-backend-parity PRIVATE ggml Threads::Threads)
    target_include_directories(test-audio8-backend-parity PRIVATE include src)
    tts_cpp_apply_ccache(test-audio8-backend-parity)
    add_test(NAME test-audio8-backend-parity COMMAND test-audio8-backend-parity)
    set_tests_properties(test-audio8-backend-parity PROPERTIES LABELS "fixture" SKIP_RETURN_CODE 77)

    # Fail-closed GGUF header reading: synthesized metadata-only files with a
    # wrong architecture and wrong-typed scalar/string keys. No fixtures.
    add_executable(test-audio8-gguf-load test/audio8/test_gguf_load.cpp
        ${_tcb_audio8_core_srcs})
    target_link_libraries(test-audio8-gguf-load PRIVATE ggml)
    target_include_directories(test-audio8-gguf-load PRIVATE include src)
    tts_cpp_apply_ccache(test-audio8-gguf-load)
    tts_cpp_register_test(test-audio8-gguf-load)

    add_executable(test-audio8-tokenizer test/audio8/test_tokenizer.cpp
        src/audio8/tokenizer.cpp src/audio8/unicode.cpp)
    target_link_libraries(test-audio8-tokenizer PRIVATE ggml)
    target_include_directories(test-audio8-tokenizer PRIVATE src)
    tts_cpp_apply_ccache(test-audio8-tokenizer)
    tts_cpp_register_test(test-audio8-tokenizer
        LABEL    "fixture"
        ARGS     "${_tcb_audio8_lm}" "${_tcb_audio8_ref}/tokenizer_ref.txt"
        REQUIRES "${_tcb_audio8_lm}" "${_tcb_audio8_ref}/tokenizer_ref.txt")

    add_executable(test-audio8-lm test/audio8/test_lm.cpp
        src/audio8/lm.cpp src/audio8/tokenizer.cpp src/audio8/unicode.cpp
        ${_tcb_audio8_core_srcs})
    target_link_libraries(test-audio8-lm PRIVATE ggml)
    target_include_directories(test-audio8-lm PRIVATE include src)
    tts_cpp_apply_ccache(test-audio8-lm)
    tts_cpp_register_test(test-audio8-lm
        LABEL    "fixture"
        ARGS     "${_tcb_audio8_lm}" "${_tcb_audio8_ref}"
        REQUIRES "${_tcb_audio8_lm}" "${_tcb_audio8_ref}/sem_logits.npy")

    # The fast-AR graph cache against the dual-path dispatch: the scheduler
    # must bypass it and must not leave it poisoned for the direct path after.
    # The load-time stacking of the LM's projections against the separate
    # weights, byte for byte, on a synthetic LM. No fixtures.
    add_executable(test-audio8-lm-fusion
        test/audio8/test_lm_fusion.cpp
        src/audio8/lm.cpp src/audio8/tokenizer.cpp src/audio8/unicode.cpp
        src/audio8/sampling.cpp ${_tcb_audio8_core_srcs})
    target_link_libraries(test-audio8-lm-fusion PRIVATE ggml)
    target_include_directories(test-audio8-lm-fusion PRIVATE include src test)
    tts_cpp_apply_ccache(test-audio8-lm-fusion)
    tts_cpp_register_test(test-audio8-lm-fusion LABEL "unit")

    add_executable(test-audio8-fast-cache-sched
        test/audio8/test_fast_cache_sched.cpp
        src/audio8/lm.cpp src/audio8/tokenizer.cpp src/audio8/unicode.cpp
        src/audio8/sampling.cpp ${_tcb_audio8_core_srcs})
    target_link_libraries(test-audio8-fast-cache-sched PRIVATE ggml)
    target_include_directories(test-audio8-fast-cache-sched PRIVATE include src test)
    tts_cpp_apply_ccache(test-audio8-fast-cache-sched)
    tts_cpp_register_test(test-audio8-fast-cache-sched
        LABEL    "fixture"
        ARGS     "${_tcb_audio8_lm}"
        REQUIRES "${_tcb_audio8_lm}")

    add_executable(test-audio8-quantised-precision
        test/audio8/test_quantised_precision.cpp
        src/audio8/lm.cpp src/audio8/tokenizer.cpp src/audio8/unicode.cpp
        src/audio8/codec_ops.cpp src/audio8/codec_decode.cpp src/audio8/codec_encode.cpp
        src/audio8/window_stream.cpp
        ${_tcb_audio8_core_srcs})
    target_link_libraries(test-audio8-quantised-precision PRIVATE ggml Threads::Threads)
    target_include_directories(test-audio8-quantised-precision PRIVATE include src)
    tts_cpp_apply_ccache(test-audio8-quantised-precision)
    tts_cpp_register_test(test-audio8-quantised-precision
        LABEL    "fixture"
        ARGS     "${_tcb_audio8_lm_q8}" "${_tcb_audio8_decoder_q8}"
        REQUIRES "${_tcb_audio8_lm_q8}" "${_tcb_audio8_decoder_q8}")

    add_executable(test-audio8-codec test/audio8/test_codec.cpp
        src/audio8/codec_ops.cpp src/audio8/codec_decode.cpp src/audio8/codec_encode.cpp
        src/audio8/window_stream.cpp
        ${_tcb_audio8_core_srcs})
    target_link_libraries(test-audio8-codec PRIVATE ggml Threads::Threads)
    target_include_directories(test-audio8-codec PRIVATE include src test)
    tts_cpp_apply_ccache(test-audio8-codec)
    tts_cpp_register_test(test-audio8-codec
        LABEL    "fixture"
        ARGS     "${_tcb_audio8_decoder}" "${_tcb_audio8_encoder}" "${_tcb_audio8_ref}"
        REQUIRES "${_tcb_audio8_decoder}" "${_tcb_audio8_encoder}"
                 "${_tcb_audio8_ref}/codec_wav.npy"
                 "${_tcb_audio8_ref}/codec_enc_codes.npy")

    add_executable(test-audio8-sampler test/audio8/test_sampler.cpp src/audio8/sampling.cpp)
    target_include_directories(test-audio8-sampler PRIVATE src)
    tts_cpp_apply_ccache(test-audio8-sampler)
    tts_cpp_register_test(test-audio8-sampler
        LABEL    "fixture"
        ARGS     "${_tcb_audio8_ref}"
        REQUIRES "${_tcb_audio8_ref}/filtered.npy" "${_tcb_audio8_ref}/meta.json")

    # The repetition-aware window, which the fixture dumps decode greedily and
    # therefore never exercise.  No fixtures, so it runs on a bare checkout.
    add_executable(test-audio8-ras test/audio8/test_ras.cpp src/audio8/sampling.cpp)
    target_include_directories(test-audio8-ras PRIVATE src)
    # A window of zero used to pop an empty deque, which is undefined rather
    # than observable: the sampler returns the same tokens either way, so only
    # a hardened container turns the regression into a failure.  Unlike
    # _GLIBCXX_DEBUG this leaves the ABI alone, and the target compiles its own
    # copy of sampling.cpp rather than linking tts-cpp, so nothing else sees it.
    target_compile_definitions(test-audio8-ras PRIVATE _GLIBCXX_ASSERTIONS)
    tts_cpp_apply_ccache(test-audio8-ras)
    tts_cpp_register_test(test-audio8-ras)

    # The top-k / top-p / temperature candidate filter on synthetic score
    # vectors, which the fixture sampler test only reaches through reference
    # dumps.  No fixtures, so it runs on a bare checkout.
    add_executable(test-audio8-sampling-filter test/audio8/test_sampling_filter.cpp
        src/audio8/sampling.cpp)
    target_include_directories(test-audio8-sampling-filter PRIVATE src)
    target_compile_definitions(test-audio8-sampling-filter PRIVATE _GLIBCXX_ASSERTIONS)
    tts_cpp_apply_ccache(test-audio8-sampling-filter)
    tts_cpp_register_test(test-audio8-sampling-filter)

    # End-to-end through the public engine, both the text path and the cloning
    # path, so it links the whole library rather than the stage sources.
    add_executable(test-audio8-engine test/audio8/test_engine.cpp)
    target_link_libraries(test-audio8-engine PRIVATE tts-cpp ggml)
    target_include_directories(test-audio8-engine PRIVATE include src test)
    tts_cpp_apply_ccache(test-audio8-engine)
    tts_cpp_register_test(test-audio8-engine
        LABEL    "fixture"
        ARGS     "${_tcb_audio8_lm}" "${_tcb_audio8_decoder}" "${_tcb_audio8_encoder}"
                 "${_tcb_audio8_ref}" "${_tcb_audio8_ref_clone}" "${_tcb_audio8_ref_wav}"
        REQUIRES "${_tcb_audio8_lm}" "${_tcb_audio8_decoder}" "${_tcb_audio8_encoder}"
                 "${_tcb_audio8_ref}/wav.npy" "${_tcb_audio8_ref_clone}/wav.npy"
                 "${_tcb_audio8_ref_wav}")

    # The three Audio8 suites again on one GPU backend, against the same torch
    # reference and the same fixtures.  Comparing the GPU to the reference rather
    # than to CPU is what makes this a correctness gate: the two arms are then
    # independent readings of one ground truth instead of a chain where CPU error
    # hides GPU error.  AUDIO8_TEST_GPU names the backend the arm is for, which
    # the harnesses assert before reading a number: both arms run the same
    # binaries, so an arm that only asked for "a GPU" would pass on the other
    # arm's backend whenever both are compiled in.
    function(tts_cpp_register_audio8_gpu_tests suffix)
        tts_cpp_register_test(test-audio8-lm-${suffix}
            EXE      test-audio8-lm
            LABEL    "gpu"
            ARGS     "${_tcb_audio8_lm}" "${_tcb_audio8_ref}"
            REQUIRES "${_tcb_audio8_lm}" "${_tcb_audio8_ref}/sem_logits.npy")
        tts_cpp_register_test(test-audio8-codec-${suffix}
            EXE      test-audio8-codec
            LABEL    "gpu"
            ARGS     "${_tcb_audio8_decoder}" "${_tcb_audio8_encoder}" "${_tcb_audio8_ref}"
            REQUIRES "${_tcb_audio8_decoder}" "${_tcb_audio8_encoder}"
                     "${_tcb_audio8_ref}/codec_wav.npy"
                     "${_tcb_audio8_ref}/codec_enc_codes.npy")
        tts_cpp_register_test(test-audio8-engine-${suffix}
            EXE      test-audio8-engine
            LABEL    "gpu"
            ARGS     "${_tcb_audio8_lm}" "${_tcb_audio8_decoder}" "${_tcb_audio8_encoder}"
                     "${_tcb_audio8_ref}" "${_tcb_audio8_ref_clone}" "${_tcb_audio8_ref_wav}"
            REQUIRES "${_tcb_audio8_lm}" "${_tcb_audio8_decoder}" "${_tcb_audio8_encoder}"
                     "${_tcb_audio8_ref}/wav.npy" "${_tcb_audio8_ref_clone}/wav.npy"
                     "${_tcb_audio8_ref_wav}")
        # Reads the marker the graph asks for, which is the half of the
        # quantised route the f32 fixtures cannot see.
        tts_cpp_register_test(test-audio8-quantised-precision-${suffix}
            EXE      test-audio8-quantised-precision
            LABEL    "gpu"
            ARGS     "${_tcb_audio8_lm_q8}" "${_tcb_audio8_decoder_q8}"
            REQUIRES "${_tcb_audio8_lm_q8}" "${_tcb_audio8_decoder_q8}")
        tts_cpp_register_test(test-audio8-fast-cache-sched-${suffix}
            EXE      test-audio8-fast-cache-sched
            LABEL    "gpu"
            ARGS     "${_tcb_audio8_lm}"
            REQUIRES "${_tcb_audio8_lm}")
        tts_cpp_register_test(test-audio8-lm-fusion-${suffix}
            EXE      test-audio8-lm-fusion
            LABEL    "gpu")
        set_property(TEST test-audio8-lm-${suffix} test-audio8-codec-${suffix}
                          test-audio8-engine-${suffix}
                          test-audio8-quantised-precision-${suffix}
                          test-audio8-fast-cache-sched-${suffix}
                          test-audio8-lm-fusion-${suffix} APPEND
                     PROPERTY ENVIRONMENT "AUDIO8_TEST_GPU=${suffix}"
                                          "TTS_CPP_GPU_BACKEND=${suffix}")
    endfunction()

    # The block formats take their own route through every backend -- Vulkan
    # picks fp32-arithmetic pipelines for them, CUDA drops the f32 marker and
    # takes the integer-dot path -- and the f32 fixtures above cannot reach
    # either.  One short greedy utterance per backend pins that the quantised
    # route still runs there and still reports the backend it ran on.
    function(tts_cpp_register_audio8_quantised_cli_test backend device_regex)
        if (NOT TARGET audio8-cli)
            return()
        endif()
        tts_cpp_register_test(test-audio8-cli-${backend}-q8_0
            EXE        audio8-cli
            LABEL      "gpu"
            ARGS       "--lm" "${_tcb_audio8_lm_q8}"
                       "--codec-decoder" "${_tcb_audio8_decoder_q8}"
                       "--text" "Audio8 ${backend} q8 regression."
                       "--greedy" "--max-frames" "2"
                       "--n-gpu-layers" "${_tcb_audio8_gpu_layers}"
                       "--out" "${CMAKE_CURRENT_BINARY_DIR}/audio8-${backend}-q8_0.wav"
            PASS_REGEX "${device_regex}"
            REQUIRES   "${_tcb_audio8_lm_q8}" "${_tcb_audio8_decoder_q8}")
        set_property(TEST test-audio8-cli-${backend}-q8_0 APPEND PROPERTY ENVIRONMENT
                     "TTS_CPP_GPU_BACKEND=${backend}")
    endfunction()

    # Audio8 allowlist: Vulkan / Metal / OpenCL / CUDA (audio8/gguf.cpp
    # GpuBackendRequirement::Vulkan | Metal | OpenCL | CUDA).
    if (TTS_CPP_GGML_CUDA)
        tts_cpp_register_audio8_gpu_tests(cuda)
        tts_cpp_register_audio8_quantised_cli_test(cuda "on CUDA")
    endif()
    if (TTS_CPP_GGML_VULKAN)
        tts_cpp_register_audio8_gpu_tests(vulkan)
        tts_cpp_register_audio8_quantised_cli_test(vulkan "on Vulkan")
    endif()
    if (TTS_CPP_GGML_METAL)
        tts_cpp_register_audio8_gpu_tests(metal)
    endif()
    if (TTS_CPP_GGML_OPENCL)
        tts_cpp_register_audio8_gpu_tests(opencl)
    endif()

    # The per-stage breakdown the Metal work is ranked by.  Needs the models but
    # none of the reference dumps: it checks the accounting, not the numbers.
    add_executable(test-audio8-timing test/audio8/test_timing.cpp)
    target_link_libraries(test-audio8-timing PRIVATE tts-cpp ggml)
    target_include_directories(test-audio8-timing PRIVATE include src)
    tts_cpp_apply_ccache(test-audio8-timing)
    tts_cpp_register_test(test-audio8-timing
        LABEL    "fixture"
        ARGS     "${_tcb_audio8_lm}" "${_tcb_audio8_decoder}" "${_tcb_audio8_encoder}"
                 "${_tcb_audio8_ref_wav}"
        REQUIRES "${_tcb_audio8_lm}" "${_tcb_audio8_decoder}" "${_tcb_audio8_encoder}"
                 "${_tcb_audio8_ref_wav}")

    # The CLI's own surface: the flags the bench scripts and the on-device
    # harness pass, and the code dump two runs get compared on.  No model, so it
    # runs everywhere.
    add_executable(test-audio8-codec-bench-cli test/audio8/test_codec_bench_cli.cpp)
    target_include_directories(test-audio8-codec-bench-cli PRIVATE src)
    tts_cpp_apply_ccache(test-audio8-codec-bench-cli)
    tts_cpp_register_test(test-audio8-codec-bench-cli LABEL "unit")

    # Model-free on every host: sidecar naming rule and window plan.
    add_executable(test-audio8-coreml-windows test/audio8/test_coreml_windows.cpp
        src/audio8/coreml_path.cpp src/coreml_sidecar_path.cpp)
    target_include_directories(test-audio8-coreml-windows PRIVATE src)
    tts_cpp_apply_ccache(test-audio8-coreml-windows)
    tts_cpp_register_test(test-audio8-coreml-windows LABEL "unit")

    # The synthesis stream that overlaps the Core ML codec with generation,
    # driven by a fake causal synthesizer. No models, so it runs everywhere.
    add_executable(test-audio8-window-stream test/audio8/test_window_stream.cpp
        src/audio8/window_stream.cpp)
    target_include_directories(test-audio8-window-stream PRIVATE src)
    target_link_libraries(test-audio8-window-stream PRIVATE Threads::Threads)
    tts_cpp_apply_ccache(test-audio8-window-stream)
    tts_cpp_register_test(test-audio8-window-stream LABEL "unit")

    # Model-free on every host: the Supertonic vocoder sidecar's naming rule.
    add_executable(test-supertonic-coreml-path test/test_supertonic_coreml_path.cpp
        src/supertonic_coreml_path.cpp src/coreml_sidecar_path.cpp)
    target_include_directories(test-supertonic-coreml-path PRIVATE src)
    tts_cpp_apply_ccache(test-supertonic-coreml-path)
    tts_cpp_register_test(test-supertonic-coreml-path LABEL "unit")

    # Model-free on every host: the weight-width floor that keeps the sidecar
    # off tiers it would substitute rather than accelerate.
    add_executable(test-supertonic-coreml-tier test/test_supertonic_coreml_tier.cpp)
    target_link_libraries(test-supertonic-coreml-tier PRIVATE tts-cpp ggml)
    target_include_directories(test-supertonic-coreml-tier PRIVATE include src)
    tts_cpp_apply_ccache(test-supertonic-coreml-tier)
    tts_cpp_register_test(test-supertonic-coreml-tier LABEL "unit")

    # Self-skips (77) without TTS_CPP_COREML or a staged
    # SUPERTONIC_COREML_TEST_MODELS_DIR holding a Supertonic GGUF with a
    # compiled `<model>-vocoder.mlmodelc` next to it; the tts-ci macOS lane
    # stages the fixture and runs the binary directly so a skip fails.
    add_executable(test-supertonic-coreml-parity test/test_supertonic_coreml_parity.cpp)
    target_link_libraries(test-supertonic-coreml-parity PRIVATE tts-cpp ggml)
    target_include_directories(test-supertonic-coreml-parity PRIVATE include src test)
    if (TTS_CPP_COREML)
        target_compile_definitions(test-supertonic-coreml-parity PRIVATE TTS_CPP_USE_COREML)
    endif()
    tts_cpp_apply_ccache(test-supertonic-coreml-parity)
    tts_cpp_register_test(test-supertonic-coreml-parity LABEL "fixture")
    set_property(TEST test-supertonic-coreml-parity PROPERTY SKIP_RETURN_CODE 77)

    # Self-skips (77) without TTS_CPP_COREML or a staged
    # AUDIO8_COREML_TEST_MODELS_DIR; CI runs the binary directly so a skip fails.
    add_executable(test-audio8-codec-coreml-parity test/audio8/test_codec_coreml_parity.cpp)
    target_link_libraries(test-audio8-codec-coreml-parity PRIVATE tts-cpp ggml)
    target_include_directories(test-audio8-codec-coreml-parity PRIVATE include src test)
    if (TTS_CPP_COREML)
        target_compile_definitions(test-audio8-codec-coreml-parity PRIVATE TTS_CPP_USE_COREML)
    endif()
    tts_cpp_apply_ccache(test-audio8-codec-coreml-parity)
    tts_cpp_register_test(test-audio8-codec-coreml-parity LABEL "fixture")
    set_property(TEST test-audio8-codec-coreml-parity PROPERTY SKIP_RETURN_CODE 77)

    # Not a ctest: CI appends its parity-gated markdown table to the job summary.
    add_executable(bench-audio8-codec-coreml test/audio8/bench_codec_coreml.cpp)
    target_link_libraries(bench-audio8-codec-coreml PRIVATE tts-cpp ggml)
    target_include_directories(bench-audio8-codec-coreml PRIVATE include src test)
    if (TTS_CPP_COREML)
        target_compile_definitions(bench-audio8-codec-coreml PRIVATE TTS_CPP_USE_COREML)
    endif()
    tts_cpp_apply_ccache(bench-audio8-codec-coreml)

    add_executable(test-audio8-cli test/audio8/test_cli.cpp src/audio8/cli.cpp)
    target_include_directories(test-audio8-cli PRIVATE src)
    tts_cpp_apply_ccache(test-audio8-cli)
    tts_cpp_register_test(test-audio8-cli
        ARGS "${CMAKE_CURRENT_BINARY_DIR}/audio8-cli-test-codes.txt")

    # ...and the same flags through the real binary, where --verbose has to reach
    # stderr and --dump-codes has to survive a synthesis.  Two frames of it.
    if (TARGET audio8-cli)
        tts_cpp_register_test(test-audio8-cli-verbose
            EXE        audio8-cli
            LABEL      "fixture"
            ARGS       "--lm" "${_tcb_audio8_lm}"
                       "--codec-decoder" "${_tcb_audio8_decoder}"
                       "--text" "Two frames is enough."
                       "--greedy" "--max-frames" "2" "--verbose"
                       "--out" "${CMAKE_CURRENT_BINARY_DIR}/audio8-cli-verbose.wav"
                       "--dump-codes" "${CMAKE_CURRENT_BINARY_DIR}/audio8-cli-verbose.txt"
            PASS_REGEX "\\[audio8-timing\\] total"
            REQUIRES   "${_tcb_audio8_lm}" "${_tcb_audio8_decoder}")
    endif()

    # ---- Parler-TTS tests (fixtures from scripts/parler/dump-reference.py) ----
    set(_tcb_parler_core_srcs
        src/parler/gguf.cpp
        src/backend_selection.cpp
        src/sched_dispatch.cpp
        src/log.cpp)

    # Fail-closed GGUF loading: synthesizes tiny metadata-only GGUFs (truncated
    # file, missing keys, missing tensors, wrong-typed key) and asserts the
    # loader rejects each one.  No fixtures, so it runs on a bare checkout.
    add_executable(test-parler-gguf-load test/parler/test_gguf_load.cpp
        ${_tcb_parler_core_srcs})
    target_link_libraries(test-parler-gguf-load PRIVATE ggml)
    target_include_directories(test-parler-gguf-load PRIVATE ggml/include include src)
    tts_cpp_apply_ccache(test-parler-gguf-load)
    tts_cpp_register_test(test-parler-gguf-load)

    add_executable(test-parler-delay test/parler/test_delay.cpp src/parler/delay.cpp)
    target_include_directories(test-parler-delay PRIVATE src)
    tts_cpp_apply_ccache(test-parler-delay)
    tts_cpp_register_test(test-parler-delay
        LABEL    "fixture"
        ARGS     "${_tcb_parler_ref}"
        REQUIRES "${_tcb_parler_ref}/delay_mask_len24.npy")

    add_executable(test-parler-sampler test/parler/test_sampler.cpp src/parler/sampler.cpp)
    target_include_directories(test-parler-sampler PRIVATE src)
    tts_cpp_apply_ccache(test-parler-sampler)
    tts_cpp_register_test(test-parler-sampler)

    add_executable(test-parler-text-norm test/parler/test_text_norm.cpp
        src/parler/text_norm.cpp)
    target_include_directories(test-parler-text-norm PRIVATE src)
    tts_cpp_apply_ccache(test-parler-text-norm)
    tts_cpp_register_test(test-parler-text-norm)

    add_executable(test-voice-controls test/test_voice_controls.cpp
        src/voice_controls.cpp)
    target_include_directories(test-voice-controls PRIVATE include src)
    tts_cpp_apply_ccache(test-voice-controls)
    tts_cpp_register_test(test-voice-controls)

    add_executable(test-cosyvoice-instruct test/test_cosyvoice_instruct.cpp
        src/cosyvoice_instruct.cpp src/voice_controls.cpp)
    target_include_directories(test-cosyvoice-instruct PRIVATE include src)
    tts_cpp_apply_ccache(test-cosyvoice-instruct)
    tts_cpp_register_test(test-cosyvoice-instruct)

    add_executable(test-supertonic-pace test/test_supertonic_pace.cpp
        src/supertonic_pace.cpp src/voice_controls.cpp)
    target_include_directories(test-supertonic-pace PRIVATE include src)
    tts_cpp_apply_ccache(test-supertonic-pace)
    tts_cpp_register_test(test-supertonic-pace)

    # Capability listing on the shipped CLIs.  Model-free: the flags print and
    # exit before anything is loaded, so these run in the default lane.  They
    # assert the flag exists and names its engine; the wording of each list is
    # pinned separately by test-voice-controls.
    foreach (_cli_target parler-cli supertonic-cli cosyvoice-cli)
        if (TARGET ${_cli_target})
            string(REPLACE "-cli" "" _cli_engine ${_cli_target})
            tts_cpp_register_test(test-${_cli_target}-list-emotions
                EXE        ${_cli_target}
                ARGS       "--list-emotions"
                PASS_REGEX "${_cli_engine} emotions: ")
            tts_cpp_register_test(test-${_cli_target}-list-paces
                EXE        ${_cli_target}
                ARGS       "--list-paces"
                PASS_REGEX "${_cli_engine} paces: ")
        endif()
    endforeach()

    # tts-cli routes three families and prints them in routed_engines() order,
    # so matching the last line proves the whole matrix was emitted.
    if (TARGET tts-cli)
        tts_cpp_register_test(test-tts-cli-list-emotions
            EXE        tts-cli
            ARGS       "--list-emotions"
            PASS_REGEX "chatterbox emotions: \\(not supported\\)")
        tts_cpp_register_test(test-tts-cli-list-paces
            EXE        tts-cli
            ARGS       "--list-paces"
            PASS_REGEX "chatterbox paces: \\(not supported\\)")
    endif()

    add_executable(test-parler-description test/parler/test_description.cpp
        src/parler/description.cpp src/voice_controls.cpp)
    target_include_directories(test-parler-description PRIVATE include src)
    tts_cpp_apply_ccache(test-parler-description)
    tts_cpp_register_test(test-parler-description)

    add_executable(test-parler-tokenizer test/parler/test_tokenizer.cpp src/parler/tokenizer.cpp)
    target_link_libraries(test-parler-tokenizer PRIVATE ggml)
    target_include_directories(test-parler-tokenizer PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-parler-tokenizer)
    tts_cpp_register_test(test-parler-tokenizer
        LABEL    "fixture"
        ARGS     "${_tcb_parler_gguf}" "${_tcb_parler_ref}"
        REQUIRES "${_tcb_parler_gguf}" "${_tcb_parler_ref}/tokenizer_corpus.json")

    add_executable(test-parler-t5 test/parler/test_t5.cpp
        src/parler/t5.cpp ${_tcb_parler_core_srcs})
    target_link_libraries(test-parler-t5 PRIVATE ggml)
    target_include_directories(test-parler-t5 PRIVATE ggml/include include src)
    tts_cpp_apply_ccache(test-parler-t5)
    tts_cpp_register_test(test-parler-t5
        LABEL    "fixture"
        ARGS     "${_tcb_parler_gguf}" "${_tcb_parler_ref}"
        REQUIRES "${_tcb_parler_gguf}" "${_tcb_parler_ref}/case0_cross_states.npy")

    add_executable(test-parler-decoder test/parler/test_decoder.cpp
        src/parler/t5.cpp src/parler/decoder.cpp src/parler/delay.cpp
        ${_tcb_parler_core_srcs})
    target_link_libraries(test-parler-decoder PRIVATE ggml)
    target_include_directories(test-parler-decoder PRIVATE ggml/include include src)
    tts_cpp_apply_ccache(test-parler-decoder)
    tts_cpp_register_test(test-parler-decoder
        LABEL    "fixture"
        ARGS     "${_tcb_parler_gguf}" "${_tcb_parler_ref}"
        REQUIRES "${_tcb_parler_gguf}" "${_tcb_parler_ref}/case0_step_logits.npy")

    # Model-free fused-projection parity + zero-copy logits view on a synthetic
    # decoder (no fixtures), so it runs on a bare checkout like the sampler test.
    add_executable(test-parler-decoder-fused test/parler/test_decoder_fused.cpp
        src/parler/decoder.cpp ${_tcb_parler_core_srcs})
    target_link_libraries(test-parler-decoder-fused PRIVATE ggml)
    target_include_directories(test-parler-decoder-fused PRIVATE ggml/include include src)
    tts_cpp_apply_ccache(test-parler-decoder-fused)
    tts_cpp_register_test(test-parler-decoder-fused)

    # Model-free: the DAC snake and conv routing on every build, plus the
    # Accelerate DAC kernels against naive references where they are compiled in.
    add_executable(test-parler-dac-cpu-kernels test/parler/test_dac_cpu_kernels.cpp
        src/parler/dac.cpp src/parler/dac_accel.cpp ${_tcb_parler_core_srcs})
    target_link_libraries(test-parler-dac-cpu-kernels PRIVATE ggml)
    target_include_directories(test-parler-dac-cpu-kernels PRIVATE ggml/include include src)
    tts_cpp_apply_ccache(test-parler-dac-cpu-kernels)
    tts_cpp_register_test(test-parler-dac-cpu-kernels LABEL "unit")

    add_executable(test-parler-dac test/parler/test_dac.cpp
        src/parler/dac.cpp src/parler/dac_accel.cpp ${_tcb_parler_core_srcs})
    target_link_libraries(test-parler-dac PRIVATE ggml)
    target_include_directories(test-parler-dac PRIVATE ggml/include include src)
    tts_cpp_apply_ccache(test-parler-dac)
    tts_cpp_register_test(test-parler-dac
        LABEL    "fixture"
        ARGS     "${_tcb_parler_gguf}" "${_tcb_parler_ref}"
        REQUIRES "${_tcb_parler_gguf}" "${_tcb_parler_ref}/dacrand_wav.npy")

    # Same binary on the GPU transposed-conv path (conv_transpose_1d_matmul), which
    # the CPU run never reaches.  Skipped on GPU-less fleets via `ctest -LE gpu`.
    if (TTS_CPP_PARLER_GPU)
        tts_cpp_register_test(test-parler-dac-gpu
            EXE      test-parler-dac
            LABEL    "gpu"
            ARGS     "${_tcb_parler_gguf}" "${_tcb_parler_ref}"
            REQUIRES "${_tcb_parler_gguf}" "${_tcb_parler_ref}/dacrand_wav.npy")
        set_property(TEST test-parler-dac-gpu APPEND PROPERTY ENVIRONMENT "PARLER_TEST_GPU=1")
        # Per-backend pins of the same harness, so a host carrying several GPU
        # backends covers each by name; TTS_CPP_GPU_BACKEND fails loud when
        # the named backend is absent, so an arm cannot pass on the CPU.
        function(tts_cpp_register_parler_dac_arm suffix)
            tts_cpp_register_test(test-parler-dac-${suffix}
                EXE      test-parler-dac
                LABEL    "gpu"
                ARGS     "${_tcb_parler_gguf}" "${_tcb_parler_ref}"
                REQUIRES "${_tcb_parler_gguf}" "${_tcb_parler_ref}/dacrand_wav.npy")
            set_property(TEST test-parler-dac-${suffix} APPEND PROPERTY ENVIRONMENT
                         "PARLER_TEST_GPU=1" "TTS_CPP_GPU_BACKEND=${suffix}")
        endfunction()
        if (TTS_CPP_GGML_CUDA)
            tts_cpp_register_parler_dac_arm(cuda)
        endif()
        if (TTS_CPP_GGML_VULKAN)
            tts_cpp_register_parler_dac_arm(vulkan)
        endif()
    endif()

    add_executable(test-parler-e2e test/parler/test_e2e.cpp
        src/parler/t5.cpp src/parler/decoder.cpp src/parler/delay.cpp
        src/parler/dac.cpp src/parler/dac_accel.cpp src/parler/sampler.cpp src/parler/tokenizer.cpp
        src/parler/bpe_tokenizer.cpp
        src/parler/text_norm.cpp src/parler/engine.cpp ${_tcb_parler_core_srcs})
    target_link_libraries(test-parler-e2e PRIVATE ggml)
    target_include_directories(test-parler-e2e PRIVATE ggml/include include src)
    tts_cpp_apply_ccache(test-parler-e2e)
    tts_cpp_register_test(test-parler-e2e
        LABEL    "fixture"
        ARGS     "${_tcb_parler_gguf}" "${_tcb_parler_ref}"
        REQUIRES "${_tcb_parler_gguf}" "${_tcb_parler_ref}/case0_greedy_delayed.npy")

    add_executable(test-parler-sched-equivalence test/parler/test_sched_equivalence.cpp)
    target_link_libraries(test-parler-sched-equivalence PRIVATE tts-cpp)
    target_include_directories(test-parler-sched-equivalence PRIVATE include)
    tts_cpp_apply_ccache(test-parler-sched-equivalence)
    tts_cpp_register_test(test-parler-sched-equivalence
        LABEL    "fixture"
        ARGS     "${_tcb_parler_gguf}"
        REQUIRES "${_tcb_parler_gguf}")

    add_executable(test-parler-engine test/parler/test_engine.cpp)
    target_link_libraries(test-parler-engine PRIVATE tts-cpp)
    target_include_directories(test-parler-engine PRIVATE include)
    tts_cpp_apply_ccache(test-parler-engine)
    tts_cpp_register_test(test-parler-engine
        LABEL    "fixture"
        ARGS     "${_tcb_parler_gguf}" "${_tcb_super2_gguf}"
        REQUIRES "${_tcb_parler_gguf}")

    add_executable(test-parler-engine-stream test/parler/test_engine_stream.cpp)
    target_link_libraries(test-parler-engine-stream PRIVATE tts-cpp)
    target_include_directories(test-parler-engine-stream PRIVATE include src)
    tts_cpp_apply_ccache(test-parler-engine-stream)
    tts_cpp_register_test(test-parler-engine-stream
        LABEL    "fixture"
        ARGS     "${_tcb_parler_gguf}"
        REQUIRES "${_tcb_parler_gguf}")

    # ---- Indic Parler-TTS (same binaries, indic fixtures; separate BPE
    #      prompt tokenizer gets its own parity test) ----
    add_executable(test-parler-bpe-tokenizer test/parler/test_bpe_tokenizer.cpp
        src/parler/bpe_tokenizer.cpp)
    target_link_libraries(test-parler-bpe-tokenizer PRIVATE ggml)
    target_include_directories(test-parler-bpe-tokenizer PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-parler-bpe-tokenizer)
    tts_cpp_register_test(test-parler-bpe-tokenizer
        LABEL    "fixture"
        ARGS     "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}"
        REQUIRES "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}/prompt_tokenizer_corpus.json")

    tts_cpp_register_test(test-parler-tokenizer-indic
        EXE      test-parler-tokenizer
        LABEL    "fixture"
        ARGS     "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}"
        REQUIRES "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}/tokenizer_corpus.json")

    tts_cpp_register_test(test-parler-t5-indic
        EXE      test-parler-t5
        LABEL    "fixture"
        ARGS     "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}"
        REQUIRES "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}/case0_cross_states.npy")

    tts_cpp_register_test(test-parler-decoder-indic
        EXE      test-parler-decoder
        LABEL    "fixture"
        ARGS     "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}"
        REQUIRES "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}/case0_step_logits.npy")

    tts_cpp_register_test(test-parler-dac-indic
        EXE      test-parler-dac
        LABEL    "fixture"
        ARGS     "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}"
        REQUIRES "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}/dacrand_wav.npy")

    if (TTS_CPP_PARLER_GPU)
        tts_cpp_register_test(test-parler-dac-indic-gpu
            EXE      test-parler-dac
            LABEL    "gpu"
            ARGS     "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}"
            REQUIRES "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}/dacrand_wav.npy")
        set_property(TEST test-parler-dac-indic-gpu APPEND PROPERTY ENVIRONMENT "PARLER_TEST_GPU=1")
    endif()

    tts_cpp_register_test(test-parler-e2e-indic
        EXE      test-parler-e2e
        LABEL    "fixture"
        ARGS     "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}"
        REQUIRES "${_tcb_parler_indic_gguf}" "${_tcb_parler_indic_ref}/case0_greedy_delayed.npy")

    tts_cpp_register_test(test-parler-sched-equivalence-indic
        EXE      test-parler-sched-equivalence
        LABEL    "fixture"
        ARGS     "${_tcb_parler_indic_gguf}"
        REQUIRES "${_tcb_parler_indic_gguf}")

    add_executable(test-s3gen test/test_s3gen.cpp src/backend_selection.cpp)
    target_link_libraries(test-s3gen PRIVATE ggml)
    target_include_directories(test-s3gen PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-s3gen)
    tts_cpp_register_test(test-s3gen
        LABEL    "fixture"
        ARGS     "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref}" "ALL"
        REQUIRES "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref_sentinel}")

    add_executable(test-s3gen-backend-agreement test/test_s3gen_backend_agreement.cpp)
    target_link_libraries(test-s3gen-backend-agreement PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-s3gen-backend-agreement PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-s3gen-backend-agreement)

    add_executable(test-chatterbox-backend-parity test/test_chatterbox_backend_parity.cpp)
    target_link_libraries(test-chatterbox-backend-parity PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-chatterbox-backend-parity PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-chatterbox-backend-parity)

    # Per-backend S3Gen parity arms. Both run the same binary and selection
    # picks on its own preference order, so each arm pins the backend it
    # covers through TTS_CPP_GPU_BACKEND; the harness fails rather than
    # falling back to CPU when that backend is unavailable. Registered only
    # for backends whose S3Gen/HiFT graphs have been checked against the
    # reference dump -- Metal is absent because it has not been.
    function(tts_cpp_register_s3gen_gpu_arm suffix)
        tts_cpp_register_test(test-s3gen-${suffix}
            EXE      test-s3gen
            LABEL    "gpu"
            ARGS     "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref}" "ALL" "--n-gpu-layers" "99"
            REQUIRES "${_tcb_s3gen_gguf}" "${_tcb_s3gen_ref_sentinel}")
        set_property(TEST test-s3gen-${suffix} APPEND
                     PROPERTY ENVIRONMENT "TTS_CPP_GPU_BACKEND=${suffix}")

        # Backs the CPU-vs-GPU claim for the T3 half, which has no usable
        # reference dump: same seed, greedy sampling, token and sample counts
        # compared against the CPU on the same build.
        tts_cpp_register_test(test-chatterbox-parity-mtl-${suffix}
            EXE      test-chatterbox-backend-parity
            LABEL    "gpu"
            ARGS     "${_tcb_t3_mtl_gguf}" "${_tcb_s3gen_mtl_gguf}" "en"
            REQUIRES "${_tcb_t3_mtl_gguf}" "${_tcb_s3gen_mtl_gguf}")
        set_property(TEST test-chatterbox-parity-mtl-${suffix} APPEND
                     PROPERTY ENVIRONMENT "TTS_CPP_GPU_BACKEND=${suffix}")

        # Same-GGUF CPU-vs-GPU waveform agreement. Quantization cancels on
        # identical weights, so the q4_0 arm holds to the same bar as f16 and
        # runs on the tier the model registry serves; neither needs a
        # reference dump. The long case crosses the 65535 CUDA grid limit,
        # standing in for the launch abort long-form input used to hit.
        tts_cpp_register_test(test-s3gen-agreement-${suffix}
            EXE      test-s3gen-backend-agreement
            LABEL    "gpu"
            ARGS     "${_tcb_s3gen_gguf}"
            REQUIRES "${_tcb_s3gen_gguf}")
        set_property(TEST test-s3gen-agreement-${suffix} APPEND
                     PROPERTY ENVIRONMENT "TTS_CPP_GPU_BACKEND=${suffix}")
        tts_cpp_register_test(test-s3gen-agreement-${suffix}-q4_0
            EXE      test-s3gen-backend-agreement
            LABEL    "gpu"
            ARGS     "${_tcb_s3gen_q4_gguf}" "--mel-tolerance" "5e-2" "--max-rms-ratio" "0.5"
            REQUIRES "${_tcb_s3gen_q4_gguf}")
        set_property(TEST test-s3gen-agreement-${suffix}-q4_0 APPEND
                     PROPERTY ENVIRONMENT "TTS_CPP_GPU_BACKEND=${suffix}")
    endfunction()

    if (TTS_CPP_GGML_CUDA)
        tts_cpp_register_s3gen_gpu_arm(cuda)
    endif()
    if (TTS_CPP_GGML_VULKAN)
        tts_cpp_register_s3gen_gpu_arm(vulkan)
    endif()

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
        src/campplus.cpp
        src/backend_selection.cpp)
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

    # Attention-free end-of-speech stop controller.  Pure host
    # logic with no model / ggml dependency, so it links only its own source
    # and always runs (no fixture REQUIRES).
    add_executable(test-t3-stop-controller
        test/test_t3_stop_controller.cpp
        src/t3_stop_controller.cpp)
    target_include_directories(test-t3-stop-controller PRIVATE src)
    tts_cpp_apply_ccache(test-t3-stop-controller)
    tts_cpp_register_test(test-t3-stop-controller LABEL "cpu")

    # Phase 2: alignment analyzer state machine.  Pure host logic,
    # no model / ggml dependency, so it always runs.
    add_executable(test-t3-alignment-analyzer
        test/test_t3_alignment_analyzer.cpp
        src/t3_alignment_analyzer.cpp)
    target_include_directories(test-t3-alignment-analyzer PRIVATE src)
    tts_cpp_apply_ccache(test-t3-alignment-analyzer)
    tts_cpp_register_test(test-t3-alignment-analyzer LABEL "cpu")

    # MTL sampler EOS-suppression.  Pure logit math but links the
    # library (sample_next_token_mtl lives in the t3_mtl TU); no model required.
    add_executable(test-mtl-sampler test/test_mtl_sampler.cpp)
    target_link_libraries(test-mtl-sampler PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-mtl-sampler PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-mtl-sampler)
    tts_cpp_register_test(test-mtl-sampler LABEL "cpu")

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

    # T3 dual-path dispatch equivalence: the forced scheduler path
    # (TTS_CPP_FORCE_SCHED) must be bit-identical to the direct path,
    # and a sched run must not poison the step-graph cache.  Two
    # registrations share one binary: with and without the opt-in
    # CHATTERBOX_T3_STEP_CACHE (latched per process on first use).
    add_executable(test-t3-sched-equivalence test/test_t3_sched_equivalence.cpp)
    target_link_libraries(test-t3-sched-equivalence PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-t3-sched-equivalence PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-t3-sched-equivalence)
    tts_cpp_register_test(test-t3-sched-equivalence
        LABEL    "fixture"
        ARGS     "${_tcb_t3_mtl_gguf}"
        REQUIRES "${_tcb_t3_mtl_gguf}")
    tts_cpp_register_test(test-t3-sched-equivalence-step-cache
        EXE      test-t3-sched-equivalence
        LABEL    "fixture"
        ARGS     "${_tcb_t3_mtl_gguf}" "--step-cache"
        REQUIRES "${_tcb_t3_mtl_gguf}")

    # S3Gen dual-path dispatch equivalence: forced-sched full synth
    # (TTS_CPP_FORCE_SCHED) must be bit-identical to the direct path,
    # A/B/A' with a model-cache teardown between phases.  Built-in voice
    # + synthetic tokens, so the only fixture is the S3Gen GGUF.
    add_executable(test-s3gen-sched-equivalence test/test_s3gen_sched_equivalence.cpp)
    target_link_libraries(test-s3gen-sched-equivalence PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-s3gen-sched-equivalence PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-s3gen-sched-equivalence)
    tts_cpp_register_test(test-s3gen-sched-equivalence
        LABEL    "fixture"
        ARGS     "${_tcb_s3gen_gguf}"
        REQUIRES "${_tcb_s3gen_gguf}")

    # Supertonic dual-path dispatch equivalence: forced-sched synth
    # (TTS_CPP_FORCE_SCHED) must be bit-identical to direct — every
    # dual-path graph rebuilds per sched pass (sched graphs are single-use
    # for allocation).  Two registrations share one binary: CPU (degenerate
    # [primary] sched) and GPU (real [metal, cpu] multi-backend sched, the
    # configuration that used to SIGSEGV on reuse).
    add_executable(test-supertonic-sched-equivalence test/test_supertonic_sched_equivalence.cpp)
    target_link_libraries(test-supertonic-sched-equivalence PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-supertonic-sched-equivalence PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-sched-equivalence)
    tts_cpp_register_test(test-supertonic-sched-equivalence
        LABEL    "fixture"
        ARGS     "${_tcb_super2_gguf}"
        REQUIRES "${_tcb_super2_gguf}")
    # gpu, not fixture: argv[2]=99 moves the graph onto the GPU, so this arm
    # must stay out of the CPU-only lane even once the fixture is staged.
    tts_cpp_register_test(test-supertonic-sched-equivalence-gpu
        EXE      test-supertonic-sched-equivalence
        LABEL    "gpu"
        ARGS     "${_tcb_super2_gguf}" "99"
        REQUIRES "${_tcb_super2_gguf}")
    # Per-backend pins of the same bit-exactness check, so a host carrying
    # several GPU backends covers each by name instead of whichever selection
    # prefers. TTS_CPP_GPU_BACKEND fails loud when the named backend is
    # absent, so an arm cannot pass on the CPU.
    function(tts_cpp_register_supertonic_sched_arm suffix)
        tts_cpp_register_test(test-supertonic-sched-equivalence-${suffix}
            EXE      test-supertonic-sched-equivalence
            LABEL    "gpu"
            ARGS     "${_tcb_super2_gguf}" "99"
            REQUIRES "${_tcb_super2_gguf}")
        set_property(TEST test-supertonic-sched-equivalence-${suffix} APPEND
                     PROPERTY ENVIRONMENT "TTS_CPP_GPU_BACKEND=${suffix}")
    endfunction()
    if (TTS_CPP_GGML_CUDA)
        tts_cpp_register_supertonic_sched_arm(cuda)
    endif()
    if (TTS_CPP_GGML_VULKAN)
        tts_cpp_register_supertonic_sched_arm(vulkan)
    endif()

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
        "hi|नमस्ते, यह एक बहुभाषी वाक् संश्लेषण परीक्षण है।|tier2"
    )
    if (EXISTS "${TTS_CPP_TEST_CANGJIE_TSV}")
        list(APPEND _mtl_phrases "zh|你好，这是一个多语言语音合成测试。|tier2")
    else()
        message(STATUS "Multilingual zh synth test: SKIPPED")
        message(STATUS "  reason: Cangjie TSV missing at ${TTS_CPP_TEST_CANGJIE_TSV}")
    endif()
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
        set(_mtl_synth_extra_args)
        if (_lang STREQUAL "zh")
            list(APPEND _mtl_synth_extra_args --cangjie-tsv "${TTS_CPP_TEST_CANGJIE_TSV}")
        endif()
        add_test(
            NAME mtl-synth-${_lang}
            COMMAND $<TARGET_FILE:test-multilingual-synth>
                --lang  "${_lang}"
                --text  "${_text}"
                --out   "${_mtl_out_dir}/${_lang}.wav"
                ${_mtl_synth_extra_args}
        )
        set_tests_properties(mtl-synth-${_lang} PROPERTIES
            LABELS "multilingual;mtl-${_tier}"
            SKIP_RETURN_CODE 77
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

    # q8_0 KV-cache regression for the MTL variant. A quantized KV
    # cache used to SIGABRT on Metal ("unsupported op 'CONT'") via the alignment
    # probe; the dequantizing-cast fix recovers q8 KV on the GPU.  These run the
    # full synth with --kv-cache-type q8_0 and validate the WAV is real audio
    # (RMS/peak/clipping/duration) — an end-to-end regression the op-level
    # test_metal_ops sentinel can't cover.  A diverse script subset
    # (Latin/Arabic/Cyrillic/Devanagari) doubles as cross-language coverage.
    # IMPORTANT: these only exercise the *original crash path* on a Metal/GPU
    # fleet — on a CPU-only runner q8 KV never crashed, so q8-vs-f32 there only
    # checks the dequant-cast is numerically sane, not the SIGABRT.  Labelled
    # `mtl-q8;gpu` so CI selects them on a Metal runner (`ctest -L mtl-q8`); the
    # alignment-drift regression (CER) is covered by test-eos-roundtrip-q8-kv
    # below.  Missing GGUFs -> SKIP (77), not fail, like the f32 tests above.
    set(_mtl_q8_phrases
        "en|Hello, this is a multilingual text-to-speech test."
        "ar|مرحبًا، هذا اختبار متعدد اللغات لتحويل النص إلى كلام."
        "ru|Привет, это многоязычный тест синтеза речи."
        "hi|नमस्ते, यह एक बहुभाषी वाक् संश्लेषण परीक्षण है।"
    )
    foreach(_entry IN LISTS _mtl_q8_phrases)
        string(REPLACE "|" ";" _parts "${_entry}")
        list(GET _parts 0 _lang)
        list(GET _parts 1 _text)
        add_test(
            NAME mtl-synth-q8-${_lang}
            COMMAND $<TARGET_FILE:test-multilingual-synth>
                --lang           "${_lang}"
                --text           "${_text}"
                --kv-cache-type  q8_0
                --out            "${_mtl_out_dir}/${_lang}-q8.wav"
        )
        set_tests_properties(mtl-synth-q8-${_lang} PROPERTIES
            LABELS "multilingual;mtl-q8;gpu"
            SKIP_RETURN_CODE 77
            TIMEOUT 180
        )
    endforeach()

    # End-to-end EOS round-trip regression.  Drives tts-cli to
    # synthesize a set of English phrases, transcribes with whisper-cli, and
    # asserts the transcription is close to the input (CER guard -> catches
    # clipping) and not much longer (ramble guard -> catches the trailing-token
    # bug re-appearing).  Pure orchestration; auto-disabled unless the MTL
    # GGUFs + whisper-cli + a Whisper model + tts-cli are all present.
    add_executable(test-eos-roundtrip test/test_eos_roundtrip.cpp)
    target_include_directories(test-eos-roundtrip PRIVATE src)
    tts_cpp_apply_ccache(test-eos-roundtrip)
    if (_mtl_asr_enabled AND TARGET tts-cli)
        tts_cpp_register_test(test-eos-roundtrip
            LABEL    "fixture;asr"
            ARGS     --tts-cli       "$<TARGET_FILE:tts-cli>"
                     --t3            "${_tcb_t3_mtl_gguf}"
                     --s3gen         "${_tcb_s3gen_mtl_gguf}"
                     --whisper-cli   "${WHISPER_CLI}"
                     --whisper-model "${WHISPER_MODEL}"
                     --lang          "en"
                     --tmp           "${_mtl_out_dir}"
            REQUIRES "${_tcb_t3_mtl_gguf}" "${_tcb_s3gen_mtl_gguf}")

        # Same round-trip on the quantized (Q4_0) T3 the mobile
        # app ships, to confirm the aligned heads still track under
        # quantization.  Uses the f16 S3Gen (the S3Gen Q4_0 path is a separate
        # converter limitation).  Auto-disabled until the Q4_0 T3 GGUF exists.
        tts_cpp_register_test(test-eos-roundtrip-q4_0
            EXE      test-eos-roundtrip
            LABEL    "fixture;asr"
            ARGS     --tts-cli       "$<TARGET_FILE:tts-cli>"
                     --t3            "${_tcb_t3_mtl_q4_gguf}"
                     --s3gen         "${_tcb_s3gen_mtl_gguf}"
                     --whisper-cli   "${WHISPER_CLI}"
                     --whisper-model "${WHISPER_MODEL}"
                     --lang          "en"
                     --tmp           "${_mtl_out_dir}"
            REQUIRES "${_tcb_t3_mtl_q4_gguf}" "${_tcb_s3gen_mtl_gguf}")

        # same EOS round-trip but with a q8_0 KV CACHE (not a quantized
        # T3).  The align-probe fix feeds the alignment mul_mat *dequantized* q8
        # keys (ggml_cast), which could shift the argmax/softmax over text columns
        # and thus EOS timing — something the WAV-sanity mtl-synth-q8-* tests can't
        # see.  This drives the CER/ramble guards under q8 KV to catch alignment
        # drift (clipping / early-cutoff / rambling) vs the f32 baseline above.
        tts_cpp_register_test(test-eos-roundtrip-q8-kv
            EXE      test-eos-roundtrip
            LABEL    "fixture;asr"
            ARGS     --tts-cli       "$<TARGET_FILE:tts-cli>"
                     --t3            "${_tcb_t3_mtl_gguf}"
                     --s3gen         "${_tcb_s3gen_mtl_gguf}"
                     --whisper-cli   "${WHISPER_CLI}"
                     --whisper-model "${WHISPER_MODEL}"
                     --lang          "en"
                     --kv-cache-type q8_0
                     --tmp           "${_mtl_out_dir}"
            REQUIRES "${_tcb_t3_mtl_gguf}" "${_tcb_s3gen_mtl_gguf}")
    else()
        message(STATUS "tts-cpp: test-eos-roundtrip built but not registered (needs whisper-cli + model + tts-cli)")
    endif()

    # chunk-streaming loudness regression. Drives the Engine API
    # directly (the path the qvac tts-ggml addon uses): batch vs streaming with
    # a low stream_cfm_steps, asserting the streaming output matches batch
    # loudness and does not clip.  PRE-FIX this FAILS (streaming ~full scale,
    # RMS 4-9x batch).  Uses the S3Gen built-in voice (no voice dir), so it
    # only needs the MTL GGUFs; auto-disabled otherwise.
    add_executable(test-streaming-loudness test/test_streaming_loudness.cpp)
    target_link_libraries(test-streaming-loudness PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-streaming-loudness PRIVATE include ggml/include src)
    tts_cpp_apply_ccache(test-streaming-loudness)
    tts_cpp_register_test(test-streaming-loudness
        LABEL    "fixture"
        ARGS     --t3 "${_tcb_t3_mtl_gguf}" --s3gen "${_tcb_s3gen_mtl_gguf}" --cfm-steps 2
        REQUIRES "${_tcb_t3_mtl_gguf}" "${_tcb_s3gen_mtl_gguf}")

    add_executable(test-tts-streaming
        test/test_streaming.cpp
        src/chatterbox_tts.cpp
        src/sched_dispatch.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-tts-streaming PRIVATE ggml tts-cpp-backend-defs)
    target_include_directories(test-tts-streaming PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-tts-streaming)
    tts_cpp_register_test(test-tts-streaming
        LABEL    "fixture"
        ARGS     "${_tcb_s3gen_gguf}" "${_tcb_streaming_ref}"
        REQUIRES "${_tcb_s3gen_gguf}" "${_tcb_streaming_ref}")

    # Pure sentence-splitter unit test (no model/fixture) -- always runs.
    # Pins split_text_for_tts after it was lifted into the shared
    # text_preprocess unit (used by both the CLI and the Engine).
    add_executable(test-text-split test/test_text_split.cpp)
    target_link_libraries(test-text-split PRIVATE tts-cpp)
    target_include_directories(test-text-split PRIVATE src)
    tts_cpp_apply_ccache(test-text-split)
    tts_cpp_register_test(test-text-split LABEL "cpu")

    # Chatterbox Turbo prompt tokenizer on a hand-built vocab: merge chains,
    # byte-level encoding, paralinguistic added-token spans, punc_norm.
    add_executable(test-gpt2-bpe test/test_gpt2_bpe.cpp src/gpt2_bpe.cpp)
    target_include_directories(test-gpt2-bpe PRIVATE src)
    tts_cpp_apply_ccache(test-gpt2-bpe)
    tts_cpp_register_test(test-gpt2-bpe LABEL "cpu")

    # Supertonic streaming chunker: pure text -> chunks policy (boundary
    # priority, tolerance windows, min-chunk floor, tail merge).  No model,
    # no ggml.
    add_executable(test-supertonic-chunker test/test_supertonic_chunker.cpp
        src/supertonic_chunker.cpp)
    target_include_directories(test-supertonic-chunker PRIVATE src)
    tts_cpp_apply_ccache(test-supertonic-chunker)
    tts_cpp_register_test(test-supertonic-chunker LABEL "unit")

    # LavaSR DSP primitives: resampler, STFT/ISTFT, Slaney mel,
    # FastLR crossover merge.  Pure host math (no model / no ggml), compiled
    # straight from the lavasr/dsp sources so it always runs in CI.
    # Fail-closed GGUF loading for both LavaSR stages: synthesizes tiny
    # metadata-only GGUFs and asserts each loader rejects a missing, empty,
    # truncated, unmarked, cross-architecture, or tensorless file.
    add_executable(test-lavasr-gguf-load
        test/test_lavasr_gguf_load.cpp
        src/lavasr/denoiser_gguf.cpp
        src/lavasr/enhancer_gguf.cpp
        src/lavasr/denoiser_core.cpp
        src/lavasr/enhancer_core.cpp)
    target_link_libraries(test-lavasr-gguf-load PRIVATE ggml)
    target_include_directories(test-lavasr-gguf-load PRIVATE ggml/include src test)
    tts_cpp_apply_ccache(test-lavasr-gguf-load)
    tts_cpp_register_test(test-lavasr-gguf-load LABEL "unit")

    add_executable(test-lavasr-dsp
        test/test_lavasr_dsp.cpp
        src/lavasr/dsp/resampler.cpp
        src/lavasr/dsp/stft_processor.cpp
        src/lavasr/dsp/mel_filterbank.cpp
        src/lavasr/dsp/fastlr_merge.cpp
        src/lavasr/dsp/overlap_add.cpp)
    target_include_directories(test-lavasr-dsp PRIVATE src)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(test-lavasr-dsp PRIVATE OpenMP::OpenMP_CXX)
    endif()
    tts_cpp_apply_ccache(test-lavasr-dsp)
    tts_cpp_register_test(test-lavasr-dsp LABEL "unit")

    # LavaSR enhancer scalar-core parity test: numerical parity of
    # the neural core (mel -> backbone -> spec head -> real/imag) against an
    # onnxruntime golden, plus a finite/length smoke test of the full enhance()
    # pipeline. Pure host math (no ggml); fixtures from
    # scripts/dump-lavasr-enhancer-fixtures.py. The end-to-end DSP comparison
    # (resampler/mel/ISTFT/FastLR vs the numpy reference) lives in
    # test-lavasr-enhancer-gguf below. DISABLED until the fixtures exist
    # (REQUIRES the golden real.npy).
    add_executable(test-lavasr-enhancer-core
        test/test_lavasr_enhancer_core.cpp
        src/lavasr/enhancer_core.cpp
        src/lavasr/enhancer.cpp
        src/lavasr/dsp/resampler.cpp
        src/lavasr/dsp/stft_processor.cpp
        src/lavasr/dsp/mel_filterbank.cpp
        src/lavasr/dsp/fastlr_merge.cpp)
    target_include_directories(test-lavasr-enhancer-core PRIVATE src)
    tts_cpp_apply_ccache(test-lavasr-enhancer-core)
    tts_cpp_register_test(test-lavasr-enhancer-core
        LABEL    "fixture"
        ARGS     "${TTS_CPP_TEST_REF_DIR}/lavasr-enhancer"
        REQUIRES "${TTS_CPP_TEST_REF_DIR}/lavasr-enhancer/real.npy")

    # LavaSR enhancer GGUF round-trip: load the converted GGUF via
    # load_enhancer_gguf() + the public tts_cpp::lavasr::Enhancer API, compare
    # the neural core to the onnxruntime golden, AND compare the full
    # enhance(pcm_in) output (resampler + mel + ISTFT + FastLR + neural) against
    # the numpy end-to-end golden (enhanced_48k.npy). Links ggml for GGUF I/O
    # (mirrors test-gguf-stream). DISABLED until the GGUF + fixtures exist
    # (convert-lavasr-enhancer-to-gguf.py + dump-lavasr-enhancer-fixtures.py).
    add_executable(test-lavasr-enhancer-gguf test/test_lavasr_enhancer_gguf.cpp)
    target_link_libraries(test-lavasr-enhancer-gguf PRIVATE tts-cpp ggml)
    target_include_directories(test-lavasr-enhancer-gguf PRIVATE include src ggml/include)
    tts_cpp_apply_ccache(test-lavasr-enhancer-gguf)
    tts_cpp_register_test(test-lavasr-enhancer-gguf
        LABEL    "fixture"
        ARGS     "${TTS_CPP_TEST_MODEL_DIR}/lavasr-enhancer.gguf" "${TTS_CPP_TEST_REF_DIR}/lavasr-enhancer"
        REQUIRES "${TTS_CPP_TEST_MODEL_DIR}/lavasr-enhancer.gguf" "${TTS_CPP_TEST_REF_DIR}/lavasr-enhancer/real.npy")

    # LavaSR enhancer GGML/GPU parity: the ggml compute-graph forward
    # (enhancer_ggml_spec_forward) vs the scalar CPU core (validated against the
    # onnx golden by test-lavasr-enhancer-core, so scalar==golden and this makes
    # ggml==scalar transitively validate the GPU port).  Self-contained (random
    # weights), so it ALWAYS runs in CI with no model download — exercising the
    # ggml CPU backend plus any registered GPU backend (Vulkan on
    # Windows/Linux, Metal, CUDA, OpenCL).  When the real enhancer fixtures are
    # present it additionally checks the production weights against the onnx
    # golden.  Links ggml for the backend registry + ggml_backend_free.
    add_executable(test-lavasr-enhancer-ggml test/test_lavasr_enhancer_ggml.cpp)
    target_link_libraries(test-lavasr-enhancer-ggml PRIVATE tts-cpp ggml)
    target_include_directories(test-lavasr-enhancer-ggml PRIVATE include src ggml/include)
    tts_cpp_apply_ccache(test-lavasr-enhancer-ggml)
    tts_cpp_register_test(test-lavasr-enhancer-ggml
        LABEL "unit"
        ARGS  "${TTS_CPP_TEST_REF_DIR}/lavasr-enhancer")

    # LavaSR denoiser ggml-graph parity: denoiser_ggml.cpp graph vs the scalar
    # denoiser_core.cpp.  Self-contained host math (no GGUF fixture), always runs in CI.
    add_executable(test-lavasr-denoiser-ggml test/test_lavasr_denoiser_ggml.cpp)
    target_link_libraries(test-lavasr-denoiser-ggml PRIVATE tts-cpp ggml)
    target_include_directories(test-lavasr-denoiser-ggml PRIVATE include src ggml/include)
    tts_cpp_apply_ccache(test-lavasr-denoiser-ggml)
    tts_cpp_register_test(test-lavasr-denoiser-ggml LABEL "unit")
    # Same binary with the real GGUF: adds the production-weights e2e + batch parity
    # where the model fixture is present.
    tts_cpp_register_test(test-lavasr-denoiser-ggml-e2e
        EXE      test-lavasr-denoiser-ggml
        LABEL    "fixture"
        ARGS     "${TTS_CPP_TEST_MODEL_DIR}/lavasr-denoiser.gguf"
        REQUIRES "${TTS_CPP_TEST_MODEL_DIR}/lavasr-denoiser.gguf")

    # LavaSR denoiser GGUF round-trip (follow-up): load the converted
    # GGUF via load_denoiser_gguf() + the public tts_cpp::lavasr::Denoiser API,
    # compare the UL-UNAS neural core (denoiser_net_forward) against the
    # onnxruntime golden (spec_in.npy -> spec_out.npy), and run a full denoise()
    # pipeline smoke (rate-preserving length + finite output). Links ggml for
    # GGUF I/O. DISABLED until the GGUF + fixtures exist
    # (convert-lavasr-denoiser-to-gguf.py + dump-lavasr-denoiser-fixtures.py).
    add_executable(test-lavasr-denoiser-gguf test/test_lavasr_denoiser_gguf.cpp)
    target_link_libraries(test-lavasr-denoiser-gguf PRIVATE tts-cpp ggml)
    target_include_directories(test-lavasr-denoiser-gguf PRIVATE include src ggml/include)
    tts_cpp_apply_ccache(test-lavasr-denoiser-gguf)
    tts_cpp_register_test(test-lavasr-denoiser-gguf
        LABEL    "fixture"
        ARGS     "${TTS_CPP_TEST_MODEL_DIR}/lavasr-denoiser.gguf" "${TTS_CPP_TEST_REF_DIR}/lavasr-denoiser"
        REQUIRES "${TTS_CPP_TEST_MODEL_DIR}/lavasr-denoiser.gguf"
                 "${TTS_CPP_TEST_REF_DIR}/lavasr-denoiser/spec_in.npy"
                 "${TTS_CPP_TEST_REF_DIR}/lavasr-denoiser/pcm_in.npy")

    # voice-clone test harness.
    #
    # Two header-only, model-free harnesses that every later cloning task builds
    # on: a finite-difference gradient checker (validates future Supertonic
    # analytic backward passes) and a speaker-similarity / WER metrics module.
    # Both register as "unit" with NO REQUIRES so they ALWAYS run on a fresh
    # checkout (the no-skip test policy) — they need no model and the metrics
    # fixtures are committed in-tree under test/fixtures/voiceclone/v1/.
    add_executable(test-voiceclone-gradcheck
        test/test_voiceclone_gradcheck.cpp
        src/voiceclone_gradcheck.cpp)
    target_include_directories(test-voiceclone-gradcheck PRIVATE src)
    tts_cpp_apply_ccache(test-voiceclone-gradcheck)
    tts_cpp_register_test(test-voiceclone-gradcheck LABEL "unit")

    add_executable(test-voiceclone-metrics
        test/test_voiceclone_metrics.cpp
        src/voiceclone_metrics.cpp)
    target_include_directories(test-voiceclone-metrics PRIVATE src)
    tts_cpp_apply_ccache(test-voiceclone-metrics)
    # The committed fixture dir is passed as argv[1] so the reproduction layer
    # always runs; the closed-form self-tests run regardless of the argument.
    tts_cpp_register_test(test-voiceclone-metrics
        LABEL "unit"
        ARGS  "${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/voiceclone/v1")

    # Ticket 4 — analytic backward of the style-dependent text-encoder tail
    # (speech-prompted attention + final layer norm).  Model-free: every
    # analytic gradient is gradchecked against finite differences, so it ALWAYS
    # runs on a fresh checkout (no-skip policy, no model/fixtures needed).
    add_executable(test-supertonic-text-encoder-backward
        test/test_supertonic_text_encoder_backward.cpp
        src/supertonic_text_encoder_backward.cpp
        src/voiceclone_gradcheck.cpp)
    target_include_directories(test-supertonic-text-encoder-backward PRIVATE src)
    tts_cpp_apply_ccache(test-supertonic-text-encoder-backward)
    tts_cpp_register_test(test-supertonic-text-encoder-backward LABEL "unit")

    # analytic backward of the vector estimator / CFM step
    # (convnext, depthwise conv, layer norm, gelu, rope + style cross-attention,
    # full field and per-step d_in/d_style).  Model-free: every analytic
    # gradient is gradchecked against finite differences via the Task 2 harness,
    # so it ALWAYS runs on a fresh checkout (no-skip policy, no model/fixtures).
    add_executable(test-supertonic-vector-estimator-backward
        test/test_supertonic_vector_estimator_backward.cpp
        src/supertonic_vector_estimator_backward.cpp
        src/voiceclone_gradcheck.cpp)
    target_include_directories(test-supertonic-vector-estimator-backward PRIVATE src)
    tts_cpp_apply_ccache(test-supertonic-vector-estimator-backward)
    tts_cpp_register_test(test-supertonic-vector-estimator-backward LABEL "unit")

    # QVAC-20983 — analytic backward of the vocoder (latent unpack, denorm,
    # causal conv1d, causal depthwise, channel layer norm, gelu, per-channel-gamma
    # convnext, affine batch norm, leaky-relu head, full chain d(loss)/d(latent)).
    # The "transposed convolution" upsampling is a fixed reshape+permute (the
    # latent unpack), so its backward is a pure permutation. Model-free: every
    # analytic gradient is gradchecked against finite differences via the Task 2
    # harness, so it ALWAYS runs on a fresh checkout (no-skip policy).
    add_executable(test-supertonic-vocoder-backward
        test/test_supertonic_vocoder_backward.cpp
        src/supertonic_vocoder_backward.cpp
        src/supertonic_vector_estimator_backward.cpp
        src/voiceclone_gradcheck.cpp)
    target_include_directories(test-supertonic-vocoder-backward PRIVATE src)
    tts_cpp_apply_ccache(test-supertonic-vocoder-backward)
    tts_cpp_register_test(test-supertonic-vocoder-backward LABEL "unit")

    # QVAC-20983 — forward-parity guard for the vocoder backward.  The gradcheck
    # above is self-referential (analytic backward vs the in-file reference
    # forward); this builds a synthetic CPU-backed `supertonic_model` from
    # deterministic weights and asserts the reference forward
    # (`VocoderBackward::forward`) matches the PRODUCTION
    # `supertonic_vocoder_forward_cpu` on the same buffers.  Catches any drift
    # between the two forwards (per-channel gamma layout, dilation schedule,
    # weight index order) the gradcheck cannot see.  Model-free (weights
    # synthesized in-memory) so it always runs in the `unit` tier.
    add_executable(test-supertonic-vocoder-backward-parity
        test/test_supertonic_vocoder_backward_parity.cpp
        src/supertonic_vocoder_backward.cpp
        src/supertonic_vector_estimator_backward.cpp)
    target_link_libraries(test-supertonic-vocoder-backward-parity PRIVATE tts-cpp ggml)
    target_include_directories(test-supertonic-vocoder-backward-parity PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-vocoder-backward-parity)
    tts_cpp_register_test(test-supertonic-vocoder-backward-parity LABEL "unit")

    # analytic backward of the CAMPPlus speaker encoder (FCM Conv2d
    # head + residual blocks, TDNN, CAM dense-TDNN blocks with context-attention
    # gating and dense concat, statistics pooling, dense head). Model-free: every
    # analytic input-gradient is gradchecked against finite differences via the
    # Task 2 harness, so it ALWAYS runs on a fresh checkout (no-skip policy, no
    # model/fixtures needed).
    add_executable(test-campplus-backward
        test/test_campplus_backward.cpp
        src/campplus_backward.cpp
        src/voiceclone_gradcheck.cpp)
    target_include_directories(test-campplus-backward PRIVATE src)
    tts_cpp_apply_ccache(test-campplus-backward)
    tts_cpp_register_test(test-campplus-backward LABEL "unit")

    # Forward-parity: the analytic double forward must match the production scalar
    # CAMPPlus forward (campplus_embed_cpu) on synthetic weights, anchoring the
    # gradcheck to the real model. Links campplus.cpp -> ggml.
    add_executable(test-campplus-backward-parity
        test/test_campplus_backward_parity.cpp
        src/campplus_backward.cpp
        src/campplus.cpp
        src/backend_selection.cpp)
    target_link_libraries(test-campplus-backward-parity PRIVATE ggml)
    target_include_directories(test-campplus-backward-parity PRIVATE ggml/include src)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(test-campplus-backward-parity PRIVATE OpenMP::OpenMP_CXX)
    endif()
    tts_cpp_apply_ccache(test-campplus-backward-parity)
    tts_cpp_register_test(test-campplus-backward-parity LABEL "unit")

    # Engine-level streaming-callback contract test for the per-sentence
    # segmentation path (Fix #2): monotonic global chunk_index, single final
    # is_last, result.pcm == concat(callbacks), accumulated stats.  Gated on
    # the chatterbox Turbo T3 + S3Gen GGUFs.
    add_executable(test-chatterbox-engine-stream test/test_chatterbox_engine_stream.cpp)
    target_link_libraries(test-chatterbox-engine-stream PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-chatterbox-engine-stream PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-chatterbox-engine-stream)
    tts_cpp_register_test(test-chatterbox-engine-stream
        LABEL    "fixture"
        ARGS     "${_tcb_t3_turbo_gguf}" "${_tcb_s3gen_gguf}"
        REQUIRES "${_tcb_t3_turbo_gguf}" "${_tcb_s3gen_gguf}")

    # output-frequency selection on the chatterbox::Engine API
    # (batch + streaming + validation).  Uses the multilingual fixtures so it
    # runs anywhere the mtl-synth tests do; auto-disabled when they're absent.
    add_executable(test-output-sample-rate test/test_output_sample_rate.cpp)
    target_link_libraries(test-output-sample-rate PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-output-sample-rate PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-output-sample-rate)
    tts_cpp_register_test(test-output-sample-rate
        LABEL    "fixture"
        ARGS     "${_tcb_t3_mtl_gguf}" "${_tcb_s3gen_mtl_gguf}"
        REQUIRES "${_tcb_t3_mtl_gguf}" "${_tcb_s3gen_mtl_gguf}")

    # Supertonic sibling of the above: native rate, 16 kHz batch
    # ratio, construction rejection, streaming result.pcm == concat, and the
    # streaming-equals-whole-buffer-resample batch-exact property.  Gated on the
    # Supertonic GGUF fixture; auto-disabled when it's absent.
    add_executable(test-output-sample-rate-supertonic test/test_output_sample_rate_supertonic.cpp)
    target_link_libraries(test-output-sample-rate-supertonic PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-output-sample-rate-supertonic PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-output-sample-rate-supertonic)
    tts_cpp_register_test(test-output-sample-rate-supertonic
        LABEL    "fixture"
        ARGS     "${_tcb_super_gguf}"
        REQUIRES "${_tcb_super_gguf}")

    # CPU-side persistent-cache validation.
    # Exercises the time_mlp / time_emb / cfm_estimator / weight_mirror
    # caches that amortise per-synth overhead on the multilingual CPU
    # path.  Links chatterbox_tts.cpp directly so it can reach the
    # internal test-hook entrypoints.
    add_executable(test-cpu-caches
        test/test_cpu_caches.cpp
        src/chatterbox_tts.cpp
        src/sched_dispatch.cpp
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

    # Streaming GGUF loader parity. Self-contained: writes
    # its own synthetic GGUF fixture to $TMPDIR, no model download needed.
    add_executable(test-gguf-stream test/test_gguf_stream.cpp)
    target_link_libraries(test-gguf-stream PRIVATE tts-cpp ggml)
    target_include_directories(test-gguf-stream PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-gguf-stream)
    tts_cpp_register_test(test-gguf-stream LABEL "unit")

    # KV-cache dtype parse + backend capability probe.
    # Fixture-free: drives chatterbox_kv_type_from_str / resolve_kv_type
    # against a CPU backend (which supports f32/f16/q8_0 flash attention).
    add_executable(test-kv-cache-type test/test_kv_cache_type.cpp)
    target_link_libraries(test-kv-cache-type PRIVATE tts-cpp ggml)
    target_include_directories(test-kv-cache-type PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-kv-cache-type)
    tts_cpp_register_test(test-kv-cache-type LABEL "unit")

    # ---- Memory-fit preflight (include/tts-cpp/fit.h + *-fit-params) ----
    # Model-free: strict CLI parsers, saturating math, JSON escaping, and the
    # exit-code contract of both fit CLIs.
    # tts- prefix: engines/parakeet already owns the bare test-fit-cli target
    # name, and the umbrella build (SPEECH_BUILD_TESTS=ON) puts every engine in
    # one CMake project, where duplicate target names are a configure error.
    add_executable(test-tts-fit-cli test/test_fit_cli.cpp)
    target_link_libraries(test-tts-fit-cli PRIVATE tts-cpp ggml)
    target_include_directories(test-tts-fit-cli PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-tts-fit-cli)
    tts_cpp_register_test(test-tts-fit-cli LABEL "unit")

    # Audio8 fit parity.  The no-argument form synthesizes a tiny audio8-lm
    # GGUF on the fly, so the LM-side byte-parity gates (weights, KV slabs,
    # slow/fast graph arenas vs a real load + run) need no fixture at all;
    # the fixture form adds the full-pipeline + codec-arena gates.
    add_executable(test-audio8-fit-params test/audio8/test_fit_params.cpp)
    target_link_libraries(test-audio8-fit-params PRIVATE tts-cpp ggml)
    target_include_directories(test-audio8-fit-params PRIVATE ggml/include src include test)
    tts_cpp_apply_ccache(test-audio8-fit-params)
    tts_cpp_register_test(test-audio8-fit-params LABEL "unit")
    tts_cpp_register_test(test-audio8-fit-params-fixture
        EXE      test-audio8-fit-params
        LABEL    "fixture"
        ARGS     "${_tcb_audio8_lm}" "${_tcb_audio8_decoder}" "${_tcb_audio8_encoder}"
        REQUIRES "${_tcb_audio8_lm}" "${_tcb_audio8_decoder}" "${_tcb_audio8_encoder}")

    # Parler fit parity.  The no-argument form synthesizes a tiny but complete
    # parler GGUF on the fly, so the byte-parity gates (weights, KV slab,
    # cross-K/V, decoder/T5/DAC arenas vs a real load + runs) need no fixture
    # at all; the fixture form runs the same gates on the real model (plus the
    # GPU fused-weight stack when n_gpu_layers > 0).
    add_executable(test-parler-fit-params test/parler/test_fit_params.cpp)
    target_link_libraries(test-parler-fit-params PRIVATE tts-cpp ggml)
    target_include_directories(test-parler-fit-params PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-parler-fit-params)
    tts_cpp_register_test(test-parler-fit-params LABEL "unit")
    tts_cpp_register_test(test-parler-fit-params-gpu
        EXE   test-parler-fit-params
        LABEL "gpu"
        ARGS  "--synthetic-gpu")
    tts_cpp_register_test(test-parler-fit-params-fixture
        EXE      test-parler-fit-params
        LABEL    "fixture"
        ARGS     "${_tcb_parler_gguf}"
        REQUIRES "${_tcb_parler_gguf}")

    # CosyVoice3 fit parity.  The no-argument form synthesizes the model set
    # on the fly (tiny metadata-driven LM/flow + full-geometry HiFT + voice),
    # so the byte-parity gates (weights, KV, LM/flow/HiFT arenas vs real
    # dispatches) need no fixture at all; the fixture form runs the same gates
    # on the real converted set.
    add_executable(test-cosyvoice-fit-params test/test_cosyvoice_fit_params.cpp)
    target_link_libraries(test-cosyvoice-fit-params PRIVATE tts-cpp ggml)
    target_include_directories(test-cosyvoice-fit-params PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-cosyvoice-fit-params)
    tts_cpp_register_test(test-cosyvoice-fit-params LABEL "unit")
    tts_cpp_register_test(test-cosyvoice-fit-params-gpu
        EXE   test-cosyvoice-fit-params
        LABEL "gpu"
        ARGS  "--synthetic-gpu")
    tts_cpp_register_test(test-cosyvoice-fit-params-fixture
        EXE      test-cosyvoice-fit-params
        LABEL    "fixture"
        ARGS     "${_tcb_cosy_llm_gguf}" "${_tcb_cosy_flow_gguf}" "${_tcb_cosy_hift_gguf}"
                 "${TTS_CPP_TEST_MODEL_DIR}/voice.gguf"
        REQUIRES "${_tcb_cosy_llm_gguf}" "${_tcb_cosy_flow_gguf}" "${_tcb_cosy_hift_gguf}"
                 "${TTS_CPP_TEST_MODEL_DIR}/voice.gguf")

    # Supertonic fit parity: metadata-only weight/arena projection vs a real
    # load + real reservations, byte for byte.  The no-argument form
    # synthesizes a metadata-only supertonic GGUF on the fly (the loader's
    # exact tensor contract at tiny dims), pinning weight-sizing parity and
    # the refusal paths with no fixture; the stage-arena parity gates stay
    # fixture-gated because the vocoder / vector-estimator channel plans are
    # hardcoded to the real architecture.
    add_executable(test-supertonic-fit-params test/test_supertonic_fit_params.cpp)
    target_link_libraries(test-supertonic-fit-params PRIVATE tts-cpp ggml)
    target_include_directories(test-supertonic-fit-params PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-fit-params)
    tts_cpp_register_test(test-supertonic-fit-params LABEL "unit")

    # fit_price_graph on a metadata-only graph: weights marked allocated with
    # no backend buffer must price under the forced scheduler shape instead of
    # aborting in ggml-alloc, and must agree with the direct pricer.
    add_executable(test-fit-price-unbuffered test/test_fit_price_unbuffered.cpp)
    target_link_libraries(test-fit-price-unbuffered PRIVATE tts-cpp ggml)
    target_include_directories(test-fit-price-unbuffered PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-fit-price-unbuffered)
    tts_cpp_register_test(test-fit-price-unbuffered LABEL "unit")
    tts_cpp_register_test(test-supertonic-fit-params-v2
        EXE      test-supertonic-fit-params
        LABEL    "fixture"
        ARGS     "${_tcb_super2_gguf}" "99"
        REQUIRES "${_tcb_super2_gguf}")
    tts_cpp_register_test(test-supertonic-fit-params-v1
        EXE      test-supertonic-fit-params
        LABEL    "fixture"
        ARGS     "${_tcb_super_gguf}" "99"
        REQUIRES "${_tcb_super_gguf}")
    # v3 is the only variant with CFG (cfg_uncond_scale != 0): this lane is
    # what pins the batch=2 [C, T] CFM loop arena and the unconditional-pass
    # host slabs -- neither v1 nor v2 reaches that branch.
    tts_cpp_register_test(test-supertonic-fit-params-v3
        EXE      test-supertonic-fit-params
        LABEL    "fixture"
        ARGS     "${_tcb_super3_gguf}" "99"
        REQUIRES "${_tcb_super3_gguf}")

    # Chatterbox fit parity: metadata-only projection vs a real T3 load +
    # eval and a real S3Gen synthesis, byte for byte per buffer/arena.
    add_executable(test-chatterbox-fit-params test/test_chatterbox_fit_params.cpp)
    target_link_libraries(test-chatterbox-fit-params PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-chatterbox-fit-params PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-chatterbox-fit-params)
    tts_cpp_register_test(test-chatterbox-fit-params-turbo
        EXE      test-chatterbox-fit-params
        LABEL    "fixture"
        ARGS     "${_tcb_t3_turbo_gguf}" "${_tcb_s3gen_gguf}"
        REQUIRES "${_tcb_t3_turbo_gguf}" "${_tcb_s3gen_gguf}")
    tts_cpp_register_test(test-chatterbox-fit-params-mtl
        EXE      test-chatterbox-fit-params
        LABEL    "fixture"
        ARGS     "${_tcb_t3_mtl_gguf}" "${_tcb_s3gen_mtl_gguf}"
        REQUIRES "${_tcb_t3_mtl_gguf}" "${_tcb_s3gen_mtl_gguf}")

    add_executable(test-s3gen-cfg-rate test/test_s3gen_cfg_rate.cpp)
    target_link_libraries(test-s3gen-cfg-rate PRIVATE tts-cpp ggml)
    target_include_directories(test-s3gen-cfg-rate PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-s3gen-cfg-rate)
    tts_cpp_register_test(test-s3gen-cfg-rate LABEL "unit")

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
    # Metal-only kernel parity check; skipped on CI fleets without Metal via
    # `ctest -LE gpu`. The test resolves the Metal device through the backend
    # registry (no direct ggml_backend_metal_init() call), so it links against
    # base ggml alone and also works when ggml-metal is a GGML_BACKEND_DL
    # dlopened backend rather than a linked target.
    if (TTS_CPP_GGML_METAL)
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
    # Supertonic 3 fixture suite.
    #
    # The harnesses above are arch-agnostic: they take (MODEL.gguf REF_DIR)
    # and don't care which Supertonic generation produced them.  We re-use
    # those same binaries (via `EXE`) to also run them against a Supertonic
    # 3 GGUF plus per-language ONNX-Runtime references. Generate each dump
    # with scripts/dump-supertonic-reference.py using the Supertonic 3
    # ONNX/assets/voice-style paths documented in README.md, stage the
    # converted GGUF, and point the build at them:
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
    # (_tcb_super3_gguf is set with the other fixture paths near the top of
    # the fixture block; the fit-parity v3 lane uses it too.)
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

    add_executable(test-supertonic-explicit-backend test/test_supertonic_explicit_backend.cpp)
    target_link_libraries(test-supertonic-explicit-backend PRIVATE tts-cpp)
    target_include_directories(test-supertonic-explicit-backend PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-explicit-backend)
    tts_cpp_register_test(test-supertonic-explicit-backend
        EXE      test-supertonic-explicit-backend
        LABEL    "fixture"
        ARGS     "${_tcb_super3_gguf}"
        REQUIRES "${_tcb_super3_gguf}")

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
    tts_cpp_register_test(test-supertonic-graph-rewrites-v3
        EXE      test-supertonic-graph-rewrites
        LABEL    "fixture"
        ARGS     "${_tcb_super3_gguf}"
        REQUIRES "${_tcb_super3_gguf}")
    # gpu: on a build with an Accelerate/CBLAS pointwise path the one-graph CFM
    # loop, its [C, T] step and the batched CFG pass are off on CPU, so that arm
    # is the one exercising them there. Without such a path the CPU arm takes
    # them too.
    tts_cpp_register_test(test-supertonic-graph-rewrites-v3-gpu
        EXE      test-supertonic-graph-rewrites
        LABEL    "gpu"
        ARGS     "${_tcb_super3_gguf}" "99"
        REQUIRES "${_tcb_super3_gguf}")
    # OpenCL audit follow-up #2 — text-encoder caches (F13, F16),
    # Phase 2A F16-weight roster (predicate-level), Phase 2D
    # profile-CSV emitter (unit-only).
    add_supertonic_harness(test-supertonic-text-encoder-caches
        test/test_supertonic_text_encoder_caches.cpp)
    add_supertonic_harness(test-supertonic-f16-weights
        test/test_supertonic_f16_weights.cpp)
    # The harness gates on the reference dump, which CI never has, so the
    # deviceless predicate and policy checks get their own registration that
    # always runs: with no arguments main exercises exactly that block.
    tts_cpp_register_test(test-supertonic-f16-policy
        EXE   test-supertonic-f16-weights
        LABEL "unit")
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
    # tts-cpp is needed for supertonic_mulmat_needs_pad(): the header-inline
    # st_mul_mat (supertonic_internal.h) calls it since the Mali output-pad
    # change, and it is defined in supertonic_gguf.cpp inside libtts-cpp.
    target_link_libraries(test-supertonic-convnext-block-fused PRIVATE tts-cpp ggml)
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

    # OpenCL bring-up unit tests. Three CPU-only
    # parity / structural tests for the dispatch + portable-op
    # primitives.  No GGUF needed; register as "unit" label so a
    # fresh checkout's ctest exercises them.  Links against tts-cpp
    # (STATIC) so the detail-namespace symbols are reachable, same
    # pattern as test-mtl-tokenizer / test-t3-mtl / test-tts-streaming.
    add_executable(test-supertonic-backend-dispatch
        test/test_supertonic_backend_dispatch.cpp)
    target_link_libraries(test-supertonic-backend-dispatch PRIVATE tts-cpp)
    target_include_directories(test-supertonic-backend-dispatch PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-backend-dispatch)
    tts_cpp_register_test(test-supertonic-backend-dispatch LABEL "unit")

    add_executable(test-kv-cache-copy test/test_kv_cache_copy.cpp)
    target_link_libraries(test-kv-cache-copy PRIVATE ggml)
    target_include_directories(test-kv-cache-copy PRIVATE ggml/include src)
    tts_cpp_apply_ccache(test-kv-cache-copy)
    tts_cpp_register_test(test-kv-cache-copy LABEL "unit")

    # T3 dual-path dispatch helper unit tests (src/sched_dispatch.{h,cpp}):
    # supports_op walk, force-env escape hatch, sched_fallback lifecycle,
    # pre-allocated-op abort guard.  CPU-only, no GGUF needed.
    add_executable(test-t3-sched-dispatch test/test_t3_sched_dispatch.cpp)
    target_link_libraries(test-t3-sched-dispatch PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-t3-sched-dispatch PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-t3-sched-dispatch)
    tts_cpp_register_test(test-t3-sched-dispatch LABEL "unit")

    # Same binary, guard bypassed: passes only while ggml still rejects the
    # graph (GGML_ABORT before qvac-ext-ggml 6cc71f22, a failed sched
    # allocation from it on), so a ggml sync that un-mirrors the guard fails here.
    tts_cpp_register_test(test-t3-sched-dispatch-abort-repro
                          EXE test-t3-sched-dispatch
                          ARGS --sched-abort-repro
                          LABEL "unit")
    set_property(TEST test-t3-sched-dispatch-abort-repro PROPERTY PASS_REGULAR_EXPRESSION
                 "pre-allocated tensor" "SKIP: no cpu-unsupported set_rows dst type")
    set_property(TEST test-t3-sched-dispatch-abort-repro PROPERTY FAIL_REGULAR_EXPRESSION
                 "ERROR:")

    # Real per-op migration through sched_dispatch: a BLAS (ACCEL) primary
    # rejects everything but matmul-class ops, so add/scale/cpy nodes
    # genuinely migrate to the CPU-last backend (multi-split schedule with
    # cross-backend traffic).  Self-skips when no ACCEL device is built in.
    add_executable(test-t3-sched-migration test/test_t3_sched_migration.cpp)
    target_link_libraries(test-t3-sched-migration PRIVATE tts-cpp ggml tts-cpp-backend-defs)
    target_include_directories(test-t3-sched-migration PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-t3-sched-migration)
    tts_cpp_register_test(test-t3-sched-migration LABEL "unit")

    add_executable(test-supertonic-portable-ops
        test/test_supertonic_portable_ops.cpp)
    target_link_libraries(test-supertonic-portable-ops PRIVATE tts-cpp)
    target_include_directories(test-supertonic-portable-ops PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-portable-ops)
    tts_cpp_register_test(test-supertonic-portable-ops LABEL "unit")

    # CPU-only unit test for the Vulkan-specific
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

    # follow-up — process-wide capability-probe cache +
    # F16 mul_mat probe + Q8_0 K/V flash-attn probe regression test.
    # CPU-only; runs on a fresh checkout under `ctest -L unit`.
    add_executable(test-supertonic-capability-cache
        test/test_supertonic_capability_cache.cpp)
    target_link_libraries(test-supertonic-capability-cache PRIVATE tts-cpp)
    target_include_directories(test-supertonic-capability-cache PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-capability-cache)
    tts_cpp_register_test(test-supertonic-capability-cache LABEL "unit")

    # follow-up — Engine::warm_up + EngineOptions::prewarm_text
    # API-surface lockdown.  CPU-only compile-time + runtime contract test;
    # the Vulkan-side first-synth-latency reduction is exercised by the
    # fixture-bound integration tests on a Vulkan-capable host.
    add_executable(test-supertonic-warm-up-api
        test/test_supertonic_warm_up_api.cpp)
    target_link_libraries(test-supertonic-warm-up-api PRIVATE tts-cpp)
    target_include_directories(test-supertonic-warm-up-api PRIVATE include)
    tts_cpp_apply_ccache(test-supertonic-warm-up-api)
    tts_cpp_register_test(test-supertonic-warm-up-api LABEL "unit")

    # round 3 — multi-device Vulkan auto-pick policy
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

    # round 6 — F16-weights deny-list API surface
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

    # round 4 — multi-dtype K/V flash-attention dispatch
    # resolver (`resolve_kv_attn_type`) — pure-logic policy split
    # from the Vulkan-only dispatch site so the behaviour matrix
    # is testable on CPU with synthetic probe inputs.
    add_executable(test-supertonic-kv-attn-type
        test/test_supertonic_kv_attn_type.cpp)
    target_link_libraries(test-supertonic-kv-attn-type PRIVATE tts-cpp)
    target_include_directories(test-supertonic-kv-attn-type PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-kv-attn-type)
    tts_cpp_register_test(test-supertonic-kv-attn-type LABEL "unit")

    # round 4 — API-surface lockdown for the new
    # EngineOptions::kv_attn_type field, supertonic_model field,
    # supertonic_kv_attn_type() thread-local accessor, and the
    # dispatch-scope `prev_kv_attn_type` for RAII teardown.
    add_executable(test-supertonic-kv-attn-type-api
        test/test_supertonic_kv_attn_type_api.cpp)
    target_link_libraries(test-supertonic-kv-attn-type-api PRIVATE tts-cpp)
    target_include_directories(test-supertonic-kv-attn-type-api PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-kv-attn-type-api)
    tts_cpp_register_test(test-supertonic-kv-attn-type-api LABEL "unit")

    # Supertonic v3 language-coverage contract.
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

    # round 7 — Vulkan env-var passthrough mechanism
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

    # round 7 — voice ttl/dp host cache
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

    # external (cloned) voice injection. The voice-JSON
    # parser + schema validation run as a model-free unit test (always on
    # a fresh checkout under `ctest -L unit`).  The preset-as-JSON parity
    # check — synthesize with a baked preset by name, then with the same
    # voice injected as in-memory tensors and via a voice JSON, and assert
    # bit-identical PCM — needs a Supertonic GGUF and is auto-disabled when
    # the fixture is absent (registered against both the v2 and v3 GGUFs).
    add_executable(test-supertonic-external-voice
        test/test_supertonic_external_voice.cpp)
    target_link_libraries(test-supertonic-external-voice PRIVATE tts-cpp)
    target_include_directories(test-supertonic-external-voice PRIVATE ggml/include src include)
    tts_cpp_apply_ccache(test-supertonic-external-voice)
    tts_cpp_register_test(test-supertonic-external-voice LABEL "unit")
    tts_cpp_register_test(test-supertonic-external-voice-parity
        EXE      test-supertonic-external-voice
        LABEL    "fixture"
        ARGS     "${_tcb_super_gguf}"
        REQUIRES "${_tcb_super_gguf}")
    tts_cpp_register_test(test-supertonic-external-voice-parity-v3
        EXE      test-supertonic-external-voice
        LABEL    "fixture"
        ARGS     "${_tcb_super3_gguf}"
        REQUIRES "${_tcb_super3_gguf}")

    # round 10 — pointer-compare upload-skip tracker
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

    # round 12 #6 — text-encoder speech-prompted-attention
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

    # round 12 #5 — pinned-host-buffer input allocation
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

    # round 13 #1 — input-scratchpad allocator helper
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

# Parler's DAC kernels (src/parler/dac_accel.cpp) are Accelerate-only: the
# test targets that compile them, or check the exactness they imply, get the
# framework on Apple and nothing elsewhere.
set(TTS_CPP_PARLER_ACCEL_TARGETS
    test-parler-dac
    test-parler-dac-cpu-kernels
    test-parler-e2e
    test-parler-engine-stream)

if (APPLE)
    find_library(TTS_CPP_ACCELERATE_FRAMEWORK Accelerate)
    if (TTS_CPP_ACCELERATE_FRAMEWORK)
        foreach(_tts_accel_target ${TTS_CPP_POINTWISE_ACCEL_TARGETS} ${TTS_CPP_PARLER_ACCEL_TARGETS})
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
