cmake_minimum_required(VERSION 3.20)
project(audiogen-cpp LANGUAGES C CXX VERSION 0.1.0)

# ACE-Step music-generation engine (text/lyrics -> LM codes -> FSQ detok ->
# text/cond encoders -> DiT flow-matching -> Oobleck VAE -> 48 kHz stereo PCM).
# Split out of tts-cpp into its own engine folder so the audiogen backend is
# a self-contained library, mirroring the per-engine folder layout the repo
# already uses for tts-cpp / parakeet-cpp. Consumes the same `ggml-speech`
# fork as the rest of the QVAC speech stack (it needs the custom `snake` and
# `col2im_1d` CPU ops that live on the qvac-ext-ggml/speech branch).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Windows: <windows.h> (pulled in via gguf_mmap.h's memory-mapping path) defines
# `min` / `max` as function-like macros, which mangle std::min / std::max in the
# VAE chunked-decode path (vae_ggml.cpp) into a C2589 "illegal token" compile
# error. NOMINMAX suppresses those macros; WIN32_LEAN_AND_MEAN / NOGDI trim the
# rest of the Win32 surface. Mirrors qvac/packages/audiogen-ggml's addon
# CMakeLists (and the tts-cpp / tts-ggml engines).
if (WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN NOGDI)
endif()

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(AUDIOGEN_STANDALONE_DEFAULT ON)
else()
    set(AUDIOGEN_STANDALONE_DEFAULT OFF)
endif()

option(AUDIOGEN_BUILD_LIBRARY     "audiogen-cpp: build the audiogen library (linkage follows BUILD_SHARED_LIBS; defaults to STATIC when unset)" ON)
option(AUDIOGEN_BUILD_EXECUTABLES "audiogen-cpp: build the CLI + per-stage smoke harnesses"        ${AUDIOGEN_STANDALONE_DEFAULT})
option(AUDIOGEN_BUILD_TESTS       "audiogen-cpp: build the CPU-only unit tests"                     ${AUDIOGEN_STANDALONE_DEFAULT})
option(AUDIOGEN_INSTALL           "audiogen-cpp: generate install rules"                            ON)
option(AUDIOGEN_USE_SYSTEM_GGML   "audiogen-cpp: use system-installed GGML (find_package(ggml)) instead of a vendored ggml. Defaults ON: the QVAC speech-stack vcpkg port `ggml-speech` (qvac-ext-ggml/speech branch) is the canonical, pre-patched ggml with the custom snake / col2im_1d ops this engine needs." ON)
option(AUDIOGEN_CCACHE            "audiogen-cpp: use ccache as compiler launcher if available"      ON)

if (ANDROID OR IOS OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(AUDIOGEN_MOBILE ON)
    set(AUDIOGEN_MINIMAX_DEFAULT OFF)
else()
    set(AUDIOGEN_MOBILE OFF)
    set(AUDIOGEN_MINIMAX_DEFAULT ON)
endif()
option(AUDIOGEN_BUILD_MINIMAX "audiogen-cpp: build the desktop MiniMax-Music3 engine (CPU default, GPU via EngineOptions::device)" ${AUDIOGEN_MINIMAX_DEFAULT})
option(AUDIOGEN_COREML "audiogen-cpp: enable Apple Core ML sidecars (ACE-Step VAE decoder and explicit MiniMax conditioner)." OFF)
option(AUDIOGEN_MINIMAX_COREML "audiogen-cpp: enable the full MiniMax Core ML runtime with explicit compute units." OFF)
if (AUDIOGEN_MINIMAX_COREML AND (NOT APPLE OR AUDIOGEN_MOBILE OR NOT AUDIOGEN_BUILD_MINIMAX))
    message(FATAL_ERROR "AUDIOGEN_MINIMAX_COREML requires desktop macOS and AUDIOGEN_BUILD_MINIMAX=ON.")
endif()
if (AUDIOGEN_COREML OR AUDIOGEN_MINIMAX_COREML)
    if (NOT APPLE)
        message(FATAL_ERROR "AUDIOGEN_COREML is Apple-only (requires the CoreML framework); leave it OFF on ${CMAKE_SYSTEM_NAME}.")
    endif()
    enable_language(OBJCXX)
endif()
if (AUDIOGEN_MOBILE)
    set(AUDIOGEN_BUILD_MINIMAX OFF CACHE BOOL "audiogen-cpp: MiniMax-Music3 is unavailable on mobile" FORCE)
endif()

# ccache shaves iterative rebuild time on developer machines. Scoped to
# audiogen's own targets; ggml has its own GGML_CCACHE option we leave alone.
if (AUDIOGEN_CCACHE)
    find_program(AUDIOGEN_CCACHE_PROGRAM ccache)
    if (AUDIOGEN_CCACHE_PROGRAM)
        message(STATUS "audiogen-cpp: using ccache at ${AUDIOGEN_CCACHE_PROGRAM}")
    endif()
endif()

function(audiogen_apply_ccache target)
    if (AUDIOGEN_CCACHE_PROGRAM)
        set_target_properties(${target} PROPERTIES
            C_COMPILER_LAUNCHER   "${AUDIOGEN_CCACHE_PROGRAM}"
            CXX_COMPILER_LAUNCHER "${AUDIOGEN_CCACHE_PROGRAM}"
        )
    endif()
endfunction()

# ---- ggml -----------------------------------------------------------------
# audiogen-cpp does not vendor ggml. In this in-tree subtree we consume the
# QVAC `ggml-speech` port (find_package(ggml)); it ships the snake / col2im_1d
# ops the VAE + DiT graphs rely on. A parent project that already declares a
# `ggml` target (e.g. a super-build) wins and we skip the find entirely.
if (NOT TARGET ggml)
    if (AUDIOGEN_USE_SYSTEM_GGML)
        find_package(ggml CONFIG REQUIRED)
        if (NOT ggml_FOUND)
            message(FATAL_ERROR "audiogen-cpp: system-installed GGML (ggml-speech) not found. "
                                "Point -Dggml_DIR=<...>/lib/cmake/ggml at a ggml built from "
                                "qvac-ext-ggml/speech (needs the snake + col2im_1d ops).")
        endif()
        if (NOT TARGET ggml)
            add_library(ggml ALIAS ggml::ggml)
        endif()
    else()
        message(FATAL_ERROR "audiogen-cpp: AUDIOGEN_USE_SYSTEM_GGML=OFF is unsupported in this "
                            "in-tree subtree (no vendored ggml/ here). Build the ggml-speech "
                            "port and set AUDIOGEN_USE_SYSTEM_GGML=ON (the default).")
    endif()
endif()

# ---- library --------------------------------------------------------------
set(AUDIOGEN_LIB_SOURCES
    src/acestep/audio_edit.cpp
    src/acestep/vae_gguf.cpp
    src/acestep/vae_ggml.cpp
    src/acestep/vae_encode_windows.cpp
    src/acestep/vae_coreml_path.cpp
    src/acestep/vae_api.cpp
    src/acestep/wav_reader.cpp
    src/acestep/dit_gguf.cpp
    src/acestep/dit_ggml.cpp
    src/acestep/textenc_ggml.cpp
    src/acestep/cond_ggml.cpp
    src/acestep/lm_ggml.cpp
    src/acestep/bpe_tokenizer.cpp
    src/acestep/lm_pipeline.cpp
    src/acestep/lyrics_alignment.cpp
    src/acestep/quality_score.cpp
    src/acestep/detok_ggml.cpp
    src/acestep/tok_ggml.cpp
    src/acestep/engine.cpp
    # Memory-fit preflight (include/audiogen-cpp/acestep/fit.h + the
    # acestep-fit-params CLI entry point, linkable by hosts).
    src/acestep/fit.cpp
    src/acestep/fit_cli.cpp
)

if (AUDIOGEN_BUILD_MINIMAX)
    list(APPEND AUDIOGEN_LIB_SOURCES
        src/minimax/logic.cpp
        src/minimax/coreml-conditioner.cpp
        src/minimax/engine.cpp
        src/minimax/fit.cpp
    )
endif()

if (AUDIOGEN_COREML)
    list(APPEND AUDIOGEN_LIB_SOURCES src/coreml/vae-decoder.mm)
    set_source_files_properties(src/coreml/vae-decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    if (AUDIOGEN_BUILD_MINIMAX)
        list(APPEND AUDIOGEN_LIB_SOURCES src/coreml/minimax-conditioner.mm)
        set_source_files_properties(src/coreml/minimax-conditioner.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    endif()
endif()

if (AUDIOGEN_MINIMAX_COREML)
    list(APPEND AUDIOGEN_LIB_SOURCES src/coreml/minimax-ane-validation.mm)
    set_source_files_properties(src/coreml/minimax-ane-validation.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

if (AUDIOGEN_BUILD_LIBRARY)
    # Compile once. The installed/exported library follows BUILD_SHARED_LIBS
    # (shared under the umbrella, because whisper caches ON). In-tree CLIs and
    # tests include src/ headers and call non-AUDIOGEN_API helpers; those
    # symbols are hidden in the DSO, so the tools link this object library
    # instead of libaudiogen-cpp.
    add_library(audiogen-cpp-objects OBJECT ${AUDIOGEN_LIB_SOURCES})
    audiogen_apply_ccache(audiogen-cpp-objects)
    target_include_directories(audiogen-cpp-objects
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/acestep
            $<$<BOOL:${AUDIOGEN_BUILD_MINIMAX}>:${CMAKE_CURRENT_SOURCE_DIR}/src/minimax>
    )
    target_link_libraries(audiogen-cpp-objects PUBLIC ggml)
    if (AUDIOGEN_COREML OR AUDIOGEN_MINIMAX_COREML)
        find_library(AUDIOGEN_COREML_FRAMEWORK     CoreML     REQUIRED)
        find_library(AUDIOGEN_FOUNDATION_FRAMEWORK Foundation REQUIRED)
        target_link_libraries(audiogen-cpp-objects PUBLIC
            ${AUDIOGEN_COREML_FRAMEWORK}
            ${AUDIOGEN_FOUNDATION_FRAMEWORK}
        )
    endif()
    if (AUDIOGEN_COREML)
        target_compile_definitions(audiogen-cpp-objects PRIVATE AUDIOGEN_USE_COREML)
        message(STATUS "audiogen: Core ML sidecars enabled (AUDIOGEN_USE_COREML)")
    endif()
    if (AUDIOGEN_MINIMAX_COREML)
        target_compile_definitions(audiogen-cpp-objects PUBLIC AUDIOGEN_USE_MINIMAX_COREML)
        message(STATUS "audiogen: full MiniMax Core ML enabled (AUDIOGEN_USE_MINIMAX_COREML)")
    endif()
    set_target_properties(audiogen-cpp-objects PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET     hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    if (DEFINED BUILD_SHARED_LIBS)
        add_library(audiogen-cpp $<TARGET_OBJECTS:audiogen-cpp-objects>)
    else()
        add_library(audiogen-cpp STATIC $<TARGET_OBJECTS:audiogen-cpp-objects>)
    endif()
    add_library(audiogen-cpp::audiogen-cpp ALIAS audiogen-cpp)

    target_include_directories(audiogen-cpp
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(audiogen-cpp PUBLIC ggml)
    if (AUDIOGEN_COREML OR AUDIOGEN_MINIMAX_COREML)
        target_link_libraries(audiogen-cpp PUBLIC
            ${AUDIOGEN_COREML_FRAMEWORK}
            ${AUDIOGEN_FOUNDATION_FRAMEWORK}
        )
    endif()

    set_target_properties(audiogen-cpp PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME audiogen-cpp
        CXX_VISIBILITY_PRESET     hidden
        VISIBILITY_INLINES_HIDDEN ON
        VERSION   ${PROJECT_VERSION}
        SOVERSION 0
    )

    get_target_property(_audiogen_type audiogen-cpp TYPE)
    if (_audiogen_type STREQUAL "SHARED_LIBRARY")
        # PRIVATE on the objects: library TUs dllexport. PUBLIC on audiogen-cpp:
        # find_package consumers dllimport. Tools that link the objects do not
        # inherit AUDIOGEN_SHARED, so they do not dllimport symbols they
        # statically pull in.
        target_compile_definitions(audiogen-cpp-objects PRIVATE AUDIOGEN_SHARED AUDIOGEN_BUILD)
        target_compile_definitions(audiogen-cpp PUBLIC AUDIOGEN_SHARED)
    endif()

    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
        target_compile_options(audiogen-cpp-objects PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:C>:-O3>
        )
    endif()
endif()

function(audiogen_link_in_tree_binary target)
    target_link_libraries(${target} PRIVATE audiogen-cpp-objects)
    target_include_directories(${target} PRIVATE
        include
        src
        src/acestep
        $<$<BOOL:${AUDIOGEN_BUILD_MINIMAX}>:${CMAKE_CURRENT_SOURCE_DIR}/src/minimax>
    )
    audiogen_apply_ccache(${target})
endfunction()

# ---- executables (CLI + per-stage smoke harnesses) ------------------------
if (AUDIOGEN_BUILD_EXECUTABLES)
    if (NOT AUDIOGEN_BUILD_LIBRARY)
        message(FATAL_ERROR "AUDIOGEN_BUILD_EXECUTABLES requires AUDIOGEN_BUILD_LIBRARY=ON")
    endif()

    # Thin drivers over engine internals (non-AUDIOGEN_API). Link the object
    # library, not the hidden shared audiogen-cpp.
    foreach (_tool
             acestep_cli:acestep-cli
             dit_smoke:dit-smoke
             textenc_smoke:textenc-smoke
             cond_smoke:cond-smoke
             lm_smoke:lm-smoke
             bpe_smoke:bpe-smoke
             lmgen_smoke:lmgen-smoke
             detok_smoke:detok-smoke
             quantize_cli:acestep-quantize
             music_cli:music-cli
             fit_cli_main:acestep-fit-params)
        string(REPLACE ":" ";" _pair ${_tool})
        list(GET _pair 0 _src)
        list(GET _pair 1 _exe)
        add_executable(${_exe} src/acestep/${_src}.cpp)
        audiogen_link_in_tree_binary(${_exe})
    endforeach()
    target_sources(music-cli PRIVATE
        src/acestep/mini_json.cpp
        src/acestep/music_cli_edit.cpp
    )

    if (AUDIOGEN_BUILD_MINIMAX)
        add_executable(mm3-replay src/minimax/mm3_replay_cli.cpp)
        audiogen_link_in_tree_binary(mm3-replay)
    endif()
endif()

# ---- tests ----------------------------------------------------------------
if (AUDIOGEN_BUILD_TESTS)
    enable_testing()
    include(CTest)

    function(audiogen_register_test target)
        cmake_parse_arguments(ART "" "LABEL" "ARGS" ${ARGN})
        if (NOT DEFINED ART_LABEL)
            set(ART_LABEL "unit")
        endif()
        add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}> ${ART_ARGS})
        set_property(TEST ${target} PROPERTY LABELS ${ART_LABEL})
    endfunction()

    # CPU-only, weight-free unit tests for the pipeline's pure logic
    # (dit schedule, philox golden vector, FSQ decode, top-k/p sampler).
    add_executable(test-acestep-units test/test_acestep_units.cpp)
    audiogen_link_in_tree_binary(test-acestep-units)
    audiogen_register_test(test-acestep-units LABEL "unit")

    add_executable(test-music-cli-edit
        test/test_music_cli_edit.cpp
        src/acestep/mini_json.cpp
        src/acestep/music_cli_edit.cpp
    )
    audiogen_link_in_tree_binary(test-music-cli-edit)
    audiogen_register_test(test-music-cli-edit LABEL "unit")

    add_executable(test-acestep-integration test/test_acestep_integration.cpp)
    audiogen_link_in_tree_binary(test-acestep-integration)
    audiogen_register_test(test-acestep-integration LABEL "integration")
    set_property(TEST test-acestep-integration PROPERTY SKIP_RETURN_CODE 77)

    # Core ML VAE decoder parity vs the ggml decode. Compiled on every
    # platform; self-skips (77) without AUDIOGEN_COREML, a staged
    # AUDIOGEN_TEST_MODELS_DIR, or a compiled vae-decoder.mlmodelc.
    add_executable(test-vae-coreml-parity test/test_vae_coreml_parity.cpp)
    audiogen_link_in_tree_binary(test-vae-coreml-parity)
    if (AUDIOGEN_COREML)
        target_compile_definitions(test-vae-coreml-parity PRIVATE AUDIOGEN_USE_COREML)
    endif()
    audiogen_register_test(test-vae-coreml-parity LABEL "integration")
    set_property(TEST test-vae-coreml-parity PROPERTY SKIP_RETURN_CODE 77)

    # Core ML VAE decode benchmark (ggml GPU vs sidecar, markdown to stdout).
    # Not a ctest: CI runs it directly and appends its table to the job
    # summary; it still fails on parity below 0.999, so a wrong-but-fast
    # sidecar cannot report a win. Same self-skip (77) contract as the
    # parity test.
    add_executable(bench-vae-coreml test/bench_vae_coreml.cpp)
    audiogen_link_in_tree_binary(bench-vae-coreml)
    if (AUDIOGEN_COREML)
        target_compile_definitions(bench-vae-coreml PRIVATE AUDIOGEN_USE_COREML)
    endif()

    # Memory-fit preflight. test-acestep-fit-cli is model-free (strict parsers,
    # saturating arithmetic, JSON escaping, CLI exit codes). test-acestep-fit-params
    # asserts the metadata-only projection matches real allocations
    # byte-for-byte; it reads the same AUDIOGEN_TEST_MODELS_DIR fixture as the
    # integration test and self-skips (77) where it is not staged.
    # acestep- prefix: engines/parakeet already owns the bare test-fit-* target
    # names, and the umbrella build (SPEECH_BUILD_TESTS=ON) puts both engines in
    # one CMake project, where duplicate target names are a configure error.
    add_executable(test-acestep-fit-cli test/test_fit_cli.cpp)
    audiogen_link_in_tree_binary(test-acestep-fit-cli)
    audiogen_register_test(test-acestep-fit-cli LABEL "unit")

    add_executable(test-acestep-fit-params test/test_fit_params.cpp)
    audiogen_link_in_tree_binary(test-acestep-fit-params)
    audiogen_register_test(test-acestep-fit-params LABEL "integration")
    set_property(TEST test-acestep-fit-params PROPERTY SKIP_RETURN_CODE 77)

    find_package(Python3 COMPONENTS Interpreter)
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-acestep-converter
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_convert_acestep.py
        )
        set_property(TEST test-acestep-converter PROPERTY LABELS "unit")
        set_property(TEST test-acestep-converter PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

        # Skips without torch under the system interpreter; the audiogen CI
        # macOS lane also runs it with the Core ML export venv, where torch is
        # installed and the assertions execute.
        add_test(NAME test-acestep-vae-exporter
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_export_vae_coreml.py
        )
        set_property(TEST test-acestep-vae-exporter PROPERTY LABELS "unit")
        set_property(TEST test-acestep-vae-exporter PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    endif()

    if (AUDIOGEN_BUILD_MINIMAX)
        add_executable(test-minimax-conditioner-layout test/test_minimax_conditioner_layout.cpp)
        audiogen_link_in_tree_binary(test-minimax-conditioner-layout)
        audiogen_register_test(test-minimax-conditioner-layout LABEL "unit")

        if (AUDIOGEN_COREML)
            add_executable(test-minimax-coreml-conditioner test/test_minimax_coreml_conditioner.cpp)
            audiogen_link_in_tree_binary(test-minimax-coreml-conditioner)
            audiogen_register_test(test-minimax-coreml-conditioner LABEL "integration")
            set_property(TEST test-minimax-coreml-conditioner PROPERTY SKIP_RETURN_CODE 77)
        endif()

        add_executable(test-minimax-coreml-options test/test_minimax_coreml_options.cpp)
        audiogen_link_in_tree_binary(test-minimax-coreml-options)
        audiogen_register_test(test-minimax-coreml-options LABEL "unit")

        add_executable(test-minimax-units test/test_minimax_units.cpp)
        audiogen_link_in_tree_binary(test-minimax-units)
        audiogen_register_test(test-minimax-units LABEL "unit")

        add_executable(test-minimax-loading-safety test/test_minimax_loading_safety.cpp)
        audiogen_link_in_tree_binary(test-minimax-loading-safety)
        audiogen_register_test(test-minimax-loading-safety LABEL "unit")

        add_executable(test-minimax-fit test/test_minimax_fit.cpp)
        audiogen_link_in_tree_binary(test-minimax-fit)
        audiogen_register_test(test-minimax-fit LABEL "unit")

        add_executable(test-minimax-fit-models test/test_minimax_fit_models.cpp)
        audiogen_link_in_tree_binary(test-minimax-fit-models)
        audiogen_register_test(test-minimax-fit-models LABEL "integration")
        set_property(TEST test-minimax-fit-models PROPERTY SKIP_RETURN_CODE 77)

        add_executable(test-minimax-integration test/test_minimax_integration.cpp)
        target_link_libraries(test-minimax-integration PRIVATE audiogen-cpp)
        target_include_directories(test-minimax-integration PRIVATE include)
        audiogen_apply_ccache(test-minimax-integration)
        audiogen_register_test(test-minimax-integration LABEL "integration")
        set_property(TEST test-minimax-integration PROPERTY SKIP_RETURN_CODE 77)

        # Model-backed quality regression over engine internals: DiT
        # determinism across computes and the on-manifold final-latent check
        # that rejects the historical "valid WAV but noise" failure.
        add_executable(test-minimax-quality test/test_minimax_quality.cpp)
        audiogen_link_in_tree_binary(test-minimax-quality)
        audiogen_register_test(test-minimax-quality LABEL "integration")
        set_property(TEST test-minimax-quality PROPERTY SKIP_RETURN_CODE 77)
        add_test(NAME test-minimax-quality-q4 COMMAND $<TARGET_FILE:test-minimax-quality> --q4)
        set_property(TEST test-minimax-quality-q4 PROPERTY LABELS "integration")
        set_property(TEST test-minimax-quality-q4 PROPERTY SKIP_RETURN_CODE 77)

        add_executable(test-minimax-metal-ops test/test_minimax_metal_ops.cpp)
        audiogen_link_in_tree_binary(test-minimax-metal-ops)
        audiogen_register_test(test-minimax-metal-ops LABEL "gpu")
        set_property(TEST test-minimax-metal-ops PROPERTY SKIP_RETURN_CODE 77)

        find_package(Python3 COMPONENTS Interpreter)
        if (Python3_Interpreter_FOUND)
            add_test(NAME test-minimax-conditioner-exporter
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_export_minimax_conditioner_coreml.py
            )
            set_property(TEST test-minimax-conditioner-exporter PROPERTY LABELS "unit")
            set_property(TEST test-minimax-conditioner-exporter PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

            add_test(NAME test-minimax-converter
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_convert_minimax.py
            )
            set_property(TEST test-minimax-converter PROPERTY LABELS "unit")
            set_property(TEST test-minimax-converter PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

            add_test(NAME test-minimax-quant-audit
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_audit_quant_types.py
            )
            set_property(TEST test-minimax-quant-audit PROPERTY LABELS "unit")
            set_property(TEST test-minimax-quant-audit PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

            add_test(NAME test-minimax-dump-compare
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_compare_ar_dumps.py
            )
            set_property(TEST test-minimax-dump-compare PROPERTY LABELS "unit")
            set_property(TEST test-minimax-dump-compare PROPERTY ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
        endif()
    endif()

    # The engine-comparison harness ships its own node:test suites
    # (benchmarks/comparison/tests/*.test.js). Registering them keeps the
    # harness's adapters/aggregation/report logic from rotting unnoticed;
    # they vanish (with a status message) on hosts without node.
    find_program(AUDIOGEN_NODE_EXECUTABLE node)
    if (AUDIOGEN_NODE_EXECUTABLE)
        # node discovers the suites itself from the working directory, so one
        # added later runs without reconfiguring (a configure-time file(GLOB)
        # would pin the list until the next cmake run). A discovery that finds
        # nothing still exits 0, so the zero-pass summary has to fail the test:
        # FAIL_REGULAR_EXPRESSION is additive, leaving node's exit code as the
        # primary signal.
        add_test(NAME test-audiogen-comparison-lib
            COMMAND ${AUDIOGEN_NODE_EXECUTABLE} --test
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/comparison/tests)
        set_property(TEST test-audiogen-comparison-lib PROPERTY LABELS "unit")
        set_property(TEST test-audiogen-comparison-lib
            PROPERTY FAIL_REGULAR_EXPRESSION "# pass 0")
    else()
        message(STATUS "audiogen: node not found - test-audiogen-comparison-lib not registered")
    endif()
endif()

# ---- install / export -----------------------------------------------------
if (AUDIOGEN_INSTALL AND AUDIOGEN_BUILD_LIBRARY)
    install(TARGETS audiogen-cpp
        EXPORT audiogen-cppTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY include/audiogen-cpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
        PATTERN "minimax" EXCLUDE
    )
    install(FILES LICENSE NOTICE
        DESTINATION ${CMAKE_INSTALL_DATADIR}/audiogen-cpp
    )
    install(FILES
        scripts/convert-acestep-to-gguf.py
        scripts/download-acestep-checkpoints.sh
        DESTINATION ${CMAKE_INSTALL_DATADIR}/audiogen-cpp/scripts
    )
    if (AUDIOGEN_BUILD_MINIMAX)
        install(DIRECTORY include/audiogen-cpp/minimax
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/audiogen-cpp
            FILES_MATCHING PATTERN "*.h"
        )
        install(FILES
            scripts/convert-minimax-music3-to-gguf.py
            scripts/download-minimax-music3.sh
            DESTINATION ${CMAKE_INSTALL_DATADIR}/audiogen-cpp/scripts
        )
    endif()
    install(EXPORT audiogen-cppTargets
        FILE audiogen-cppTargets.cmake
        NAMESPACE audiogen-cpp::
        DESTINATION ${CMAKE_INSTALL_DATADIR}/audiogen-cpp
    )
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/audiogen-cppConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/audiogen-cppConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/audiogen-cpp
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/audiogen-cppConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMinorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/audiogen-cppConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/audiogen-cppConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_DATADIR}/audiogen-cpp
    )
endif()
