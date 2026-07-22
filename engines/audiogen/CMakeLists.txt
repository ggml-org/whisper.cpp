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
    src/acestep/vae_gguf.cpp
    src/acestep/vae_ggml.cpp
    src/acestep/vae_api.cpp
    src/acestep/dit_gguf.cpp
    src/acestep/dit_ggml.cpp
    src/acestep/textenc_ggml.cpp
    src/acestep/cond_ggml.cpp
    src/acestep/lm_ggml.cpp
    src/acestep/bpe_tokenizer.cpp
    src/acestep/lm_pipeline.cpp
    src/acestep/detok_ggml.cpp
    src/acestep/engine.cpp
)

if (AUDIOGEN_BUILD_LIBRARY)
    if (DEFINED BUILD_SHARED_LIBS)
        add_library(audiogen-cpp ${AUDIOGEN_LIB_SOURCES})
    else()
        add_library(audiogen-cpp STATIC ${AUDIOGEN_LIB_SOURCES})
    endif()
    add_library(audiogen-cpp::audiogen-cpp ALIAS audiogen-cpp)
    audiogen_apply_ccache(audiogen-cpp)

    target_include_directories(audiogen-cpp
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/acestep
    )
    target_link_libraries(audiogen-cpp PUBLIC ggml)

    set_target_properties(audiogen-cpp PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME audiogen-cpp
        CXX_VISIBILITY_PRESET     hidden
        VISIBILITY_INLINES_HIDDEN ON
        VERSION   ${PROJECT_VERSION}
        SOVERSION 0
    )

    # Symbol-export contract for shared builds (see include/audiogen-cpp/export.h).
    # No-op for the default STATIC build, where consumers (and the in-tree smoke
    # tools / unit tests) see the whole symbol table.
    get_target_property(_audiogen_type audiogen-cpp TYPE)
    if (_audiogen_type STREQUAL "SHARED_LIBRARY")
        target_compile_definitions(audiogen-cpp PUBLIC  AUDIOGEN_SHARED)
        target_compile_definitions(audiogen-cpp PRIVATE AUDIOGEN_BUILD)
    endif()

    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
        target_compile_options(audiogen-cpp PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:C>:-O3>
        )
    endif()
endif()

# ---- executables (CLI + per-stage smoke harnesses) ------------------------
if (AUDIOGEN_BUILD_EXECUTABLES)
    if (NOT AUDIOGEN_BUILD_LIBRARY)
        message(FATAL_ERROR "AUDIOGEN_BUILD_EXECUTABLES requires AUDIOGEN_BUILD_LIBRARY=ON")
    endif()

    # Each harness is a thin driver over the audiogen-cpp library; they reach
    # internal (non-AUDIOGEN_API) helpers, which resolve against the default
    # STATIC archive. Include src/ so they can pull the internal stage headers.
    foreach (_tool
             acestep_cli:acestep-cli
             dit_smoke:dit-smoke
             textenc_smoke:textenc-smoke
             cond_smoke:cond-smoke
             lm_smoke:lm-smoke
             bpe_smoke:bpe-smoke
             lmgen_smoke:lmgen-smoke
             detok_smoke:detok-smoke
             music_cli:music-cli)
        string(REPLACE ":" ";" _pair ${_tool})
        list(GET _pair 0 _src)
        list(GET _pair 1 _exe)
        add_executable(${_exe} src/acestep/${_src}.cpp)
        target_link_libraries(${_exe} PRIVATE audiogen-cpp)
        target_include_directories(${_exe} PRIVATE include src src/acestep)
        audiogen_apply_ccache(${_exe})
    endforeach()
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
    target_link_libraries(test-acestep-units PRIVATE audiogen-cpp)
    target_include_directories(test-acestep-units PRIVATE include src src/acestep)
    audiogen_apply_ccache(test-acestep-units)
    audiogen_register_test(test-acestep-units LABEL "unit")
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
    )
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
