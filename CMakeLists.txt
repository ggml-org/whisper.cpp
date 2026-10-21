# QVAC speech superbuild: feature-gated umbrella over the vendored
# whisper.cpp (third_party/, git subtree, see third_party/whisper.cpp/PATCHES.md)
# and our engines (engines/). Every component consumes ONE system ggml —
# the ggml-speech vcpkg port built from qvac-ext-ggml@speech — so the
# stack shares a single ggml pin and file set; the ggml/ tree vendored
# inside the whisper subtree is never compiled.
#
# Each engine also stays independently configurable standalone
# (cmake -S engines/<name>), which is what the per-engine vcpkg ports and
# CI lanes use today; this umbrella is additive and is the base for the
# future single speech port (QIP #94, Ticket 4).
cmake_minimum_required(VERSION 3.20)
project(qvac-speech LANGUAGES C CXX)

option(SPEECH_BUILD_WHISPER     "speech: build the whisper engine (third_party/whisper.cpp)" ON)
option(SPEECH_BUILD_PARAKEET    "speech: build the parakeet engine (engines/parakeet)"       ON)
option(SPEECH_BUILD_TTS         "speech: build the TTS engines (engines/tts)"                ON)
option(SPEECH_BUILD_AUDIOGEN    "speech: build the audiogen engine (engines/audiogen)"       ON)
option(SPEECH_BUILD_TESTS       "speech: build the engines' test harnesses"                  OFF)
option(SPEECH_BUILD_WHISPER_TESTS "speech: build whisper's test harnesses too (fixture-only tests run model-free; the transcription tests additionally need downloaded whisper models — filter by label)" OFF)
option(SPEECH_BUILD_EXECUTABLES "speech: build the engines' CLI executables"                 ON)

if (SPEECH_BUILD_TESTS)
    enable_testing()
endif()

# One ggml for everything. Resolved from the ggml-speech vcpkg port (or
# any qvac-ext-ggml@speech install on CMAKE_PREFIX_PATH). Every
# subproject below guards its own ggml acquisition with
# `if (NOT TARGET ggml)`, so defining the target here short-circuits
# all of them onto this single import.
find_package(ggml CONFIG REQUIRED)
add_library(ggml ALIAS ggml::ggml)

if (SPEECH_BUILD_WHISPER)
    # CACHE FORCE throughout this block: whisper's cmake_minimum_required(3.5)
    # leaves CMP0077 OLD in its scope, so its option() defaults would clobber
    # plain normal-variable set()s on the first configure.
    set(WHISPER_USE_SYSTEM_GGML ON  CACHE BOOL "umbrella: one shared system ggml" FORCE)
    # The umbrella (and speech-cpp port) ships the release in UPSTREAM_PIN.
    # Override upstream's nightly default, including stale build caches.
    set(WHISPER_BUILD_IS_DEV OFF CACHE BOOL "umbrella: pinned Whisper release" FORCE)
    set(WHISPER_BUILD_TESTS     ${SPEECH_BUILD_WHISPER_TESTS} CACHE BOOL "umbrella: see SPEECH_BUILD_WHISPER_TESTS" FORCE)
    set(WHISPER_BUILD_EXAMPLES  ${SPEECH_BUILD_EXECUTABLES} CACHE BOOL "umbrella: whisper-cli lives in examples/" FORCE)
    # Upstream ships its own C-API parakeet which collides with
    # engines/parakeet — gate it off (QVAC patch in the subtree, see
    # third_party/whisper.cpp/PATCHES.md). engines/parakeet's library
    # artifacts moved to the qvac-parakeet namespace, but upstream's
    # `parakeet-cli` target name still clashes with ours, and its
    # install payload (lib/libparakeet.*, lib/cmake/parakeet/,
    # parakeet.pc) is owned by the standalone whisper-cpp vcpkg port —
    # shipping it from here too would recreate the file-ownership
    # conflict this gate exists to avoid.
    # CACHE FORCE because whisper's cmake_minimum_required(3.5) leaves
    # CMP0077 OLD in its scope: a plain normal-variable set() here would
    # be clobbered by its option() default on the first configure.
    set(WHISPER_BUILD_PARAKEET OFF CACHE BOOL "umbrella: upstream parakeet collides with engines/parakeet" FORCE)
    add_subdirectory(third_party/whisper.cpp)
endif()

if (SPEECH_BUILD_PARAKEET)
    set(PARAKEET_USE_SYSTEM_GGML ON)
    set(PARAKEET_BUILD_TESTS       ${SPEECH_BUILD_TESTS})
    set(PARAKEET_BUILD_EXECUTABLES ${SPEECH_BUILD_EXECUTABLES})
    set(PARAKEET_BUILD_EXAMPLES    OFF)
    add_subdirectory(engines/parakeet)
endif()

if (SPEECH_BUILD_TTS)
    # TTS_CPP_USE_SYSTEM_GGML already defaults ON in this tree.
    set(TTS_CPP_BUILD_TESTS       ${SPEECH_BUILD_TESTS})
    set(TTS_CPP_BUILD_EXECUTABLES ${SPEECH_BUILD_EXECUTABLES})
    add_subdirectory(engines/tts)
endif()

if (SPEECH_BUILD_AUDIOGEN)
    # AUDIOGEN_USE_SYSTEM_GGML already defaults ON in this tree, and the
    # engine short-circuits onto the umbrella's shared ggml via its own
    # `if (NOT TARGET ggml)` guard.
    set(AUDIOGEN_BUILD_TESTS       ${SPEECH_BUILD_TESTS})
    set(AUDIOGEN_BUILD_EXECUTABLES ${SPEECH_BUILD_EXECUTABLES})
    add_subdirectory(engines/audiogen)
endif()

# Whisper-caller GPU-picker helper (engines/whisper-gpu-picker.h) — header-only,
# consumed by the packages/asr-ggml addon that wraps third_party/whisper.cpp.
# From this repo we only own the header + its model-free unit test, gated on
# SPEECH_BUILD_TESTS (no ggml backends actually loaded, no whisper model
# required, runs on every host in the matrix).
if (SPEECH_BUILD_TESTS)
    add_executable(test-whisper-gpu-picker engines/whisper-gpu-picker.test.cpp)
    target_link_libraries(test-whisper-gpu-picker PRIVATE ggml)
    target_include_directories(test-whisper-gpu-picker PRIVATE engines)
    # ggml-backend.h relies on C++11 attributes ([[noreturn]]) and
    # std::initializer_list; Apple clang defaults to an older standard when the
    # target does not declare one, so pin C++17 explicitly.
    target_compile_features(test-whisper-gpu-picker PRIVATE cxx_std_17)
    add_test(NAME test-whisper-gpu-picker
             COMMAND $<TARGET_FILE:test-whisper-gpu-picker>)
    set_property(TEST test-whisper-gpu-picker PROPERTY LABELS unit)

    # The MOSS GPU suites' accelerator-arm table (engines/test/moss_gpu_arm.h)
    # is shared by tts and parakeet tests; this pins the names it accepts.
    add_executable(test-moss-gpu-arm engines/test/moss_gpu_arm.test.cpp)
    target_include_directories(test-moss-gpu-arm PRIVATE engines/test)
    target_compile_features(test-moss-gpu-arm PRIVATE cxx_std_17)
    add_test(NAME test-moss-gpu-arm COMMAND $<TARGET_FILE:test-moss-gpu-arm>)
    set_property(TEST test-moss-gpu-arm PROPERTY LABELS unit)
endif()
