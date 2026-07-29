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
cmake_minimum_required(VERSION 3.19)
project(qvac-speech LANGUAGES C CXX)

option(SPEECH_BUILD_WHISPER     "speech: build the whisper engine (third_party/whisper.cpp)" ON)
option(SPEECH_BUILD_PARAKEET    "speech: build the parakeet engine (engines/parakeet)"       ON)
option(SPEECH_BUILD_TTS         "speech: build the TTS engines (engines/tts)"                ON)
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
