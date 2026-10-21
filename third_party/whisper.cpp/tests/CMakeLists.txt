set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (EMSCRIPTEN)
    #
    # test-whisper-js

    set(TEST_TARGET test-whisper-js)

    add_test(NAME ${TEST_TARGET}
        COMMAND node test-whisper.js --experimental-wasm-threads
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )

    return()
endif()

set(TEST_TARGET test-whisper-cli-tiny)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:whisper-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.bin -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "tiny;gh")

set(TEST_TARGET test-whisper-cli-tiny.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:whisper-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.en.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "tiny;en")

set(TEST_TARGET test-whisper-cli-base)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:whisper-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-base.bin -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "base")

set(TEST_TARGET test-whisper-cli-base.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:whisper-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-base.en.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "base;en")

set(TEST_TARGET test-whisper-cli-small)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:whisper-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-small.bin -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "small")

set(TEST_TARGET test-whisper-cli-small.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:whisper-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-small.en.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "small;en")

set(TEST_TARGET test-whisper-cli-medium)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:whisper-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-medium.bin -l fr
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "medium")

set(TEST_TARGET test-whisper-cli-medium.en)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:whisper-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-medium.en.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "medium;en")

set(TEST_TARGET test-whisper-cli-large)
add_test(NAME ${TEST_TARGET}
    COMMAND $<TARGET_FILE:whisper-cli>
    -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-large.bin
    -f ${PROJECT_SOURCE_DIR}/samples/jfk.wav)
set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "large")

if (WHISPER_COMMON_FFMPEG)
    set(TEST_TARGET test-whisper-cli-tiny-mp3)
    # Check with reviewers: any way to check the output transcription via ctest (diff, ...)?
    add_test(NAME ${TEST_TARGET}
      COMMAND $<TARGET_FILE:whisper-cli>
      -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.en.bin
      -f ${PROJECT_SOURCE_DIR}/samples/jfk.mp3)
    set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "tiny;mp3")
endif()

# UTF-8 helper unit test
set(UTF8_TEST test-common-utf8)
add_executable(${UTF8_TEST} ${UTF8_TEST}.cpp)
target_include_directories(${UTF8_TEST} PRIVATE ../examples)
target_link_libraries(${UTF8_TEST} PRIVATE common whisper)
add_test(NAME ${UTF8_TEST} COMMAND ${UTF8_TEST})
set_tests_properties(${UTF8_TEST} PROPERTIES LABELS "unit")

# decoder logits-slice suffix-offset unit test (header-only, no model needed)
set(LOGITS_SLICE_TEST test-whisper-logits-slice)
add_executable(${LOGITS_SLICE_TEST} ${LOGITS_SLICE_TEST}.cpp)
target_include_directories(${LOGITS_SLICE_TEST} PRIVATE ../src)
add_test(NAME ${LOGITS_SLICE_TEST} COMMAND ${LOGITS_SLICE_TEST})
set_tests_properties(${LOGITS_SLICE_TEST} PROPERTIES LABELS "unit;gh")

# language lookup-table unit test (QVAC patch, see PATCHES.md; no model needed)
set(LANG_TEST test-whisper-lang)
add_executable(${LANG_TEST} ${LANG_TEST}.cpp)
target_include_directories(${LANG_TEST} PRIVATE ../include)
target_link_libraries(${LANG_TEST} PRIVATE whisper)
add_test(NAME ${LANG_TEST} COMMAND ${LANG_TEST})
set_tests_properties(${LANG_TEST} PROPERTIES LABELS "unit;gh")

# default-parameter contract test (QVAC patch, see PATCHES.md; no model needed)
set(PARAMS_TEST test-whisper-params)
add_executable(${PARAMS_TEST} ${PARAMS_TEST}.cpp)
target_include_directories(${PARAMS_TEST} PRIVATE ../include)
target_link_libraries(${PARAMS_TEST} PRIVATE whisper)
add_test(NAME ${PARAMS_TEST} COMMAND ${PARAMS_TEST})
set_tests_properties(${PARAMS_TEST} PROPERTIES LABELS "unit;gh")

# Memory-fit preflight: model-free unit test for the CLI helpers and the
# whisper_fit_params argument/status contract (QVAC patch, see PATCHES.md)
set(FIT_CLI_TEST test-whisper-fit-cli)
add_executable(${FIT_CLI_TEST} ${FIT_CLI_TEST}.cpp)
target_include_directories(${FIT_CLI_TEST} PRIVATE ../include ../examples)
target_link_libraries(${FIT_CLI_TEST} PRIVATE whisper)
add_test(NAME ${FIT_CLI_TEST} COMMAND ${FIT_CLI_TEST})
set_tests_properties(${FIT_CLI_TEST} PROPERTIES LABELS "unit;gh")

# Memory-fit preflight: byte-parity gates against a REAL load of the committed
# tiny + silero fixtures -- the projection must equal what whisper actually
# allocates (QVAC patch, see PATCHES.md). Registered twice:
#   - ${FIT_TEST}: CPU parity (`unit` label, model-download-free, every CI lane).
#     On CPU the main pool IS host memory, so the device/host split is trivial
#     there -- the split is only really exercised by:
#   - ${FIT_TEST}-gpu: the same parity gates on a GPU/IGPU backend, plus a
#     driver-truth check of the device/host classifier (see the test source).
#     Labeled `gpu`: the CPU CI lanes exclude it (`ctest -LE 'gpu|...'`,
#     whisper-ci.yml) and any lane with a real GPU picks it up via
#     `ctest -L gpu` -- the same gate the engines' self-hosted GPU lanes use
#     (parakeet-ci.yml / tts-ci.yml). Exits 77 (= ctest SKIP, green but
#     reported as skipped, never as passed) when the build or machine
#     exposes no GPU device, so a mislabeled lane cannot green-wash it.
set(FIT_TEST test-whisper-fit-params)
add_executable(${FIT_TEST} ${FIT_TEST}.cpp)
target_include_directories(${FIT_TEST} PRIVATE ../include)
target_link_libraries(${FIT_TEST} PRIVATE whisper)
target_compile_definitions(${FIT_TEST} PRIVATE
    WHISPER_FIT_TEST_MODEL="${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.bin"
    WHISPER_FIT_TEST_VAD_MODEL="${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin")
add_test(NAME ${FIT_TEST} COMMAND ${FIT_TEST})
set_tests_properties(${FIT_TEST} PROPERTIES LABELS "unit;tiny;gh")
add_test(NAME ${FIT_TEST}-gpu COMMAND ${FIT_TEST}
    ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.bin
    ${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin
    gpu)
set_tests_properties(${FIT_TEST}-gpu PROPERTIES LABELS "gpu;tiny" SKIP_RETURN_CODE 77)

# Memory-fit preflight from the registry's weightless descriptions: projecting
# a committed *.fit.gguf must equal projecting the model it describes (QVAC
# patch, see PATCHES.md). CPU form under `unit`, GPU form under `gpu`, as above.
set(FIT_DESCRIPTION_TEST test-whisper-fit-description)
set(FIT_DESCRIPTION_FIXTURES
    ${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin
    ${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.fit.gguf
    ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.bin
    ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.fit.gguf
    ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-bci.bin
    ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-bci.fit.gguf)
add_executable(${FIT_DESCRIPTION_TEST} ${FIT_DESCRIPTION_TEST}.cpp)
target_include_directories(${FIT_DESCRIPTION_TEST} PRIVATE ../include)
target_link_libraries(${FIT_DESCRIPTION_TEST} PRIVATE whisper)
add_test(NAME ${FIT_DESCRIPTION_TEST} COMMAND ${FIT_DESCRIPTION_TEST} ${FIT_DESCRIPTION_FIXTURES})
set_tests_properties(${FIT_DESCRIPTION_TEST} PROPERTIES LABELS "unit;tiny;gh")
add_test(NAME ${FIT_DESCRIPTION_TEST}-gpu COMMAND ${FIT_DESCRIPTION_TEST} ${FIT_DESCRIPTION_FIXTURES} gpu)
set_tests_properties(${FIT_DESCRIPTION_TEST}-gpu PROPERTIES LABELS "gpu;tiny" SKIP_RETURN_CODE 77)

set(BUFFER_LOADER_TEST test-whisper-buffer-loader)
add_executable(${BUFFER_LOADER_TEST} ${BUFFER_LOADER_TEST}.cpp)
target_include_directories(${BUFFER_LOADER_TEST} PRIVATE ../include ../examples)
target_link_libraries(${BUFFER_LOADER_TEST} PRIVATE common whisper)
add_test(NAME ${BUFFER_LOADER_TEST} COMMAND ${BUFFER_LOADER_TEST})
set_tests_properties(${BUFFER_LOADER_TEST} PROPERTIES LABELS "unit;gh")

# whisper_full() with n_samples == 0 must not read an uninitialized mel (#3978)
set(ZERO_SAMPLES_TEST test-whisper-zero-samples)
add_executable(${ZERO_SAMPLES_TEST} ${ZERO_SAMPLES_TEST}.cpp)
target_include_directories(${ZERO_SAMPLES_TEST} PRIVATE ../include ../examples)
target_link_libraries(${ZERO_SAMPLES_TEST} PRIVATE common whisper)
target_compile_definitions(${ZERO_SAMPLES_TEST} PRIVATE
    WHISPER_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.bin")
add_test(NAME ${ZERO_SAMPLES_TEST} COMMAND ${ZERO_SAMPLES_TEST})
set_tests_properties(${ZERO_SAMPLES_TEST} PROPERTIES LABELS "tiny;gh")


# VAD test tests VAD in isolation
set(VAD_TEST test-vad)
add_executable(${VAD_TEST} ${VAD_TEST}.cpp)
target_include_directories(${VAD_TEST} PRIVATE ../include ../examples)
target_link_libraries(${VAD_TEST} PRIVATE common whisper)
target_compile_definitions(${VAD_TEST} PRIVATE
    VAD_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin"
    SAMPLE_PATH="${PROJECT_SOURCE_DIR}/samples/jfk.wav")
add_test(NAME ${VAD_TEST} COMMAND ${VAD_TEST})
set_tests_properties(${VAD_TEST} PROPERTIES LABELS "unit")

# VAD test full uses whisper_full with VAD enabled
set(VAD_TEST test-vad-full)
add_executable(${VAD_TEST} ${VAD_TEST}.cpp)
target_include_directories(${VAD_TEST} PRIVATE ../include ../examples)
target_link_libraries(${VAD_TEST} PRIVATE common whisper)
target_compile_definitions(${VAD_TEST} PRIVATE
    WHISPER_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/ggml-base.en.bin"
    VAD_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin"
    SAMPLE_PATH="${PROJECT_SOURCE_DIR}/samples/jfk.wav")
add_test(NAME ${VAD_TEST} COMMAND ${VAD_TEST})
set_tests_properties(${VAD_TEST} PROPERTIES LABELS "base;en")

# Streaming VAD test pins the upstream PR #3677 contract:
# whisper_vad_detect_speech_no_reset + whisper_vad_reset_state must
# combine to yield exactly the same per-chunk probabilities as the
# original whisper_vad_detect_speech call. Uses only the silero VAD
# model (no whisper transcribe model needed).
set(VAD_TEST test-vad-streaming)
add_executable(${VAD_TEST} ${VAD_TEST}.cpp)
target_include_directories(${VAD_TEST} PRIVATE ../include ../examples)
target_link_libraries(${VAD_TEST} PRIVATE common whisper)
target_compile_definitions(${VAD_TEST} PRIVATE
    VAD_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin"
    SAMPLE_PATH="${PROJECT_SOURCE_DIR}/samples/jfk.wav")
add_test(NAME ${VAD_TEST} COMMAND ${VAD_TEST})
set_tests_properties(${VAD_TEST} PROPERTIES LABELS "unit;vad")

# VAD use_gpu contract test (QVAC patch, see PATCHES.md): use_gpu=true must
# initialize and produce CPU-matching probabilities. Exercises the real GPU
# path on GPU builds and falls back to CPU-vs-CPU elsewhere.
set(VAD_TEST test-vad-gpu)
add_executable(${VAD_TEST} ${VAD_TEST}.cpp)
target_include_directories(${VAD_TEST} PRIVATE ../include ../examples)
target_link_libraries(${VAD_TEST} PRIVATE common whisper)
target_compile_definitions(${VAD_TEST} PRIVATE
    VAD_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin"
    SAMPLE_PATH="${PROJECT_SOURCE_DIR}/samples/jfk.wav")
add_test(NAME ${VAD_TEST} COMMAND ${VAD_TEST})
set_tests_properties(${VAD_TEST} PROPERTIES LABELS "unit;vad")

if (WHISPER_BUILD_PARAKEET) # QVAC patch, see PATCHES.md: these link the gated `parakeet` target
# Parakeet model loading test
set(PARAKEET_TEST test-parakeet)
add_executable(${PARAKEET_TEST} ${PARAKEET_TEST}.cpp)
target_include_directories(${PARAKEET_TEST} PRIVATE ../include ../examples)
target_link_libraries(${PARAKEET_TEST} PRIVATE parakeet common whisper)
target_compile_definitions(${PARAKEET_TEST} PRIVATE
    PARAKEET_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-ggml-parakeet-tdt.bin"
    PARAKEET_BAD_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-ggml-parakeet-tdt-bad-nfft0.bin"
    SAMPLE_PATH="${PROJECT_SOURCE_DIR}/samples/jfk.wav")
add_test(NAME ${PARAKEET_TEST} COMMAND ${PARAKEET_TEST})
set_tests_properties(${PARAKEET_TEST} PROPERTIES LABELS "parakeet;gh")

# The following parakeet test require a real ggml-parakeet-tdt model to have
# been converted or downloaded:
# $ hf download danbev/parakeet parakeet-tdt-0.6b-v3-f32.bin --local-dir models
#
# And also required more audio samples that are shipped by default. These can
# downloaded by running:
# $ make samples
function(add_parakeet_transcription_test TEST_TARGET TEST_SOURCE SAMPLE_PATH EXPECTED_TRANSCRIPTION_PATH)
    set(TRANSCRIPTION_SIMILARITY_THRESHOLD "1.0")
    if (ARGC GREATER 4)
        set(TRANSCRIPTION_SIMILARITY_THRESHOLD "${ARGV4}")
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCE})
    target_include_directories(${TEST_TARGET} PRIVATE ../include ../examples)
    target_link_libraries(${TEST_TARGET} PRIVATE parakeet common whisper)
    target_compile_definitions(${TEST_TARGET} PRIVATE
        PARAKEET_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/ggml-parakeet-tdt-0.6b-v3-f32.bin"
        SAMPLE_PATH="${PROJECT_SOURCE_DIR}/${SAMPLE_PATH}"
        EXPECTED_TRANSCRIPTION_PATH="${PROJECT_SOURCE_DIR}/${EXPECTED_TRANSCRIPTION_PATH}"
        TRANSCRIPTION_SIMILARITY_THRESHOLD=${TRANSCRIPTION_SIMILARITY_THRESHOLD})

    add_custom_target(run-${TEST_TARGET}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        DEPENDS ${TEST_TARGET}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endfunction()

add_parakeet_transcription_test(
    test-parakeet-full-jfk
    test-parakeet-full.cpp
    samples/jfk.wav
    tests/parakeet-expected-jfk-output.txt)

add_parakeet_transcription_test(
    test-parakeet-full-gb1
    test-parakeet-full.cpp
    samples/gb1.wav
    tests/parakeet-expected-gb1-output.txt)

add_parakeet_transcription_test(
    test-parakeet-full-diffusion
    test-parakeet-full.cpp
    samples/diffusion2023-07-03.flac
    tests/parakeet-expected-diffusion-output.txt
    0.95)
endif() # WHISPER_BUILD_PARAKEET
