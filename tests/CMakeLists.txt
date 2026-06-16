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
target_link_libraries(${UTF8_TEST} PRIVATE common)
add_test(NAME ${UTF8_TEST} COMMAND ${UTF8_TEST})
set_tests_properties(${UTF8_TEST} PROPERTIES LABELS "unit")

# VAD test tests VAD in isolation
set(VAD_TEST test-vad)
add_executable(${VAD_TEST} ${VAD_TEST}.cpp)
target_include_directories(${VAD_TEST} PRIVATE ../include ../ggml/include ../examples)
target_link_libraries(${VAD_TEST} PRIVATE common)
target_compile_definitions(${VAD_TEST} PRIVATE
    VAD_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin"
    SAMPLE_PATH="${PROJECT_SOURCE_DIR}/samples/jfk.wav")
add_test(NAME ${VAD_TEST} COMMAND ${VAD_TEST})
set_tests_properties(${VAD_TEST} PROPERTIES LABELS "unit")

# VAD test full uses whisper_full with VAD enabled
set(VAD_TEST test-vad-full)
add_executable(${VAD_TEST} ${VAD_TEST}.cpp)
target_include_directories(${VAD_TEST} PRIVATE ../include ../ggml/include ../examples)
target_link_libraries(${VAD_TEST} PRIVATE common)
target_compile_definitions(${VAD_TEST} PRIVATE
    WHISPER_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/ggml-base.en.bin"
    VAD_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-silero-v6.2.0-ggml.bin"
    SAMPLE_PATH="${PROJECT_SOURCE_DIR}/samples/jfk.wav")
add_test(NAME ${VAD_TEST} COMMAND ${VAD_TEST})
set_tests_properties(${VAD_TEST} PROPERTIES LABELS "base;en")

# Parakeet model loading test
set(PARAKEET_TEST test-parakeet)
add_executable(${PARAKEET_TEST} ${PARAKEET_TEST}.cpp)
target_include_directories(${PARAKEET_TEST} PRIVATE ../include ../ggml/include ../examples)
target_link_libraries(${PARAKEET_TEST} PRIVATE parakeet common)
target_compile_definitions(${PARAKEET_TEST} PRIVATE
    PARAKEET_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/for-tests-ggml-parakeet-tdt.bin"
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
    target_include_directories(${TEST_TARGET} PRIVATE ../include ../ggml/include ../examples)
    target_link_libraries(${TEST_TARGET} PRIVATE parakeet common)
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

