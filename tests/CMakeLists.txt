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

if (WHISPER_FFMPEG)
    set(TEST_TARGET test-whisper-cli-tiny-mp3)
    # Check with reviewers: any way to check the output transcription via ctest (diff, ...)?
    add_test(NAME ${TEST_TARGET}
      COMMAND $<TARGET_FILE:whisper-cli>
      -m ${PROJECT_SOURCE_DIR}/models/for-tests-ggml-tiny.en.bin
      -f ${PROJECT_SOURCE_DIR}/samples/jfk.mp3)
    set_tests_properties(${TEST_TARGET} PROPERTIES LABELS "tiny;mp3")
endif()

# VAD test tests VAD in isolation
set(VAD_TEST test-vad)
add_executable(${VAD_TEST} ${VAD_TEST}.cpp)
target_include_directories(${VAD_TEST} PRIVATE ../include ../ggml/include ../examples)
target_link_libraries(${VAD_TEST} PRIVATE common)
add_test(NAME ${VAD_TEST} COMMAND ${VAD_TEST})
set_tests_properties(${VAD_TEST} PROPERTIES LABELS "unit")

# VAD test full uses whisper_full with VAD enabled
set(VAD_TEST test-vad-full)
add_executable(${VAD_TEST} ${VAD_TEST}.cpp)
target_include_directories(${VAD_TEST} PRIVATE ../include ../ggml/include ../examples)
target_link_libraries(${VAD_TEST} PRIVATE common)
add_test(NAME ${VAD_TEST} COMMAND ${VAD_TEST})
set_tests_properties(${VAD_TARGET} PROPERTIES LABELS "base;en")
