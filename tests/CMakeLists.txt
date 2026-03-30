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

# Speaker embedding inference test
add_executable(test-speaker-embedding test-speaker-embedding.cpp ../src/whisper-diarize.cpp ../src/whisper-speaker.cpp)
target_include_directories(test-speaker-embedding PRIVATE ../include ../ggml/include)
target_link_libraries(test-speaker-embedding PRIVATE ggml m)

# Embedding quality (speaker discrimination)
add_executable(test-embedding-quality test-embedding-quality.cpp ../src/whisper-diarize.cpp ../src/whisper-speaker.cpp)
target_include_directories(test-embedding-quality PRIVATE ../include ../ggml/include ../src)
target_link_libraries(test-embedding-quality PRIVATE ggml m)

# Clustering algorithm test
add_executable(test-clustering test-clustering.cpp ../src/whisper-diarize.cpp ../src/whisper-speaker.cpp)
target_include_directories(test-clustering PRIVATE ../include ../ggml/include)
target_link_libraries(test-clustering PRIVATE ggml m)
target_compile_options(test-clustering PRIVATE -std=c++11 -Wall -Wextra)

# Diarization integration test
add_executable(test-diarization-integration test-diarization-integration.cpp)
target_include_directories(test-diarization-integration PRIVATE ../include ../ggml/include)
target_link_libraries(test-diarization-integration PRIVATE whisper)

# CLI diarization end-to-end test
add_executable(test-cli-diarization test-cli-diarization.cpp)
target_include_directories(test-cli-diarization PRIVATE ../include ../examples)
target_link_libraries(test-cli-diarization PRIVATE whisper common)
target_compile_options(test-cli-diarization PRIVATE -std=c++17)

# DER benchmark
add_executable(benchmark-der benchmark-der.cpp)
target_include_directories(benchmark-der PRIVATE ../include ../ggml/include)
target_link_libraries(benchmark-der PRIVATE ggml m)
target_compile_options(benchmark-der PRIVATE -std=c++17)

# AddressSanitizer support
if (WHISPER_SANITIZE_ADDRESS)
    message(STATUS "Enabling AddressSanitizer for benchmark-der")
    target_compile_options(benchmark-der PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(benchmark-der PRIVATE -fsanitize=address)
endif()

# Register tests
enable_testing()
add_test(NAME speaker-embedding COMMAND test-speaker-embedding WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME clustering COMMAND test-clustering WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME diarization-integration COMMAND test-diarization-integration WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME cli-diarization COMMAND test-cli-diarization WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Register benchmark-der test
if (WHISPER_DIARIZE)
    add_test(NAME benchmark-der COMMAND benchmark-der WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(benchmark-der PROPERTIES LABELS "diarization;benchmark")

    # ASan leak detection (Linux only)
    if (WHISPER_SANITIZE_ADDRESS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set_tests_properties(benchmark-der PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1:strip_path_prefix=${CMAKE_SOURCE_DIR}/")
    endif()
endif()

# Test: Reference embedding comparison (feed PyTorch mel → compare embeddings)
add_executable(test-ref-compare test-ref-compare.cpp ../src/whisper-diarize.cpp ../src/whisper-speaker.cpp)
target_include_directories(test-ref-compare PRIVATE ../include ../ggml/include ../src)
target_link_libraries(test-ref-compare PRIVATE ggml m)

# Test: Mel feature comparison
add_executable(test-mel-compare test-mel-compare.cpp ../src/whisper-diarize.cpp ../src/whisper-speaker.cpp)
target_include_directories(test-mel-compare PRIVATE ../include ../ggml/include ../src)
target_link_libraries(test-mel-compare PRIVATE ggml m)
