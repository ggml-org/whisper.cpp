cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(chatterbox_hexagon_benchmark LANGUAGES CXX)
    include(CTest)
endif()

# Installed public headers/targets only. Point CMAKE_PREFIX_PATH (or Android's
# CMAKE_FIND_ROOT_PATH) at the matching Speech and shared GGML installations.
if(NOT TARGET tts-cpp::tts-cpp)
    find_package(tts-cpp CONFIG REQUIRED)
endif()
add_executable(chatterbox-hexagon ../chatterbox-hexagon.cpp)
target_compile_features(chatterbox-hexagon PRIVATE cxx_std_17)
target_link_libraries(chatterbox-hexagon PRIVATE tts-cpp::tts-cpp)
if(MSVC)
    target_compile_options(chatterbox-hexagon PRIVATE /W4)
else()
    target_compile_options(chatterbox-hexagon PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Build the test double through CMake's compiler/toolchain, so Python does not
# assume GCC flags or try to compile an Android binary with the host compiler.
# These native-only contracts test this measurement tool, not inference math.
if((BUILD_TESTING OR TTS_CPP_BUILD_TESTS) AND NOT CMAKE_CROSSCOMPILING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        find_package(Threads REQUIRED)
        add_executable(chatterbox-hexagon-contract-stub
            ../chatterbox-hexagon.cpp contract-engine.cpp)
        target_compile_features(chatterbox-hexagon-contract-stub PRIVATE cxx_std_17)
        target_include_directories(chatterbox-hexagon-contract-stub PRIVATE
            $<TARGET_PROPERTY:tts-cpp::tts-cpp,INTERFACE_INCLUDE_DIRECTORIES>)
        target_link_libraries(chatterbox-hexagon-contract-stub PRIVATE Threads::Threads)
        add_test(NAME test-chatterbox-measurement-driver
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-driver.py
                --binary $<TARGET_FILE:chatterbox-hexagon>
                --stub-binary $<TARGET_FILE:chatterbox-hexagon-contract-stub>)
        set_tests_properties(test-chatterbox-measurement-driver PROPERTIES
            LABELS "unit" TIMEOUT 120)

        set(_chatterbox_fit_test_args)
        if(TARGET test-chatterbox-fit-params)
            list(APPEND _chatterbox_fit_test_args
                --fit-binary $<TARGET_FILE:test-chatterbox-fit-params>)
        endif()
        add_test(NAME test-chatterbox-collection
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../test-chatterbox-hexagon-validation.py
                ${_chatterbox_fit_test_args})
        set_tests_properties(test-chatterbox-collection PROPERTIES
            LABELS "unit" TIMEOUT 120)
    else()
        message(STATUS "Chatterbox measurement-driver contracts not registered: Python3 is unavailable")
    endif()
endif()
