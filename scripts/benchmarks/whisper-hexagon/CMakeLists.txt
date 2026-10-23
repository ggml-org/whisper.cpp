cmake_minimum_required(VERSION 3.16)
project(whisper_hexagon_benchmark LANGUAGES CXX)

# Standalone installed-library consumer: no source-tree headers or libraries.
# Set CMAKE_PREFIX_PATH to the matching Whisper and shared GGML install prefixes.
find_package(whisper CONFIG REQUIRED)
add_executable(whisper-hexagon ../whisper-hexagon.cpp)
target_compile_features(whisper-hexagon PRIVATE cxx_std_11)
target_link_libraries(whisper-hexagon PRIVATE whisper::whisper)
if(MSVC)
    target_compile_options(whisper-hexagon PRIVATE /W4)
else()
    target_compile_options(whisper-hexagon PRIVATE -Wall -Wextra -Wpedantic)
endif()
