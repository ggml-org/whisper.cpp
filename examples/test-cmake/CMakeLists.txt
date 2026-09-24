cmake_minimum_required(VERSION 3.14)
project(whisper-simple)

set(CMAKE_CXX_STANDARD 17)

# request a version that is satisfied by both dev (1.9.2-dev) and release (1.9.2) installs
# (find_package does not accept a pre-release version as an argument)
find_package(whisper 1.9.2 REQUIRED)

add_executable(test-cmake test-cmake.cpp)

target_link_libraries(test-cmake PRIVATE whisper)

target_compile_definitions(test-cmake PRIVATE
    WHISPER_BUILD_NUMBER=${WHISPER_BUILD_NUMBER}
    WHISPER_BUILD_COMMIT="${WHISPER_BUILD_COMMIT}"
)
