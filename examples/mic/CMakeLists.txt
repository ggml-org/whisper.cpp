cmake_minimum_required(VERSION 3.10)
project(whisper-mic)

add_executable(whisper-mic whisper-mic.cpp)

target_include_directories(whisper-mic PRIVATE
    ${CMAKE_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/../..
    ${CMAKE_SOURCE_DIR}/../../include
    ${CMAKE_SOURCE_DIR}/../../ggml/include
)

target_link_libraries(whisper-mic PRIVATE whisper pthread dl m rt)

set_target_properties(whisper-mic PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
