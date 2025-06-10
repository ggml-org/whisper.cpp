cmake_minimum_required(VERSION 3.10)

project(whisper.cpp)

set(CMAKE_CXX_STANDARD 17)
set(WHISPER_LIB_DIR ${CMAKE_SOURCE_DIR}/../../../../../../..)

# Path to external GGML, otherwise uses the copy in whisper.cpp.
option(GGML_HOME "whisper: Path to external GGML source" OFF)

set(
    SOURCE_FILES
    ${WHISPER_LIB_DIR}/src/whisper.cpp
    ${CMAKE_SOURCE_DIR}/jni.c
    )

# TODO: this needs to be updated to work with the new ggml CMakeLists

if (NOT GGML_HOME)
    set(
        SOURCE_FILES
        ${SOURCE_FILES}
        ${WHISPER_LIB_DIR}/ggml/src/ggml.c
        ${WHISPER_LIB_DIR}/ggml/src/ggml-alloc.c
        ${WHISPER_LIB_DIR}/ggml/src/ggml-backend.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-backend-reg.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-quants.c
        ${WHISPER_LIB_DIR}/ggml/src/ggml-threading.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/ggml-cpu.c
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/ggml-cpu.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/repack.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/hbm.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/quants.c
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/traits.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/unary-ops.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/binary-ops.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/vec.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/ops.cpp
        )
endif()

find_library(LOG_LIB log)

function(build_library target_name)
    add_library(
        ${target_name}
        SHARED
        ${SOURCE_FILES}
    )

    target_compile_definitions(${target_name} PUBLIC GGML_USE_CPU)

    if (${target_name} STREQUAL "whisper_v8fp16_va")
        target_compile_options(${target_name} PRIVATE -march=armv8.2-a+fp16)
        set(GGML_COMPILE_OPTIONS                      -march=armv8.2-a+fp16)
    elseif (${target_name} STREQUAL "whisper_vfpv4")
        target_compile_options(${target_name} PRIVATE -mfpu=neon-vfpv4)
        set(GGML_COMPILE_OPTIONS                      -mfpu=neon-vfpv4)
    endif ()

    if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE -O3)
        target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
        target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)

        target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
        target_link_options(${target_name} PRIVATE -Wl,--exclude-libs,ALL)
        target_link_options(${target_name} PRIVATE -flto)
    endif ()

    if (GGML_HOME)
        include(FetchContent)
        FetchContent_Declare(ggml SOURCE_DIR ${GGML_HOME})
        FetchContent_MakeAvailable(ggml)

        target_compile_options(ggml PRIVATE ${GGML_COMPILE_OPTIONS})
        target_link_libraries(${target_name} ${LOG_LIB} android ggml)
    else()
        target_link_libraries(${target_name} ${LOG_LIB} android)
    endif()


endfunction()

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    build_library("whisper_v8fp16_va")
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    build_library("whisper_vfpv4")
endif ()

build_library("whisper") # Default target

include_directories(${WHISPER_LIB_DIR})
include_directories(${WHISPER_LIB_DIR}/src)
include_directories(${WHISPER_LIB_DIR}/include)
include_directories(${WHISPER_LIB_DIR}/ggml/include)
include_directories(${WHISPER_LIB_DIR}/ggml/src)
include_directories(${WHISPER_LIB_DIR}/ggml/src/ggml-cpu)

