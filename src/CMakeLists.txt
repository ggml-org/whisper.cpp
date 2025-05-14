cmake_minimum_required(VERSION 3.16)

project(whisper)

# Add whisper.cpp root directory to the include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# --- New Option for Dynamic Backend Loading ---
option(WHISPER_BACKEND_DL "Build backends (OpenVINO, Metal/CoreML) as shared libraries" OFF)
# -------------------------------------------

# Find liburing only if explicitly requested (used in main)
option(WHISPER_BUILD_LIBURING "Build with io_uring support" OFF)
if (WHISPER_BUILD_LIBURING)
    find_package(liburing QUIET)
    if (liburing_FOUND)
        message(STATUS "Found liburing: ${liburing_LIBRARIES}")
        include_directories(${liburing_INCLUDE_DIRS})
        list(APPEND WHISPER_LINK_LIBS ${liburing_LIBRARIES})
        add_definitions(-DWHISPER_USE_LIBURING)
    else()
        message(WARNING "liburing not found. io_uring support disabled.")
        option(WHISPER_BUILD_LIBURING "Build with io_uring support" OFF) # Disable the option if liburing is not found
    endif()
endif()

# Add command-line example sources only if explicitly requested
option(WHISPER_BUILD_EXAMPLES "Build whisper.cpp examples" OFF)
if (WHISPER_BUILD_EXAMPLES)
    # These sources are not part of the library target itself
    add_subdirectory(examples)
endif()

# Add tests if explicitly requested
option(WHISPER_BUILD_TESTS "Build whisper.cpp tests" OFF)
if (WHISPER_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# --- Backend Configuration ---
# OpenVINO Backend
option(WHISPER_BUILD_OPENVINO "Build with OpenVINO backend" OFF)
if (WHISPER_BUILD_OPENVINO)
    message(STATUS "Building with OpenVINO backend")
    find_package(OpenVINO REQUIRED)
    include_directories(${OpenVINO_INCLUDE_DIRS})

    # --- Conditional compilation for OpenVINO as shared lib ---
    if (WHISPER_BACKEND_DL)
        # Compile OpenVINO backend into a separate shared library
        add_library(whisper_openvino_backend SHARED whisper-openvino.cpp)
        target_link_libraries(whisper_openvino_backend PRIVATE ${OpenVINO_LIBRARIES})
        target_include_directories(whisper_openvino_backend PRIVATE ${OpenVINO_INCLUDE_DIRS})
        target_compile_definitions(whisper_openvino_backend PRIVATE -DGGML_USE_OPENVINO)
        # Add a define to the main whisper library indicating dynamic loading
        add_definitions(-DWHISPER_USE_OPENVINO_DL)

        # The whisper-openvino.cpp source is *not* added to the main whisper library
        set(WHISPER_OPENVINO_SRC "")
        set(WHISPER_OPENVINO_LINK_LIB whisper_openvino_backend)

    else()
        # Compile OpenVINO backend into the main whisper library (static linking)
        list(APPEND WHISPER_SRC whisper-openvino.cpp)
        list(APPEND WHISPER_LINK_LIBS ${OpenVINO_LIBRARIES})
        add_definitions(-DGGML_USE_OPENVINO)

        set(WHISPER_OPENVINO_SRC whisper-openvino.cpp)
        set(WHISPER_OPENVINO_LINK_LIB ${OpenVINO_LIBRARIES})
    endif()
    # ---------------------------------------------------------
endif()

# CoreML Backend (implies Metal backend on Apple)
# The CoreML backend is built into ggml-metal.m
option(WHISPER_COREML "Build with CoreML backend (Apple only)" OFF)
if (WHISPER_COREML AND APPLE)
    message(STATUS "Building with CoreML backend")
    add_definitions(-DGGML_USE_COREML)

    # --- Conditional compilation for Metal/CoreML as shared lib ---
    if (WHISPER_BACKEND_DL)
        # Build ggml as a shared library when CoreML/Metal is enabled and dynamic linking is requested
        set(GGML_LIB_TYPE SHARED)
        # Add a define to the main whisper library indicating dynamic loading
        add_definitions(-DWHISPER_USE_METAL_DL)
        message(STATUS "Building ggml library as SHARED for CoreML/Metal dynamic loading")
    else()
        # Build ggml as a static library (default behavior)
        set(GGML_LIB_TYPE STATIC)
        message(STATUS "Building ggml library as STATIC for CoreML/Metal static linking")
    endif()
    # --------------------------------------------------------------

else()
    # Build ggml as a static library if CoreML is not enabled or not on Apple
    set(GGML_LIB_TYPE STATIC)
    message(STATUS "Building ggml library as STATIC (CoreML/Metal disabled or not dynamic)")
endif()
# --- End Backend Configuration ---


# Common sources for whisper and ggml
set(COMMON_SRC
    whisper.cpp
    ggml.c
    ggml-alloc.c
    ggml-backend.c
    ggml-context.c
    ggml-cuda.cu # Placeholder, only compiled if CUDA is enabled
    ggml-impl.h
    ggml-kompute.cpp # Placeholder, only compiled if KOMPUTE is enabled
    ggml-metal.m   # Placeholder, only compiled if METAL is enabled
    ggml-opencl.c  # Placeholder, only compiled if OpenCL is enabled
    ggml-vulkan.cpp # Placeholder, only compiled if VULKAN is enabled
)

# Determine which GGML backend sources to include
set(GGML_BACKEND_SRC "")

# --- GGML BACKENDS ---
# CUDA Backend
option(WHISPER_BUILD_CUDA "Build with CUDA backend" OFF)
if (WHISPER_BUILD_CUDA)
    if (NOT CMAKE_CUDA_COMPILER)
        find_package(CUDA REQUIRED)
    endif()

    enable_language(CUDA)
    list(APPEND GGML_BACKEND_SRC ggml-cuda.cu)
    add_definitions(-DGGML_USE_CUDA)
    add_definitions(-DGGML_CUDA_DMMV_X=32)
    add_definitions(-DGGML_CUDA_DMMV_Y=1)
    add_definitions(-DGGML_CUDA_MMV_X=32)
    add_definitions(-DGGML_CUDA_MMV_Y=1)
    add_definitions(-DGGML_CUDA_MMQ_X=32)
    add_definitions(-DGGML_CUDA_MMQ_Y=1)
    add_definitions(-DGGML_CUDA_K_QUANTS_PER_ITERATION=1)
    add_definitions(-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128)
    add_definitions(-DGGML_CUDA_MAX_BATCH_SIZE=512)
    add_definitions(-DGGML_CUDA_MAX_DEVICES=8)

    # Add architecture flags
    set(CUDA_ARCH_FLAGS "")
    if (NOT CUDA_ARCHITECTURES)
        message(STATUS "CUDA_ARCHITECTURES is not set, compiling for all detected architectures.")
        message(STATUS "If this is not desired, set CUDA_ARCHITECTURES to e.g. '60;70;75;80;86;89;90'.")
        list(APPEND CUDA_ARCH_FLAGS "-arch=all-major")
    else()
        foreach(arch ${CUDA_ARCHITECTURES})
            list(APPEND CUDA_ARCH_FLAGS "-arch=sm_${arch}")
        endforeach()
    endif()

    target_compile_options(ggml PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_ARCH_FLAGS}>)
endif()

# KOMPUTE Backend
option(WHISPER_BUILD_KOMPUTE "Build with Kompute backend" OFF)
if (WHISPER_BUILD_KOMPUTE)
    find_package(Kompute REQUIRED)
    list(APPEND GGML_BACKEND_SRC ggml-kompute.cpp)
    add_definitions(-DGGML_USE_KOMPUTE)
    list(APPEND GGML_BACKEND_LINK_LIBS Kompute::Kompute)
endif()

# Metal Backend (Apple only)
# Handled by WHISPER_COREML option and GGML_LIB_TYPE logic above
if (WHISPER_COREML AND APPLE)
    list(APPEND GGML_BACKEND_SRC ggml-metal.m)
    add_definitions(-DGGML_USE_METAL)
    # Metal framework needed for linking the library that uses it
    list(APPEND GGML_BACKEND_LINK_LIBS "-framework" "Metal" "-framework" "Foundation" "-framework" "CoreGraphics")
endif()

# OpenCL Backend
option(WHISPER_BUILD_OPENCL "Build with OpenCL backend" OFF)
if (WHISPER_BUILD_OPENCL)
    find_package(OpenCL REQUIRED)
    list(APPEND GGML_BACKEND_SRC ggml-opencl.c)
    add_definitions(-DGGML_USE_OPENCL)
    list(APPEND GGML_BACKEND_LINK_LIBS OpenCL::OpenCL)
endif()

# Vulkan Backend
option(WHISPER_BUILD_VULKAN "Build with Vulkan backend" OFF)
if (WHISPER_BUILD_VULKAN)
    find_package(Vulkan REQUIRED)
    list(APPEND GGML_BACKEND_SRC ggml-vulkan.cpp)
    add_definitions(-DGGML_USE_VULKAN)
    list(APPEND GGML_BACKEND_LINK_LIBS Vulkan::Vulkan)
endif()
# --- END GGML BACKENDS ---


# --- GGML Library Definition ---
# Define the ggml library with selected backends and determined type (STATIC/SHARED)
add_library(ggml ${GGML_LIB_TYPE}
    ${COMMON_SRC}
    ${GGML_BACKEND_SRC}
)

# Link backend specific libraries to ggml
target_link_libraries(ggml PRIVATE ${GGML_BACKEND_LINK_LIBS})

# Setup include directories for ggml
target_include_directories(ggml PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)

# Common compile definitions for ggml
target_compile_definitions(ggml PRIVATE
    -D_CRT_SECURE_NO_DEPRECATE
    -D_CRT_NONSTDC_NO_DEPRECATE
    -DGGML_SHARED
    -DGGML_BUILD=1
)

# Platform specific compile definitions for ggml
if (MSVC)
    target_compile_definitions(ggml PRIVATE -D_WIN32_WINNT=0x0601) # for winapi
endif()

if (APPLE)
    target_compile_options(ggml PRIVATE -Wno-unguarded-availability-new)
    target_link_options(ggml PRIVATE -Wl,-allow_multiple_dylibs) # Allow multiple copies of ggml/whisper dylib
endif()
# --- End GGML Library Definition ---


# --- WHISPER Library Definition ---
# Sources for the main whisper library.
# Note: whisper-openvino.cpp is added only if WHISPER_BACKEND_DL is OFF.
set(WHISPER_SRC
    whisper.cpp
    ${WHISPER_OPENVINO_SRC} # Added here if compiled statically
)

# Define the main whisper library
# It will be static or shared based on CMAKE_CXX_STANDARD_LIBRARIES implicit link type
add_library(whisper ${WHISPER_SRC})

# Link libraries to whisper
# Links ggml, OpenVINO (statically or dynamically), and other common libs
target_link_libraries(whisper
    PUBLIC
        ggml # Always links ggml
        ${WHISPER_OPENVINO_LINK_LIB} # Links OpenVINO backend library/libs (dynamic or static)
        ${WHISPER_LINK_LIBS} # Includes liburing if found
)

# Setup include directories for whisper
target_include_directories(whisper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_BINARY_DIR} # For ggml-opencl.h etc
)

# Common compile definitions for whisper
target_compile_definitions(whisper PRIVATE
    -D_CRT_SECURE_NO_DEPRECATE
    -D_CRT_NONSTDC_NO_DEPRECATE
    -DWHISPER_SHARED
    -DWHISPER_BUILD=1
)

# Platform specific compile definitions for whisper
if (MSVC)
    target_compile_definitions(whisper PRIVATE -D_WIN32_WINNT=0x0601) # for winapi
endif()

if (APPLE)
    target_compile_options(whisper PRIVATE -Wno-unguarded-availability-new)
    target_link_options(whisper PRIVATE -Wl,-allow_multiple_dylibs) # Allow multiple copies of ggml/whisper dylib
endif()
# --- End WHISPER Library Definition ---

# Install targets
install(TARGETS whisper ggml
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install whisper.h header
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../whisper.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

