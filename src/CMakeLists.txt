# TODO: should not use this
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()


if (WHISPER_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

if (WHISPER_COREML)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREML_FRAMEWORK CoreML)

    if (COREML_FRAMEWORK)
        message(STATUS "CoreML framework found")

        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_USE_COREML)
    else()
        message(FATAL_ERROR "CoreML framework not found")
    endif()

    if (WHISPER_COREML_ALLOW_FALLBACK)
        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_COREML_ALLOW_FALLBACK)
    endif()
endif()

if (WHISPER_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()

if (WHISPER_VITISAI)
    find_package(FlexmlRT REQUIRED)

    # Legacy RAI overrides are required by FlexMLRT older than 1.8.0
    set(WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES_MODE "AUTO" CACHE STRING
        "Legacy RAI override mode for FlexMLRT (AUTO|ON|OFF)")
    set_property(CACHE WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES_MODE PROPERTY STRINGS AUTO ON OFF)

    string(TOUPPER "${WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES_MODE}" _flexmlrt_legacy_mode)
    set(_flexmlrt_legacy_hint "Set -DWHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES_MODE=ON or OFF explicitly.")

    if (NOT _flexmlrt_legacy_mode MATCHES "^(AUTO|ON|OFF)$")
        message(FATAL_ERROR
            "Invalid WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES_MODE='${WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES_MODE}'. "
            "Expected AUTO, ON, or OFF.")
    endif()

    if (_flexmlrt_legacy_mode STREQUAL "AUTO")
        if (NOT FlexmlRT_DIR)
            message(FATAL_ERROR
                "FlexmlRT_DIR is unset after find_package(FlexmlRT). ${_flexmlrt_legacy_hint}")
        endif()

        # FlexmlRT_DIR points to <pkg_root>/share/cmake/FlexmlRT.
        get_filename_component(_flexmlrt_init_py "${FlexmlRT_DIR}/../../../__init__.py" ABSOLUTE)
        if (NOT EXISTS "${_flexmlrt_init_py}")
            message(FATAL_ERROR
                "flexmlrt __init__.py not found at ${_flexmlrt_init_py}. ${_flexmlrt_legacy_hint}")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_flexmlrt_init_py}")

        file(STRINGS "${_flexmlrt_init_py}" _flexmlrt_version_lines
             REGEX "^VERSION[ \t]*=[ \t]*\"[0-9]+\\.[0-9]+\\.[0-9]+")
        if (NOT _flexmlrt_version_lines MATCHES "\"([0-9]+\\.[0-9]+\\.[0-9]+)")
            message(FATAL_ERROR
                "Could not parse flexmlrt VERSION from ${_flexmlrt_init_py}. ${_flexmlrt_legacy_hint}")
        endif()
        set(_flexmlrt_version "${CMAKE_MATCH_1}")

        if (_flexmlrt_version VERSION_LESS "1.8.0")
            set(WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES 1)
        else()
            set(WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES 0)
        endif()
        message(STATUS "Detected flexmlrt VERSION=${_flexmlrt_version} from ${_flexmlrt_init_py} (legacy overrides=${WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES})")
    else()
        if (_flexmlrt_legacy_mode STREQUAL "ON")
            set(WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES 1)
        else()
            set(WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES 0)
        endif()
        message(STATUS "FlexMLRT legacy RAI overrides forced ${_flexmlrt_legacy_mode}")
    endif()
endif()

#
# libraries
#

# whisper.coreml

if (WHISPER_COREML)
    set(TARGET whisper.coreml)

    add_library(${TARGET}
        coreml/whisper-compat.m
        coreml/whisper-encoder.h
        coreml/whisper-encoder.mm
        coreml/whisper-encoder-impl.h
        coreml/whisper-encoder-impl.m
        )

    include(DefaultTargetOptions)

    target_include_directories(${TARGET} PUBLIC
        .
        )

    target_link_libraries(${TARGET} PRIVATE ${FOUNDATION_FRAMEWORK} ${COREML_FRAMEWORK})

    set_target_properties(${TARGET} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
        )

    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
    install(TARGETS ${TARGET} LIBRARY)
endif()

if (WHISPER_OPENVINO)
    set(TARGET whisper.openvino)

    add_library(${TARGET} OBJECT
        openvino/whisper-openvino-encoder.h
        openvino/whisper-openvino-encoder.cpp
        )

    target_include_directories(${TARGET} PUBLIC
        .
        )

    set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_USE_OPENVINO)

    target_link_libraries(${TARGET} PRIVATE ggml openvino::runtime)
    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

if (WHISPER_VITISAI)
    set(TARGET whisper.vitisai)

    add_library(${TARGET} OBJECT
        vitisai/whisper-vitisai-helpers.h
        vitisai/whisper-vitisai-helpers.cpp
        vitisai/whisper-vitisai-encoder.h
        vitisai/whisper-vitisai-encoder.cpp
        )

    target_include_directories(${TARGET} PUBLIC .)

    set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_USE_VITISAI)

    # FlexMLRT headers and this plugin require C++17. Keep it PRIVATE so the
    # C++11 requirement of the whisper target is not bumped.
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    target_compile_definitions(${TARGET} PRIVATE
        WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES=${WHISPER_FLEXMLRT_LEGACY_RAI_OVERRIDES}
    )

    target_link_libraries(${TARGET} PRIVATE ggml flexmlrt::flexmlrt)
    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

# whisper

add_library(whisper
            ../include/whisper.h
            whisper-arch.h
            whisper.cpp
            )

add_library(parakeet
            ../include/parakeet.h
            parakeet-arch.h
            parakeet.cpp
            )

target_include_directories(parakeet PUBLIC . ../include)
target_compile_features   (parakeet PUBLIC cxx_std_11)
target_link_libraries(parakeet PUBLIC ggml Threads::Threads)

# Set the version numbers
set_target_properties(whisper PROPERTIES
    VERSION ${WHISPER_VERSION_BASE}
    SOVERSION ${WHISPER_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

set_target_properties(parakeet PROPERTIES
    VERSION ${WHISPER_VERSION_BASE}
    SOVERSION ${WHISPER_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(whisper PUBLIC . ../include)
target_compile_features   (whisper PUBLIC cxx_std_11) # don't bump

if (CMAKE_CXX_BYTE_ORDER STREQUAL "BIG_ENDIAN")
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_BIG_ENDIAN)
    set(PARAKEET_EXTRA_FLAGS ${PARAKEET_EXTRA_FLAGS} -DPARAKEET_BIG_ENDIAN)
endif()

if (WHISPER_EXTRA_FLAGS)
    target_compile_options(whisper PRIVATE ${WHISPER_EXTRA_FLAGS})
endif()

if (PARAKEET_EXTRA_FLAGS)
    target_compile_options(parakeet PRIVATE ${PARAKEET_EXTRA_FLAGS})
endif()

find_package(Threads REQUIRED)
target_link_libraries(whisper PUBLIC ggml Threads::Threads)

if (WHISPER_COREML)
    target_link_libraries(whisper PRIVATE whisper.coreml)
endif()

if (WHISPER_OPENVINO)
    target_link_libraries(whisper PRIVATE whisper.openvino)
endif()

if (WHISPER_VITISAI)
    target_link_libraries(whisper PRIVATE whisper.vitisai)
endif()

if (WHISPER_MKL)
    target_link_libraries(whisper PRIVATE MKL::MKL)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(whisper PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(whisper PRIVATE WHISPER_SHARED WHISPER_BUILD)

    set_target_properties(parakeet PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(parakeet PRIVATE PARAKEET_SHARED PARAKEET_BUILD)
endif()
