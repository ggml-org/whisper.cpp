# SDL2 CMake configuration file:
# This file is meant to be placed in a cmake subfolder of SDL2-devel-2.x.y-VC

cmake_minimum_required(VERSION 3.0...3.28)

include(FeatureSummary)
set_package_properties(SDL2 PROPERTIES
    URL "https://www.libsdl.org/"
    DESCRIPTION "low level access to audio, keyboard, mouse, joystick, and graphics hardware"
)

# Copied from `configure_package_config_file`
macro(set_and_check _var _file)
    set(${_var} "${_file}")
    if(NOT EXISTS "${_file}")
        message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
    endif()
endmacro()

# Copied from `configure_package_config_file`
macro(check_required_components _NAME)
    foreach(comp ${${_NAME}_FIND_COMPONENTS})
        if(NOT ${_NAME}_${comp}_FOUND)
            if(${_NAME}_FIND_REQUIRED_${comp})
                set(${_NAME}_FOUND FALSE)
            endif()
        endif()
    endforeach()
endmacro()

set(SDL2_FOUND TRUE)

if(CMAKE_SIZEOF_VOID_P STREQUAL "4")
    set(_sdl_arch_subdir "x86")
elseif(CMAKE_SIZEOF_VOID_P STREQUAL "8")
    set(_sdl_arch_subdir "x64")
else()
    set(SDL2_FOUND FALSE)
    return()
endif()

# For compatibility with autotools sdl2-config.cmake, provide SDL2_* variables.

set_and_check(SDL2_PREFIX       "${CMAKE_CURRENT_LIST_DIR}/..")
set_and_check(SDL2_EXEC_PREFIX  "${CMAKE_CURRENT_LIST_DIR}/..")
set_and_check(SDL2_INCLUDE_DIR  "${SDL2_PREFIX}/include")
set(SDL2_INCLUDE_DIRS           "${SDL2_INCLUDE_DIR}")
set_and_check(SDL2_BINDIR       "${SDL2_PREFIX}/lib/${_sdl_arch_subdir}")
set_and_check(SDL2_LIBDIR       "${SDL2_PREFIX}/lib/${_sdl_arch_subdir}")

set(SDL2_LIBRARIES      SDL2::SDL2main SDL2::SDL2)
set(SDL2MAIN_LIBRARY    SDL2::SDL2main)
set(SDL2TEST_LIBRARY    SDL2::SDL2test)


# All targets are created, even when some might not be requested though COMPONENTS.
# This is done for compatibility with CMake generated SDL2-target.cmake files.

set(_sdl2_library     "${SDL2_LIBDIR}/SDL2.lib")
set(_sdl2_dll_library "${SDL2_BINDIR}/SDL2.dll")
if(EXISTS "${_sdl2_library}" AND EXISTS "${_sdl2_dll_library}")
    if(NOT TARGET SDL2::SDL2)
        add_library(SDL2::SDL2 SHARED IMPORTED)
        set_target_properties(SDL2::SDL2
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${SDL2_INCLUDE_DIRS}"
                IMPORTED_IMPLIB "${_sdl2_library}"
                IMPORTED_LOCATION "${_sdl2_dll_library}"
                COMPATIBLE_INTERFACE_BOOL "SDL2_SHARED"
                INTERFACE_SDL2_SHARED "ON"
                COMPATIBLE_INTERFACE_STRING "SDL_VERSION"
                INTERFACE_SDL_VERSION "SDL2"
        )
    endif()
    set(SDL2_SDL2_FOUND TRUE)
else()
    set(SDL2_SDL2_FOUND FALSE)
endif()
unset(_sdl2_library)
unset(_sdl2_dll_library)

set(SDL2_SDL2-static_FOUND FALSE)

set(_sdl2main_library "${SDL2_LIBDIR}/SDL2main.lib")
if(EXISTS "${_sdl2main_library}")
    if(NOT TARGET SDL2::SDL2main)
        add_library(SDL2::SDL2main STATIC IMPORTED)
        set_target_properties(SDL2::SDL2main
        PROPERTIES
            IMPORTED_LOCATION "${_sdl2main_library}"
            COMPATIBLE_INTERFACE_STRING "SDL_VERSION"
            INTERFACE_SDL_VERSION "SDL2"
        )
    endif()
    set(SDL2_SDL2main_FOUND TRUE)
else()
    set(SDL2_SDL2main_FOUND FALSE)
endif()
unset(_sdl2main_library)

set(_sdl2test_library "${SDL2_LIBDIR}/SDL2test.lib")
if(EXISTS "${_sdl2test_library}")
    if(NOT TARGET SDL2::SDL2test)
        add_library(SDL2::SDL2test STATIC IMPORTED)
        set_target_properties(SDL2::SDL2test
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${SDL2_INCLUDE_DIRS}"
                IMPORTED_LOCATION "${_sdl2test_library}"
                COMPATIBLE_INTERFACE_STRING "SDL_VERSION"
                INTERFACE_SDL_VERSION "SDL2"
        )
    endif()
    set(SDL2_SDL2test_FOUND TRUE)
else()
    set(SDL2_SDL2test_FOUND FALSE)
endif()
unset(_sdl2test_library)

check_required_components(SDL2)
