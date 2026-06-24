# whisper-gui: a minimal desktop GUI built on Dear ImGui (SDL2 + OpenGL3).
#
# This target is fully self-contained / air-gapped friendly: both Dear ImGui
# and SDL2 are vendored under deps/ and built from source, so a fresh clone
# builds with no network and no SDL2/ImGui system packages. The only thing the
# build machine must provide are the OS windowing + OpenGL development headers
# (X11/Wayland + GL on Linux), which ship with any desktop's base dev packages
# and cannot be vendored.
#
# It is intentionally decoupled from WHISPER_SDL2 (which uses a *system* SDL2
# via find_package) so it can build on machines that have no SDL2 installed.

option(WHISPER_GUI "whisper: build the self-contained desktop GUI (vendored SDL2 + Dear ImGui)" OFF)

if (NOT WHISPER_GUI)
    return()
endif()

set(TARGET whisper-gui)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/imgui)
set(SDL_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/deps/SDL)

# --- vendored SDL2 (static) ---------------------------------------------------
# build a static SDL2, no shared lib, no tests, and don't install it.
set(SDL_SHARED          OFF CACHE BOOL "" FORCE)
set(SDL_STATIC          ON  CACHE BOOL "" FORCE)
set(SDL_STATIC_PIC      ON  CACHE BOOL "" FORCE)
set(SDL_TEST            OFF CACHE BOOL "" FORCE)
set(SDL2_DISABLE_INSTALL ON CACHE BOOL "" FORCE)

if (NOT TARGET SDL2-static)
    add_subdirectory(${SDL_DIR} ${CMAKE_CURRENT_BINARY_DIR}/SDL EXCLUDE_FROM_ALL)
endif()

# --- OpenGL (provided by the OS / GPU driver, never vendored) ------------------
find_package(OpenGL REQUIRED)

# --- the executable -----------------------------------------------------------
add_executable(${TARGET}
    main.cpp
    diarization.cpp
    neural_diarize.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

include(DefaultTargetOptions)

target_compile_features(${TARGET} PRIVATE cxx_std_17) # std::filesystem

if (MSVC)
    # json.hpp / Dear ImGui translation units need /bigobj; /utf-8 for sources
    target_compile_options(${TARGET} PRIVATE /bigobj /utf-8)
endif()

target_include_directories(${TARGET} PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/..   # examples/json.hpp
)

target_link_libraries(${TARGET} PRIVATE
    common
    whisper
    SDL2::SDL2-static
    SDL2::SDL2main
    OpenGL::GL
    ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS ${TARGET} RUNTIME)

# Windows: a tiny double-clickable launcher that sets the working directory to
# the repo root (so models\ and diarize.py resolve) and starts whisper-gui.exe.
if (WIN32)
    add_executable(whisper-gui-launcher WIN32 launcher.cpp)
    include(DefaultTargetOptions)
    set_target_properties(whisper-gui-launcher PROPERTIES OUTPUT_NAME "WhisperGUI")
    if (MINGW)
        # wWinMain (wide entry point) needs -municode under MinGW
        target_compile_options(whisper-gui-launcher PRIVATE -municode)
        target_link_options(whisper-gui-launcher PRIVATE -municode)
    endif()
    install(TARGETS whisper-gui-launcher RUNTIME)
endif()
