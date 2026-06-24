if (WHISPER_SDL2)
    set(TARGET whisper-gui)

    # the GUI renders with OpenGL (via the Dear ImGui SDL2 + OpenGL3 backend)
    find_package(OpenGL QUIET)
    if (NOT OpenGL_FOUND)
        message(WARNING "whisper-gui: OpenGL not found - skipping target")
        return()
    endif()

    # Dear ImGui is a small, header-light immediate-mode GUI library.
    # It ships no CMakeLists, so we fetch the sources and compile the few
    # files we need directly into this target - nothing is vendored in-tree.
    include(FetchContent)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.5
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(imgui)

    add_executable(${TARGET}
        main.cpp
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )

    include(DefaultTargetOptions)

    target_include_directories(${TARGET} PRIVATE
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${SDL2_INCLUDE_DIRS}
    )

    target_link_libraries(${TARGET} PRIVATE
        common
        whisper
        ${SDL2_LIBRARIES}
        OpenGL::GL
        ${CMAKE_THREAD_LIBS_INIT}
    )

    install(TARGETS ${TARGET} RUNTIME)
endif ()
