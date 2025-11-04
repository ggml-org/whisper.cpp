if (WHISPER_SDL2)
    # stream
    set(TARGET lsp)
    add_executable(${TARGET} lsp.cpp)

    include(DefaultTargetOptions)

    target_link_libraries(${TARGET} PRIVATE common json_cpp common-sdl whisper ${CMAKE_THREAD_LIBS_INIT})
endif ()
