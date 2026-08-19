if (WHISPER_SDL2)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    if (WHISPER_USE_SYSTEM_LLAMA)
        message(STATUS "Using system-provided llama.cpp, skipping llama.cpp fetch")
        find_package(llama 0.1.2 REQUIRED)
    else()
        include(FetchContent)

        set(LLAMA_TAG "v0.1.2")

        set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)

        FetchContent_Declare(
            llama
            GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
            GIT_TAG        ${LLAMA_TAG}
        )

        message(STATUS "Fetching llama.cpp dependency (${LLAMA_TAG})...")
        FetchContent_MakeAvailable(llama)
    endif()

    set(TARGET whisper-talk-llama)
    add_executable(${TARGET} talk-llama.cpp)
    target_link_libraries(${TARGET} PRIVATE llama)
    target_include_directories(${TARGET} PRIVATE . ${SDL2_INCLUDE_DIRS})

    target_link_libraries(${TARGET} PRIVATE common common-sdl whisper ${SDL2_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TARGET} RUNTIME)

    if(WIN32)
        # It requires Windows 8.1 or later for PrefetchVirtualMemory
        target_compile_definitions(${TARGET} PRIVATE -D_WIN32_WINNT=0x0602)
    endif()

    include(DefaultTargetOptions)
endif ()
