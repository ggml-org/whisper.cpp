if (WHISPER_SDL2)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    set(TARGET whisper-talk-llama)
    add_executable(${TARGET} talk-llama.cpp
        llama.cpp
        llama-adapter.cpp
        llama-arch.cpp
        llama-batch.cpp
        llama-chat.cpp
        llama-context.cpp
        llama-cparams.cpp
        llama-grammar.cpp
        llama-graph.cpp
        llama-hparams.cpp
        llama-impl.cpp
        llama-io.cpp
        llama-kv-cache-unified.cpp
        llama-kv-cache-unified-iswa.cpp
        llama-memory-recurrent.cpp
        llama-memory-hybrid.cpp
        llama-memory.cpp
        llama-mmap.cpp
        llama-model-loader.cpp
        llama-model-saver.cpp
        llama-model.cpp
        llama-quant.cpp
        llama-sampling.cpp
        llama-vocab.cpp
        unicode.cpp
        unicode-data.cpp)
    target_include_directories(${TARGET} PRIVATE ${SDL2_INCLUDE_DIRS})

    target_link_libraries(${TARGET} PRIVATE common common-sdl whisper ${SDL2_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

    if(WIN32)
        # It requires Windows 8.1 or later for PrefetchVirtualMemory
        target_compile_definitions(${TARGET} PRIVATE -D_WIN32_WINNT=0x0602)
    endif()

    include(DefaultTargetOptions)
endif ()
