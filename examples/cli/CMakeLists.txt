set(TARGET whisper-cli)
add_executable(${TARGET} cli.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common whisper ${FFMPEG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    target_sources(${TARGET} PRIVATE win32.manifest)
endif()

install(TARGETS ${TARGET} RUNTIME)
