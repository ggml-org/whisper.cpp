set(TARGET parakeet-cli)
add_executable(${TARGET} parakeet-cli.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common parakeet ${FFMPEG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)
