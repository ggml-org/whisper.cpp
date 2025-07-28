# Build whisper-stream-socket (Unix domain socket streaming server)

set(TARGET whisper-stream-socket)

add_executable(${TARGET} stream_socket.cpp)

include(DefaultTargetOptions)

find_package(Threads REQUIRED)

target_link_libraries(${TARGET} PRIVATE common whisper ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)
