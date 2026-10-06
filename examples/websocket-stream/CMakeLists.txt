set(TARGET whisper-websocket-stream)
add_executable(${TARGET} main.cpp whisper-server.cpp message-buffer.cpp)
find_package(ixwebsocket)
find_package(CURL REQUIRED)
include(DefaultTargetOptions)
target_link_libraries(${TARGET} PRIVATE common whisper ixwebsocket z CURL::libcurl ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)
