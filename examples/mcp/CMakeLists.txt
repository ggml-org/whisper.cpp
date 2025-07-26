set(TARGET whisper-mcp-server)
add_executable(${TARGET} mcp-server.cpp stdio-transport.cpp mcp-handler.cpp)
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(DEMO_TARGET mcp-demo)
add_executable(${DEMO_TARGET} mcp-demo.cpp stdio-client.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common json_cpp whisper ${CMAKE_THREAD_LIBS_INIT})

# mcp_client only needs json_cpp and threading not whisper.
target_link_libraries(${DEMO_TARGET} PRIVATE json_cpp ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} ${DEMO_TARGET} RUNTIME)
