set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET parakeet-server)
add_executable(${TARGET} parakeet-server.cpp)

include(DefaultTargetOptions)

target_sources(${TARGET} PRIVATE ../server-common.cpp)

target_link_libraries(${TARGET} PRIVATE common json_cpp parakeet ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

install(TARGETS ${TARGET} RUNTIME)
