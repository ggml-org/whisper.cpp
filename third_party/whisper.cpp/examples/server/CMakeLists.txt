set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET whisper-server)
add_executable(${TARGET} server.cpp httplib.h)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common json_cpp whisper ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

install(TARGETS ${TARGET} RUNTIME)
