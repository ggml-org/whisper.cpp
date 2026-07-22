set(TARGET parakeet-quantize)
add_executable(${TARGET} parakeet-quantize.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common parakeet ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS ${TARGET} RUNTIME)
