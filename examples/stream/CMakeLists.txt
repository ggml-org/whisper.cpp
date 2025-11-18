if (WHISPER_SDL2)
    set(TARGET whisper-stream)
    add_executable(${TARGET} stream.cpp)

    include(DefaultTargetOptions)

    target_link_libraries(${TARGET} PRIVATE common common-sdl whisper ${CMAKE_THREAD_LIBS_INIT})

    install(TARGETS ${TARGET} RUNTIME)
endif ()

# --- ADD THIS ENTIRE BLOCK AT THE END ---

# Find Boost for the WebSocket server
find_package(Boost REQUIRED COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIRS})

# Define the WebSocket server executable
set(WS_SERVER_TARGET whisper-ws) # You can change this name if you like
add_executable(${WS_SERVER_TARGET} server.cpp)

# Apply default options
include(DefaultTargetOptions)

# Link libraries needed by the WebSocket server
target_link_libraries(${WS_SERVER_TARGET} PRIVATE common common-sdl whisper Boost::system Boost::thread ${CMAKE_THREAD_LIBS_INIT})
# Make it installable (optional but good practice)
install(TARGETS ${WS_SERVER_TARGET} RUNTIME)

# ----------------------------------------
