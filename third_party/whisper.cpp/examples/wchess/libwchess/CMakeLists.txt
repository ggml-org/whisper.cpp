add_library(wchess-core STATIC
    WChess.cpp
    WChess.h
    Chessboard.cpp
    Chessboard.h
)

target_link_libraries(wchess-core
    PUBLIC
    whisper
    common
)

target_include_directories(wchess-core
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)

# add_executable(test-chessboard test-chessboard.cpp Chessboard.cpp)
