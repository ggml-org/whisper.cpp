set(TARGET wchess.wasm)

add_executable(${TARGET}
    wchess.wasm.cpp
    )

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE
    common
    wchess-core
    )

unset(EXTRA_FLAGS)

if (WHISPER_WASM_SINGLE_FILE)
    set(EXTRA_FLAGS "-s SINGLE_FILE=1")
    message(STATUS "Embedding WASM inside chess.js")

    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/bin/${TARGET}.js
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/js/chess.js
        )
endif()

set_target_properties(${TARGET} PROPERTIES LINK_FLAGS " \
    --bind \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE=8 \
    -s INITIAL_MEMORY=1024MB \
    -s TOTAL_MEMORY=1024MB \
    -s FORCE_FILESYSTEM=1 \
    -s EXPORTED_RUNTIME_METHODS=\"['print', 'printErr', 'ccall', 'cwrap', 'HEAPU8']]\" \
    ${EXTRA_FLAGS} \
    ")


add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/chessboardjs-1.0.0
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/jquery-3.7.1.min.js
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/js/
    )

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index-tmpl.html  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/index.html @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/helpers.js    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/js/helpers.js @ONLY)
