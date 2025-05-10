#
# libstream
#

set(TARGET libstream)

add_executable(${TARGET}
    emscripten.cpp
    )

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE
    whisper
    )

unset(EXTRA_FLAGS)

if (WHISPER_WASM_SINGLE_FILE)
    set(EXTRA_FLAGS "-s SINGLE_FILE=1")
    message(STATUS "Embedding WASM inside stream.js")

    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/bin/libstream.js
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/stream.wasm/stream.js
        )
endif()

set_target_properties(${TARGET} PROPERTIES LINK_FLAGS " \
    --bind \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE=8 \
    -s INITIAL_MEMORY=1024MB \
    -s TOTAL_MEMORY=1024MB \
    -s FORCE_FILESYSTEM=1 \
    -s EXPORTED_RUNTIME_METHODS=\"['print', 'printErr', 'ccall', 'cwrap', 'HEAPU8']\" \
    ${EXTRA_FLAGS} \
    ")

#
# stream.wasm
#

set(TARGET stream.wasm)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index-tmpl.html  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/index.html @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../helpers.js    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}/helpers.js @ONLY)
