if (EMSCRIPTEN)
    add_subdirectory(javascript)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/javascript/publish.log
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/javascript/whisper.js
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/javascript/libwhisper.worker.js
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/javascript/package.json
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/javascript
        COMMAND npm publish
        COMMAND touch publish.log
        COMMENT "Publishing npm module v${PROJECT_VERSION}"
        VERBATIM
        )

    add_custom_target(publish-npm
        DEPENDS javascript/publish.log
        )
endif()
