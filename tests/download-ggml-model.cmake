if (NOT DEFINED MODEL_URL OR NOT DEFINED MODEL_PATH)
    message(FATAL_ERROR "MODEL_URL and MODEL_PATH must both be defined")
endif()

if (EXISTS "${MODEL_PATH}")
    message(STATUS "model already present, skipping download: ${MODEL_PATH}")
    return()
endif()

message(STATUS "downloading ${MODEL_URL}")

file(DOWNLOAD "${MODEL_URL}" "${MODEL_PATH}.tmp"
     STATUS download_status
     SHOW_PROGRESS)

list(GET download_status 0 status_code)
if (NOT status_code EQUAL 0)
    list(GET download_status 1 status_string)
    file(REMOVE "${MODEL_PATH}.tmp")
    message(FATAL_ERROR "failed to download ${MODEL_URL}: ${status_string}")
endif()

file(RENAME "${MODEL_PATH}.tmp" "${MODEL_PATH}")

message(STATUS "model saved to ${MODEL_PATH}")
