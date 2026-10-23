# QVAC (see PATCHES.md): subprocess checks for both native frontends. Deliberate
# missing paths prove device errors/listing happen before any model load.
cmake_minimum_required(VERSION 3.5)

if (NOT DEFINED CLI OR NOT DEFINED FIT OR NOT DEFINED TEST_DIR)
    message(FATAL_ERROR "CLI, FIT and TEST_DIR are required")
endif()

set(missing_model "${TEST_DIR}/missing-device-test-model.bin")
set(missing_audio "${TEST_DIR}/missing-device-test-audio.wav")
set(invalid_env "1invalid-environment-target")
set(absent_device "whisper-test-device-that-is-not-registered")

function(check_frontend name executable expected_status environment expected forbidden)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "WHISPER_ARG_DEVICE=${environment}" "${executable}" ${ARGN}
        RESULT_VARIABLE status
        OUTPUT_VARIABLE output
        ERROR_VARIABLE errors
        TIMEOUT 30)
    set(combined "${output}\n${errors}")
    if (NOT "${status}" STREQUAL "${expected_status}")
        message(FATAL_ERROR "${name}: expected exit ${expected_status}, got ${status}\n${combined}")
    endif()
    string(FIND "${combined}" "${expected}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "${name}: missing '${expected}'\n${combined}")
    endif()
    if (NOT "${forbidden}" STREQUAL "")
        string(FIND "${combined}" "${forbidden}" found)
        if (NOT found EQUAL -1)
            message(FATAL_ERROR "${name}: unexpected '${forbidden}'\n${combined}")
        endif()
    endif()
    message(STATUS "${name}: passed (exit ${status})")
endfunction()

# Listing must work with no model/audio and must not parse an ignored env target.
foreach(frontend CLI FIT)
    check_frontend("${frontend} listing" "${${frontend}}" 0 "${invalid_env}"
        "Available GGML devices (GPU indices count only GPU/IGPU devices):" "WHISPER_ARG_DEVICE:"
        --list-devices --model "${missing_model}")
endforeach()

# Explicit errors precede model errors, and explicit flags override the env.
foreach(frontend CLI FIT)
    if (frontend STREQUAL "CLI")
        set(error_status 1)
        set(device_flags --device -dev)
    else()
        set(error_status 2)
        set(device_flags --device --gpu-device -dev)
    endif()
    foreach(flag IN LISTS device_flags)
        check_frontend("${frontend} ${flag} unavailable" "${${frontend}}" ${error_status} "${invalid_env}"
            "GPU device '${absent_device}' is unavailable" "WHISPER_ARG_DEVICE:"
            --model "${missing_model}" ${flag} "${absent_device}")
        check_frontend("${frontend} ${flag} overflow" "${${frontend}}" ${error_status} "${invalid_env}"
            "invalid GPU device index:" "WHISPER_ARG_DEVICE:"
            --model "${missing_model}" ${flag} 999999999999999999999999)
        check_frontend("${frontend} ${flag} missing value" "${${frontend}}" ${error_status} "${invalid_env}"
            "requires a value" "WHISPER_ARG_DEVICE:"
            --model "${missing_model}" ${flag})
        check_frontend("${frontend} ${flag} conflict after --no-gpu" "${${frontend}}" ${error_status} "${invalid_env}"
            "cannot be combined with --no-gpu" "WHISPER_ARG_DEVICE:"
            --model "${missing_model}" --no-gpu ${flag} HTP0)
        check_frontend("${frontend} ${flag} conflict before --no-gpu" "${${frontend}}" ${error_status} "${invalid_env}"
            "cannot be combined with --no-gpu" "WHISPER_ARG_DEVICE:"
            --model "${missing_model}" ${flag} HTP0 --no-gpu)
    endforeach()
    check_frontend("${frontend} invalid environment" "${${frontend}}" ${error_status} "${invalid_env}"
        "WHISPER_ARG_DEVICE: invalid GPU device index" ""
        --model "${missing_model}")
endforeach()

# CPU mode suppresses the environment and reaches the ordinary input/model
# validation. These paths neither require a GPU nor open a real model.
check_frontend("CLI CPU ignores env" "${CLI}" 2 "${invalid_env}"
    "no input files specified" "WHISPER_ARG_DEVICE:"
    --model "${missing_model}" --file "${missing_audio}" --no-gpu)
check_frontend("FIT CPU ignores env" "${FIT}" 2 "${invalid_env}"
    "model-unreadable" "WHISPER_ARG_DEVICE:"
    --model "${missing_model}" --no-gpu)
