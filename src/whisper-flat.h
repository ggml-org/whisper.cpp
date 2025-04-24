#pragma once

#ifdef WHISPER_SHARED
#    ifdef _WIN32
#        ifdef WHISPER_BUILD
#            define WHISPER_FLAT_API __declspec(dllexport)
#        else
#            define WHISPER_FLAT_API __declspec(dllimport)
#        endif
#    else
#        define WHISPER_FLAT_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define WHISPER_FLAT_API
#endif

#include "whisper.h"

#ifdef __cplusplus
extern "C" {
#endif

    WHISPER_FLAT_API void whisper_flat_backend_load_all(void);
    WHISPER_FLAT_API struct whisper_timings * whisper_flat_get_timings_with_state(struct whisper_state * state);
    WHISPER_FLAT_API struct whisper_state * whisper_flat_get_state_from_context(struct whisper_context * ctx);
    WHISPER_FLAT_API const char * whisper_flat_get_system_info_json(void);

#ifdef __cplusplus
}
#endif

