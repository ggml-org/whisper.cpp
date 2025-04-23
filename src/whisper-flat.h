#pragma once

#ifdef WHISPER_SHARED
#    ifdef _WIN32
#        ifdef WHISPER_BUILD
#            define WHISPER_API __declspec(dllexport)
#        else
#            define WHISPER_API __declspec(dllimport)
#        endif
#    else
#        define WHISPER_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define WHISPER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

    WHISPER_API void whisper_flat_backend_load_all(void);

#ifdef __cplusplus
}
#endif

