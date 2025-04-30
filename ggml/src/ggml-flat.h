#pragma once

#ifdef BINDINGS_FLAT
#endif

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_FLAT_API __declspec(dllexport) extern
#        else
#            define GGML_FLAT_API __declspec(dllimport) extern
#        endif
#    else
#        define GGML_FLAT_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define GGML_FLAT_API extern
#endif

#ifdef  __cplusplus
extern "C" {
#endif

    #ifdef GGML_BINDINGS_FLAT
    GGML_FLAT_API void ggml_flat_test(void);
    #endif

#ifdef  __cplusplus
}
#endif
