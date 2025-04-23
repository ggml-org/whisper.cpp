#pragma once

#ifdef BINDINGS_FLAT
#endif

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_API __declspec(dllexport) extern
#        else
#            define GGML_API __declspec(dllimport) extern
#        endif
#    else
#        define GGML_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define GGML_API extern
#endif

#ifdef  __cplusplus
extern "C" {
#endif



#ifdef  __cplusplus
}
#endif
