#pragma once

// PARAKEET_API marks exported symbols when building/using a shared library; empty for static.
//
// PARAKEET_SHARED — define when linking against or building libparakeet as a DLL/shared object.
// PARAKEET_BUILD — define only in translation units that compile the library (export vs import on Windows).

#ifdef PARAKEET_SHARED
#  if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef PARAKEET_BUILD
#      define PARAKEET_API __declspec(dllexport)
#    else
#      define PARAKEET_API __declspec(dllimport)
#    endif
#  else
#    define PARAKEET_API __attribute__((visibility("default")))
#  endif
#else
#  define PARAKEET_API
#endif
