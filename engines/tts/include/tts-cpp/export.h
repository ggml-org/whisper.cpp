#pragma once

// TTS_CPP_API marks symbols that are part of the public C++ surface.
// Annotating a class / function / variable with TTS_CPP_API guarantees:
//
//   - on Windows + MSVC, when the library is built / consumed as a DLL
//     (TTS_CPP_SHARED defined), the symbol gets the right
//     dllexport / dllimport modifier.
//   - on GCC / clang, the symbol stays at default ELF visibility even
//     when the library target sets CXX_VISIBILITY_PRESET=hidden + 
//     VISIBILITY_INLINES_HIDDEN=ON (which we do, mirroring parakeet),
//     and even in static builds.
//
// The visibility("default") fallback in the static-build case is what
// lets a host project that vendors libtts-cpp.a inside its own .so
// (e.g. a Bare addon / a Node native module) re-export the API
// surface to its own consumers without having to rewrap every
// function in a wrapper marked __attribute__((visibility("default"))).
//
// TTS_CPP_SHARED  define when linking against or building libtts-cpp as a
//                 DLL / shared object.  Set automatically as a PUBLIC
//                 compile definition on the tts-cpp target when the
//                 library is built shared, so consumers picking the
//                 target up via find_package(tts-cpp) inherit it
//                 transparently.
//
// TTS_CPP_BUILD   define only inside translation units that compile the
//                 library itself (set as a PRIVATE compile definition on
//                 the tts-cpp target).  Flips Windows from `dllimport`
//                 (consumer side) to `dllexport` (library side).

#ifdef TTS_CPP_SHARED
#  if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef TTS_CPP_BUILD
#      define TTS_CPP_API __declspec(dllexport)
#    else
#      define TTS_CPP_API __declspec(dllimport)
#    endif
#  else
#    define TTS_CPP_API __attribute__((visibility("default")))
#  endif
#else
#  if defined(__GNUC__) || defined(__clang__)
#    define TTS_CPP_API __attribute__((visibility("default")))
#  else
#    define TTS_CPP_API
#  endif
#endif
