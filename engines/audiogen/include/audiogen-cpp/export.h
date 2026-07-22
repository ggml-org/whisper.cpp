#pragma once

// AUDIOGEN_API marks symbols that are part of the public C++ surface.
// Annotating a class / function / variable with AUDIOGEN_API guarantees:
//
//   - on Windows + MSVC, when the library is built / consumed as a DLL
//     (AUDIOGEN_SHARED defined), the symbol gets the right
//     dllexport / dllimport modifier.
//   - on GCC / clang, the symbol stays at default ELF visibility even
//     when the library target sets CXX_VISIBILITY_PRESET=hidden +
//     VISIBILITY_INLINES_HIDDEN=ON (which we do, mirroring parakeet),
//     and even in static builds.
//
// The visibility("default") fallback in the static-build case is what
// lets a host project that vendors libaudiogen-cpp.a inside its own .so
// (e.g. a Bare addon / a Node native module) re-export the API
// surface to its own consumers without having to rewrap every
// function in a wrapper marked __attribute__((visibility("default"))).
//
// AUDIOGEN_SHARED  define when linking against or building libaudiogen-cpp
//                  as a DLL / shared object.  Set automatically as a PUBLIC
//                  compile definition on the audiogen-cpp target when the
//                  library is built shared, so consumers picking the target
//                  up via find_package(audiogen-cpp) inherit it transparently.
//
// AUDIOGEN_BUILD   define only inside translation units that compile the
//                  library itself (set as a PRIVATE compile definition on
//                  the audiogen-cpp target).  Flips Windows from `dllimport`
//                  (consumer side) to `dllexport` (library side).

#ifdef AUDIOGEN_SHARED
#  if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef AUDIOGEN_BUILD
#      define AUDIOGEN_API __declspec(dllexport)
#    else
#      define AUDIOGEN_API __declspec(dllimport)
#    endif
#  else
#    define AUDIOGEN_API __attribute__((visibility("default")))
#  endif
#else
#  if defined(__GNUC__) || defined(__clang__)
#    define AUDIOGEN_API __attribute__((visibility("default")))
#  else
#    define AUDIOGEN_API
#  endif
#endif
