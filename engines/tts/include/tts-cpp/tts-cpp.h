#pragma once

// Top-level tts-cpp library entry points.
//
// The library currently ships the Chatterbox model pipeline.  Additional
// engines (e.g. the Chatterbox multilingual model, other TTS backends) will
// land under the same umbrella and should prefer headers under
// <tts-cpp/...> for the generic API and <tts-cpp/<engine>/...> for
// engine-specific details.
//
// Two layers of API are exposed today:
//
//   1. High-level text -> wav via the CLI dispatcher.  The current
//      implementation wraps the CLI's argv path; a proper struct-based
//      public API will land as the code is split out of src/main.cpp.
//      Until then, callers building against the library can invoke
//      `tts_cpp_cli_main(argc, argv)` with the same flags accepted by the
//      `tts-cli` executable.
//
//   2. Lower-level per-engine APIs, e.g. the Chatterbox S3Gen + HiFT
//      back-half in <tts-cpp/chatterbox/s3gen_pipeline.h>.

#include "tts-cpp/export.h"

#ifdef __cplusplus
extern "C" {
#endif

TTS_CPP_API int tts_cpp_cli_main(int argc, char ** argv);

#ifdef __cplusplus
}
#endif
