#pragma once

// external voice injection.
//
// A Supertonic "voice" is just two small float tensors: `style_ttl`
// (timbre / identity, e.g. [1,50,256]) and `style_dp` (rhythm / pacing,
// e.g. [1,8,16]).  The 10 baked-in presets are these tensors stored in
// the GGUF; a "cloned" voice is simply a different pair.  The tensors are
// model-format-agnostic — a voice JSON produced from the ONNX models is
// the exact same numbers our GGML model consumes — so this loader gives a
// runtime path to synthesize with an externally supplied voice instead of
// only a baked-in name.
//
// Canonical JSON shape (matches scripts/convert-supertonic2-to-gguf.py,
// which writes voice["style_ttl"]["data"] / voice["style_dp"]["data"]
// straight into the GGUF tensors):
//
//   {
//     "style_ttl": { "data": [ ... ], "shape": [1, 50, 256] },
//     "style_dp":  { "data": [ ... ], "shape": [1,  8,  16] },
//     "metadata":  { "name": "marco", ... }
//   }
//
// `shape` is optional and only used to cross-check the flattened `data`
// length.  A bare `"style_ttl": [ ... ]` array form is also accepted.
// The flattened `data` order is row-major and identical to the host
// float layout the synthesis pipeline expects, so feeding a preset's JSON
// is bit-for-bit equivalent to selecting that preset by name.

#include <string>
#include <vector>

namespace tts_cpp::supertonic::detail {

struct external_voice {
    std::string        name;  // optional; from top-level "name" or metadata.name
    std::vector<float> ttl;   // flattened style_ttl, row-major
    std::vector<float> dp;    // flattened style_dp,  row-major
};

// Parse a Supertonic voice JSON from an in-memory string.  Returns false
// and (when `err` is non-null) sets a human-readable message on any
// malformed / missing field.  Does NOT validate element counts against a
// model — the Engine cross-checks `ttl`/`dp` sizes against the loaded
// model's baked voice tensors so a mis-shaped voice fails loudly at
// construction time.
bool parse_supertonic_voice_json(const std::string & json_text,
                                 external_voice & out,
                                 std::string * err);

// Same as parse_supertonic_voice_json but reads the JSON from a file.
bool load_supertonic_voice_json_file(const std::string & path,
                                     external_voice & out,
                                     std::string * err);

}  // namespace tts_cpp::supertonic::detail
