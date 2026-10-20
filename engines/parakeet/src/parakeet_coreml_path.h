#pragma once

#include <string>

namespace parakeet {

// Derives the Core ML encoder sidecar path from a GGUF path: strips the file
// extension and any trailing quantisation tag (dot- or dash-separated, e.g.
// `.f16`, `.bf16`, `.q8_0`, `-q8_0`), then appends `-encoder.mlmodelc`. The tag is
// dropped so a single exported encoder is shared across decoder quantisations
// of the same model:
//   models/parakeet-tdt-0.6b-v3.q8_0.gguf -> models/parakeet-tdt-0.6b-v3-encoder.mlmodelc
//   models/parakeet-tdt-0.6b-v3.f16.gguf  -> models/parakeet-tdt-0.6b-v3-encoder.mlmodelc
//   models/parakeet-tdt-0.6b-v3.bf16.gguf -> models/parakeet-tdt-0.6b-v3-encoder.mlmodelc
// Pure string logic; compiled on every platform so the naming contract can be
// unit-tested without Apple frameworks or a model.
std::string coreml_encoder_sidecar_path(const std::string & gguf_path);

// Sortformer v2.1 AOSC sidecar: the subsampling output/cache concatenation is
// supplied directly, so this bundle contains only the FastConformer block stack.
std::string coreml_bypass_encoder_sidecar_path(const std::string & gguf_path);

// Same derivation, `-decoder.mlmodelc`: the batched TDT decoder sidecar
// (src/coreml/parakeet-decoder.h, exported by
// scripts/export-decoder-coreml-ane.py). The encoder and decoder sidecars are
// independent: a GGUF may have either, both, or neither next to it.
std::string coreml_decoder_sidecar_path(const std::string & gguf_path);

}  // namespace parakeet
