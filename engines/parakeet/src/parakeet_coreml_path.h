#pragma once

#include <cstdint>
#include <string>
#include <vector>

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

std::string coreml_diarizer_sidecar_path(const std::string & gguf_path, int capacity);

int coreml_diarizer_sidecar_capacity(const std::string & gguf_path,
                                     const std::string & candidate_path);

// Smallest capacity in the ascending list that holds `frames` rows, 0 if none.
int coreml_diarizer_capacity_for(const std::vector<int> & sorted_capacities, int frames);

struct CoremlDiarizerBundle {
    int capacity = 0;
    uint64_t weights_bytes = 0;
};

// Every call of the range uses between min_frames and max_frames rows.
struct CoremlFrameRange {
    int min_frames = 0;
    int max_frames = 0;
};

// Projected system RAM of the diarizer bundles (ascending by capacity) that
// calls in `ranges` can load. A range charges every capacity from the one
// that covers min_frames up to the one that covers max_frames, or all larger
// capacities when max_frames exceeds them. Measured on an M4: the Neural
// Engine keeps about two copies of the weights, and the first bundle adds a
// fixed runtime cost.
uint64_t coreml_diarizer_resident_bytes(const std::vector<CoremlDiarizerBundle> & bundles,
                                        const std::vector<CoremlFrameRange> & ranges);

}  // namespace parakeet
