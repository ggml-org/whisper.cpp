#pragma once

// Multilingual streaming chunker for the Supertonic engine.
//
// Splits an input string into a list of substrings sized for per-chunk
// synthesis, preferring natural boundaries when available:
//
//   1. sentence-end punctuation  (. ? ! 。 ？ ！ ‼ ⁇ ⁈ ⁉ । ॥)
//   2. clause-end punctuation    (, ; : ， 、 ； ： ؛ ، and closing brackets)
//   3. whitespace                (handles CJK/Thai/Lao/Khmer where 1+2 are absent)
//   4. hard cut                  (last-resort cap at the upper tolerance bound)
//
// Token grain matches `supertonic_text_to_ids` (one ID per Unicode code
// point after normalization), so the input character count IS the token
// count that the engine will see.  No model tokenizer call is required
// for sizing.

#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp::supertonic::detail {

// Split `text` into chunks sized roughly `target_tokens` code points
// each, snapping to the best available boundary within ±`tolerance_pct`
// of the target.  When `first_chunk_tokens > 0`, the first chunk uses
// that smaller target instead (latency knob — first audio lands earlier
// while subsequent chunks stay large to keep throughput up).
//
// `min_chunk_tokens` is a hard floor on every chunk's size: the
// effective target is `max(target_tokens, min_chunk_tokens)` (and
// similarly for first-chunk).  The trailing chunk is merged into the
// previous one if it ends up below the floor.  Default 30 — empirically
// the model emits dropped/muddled phonemes when fed shorter stubs.
//
// Leading/trailing whitespace on each chunk is trimmed.  Adjacent chunks
// concatenated back together (modulo trimmed whitespace) reproduce the
// input.  Empty / whitespace-only chunks are not emitted.
std::vector<std::string> split_for_streaming(
    const std::string & text,
    int target_tokens,
    int first_chunk_tokens = 0,
    int tolerance_pct      = 20,
    int min_chunk_tokens   = 30);

// Sentence-end predicate over a Unicode code point.  Public so the
// engine's per-chunk "does this end on a natural sentence terminator?"
// helper can share the table with the chunker's boundary search —
// keeps additions (e.g. Ethiopic ።, Tibetan ། in the future) in one
// place.  See supertonic_chunker.cpp for the full set.
bool is_sentence_end_cp(uint32_t cp);

} // namespace tts_cpp::supertonic::detail
