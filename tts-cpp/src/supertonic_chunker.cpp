#include "supertonic_chunker.h"

#include <algorithm>
#include <cstdint>

namespace tts_cpp::supertonic::detail {
namespace {

// Minimal UTF-8 decoder — same shape as the anon-namespace helpers in
// supertonic_preprocess.cpp.  Kept local so the chunker has no cross-file
// dependency beyond its own header.  Replaces malformed sequences with
// U+FFFD and a 1-byte advance (matches preprocess behaviour for parity).
bool utf8_decode(const char * s, size_t len, size_t & pos, uint32_t & cp) {
    if (pos >= len) return false;
    uint8_t b0 = (uint8_t) s[pos];
    if (b0 < 0x80) { cp = b0; pos += 1; return true; }
    int extra = 0;
    if      ((b0 & 0xE0) == 0xC0) { cp = b0 & 0x1F; extra = 1; }
    else if ((b0 & 0xF0) == 0xE0) { cp = b0 & 0x0F; extra = 2; }
    else if ((b0 & 0xF8) == 0xF0) { cp = b0 & 0x07; extra = 3; }
    else                          { cp = 0xFFFD; pos += 1; return true; }
    if (pos + 1 + extra > len)    { cp = 0xFFFD; pos += 1; return true; }
    for (int i = 0; i < extra; ++i) {
        uint8_t b = (uint8_t) s[pos + 1 + i];
        if ((b & 0xC0) != 0x80) { cp = 0xFFFD; pos += 1; return true; }
        cp = (cp << 6) | (b & 0x3F);
    }
    pos += 1 + extra;
    return true;
}

struct cp_at {
    uint32_t cp;        // code point
    size_t   byte_pos;  // byte offset of this code point in the source string
};

std::vector<cp_at> decode_with_byte_offsets(const std::string & s) {
    std::vector<cp_at> out;
    out.reserve(s.size());
    size_t pos = 0;
    while (pos < s.size()) {
        size_t   start = pos;
        uint32_t cp    = 0;
        if (!utf8_decode(s.data(), s.size(), pos, cp)) break;
        out.push_back({cp, start});
    }
    return out;
}

bool is_space_cp(uint32_t cp) {
    return cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D ||
           cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

// Clause-end punctuation (lower priority than sentence-end).  Includes
// CJK and Arabic equivalents.  Closing brackets count — a clause that
// just ended a parenthetical is a reasonable break point too.
bool is_clause_end_cp(uint32_t cp) {
    switch (cp) {
        case 0x002C: // ,
        case 0x003B: // ;
        case 0x003A: // :
        case 0xFF0C: // ， fullwidth comma
        case 0x3001: // 、 ideographic comma
        case 0xFF1B: // ； fullwidth semicolon
        case 0xFF1A: // ： fullwidth colon
        case 0x060C: // ،  Arabic comma
        case 0x061B: // ؛  Arabic semicolon
        case 0x0029: // )
        case 0x005D: // ]
        case 0x007D: // }
        case 0xFF09: // ）
            return true;
        default:
            return false;
    }
}

// Scan for the first index in (lo, hi] where pred(cps[idx-1].cp) is true.
// Right-first sweep from `target`, then leftward — chunks that end ON
// the punctuation/space read more naturally than chunks that end one
// character before it.  Returns SIZE_MAX if no match.
size_t scan_for(const std::vector<cp_at> & cps,
                size_t target,
                size_t lo,
                size_t hi,
                bool (*pred)(uint32_t))
{
    if (hi <= lo + 1) return SIZE_MAX;
    const size_t t = std::clamp(target, lo + 1, hi);
    for (size_t r = t; r <= hi; ++r) {
        if (pred(cps[r - 1].cp)) return r;
    }
    for (size_t r = t; r > lo + 1; --r) {
        if (pred(cps[r - 2].cp)) return r - 1;
    }
    return SIZE_MAX;
}

// Find the best boundary index for splitting.  Two windows:
//
//   `sent_lo..sent_hi`  — wide window for sentence-end punctuation.
//                          Sentence prosody dominates audio quality on
//                          this model (the duration predictor and
//                          attention run per-chunk, so chunk-aligned
//                          sentence breaks let the model phrase
//                          naturally), so sentence search reaches
//                          much further than clause/whitespace.
//
//   `norm_lo..norm_hi`  — tight user-controlled window for clause and
//                          whitespace fallbacks when no sentence is in
//                          reach.  Hard-cut at `norm_hi` as last
//                          resort.  Continuation flag in the engine
//                          makes the resulting mid-clause chunk audio
//                          tolerable; the bigger seam artifacts (small
//                          pauses, rate shifts) are inherent to
//                          per-chunk synthesis on a non-streaming-
//                          trained model and can't be removed at this
//                          layer.
//
// Returns the index AFTER the break (chunk = cps[start..break)).
size_t pick_break(const std::vector<cp_at> & cps,
                  size_t target,
                  size_t sent_lo, size_t sent_hi,
                  size_t norm_lo, size_t norm_hi)
{
    if (size_t b = scan_for(cps, target, sent_lo, sent_hi, is_sentence_end_cp);
        b != SIZE_MAX) return b;
    if (size_t b = scan_for(cps, target, norm_lo, norm_hi, is_clause_end_cp);
        b != SIZE_MAX) return b;
    if (size_t b = scan_for(cps, target, norm_lo, norm_hi, is_space_cp);
        b != SIZE_MAX) return b;
    return norm_hi;  // hard cut
}

std::string slice_to_string(const std::vector<cp_at> & cps,
                            size_t start_idx,
                            size_t end_idx,
                            const std::string & source) {
    if (start_idx >= end_idx) return {};
    const size_t byte_start = cps[start_idx].byte_pos;
    const size_t byte_end   = (end_idx < cps.size())
                                ? cps[end_idx].byte_pos
                                : source.size();
    std::string out = source.substr(byte_start, byte_end - byte_start);

    // Trim leading + trailing whitespace at the code-point level.  Done
    // by scanning the slice — cheaper than re-decoding given the slice
    // is typically tens of bytes.
    size_t l = 0;
    while (l < out.size() && (out[l] == ' ' || out[l] == '\t' ||
                              out[l] == '\n' || out[l] == '\r')) ++l;
    size_t r = out.size();
    while (r > l && (out[r - 1] == ' ' || out[r - 1] == '\t' ||
                     out[r - 1] == '\n' || out[r - 1] == '\r')) --r;
    return out.substr(l, r - l);
}

} // namespace

// Sentence-end punctuation across ASCII, CJK, Devanagari, and the
// extended Unicode punctuation range.  Conservative — symbols that
// can be sentence-terminating but ambiguous (e.g. ellipsis "…") are
// intentionally excluded since they often continue a thought.
//
// Public (declared in supertonic_chunker.h) so the engine's per-chunk
// "does this end on a natural sentence terminator?" helper shares the
// same table — additions (e.g. Ethiopic ።, Tibetan ། later) land in
// one place instead of needing to be synced across compilation units.
bool is_sentence_end_cp(uint32_t cp) {
    switch (cp) {
        case 0x002E: // .
        case 0x003F: // ?
        case 0x0021: // !
        case 0x3002: // 。  CJK ideographic full stop
        case 0xFF1F: // ？ fullwidth question mark
        case 0xFF01: // ！ fullwidth exclamation mark
        case 0x203C: // ‼ double exclamation
        case 0x2047: // ⁇ double question
        case 0x2048: // ⁈ question exclamation
        case 0x2049: // ⁉ exclamation question
        case 0x0964: // ।  Devanagari danda
        case 0x0965: // ॥  Devanagari double danda
        case 0x06D4: // ۔  Urdu full stop
            return true;
        default:
            return false;
    }
}

std::vector<std::string> split_for_streaming(
    const std::string & text,
    int target_tokens,
    int first_chunk_tokens,
    int tolerance_pct,
    int min_chunk_tokens)
{
    std::vector<std::string> out;
    if (target_tokens <= 0 || text.empty()) {
        // Caller is responsible for falling back to the batch path when
        // target_tokens <= 0; returning a single-element vector here so
        // the chunker remains usable as a defensive no-op splitter.
        if (!text.empty()) out.push_back(text);
        return out;
    }

    const std::vector<cp_at> cps = decode_with_byte_offsets(text);
    if (cps.empty()) return out;

    const int tol_pct        = std::clamp(tolerance_pct, 0, 100);
    const int min_chunk      = std::max(1, min_chunk_tokens);
    // Effective targets clamp up to min_chunk so the chunker never aims
    // for a sub-minimum chunk (the model glitches on stub input below
    // ~30 tokens — verified empirically on multiple seeds and texts).
    const int target_eff     = std::max(target_tokens, min_chunk);
    const int first_eff      = first_chunk_tokens > 0
                                   ? std::max(first_chunk_tokens, min_chunk)
                                   : 0;

    const size_t total = cps.size();
    size_t       start = 0;
    int          chunk_idx = 0;

    while (start < total) {
        const int target_this = (chunk_idx == 0 && first_eff > 0)
                                    ? first_eff
                                    : target_eff;

        // Tight window — for clause/whitespace boundaries and the
        // hard-cut fallback.  Driven by the user-supplied tolerance.
        // Lower bound is bumped to start + min_chunk so a break can't
        // produce a sub-minimum chunk on this iteration.
        int norm_lo_rel = std::max(1, target_this - target_this * tol_pct / 100);
        int norm_hi_rel = target_this + target_this * tol_pct / 100;
        norm_lo_rel     = std::max(norm_lo_rel, min_chunk);
        norm_hi_rel     = std::max(norm_hi_rel, norm_lo_rel);

        // Wide window — sentence-end search.  Reaches back to half the
        // effective target (so a sentence break that yields a too-small
        // chunk is rejected by the min_chunk floor) and forward to 2×
        // the target.  2× is empirical: catches a long-but-reasonable
        // first sentence in multi-sentence text (~75-90 chars at
        // target=50), but narrow enough that for a genuinely runaway
        // sentence (>2× target with no internal periods), the chunker
        // falls through to whitespace and produces multiple sub-
        // sentence chunks instead of slurping the whole tail as one
        // huge "sentence-aligned" chunk.
        int sent_lo_rel = std::max(1, target_this / 2);
        int sent_hi_rel = target_this * 2;
        sent_lo_rel     = std::max(sent_lo_rel, min_chunk);
        sent_hi_rel     = std::max(sent_hi_rel, sent_lo_rel);

        const size_t norm_lo = std::min(start + (size_t) norm_lo_rel, total);
        const size_t norm_hi = std::min(start + (size_t) norm_hi_rel, total);
        const size_t sent_lo = std::min(start + (size_t) sent_lo_rel, total);
        const size_t sent_hi = std::min(start + (size_t) sent_hi_rel, total);

        size_t brk;
        if (norm_hi <= start + 1 || total - start <= (size_t) norm_hi_rel) {
            // Entire remainder fits inside this chunk's upper tolerance —
            // take it all.  Avoids leaving a tiny sub-tolerance tail.
            brk = total;
        } else {
            const size_t target_abs = std::min(start + (size_t) target_this, total);
            brk = pick_break(cps, target_abs,
                             sent_lo, sent_hi,
                             norm_lo, norm_hi);
        }

        std::string chunk = slice_to_string(cps, start, brk, text);
        if (!chunk.empty()) out.push_back(std::move(chunk));
        start = brk;
        ++chunk_idx;
    }

    // Tail-merge heuristic: if the last chunk is genuinely tiny, fold
    // it into the previous chunk to avoid paying full pipeline cost for
    // a handful of trailing tokens.  Mirrors chatterbox_engine.cpp:608.
    //
    // Threshold is intentionally `max(6, target_tokens/3)`, NOT
    // `min_chunk_tokens` — using min_chunk here would merge any
    // last-chunk shorter than the floor, which can swallow a complete
    // final sentence (e.g. Korean "공원에서 산책하기 좋은 날이다."
    // is 18 code points, below a min_chunk=30 floor, but is itself a
    // valid sentence-aligned chunk that the model handles fine because
    // CJK information density per code point is much higher than ASCII).
    // The min_chunk floor governs what the chunker proactively *aims
    // for*, not what it does with whatever's left after the last natural
    // boundary.
    if (out.size() >= 2) {
        const std::vector<cp_at> tail_cps = decode_with_byte_offsets(out.back());
        const int                tail_thresh = std::max(6, target_tokens / 3);
        if ((int) tail_cps.size() < tail_thresh) {
            std::string merged = out[out.size() - 2];
            if (!merged.empty() && !out.back().empty()) merged.push_back(' ');
            merged += out.back();
            out.pop_back();
            out.back() = std::move(merged);
        }
    }

    return out;
}

} // namespace tts_cpp::supertonic::detail
