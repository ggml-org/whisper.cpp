#!/usr/bin/env python3
"""Generate src/audio8/unicode_tables.inc.

The Audio8 tokenizer needs two pieces of Unicode data that the C++ standard
library does not provide:

  general categories -- the Qwen2 pre-tokenizer splits on \\p{L}, \\p{N} and
      \\s, so the engine has to know which codepoints are letters, numbers and
      whitespace. Approximating "letter" as "ASCII letter or any non-ASCII" is
      wrong for exactly the characters that show up in real prompts: emoji,
      CJK punctuation and currency signs are neither letters nor numbers, and
      splitting them as letters merges them into the neighbouring word.
      Whitespace is Unicode-wide too, so a non-breaking or ideographic space
      splits like a plain one.

  NFC -- the tokenizer normalises to Normalization Form C before splitting.
      Text that arrives decomposed ("e" + combining acute rather than "e
      acute") otherwise tokenizes to a different id sequence than the model
      was trained on.

Both are emitted as sorted tables the tokenizer binary-searches. Hangul is
composed and decomposed arithmetically at run time, so its 11k syllables stay
out of the tables.

Composition pairs are derived by asking the reference normaliser rather than by
reading the exclusion list: a pair composes exactly when normalising the two
codepoints together yields the single one. That keeps the generator honest
about composition exclusions and non-starter decompositions without
reimplementing either rule.

    python3 gen-audio8-unicode-tables.py --out src/audio8/unicode_tables.inc
"""
import argparse
import unicodedata

HANGUL_SYLLABLE_FIRST = 0xAC00
HANGUL_SYLLABLE_LAST = 0xD7A3
CODEPOINT_LIMIT = 0x110000
COLUMNS = 4


def is_hangul_syllable(codepoint):
    return HANGUL_SYLLABLE_FIRST <= codepoint <= HANGUL_SYLLABLE_LAST


def ranges_where(predicate):
    ranges = []
    start = None
    for codepoint in range(CODEPOINT_LIMIT):
        inside = predicate(codepoint)
        if inside and start is None:
            start = codepoint
        elif not inside and start is not None:
            ranges.append((start, codepoint - 1))
            start = None
    if start is not None:
        ranges.append((start, CODEPOINT_LIMIT - 1))
    return ranges


def category_ranges(prefix):
    return ranges_where(
        lambda codepoint: unicodedata.category(chr(codepoint)).startswith(prefix)
    )


# The regex crate's \s is Unicode White_Space, which unicodedata does not
# expose. It is the separator categories plus a fixed list of control and
# format characters, none of which have moved since Unicode 4.
WHITESPACE_CATEGORIES = ("Zs", "Zl", "Zp")
WHITESPACE_EXTRA = (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x85)


def is_whitespace(codepoint):
    return (
        codepoint in WHITESPACE_EXTRA
        or unicodedata.category(chr(codepoint)) in WHITESPACE_CATEGORIES
    )


def whitespace_ranges():
    return ranges_where(is_whitespace)


def combining_classes():
    return [
        (codepoint, unicodedata.combining(chr(codepoint)))
        for codepoint in range(CODEPOINT_LIMIT)
        if unicodedata.combining(chr(codepoint))
    ]


def canonical_parts(codepoint):
    decomposition = unicodedata.decomposition(chr(codepoint))
    if not decomposition or decomposition.startswith("<"):
        return None
    return [int(part, 16) for part in decomposition.split()]


def full_decomposition(codepoint):
    parts = canonical_parts(codepoint)
    if parts is None:
        return [codepoint]
    expanded = []
    for part in parts:
        expanded.extend(full_decomposition(part))
    return expanded


def decompositions():
    table = []
    for codepoint in range(CODEPOINT_LIMIT):
        if is_hangul_syllable(codepoint) or canonical_parts(codepoint) is None:
            continue
        table.append((codepoint, full_decomposition(codepoint)))
    return table


def composes(first, second, target):
    return unicodedata.normalize("NFC", chr(first) + chr(second)) == chr(target)


def compositions():
    table = []
    for codepoint in range(CODEPOINT_LIMIT):
        if is_hangul_syllable(codepoint):
            continue
        parts = canonical_parts(codepoint)
        if parts is None or len(parts) != 2:
            continue
        if composes(parts[0], parts[1], codepoint):
            table.append((parts[0], parts[1], codepoint))
    return sorted(table)


def format_rows(rows, formatter):
    lines = []
    for index in range(0, len(rows), COLUMNS):
        chunk = rows[index:index + COLUMNS]
        lines.append("    " + " ".join(formatter(row) for row in chunk))
    return "\n".join(lines)


def emit_ranges(name, ranges):
    body = format_rows(ranges, lambda r: "{0x%05X, 0x%05X}," % r)
    return (
        f"static const audio8_range k_audio8_{name}[] = {{\n{body}\n}};\n"
    )


def emit_combining(entries):
    body = format_rows(entries, lambda e: "{0x%05X, %3d}," % e)
    return f"static const audio8_combining k_audio8_combining[] = {{\n{body}\n}};\n"


def emit_compositions(entries):
    body = format_rows(entries, lambda e: "{0x%05X, 0x%05X, 0x%05X}," % e)
    return (
        f"static const audio8_composition k_audio8_compositions[] = {{\n{body}\n}};\n"
    )


def emit_decompositions(entries):
    values = []
    index = []
    for codepoint, expansion in entries:
        index.append((codepoint, len(values), len(expansion)))
        values.extend(expansion)
    data = format_rows(values, lambda v: "0x%05X," % v)
    table = format_rows(index, lambda e: "{0x%05X, %6d, %d}," % e)
    return (
        f"static const uint32_t k_audio8_decomposition_data[] = {{\n{data}\n}};\n\n"
        f"static const audio8_decomposition k_audio8_decompositions[] = {{\n{table}\n}};\n"
    )


def header():
    return (
        "// Generated by scripts/gen-audio8-unicode-tables.py. Do not edit.\n"
        "//\n"
        "// Letter, number and whitespace ranges back the Qwen2 pre-tokenizer's\n"
        "// \\p{L}, \\p{N} and \\s classes; the combining, composition and\n"
        "// decomposition tables back NFC normalisation. Hangul is handled\n"
        "// arithmetically and is not listed here.\n"
        f"// Unicode {unicodedata.unidata_version}.\n\n"
        "struct audio8_range { uint32_t first; uint32_t last; };\n"
        "struct audio8_combining { uint32_t cp; uint8_t ccc; };\n"
        "struct audio8_composition { uint32_t first; uint32_t second; uint32_t composed; };\n"
        "struct audio8_decomposition { uint32_t cp; uint32_t offset; uint32_t length; };\n\n"
    )


def build():
    sections = [
        header(),
        emit_ranges("letters", category_ranges("L")),
        "",
        emit_ranges("numbers", category_ranges("N")),
        "",
        emit_ranges("whitespace", whitespace_ranges()),
        "",
        emit_combining(combining_classes()),
        "",
        emit_compositions(compositions()),
        "",
        emit_decompositions(decompositions()),
    ]
    return "\n".join(sections)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    text = build()
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
