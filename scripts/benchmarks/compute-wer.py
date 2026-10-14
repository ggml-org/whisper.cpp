#!/usr/bin/env python3
# Compute Word Error Rate (WER) for one hypothesis against one reference
# transcript. Invoked by scripts/benchmarks/run-family.sh when a family's
# spec in families.json carries a `correctness` block.
#
# Usage:
#   compute-wer.py --hypothesis-text "HYP STRING" \
#                  --reference REF_FILE \
#                  [--normalizer english|none] \
#                  [--hypothesis-format plain|moss-segments] \
#                  [--json-out PATH]
#
#   compute-wer.py --hypothesis-file HYP_FILE ...   # same, but hyp from file
#   compute-wer.py --self-test                       # run built-in checks
#
# Output JSON schema (single line, no trailing newline suppressed):
#   {
#     "wer":          0.0,
#     "n_ref_words":  17,
#     "n_hyp_words":  17,
#     "n_edits":      {"sub": 0, "ins": 0, "del": 0},
#     "normalizer":   "english",
#     "hypothesis_format": "plain",
#     "reference":    "path/to/ref.txt"
#   }
#
# --hypothesis-format names the shape of the hypothesis text. `plain` scores
# it as is. `moss-segments` is what the moss-transcribe CLI prints: one
# `[start-end] Sxx: text` line per segment (or the raw `[t][Sxx]text[t]`
# transcript when no segment parsed), and only the text is scored.
#
# Implementation: pure-Python word-level Levenshtein DP so the CI runners
# don't need `jiwer` installed. If `jiwer` IS importable, we use it to
# cross-check our own numbers under --self-test but never at runtime — a
# CI-installed jiwer with a different normalizer would silently change the
# reported WER across dispatches.
#
# The English normalizer is a stripped-down port of
# openai/whisper's EnglishTextNormalizer (BasicTextNormalizer path). It
# handles the punctuation / case / whitespace differences that show up
# between parakeet's raw output and the expected reference file — nothing
# fancier (no contraction expansion, no number normalization). If a family
# needs richer normalization later, add a new mode here rather than
# monkey-patching the callers.

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import unicodedata


# ---- normalizers ------------------------------------------------------------


_PUNCT_STRIP_RE = re.compile(r"[^\w\s']")   # keep word chars, whitespace, apostrophes
_WHITESPACE_RE  = re.compile(r"\s+")


def normalize_english(text: str) -> str:
    """Case-fold, strip punctuation (except apostrophes), collapse whitespace.

    Deliberately conservative — matches how whisper.cpp's librispeech eval
    treats reference/hypothesis pairs before comparison. Any richer
    normalization (contractions, numbers-to-words) should be a new mode,
    not a silent upgrade of this one.
    """
    # NFC-normalize to fold pre-composed accents so the punctuation strip
    # never leaves a stray combining mark behind.
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _PUNCT_STRIP_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_none(text: str) -> str:
    """Only collapse whitespace — leave case and punctuation untouched.

    Use for languages / scripts where the English normalizer would strip
    meaningful marks (e.g. Devanagari punctuation).
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


NORMALIZERS = {
    "english": normalize_english,
    "none":    normalize_none,
}


# ---- hypothesis formats -----------------------------------------------------


_MOSS_SEGMENT_PREFIX_RE = re.compile(r"^\s*\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]\s*S\d+:\s*")
_MOSS_RAW_MARKUP_RE     = re.compile(r"\[(?:\d+(?:\.\d+)?|S\d+)\]")


def strip_moss_segment_line(line: str) -> str:
    return _MOSS_RAW_MARKUP_RE.sub(" ", _MOSS_SEGMENT_PREFIX_RE.sub("", line))


def extract_moss_segments_text(text: str) -> str:
    """Return the spoken text of a moss-transcribe stdout capture.

    Each segment line is `[start-end] Sxx: text`; the prefix is dropped.
    The raw transcript markup `[seconds]` and `[Sxx]` that the CLI prints
    when no segment parsed is dropped as well, so the timestamps and
    speaker labels never count as words.
    """
    return "\n".join(strip_moss_segment_line(line) for line in text.splitlines())


def extract_plain_text(text: str) -> str:
    return text


HYPOTHESIS_FORMATS = {
    "plain":         extract_plain_text,
    "moss-segments": extract_moss_segments_text,
}


# ---- WER computation --------------------------------------------------------


def _edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Word-level Levenshtein DP. Returns (substitutions, insertions, deletions).

    Insertion = word in hyp but not in ref (extra word). Deletion = word in
    ref but not in hyp (missing word). Substitution = word replaced by
    another. Cost is 1 for each; on equal-cost ties `min()` compares the
    tuple's next element (substitution count), so the breakdown minimizes
    subs, then ins, then dels — deterministic across Python versions but
    arbitrary among equally-optimal alignments. Total WER is unaffected.
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, sub, ins, del) to align ref[:i] with hyp[:j].
    # We only need two rows at a time, but track full grid for backtrace-free
    # counting via per-cell tuples.
    prev: list[tuple[int, int, int, int]] = [(j, 0, j, 0) for j in range(m + 1)]
    for i in range(1, n + 1):
        curr: list[tuple[int, int, int, int]] = [(i, 0, 0, i)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr.append(prev[j - 1])
                continue
            # Three candidates: substitute, insert, delete.
            sub_c, sub_s, sub_i, sub_d = prev[j - 1]
            ins_c, ins_s, ins_i, ins_d = curr[j - 1]
            del_c, del_s, del_i, del_d = prev[j]
            sub_c += 1; sub_s += 1
            ins_c += 1; ins_i += 1
            del_c += 1; del_d += 1
            # Prefer sub, then del, then ins on ties (arbitrary but stable).
            best = min(
                (sub_c, sub_s, sub_i, sub_d),
                (del_c, del_s, del_i, del_d),
                (ins_c, ins_s, ins_i, ins_d),
            )
            curr.append(best)
        prev = curr
    _, subs, ins, dels = prev[m]
    return subs, ins, dels


def compute_wer(
    hypothesis: str,
    reference:  str,
    normalizer: str = "english",
    hypothesis_format: str = "plain",
) -> dict:
    """Compute WER + edit breakdown against `reference`. See module docstring
    for the returned schema."""
    if normalizer not in NORMALIZERS:
        raise ValueError(f"unknown normalizer: {normalizer!r} "
                         f"(known: {sorted(NORMALIZERS)})")
    if hypothesis_format not in HYPOTHESIS_FORMATS:
        raise ValueError(f"unknown hypothesis format: {hypothesis_format!r} "
                         f"(known: {sorted(HYPOTHESIS_FORMATS)})")
    norm = NORMALIZERS[normalizer]
    ref_words = norm(reference).split()
    hyp_words = norm(HYPOTHESIS_FORMATS[hypothesis_format](hypothesis)).split()

    if not ref_words:
        # WER is undefined against an empty reference. Report a WER of 0.0
        # when the hyp is also empty, else 1.0 with the hyp treated as pure
        # insertions — the shape callers can still render.
        return {
            "wer":         0.0 if not hyp_words else 1.0,
            "n_ref_words": 0,
            "n_hyp_words": len(hyp_words),
            "n_edits":     {"sub": 0, "ins": len(hyp_words), "del": 0},
            "normalizer":  normalizer,
            "hypothesis_format": hypothesis_format,
        }

    subs, ins, dels = _edit_counts(ref_words, hyp_words)
    return {
        "wer":         (subs + ins + dels) / len(ref_words),
        "n_ref_words": len(ref_words),
        "n_hyp_words": len(hyp_words),
        "n_edits":     {"sub": subs, "ins": ins, "del": dels},
        "normalizer":  normalizer,
        "hypothesis_format": hypothesis_format,
    }


# ---- self-test --------------------------------------------------------------


_MOSS_SEGMENT_CASES: list[tuple[str, str, str, float]] = [
    # (label, ref, moss-transcribe stdout, expected WER under moss-segments)
    ("two segments",
     "the quick brown fox jumps over the lazy dog",
     "[0.00-2.50] S1: The quick brown fox\n[2.50-5.00] S2: jumps over the lazy dog.\n",
     0.0),
    ("segment with a missing word",
     "the quick brown fox",
     "[0.00-2.50] S1: the brown fox\n",
     0.25),
    ("raw markup without parsed segments",
     "the quick brown fox",
     "[0.00][S1]the quick brown fox[2.50]\n",
     0.0),
    ("empty capture",
     "the quick brown fox",
     "",
     1.0),
]


def _self_test_moss_segments() -> list[str]:
    """The moss-segments format must drop timestamps and speaker labels, and
    the plain format must still count them, or the option is not wired."""
    failures: list[str] = []
    for label, ref, hyp, want in _MOSS_SEGMENT_CASES:
        got = compute_wer(hyp, ref, normalizer="english", hypothesis_format="moss-segments")
        if abs(got["wer"] - want) > 1e-6:
            failures.append(f"moss-segments {label}: wer {got['wer']} != {want}")
    label, ref, hyp, _want = _MOSS_SEGMENT_CASES[0]
    plain = compute_wer(hyp, ref, normalizer="english", hypothesis_format="plain")
    if plain["wer"] == 0.0:
        failures.append("plain format must keep the segment markup as words")
    return failures


def _self_test() -> int:
    """Runs known WER cases + optional cross-check against jiwer if present."""
    cases: list[tuple[str, str, str, float, tuple[int, int, int]]] = [
        # (label, ref, hyp, expected_wer, (sub, ins, del))
        ("identical",    "the quick brown fox", "the quick brown fox", 0.00, (0, 0, 0)),
        ("one sub",      "the quick brown fox", "the slow  brown fox", 0.25, (1, 0, 0)),
        ("one ins",      "the quick brown fox", "the quick brown red fox", 0.25, (0, 1, 0)),
        ("one del",      "the quick brown fox", "the quick fox",       0.25, (0, 0, 1)),
        ("all wrong",    "aaa bbb ccc",          "xxx yyy zzz",         1.00, (3, 0, 0)),
        ("empty hyp",    "aaa bbb ccc",          "",                    1.00, (0, 0, 3)),
        ("empty ref",    "",                      "aaa bbb",            1.00, (0, 2, 0)),
        ("both empty",   "",                      "",                    0.00, (0, 0, 0)),
        ("punct/case",   "Hello, world!",        "hello world",         0.00, (0, 0, 0)),
    ]
    failed: list[str] = []
    for label, ref, hyp, want_wer, (want_sub, want_ins, want_del) in cases:
        got = compute_wer(hyp, ref, normalizer="english")
        if abs(got["wer"] - want_wer) > 1e-6:
            failed.append(f"{label}: wer {got['wer']} != {want_wer}")
            continue
        edits = got["n_edits"]
        # Only assert edit counts when ref is non-empty; the empty-ref path
        # takes the special-case branch above whose split we don't want to
        # over-constrain.
        if ref and (edits["sub"], edits["ins"], edits["del"]) != (want_sub, want_ins, want_del):
            failed.append(
                f"{label}: edits {edits} != "
                f"{{'sub': {want_sub}, 'ins': {want_ins}, 'del': {want_del}}}"
            )

    # `normalizer=none` sanity check: punctuation makes the words unequal.
    got = compute_wer("hello world", "Hello, world!", normalizer="none")
    if got["wer"] == 0.0:
        failed.append("normalizer=none: punctuation should not be stripped")

    failed.extend(_self_test_moss_segments())

    # Cross-check against jiwer if installed. Never fail the self-test on
    # jiwer differences alone — jiwer's default normalizer differs from
    # ours — but log the delta so a developer sees whether their tuning
    # matches upstream expectations.
    try:
        import jiwer   # type: ignore[import-not-found]
        # jiwer.wer defaults to word-level; use its own normalization to
        # avoid the whisper-style pre-normalizer difference biasing the
        # cross-check. Comparing on the raw text is closest to apples-to-apples.
        for label, ref, hyp, _want, _edits in cases:
            if not ref or not hyp:
                continue  # jiwer raises on empty; skip.
            ours = compute_wer(hyp, ref, normalizer="english")["wer"]
            theirs = float(jiwer.wer(ref.lower(), hyp.lower()))
            if abs(ours - theirs) > 1e-6:
                print(f"note: jiwer disagreement on {label!r}: "
                      f"ours={ours} theirs={theirs}", file=sys.stderr)
    except ImportError:
        pass

    if failed:
        for f in failed:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("compute-wer.py: self-test ok")
    return 0


# ---- CLI --------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute WER for one hyp/ref pair.")
    ap.add_argument("--hypothesis-text", help="Hypothesis transcript as an inline string.")
    ap.add_argument("--hypothesis-file", type=pathlib.Path,
                    help="Hypothesis transcript from a file (mutually exclusive with --hypothesis-text).")
    ap.add_argument("--reference",       type=pathlib.Path,
                    help="Reference transcript file.")
    ap.add_argument("--normalizer",      default="english",
                    choices=sorted(NORMALIZERS),
                    help="Text normalizer applied to both sides before comparison (default: english).")
    ap.add_argument("--hypothesis-format", default="plain",
                    choices=sorted(HYPOTHESIS_FORMATS),
                    help="Shape of the hypothesis text: plain, or moss-segments for moss-transcribe's "
                         "'[start-end] Sxx: text' lines (default: plain).")
    ap.add_argument("--json-out",        type=pathlib.Path,
                    help="Write result JSON to this path in addition to stdout.")
    ap.add_argument("--self-test", action="store_true",
                    help="Run built-in WER cases and exit.")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if not args.reference:
        print("--reference is required", file=sys.stderr); return 2
    # Presence check (is-None), not truthiness — an empty `--hypothesis-text ""`
    # is a legitimate 100%-miss case (model emitted nothing); the driver relies
    # on this so it can score empty transcripts as WER 1.0 instead of skipping.
    if (args.hypothesis_text is None) == (args.hypothesis_file is None):
        print("exactly one of --hypothesis-text / --hypothesis-file is required",
              file=sys.stderr); return 2

    ref_text = args.reference.read_text(encoding="utf-8")
    if args.hypothesis_file is not None:
        hyp_text = args.hypothesis_file.read_text(encoding="utf-8")
    else:
        hyp_text = args.hypothesis_text

    result = compute_wer(hyp_text, ref_text, normalizer=args.normalizer,
                         hypothesis_format=args.hypothesis_format)
    result["reference"] = str(args.reference)

    line = json.dumps(result, sort_keys=True)
    print(line)
    if args.json_out:
        args.json_out.write_text(line + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
