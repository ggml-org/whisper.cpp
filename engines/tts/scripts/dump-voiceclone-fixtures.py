#!/usr/bin/env python3
"""Generate the voice-clone test-harness golden fixtures.

The harness has two fixture tiers:

  * SMALL / committed (this script's default `synthetic` mode).
    Tiny, deterministic, model-free arrays whose expected metric scores are
    computed here with the *reference* (pure-Python) implementation of each
    metric and written next to the inputs.  The C++ self-test
    (test/test_voiceclone_metrics.cpp) then recomputes the same metric and
    asserts it reproduces the committed score, so every harness run cross-checks
    the C++ port against this independent reference.  The inputs use values that
    are exactly representable in float32 (integers and halves), so the only
    cross-language difference is the float64 metric arithmetic itself, which the
    test tolerances comfortably cover.  These files are small enough to live in
    git and let the `unit` ctest tier run with no downloads (synthetic mode does
    not import numpy on purpose, so it works on a bare Python).

  * HEAVY / off-device (the `reference` mode, documented below, NOT run in CI).
    The real speaker-similarity references the design calls for:
      - target speaker embeddings from CAMPPlus (on-device) and from
        WavLM-base-plus-sv / ECAPA-TDNN / ResNet (off-device, SpeechBrain),
      - WavLM-Large layer-3 time-averaged (mu, sigma) statistics used as the
        inverse-optimization objective (Kim et al., gradient-based style
        extraction),
      - reference cloned voice vectors (style_ttl / style_dp),
      - expected per-stage gradients (PyTorch autograd through the
        onnx2torch-converted Supertonic pipeline) for the gradcheck.
    These require the heavy models + audio corpus and are produced on a
    workstation/GPU, versioned out-of-tree, and pointed at via
    -DTTS_CPP_TEST_REF_DIR.  This mode is a stub here on purpose: it documents
    the contract so a later cloning task fills it in alongside the C++ analytic
    gradients it validates.

Usage:
    python3 scripts/dump-voiceclone-fixtures.py synthetic \\
        --out test/fixtures/voiceclone/v1
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


# Minimal NumPy .npy v1.0 writer (little-endian float32), byte-compatible with
# the reader in src/npy.h.  Implemented here so synthetic mode needs no numpy.

def write_npy_f32(path: Path, shape: tuple[int, ...], data: list[float]) -> None:
    header = ("{'descr': '<f4', 'fortran_order': False, 'shape': %s, }"
              % _shape_repr(shape))
    preamble = 10  # 6 magic + 2 version + 2 header-len
    total = preamble + len(header) + 1
    while total % 64 != 0:
        header += " "
        total += 1
    header += "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY")
        f.write(bytes([1, 0]))
        f.write(struct.pack("<H", len(header)))
        f.write(header.encode("latin1"))
        f.write(struct.pack("<%df" % len(data), *data))


def _shape_repr(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return "(%d,)" % shape[0]
    return "(" + ", ".join(str(s) for s in shape) + ")"


def to_f32(values: list[float]) -> list[float]:
    # Round every value through float32 so the committed bytes and the expected
    # score are computed from the exact same numbers the C++ side will read.
    return [struct.unpack("<f", struct.pack("<f", v))[0] for v in values]


# Metric reference implementations.  These MUST stay numerically identical to
# the C++ definitions in src/voiceclone_metrics.h; the committed fixtures exist
# to catch any drift between the two.

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    denom = na * nb
    return 0.0 if denom <= 0.0 else dot / denom


def time_avg_stats(h: list[list[float]]) -> tuple[list[float], list[float]]:
    t = len(h)
    d = len(h[0])
    mu = [sum(h[i][j] for i in range(t)) / t for j in range(d)]
    sigma = [math.sqrt(sum((h[i][j] - mu[j]) ** 2 for i in range(t)) / t) for j in range(d)]
    return mu, sigma


def wavlm_style_loss(h_ref: list[list[float]], h_tgt: list[list[float]]) -> float:
    mu_r, sig_r = time_avg_stats(h_ref)
    mu_t, sig_t = time_avg_stats(h_tgt)
    return (sum((a - b) ** 2 for a, b in zip(mu_r, mu_t))
            + sum((a - b) ** 2 for a, b in zip(sig_r, sig_t)))


def word_error_rate(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    n, m = len(r), len(h)
    if n == 0:
        return 0.0 if m == 0 else float(m)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    return dp[n][m] / n


def flatten(rows: list[list[float]]) -> list[float]:
    return [v for row in rows for v in row]


def generate_synthetic(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # --- cosine similarity fixture -----------------------------------------
    # Values exactly representable in float32 (integers + halves).
    embed_a = to_f32([1.0, 2.0, 3.0, 4.0, 0.0, -1.0, -2.0, 0.5])
    embed_b = to_f32([2.0, 1.0, 0.0, 4.0, 1.0, -1.0, 0.0, 0.25])
    write_npy_f32(out / "embed_a.npy", (len(embed_a),), embed_a)
    write_npy_f32(out / "embed_b.npy", (len(embed_b),), embed_b)
    write_npy_f32(out / "expected_cosine.npy", (1,), [cosine_similarity(embed_a, embed_b)])

    # --- WavLM layer-3 style loss fixture ----------------------------------
    # Two short [T, D] matrices standing in for WavLM L3 hidden states; differ
    # in length on purpose (the loss is time-averaged so T may differ).
    hidden_ref = [
        to_f32([0.0, 1.0, 2.0, -1.0, 0.5, 3.0]),
        to_f32([2.0, 1.0, 0.0, 1.0, -0.5, 1.0]),
        to_f32([4.0, 3.0, -2.0, 0.0, 0.5, 2.0]),
        to_f32([-2.0, 1.0, 2.0, 1.0, 1.5, 0.0]),
        to_f32([0.0, 0.0, 2.0, -1.0, -0.5, 4.0]),
    ]
    hidden_target = [
        to_f32([1.0, 0.0, 1.0, 0.0, 0.5, 2.0]),
        to_f32([1.0, 2.0, -1.0, 2.0, 0.5, 1.0]),
        to_f32([3.0, 1.0, 0.0, 1.0, -0.5, 3.0]),
        to_f32([-1.0, 2.0, 1.0, 0.0, 1.5, 1.0]),
    ]
    write_npy_f32(out / "hidden_ref.npy", (len(hidden_ref), len(hidden_ref[0])), flatten(hidden_ref))
    write_npy_f32(out / "hidden_target.npy", (len(hidden_target), len(hidden_target[0])), flatten(hidden_target))
    write_npy_f32(out / "expected_style_loss.npy", (1,), [wavlm_style_loss(hidden_ref, hidden_target)])

    # --- WER fixture --------------------------------------------------------
    ref_text = "the quick brown fox jumps over the lazy dog"
    hyp_text = "the quick brown box jumps over a lazy dog"  # box<->fox, a<->the
    (out / "wer_ref.txt").write_text(ref_text + "\n", encoding="utf-8")
    (out / "wer_hyp.txt").write_text(hyp_text + "\n", encoding="utf-8")
    write_npy_f32(out / "expected_wer.npy", (1,), [word_error_rate(ref_text, hyp_text)])

    print(f"wrote synthetic voice-clone fixtures to {out}")
    print(f"  cosine     = {cosine_similarity(embed_a, embed_b):.6f}")
    print(f"  style_loss = {wavlm_style_loss(hidden_ref, hidden_target):.6f}")
    print(f"  wer        = {word_error_rate(ref_text, hyp_text):.6f}")


def generate_reference(out: Path) -> None:
    raise SystemExit(
        "reference (heavy/off-device) fixtures are not generated by this stub.\n"
        "They require CAMPPlus + WavLM-Large + ECAPA-TDNN + ResNet + the audio\n"
        "corpus and a PyTorch Supertonic (onnx2torch) build for per-stage\n"
        "gradients.  A later cloning task wires this mode alongside the C++\n"
        "analytic gradients it validates; see the module docstring for the\n"
        "contract and point the build at the result via -DTTS_CPP_TEST_REF_DIR."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("synthetic", "reference"),
                   help="synthetic = small committed fixtures; reference = heavy off-device (stub)")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent.parent / "test/fixtures/voiceclone/v1",
                   help="output directory (default: test/fixtures/voiceclone/v1)")
    args = p.parse_args()

    if args.mode == "synthetic":
        generate_synthetic(args.out)
    else:
        generate_reference(args.out)


if __name__ == "__main__":
    main()
