#!/usr/bin/env python3
"""Grid-search Parler quant recipes by teacher-forced argmax agreement.

For each variant: convert a temp GGUF (convert-to-gguf.py), run
test-parler-decoder in PARLER_TEST_REPORT_ONLY mode against the fixtures,
parse the aggregate "argmax agreement" line, append a row to a TSV, delete
the GGUF (unless --keep-ggufs). Baselines measure existing GGUFs directly.

Machine-serial by design: convert is disk/CPU bound and the report binary is
multi-threaded already.

t5=F16 is deliberately absent from every variant: f16 T5 weights make ggml
cast activation rows to f16 for the dot product and flan-T5 activations
overflow that range (NaN). Decoder-side F16 tiers are safe — the all-f16
GGUF passes strict parity. LM heads never go below Q8_0 (6-bit heads were
shown to derail sampled decoding on large-v1).

Usage:
  python3 quant-grid.py --ref artifacts/parler-ref \
      --decoder build-parler/test-parler-decoder \
      --baseline f16=models/parler-mini-v1-f16.gguf \
      [--imatrix models/parler-mini-v1-imatrix.npz] \
      --out-dir /tmp/parler-grid
"""

import argparse
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(SCRIPT_DIR, "convert-to-gguf.py")

# (label, dtype, recipe override or None, use imatrix)
# 2026-07-16 findings (full tables in the results TSV / PROGRESS ledger):
# heads=F16 dominates (+3.4pt at q8-class for +10 MB), tables < +1pt, bulk
# ladder Q4_K 83.1 < IQ4_NL 85.8 < Q5_K 89.8 ~ Q5_0 90.4 < Q6_K+headF16
# 94.9 < Q8_0+tab/headF16 98.1 (f16 ceiling 99.61). Shipped recipes are
# q6_k/q8_0 only; the overrides below reproduce dropped sub-q6 tiers for
# future research (e.g. an imatrix pass — never yet measured).
GRID = [
    ("q6_k+im",   "q6_k", None,                     True),
    ("q4_k_m+im", "q6_k", "bulk=Q4_K,heads=Q8_0",   True),
    ("q5_0+im",   "q6_k", "bulk=Q5_0,heads=Q8_0",   True),
    ("q5_k+im",   "q6_k", "bulk=Q5_K,heads=Q8_0",   True),
    ("iq4_nl+im", "q6_k", "bulk=IQ4_NL,heads=Q8_0", True),
]

AGREE_RE = re.compile(r"argmax agreement: (\d+)/(\d+) \(([\d.]+)%\)")


def log(msg):
    print(f"[grid] {msg}", flush=True)


def measure(decoder, gguf, ref):
    env = dict(os.environ, PARLER_TEST_REPORT_ONLY="1")
    p = subprocess.run([decoder, gguf, ref], env=env, timeout=900,
                       capture_output=True, text=True)
    txt = p.stdout + p.stderr
    m = None
    for m in AGREE_RE.finditer(txt):
        pass
    if p.returncode != 0 or m is None or "REPORT DONE" not in txt:
        tail = "\n".join(txt.splitlines()[-8:])
        return None, f"decoder rc={p.returncode}\n{tail}"
    return (int(m.group(1)), int(m.group(2)), float(m.group(3))), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="parler-tts/parler-tts-mini-v1")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--imatrix", default=None)
    ap.add_argument("--baseline", action="append", default=[],
                    help="label=path of an existing GGUF to measure as-is (repeatable)")
    ap.add_argument("--only", default=None, help="substring filter on variant labels")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-ggufs", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tsv = os.path.join(args.out_dir, "grid-results.tsv")
    rows = []

    def record(label, spec, size_mb, agree, err):
        if agree:
            a, b, pct = agree
            row = f"{label}\t{spec}\t{size_mb:.0f}\t{a}/{b}\t{pct:.2f}"
        else:
            row = f"{label}\t{spec}\t{size_mb:.0f}\tFAILED\t-"
            log(f"{label} FAILED:\n{err}")
        rows.append(row)
        with open(tsv, "a") as f:
            f.write(row + "\n")
        log(f"result: {row}")

    for spec in args.baseline:
        label, _, path = spec.partition("=")
        log(f"baseline {label}: {path}")
        agree, err = measure(args.decoder, path, args.ref)
        record(label, "baseline", os.path.getsize(path) / 1e6, agree, err)

    for label, dtype, recipe, use_im in GRID:
        if args.only and args.only not in label:
            continue
        if use_im and not args.imatrix:
            log(f"skip {label} (no --imatrix)")
            continue
        spec = dtype + (f":{recipe}" if recipe else "") + (":im" if use_im else "")
        gguf = os.path.join(args.out_dir, f"grid-{label.replace('+', '_')}.gguf")
        cmd = [sys.executable, CONVERTER, "--model-id", args.model_id,
               "--dtype", dtype, "--out", gguf]
        if recipe:
            cmd += ["--recipe", recipe]
        if use_im:
            cmd += ["--imatrix", args.imatrix]
        log(f"convert {label}: {spec}")
        t0 = time.time()
        p = subprocess.run(cmd, timeout=1800, capture_output=True, text=True)
        if p.returncode != 0:
            record(label, spec, 0, None,
                   "convert failed:\n" + "\n".join((p.stdout + p.stderr).splitlines()[-10:]))
            continue
        size_mb = os.path.getsize(gguf) / 1e6
        log(f"convert {label} done in {time.time() - t0:.0f}s ({size_mb:.0f} MB)")
        agree, err = measure(args.decoder, gguf, args.ref)
        record(label, spec, size_mb, agree, err)
        if not args.keep_ggufs:
            os.remove(gguf)

    log("---- final table (label / spec / MB / agree / %) ----")
    for row in rows:
        log(row)
    log(f"TSV: {tsv}")


if __name__ == "__main__":
    main()
