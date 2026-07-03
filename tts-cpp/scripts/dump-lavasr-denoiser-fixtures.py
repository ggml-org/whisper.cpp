#!/usr/bin/env python3
"""Dump onnxruntime golden fixtures for the LavaSR denoiser scalar-core parity
test (test/test_lavasr_denoiser_gguf.cpp).

Runs the UL-UNAS denoiser ONNX (Topping1/LavaSRcpp
`denoiser_core_legacy_fixed63.onnx`, input `spec_ri` [1,2,T,257] real/imag) on a
deterministic random spectrogram and writes:
  <out-dir>/spec_in.npy   float32 [2, T, 257]   network input  (real, imag)
  <out-dir>/spec_out.npy  float32 [2, T, 257]   onnxruntime output (spec_enh_ri)

The C++ test loads the GGUF, runs denoiser_net_forward on spec_in and compares
against spec_out.  Mirrors scripts/dump-lavasr-enhancer-fixtures.py.

Usage:
  python dump-lavasr-denoiser-fixtures.py \\
      --denoiser denoiser_core_legacy_fixed63.onnx \\
      --out-dir  <ref-dir>/lavasr-denoiser [--frames 63] [--seed 0]
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoiser", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", type=int, default=63)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    T, F = args.frames, 257
    rng = np.random.default_rng(args.seed)
    # Scale ~ a real log-STFT magnitude range so the mask operates in-distribution.
    spec_in = (rng.standard_normal((1, 2, T, F)).astype(np.float32)) * 0.5

    sess = ort.InferenceSession(args.denoiser, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name
    spec_out = sess.run([oname], {iname: spec_in})[0]  # (1,2,T,257)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "spec_in.npy"), spec_in[0].astype(np.float32))
    np.save(os.path.join(args.out_dir, "spec_out.npy"), spec_out[0].astype(np.float32))
    print(f"wrote spec_in.npy / spec_out.npy [2,{T},{F}] to {args.out_dir}")
    print(f"  onnx out range: [{spec_out.min():.4f}, {spec_out.max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
