#!/usr/bin/env python3
"""Convert the LavaSR denoiser (UL-UNAS) ONNX into a single GGUF for the tts-cpp
CPU/GGML denoiser.

LavaSR is a two-stage pipeline:
  (1) UL-UNAS denoiser  [this script]                         -> cleaned audio
  (2) Vocos BWE enhancer [convert-lavasr-enhancer-to-gguf.py] -> 48 kHz

UL-UNAS (arXiv:2503.00340, github.com/Xiaobin-Rong/ul-unas) is a GTCRN-family
TF-domain U-Net operating on 16 kHz STFT frames (n_fft=512, hop=256):
  ERB band-merge -> encoder (grouped depthwise-separable conv blocks with affine
  PReLU + causal time-frequency attention) -> 2x dual-path grouped RNN (DPGRNN)
  bottleneck -> decoder (with additive U-Net skips) -> real ratio mask * spec.

The reference model is the default `ULUNAS()` config
(github.com/Xiaobin-Rong/ul-unas), matching Topping1/LavaSRcpp release asset
`denoiser_core_legacy_fixed63.onnx` (the STFT-stripped network: input
`spec_ri` [1,2,T,257] real/imag, output `spec_enh_ri` [1,2,T,257]).

The ONNX initializers are already in the layout the C++ scalar core reads, so we
copy them verbatim (dropping the `model.` prefix) and record the architecture as
GGUF metadata:
  Conv weights:  [out, in/groups, kt, kf]   (ONNX/PyTorch order)
  Linear/FC W:   [out, in]                   (y[o] = sum_i W[o,i] x[i] + b[o])
  GRU gates:     weight_ih/hh_l0 [3H,*] (+ _reverse for BiGRU), gate order r,z,n

Usage:
  python convert-lavasr-denoiser-to-gguf.py \\
      --denoiser denoiser_core_legacy_fixed63.onnx \\
      --out      lavasr-denoiser.gguf \\
      --ftype    f32            # or f16
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import onnx
from gguf import GGUFWriter
from onnx import numpy_helper

ARCH = "lavasr-denoiser"

# STFT / rate params (must match ulunas.py + the @qvac/tts-onnx LavaSRDenoiser).
N_FFT = 512
HOP = 256
WIN = 512
SPEC_BINS = N_FFT // 2 + 1  # 257
WORK_SR = 16000             # UL-UNAS operates on 16 kHz audio
ERB_LOW = 65               # linear bins kept before the ERB band-merge
ERB_HIGH = 64              # ERB bands (erb_fc: 192 -> 64)
FREQ_COMP_RATIO = 4        # cTFA frequency-folding group size (FA r=4)
# Chunked inference (fixed-length ONNX export): 63-frame windows, hop 21, with a
# squared-Hann overlap-add — mirrors @qvac/tts-onnx LavaSRDenoiser.
CHUNK_FRAMES = 63
CHUNK_HOP = 21
BN_EPS = 1e-5              # nn.BatchNorm2d default
LN_EPS = 1e-8             # DPGRNN LayerNorm eps

# The C++ scalar core (denoiser_core.cpp) hardwires the default ULUNAS() topology
# (5 encoder/decoder blocks, 12-ch depthwise convs, erb 65/64, 2x DPGRNN 16-wide,
# cTFA r=4). Fail fast at conversion if the ONNX doesn't match, so a differently
# NAS'd export can't produce a "valid" GGUF that the loader accepts but the
# forward silently mis-runs. Names are post-"model."-strip; this is a superset of
# the loader's check_shape() guard.
EXPECTED_TENSOR_COUNT = 409
EXPECTED_SHAPES = {
    "erb.erb_fc.weight": (ERB_HIGH, SPEC_BINS - ERB_LOW),    # (64, 192)
    "erb.ierb_fc.weight": (SPEC_BINS - ERB_LOW, ERB_HIGH),   # (192, 64)
    "encoder.en_convs.0.ops.1.weight": (12, 1, 3, 3),        # first enc depthwise (12 ch)
    "encoder.en_convs.4.pconv.0.weight": (16, 16, 1, 1),     # last enc pointwise (16 ch)
    "decoder.de_convs.0.pconv.0.weight": (32, 8, 1, 1),      # first dec pointwise (skip-concat)
    "decoder.de_convs.4.ops.1.weight": (12, 1, 3, 3),        # last dec conv -> mask
    "dpgrnn.0.intra_fc.weight": (16, 16),
    "dpgrnn.1.intra_fc.weight": (16, 16),
}


def check_topology(inits: dict) -> list:
    """Return a list of human-readable problems if `inits` (raw ONNX initializer
    name -> array) doesn't match the expected UL-UNAS topology; empty == OK."""
    stripped = {
        (n[len("model."):] if n.startswith("model.") else n): a
        for n, a in inits.items()
    }
    problems = []
    for name, want in EXPECTED_SHAPES.items():
        if name not in stripped:
            problems.append(f"missing tensor '{name}'")
        elif tuple(stripped[name].shape) != want:
            problems.append(f"'{name}' shape {tuple(stripped[name].shape)} != {want}")
    if ERB_LOW + ERB_HIGH != 129:
        problems.append(f"erb_low+erb_high={ERB_LOW + ERB_HIGH} != 129")
    if len(inits) != EXPECTED_TENSOR_COUNT:
        problems.append(f"tensor count {len(inits)} != {EXPECTED_TENSOR_COUNT}")
    return problems


def f16_ok(name: str, arr: np.ndarray) -> bool:
    """Downcast only the large matmul weights; keep precision-sensitive params
    (affine PReLU, LayerNorm, BatchNorm, ERB filterbank, all biases) in f32."""
    if arr.ndim < 2 or arr.dtype != np.float32:
        return False
    if "_ln." in name or ".erb." in name:
        return False
    base = name.rsplit(".", 1)[-1]
    if base in ("affine_weight", "affine_bias"):
        return False
    return base == "weight" or base.startswith("weight_ih") or base.startswith("weight_hh")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--denoiser", required=True, help="UL-UNAS denoiser ONNX")
    ap.add_argument("--out", required=True, help="output GGUF path")
    ap.add_argument("--ftype", choices=["f32", "f16"], default="f32")
    args = ap.parse_args()

    model = onnx.load(args.denoiser, load_external_data=True)
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    if not inits:
        sys.stderr.write("error: ONNX has no initializers\n")
        return 1

    problems = check_topology(inits)
    if problems:
        sys.stderr.write(
            "error: ONNX does not match the expected UL-UNAS topology that the "
            "tts-cpp core hardwires:\n"
        )
        for p in problems:
            sys.stderr.write(f"  - {p}\n")
        sys.stderr.write(
            "Refusing to write a GGUF the C++ forward would silently mis-run.\n"
        )
        return 1

    with open(args.denoiser, "rb") as fh:
        src_md5 = hashlib.md5(fh.read()).hexdigest()

    writer = GGUFWriter(args.out, ARCH)
    writer.add_uint32("lavasr.denoiser.n_fft", N_FFT)
    writer.add_uint32("lavasr.denoiser.hop", HOP)
    writer.add_uint32("lavasr.denoiser.win", WIN)
    writer.add_uint32("lavasr.denoiser.spec_bins", SPEC_BINS)
    writer.add_uint32("lavasr.denoiser.work_sample_rate", WORK_SR)
    writer.add_uint32("lavasr.denoiser.erb_low", ERB_LOW)
    writer.add_uint32("lavasr.denoiser.erb_high", ERB_HIGH)
    writer.add_uint32("lavasr.denoiser.freq_comp_ratio", FREQ_COMP_RATIO)
    writer.add_uint32("lavasr.denoiser.chunk_frames", CHUNK_FRAMES)
    writer.add_uint32("lavasr.denoiser.chunk_hop", CHUNK_HOP)
    writer.add_float32("lavasr.denoiser.batchnorm_eps", BN_EPS)
    writer.add_float32("lavasr.denoiser.layernorm_eps", LN_EPS)
    # Provenance: which ONNX this GGUF was converted from (name + MD5), so a
    # baked model can be traced back to its source export.
    writer.add_string("lavasr.denoiser.source_onnx", os.path.basename(args.denoiser))
    writer.add_string("lavasr.denoiser.source_md5", src_md5)

    print("tensors:")
    n_f16 = 0
    for name in sorted(inits):
        arr = np.ascontiguousarray(inits[name])
        # Drop the wrapper "model." prefix -> encoder.* / decoder.* / dpgrnn.* / erb.*
        out_name = name[len("model."):] if name.startswith("model.") else name
        if f16_ok(name, arr) and args.ftype == "f16":
            arr = arr.astype(np.float16)
            n_f16 += 1
        elif arr.dtype not in (np.float32, np.float16):
            arr = arr.astype(np.float32)
        writer.add_tensor(out_name, arr)
        print(f"  {out_name:52s} {str(arr.dtype):8s} {list(arr.shape)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nWrote {args.out} (arch={ARCH}, ftype={args.ftype}, "
          f"tensors={len(inits)}, f16={n_f16})")
    print(f"source: {os.path.basename(args.denoiser)} md5={src_md5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
