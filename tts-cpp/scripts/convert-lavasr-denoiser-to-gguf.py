#!/usr/bin/env python3
"""Convert the LavaSR denoiser (UL-UNAS) ONNX into a single GGUF for the tts-cpp
CPU/GGML denoiser.

SCAFFOLD — QVAC-16579 follow-up (LavaSR denoiser stage).  The tensor mapping is
stubbed until the reference ONNX is finalised; this file pins down the CLI and
the planned GGUF layout so the implementation can drop straight in.  See the
working reference for conventions: convert-lavasr-enhancer-to-gguf.py.

LavaSR is a two-stage pipeline:
  (1) UL-UNAS denoiser  [this script]                         -> cleaned audio
  (2) Vocos BWE enhancer [convert-lavasr-enhancer-to-gguf.py] -> 48 kHz

UL-UNAS (arXiv:2503.00340, github.com/Xiaobin-Rong/ul-unas) is a TF-domain U-Net:
  * STFT input -> encoder (efficient depthwise-separable conv blocks + affine
    PReLU) -> bottleneck -> decoder (with U-Net skips) -> mask -> ISTFT
  * causal time-frequency attention (cTFA) per block:
        TA = GRU   -> FC -> Sigmoid                     (temporal, causal)
        FA = BiGRU over folded frequency (R=4) -> FC -> Sigmoid (spectral)

Planned GGUF layout (mirrors convert-lavasr-enhancer-to-gguf.py conventions):
  ARCH = "lavasr-denoiser"
  metadata: n_fft, hop, win, spec_bins, work_sample_rate, freq_fold_r, ...
  Conv weights:  [out, in/groups, K]       (ONNX order)
  Linear/GRU W:  transposed to [out, in]   (so the C++ scalar core reads
                 ne=[in,out] and does y[o] = sum_i W[o,i] x[i] + b[o])
  GRU gates:     W_ir/W_iz/W_in, W_hr/W_hz/W_hn + biases; BiGRU: fwd + bwd sets
  affine PReLU:  per-channel alpha (slope) + beta (affine shift)

Usage (planned):
  python convert-lavasr-denoiser-to-gguf.py \\
      --denoiser ul_unas_denoiser.onnx \\
      --out      lavasr-denoiser.gguf \\
      --ftype    f16            # or f32
"""
import argparse
import sys

ARCH = "lavasr-denoiser"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--denoiser", required=True, help="UL-UNAS denoiser ONNX")
    ap.add_argument("--out", required=True, help="output GGUF path")
    ap.add_argument("--ftype", choices=["f32", "f16"], default="f16")
    ap.parse_args()

    # TODO(QVAC-16579 follow-up: LavaSR denoiser stage): load the ONNX, map the
    # UL-UNAS layers to the tensor names/layout documented above, and write the
    # metadata + weights with gguf.GGUFWriter (see the enhancer converter for a
    # complete, working reference).
    sys.stderr.write(
        "convert-lavasr-denoiser-to-gguf.py is a scaffold: the UL-UNAS "
        "ONNX->GGUF mapping is not implemented yet (QVAC-16579 follow-up: "
        "LavaSR denoiser stage). See the module docstring for the planned "
        "layout.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
