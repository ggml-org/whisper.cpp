#!/usr/bin/env python3
"""Requantize a chatterbox GGUF (T3 or S3Gen) to a smaller dtype.

`llama-quantize` refuses to touch either GGUF because neither
`chatterbox` nor `chatterbox-s3gen` is a llama.cpp-known arch.  This
tool walks the GGUF tensor-by-tensor and rewrites it with the big 2-D
weight matrices stored as `Q8_0` / `Q5_0` / `Q4_0`, leaving the
numerically-sensitive tensors (embedding tables accessed via get_rows,
biases, norm scales, filterbank / STFT bases, positional embeddings,
builtin voice conditioning) at their source dtype.

Works for both models because the deny-list covers the union of
patterns that either side uses for "keep-as-F32/F16".

Usage:

    # T3 Q8_0
    python scripts/requantize-gguf.py \\
        models/chatterbox-t3-turbo.gguf \\
        models/t3-q8_0.gguf q8_0

    # S3Gen Q8_0
    python scripts/requantize-gguf.py \\
        models/chatterbox-s3gen.gguf \\
        models/chatterbox-s3gen-q8_0.gguf q8_0

    # Q4_0 is the same, last arg is just `q4_0`.

    # F16 downcast for HiFT conv kernels (multilingual S3Gen — see §3.24).
    # `--name-filter hift/` constrains the rewrite to a name substring;
    # everything else is passed through at its source dtype.  Two-pass
    # use:
    #   1. F32→F16 for HiFT conv kernels in the F16 source GGUF
    #   2. F16→Q4_0 for the CFM transformer linears (no name filter)
    python scripts/requantize-gguf.py \\
        models/chatterbox-s3gen-mtl-f16.gguf \\
        /tmp/intermediate.gguf f16 --name-filter hift/
    python scripts/requantize-gguf.py \\
        /tmp/intermediate.gguf \\
        models/chatterbox-s3gen-mtl-q4_0_hift_f16.gguf q4_0

Quality trade-off (measured on a representative paragraph, Metal / M3 Ultra):
  F32 (default)   — baseline
  Q8_0            — essentially bit-exact, cos-sim > 0.99 vs baseline
  Q4_0            — different CFM ODE trajectory → different sample;
                    subjective quality equal, cos-sim falls to ~0.66
  F16 (--name-filter hift/) — HiFT conv kernels at half precision; PCM
                    cosine 0.9999 vs the corresponding all-F32-HiFT
                    baseline (audio essentially indistinguishable).
                    `[hift_decode]` ~3 % faster on M3 Ultra Metal
                    (124.9 → 121.3 ms median across 3 invocations);
                    GGUF ~33 MB smaller.  See PROGRESS.md §3.24.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import gguf


# Names we NEVER touch: they're read as raw F32 by the C++ loader, or
# they're accessed via ggml_get_rows (embedding tables), or they're
# numerically sensitive (filterbanks, STFT bases, voice conditioning,
# position embeddings, norm/bias params).  Works for both T3 (GPT-2-
# style names) and S3Gen (custom per-module names).
_DENY_SUBSTRINGS = (
    # Raw-F32 access in the C++ loader
    "flow/input_embedding",     # S3Gen speech embedding table (read as F32 for CPU-side lookup)
    "/builtin/",                # voice conditioning tensors, loaded directly
    # Embedding tables (accessed via ggml_get_rows — safer as F16/F32)
    "text_emb",                 # T3 text token embedding
    "speech_emb",               # T3 speech token embedding
    "wte",                      # GPT-2 word token embedding
    "wpe",                      # GPT-2 learned position embedding
    # Spectral bases / positional encodings (bit-exact numerics)
    "stft_basis",               # STFT analysis / synthesis
    "mel_filterbank",           # mel filterbank
    "mel_fb",                   # T3 VoiceEncoder and S3Gen mel filterbank tensors
    "pos_emb",                  # positional embeddings — small, keep F32
    "pe/pe",                    # conformer pos enc
    "pre_attention_query",      # MTL T3 perceiver: learned query embedding
                                # (CLS-like).  Used as an *activation* (passed
                                # as the right-hand side of mul_mat after
                                # reshape), not a weight, so quantising it
                                # breaks ggml_reshape_2d / ggml_norm /
                                # ggml_mul_mat-as-src1 in build_perceiver.
                                # Pre-existing latent bug: was always wrongly
                                # quantizable (3-D shape (1024, 32, 1) clears
                                # the K%32==0 gate); only surfaced now because
                                # the shipped q4_0 GGUF was produced via an
                                # earlier code path that kept it at source
                                # dtype.
    # Biases / norms / scale params — always 1-D or near-1-D
    "/b",                       # legacy biases (gpt-2 /b, s3gen /b)
    "/bias",                    # pytorch-style bias
    "/bn/",                     # batchnorm params
    "/norm/",                   # layernorms
    "/ln_",                     # GPT-2 style layernorms (ln_1, ln_2, ln_f)
    "/scale",                   # legacy scale weights (narrowed from the
                                # old "/s" glob so HiFT source_* conv
                                # weights are no longer incidentally
                                # excluded.  The `kernel_mul_mv_f32_f16`
                                # / `_4` / `_short` Metal kernel variants
                                # that HiFT source_* conv1d needs are
                                # shipped in patches/ggml-metal-
                                # chatterbox-ops.patch as of PROGRESS
                                # §3.26, so this deny is no longer
                                # necessary for correctness.  With the
                                # kernel in place, the 21 source_*
                                # conv-kernel weights go through the
                                # --name-filter hift/ recipe at f16 and
                                # the GGUF shrinks by ~7.7 MB with WAV
                                # parity (cos 1.000000, rms-diff 0.035 %,
                                # max abs 4/32767).  See §3.26.)
    "alpha",                    # Snake activation alphas
    "beta",
    "gamma",
    # Voice-cloning preprocessing encoders — NEVER quantize.  These are
    # small specialised models whose dynamic range is too tight for Q4/Q8
    # block quantization; the resulting encoder output drifts so badly that
    # the voice-cloning tensors become unusable (we've seen speaker_emb
    # collapse to zeros, prompt_token to a single constant value, and
    # CAMPPlus embedding go antipodal to its F32 counterpart).  Keeping
    # them at source dtype costs ~40 MB across both GGUFs but is the
    # difference between a working clone and garbage audio.
    "voice_encoder/",           # T3 VoiceEncoder (3-layer bi-LSTM + projection)
    "campplus/",                # S3Gen CAMPPlus (TDNN x-vector extractor)
    "s3tokv2/",                 # S3Gen S3TokenizerV2 (conformer + FSQ quantizer)
)


# Suffix-anchored denies.  Use this for one-letter param names that would
# otherwise hit too many incidental substring matches.  The classic case
# is the GPT-2 / Llama RMSNorm scale tensor `.../ln_attn/g`, `.../norm/g`:
# matched as a substring, "/g" also wrongly catches `.../mlp/gate/w` (30
# tensors × ~4 MB each ≈ 120 MB on the multilingual T3 Q4_0 GGUF) and is
# the reason §3.23 observed `mlp_gate` shipping as F16 while `mlp_up`
# shipped as Q4_0 — a converter bug, not by design.
_DENY_SUFFIXES = (
    "/g",                       # GPT-2 / Llama RMSNorm / LayerNorm scale at end of path
)


# Tensor element dtypes we're willing to quantize from.  F16 is T3's
# default for its big projection weights; F32 is S3Gen's default.
_QUANTIZABLE_SRC_DTYPES = {
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
}


_QUANT_TYPE = {
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
    # F16 is a downcast, not a block quant — block_size = 1 in
    # GGML_QUANT_SIZES, so the shape gates in should_quantize accept any
    # 2-D / 3-D weight tensor.  Useful for the 3-D HiFT conv kernels
    # (K in {3, 7, 11, 16}) that none of the 32-block quants can take.
    "f16":  gguf.GGMLQuantizationType.F16,
}


def should_quantize(name: str, shape: tuple[int, ...], qtype: gguf.GGMLQuantizationType) -> bool:
    # Keep tiny tensors at full precision.
    n_elements = 1
    for d in shape:
        n_elements *= d
    if n_elements < 1024:
        return False

    # Deny-list.
    for s in _DENY_SUBSTRINGS:
        if s in name:  # case-sensitive for path-like names
            return False
    for s in _DENY_SUFFIXES:
        if name.endswith(s):  # one-letter param names that would over-match as substring
            return False

    block = gguf.GGML_QUANT_SIZES[qtype][0]

    # 2D matmul weights: ggml shape (ne0, ne1) = (reduction_dim, output).
    # GGUFReader exposes shape in numpy (reversed) order, so the
    # reduction dim is shape[-1].  Quantization quantises along the
    # last numpy axis, so shape[-1] must be a multiple of the block.
    if len(shape) == 2:
        return shape[-1] % block == 0

    # 3D conv kernels: ggml shape (K, IC, OC) -> numpy (OC, IC, K).
    # `gguf.quants.quantize` quantises along the LAST numpy axis, which is K
    # for a conv kernel.  HiFT conv kernels have K in {3, 7, 11, 16}; none
    # are multiples of any block size we ship here (32).
    #
    # Quantising along K*IC instead would need a numpy reshape to
    # (OC, K*IC) before `quantize` and then storing the result with ggml
    # shape (K*IC, OC) — i.e. a 2-D on-disk tensor.  But the C++ side's
    # `conv1d_f32` calls `ggml_im2col(kernel, ...)` which derives the
    # kernel size from `kernel->ne[0]`; collapsing K into a flattened
    # (K*IC) ne[0] would silently break im2col window extraction.
    #
    # So 3-D quantisation only works when K alone meets the block-size
    # constraint.  We still gate on it (instead of returning False
    # outright) so any future converter that ships K-aligned conv
    # kernels gets the win for free; for the current HiFT stack this
    # path stays a no-op and the caller logs the kept-as-source-dtype
    # tensors via stats.kept.
    if len(shape) == 3:
        return shape[-1] % block == 0

    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="Source GGUF (F32/F16)")
    ap.add_argument("dst", type=Path, help="Output GGUF")
    ap.add_argument("dtype", choices=_QUANT_TYPE.keys(), help="Target quant dtype")
    ap.add_argument(
        "--name-filter",
        default=None,
        help=("Substring filter on tensor names; only tensors whose name "
              "contains this substring are touched.  All other tensors "
              "are passed through at their source dtype.  Useful for "
              "applying f16 to HiFT conv kernels in a Q4_0 source GGUF "
              "without disturbing the existing Q4_0 CFM weights."),
    )
    args = ap.parse_args()

    qtype = _QUANT_TYPE[args.dtype]
    name_filter = args.name_filter

    src = gguf.GGUFReader(args.src, "r")
    arch = src.fields.get("general.architecture")
    arch_name = ""
    if arch is not None:
        arch_name = bytes(arch.parts[arch.data[0]]).decode("utf-8")

    writer = gguf.GGUFWriter(args.dst, arch_name or "chatterbox-s3gen")

    # Copy all metadata (KV fields) verbatim.  Skip the ones the writer
    # sets itself to avoid duplicates.
    _SKIP_KEYS = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "general.architecture",
    }
    for key, field in src.fields.items():
        if key in _SKIP_KEYS:
            continue
        val_type = field.types[0] if field.types else None
        parts = [field.parts[i] for i in field.data]
        if val_type is None:
            continue
        if val_type == gguf.GGUFValueType.ARRAY:
            sub_type = field.types[1] if len(field.types) > 1 else None
            if sub_type == gguf.GGUFValueType.STRING:
                values = [bytes(p).decode("utf-8") for p in parts]
                writer.add_array(key, values)
            else:
                arr = np.concatenate([np.asarray(p) for p in parts]).tolist()
                writer.add_array(key, arr)
        elif val_type == gguf.GGUFValueType.STRING:
            writer.add_string(key, bytes(parts[0]).decode("utf-8"))
        elif val_type == gguf.GGUFValueType.BOOL:
            writer.add_bool(key, bool(parts[0][0]))
        elif val_type in (gguf.GGUFValueType.UINT8, gguf.GGUFValueType.UINT16,
                          gguf.GGUFValueType.UINT32, gguf.GGUFValueType.UINT64):
            writer.add_uint32(key, int(parts[0][0]))
        elif val_type in (gguf.GGUFValueType.INT8, gguf.GGUFValueType.INT16,
                          gguf.GGUFValueType.INT32, gguf.GGUFValueType.INT64):
            writer.add_int32(key, int(parts[0][0]))
        elif val_type in (gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64):
            writer.add_float32(key, float(parts[0][0]))

    quantized_count = 0
    kept_count = 0
    src_bytes = 0
    dst_bytes = 0

    for t in src.tensors:
        # GGUFReader returns shape in numpy-style reversed order.
        shape = tuple(int(d) for d in reversed(t.shape) if d > 0)
        if not shape:
            shape = (int(t.shape[0]),)

        data = np.asarray(t.data)
        src_bytes += data.nbytes

        in_filter = name_filter is None or name_filter in t.name
        if (in_filter and t.tensor_type in _QUANTIZABLE_SRC_DTYPES
                      and t.tensor_type != qtype
                      and should_quantize(t.name, shape, qtype)):
            # Reshape to natural (shape).  GGUF raw data is contiguous in
            # the original order, but reversed() above gives element-shape
            # which is what `quantize()` expects.
            arr = data.astype(np.float32).reshape(shape)
            qdata = gguf.quants.quantize(arr, qtype)
            writer.add_tensor(t.name, qdata, raw_shape=qdata.shape, raw_dtype=qtype)
            quantized_count += 1
            dst_bytes += qdata.nbytes
        else:
            # Pass through unchanged.  Preserve original dtype.
            #
            # For already-quantised inputs (Q-type sources) the GGUF data
            # is opaque packed bytes (Q4_0: 18 B / 32 elements ≈ 0.56 B
            # per element), so a numpy-shape reshape against the
            # element-shape would fail with a size-mismatch.  Float-type
            # sources have block_size=1 in GGML_QUANT_SIZES so the
            # reshape works as before.
            block_size, type_size = gguf.GGML_QUANT_SIZES[t.tensor_type]
            if block_size == 1:
                arr = data.reshape(shape)
                writer.add_tensor(t.name, arr, raw_shape=arr.shape, raw_dtype=t.tensor_type)
            else:
                # Q-type passthrough.  gguf-0.18+ `add_tensor_info` treats
                # `raw_shape` as **byte shape** for uint8 tensors (the
                # innermost dim is bytes per row, not elements per row).
                # Convert: byte_inner = elements_inner / block * type_size.
                # Earlier versions of this script hit
                # `ValueError: Quantized tensor bytes per row (N) is not a
                # multiple of Q4_0 type size (18)` when re-quantising a
                # GGUF that already had Q-type tensors — see §3.26.
                byte_inner = shape[-1] // block_size * type_size
                byte_shape = tuple(list(shape[:-1]) + [byte_inner])
                writer.add_tensor(t.name, data, raw_shape=byte_shape, raw_dtype=t.tensor_type)
            kept_count += 1
            dst_bytes += data.nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"arch: {arch_name!r}")
    print(f"quantized: {quantized_count} tensors to {args.dtype.upper()}")
    print(f"kept:      {kept_count} tensors as source dtype")
    print(f"size:      {src_bytes / 1e6:.1f} MB  →  {dst_bytes / 1e6:.1f} MB  "
          f"({dst_bytes / src_bytes * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
