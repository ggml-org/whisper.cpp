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
    "flow/spk_embed_affine",    # speaker-embedding affine (w + b): read as raw F32 by
                                # cached_cpu_weights_f32 in the s3gen synth path, so a
                                # block-quantized tensor would be byte-reinterpreted as
                                # float -> NaN speaker embedding -> all-NaN mel -> noise.
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


# Supertonic (v1/v2/v3) keep-as-source-dtype roster.  Unlike the chatterbox
# _DENY_SUBSTRINGS roster above, Supertonic tensors are stored under OPAQUE
# names (`supertonic/<stage>/tNNNN`) plus the descriptive `voices/<id>/ttl`,
# so a substring deny against the storage name catches nothing.  Instead we
# resolve each opaque name back to its *logical* PyTorch/ONNX source name via
# the converter-emitted alias arrays (supertonic.source_names /
# .tensor_names / .source_aliases / .source_alias_targets) and deny by
# logical-name substring.
#
# The roster targets exactly the tensors the C++ loader reads as **raw F32**
# (uploaded as graph inputs / constants, NOT consumed through ggml's
# dequantizing matmul/get_rows paths).  Block-quantizing those reinterprets
# Q4/Q8 bytes as floats -> NaN -> a crash (q4_0) or decorrelated audio
# (q8_0).  The genuine projection weights (`onnx::MatMul_*`, `*.linear.weight`)
# stay quantizable because ggml dequantizes them in-op.
#
# Why this only began biting in v3: the CFG `uncond_masker.*_special_token`
# tensors are [50, 256] (last-dim 256 % 32 == 0 -> quantizable) AND read raw,
# and they simply do not exist in v1/v2 (no classifier-free guidance there).
_SUPERTONIC_KEEP_F32_LOGICAL = (
    "Expand_output",        # vector_estimator style-key constant (/Expand_output_0)
    "uncond_masker",        # v3 CFG null text/style tokens (read raw)
    "special_token",        # ditto (style_key/value/text_special_token)
    "char_embedder",        # text/duration char embedding tables
    "text_embedder",        # embedder wrappers
    "sentence_token",       # duration sentence-encoder learned token
    "style_key",            # style-token-layer learned key (read raw)
    "style_value",
    "style_token",
    "rope",                 # rotary position frequencies
    "theta",
)
# Storage-name (opaque) substrings — for tensors with no logical alias.
_SUPERTONIC_KEEP_F32_STORAGE = (
    "/voices/",             # per-voice conditioning embeddings (ttl), read raw
    # The duration predictor runs as a scalar CPU continuation that reads
    # EVERY weight through `read_f32` / `cached_read_f32` (a raw F32 byte
    # copy, no in-op dequant), so any block-quantized duration weight is
    # reinterpreted as floats -> SIGBUS (q4_0) / corrupted phoneme timings
    # -> fully decorrelated audio (q8_0).  The stage is tiny, so keeping it
    # at source dtype costs almost nothing.  (The text/vector matmuls stay
    # quantizable: those flow through ggml mul_mat, which dequantizes in-op.)
    "supertonic/duration/",
)


def build_supertonic_keep_f32(src: gguf.GGUFReader) -> set[str]:
    """Resolve the Supertonic raw-read roster to a set of opaque storage
    tensor names that must NOT be quantized.  Returns empty set when the
    alias metadata is absent (non-Supertonic GGUF)."""
    def _arr(key: str) -> list[str]:
        fld = src.fields.get(key)
        if fld is None:
            return []
        return [bytes(fld.parts[i]).decode("utf-8", "replace") for i in fld.data]

    source_names = _arr("supertonic.source_names")
    tensor_names = _arr("supertonic.tensor_names")
    alias_names = _arr("supertonic.source_aliases")
    alias_targets = _arr("supertonic.source_alias_targets")

    # opaque storage name -> set of logical source names pointing at it.
    storage_to_logical: dict[str, set[str]] = {}
    for logical, storage in zip(source_names, tensor_names):
        storage_to_logical.setdefault(storage, set()).add(logical)
    # aliases map an extra logical name onto an existing source name; chase
    # them to the same storage tensor so e.g. `...W_key.linear.weight` and its
    # `onnx::MatMul_*` bridge alias both resolve.
    logical_to_storage: dict[str, str] = {
        logical: storage for logical, storage in zip(source_names, tensor_names)
    }
    for alias, target in zip(alias_names, alias_targets):
        storage = logical_to_storage.get(target)
        if storage is not None:
            storage_to_logical.setdefault(storage, set()).add(alias)

    keep: set[str] = set()
    for storage, logicals in storage_to_logical.items():
        if any(any(p in lg for p in _SUPERTONIC_KEEP_F32_LOGICAL) for lg in logicals):
            keep.add(storage)
    # Storage-name fallback (voices have no logical alias).
    for t in src.tensors:
        if any(p in t.name for p in _SUPERTONIC_KEEP_F32_STORAGE):
            keep.add(t.name)
    return keep


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
        # Genuine K-aligned conv kernels (K a multiple of the block).  Rare
        # in this stack (HiFT K in {3,7,11,16} never qualifies).
        if shape[-1] % block == 0:
            return True
        # Pointwise (1x1) conv: stored 3-D as ggml ne=[K=1, IC, OC] (numpy
        # (OC, IC, 1)).  Algebraically it's a 2-D matmul [IC, OC]; the K=1
        # axis is trivial.  Squeeze the singleton and gate on the real
        # reduction dim (IC).  When quantized we store it squeezed-2D and the
        # C++ loader re-expands it to [1, IC, OC] at load (driven by the
        # `supertonic.pwconv_squeezed` roster this script emits).  This is
        # where the bulk of a Supertonic GGUF lives (ConvNeXt pwconv1/pwconv2
        # in the vocoder + vector_estimator), so unlocking it is what takes a
        # Q8_0 GGUF from ~96% of F32 down to ~32%.
        squeezed = tuple(d for d in shape if d != 1)
        if len(squeezed) == 2 and squeezed[-1] % block == 0:
            return True
        return False

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

    # Supertonic stores weights under opaque `tNNNN` names; resolve the
    # raw-read roster via the alias arrays so we never block-quantize a
    # tensor the C++ loader reads as raw F32 (voices, CFG null tokens,
    # style/expand constants, embedding tables).  Empty for other archs.
    keep_f32_storage: set[str] = set()
    if arch_name.startswith("supertonic"):
        keep_f32_storage = build_supertonic_keep_f32(src)

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
    # Storage names of pointwise (K=1) convs we squeezed 3-D [1,IC,OC] -> 2-D
    # [IC,OC] before quantizing.  The C++ loader re-expands these to 3-D at
    # load so the conv/matmul call sites see the original shape.
    pwconv_squeezed_names: list[str] = []

    for t in src.tensors:
        # GGUFReader returns shape in numpy-style reversed order.
        shape = tuple(int(d) for d in reversed(t.shape) if d > 0)
        if not shape:
            shape = (int(t.shape[0]),)

        data = np.asarray(t.data)
        src_bytes += data.nbytes

        in_filter = name_filter is None or name_filter in t.name
        if (in_filter and t.name not in keep_f32_storage
                      and t.tensor_type in _QUANTIZABLE_SRC_DTYPES
                      and t.tensor_type != qtype
                      and should_quantize(t.name, shape, qtype)):
            # Reshape to natural (shape).  GGUF raw data is contiguous in
            # the original order, but reversed() above gives element-shape
            # which is what `quantize()` expects.
            #
            # Pointwise (1x1) conv: squeeze the singleton K dim so `quantize`
            # sees a 2-D [.., IC] matrix it can block-quantize along IC.  The
            # squeezed-2D bytes are layout-identical to the 3-D [1,IC,OC]
            # tensor, so the loader re-expands by name (no permutation).
            quant_shape = shape
            q_block = gguf.GGML_QUANT_SIZES[qtype][0]
            squeezed_pwconv = (
                len(shape) == 3 and (shape[-1] % q_block != 0) and (1 in shape)
            )
            if squeezed_pwconv:
                quant_shape = tuple(d for d in shape if d != 1)
            arr = data.astype(np.float32).reshape(quant_shape)
            qdata = gguf.quants.quantize(arr, qtype)
            writer.add_tensor(t.name, qdata, raw_shape=qdata.shape, raw_dtype=qtype)
            if squeezed_pwconv:
                pwconv_squeezed_names.append(t.name)
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

    # Emit the pointwise-conv re-expansion roster so the C++ loader restores
    # the original 3-D [1,IC,OC] shape for these squeezed-2-D quantized
    # tensors.  Only present when we actually squeezed something.
    if pwconv_squeezed_names:
        writer.add_array("supertonic.pwconv_squeezed", pwconv_squeezed_names)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"arch: {arch_name!r}")
    print(f"quantized: {quantized_count} tensors to {args.dtype.upper()}")
    if pwconv_squeezed_names:
        print(f"           ({len(pwconv_squeezed_names)} of them pointwise K=1 convs "
              f"squeezed 3D->2D; loader re-expands via supertonic.pwconv_squeezed)")
    print(f"kept:      {kept_count} tensors as source dtype")
    if keep_f32_storage:
        print(f"           ({len(keep_f32_storage)} of them via the Supertonic "
              f"raw-read roster: voices / CFG null tokens / style+expand "
              f"constants / embeddings)")
    print(f"size:      {src_bytes / 1e6:.1f} MB  →  {dst_bytes / 1e6:.1f} MB  "
          f"({dst_bytes / src_bytes * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
