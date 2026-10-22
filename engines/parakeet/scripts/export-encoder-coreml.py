#!/usr/bin/env python3
"""Export a Parakeet CTC/IndicConformer, Unified RNN-T, TDT, EOU, Nemotron, or Sortformer v2.1
FastConformer encoder
from GGUF to a Core ML package consumed by the parakeet.cpp Engine
(the encoder I/O contract lives in src/coreml/parakeet-encoder.h).

The exported graph reuses the pure-PyTorch reference encoder in
ref-encoder-from-gguf.py so it matches the ggml encoder numerically.

Three input-shape modes:
  - Fixed (default): torch.jit.trace at a single mel length (from a sample wav
    or an explicit count). Unified RNN-T and TDT treat that length as a
    capacity. Fixed TDT adds a mel_valid_mask input to isolate padding throughout
    subsampling and the Conformer stack. CTC/IndicConformer uses the same
    capacity contract. EOU,
    Nemotron, and Sortformer batch routing require the exact
    exported length.
  - Sortformer AOSC (--bypass-pre-encode): trace only the Conformer block stack
    at a fixed encoder-frame capacity. The runtime pads shorter cache/FIFO/chunk
    slabs, supplies an attention-validity mask, and crops the output.
  - Flexible (--flexible): export a Core ML RangeDim time axis via torch.export
    so one encoder serves any mel length in [--min-frames, --max-frames] (others
    fall back to ggml). CORRECTNESS-ONLY, NOT FOR ACCELERATION: measured on Apple
    M3, the shape-generic graph places 0 ops on the Neural Engine and runs on CPU
    (~8x slower than the ggml-Metal encoder at ~2000 frames); Core ML only puts
    the clean fixed-shape jit.trace graph on the ANE. Use --flexible for
    numerical experiments; for ANE speedup export a fixed single length (the
    default), which ran ~1.4x faster than ggml-Metal at ~2000 frames. The graph
    is still numerically identical to the fixed path (masks dropped -- a no-op
    for full-valid input, positional table baked as a buffer, rel-shift as an
    int32 gather; see rel_pos_mha_flex). Needs coremltools>=8 / torch>=2.3. The
    exporter prints the ANE/CPU op placement when --compile-dir is set.

Example:

  python scripts/export-encoder-coreml.py \
      --gguf   models/parakeet-tdt-0.6b-v3.f16.gguf \
      --n-mel-frames 1501 \
      --palettize-bits 6 --palettize-group-size 16 \
      --out    models/parakeet-tdt-0.6b-v3-encoder.mlpackage \
      --compile-dir models

  # Unified RNN-T fixed-capacity encoder:
  python scripts/export-encoder-coreml.py \
      --gguf models/parakeet-unified-en-0.6b.q8_0.gguf \
      --n-mel-frames 1501 \
      --palettize-bits 6 --palettize-group-size 16 \
      --out models/parakeet-unified-en-0.6b-encoder.mlpackage \
      --compile-dir models

  # IndicConformer multilingual CTC fixed-capacity encoder:
  python scripts/export-encoder-coreml.py \
      --gguf models/indic-conformer-600m-multilingual.f16.gguf \
      --n-mel-frames 1501 \
      --palettize-bits 6 --palettize-group-size 16 \
      --out models/indic-conformer-600m-multilingual-encoder.mlpackage \
      --compile-dir models

  # EOU fixed shape (the 11-second fixture produces 1101 mel frames):
  python scripts/export-encoder-coreml.py \
      --gguf models/parakeet_realtime_eou_120m-v1.f16.gguf \
      --wav test/samples/jfk.wav \
      --palettize-bits 6 --palettize-group-size 16 \
      --out models/parakeet_realtime_eou_120m-v1-encoder.mlpackage \
      --compile-dir models

  # Sortformer v2.1 batch path: mel -> FastConformer output.
  python scripts/export-encoder-coreml.py \
      --gguf models/diar_streaming_sortformer_4spk-v2.1.f16.gguf \
      --wav test/samples/diarization-sample-16k.wav \
      --palettize-bits 6 --palettize-group-size 16 \
      --out models/diar_streaming_sortformer_4spk-v2.1-encoder.mlpackage \
      --compile-dir models

  # Sortformer v2.1 AOSC path: cached pre-encode embeddings -> encoder output.
  # 410 covers the default 188-row speaker cache, 188-row FIFO, and the
  # subsampled 80 ms + 2000 ms + 560 ms context/chunk window.
  python scripts/export-encoder-coreml.py \
      --gguf models/diar_streaming_sortformer_4spk-v2.1.f16.gguf \
      --bypass-pre-encode --n-encoder-frames 410 \
      --palettize-bits 6 --palettize-group-size 16 \
      --out models/diar_streaming_sortformer_4spk-v2.1-encoder-bypass-pre-encode.mlpackage \
      --compile-dir models

  # variable-length (best-effort; validate on device):
  python scripts/export-encoder-coreml.py \
      --gguf models/parakeet-tdt-0.6b-v3.f16.gguf --wav test/samples/jfk.wav \
      --flexible --max-frames 3000 \
      --out models/parakeet-tdt-0.6b-v3-encoder.mlpackage --compile-dir models
"""
import argparse
import importlib.util
import math
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct


def load_reference_encoder(scripts_dir):
    path = scripts_dir / "ref-encoder-from-gguf.py"
    spec = importlib.util.spec_from_file_location("parakeet_ref_encoder", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BiasTolerantWeights(dict):
    def __missing__(self, key):
        if key.endswith(".bias"):
            return None
        raise KeyError(key)


def resolve_hop_length(meta):
    for key, value in meta.items():
        if "hop" in key.lower():
            return int(value)
    return 160


def mel_frames_for_wav(wav_path, hop_length):
    with wave.open(str(wav_path), "rb") as reader:
        num_samples = reader.getnframes()
    return 1 + num_samples // hop_length


def model_type(meta):
    return str(meta.get("parakeet.model.type", "ctc"))


def validate_export_contract(meta, flexible=False, bypass_pre_encode=False):
    kind = model_type(meta)
    if kind not in ("ctc", "rnnt", "tdt", "eou", "nemotron", "sortformer"):
        raise ValueError(
            "Core ML encoder export supports CTC/IndicConformer, Unified RNN-T, "
            "TDT, EOU, Nemotron, and "
            f"Sortformer v2.1, got {kind!r}")
    if bypass_pre_encode and kind != "sortformer":
        raise ValueError("--bypass-pre-encode is supported only for Sortformer v2.1")
    if kind in ("ctc", "rnnt", "tdt"):
        family = {"ctc": "CTC/IndicConformer", "rnnt": "Unified RNN-T", "tdt": "TDT"}[kind]
        if bool(meta.get("parakeet.encoder.causal_downsampling", False)):
            raise ValueError(
                f"{family} Core ML export requires non-causal downsampling")
        if str(meta.get(
                "parakeet.encoder.conv_context_size", "default")) == "causal":
            raise ValueError(
                f"{family} Core ML export requires non-causal convolution")
        if str(meta.get(
                "parakeet.encoder.conv_norm_type", "batch_norm")) != "batch_norm":
            raise ValueError(
                f"{family} Core ML export requires batch-normalized convolution")
        left = int(meta.get("parakeet.encoder.att_context_size_left", -1))
        right = int(meta.get("parakeet.encoder.att_context_size_right", -1))
        if left >= 0 or right >= 0:
            raise ValueError(
                f"{family} Core ML export requires unbounded attention context")
    if kind == "sortformer":
        variant = str(meta.get("parakeet.model_variant", ""))
        if variant != "sortformer-streaming-v2.1-aosc":
            raise ValueError(
                "Sortformer Core ML export requires "
                "parakeet.model_variant='sortformer-streaming-v2.1-aosc', "
                f"got {variant!r}")
        if flexible:
            raise ValueError("Sortformer Core ML export requires a fixed shape; omit --flexible")
        if bool(meta.get("parakeet.encoder.causal_downsampling", False)):
            raise ValueError("Sortformer Core ML export requires non-causal downsampling")
        if str(meta.get("parakeet.encoder.conv_context_size", "default")) == "causal":
            raise ValueError("Sortformer Core ML export requires non-causal convolution")
        if str(meta.get("parakeet.encoder.att_context_style", "regular")) != "regular":
            raise ValueError("Sortformer Core ML export requires regular full-context attention")
        left = int(meta.get("parakeet.encoder.att_context_size_left", -1))
        right = int(meta.get("parakeet.encoder.att_context_size_right", -1))
        if left >= 0 or right >= 0:
            raise ValueError("Sortformer Core ML export requires unbounded attention context")
    if kind == "eou":
        if flexible:
            raise ValueError("EOU Core ML export requires a fixed shape; omit --flexible")
        expected = {
            "parakeet.encoder.causal_downsampling": True,
            "parakeet.encoder.conv_context_size": "causal",
            "parakeet.encoder.conv_norm_type": "layer_norm",
            "parakeet.encoder.att_context_style": "chunked_limited",
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"unsupported EOU encoder metadata: {key}={meta.get(key)!r}, expected {value!r}")
        if int(meta.get("parakeet.encoder.att_context_size_left", -1)) < 0 or \
           int(meta.get("parakeet.encoder.att_context_size_right", -1)) < 0:
            raise ValueError("EOU Core ML export requires finite attention context metadata")
    if kind == "nemotron":
        if flexible:
            raise ValueError("Nemotron Core ML export requires a fixed shape; omit --flexible")
        expected = {
            "parakeet.encoder.causal_downsampling": True,
            "parakeet.encoder.conv_context_size": "causal",
            "parakeet.encoder.conv_norm_type": "layer_norm",
            "parakeet.encoder.att_context_style": "chunked_limited",
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(
                    f"unsupported Nemotron encoder metadata: {key}={meta.get(key)!r}, "
                    f"expected {value!r}")
        if int(meta.get("parakeet.encoder.att_context_size_left", -1)) < 0 or \
           int(meta.get("parakeet.encoder.att_context_size_right", -1)) < 0:
            raise ValueError(
                "Nemotron Core ML export requires finite attention context metadata")
    return kind


def causal_subsampling(mel, weights):
    x = mel.unsqueeze(0).transpose(1, 2).unsqueeze(1)

    def pad(value):
        # PyTorch pads the last dimension first: frequency (2,1), then time (2,1).
        return F.pad(value, (2, 1, 2, 1))

    x = F.conv2d(pad(x), weights["encoder.subsampling.conv0.weight"],
                 bias=weights["encoder.subsampling.conv0.bias"], stride=2)
    x = F.relu(x)
    x = F.conv2d(pad(x), weights["encoder.subsampling.conv1_dw.weight"],
                 bias=weights["encoder.subsampling.conv1_dw.bias"], stride=2,
                 groups=x.size(1))
    x = F.conv2d(x, weights["encoder.subsampling.conv1_pw.weight"],
                 bias=weights["encoder.subsampling.conv1_pw.bias"])
    x = F.relu(x)
    x = F.conv2d(pad(x), weights["encoder.subsampling.conv2_dw.weight"],
                 bias=weights["encoder.subsampling.conv2_dw.bias"], stride=2,
                 groups=x.size(1))
    x = F.conv2d(x, weights["encoder.subsampling.conv2_pw.weight"],
                 bias=weights["encoder.subsampling.conv2_pw.bias"])
    x = F.relu(x)
    x = x.permute(0, 2, 1, 3).flatten(2)
    return F.linear(x, weights["encoder.subsampling.out.weight"],
                    weights["encoder.subsampling.out.bias"])


def chunked_attention_mask(length, left, right, dtype, device):
    chunk = right + 1
    query = torch.arange(length, device=device).unsqueeze(1)
    key = torch.arange(length, device=device).unsqueeze(0)
    chunk_start = (query // chunk) * chunk
    visible = (key >= chunk_start - left) & (key < chunk_start + chunk)
    zeros = torch.zeros((length, length), dtype=dtype, device=device)
    blocked = torch.full((length, length), -1.0e30, dtype=dtype, device=device)
    return torch.where(visible, zeros, blocked).unsqueeze(0).unsqueeze(0)


def rel_pos_mha_fixed(ref, x, pos_emb, weights, prefix, n_heads, att_mask=None):
    d_model = x.size(-1)
    head_dim = d_model // n_heads
    batch, length, _ = x.shape
    q = F.linear(x, weights[f"{prefix}.q.weight"], weights[f"{prefix}.q.bias"]).view(batch, length, n_heads, head_dim)
    k = F.linear(x, weights[f"{prefix}.k.weight"], weights[f"{prefix}.k.bias"]).view(batch, length, n_heads, head_dim).transpose(1, 2)
    v = F.linear(x, weights[f"{prefix}.v.weight"], weights[f"{prefix}.v.bias"]).view(batch, length, n_heads, head_dim).transpose(1, 2)
    p = F.linear(pos_emb, weights[f"{prefix}.pos.weight"]).view(1, -1, n_heads, head_dim).transpose(1, 2)
    q_u = (q + weights[f"{prefix}.pos_bias_u"]).transpose(1, 2)
    q_v = (q + weights[f"{prefix}.pos_bias_v"]).transpose(1, 2)
    matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
    matrix_bd = ref.rel_shift(torch.matmul(q_v, p.transpose(-2, -1)))[:, :, :, :length]
    scores = (matrix_ac + matrix_bd) / math.sqrt(head_dim)
    if att_mask is not None:
        scores = scores + att_mask
    attn = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(attn, v).transpose(1, 2).reshape(batch, length, d_model)
    return F.linear(ctx, weights[f"{prefix}.out.weight"], weights[f"{prefix}.out.bias"])


def conformer_conv(ref, x, weights, prefix, causal=False, layer_norm=False,
                   sequence_mask=None):
    x = x.transpose(1, 2)
    x = F.conv1d(x, weights[f"{prefix}.pw1.weight"].squeeze(-1).unsqueeze(-1),
                 weights[f"{prefix}.pw1.bias"])
    x = F.glu(x, dim=1)
    if sequence_mask is not None:
        # Keep appended capacity rows identical to the implicit zero padding at
        # the boundary of the unpadded ggml depthwise convolution.
        x = x * sequence_mask.transpose(1, 2)
    depthwise = weights[f"{prefix}.dw.weight"]
    groups = int(depthwise.shape[0])
    kernel = int(depthwise.shape[-1])
    if causal:
        x = F.pad(x, (kernel - 1, 0))
        padding = 0
    else:
        padding = (kernel - 1) // 2
    x = F.conv1d(x, depthwise, weights[f"{prefix}.dw.bias"], padding=padding, groups=groups)
    if layer_norm:
        x = ref.layer_norm(x.transpose(1, 2),
                           weights[f"{prefix}.norm.weight"],
                           weights[f"{prefix}.norm.bias"]).transpose(1, 2)
    else:
        scale = weights[f"{prefix}.bn.scale"].view(1, -1, 1)
        shift = weights[f"{prefix}.bn.shift"].view(1, -1, 1)
        x = x * scale + shift
    x = F.silu(x)
    x = F.conv1d(x, weights[f"{prefix}.pw2.weight"].squeeze(-1).unsqueeze(-1),
                 weights[f"{prefix}.pw2.bias"])
    return x.transpose(1, 2)


def conformer_block(ref, x, pos_emb, weights, index, n_heads,
                    att_mask=None, causal_conv=False, conv_layer_norm=False,
                    sequence_mask=None):
    p = f"encoder.blk.{index}"
    x = x + 0.5 * ref.conformer_ff(
        ref.layer_norm(x, weights[f"{p}.norm_ff1.weight"], weights[f"{p}.norm_ff1.bias"]),
        weights, f"{p}.ff1")
    if sequence_mask is not None:
        x = x * sequence_mask
    x = x + rel_pos_mha_fixed(ref,
        ref.layer_norm(x, weights[f"{p}.norm_attn.weight"], weights[f"{p}.norm_attn.bias"]),
        pos_emb, weights, f"{p}.attn", n_heads, att_mask)
    if sequence_mask is not None:
        x = x * sequence_mask
    x = x + conformer_conv(ref,
        ref.layer_norm(x, weights[f"{p}.norm_conv.weight"], weights[f"{p}.norm_conv.bias"]),
        weights, f"{p}.conv", causal_conv, conv_layer_norm, sequence_mask)
    if sequence_mask is not None:
        x = x * sequence_mask
    x = x + 0.5 * ref.conformer_ff(
        ref.layer_norm(x, weights[f"{p}.norm_ff2.weight"], weights[f"{p}.norm_ff2.bias"]),
        weights, f"{p}.ff2")
    x = ref.layer_norm(x, weights[f"{p}.norm_out.weight"], weights[f"{p}.norm_out.bias"])
    return x if sequence_mask is None else x * sequence_mask


def conformer_stack_forward(ref, x, weights, meta, attention_mask=None,
                            sequence_mask=None):
    d_model = meta["parakeet.encoder.d_model"]
    n_layers = meta["parakeet.encoder.n_layers"]
    n_heads = meta["parakeet.encoder.n_heads"]
    kind = model_type(meta)
    if meta.get("parakeet.encoder.xscaling", True):
        x = x * math.sqrt(d_model)
    if sequence_mask is not None:
        x = x * sequence_mask
    length = x.size(1)
    pe = ref.sinusoidal_rel_pe(
        max(length, meta.get("parakeet.encoder.pos_emb_max_len", 5000)), d_model, dtype=x.dtype)
    center = pe.size(1) // 2 + 1
    pos_emb = pe[:, center - length: center + length - 1]
    att_mask = attention_mask
    causal_conv = False
    conv_layer_norm = False
    if kind in ("eou", "nemotron"):
        att_mask = chunked_attention_mask(
            length,
            int(meta["parakeet.encoder.att_context_size_left"]),
            int(meta["parakeet.encoder.att_context_size_right"]),
            x.dtype, x.device)
        causal_conv = True
        conv_layer_norm = True
    for index in range(n_layers):
        x = conformer_block(ref, x, pos_emb, weights, index, n_heads,
                            att_mask, causal_conv, conv_layer_norm, sequence_mask)
    return x


def encoder_forward(ref, mel, weights, meta):
    if model_type(meta) in ("eou", "nemotron"):
        x = causal_subsampling(mel, weights)
    else:
        x, _ = ref.subsampling(mel, weights)
    return conformer_stack_forward(ref, x, weights, meta)


def subsampling_shape_generic(mel, weights):
    # ref.subsampling masks each conv stage to a valid length derived from
    # torch.arange(T), which bakes the traced mel length into the graph. For the
    # full-valid offline input those masks are all-ones (a no-op), so drop them
    # here to keep the stride-2 conv stack polymorphic in the time axis -- the
    # prerequisite for a dynamic-length (RangeDim) export.
    x = mel.unsqueeze(0).transpose(1, 2).unsqueeze(1)
    x = F.conv2d(x, weights["encoder.subsampling.conv0.weight"],
                 bias=weights["encoder.subsampling.conv0.bias"], stride=2, padding=1)
    x = F.relu(x)
    x = F.conv2d(x, weights["encoder.subsampling.conv1_dw.weight"],
                 bias=weights["encoder.subsampling.conv1_dw.bias"],
                 stride=2, padding=1, groups=x.size(1))
    x = F.conv2d(x, weights["encoder.subsampling.conv1_pw.weight"],
                 bias=weights["encoder.subsampling.conv1_pw.bias"], stride=1, padding=0)
    x = F.relu(x)
    x = F.conv2d(x, weights["encoder.subsampling.conv2_dw.weight"],
                 bias=weights["encoder.subsampling.conv2_dw.bias"],
                 stride=2, padding=1, groups=x.size(1))
    x = F.conv2d(x, weights["encoder.subsampling.conv2_pw.weight"],
                 bias=weights["encoder.subsampling.conv2_pw.bias"], stride=1, padding=0)
    x = F.relu(x)
    x = x.permute(0, 2, 1, 3).flatten(2)
    return F.linear(x, weights["encoder.subsampling.out.weight"],
                    weights["encoder.subsampling.out.bias"])


def encoder_frames_for_mel(ref, meta, n_mel_frames):
    # Post-subsampling encoder frame count: the same three stride-2 convs
    # run_encoder uses, mirrored here to size the fixed positional-embedding
    # buffer to the longest supported utterance.
    causal = meta.get("parakeet.encoder.causal_downsampling", False)

    def nxt(length):
        return (length // 2 + 1) if causal else ref._conv_out_len(length, 3, 2, 1)

    return nxt(nxt(nxt(n_mel_frames)))


def rel_shift_gather(matrix_bd, T):
    # Transformer-XL rel-shift + crop to T key positions as a single int32 gather:
    # out[.., i, j] = matrix_bd[.., i, (j - i) + (T - 1)]. Equivalent to the ref
    # pad/reshape rel_shift followed by [:, :, :, :T], but with no double-dynamic
    # reshape (which coremltools cannot build a shape tensor for).
    i = torch.arange(T, device=matrix_bd.device).unsqueeze(1)
    j = torch.arange(T, device=matrix_bd.device).unsqueeze(0)
    idx = (j - i + (T - 1)).to(torch.int32)
    idx = idx.unsqueeze(0).unsqueeze(0).expand(1, matrix_bd.size(1), -1, -1)
    return torch.gather(matrix_bd, 3, idx)


def rel_pos_mha_flex(x, pos_emb, W, prefix, n_heads):
    # Export-friendly relative-position attention: head splits reshape the dynamic
    # time axis via -1 (no sym_size in the shape tensor) and the rel-shift is the
    # int32 gather above, so the only dynamic-shape ops left are matmul/softmax,
    # which coremltools handles. Numerically identical to ref.rel_pos_mha.
    d_model = x.size(-1)
    head_dim = d_model // n_heads
    s_d_k = math.sqrt(head_dim)

    q = F.linear(x, W[f"{prefix}.q.weight"], W[f"{prefix}.q.bias"]).reshape(1, -1, n_heads, head_dim)
    k = F.linear(x, W[f"{prefix}.k.weight"], W[f"{prefix}.k.bias"]).reshape(1, -1, n_heads, head_dim)
    v = F.linear(x, W[f"{prefix}.v.weight"], W[f"{prefix}.v.bias"]).reshape(1, -1, n_heads, head_dim)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    p = F.linear(pos_emb, W[f"{prefix}.pos.weight"]).reshape(1, -1, n_heads, head_dim).transpose(1, 2)

    q_u = (q + W[f"{prefix}.pos_bias_u"]).transpose(1, 2)
    q_v = (q + W[f"{prefix}.pos_bias_v"]).transpose(1, 2)

    matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
    matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))
    matrix_bd = rel_shift_gather(matrix_bd, matrix_ac.size(-1))

    scores = (matrix_ac + matrix_bd) / s_d_k
    attn = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(attn, v).transpose(1, 2).reshape(1, -1, d_model)
    return F.linear(ctx, W[f"{prefix}.out.weight"], W[f"{prefix}.out.bias"])


def conformer_block_flex(ref, x, pos_emb, weights, index, n_heads):
    p = f"encoder.blk.{index}"
    x = x + 0.5 * ref.conformer_ff(
        ref.layer_norm(x, weights[f"{p}.norm_ff1.weight"], weights[f"{p}.norm_ff1.bias"]),
        weights, f"{p}.ff1")
    x = x + rel_pos_mha_flex(
        ref.layer_norm(x, weights[f"{p}.norm_attn.weight"], weights[f"{p}.norm_attn.bias"]),
        pos_emb, weights, f"{p}.attn", n_heads)
    x = x + conformer_conv(ref,
        ref.layer_norm(x, weights[f"{p}.norm_conv.weight"], weights[f"{p}.norm_conv.bias"]),
        weights, f"{p}.conv")
    x = x + 0.5 * ref.conformer_ff(
        ref.layer_norm(x, weights[f"{p}.norm_ff2.weight"], weights[f"{p}.norm_ff2.bias"]),
        weights, f"{p}.ff2")
    return ref.layer_norm(x, weights[f"{p}.norm_out.weight"], weights[f"{p}.norm_out.bias"])


def encoder_forward_flexible(ref, mel, weights, meta, pe_table):
    # Variable-length forward: masking-free subsampling + a precomputed relative
    # positional buffer gathered by the symbolic sequence length, feeding an
    # export-friendly attention (rel_pos_mha_flex) whose only dynamic-shape ops
    # are matmul/softmax.
    d_model = meta["parakeet.encoder.d_model"]
    n_layers = meta["parakeet.encoder.n_layers"]
    n_heads = meta["parakeet.encoder.n_heads"]
    x = subsampling_shape_generic(mel, weights)
    if meta.get("parakeet.encoder.xscaling", True):
        x = x * math.sqrt(d_model)
    length = x.size(1)
    center = pe_table.size(1) // 2 + 1
    # Gather the 2*length-1 relative-position rows [center-length, center+length-1)
    # with an explicit int64 index rather than a Python slice: the conv-derived
    # `length` lowers to a float var in Core ML MIL, which makes a slice `begin`
    # float (a mixed-dtype concat coremltools rejects); an int64 gather index
    # sidesteps that.
    offsets = torch.arange(2 * length - 1, device=pe_table.device)
    pos_index = (center - length + offsets).to(torch.int32)
    pos_emb = pe_table.index_select(1, pos_index)
    for index in range(n_layers):
        x = conformer_block_flex(ref, x, pos_emb, weights, index, n_heads)
    return x


class EncoderModule(torch.nn.Module):
    def __init__(self, ref, weights, meta, flexible=False, pe_table=None):
        super().__init__()
        self.ref = ref
        self.meta = meta
        self.flexible = flexible
        self._keys = list(weights.keys())
        self._buffers_by_key = {}
        for i, key in enumerate(self._keys):
            name = f"w_{i}"
            self.register_buffer(name, weights[key].contiguous().float())
            self._buffers_by_key[key] = name
        if pe_table is not None:
            self.register_buffer("pe_table", pe_table.contiguous().float())

    def forward(self, mel):
        weights = BiasTolerantWeights(
            (key, getattr(self, self._buffers_by_key[key])) for key in self._keys)
        if self.flexible:
            return encoder_forward_flexible(self.ref, mel, weights, self.meta, self.pe_table)[0]
        return encoder_forward(self.ref, mel, weights, self.meta)[0]


class MaskedEncoderModule(EncoderModule):
    """Fixed-capacity offline TDT encoder with a contiguous valid mel prefix.

    The second input stays a tensor throughout tracing: shorter utterances must
    not inherit the example length baked into the exported graph.
    """

    def forward(self, mel, mel_valid_mask):
        weights = BiasTolerantWeights(
            (key, getattr(self, self._buffers_by_key[key])) for key in self._keys)
        # Keep length arithmetic integer even under Core ML's FP16 transform:
        # FP16 cannot represent every frame count above 2048.
        valid_mel_frames = mel_valid_mask.to(torch.int32).sum(dim=0)
        x, valid_frames = self.ref.subsampling(mel, weights, valid_mel_frames)
        valid = (torch.arange(x.size(1), device=x.device) < valid_frames).to(x.dtype)
        sequence_mask = valid.reshape(1, -1, 1)
        # Finite in FP16; a fully masked padded query must not create NaNs.
        attention_mask = (1.0 - valid).reshape(1, 1, 1, -1) * -1.0e4
        return conformer_stack_forward(
            self.ref, x, weights, self.meta, attention_mask, sequence_mask)[0]


def prepare_bypass_encoder_inputs(pre_encode, valid_mask):
    encoder_input = pre_encode.transpose(0, 1).unsqueeze(0)
    sequence_mask = valid_mask.reshape(1, -1, 1)
    attention_mask = (1.0 - valid_mask).reshape(1, 1, 1, -1) * -1.0e4
    return encoder_input, sequence_mask, attention_mask


class BypassEncoderModule(torch.nn.Module):
    """FastConformer blocks with subsampling/pre-encode intentionally omitted."""

    def __init__(self, ref, weights, meta):
        super().__init__()
        self.ref = ref
        self.meta = meta
        self._keys = list(weights.keys())
        self._buffers_by_key = {}
        for i, key in enumerate(self._keys):
            name = f"w_{i}"
            self.register_buffer(name, weights[key].contiguous().float())
            self._buffers_by_key[key] = name

    def forward(self, pre_encode, valid_mask):
        weights = BiasTolerantWeights(
            (key, getattr(self, self._buffers_by_key[key])) for key in self._keys)
        x, sequence_mask, attention_mask = prepare_bypass_encoder_inputs(
            pre_encode, valid_mask)
        return conformer_stack_forward(
            self.ref, x, weights, self.meta, attention_mask=attention_mask,
            sequence_mask=sequence_mask)[0]


def compile_mlmodelc(mlpackage_path, compile_dir):
    subprocess.run(
        ["xcrun", "coremlc", "compile", str(mlpackage_path), str(compile_dir)],
        check=True)
    compiled = Path(compile_dir) / (Path(mlpackage_path).stem + ".mlmodelc")
    if not compiled.is_dir():
        raise RuntimeError(f"coremlc did not produce expected bundle: {compiled}")
    return compiled


def report_compute_placement(mlmodelc_path):
    # Report Core ML op placement (Neural Engine / GPU / CPU) so a CPU-bound export
    # -- which runs far slower than the ggml-Metal encoder -- is caught here instead
    # of silently shipping a slow sidecar. Requires macOS 14.4+ / coremltools>=8.
    try:
        import coremltools as ct
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(str(mlmodelc_path), compute_units=ct.ComputeUnit.ALL)
    except Exception as exc:
        print(f"[export] compute-plan check skipped ({type(exc).__name__}: {exc})")
        return

    from collections import Counter

    def device_name(device):
        if device is None:
            return "None"
        return type(device).__name__.replace("ML", "").replace("ComputeDevice", "")

    counts = Counter()

    def walk(block):
        for op in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            counts[device_name(usage.preferred_compute_device) if usage is not None else "None"] += 1
            nested = getattr(op, "blocks", None) or getattr(op, "block", None)
            if nested:
                for inner in (nested if isinstance(nested, (list, tuple)) else [nested]):
                    walk(inner)

    walk(plan.model_structure.program.functions["main"].block)
    compute_ops = sum(v for k, v in counts.items() if k != "None")
    summary = ", ".join(f"{k}={v}" for k, v in counts.most_common())
    print(f"[export] compute-plan op placement: {summary}")
    if compute_ops > 0 and counts.get("NeuralEngine", 0) == 0:
        print("[export] WARNING: 0 ops on the Apple Neural Engine -- this model will run on "
              "CPU and be much slower than the ggml-Metal encoder. Flexible/dynamic shapes "
              "force Core ML onto CPU; export a fixed single length for ANE acceleration.")


def palettize_encoder(mlmodel, bits, group_size):
    """Apply the grouped-channel LUT layout used by ANE-focused speech models.

    Grouped-channel palettization requires the macOS 15 / iOS 18 model format;
    the caller selects that deployment target before conversion.
    """
    import coremltools.optimize as cto

    op_config = cto.coreml.OpPalettizerConfig(
        mode="kmeans",
        nbits=bits,
        granularity="per_grouped_channel",
        group_size=group_size,
    )
    config = cto.coreml.OptimizationConfig(global_config=op_config)
    print(f"[export] palettizing weights: {bits}-bit grouped-channel LUTs, "
          f"group_size={group_size}")
    return cto.coreml.palettize_weights(mlmodel, config)


def convert_fixed(ref, weights, meta, example, n_mels, n_mel_frames, d_model,
                  precision, io_dtype, deployment_target):
    masked = model_type(meta) == "tdt"
    model = (MaskedEncoderModule if masked else EncoderModule)(ref, weights, meta).eval()
    inputs = [ct.TensorType(name="mel", shape=(n_mels, n_mel_frames), dtype=io_dtype)]
    trace_inputs = (example,)
    if masked:
        trace_inputs += (torch.ones(n_mel_frames, dtype=torch.float32),)
        inputs.append(ct.TensorType(name="mel_valid_mask", shape=(n_mel_frames,),
                                    dtype=io_dtype))
    with torch.inference_mode():
        reference_out = model(*trace_inputs)
    print(f"[export] fixed n_mels={n_mels} d_model={d_model} n_mel_frames={n_mel_frames} "
          f"encoder_frames={reference_out.shape[0]}")
    traced = torch.jit.trace(model, trace_inputs, check_trace=False)
    return ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name="encoder_out", dtype=io_dtype)],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=precision,
        minimum_deployment_target=deployment_target,
    )


def convert_bypass_fixed(ref, weights, meta, example, d_model, n_encoder_frames,
                         precision, io_dtype, deployment_target):
    model = BypassEncoderModule(ref, weights, meta).eval()
    valid_mask = torch.ones(n_encoder_frames, dtype=torch.float32)
    with torch.inference_mode():
        reference_out = model(example, valid_mask)
    print(f"[export] fixed bypass-pre-encode d_model={d_model} "
          f"encoder_frames={n_encoder_frames} output_frames={reference_out.shape[0]}")
    traced = torch.jit.trace(model, (example, valid_mask), check_trace=False)
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="pre_encode", shape=(d_model, n_encoder_frames),
                              dtype=io_dtype),
                ct.TensorType(name="valid_mask", shape=(n_encoder_frames,),
                              dtype=io_dtype)],
        outputs=[ct.TensorType(name="encoder_out", dtype=io_dtype)],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=precision,
        minimum_deployment_target=deployment_target,
    )


def convert_flexible(ref, weights, meta, example, n_mels, n_mel_frames, d_model,
                     min_frames, max_frames, precision, io_dtype, deployment_target):
    min_frames = min_frames if min_frames is not None else 1
    max_frames = max(max_frames if max_frames is not None else n_mel_frames, n_mel_frames)
    pos_emb_max_len = int(meta.get("parakeet.encoder.pos_emb_max_len", 5000))
    pe_len = max(pos_emb_max_len, encoder_frames_for_mel(ref, meta, max_frames))
    pe_table = ref.sinusoidal_rel_pe(pe_len, d_model)

    model = EncoderModule(ref, weights, meta, flexible=True, pe_table=pe_table).eval()
    with torch.inference_mode():
        reference_out = model(example)
    print(f"[export] flexible n_mels={n_mels} d_model={d_model} "
          f"mel_frames=[{min_frames},{max_frames}] traced={n_mel_frames} "
          f"encoder_frames={reference_out.shape[0]} pe_len={pe_len}")

    # torch.export keeps the time axis symbolic (unlike jit.trace, which bakes it),
    # so the dynamic pos-emb slice survives conversion. strict=False uses the
    # non-strict tracer, which tolerates the Python-level weights dict.
    #
    # The three stride-2 subsampling convs make torch.export emit conservative
    # guards on the mel length (max(1, .), divisibility, reshape-product equality)
    # that no single range satisfies. backed_size_oblivious treats symbolic sizes
    # as >= 2, which discharges those guards without changing numerics; draft_export
    # (guards -> runtime asserts) is the fallback if a build still trips one. The
    # import is local so a private-config rename can only break --flexible, never
    # the default fixed export.
    import torch.fx.experimental._config as fx_config
    fx_config.backed_size_oblivious = True
    time_dim = torch.export.Dim("mel_t", min=min_frames, max=max_frames)
    dynamic_shapes = {"mel": {1: time_dim}}
    try:
        exported = torch.export.export(model, (example,), dynamic_shapes=dynamic_shapes,
                                       strict=False)
    except Exception as exc:
        print(f"[export] export guard violation ({type(exc).__name__}); "
              f"retrying via draft_export")
        exported = torch.export.draft_export(model, (example,), dynamic_shapes=dynamic_shapes)
        if isinstance(exported, tuple):
            exported = exported[0]
    # torch>=2.5 export produces a TRAINING-dialect program; coremltools converts
    # only core ATen / EDGE, so lower it first.
    exported = exported.run_decompositions({})
    return ct.convert(
        exported,
        inputs=[ct.TensorType(
            name="mel",
            shape=ct.Shape(shape=(n_mels, ct.RangeDim(
                lower_bound=min_frames, upper_bound=max_frames, default=n_mel_frames))),
            dtype=io_dtype)],
        outputs=[ct.TensorType(name="encoder_out", dtype=io_dtype)],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=precision,
        minimum_deployment_target=deployment_target,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="output .mlpackage path")
    ap.add_argument("--scripts", type=Path, default=Path(__file__).resolve().parent)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", type=Path, help="size the encoder to this wav's mel length")
    group.add_argument("--n-mel-frames", type=int, help="fixed mel length to trace")
    group.add_argument("--n-encoder-frames", type=int,
                       help="fixed capacity for --bypass-pre-encode (410 covers "
                            "the default v2.1 AOSC geometry)")
    ap.add_argument("--bypass-pre-encode", action="store_true",
                    help="export the Sortformer v2.1 conformer stack without subsampling; "
                         "input is a fixed (d_model, n_encoder_frames) embedding slab")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--palettize-bits", type=int, choices=[4, 6, 8], default=None,
                    help="optionally compress fixed-export weights into grouped-channel LUTs; "
                         "6 matches the ANE-oriented Voz layout and requires macOS 15 / iOS 18")
    ap.add_argument("--palettize-group-size", type=int, default=16,
                    help="channels sharing each LUT with --palettize-bits (default: 16)")
    ap.add_argument("--compile-dir", type=Path, default=None,
                    help="if set, compile the .mlpackage to a .mlmodelc here via coremlc")
    ap.add_argument("--flexible", action="store_true",
                    help="export a variable-length (Core ML RangeDim) encoder via "
                         "torch.export. CORRECTNESS-ONLY: runs on CPU (0 ANE ops, ~8x slower "
                         "than ggml-Metal); export a fixed single length for ANE acceleration. "
                         "Needs coremltools>=8 / torch>=2.3")
    ap.add_argument("--min-frames", type=int, default=None,
                    help="[--flexible] minimum mel length the RangeDim accepts (default 1)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="[--flexible] maximum mel length the RangeDim accepts; set to your "
                         "longest expected utterance (default: the traced length)")
    args = ap.parse_args()
    if args.bypass_pre_encode != (args.n_encoder_frames is not None):
        ap.error("--bypass-pre-encode and --n-encoder-frames must be used together")
    if args.bypass_pre_encode and args.flexible:
        ap.error("--bypass-pre-encode does not support --flexible")
    if args.n_encoder_frames is not None and args.n_encoder_frames <= 0:
        ap.error("--n-encoder-frames must be greater than zero")
    if args.flexible and args.palettize_bits is not None:
        ap.error("--palettize-bits requires a fixed-shape export (omit --flexible)")
    if args.palettize_group_size <= 0:
        ap.error("--palettize-group-size must be greater than zero")

    ref = load_reference_encoder(args.scripts)
    weights, meta = ref.load_gguf(args.gguf)
    try:
        kind = validate_export_contract(meta, flexible=args.flexible,
                                        bypass_pre_encode=args.bypass_pre_encode)
    except ValueError as exc:
        ap.error(str(exc))

    d_model = int(meta["parakeet.encoder.d_model"])
    n_mels = int(weights["preproc.mel_filterbank"].shape[0])
    if args.bypass_pre_encode:
        n_mel_frames = None
    elif args.n_mel_frames is not None:
        n_mel_frames = args.n_mel_frames
    else:
        n_mel_frames = mel_frames_for_wav(args.wav, resolve_hop_length(meta))

    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    io_dtype = np.float16 if args.precision == "fp16" else np.float32
    deployment_target = ct.target.macOS15 if args.palettize_bits is not None else ct.target.macOS13
    print(f"[export] model_type={kind} bypass_pre_encode={args.bypass_pre_encode}")
    if args.bypass_pre_encode:
        example = torch.zeros(d_model, args.n_encoder_frames, dtype=torch.float32)
        mlmodel = convert_bypass_fixed(ref, weights, meta, example, d_model,
                                       args.n_encoder_frames, precision, io_dtype,
                                       deployment_target)
    elif args.flexible:
        example = torch.zeros(n_mels, n_mel_frames, dtype=torch.float32)
        mlmodel = convert_flexible(ref, weights, meta, example, n_mels, n_mel_frames,
                                   d_model, args.min_frames, args.max_frames,
                                   precision, io_dtype, deployment_target)
    else:
        example = torch.zeros(n_mels, n_mel_frames, dtype=torch.float32)
        mlmodel = convert_fixed(ref, weights, meta, example, n_mels, n_mel_frames,
                                d_model, precision, io_dtype, deployment_target)
    if args.palettize_bits is not None:
        mlmodel = palettize_encoder(mlmodel, args.palettize_bits,
                                    args.palettize_group_size)
    mlmodel.save(str(args.out))
    print(f"[export] saved {args.out}")

    if args.compile_dir is not None:
        compiled = compile_mlmodelc(args.out, args.compile_dir)
        print(f"[export] compiled .mlmodelc into {args.compile_dir}")
        report_compute_placement(compiled)


if __name__ == "__main__":
    main()
