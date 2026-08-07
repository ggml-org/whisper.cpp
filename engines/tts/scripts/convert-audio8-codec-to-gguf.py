#!/usr/bin/env python3
"""Convert the Audio8 TTS neural codec (codec.pth) to GGUF.

The codec is a DAC-style residual-vector-quantised autoencoder running at
44.1 kHz with 2048 samples per frame. Its two halves are independent at
inference time, so they are emitted separately:

  encoder  -- convolutional analysis stack, a windowed transformer, the
              downsampling pre-module and the quantiser input projections.
              Only needed to enrol a reference voice from a WAV.
  decoder  -- the codebooks, quantiser output projections, the windowed
              post-module, the upsampling stack and the synthesis convolutions.
              Needed for every synthesis.

Both halves carry the codebooks, so each file stands alone. Weight
normalisation is folded into a single `weight` tensor, and every fused `wqkv`
is split into q/k/v with the q and k rows reordered into the split-half RoPE
layout the engine rotates in (see convert-audio8-lm-to-gguf.py for why that
leaves attention unchanged). RoPE tables are baked as `_cos` / `_sin` planes at
the reference's own bfloat16 precision, matching how the checkpoint builds them
under float32 inference.

    python3 convert-audio8-codec-to-gguf.py \\
        --model-dir models/Audio8-TTS-Preview-0.6b \\
        --part decoder --outfile audio8-codec-decoder-f32.gguf --dtype f32
"""
import argparse
import os
import re
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio8_reference import (  # noqa: E402
    BLOCK,
    CODEC_ARCH as ARCH,
    CODEC_KV as KV,
    CODEC_PARTS,
    DENSE,
    DEPTHWISE_MARKER,
    EXACT,
    FUSED_ATTENTION_SUFFIX,
    HEAD_DIM,
    QUANT_BLOCK,
    ROPE_PRECISIONS,
    STORAGE_TYPES,
    codec_head_counts,
    expand_fused,
    flush,
    format_tally,
    is_shared,
    load_config,
    raw_codec_state,
    rename_codec_key,
    rope_planes,
    select_part,
    to_engine_layout,
    write_tensors,
)

LATENT_DIM = 1024
CODEBOOK_DIM = 8
CODEC_NORM_EPS = 1e-5
CONVNEXT_NORM_EPS = 1e-6
CODEC_ROPE_THETA = 10000.0
ENCODER_STRIDES = (2, 4, 8, 8)
DECODER_STRIDES = (8, 8, 4, 2)
RESIDUAL_DILATIONS = (1, 3, 9)
ENCODER_WINDOW = 512
QUANTIZER_WINDOW = 128
SNAKE_EPSILON = 1e-9

LEGACY_WEIGHT_NORM = ("weight_g", "weight_v")
PARAMETRIZED_WEIGHT_NORM = (
    "parametrizations.weight.original0",
    "parametrizations.weight.original1",
)


@dataclass(frozen=True)
class TransformerSpec:
    n_layer: int
    n_head: int
    n_kv: int
    dim: int
    inter: int
    window: int
    head_dim: int = HEAD_DIM
    rope_theta: float = CODEC_ROPE_THETA
    norm_eps: float = CODEC_NORM_EPS


def encoder_spec():
    return TransformerSpec(
        n_layer=4, n_head=LATENT_DIM // HEAD_DIM, n_kv=LATENT_DIM // HEAD_DIM,
        dim=LATENT_DIM, inter=LATENT_DIM * 3, window=ENCODER_WINDOW,
    )


def pre_spec():
    return TransformerSpec(
        n_layer=8, n_head=16, n_kv=16, dim=LATENT_DIM, inter=3072,
        window=QUANTIZER_WINDOW,
    )


def post_spec(config):
    return TransformerSpec(
        n_layer=int(config["codec_post_n_layer"]),
        n_head=int(config["codec_post_n_head"]),
        n_kv=int(config["codec_post_n_local_heads"]),
        dim=LATENT_DIM,
        inter=int(config["codec_post_intermediate_size"]),
        window=QUANTIZER_WINDOW,
    )


def load_codec_arrays(model_dir, filename):
    import torch

    state = raw_codec_state(model_dir, filename)
    return {key: value.to(torch.float32).numpy() for key, value in state.items()}


def norm_except_first_dim(v):
    axes = tuple(range(1, v.ndim))
    return np.sqrt(np.square(v).sum(axis=axes, keepdims=True))


def fold_pair(g, v):
    return (g * v / norm_except_first_dim(v)).astype(np.float32)


def weight_norm_bases(state):
    bases = {}
    for key in state:
        for gain, direction in (LEGACY_WEIGHT_NORM, PARAMETRIZED_WEIGHT_NORM):
            if key.endswith("." + gain):
                bases[key[: -len(gain) - 1]] = (gain, direction)
    return bases


def fold_weight_norm(state):
    bases = weight_norm_bases(state)
    folded = {}
    consumed = set()
    for base, (gain, direction) in bases.items():
        gain_key = f"{base}.{gain}"
        direction_key = f"{base}.{direction}"
        folded[f"{base}.weight"] = fold_pair(state[gain_key], state[direction_key])
        consumed.update((gain_key, direction_key))
    for key, value in state.items():
        if key not in consumed:
            folded[key] = value
    return folded


def expand_attention(name, array):
    if not name.endswith(FUSED_ATTENTION_SUFFIX):
        return {name: array}
    base = name[: -len(FUSED_ATTENTION_SUFFIX)].rstrip("/")
    n_head, n_kv = codec_head_counts(array.shape[0], array.shape[1])
    return expand_fused(base, array, n_head, n_kv)


def renamed_tensors(state):
    named = {}
    for key, array in state.items():
        named.update(expand_attention(rename_codec_key(key), array))
    return {name: to_engine_layout(name, array) for name, array in named.items()}


def check_no_projection_weights(named):
    unexpected = [n for n in named if "input_proj" in n or "output_proj" in n]
    if unexpected:
        raise ValueError(
            f"window transformers are expected to be projection-free: {unexpected[:3]}"
        )


def check_partition(named, encoder_part, decoder_part):
    missing = set(named) - set(encoder_part) - set(decoder_part)
    if missing:
        raise ValueError(f"tensors assigned to neither half: {sorted(missing)[:5]}")


def rope_tables(specs, max_frames, precision):
    tables = {}
    for name, spec in specs.items():
        tables.update(rope_planes(
            f"rope/{name}", max_frames, spec.head_dim, spec.rope_theta, precision
        ))
    return tables


def write_transformer_metadata(writer, name, spec):
    for key, value in dict(
        n_layer=spec.n_layer, n_head=spec.n_head, n_kv=spec.n_kv, dim=spec.dim,
        inter=spec.inter, window=spec.window, head_dim=spec.head_dim,
    ).items():
        writer.add_uint32(f"{KV}.{name}.{key}", int(value))
    writer.add_float32(f"{KV}.{name}.rope_theta", float(spec.rope_theta))
    writer.add_float32(f"{KV}.{name}.norm_eps", float(spec.norm_eps))


def write_metadata(writer, config, specs, part, max_frames):
    for key, value in dict(
        sample_rate=config["codec_sample_rate"],
        frame_size=config["codec_frame_size"],
        num_codebooks=config["num_codebooks"],
        latent_dim=LATENT_DIM,
        codebook_dim=CODEBOOK_DIM,
        max_frames=max_frames,
    ).items():
        writer.add_uint32(f"{KV}.{key}", int(value))
    writer.add_float32(f"{KV}.convnext_norm_eps", CONVNEXT_NORM_EPS)
    writer.add_float32(f"{KV}.snake_epsilon", SNAKE_EPSILON)
    writer.add_array(f"{KV}.encoder_strides", list(ENCODER_STRIDES))
    writer.add_array(f"{KV}.decoder_strides", list(DECODER_STRIDES))
    writer.add_array(f"{KV}.residual_dilations", list(RESIDUAL_DILATIONS))
    writer.add_string(f"{KV}.part", part)
    for name, spec in specs.items():
        write_transformer_metadata(writer, name, spec)


def write_codebook_metadata(writer, named):
    semantic = named["q/sem/0/codebook/weight"]
    residual = named["q/res/0/codebook/weight"]
    writer.add_uint32(f"{KV}.semantic_codebook_size", int(semantic.shape[0]))
    writer.add_uint32(f"{KV}.residual_codebook_size", int(residual.shape[0]))
    writer.add_uint32(
        f"{KV}.residual_codebooks",
        sum(1 for n in named if re.fullmatch(r"q/res/\d+/codebook/weight", n)),
    )


def is_body_matmul(name, array):
    return (
        array.ndim == 2
        and "/blk/" in name
        and not name.endswith(("_norm", "_scale"))
        and array.shape[1] % QUANT_BLOCK == 0
    )


def is_matmul_operand(name, array):
    """A depthwise kernel is one scalar per channel per tap, so the engine
    applies it element-wise rather than contracting with it, and f16 would have
    to be widened at every use."""
    return array.ndim >= 2 and DEPTHWISE_MARKER not in name


def role_of(name, array):
    """The codebooks stay exact even though they are read like a table: the
    encoder picks codes by nearest neighbour, so rounding them moves the
    argmin. The same goes for the RoPE tables and every parameter the engine
    reads element-wise, the snake alphas among them, which sit inside a sine."""
    if is_body_matmul(name, array):
        return BLOCK
    if is_shared(name) or name.startswith("rope/"):
        return EXACT
    return DENSE if is_matmul_operand(name, array) else EXACT


def part_specs(config, part):
    if part == "encoder":
        return {"enc_tf": encoder_spec(), "pre_tf": pre_spec()}
    return {"post_tf": post_spec(config)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--codec-file", default="codec.pth")
    parser.add_argument("--part", choices=CODEC_PARTS, required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--dtype", choices=STORAGE_TYPES, default="f32")
    parser.add_argument("--max-frames", type=int, default=4096)
    parser.add_argument("--rope-precision", choices=ROPE_PRECISIONS, default="bf16",
                        help="bf16 reproduces the reference's own table rounding")
    return parser.parse_args()


def main():
    import gguf

    args = parse_args()
    config = load_config(args.model_dir)
    state = fold_weight_norm(load_codec_arrays(args.model_dir, args.codec_file))
    named = renamed_tensors(state)
    check_no_projection_weights(named)

    encoder_part = select_part(named, "encoder")
    decoder_part = select_part(named, "decoder")
    check_partition(named, encoder_part, decoder_part)
    selected = encoder_part if args.part == "encoder" else decoder_part

    specs = part_specs(config, args.part)
    selected.update(rope_tables(specs, args.max_frames, args.rope_precision))

    writer = gguf.GGUFWriter(args.outfile, ARCH)
    write_metadata(writer, config, specs, args.part, args.max_frames)
    writer.add_string(f"{KV}.rope_precision", args.rope_precision)
    write_codebook_metadata(writer, selected)
    tally = write_tensors(writer, selected, args.dtype, role_of)
    flush(writer)
    print(f"wrote {len(selected)} {args.part} tensors ({format_tally(tally)}) "
          f"-> {args.outfile}")


if __name__ == "__main__":
    main()
