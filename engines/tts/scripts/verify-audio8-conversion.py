#!/usr/bin/env python3
"""Check converted Audio8 GGUFs tensor-by-tensor against the PyTorch checkpoint.

Every tensor the converters emit is either a straight copy, a slice of a fused
`wqkv`, a gather of embedding rows, or a folded weight-norm pair. This script
rebuilds each expectation from the reference implementation -- the codec is
instantiated through the checkpoint's own remote code so the folding is
compared against what PyTorch itself materialises -- and reports the worst
absolute deviation per tensor group.

What that does and does not establish is worth being precise about. Values,
shapes, storage types and the weight-norm folding are compared against the
checkpoint. The codec's *layout* is not: `codec_expectations` reaches the
engine's axis order through the same `to_engine_layout` and `expand_fused` the
converter uses, so a wrong entry in TRANSPOSED_CONV, a wrong axis in the
tap-major kernel, or a wrong offset in the qkv split cancels on both sides and
passes. The language model's q/k permutation is the one layout that is checked
independently, by `verify_rope_layout`, which re-derives attention scores from
the reference rotation; the codec still wants the same treatment.

An f32 conversion is expected to be exact; f16 and quantised builds are checked
against the tolerance implied by their storage format. RoPE tables are only
compared when the GGUF says it baked them at the reference's own bfloat16
rounding, since an f32 build is deliberately a different table.

    python3 verify-audio8-conversion.py \\
        --model-dir models/Audio8-TTS-Preview-0.6b \\
        --lm-gguf audio8-lm-f32.gguf \\
        --codec-encoder-gguf audio8-codec-encoder-f32.gguf \\
        --codec-decoder-gguf audio8-codec-decoder-f32.gguf
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio8_reference import (  # noqa: E402
    CODEC_KV,
    FUSED_ATTENTION_SUFFIX,
    HEAD_DIM,
    LM_KV,
    codec_head_counts,
    expand_fused,
    load_codec_module,
    load_config,
    rename_codec_key,
    select_part,
    split_qkv,
    to_engine_layout,
)

LM_ROPE_TENSORS = ("lm/rope_cos", "lm/rope_sin", "fast/rope_cos", "fast/rope_sin")
ROTATION_POSITIONS = 64
ABSOLUTE_FLOOR = 1e-8
# A block-quantised value can miss its original by up to one reconstruction
# step: q8_0 spreads 255 levels across a block, q4_0 only 16, and q4_0's scale
# is asymmetric, so its worst case is a whole step rather than half of one.
RELATIVE_TOLERANCE = {"F32": 1e-6, "F16": 1e-3, "Q8_0": 1.0 / 127.0, "Q4_0": 1.0 / 8.0}
DEFAULT_RELATIVE_TOLERANCE = 1.0 / 8.0


def tolerance_for(tensor_type_name, reference):
    relative = RELATIVE_TOLERANCE.get(tensor_type_name, DEFAULT_RELATIVE_TOLERANCE)
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    return ABSOLUTE_FLOOR + relative * scale


def read_gguf(path):
    import gguf

    reader = gguf.GGUFReader(path)
    fields = {name: field.contents() for name, field in reader.fields.items()}
    tensors = {
        tensor.name: (dequantize(tensor), tensor.tensor_type.name)
        for tensor in reader.tensors
    }
    return fields, tensors


def dequantize(tensor):
    import gguf

    if tensor.tensor_type in (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
    ):
        return np.array(tensor.data, dtype=np.float32).reshape(
            tuple(int(d) for d in reversed(tensor.shape))
        )
    return gguf.quants.dequantize(tensor.data, tensor.tensor_type).astype(np.float32)


def load_lm_state(model_dir):
    import torch
    from safetensors.torch import load_file

    tensors = load_file(f"{model_dir}/model.safetensors")
    return {k: v.to(torch.float32).numpy() for k, v in tensors.items()}


def lm_branch_expectations(state, source, target, depth, n_head, n_kv, head_dim, bias):
    expected = {}
    for index in range(depth):
        source_prefix = f"{source}.{index}"
        target_prefix = f"{target}/blk/{index}"
        expected.update(expand_fused(
            target_prefix, state[f"{source_prefix}.attention.wqkv.weight"],
            n_head, n_kv, head_dim,
        ))
        if bias:
            expected.update(expand_fused(
                target_prefix, state[f"{source_prefix}.attention.wqkv.bias"],
                n_head, n_kv, head_dim, suffix="_b",
            ))
        expected[f"{target_prefix}/wo"] = state[f"{source_prefix}.attention.wo.weight"]
        expected[f"{target_prefix}/attn_norm"] = state[f"{source_prefix}.attention_norm.weight"]
        expected[f"{target_prefix}/ffn_norm"] = state[f"{source_prefix}.ffn_norm.weight"]
        for gate in ("w1", "w2", "w3"):
            expected[f"{target_prefix}/{gate}"] = state[
                f"{source_prefix}.feed_forward.{gate}.weight"
            ]
    return expected


def semantic_head(embeddings, config):
    begin = int(config["semantic_begin_id"])
    end = int(config["semantic_end_id"])
    rows = list(range(begin, end + 1)) + [int(config["eos_token_id"])]
    return embeddings[rows]


def reference_rope(model_dir, length, head_dim, theta):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    precompute = get_class_from_dynamic_module(
        "modeling_arktts._precompute_rope", model_dir
    )
    return precompute(length, head_dim, theta).float().numpy()


def rope_plane_expectations(prefix, table):
    return {
        f"{prefix}_cos": np.ascontiguousarray(table[..., 0]),
        f"{prefix}_sin": np.ascontiguousarray(table[..., 1]),
    }


def lm_rope_expectations(model_dir, config):
    theta = float(config["rope_base"])
    expectations = rope_plane_expectations("lm/rope", reference_rope(
        model_dir, int(config["max_seq_len"]), int(config["head_dim"]), theta
    ))
    expectations.update(rope_plane_expectations("fast/rope", reference_rope(
        model_dir, int(config["num_codebooks"]), int(config["fast_head_dim"]), theta
    )))
    return expectations


def lm_expectations(config, state):
    expected = {
        "lm/tok_emb": state["embeddings.weight"],
        "lm/codebook_emb": state["codebook_embeddings.weight"],
        "lm/norm": state["norm.weight"],
        "lm/sem_head": semantic_head(state["embeddings.weight"], config),
        "fast/emb": state["fast_embeddings.weight"],
        "fast/norm": state["fast_norm.weight"],
        "fast/out": state["fast_output.weight"],
    }
    expected.update(lm_branch_expectations(
        state, "layers", "lm", int(config["n_layer"]), int(config["n_head"]),
        int(config["n_local_heads"]), int(config["head_dim"]),
        bool(config["attention_qkv_bias"]),
    ))
    expected.update(lm_branch_expectations(
        state, "fast_layers", "fast", int(config["n_fast_layer"]),
        int(config["fast_n_head"]), int(config["fast_n_local_heads"]),
        int(config["fast_head_dim"]), bool(config["fast_attention_qkv_bias"]),
    ))
    return expected


def module_weight(module):
    import torch

    weight = getattr(module, "weight", None)
    return weight if isinstance(weight, torch.Tensor) else None


def materialised_parameters(codec):
    named = {}
    for name, module in codec.named_modules():
        if "parametrizations" in name:
            continue
        weight = module_weight(module)
        if weight is not None:
            named[f"{name}.weight"] = weight.detach().numpy()
    for name, parameter in codec.named_parameters():
        named.setdefault(name, parameter.detach().numpy())
    return named


def is_parametrization_artefact(key):
    return "parametrizations" in key or key.endswith(("weight_g", "weight_v"))


def codec_expectations(model_dir):
    codec = load_codec_module(model_dir)
    expected = {}
    for key, array in materialised_parameters(codec).items():
        if is_parametrization_artefact(key):
            continue
        name = rename_codec_key(key)
        if name.endswith(FUSED_ATTENTION_SUFFIX):
            base = name[: -len(FUSED_ATTENTION_SUFFIX)].rstrip("/")
            n_head, n_kv = codec_head_counts(array.shape[0], array.shape[1])
            expected.update(expand_fused(base, array, n_head, n_kv, HEAD_DIM))
        else:
            expected[name] = array
    return {name: to_engine_layout(name, array) for name, array in expected.items()}


def group_of(name):
    return re.sub(r"/\d+", "/N", name)


def compare(actual, expected, label):
    worst = defaultdict(float)
    counts = defaultdict(int)
    failures = []
    for name, (array, type_name) in sorted(actual.items()):
        if name not in expected:
            failures.append(f"{label}: {name} has no reference counterpart")
            continue
        reference = np.asarray(expected[name], dtype=np.float32)
        if array.shape != reference.shape:
            failures.append(
                f"{label}: {name} shape {array.shape} != reference {reference.shape}"
            )
            continue
        deviation = float(np.max(np.abs(array - reference))) if array.size else 0.0
        group = group_of(name)
        worst[group] = max(worst[group], deviation)
        counts[group] += 1
        if deviation > tolerance_for(type_name, reference):
            failures.append(
                f"{label}: {name} deviates by {deviation:.3e} (type {type_name})"
            )
    return worst, counts, failures


def report(label, worst, counts):
    print(f"\n{label}: {sum(counts.values())} tensors in {len(counts)} groups")
    for group in sorted(worst):
        print(f"  {group:52s} x{counts[group]:<4d} max|delta| {worst[group]:.3e}")


def missing_from_gguf(emitted, expected, label):
    absent = sorted(set(expected) - set(emitted))
    return [f"{label}: {name} was not emitted" for name in absent]


def reference_rotate(heads, table):
    """The checkpoint's own rotation: adjacent pairs against a complex table."""
    pairs = heads.reshape(*heads.shape[:-1], -1, 2)
    return np.stack(
        (
            pairs[..., 0] * table[..., 0] - pairs[..., 1] * table[..., 1],
            pairs[..., 1] * table[..., 0] + pairs[..., 0] * table[..., 1],
        ),
        axis=-1,
    ).reshape(heads.shape)


def split_half_rotate(heads, cosine, sine):
    """The engine's rotation: halves against separate cosine and sine planes."""
    half = heads.shape[-1] // 2
    lower, upper = heads[..., :half], heads[..., half:]
    return np.concatenate(
        (lower * cosine - upper * sine, upper * cosine + lower * sine), axis=-1
    )


def first_head(rows, activations, head_dim):
    return (activations @ rows[:head_dim].T)


def attention_scores(query, key):
    return query @ key.T


def rotation_deviation(reference, emitted, table, head_dim):
    """Attention scores are the only thing the permutation has to preserve, so
    they are what gets compared: the reference's rotation of the original rows
    against the engine's rotation of the reordered ones. Both sides rotate with
    the reference table, so the permutation is checked whatever precision the
    GGUF baked its own planes at."""
    rng = np.random.default_rng(0)
    cosine, sine = table[..., 0], table[..., 1]
    length = min(int(table.shape[0]), ROTATION_POSITIONS)
    activations = rng.standard_normal(
        (length, reference["wq"].shape[1]), dtype=np.float32
    )
    expected = attention_scores(*(
        reference_rotate(first_head(reference[part], activations, head_dim), table[:length])
        for part in ("wq", "wk")
    ))
    got = attention_scores(*(
        split_half_rotate(
            first_head(emitted[part], activations, head_dim), cosine[:length], sine[:length]
        )
        for part in ("wq", "wk")
    ))
    return float(np.max(np.abs(expected - got))) / float(np.max(np.abs(expected)))


def reference_qk(state, config, head_dim):
    query, key, _ = split_qkv(
        state["layers.0.attention.wqkv.weight"], int(config["n_head"]),
        int(config["n_local_heads"]), head_dim,
    )
    return {"wq": query, "wk": key}


def verify_rope_layout(model_dir, state, tensors, config):
    """The reordered q/k rows and the split planes are only correct together,
    so they are checked together rather than value by value. The bar is what
    the q/k storage can represent: the scores are computed from the quantised
    rows but compared against the full-precision reference."""
    head_dim = int(config["head_dim"])
    table = reference_rope(
        model_dir, int(config["max_seq_len"]), head_dim, float(config["rope_base"])
    )
    emitted = {part: tensors[f"lm/blk/0/{part}"] for part in ("wq", "wk")}
    type_name = emitted["wq"][1]
    deviation = rotation_deviation(
        reference_qk(state, config, head_dim),
        {part: value for part, (value, _) in emitted.items()},
        table, head_dim,
    )
    tolerance = RELATIVE_TOLERANCE.get(type_name, DEFAULT_RELATIVE_TOLERANCE)
    print(f"\nrope layout: attention scores match the reference to {deviation:.3e} "
          f"(q/k stored as {type_name}, tolerance {tolerance:.3e})")
    if deviation > tolerance:
        return [f"lm: split-half rotation deviates by {deviation:.3e}"]
    return []


def bakes_reference_rope(fields, namespace):
    return fields.get(f"{namespace}.rope_precision") == "bf16"


def drop_lm_rope(tensors):
    return {n: v for n, v in tensors.items() if n not in LM_ROPE_TENSORS}


def verify_lm(model_dir, path):
    fields, tensors = read_gguf(path)
    config = load_config(model_dir)
    state = load_lm_state(model_dir)
    expected = lm_expectations(config, state)
    if bakes_reference_rope(fields, LM_KV):
        expected.update(lm_rope_expectations(model_dir, config))
    else:
        print(f"  note: {path} bakes non-reference RoPE tables; skipping their check")
        tensors = drop_lm_rope(tensors)
    worst, counts, failures = compare(tensors, expected, "lm")
    report(f"lm ({path})", worst, counts)
    failures += missing_from_gguf(tensors, expected, "lm")
    return failures + verify_rope_layout(model_dir, state, tensors, config)


def reference_codec_rope(model_dir, length, head_dim, theta):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    rope = get_class_from_dynamic_module("modeling_arktts_codec._rope", model_dir)
    return rope(length, head_dim, theta).float().numpy()


def declared_transformers(fields):
    """The transformers this half says it carries. Taken from the metadata the
    converter writes per part rather than from the tensors, so a missing RoPE
    table shows up as missing instead of going unlooked-for."""
    prefix, suffix = f"{CODEC_KV}.", ".head_dim"
    return sorted(
        key[len(prefix):-len(suffix)]
        for key in fields if key.startswith(prefix) and key.endswith(suffix)
    )


def rope_plane_names(fields):
    return {
        f"rope/{transformer}_{plane}"
        for transformer in declared_transformers(fields)
        for plane in ("cos", "sin")
    }


def codec_rope_expectations(model_dir, fields):
    frames = int(fields[f"{CODEC_KV}.max_frames"])
    expectations = {}
    for transformer in declared_transformers(fields):
        expectations.update(rope_plane_expectations(
            f"rope/{transformer}",
            reference_codec_rope(
                model_dir,
                frames,
                int(fields[f"{CODEC_KV}.{transformer}.head_dim"]),
                float(fields[f"{CODEC_KV}.{transformer}.rope_theta"]),
            ),
        ))
    return expectations


def drop_rope(tensors):
    return {n: v for n, v in tensors.items() if not n.startswith("rope/")}


def verify_codec(model_dir, path, part, checkpoint, fields, tensors):
    label = f"codec/{part}"
    declared = fields.get(f"{CODEC_KV}.part")
    if declared != part:
        return [f"codec: {path} declares part {declared!r}, expected {part!r}"]
    emitted = set(tensors)
    expected = select_part(checkpoint, part)
    # RoPE tables are the one thing the checkpoint does not hold, so their
    # names are required even in a build whose values cannot be compared.
    required = set(expected) | rope_plane_names(fields)
    if bakes_reference_rope(fields, CODEC_KV):
        expected.update(codec_rope_expectations(model_dir, fields))
    else:
        print(f"  note: {path} bakes non-reference RoPE tables; skipping their check")
        tensors = drop_rope(tensors)
    worst, counts, failures = compare(tensors, expected, label)
    report(f"{label} ({path})", worst, counts)
    return failures + missing_from_gguf(emitted, required, label)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--lm-gguf")
    parser.add_argument("--codec-encoder-gguf")
    parser.add_argument("--codec-decoder-gguf")
    args = parser.parse_args()
    # Without one there is nothing to check, and a run that checks nothing
    # would otherwise end in the success line.
    if not (args.lm_gguf or args.codec_encoder_gguf or args.codec_decoder_gguf):
        parser.error("name at least one GGUF: --lm-gguf, --codec-encoder-gguf "
                     "or --codec-decoder-gguf")
    return args


def codec_paths(args):
    requested = (
        (args.codec_encoder_gguf, "encoder"),
        (args.codec_decoder_gguf, "decoder"),
    )
    return [(path, part) for path, part in requested if path]


def verify_codec_halves(model_dir, args):
    """Each half is checked against what the split says it should carry, not
    against the pair's union: a tensor in the wrong file satisfies the union
    while the runtime loader still rejects the half it is missing from."""
    paths = codec_paths(args)
    if not paths:
        return []
    checkpoint = codec_expectations(model_dir)
    print("\nnote: the codec expectation is built through the converter's own "
          "layout helpers,\n      so tensor values are checked but their axis "
          "order is not")
    failures = []
    for path, part in paths:
        fields, tensors = read_gguf(path)
        failures += verify_codec(model_dir, path, part, checkpoint, fields, tensors)
    return failures


def main():
    args = parse_args()
    failures = []
    if args.lm_gguf:
        failures += verify_lm(args.model_dir, args.lm_gguf)
    failures += verify_codec_halves(args.model_dir, args)
    if failures:
        print(f"\nFAILED with {len(failures)} problem(s):")
        for line in failures[:40]:
            print(f"  {line}")
        sys.exit(1)
    print("\nevery converted tensor matches the reference checkpoint in value, "
          "shape and dtype")


if __name__ == "__main__":
    main()
