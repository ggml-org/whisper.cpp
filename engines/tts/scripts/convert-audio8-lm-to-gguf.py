#!/usr/bin/env python3
"""Convert the Audio8 TTS DualAR model (model.safetensors) to GGUF.

The checkpoint holds two autoregressive transformers that share a block shape:

  slow AR : 24 layers, hidden 896, 14 heads / 2 KV heads, head_dim 64,
            SwiGLU intermediate 4864, RMSNorm eps 1e-6, RoPE theta 1e6, QKV bias.
            Emits one semantic token per 21.5 Hz codec frame.
  fast AR :  4 layers, same shape without QKV bias. Expands each semantic token
            into the frame's remaining codec codebooks.

The fused `wqkv` matrices are split into separate q/k/v tensors so the ggml
graph can quantize and address them individually. The text head is tied to the
input embedding; because the semantic sampler masks every logit outside
[semantic_begin, semantic_end] plus EOS, we additionally emit `lm/sem_head`,
the 4097 rows of that embedding the sampler can actually select. Restricting
the head to those rows is exact -- the masked-out logits contribute nothing to
the top-k/top-p renormalisation -- and turns a 155776-row projection per step
into a 4097-row one.

The RoPE tables are baked rather than recomputed at run time. The reference
builds them in bfloat16 even under float32 inference, and that ~2e-3 rounding
is not cosmetic: recomputing the tables in float32 changes the greedy fast-AR
codebook choices on the very first frame and the trajectories separate from
there. Baking the reference's own table is what makes exact parity reachable.
They are emitted as separate `_cos` and `_sin` planes, which is the shape the
engine's rotation consumes.

The reference rotates the adjacent pair (2j, 2j+1) of each head; ggml and the
rest of this tree rotate (j, head_dim / 2 + j). Rather than pay for a strided
gather in every attention block, the q and k projection rows are reordered here
into the split-half layout. Attention cannot tell the difference, because the
same permutation lands on q and k and their dot product does not depend on the
order of the terms.

The Qwen2 byte-level BPE vocabulary, merges and added tokens are embedded so
the GGUF is self-contained.

    python3 convert-audio8-lm-to-gguf.py \\
        --model-dir models/Audio8-TTS-Preview-0.6b \\
        --outfile audio8-lm-f32.gguf --dtype f32
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio8_reference import (  # noqa: E402
    BLOCK,
    DENSE,
    EXACT,
    LM_ARCH as ARCH,
    LM_KV as KV,
    QUANT_BLOCK,
    ROPE_PRECISIONS,
    STORAGE_TYPES,
    expand_fused,
    flush,
    format_tally,
    load_config,
    rope_planes,
    write_tensors,
)

EMBEDDING_TABLES = ("lm/tok_emb", "lm/codebook_emb", "fast/emb")


@dataclass(frozen=True)
class Hyperparams:
    depth: int
    hidden: int
    n_head: int
    n_kv: int
    head_dim: int
    inter: int
    vocab: int
    fast_depth: int
    fast_hidden: int
    fast_n_head: int
    fast_n_kv: int
    fast_head_dim: int
    fast_inter: int
    num_codebooks: int
    codebook_size: int
    semantic_begin: int
    semantic_end: int
    eos: int
    pad: int
    max_seq_len: int
    ras_window: int
    rope_theta: float
    rms_eps: float
    ras_top_p: float
    ras_temperature: float
    norm_fast_input: bool
    qkv_bias: bool
    fast_qkv_bias: bool


def read_hyperparams(config):
    return Hyperparams(
        depth=int(config["n_layer"]),
        hidden=int(config["dim"]),
        n_head=int(config["n_head"]),
        n_kv=int(config["n_local_heads"]),
        head_dim=int(config["head_dim"]),
        inter=int(config["intermediate_size"]),
        vocab=int(config["vocab_size"]),
        fast_depth=int(config["n_fast_layer"]),
        fast_hidden=int(config["fast_dim"]),
        fast_n_head=int(config["fast_n_head"]),
        fast_n_kv=int(config["fast_n_local_heads"]),
        fast_head_dim=int(config["fast_head_dim"]),
        fast_inter=int(config["fast_intermediate_size"]),
        num_codebooks=int(config["num_codebooks"]),
        codebook_size=int(config["codebook_size"]),
        semantic_begin=int(config["semantic_begin_id"]),
        semantic_end=int(config["semantic_end_id"]),
        eos=int(config["eos_token_id"]),
        pad=int(config["pad_token_id"]),
        max_seq_len=int(config["max_seq_len"]),
        ras_window=int(config["ras_window_size"]),
        rope_theta=float(config["rope_base"]),
        rms_eps=float(config["norm_eps"]),
        ras_top_p=float(config["ras_top_p"]),
        ras_temperature=float(config["ras_temperature"]),
        norm_fast_input=bool(config["norm_fastlayer_input"]),
        qkv_bias=bool(config["attention_qkv_bias"]),
        fast_qkv_bias=bool(config["fast_attention_qkv_bias"]),
    )


def check_supported(hp, config):
    if config["model_type"] != "arktts":
        raise ValueError(f"unexpected model_type {config['model_type']!r}")
    if not config["tie_word_embeddings"]:
        raise ValueError("the semantic head assumes a tied input embedding")
    if config["attention_qk_norm"] or config["fast_attention_qk_norm"]:
        raise ValueError("QK-norm is not supported")
    if config["attention_o_bias"] or config["fast_attention_o_bias"]:
        raise ValueError("attention output bias is not supported")
    if hp.fast_hidden != hp.hidden:
        raise ValueError("fast_project_in is only an identity when fast_dim == dim")
    if hp.semantic_end - hp.semantic_begin + 1 != hp.codebook_size:
        raise ValueError("the semantic token range must cover exactly one codebook")


def load_state_dict(model_dir):
    import torch
    from safetensors.torch import load_file

    tensors = load_file(os.path.join(model_dir, "model.safetensors"))
    return {k: v.to(torch.float32).numpy() for k, v in tensors.items()}


def attention_tensors(prefix, block, n_head, n_kv, head_dim, has_bias):
    named = {
        f"{prefix}/wo": block["attention.wo.weight"],
        f"{prefix}/attn_norm": block["attention_norm.weight"],
    }
    named.update(expand_fused(
        prefix, block["attention.wqkv.weight"], n_head, n_kv, head_dim
    ))
    if has_bias:
        named.update(expand_fused(
            prefix, block["attention.wqkv.bias"], n_head, n_kv, head_dim, suffix="_b"
        ))
    return named


def feed_forward_tensors(prefix, block):
    return {
        f"{prefix}/w1": block["feed_forward.w1.weight"],
        f"{prefix}/w2": block["feed_forward.w2.weight"],
        f"{prefix}/w3": block["feed_forward.w3.weight"],
        f"{prefix}/ffn_norm": block["ffn_norm.weight"],
    }


def collect_block(state, source_prefix, index):
    head = f"{source_prefix}.{index}."
    return {k[len(head):]: v for k, v in state.items() if k.startswith(head)}


def branch_tensors(state, source_prefix, out_prefix, depth, n_head, n_kv, head_dim, has_bias):
    named = {}
    for index in range(depth):
        block = collect_block(state, source_prefix, index)
        prefix = f"{out_prefix}/blk/{index}"
        named.update(attention_tensors(prefix, block, n_head, n_kv, head_dim, has_bias))
        named.update(feed_forward_tensors(prefix, block))
    return named


def semantic_head(embeddings, hp):
    rows = list(range(hp.semantic_begin, hp.semantic_end + 1)) + [hp.eos]
    return np.ascontiguousarray(embeddings[rows])


def rope_tables(hp, precision):
    tables = rope_planes("lm/rope", hp.max_seq_len, hp.head_dim, hp.rope_theta, precision)
    tables.update(rope_planes(
        "fast/rope", hp.num_codebooks, hp.fast_head_dim, hp.rope_theta, precision
    ))
    return tables


def model_tensors(state, hp):
    named = {
        "lm/tok_emb": state["embeddings.weight"],
        "lm/codebook_emb": state["codebook_embeddings.weight"],
        "lm/norm": state["norm.weight"],
        "lm/sem_head": semantic_head(state["embeddings.weight"], hp),
        "fast/emb": state["fast_embeddings.weight"],
        "fast/norm": state["fast_norm.weight"],
        "fast/out": state["fast_output.weight"],
    }
    named.update(branch_tensors(
        state, "layers", "lm", hp.depth, hp.n_head, hp.n_kv, hp.head_dim, hp.qkv_bias
    ))
    named.update(branch_tensors(
        state, "fast_layers", "fast", hp.fast_depth, hp.fast_n_head, hp.fast_n_kv,
        hp.fast_head_dim, hp.fast_qkv_bias,
    ))
    return named


def write_metadata(writer, hp):
    for key, value in dict(
        depth=hp.depth, hidden=hp.hidden, n_head=hp.n_head, n_kv=hp.n_kv,
        head_dim=hp.head_dim, inter=hp.inter, vocab=hp.vocab,
        fast_depth=hp.fast_depth, fast_hidden=hp.fast_hidden,
        fast_n_head=hp.fast_n_head, fast_n_kv=hp.fast_n_kv,
        fast_head_dim=hp.fast_head_dim, fast_inter=hp.fast_inter,
        num_codebooks=hp.num_codebooks, codebook_size=hp.codebook_size,
        semantic_begin=hp.semantic_begin, semantic_end=hp.semantic_end,
        eos=hp.eos, pad=hp.pad, max_seq_len=hp.max_seq_len,
        ras_window=hp.ras_window,
    ).items():
        writer.add_uint32(f"{KV}.{key}", int(value))
    for key, value in dict(
        rope_theta=hp.rope_theta, rms_eps=hp.rms_eps,
        ras_top_p=hp.ras_top_p, ras_temperature=hp.ras_temperature,
    ).items():
        writer.add_float32(f"{KV}.{key}", float(value))
    writer.add_bool(f"{KV}.norm_fast_input", hp.norm_fast_input)
    writer.add_bool(f"{KV}.qkv_bias", hp.qkv_bias)
    writer.add_bool(f"{KV}.fast_qkv_bias", hp.fast_qkv_bias)


def read_tokenizer(model_dir):
    with open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8") as handle:
        return json.load(handle)


def token_list(tokenizer, vocab_size):
    tokens = [""] * vocab_size
    for text, index in tokenizer["model"]["vocab"].items():
        tokens[index] = text
    for entry in tokenizer["added_tokens"]:
        tokens[entry["id"]] = entry["content"]
    return tokens


def merge_list(tokenizer):
    return [" ".join(pair) for pair in tokenizer["model"]["merges"]]


def added_token_ids(tokenizer):
    return [entry["id"] for entry in tokenizer["added_tokens"]]


def write_tokenizer(writer, model_dir, hp):
    tokenizer = read_tokenizer(model_dir)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(token_list(tokenizer, hp.vocab))
    writer.add_token_merges(merge_list(tokenizer))
    writer.add_array("tokenizer.ggml.added_token_ids", added_token_ids(tokenizer))
    writer.add_eos_token_id(hp.eos)
    writer.add_pad_token_id(hp.pad)


def is_body_matmul(name, array):
    return (
        array.ndim == 2
        and "/blk/" in name
        and not name.endswith(("_norm", "_b"))
        and array.shape[1] % QUANT_BLOCK == 0
    )


def role_of(name, array):
    """Everything outside the body and the embedding tables stays exact: that
    leaves the RoPE tables, the two sampling heads and the norms, which between
    them decide every greedy choice and add up to a few megabytes."""
    if is_body_matmul(name, array):
        return BLOCK
    if name in EMBEDDING_TABLES:
        return DENSE
    return EXACT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--dtype", choices=STORAGE_TYPES, default="f32")
    parser.add_argument("--rope-precision", choices=ROPE_PRECISIONS, default="bf16",
                        help="bf16 reproduces the reference's own table rounding")
    return parser.parse_args()


def main():
    import gguf

    args = parse_args()
    config = load_config(args.model_dir)
    hp = read_hyperparams(config)
    check_supported(hp, config)
    print(
        f"slow AR: depth={hp.depth} hidden={hp.hidden} n_head={hp.n_head} "
        f"n_kv={hp.n_kv} head_dim={hp.head_dim} inter={hp.inter}"
    )
    print(
        f"fast AR: depth={hp.fast_depth} hidden={hp.fast_hidden} "
        f"n_head={hp.fast_n_head} n_kv={hp.fast_n_kv} inter={hp.fast_inter}"
    )

    named = model_tensors(load_state_dict(args.model_dir), hp)
    named.update(rope_tables(hp, args.rope_precision))
    writer = gguf.GGUFWriter(args.outfile, ARCH)
    write_metadata(writer, hp)
    writer.add_string(f"{KV}.rope_precision", args.rope_precision)
    write_tokenizer(writer, args.model_dir, hp)
    tally = write_tensors(writer, named, args.dtype, role_of)
    flush(writer)
    print(f"wrote {len(named)} tensors ({format_tally(tally)}) -> {args.outfile}")


if __name__ == "__main__":
    main()
