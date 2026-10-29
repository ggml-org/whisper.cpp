#!/usr/bin/env python3
"""Convert a Hugging Face MOSS-SoundEffect checkpoint (diffusers layout:
Qwen3 text encoder, Wan audio DiT, continuous DAC VAE) into the single
moss-sfx GGUF that the tts-cpp MOSS sound-effect engine loads.

usage: convert-moss-sfx-to-gguf.py <hf_checkpoint_dir> [--outtype f16|bf16|f32|q8_0|q4_0]
           [--text-block-outtype TYPE] [--dit-block-outtype TYPE] [--outfile out.gguf]
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import gguf
import numpy as np

ARCH = "moss-sfx"
TEXT_LENGTH = 512
DEFAULT_STEPS = 100
DEFAULT_GUIDANCE = 4.0
DEFAULT_SHIFT = 5.0
DEFAULT_MAX_SECONDS = 30.0
SNAKE_EPSILON = 1e-9
RESIDUAL_UNITS = 3
TEXT_LAYER_TENSORS = 11
DIT_LAYER_TENSORS = 27
DIT_GLOBAL_TENSORS = 15
TEXT_GLOBAL_TENSORS = 2
SNAKE_TENSORS = 2
CONV_TENSORS = 2
POST_QUANT_TENSORS = 2
DEFAULT_TRAIN_TIMESTEPS = 1000
DEFAULT_PAD_TOKEN = "<|endoftext|>"

SAFETENSORS_DTYPES = {
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
}

OUTTYPES = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "bf16": gguf.GGMLQuantizationType.BF16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
}

TEXT_BLOCK_PREFIX = "text.blk."
DIT_BLOCK_PREFIX = "dit.blk."

TEXT_LAYER_RENAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}

DIT_LAYER_RENAMES = {
    "attn1.to_q.weight": "self_q.weight",
    "attn1.to_q.bias": "self_q.bias",
    "attn1.to_k.weight": "self_k.weight",
    "attn1.to_k.bias": "self_k.bias",
    "attn1.to_v.weight": "self_v.weight",
    "attn1.to_v.bias": "self_v.bias",
    "attn1.to_out.0.weight": "self_o.weight",
    "attn1.to_out.0.bias": "self_o.bias",
    "attn1.norm_q.weight": "self_norm_q.weight",
    "attn1.norm_k.weight": "self_norm_k.weight",
    "attn2.to_q.weight": "cross_q.weight",
    "attn2.to_q.bias": "cross_q.bias",
    "attn2.to_k.weight": "cross_k.weight",
    "attn2.to_k.bias": "cross_k.bias",
    "attn2.to_v.weight": "cross_v.weight",
    "attn2.to_v.bias": "cross_v.bias",
    "attn2.to_out.0.weight": "cross_o.weight",
    "attn2.to_out.0.bias": "cross_o.bias",
    "attn2.norm_q.weight": "cross_norm_q.weight",
    "attn2.norm_k.weight": "cross_norm_k.weight",
    "norm2.weight": "cross_norm.weight",
    "norm2.bias": "cross_norm.bias",
    "ffn.net.0.proj.weight": "ffn_up.weight",
    "ffn.net.0.proj.bias": "ffn_up.bias",
    "ffn.net.2.weight": "ffn_down.weight",
    "ffn.net.2.bias": "ffn_down.bias",
    "scale_shift_table": "modulation",
}

DIT_GLOBAL_RENAMES = {
    "patch_embedding.weight": "dit.patch_embd.weight",
    "patch_embedding.bias": "dit.patch_embd.bias",
    "condition_embedder.text_embedder.linear_1.weight": "dit.text_embd_1.weight",
    "condition_embedder.text_embedder.linear_1.bias": "dit.text_embd_1.bias",
    "condition_embedder.text_embedder.linear_2.weight": "dit.text_embd_2.weight",
    "condition_embedder.text_embedder.linear_2.bias": "dit.text_embd_2.bias",
    "condition_embedder.time_embedder.linear_1.weight": "dit.time_embd_1.weight",
    "condition_embedder.time_embedder.linear_1.bias": "dit.time_embd_1.bias",
    "condition_embedder.time_embedder.linear_2.weight": "dit.time_embd_2.weight",
    "condition_embedder.time_embedder.linear_2.bias": "dit.time_embd_2.bias",
    "condition_embedder.time_proj.weight": "dit.time_proj.weight",
    "condition_embedder.time_proj.bias": "dit.time_proj.bias",
    "proj_out.weight": "dit.head.weight",
    "proj_out.bias": "dit.head.bias",
    "scale_shift_table": "dit.head.modulation",
}


@dataclass(frozen=True)
class TensorLocation:
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    data_start: int


class SafeTensorsIndex:
    def __init__(self, model_dir: Path):
        self.locations: OrderedDict[str, TensorLocation] = OrderedDict()
        self._index_shards(model_dir)
        if not self.locations:
            raise FileNotFoundError(f"no safetensors files under {model_dir}")

    def _index_shards(self, model_dir: Path) -> None:
        for shard_path in sorted(model_dir.glob("*.safetensors")):
            self._index_shard(shard_path)

    def _index_shard(self, shard_path: Path) -> None:
        with shard_path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        for tensor_name, meta in header.items():
            if tensor_name == "__metadata__":
                continue
            self.locations[tensor_name] = TensorLocation(
                shard=shard_path,
                dtype=meta["dtype"],
                shape=tuple(int(v) for v in meta["shape"]),
                data_offsets=(int(meta["data_offsets"][0]), int(meta["data_offsets"][1])),
                data_start=data_start,
            )

    def __iter__(self) -> Iterator[str]:
        return iter(self.locations.keys())

    def __contains__(self, name: str) -> bool:
        return name in self.locations

    def load(self, name: str) -> np.ndarray:
        loc = self.locations[name]
        offset = loc.data_start + loc.data_offsets[0]
        if loc.dtype == "BF16":
            raw = np.memmap(loc.shard, mode="r", dtype=np.uint16, offset=offset, shape=loc.shape)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        dtype = SAFETENSORS_DTYPES.get(loc.dtype)
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype {loc.dtype!r} for {name!r}")
        return np.asarray(np.memmap(loc.shard, mode="r", dtype=dtype, offset=offset, shape=loc.shape))


@dataclass(frozen=True)
class MatrixTypes:
    default: str
    text_blocks: str
    dit_blocks: str

    def for_tensor(self, name: str) -> str:
        if name.startswith(TEXT_BLOCK_PREFIX):
            return self.text_blocks
        if name.startswith(DIT_BLOCK_PREFIX):
            return self.dit_blocks
        return self.default

    def tag(self) -> str:
        overrides = [f"{part}-{outtype}" for part, outtype in (("text", self.text_blocks), ("dit", self.dit_blocks))
                     if outtype != self.default]
        return "-".join([self.default, *overrides])


class Emitter:
    def __init__(self, writer: gguf.GGUFWriter, types: MatrixTypes):
        self.writer = writer
        self.types = types
        self.names: set[str] = set()

    def matrix_type(self, name: str) -> gguf.GGMLQuantizationType:
        return OUTTYPES[self.types.for_tensor(name)]

    def matrix(self, name: str, tensor: np.ndarray) -> None:
        data = np.ascontiguousarray(tensor, dtype=np.float32)
        qtype = self.matrix_type(name)
        if qtype == gguf.GGMLQuantizationType.F32:
            self._add(name, data)
            return
        if qtype == gguf.GGMLQuantizationType.F16:
            self._add(name, data.astype(np.float16))
            return
        self._add(name, gguf.quants.quantize(data, qtype), raw_dtype=qtype)

    def conv(self, name: str, tensor: np.ndarray) -> None:
        self._add(name, np.ascontiguousarray(tensor, dtype=np.float16))

    def exact(self, name: str, tensor: np.ndarray) -> None:
        self._add(name, np.ascontiguousarray(tensor, dtype=np.float32))

    def _add(self, name: str, data: np.ndarray, raw_dtype: Any = None) -> None:
        if name in self.names:
            raise RuntimeError(f"duplicate tensor {name}")
        self.writer.add_tensor(name, data, raw_dtype=raw_dtype)
        self.names.add(name)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def map_text_tensor_name(name: str) -> str | None:
    if name.startswith("model."):
        name = name[len("model."):]
    if name == "embed_tokens.weight":
        return "text.token_embd.weight"
    if name == "norm.weight":
        return "text.output_norm.weight"
    if (match := re.fullmatch(r"layers\.(\d+)\.(.+)", name)) is not None:
        mapped = TEXT_LAYER_RENAMES.get(match.group(2))
        return f"text.blk.{match.group(1)}.{mapped}" if mapped else None
    return None


def map_dit_tensor_name(name: str) -> str | None:
    if name in DIT_GLOBAL_RENAMES:
        return DIT_GLOBAL_RENAMES[name]
    if (match := re.fullmatch(r"blocks\.(\d+)\.(.+)", name)) is not None:
        mapped = DIT_LAYER_RENAMES.get(match.group(2))
        return f"dit.blk.{match.group(1)}.{mapped}" if mapped else None
    return None


def is_text_matrix(mapped: str) -> bool:
    return mapped.endswith(".weight") and "norm" not in mapped


def is_dit_matrix(mapped: str) -> bool:
    return mapped.endswith(".weight") and "norm" not in mapped and "patch_embd" not in mapped


def squeeze_leading(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(tensor.shape[1:]) if tensor.ndim == 3 and tensor.shape[0] == 1 else tensor


def emit_text_encoder(emitter: Emitter, index: SafeTensorsIndex) -> None:
    for name in index:
        mapped = map_text_tensor_name(name)
        if mapped is None:
            continue
        tensor = index.load(name)
        if is_text_matrix(mapped):
            emitter.matrix(mapped, tensor)
        else:
            emitter.exact(mapped, tensor)


def emit_dit(emitter: Emitter, index: SafeTensorsIndex) -> None:
    for name in index:
        mapped = map_dit_tensor_name(name)
        if mapped is None:
            raise RuntimeError(f"unmapped DiT tensor {name}")
        tensor = index.load(name)
        if mapped == "dit.patch_embd.weight":
            emitter.exact(mapped, tensor.reshape(tensor.shape[0], -1))
        elif is_dit_matrix(mapped):
            emitter.matrix(mapped, tensor)
        else:
            emitter.exact(mapped, squeeze_leading(tensor))


def fold_weight_norm(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, v.ndim))
    norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=axes, keepdims=True))
    return (v * (g.reshape(norm.shape) / norm)).astype(np.float32)


def transposed_conv_columns(weight: np.ndarray) -> np.ndarray:
    in_channels, out_channels, kernel = weight.shape
    return weight.reshape(in_channels, out_channels * kernel).T


def column(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(-1, 1)


class VaeWeights:
    def __init__(self, loader: Callable[[str], np.ndarray]):
        self.loader = loader

    def plain(self, name: str) -> np.ndarray:
        return np.asarray(self.loader(name), dtype=np.float32)

    def folded(self, prefix: str) -> np.ndarray:
        return fold_weight_norm(self.plain(prefix + ".weight_g"), self.plain(prefix + ".weight_v"))


def load_vae_weights(vae_dir: Path) -> tuple[VaeWeights, dict[str, Any]]:
    config = read_json(vae_dir / "config.json") if (vae_dir / "config.json").exists() else {}
    if list(vae_dir.glob("*.safetensors")):
        index = SafeTensorsIndex(vae_dir)
        return VaeWeights(index.load), config
    weights, kwargs = load_vae_checkpoint(vae_dir)
    return weights, {**config, **kwargs}


def load_vae_checkpoint(vae_dir: Path) -> tuple[VaeWeights, dict[str, Any]]:
    import torch

    checkpoint_path = next(iter(sorted(vae_dir.glob("*.pth"))), None)
    if checkpoint_path is None:
        raise FileNotFoundError(f"no VAE weights under {vae_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    kwargs = checkpoint.get("metadata", {}).get("kwargs", {})
    if not kwargs.get("continuous", False):
        raise ValueError("only the continuous DAC VAE is supported")
    return VaeWeights(lambda name: state[name].float().numpy()), kwargs


def emit_snake(emitter: Emitter, name: str, alpha: np.ndarray) -> None:
    alpha = alpha.reshape(-1)
    emitter.exact(name + ".alpha", column(alpha))
    emitter.exact(name + ".inv", column(1.0 / (alpha + SNAKE_EPSILON)))


def emit_conv(emitter: Emitter, name: str, weights: VaeWeights, prefix: str) -> None:
    emitter.conv(name + ".weight", weights.folded(prefix))
    emitter.exact(name + ".bias", column(weights.plain(prefix + ".bias")))


def emit_residual_unit(emitter: Emitter, name: str, weights: VaeWeights, prefix: str) -> None:
    emit_snake(emitter, name + ".snake1", weights.plain(prefix + ".block.0.alpha"))
    emit_conv(emitter, name + ".conv1", weights, prefix + ".block.1")
    emit_snake(emitter, name + ".snake2", weights.plain(prefix + ".block.2.alpha"))
    emit_conv(emitter, name + ".conv2", weights, prefix + ".block.3")


def emit_decoder_block(emitter: Emitter, block: int, weights: VaeWeights) -> None:
    name = f"vae.blk.{block}"
    prefix = f"decoder.model.{block + 1}.block"
    emit_snake(emitter, name + ".snake", weights.plain(prefix + ".0.alpha"))
    emitter.conv(name + ".up.weight", transposed_conv_columns(weights.folded(prefix + ".1")))
    emitter.exact(name + ".up.bias", column(weights.plain(prefix + ".1.bias")))
    for unit in range(RESIDUAL_UNITS):
        emit_residual_unit(emitter, f"{name}.res.{unit}", weights, f"{prefix}.{unit + 2}")


def emit_decoder_blocks(emitter: Emitter, weights: VaeWeights, n_blocks: int) -> None:
    for block in range(n_blocks):
        emit_decoder_block(emitter, block, weights)


def emit_vae(emitter: Emitter, weights: VaeWeights, rates: list[int]) -> None:
    post = weights.plain("post_quant_conv.weight")
    emitter.exact("vae.post_quant.weight", post.reshape(post.shape[0], post.shape[1]))
    emitter.exact("vae.post_quant.bias", column(weights.plain("post_quant_conv.bias")))
    emit_conv(emitter, "vae.conv_in", weights, "decoder.model.0")
    emit_decoder_blocks(emitter, weights, len(rates))
    last = len(rates) + 1
    emit_snake(emitter, "vae.snake_out", weights.plain(f"decoder.model.{last}.alpha"))
    emit_conv(emitter, "vae.conv_out", weights, f"decoder.model.{last + 1}")


def vocabulary_size(vocab: dict[str, int], added_tokens: list[dict]) -> int:
    return max([max(vocab.values())] + [int(added["id"]) for added in added_tokens]) + 1


def place_vocabulary(tokens: list[str], vocab: dict[str, int]) -> None:
    for token, token_id in vocab.items():
        tokens[token_id] = token


def place_added_tokens(tokens: list[str], added_tokens: list[dict]) -> None:
    for added in added_tokens:
        tokens[int(added["id"])] = added["content"]


def place_tokens(tokens: list[str], vocab: dict[str, int], added_tokens: list[dict]) -> None:
    place_vocabulary(tokens, vocab)
    place_added_tokens(tokens, added_tokens)


def read_tokenizer(tokenizer_dir: Path) -> tuple[list[str], list[str]]:
    tokenizer = read_json(tokenizer_dir / "tokenizer.json")
    model = tokenizer["model"]
    added_tokens = tokenizer.get("added_tokens", [])
    tokens = [""] * vocabulary_size(model["vocab"], added_tokens)
    place_tokens(tokens, model["vocab"], added_tokens)
    merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in model["merges"]]
    return tokens, merges


def pad_token_id(tokenizer_dir: Path, tokens: list[str]) -> int:
    config = read_json(tokenizer_dir / "tokenizer_config.json")
    pad = config.get("pad_token")
    pad = pad.get("content") if isinstance(pad, dict) else pad
    return tokens.index(pad or DEFAULT_PAD_TOKEN)


def add_tokenizer(writer: gguf.GGUFWriter, tokenizer_dir: Path) -> None:
    tokens, merges = read_tokenizer(tokenizer_dir)
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.merges", merges)
    writer.add_uint32("tokenizer.ggml.padding_token_id", pad_token_id(tokenizer_dir, tokens))


def add_text_metadata(writer: gguf.GGUFWriter, config: dict[str, Any]) -> None:
    n_heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim", config["hidden_size"] // n_heads))
    key = f"{ARCH}.text"
    writer.add_uint32(f"{key}.block_count", int(config["num_hidden_layers"]))
    writer.add_uint32(f"{key}.context_length", TEXT_LENGTH)
    writer.add_uint32(f"{key}.embedding_length", int(config["hidden_size"]))
    writer.add_uint32(f"{key}.feed_forward_length", int(config["intermediate_size"]))
    writer.add_uint32(f"{key}.attention.head_count", n_heads)
    writer.add_uint32(f"{key}.attention.head_count_kv", int(config["num_key_value_heads"]))
    writer.add_uint32(f"{key}.attention.key_length", head_dim)
    writer.add_float32(f"{key}.rope.freq_base", float(config.get("rope_theta", 10000.0)))
    writer.add_float32(f"{key}.attention.layer_norm_rms_epsilon", float(config.get("rms_norm_eps", 1e-6)))


def add_dit_metadata(writer: gguf.GGUFWriter, config: dict[str, Any]) -> None:
    key = f"{ARCH}.dit"
    if config.get("vae_type", "dac") != "dac" or config.get("has_image_input", False):
        raise ValueError("only the text-conditioned DAC DiT is supported")
    if list(config.get("patch_size", [1])) != [1]:
        raise ValueError("only patch size 1 is supported")
    writer.add_uint32(f"{key}.block_count", int(config["num_layers"]))
    writer.add_uint32(f"{key}.embedding_length", int(config["dim"]))
    writer.add_uint32(f"{key}.feed_forward_length", int(config["ffn_dim"]))
    writer.add_uint32(f"{key}.attention.head_count", int(config["num_heads"]))
    writer.add_uint32(f"{key}.in_channels", int(config["in_dim"]))
    writer.add_uint32(f"{key}.out_channels", int(config["out_dim"]))
    writer.add_uint32(f"{key}.text_dim", int(config["text_dim"]))
    writer.add_uint32(f"{key}.freq_dim", int(config["freq_dim"]))
    writer.add_float32(f"{key}.epsilon", float(config["eps"]))


def add_vae_metadata(writer: gguf.GGUFWriter, kwargs: dict[str, Any]) -> None:
    key = f"{ARCH}.vae"
    rates = [int(rate) for rate in kwargs["decoder_rates"]]
    writer.add_uint32(f"{key}.latent_dim", int(kwargs["latent_dim"]))
    writer.add_uint32(f"{key}.decoder_dim", int(kwargs["decoder_dim"]))
    writer.add_array(f"{key}.decoder_rates", rates)
    writer.add_uint32(f"{key}.sample_rate", int(kwargs["sample_rate"]))


def add_generation_metadata(writer: gguf.GGUFWriter, index: dict[str, Any], scheduler: dict[str, Any]) -> None:
    writer.add_float32(f"{ARCH}.max_seconds", float(index.get("max_inference_seconds", DEFAULT_MAX_SECONDS)))
    writer.add_float32(f"{ARCH}.sigma_shift", float(scheduler.get("shift", DEFAULT_SHIFT)))
    writer.add_uint32(f"{ARCH}.num_train_timesteps", int(scheduler.get("num_train_timesteps", DEFAULT_TRAIN_TIMESTEPS)))
    writer.add_uint32(f"{ARCH}.default_steps", DEFAULT_STEPS)
    writer.add_float32(f"{ARCH}.default_guidance", DEFAULT_GUIDANCE)


def vae_tensor_count(n_blocks: int) -> int:
    per_unit = 2 * SNAKE_TENSORS + 2 * CONV_TENSORS
    per_block = SNAKE_TENSORS + CONV_TENSORS + RESIDUAL_UNITS * per_unit
    edges = POST_QUANT_TENSORS + CONV_TENSORS + SNAKE_TENSORS + CONV_TENSORS
    return edges + n_blocks * per_block


def expected_tensor_count(text: dict[str, Any], dit: dict[str, Any], n_blocks: int) -> int:
    return (TEXT_GLOBAL_TENSORS + TEXT_LAYER_TENSORS * int(text["num_hidden_layers"]) +
            DIT_GLOBAL_TENSORS + DIT_LAYER_TENSORS * int(dit["num_layers"]) +
            vae_tensor_count(n_blocks))


@dataclass
class Checkpoint:
    model_dir: Path
    text: dict[str, Any]
    dit: dict[str, Any]
    index: dict[str, Any]
    scheduler: dict[str, Any]
    vae_weights: VaeWeights
    vae: dict[str, Any]

    @property
    def rates(self) -> list[int]:
        return [int(rate) for rate in self.vae["decoder_rates"]]


def read_checkpoint(model_dir: Path) -> Checkpoint:
    vae_weights, vae_kwargs = load_vae_weights(model_dir / "vae")
    return Checkpoint(
        model_dir=model_dir,
        text=read_json(model_dir / "text_encoder" / "config.json"),
        dit=read_json(model_dir / "transformer" / "config.json"),
        index=read_json(model_dir / "model_index.json"),
        scheduler=read_json(model_dir / "scheduler" / "scheduler_config.json"),
        vae_weights=vae_weights,
        vae=vae_kwargs,
    )


def add_metadata(writer: gguf.GGUFWriter, checkpoint: Checkpoint) -> None:
    writer.add_type("model")
    writer.add_name(checkpoint.model_dir.name)
    add_text_metadata(writer, checkpoint.text)
    add_dit_metadata(writer, checkpoint.dit)
    add_vae_metadata(writer, checkpoint.vae)
    add_generation_metadata(writer, checkpoint.index, checkpoint.scheduler)
    add_tokenizer(writer, checkpoint.model_dir / "tokenizer")


def emit_tensors(writer: gguf.GGUFWriter, checkpoint: Checkpoint, types: MatrixTypes) -> None:
    emitter = Emitter(writer, types)
    emit_text_encoder(emitter, SafeTensorsIndex(checkpoint.model_dir / "text_encoder"))
    emit_dit(emitter, SafeTensorsIndex(checkpoint.model_dir / "transformer"))
    emit_vae(emitter, checkpoint.vae_weights, checkpoint.rates)
    expected = expected_tensor_count(checkpoint.text, checkpoint.dit, len(checkpoint.rates))
    if len(emitter.names) != expected:
        raise RuntimeError(
            f"emitted {len(emitter.names)} tensors but the architecture needs {expected}; "
            "the checkpoint uses tensor names this converter does not map")


def write_gguf(writer: gguf.GGUFWriter) -> None:
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)


def convert(model_dir: Path, outfile: Path, types: MatrixTypes) -> None:
    checkpoint = read_checkpoint(model_dir)
    writer = gguf.GGUFWriter(path=outfile, arch=ARCH)
    try:
        add_metadata(writer, checkpoint)
        emit_tensors(writer, checkpoint, types)
        write_gguf(writer)
        print(f"{ARCH}: wrote {outfile}")
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--outtype", choices=tuple(OUTTYPES), default="f16",
                        help="type of the linear weights and the text token embedding")
    parser.add_argument("--text-block-outtype", choices=tuple(OUTTYPES), default=None,
                        help="type of the text-encoder block linear weights (text.blk.*); defaults to --outtype")
    parser.add_argument("--dit-block-outtype", choices=tuple(OUTTYPES), default=None,
                        help="type of the DiT block linear weights (dit.blk.*); defaults to --outtype")
    parser.add_argument("--outfile", type=Path, default=None)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    types = MatrixTypes(args.outtype, args.text_block_outtype or args.outtype, args.dit_block_outtype or args.outtype)
    outfile = args.outfile or model_dir / f"{model_dir.name}-{types.tag()}.gguf"
    convert(model_dir, outfile, types)


if __name__ == "__main__":
    main()
