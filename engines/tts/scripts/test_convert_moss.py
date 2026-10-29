#!/usr/bin/env python3
"""Converter coverage for the MOSS scripts: fabricates a tiny MossTTSDelay
checkpoint, a tiny MOSS audio tokenizer, a tiny MOSS-SoundEffect pipeline and a
tiny MOSS-Speech checkpoint on disk, runs the converters, and validates the
emitted GGUFs (tensor census, mapped names, folded weights, metadata) with the
gguf reader. The MOSS-Speech codec helpers run in-process when torch,
safetensors and soundfile are installed. Skips cleanly when numpy or gguf are
absent.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    import gguf
except ImportError as error:
    print(f"SKIP: {error.name} is not installed")
    sys.exit(0)

SCRIPTS = Path(__file__).resolve().parent

N_EMBD = 16
N_FF = 32
N_HEADS = 2
N_KV_HEADS = 1
HEAD_DIM = 8
N_VQ = 2
TEXT_VOCAB = 24
AUDIO_HEAD = 8
D_MODEL = 8
RVQ_DIM = 8
CODE_DIM = 4
CODE_SIZE = 8
OUT_DIM = 4
SFX_WIDTH = 32
SFX_FF = 64
SFX_HEAD_DIM = 16
SFX_LATENT = 4
SFX_DECODER_DIM = 8
SFX_RATES = [2, 3]
SFX_KERNEL = 7
SFX_GAIN = 2.0
SPEECH_EMBD = 32
SPEECH_FF = 64
SPEECH_HEADS = 2
SPEECH_KV_HEADS = 1
SPEECH_HEAD_DIM = 16
SPEECH_TEXT_VOCAB = 12
SPEECH_AUDIO_VOCAB = 10
SPEECH_PLACEHOLDER = 9
SPEECH_START = 8
SPEECH_END = 7
SPEECH_AUDIO_PAD = 6
SPEECH_TENSORS = 6 + 11 * 3
VQ_EMBD = 8
VQ_MELS = 4
VQ_CODES = 6
VQ_KERNEL = 3
SFX_BLOCK_MATRICES = ("text.blk.0.attn_q.weight", "text.blk.0.ffn_down.weight",
                      "dit.blk.0.self_q.weight", "dit.blk.0.cross_v.weight", "dit.blk.0.ffn_down.weight")
SFX_GLOBAL_MATRICES = ("text.token_embd.weight", "dit.text_embd_1.weight", "dit.time_embd_2.weight",
                       "dit.time_proj.weight", "dit.head.weight")


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict] = {}
    blobs = []
    offset = 0
    for name, tensor in tensors.items():
        data = np.ascontiguousarray(tensor, dtype=np.float32).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    payload = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(payload)))
        f.write(payload)
        for blob in blobs:
            f.write(blob)


def make_backbone_checkpoint(root: Path) -> Path:
    model_dir = root / "backbone"
    model_dir.mkdir()
    tensors: dict[str, np.ndarray] = {
        "language_model.embed_tokens.weight": np.zeros((TEXT_VOCAB, N_EMBD)),
        "language_model.norm.weight": np.zeros(N_EMBD),
    }
    layer = "language_model.layers.0."
    tensors[layer + "input_layernorm.weight"] = np.zeros(N_EMBD)
    tensors[layer + "self_attn.q_proj.weight"] = np.zeros((N_HEADS * HEAD_DIM, N_EMBD))
    tensors[layer + "self_attn.k_proj.weight"] = np.zeros((N_KV_HEADS * HEAD_DIM, N_EMBD))
    tensors[layer + "self_attn.v_proj.weight"] = np.zeros((N_KV_HEADS * HEAD_DIM, N_EMBD))
    tensors[layer + "self_attn.o_proj.weight"] = np.zeros((N_EMBD, N_HEADS * HEAD_DIM))
    tensors[layer + "self_attn.q_norm.weight"] = np.zeros(HEAD_DIM)
    tensors[layer + "self_attn.k_norm.weight"] = np.zeros(HEAD_DIM)
    tensors[layer + "post_attention_layernorm.weight"] = np.zeros(N_EMBD)
    tensors[layer + "mlp.gate_proj.weight"] = np.zeros((N_FF, N_EMBD))
    tensors[layer + "mlp.up_proj.weight"] = np.zeros((N_FF, N_EMBD))
    tensors[layer + "mlp.down_proj.weight"] = np.zeros((N_EMBD, N_FF))
    for i in range(N_VQ):
        tensors[f"emb_ext.{i}.weight"] = np.zeros((AUDIO_HEAD, N_EMBD))
    for k in range(N_VQ + 1):
        rows = TEXT_VOCAB if k == 0 else AUDIO_HEAD
        tensors[f"lm_heads.{k}.weight"] = np.zeros((rows, N_EMBD))
    write_safetensors(model_dir / "model.safetensors", tensors)

    config = {
        "model_type": "moss_tts_delay",
        "n_vq": N_VQ,
        "audio_vocab_size": AUDIO_HEAD - 1,
        "audio_pad_code": AUDIO_HEAD - 1,
        "audio_start_token_id": 3,
        "audio_end_token_id": 4,
        "audio_user_slot_token_id": 5,
        "audio_assistant_gen_slot_token_id": 6,
        "audio_assistant_delay_slot_token_id": 7,
        "sampling_rate": 24000,
        "language_config": {
            "num_hidden_layers": 1,
            "hidden_size": N_EMBD,
            "intermediate_size": N_FF,
            "num_attention_heads": N_HEADS,
            "num_key_value_heads": N_KV_HEADS,
            "head_dim": HEAD_DIM,
            "max_position_embeddings": 512,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    vocab = {f"tok{i}": i for i in range(TEXT_VOCAB)}
    tokenizer = {
        "model": {"vocab": vocab, "merges": ["a b"]},
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>"},
            {"id": 1, "content": "<|im_start|>"},
            {"id": 2, "content": "<|im_end|>"},
        ],
    }
    (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer))
    return model_dir


def make_codec_checkpoint(root: Path) -> Path:
    model_dir = root / "tokenizer-model"
    model_dir.mkdir()
    tensors: dict[str, np.ndarray] = {
        "quantizer.input_proj.weight": np.zeros((RVQ_DIM, RVQ_DIM, 1)),
        "quantizer.input_proj.bias": np.zeros(RVQ_DIM),
        "quantizer.output_proj.weight": np.zeros((OUT_DIM, RVQ_DIM, 1)),
        "quantizer.output_proj.bias": np.zeros(OUT_DIM),
    }
    for iq in range(N_VQ):
        q = f"quantizer.quantizers.{iq}."
        tensors[q + "codebook.weight"] = np.zeros((CODE_SIZE, CODE_DIM))
        tensors[q + "in_proj.weight"] = np.zeros((CODE_DIM, RVQ_DIM, 1))
        tensors[q + "in_proj.bias"] = np.zeros(CODE_DIM)
        tensors[q + "out_proj.weight"] = np.zeros((RVQ_DIM, CODE_DIM, 1))
        tensors[q + "out_proj.bias"] = np.zeros(RVQ_DIM)
    for prefix, transformer_in in (("encoder.1.", RVQ_DIM), ("decoder.0.", OUT_DIM)):
        layer = prefix + "transformer.layers.0."
        tensors[layer + "self_attn.in_projs.0.weight"] = np.zeros((3 * D_MODEL, D_MODEL, 1))
        tensors[layer + "self_attn.out_projs.0.weight"] = np.zeros((D_MODEL, D_MODEL, 1))
        tensors[layer + "linear1.weight"] = np.zeros((2 * D_MODEL, D_MODEL, 1))
        tensors[layer + "linear2.weight"] = np.zeros((D_MODEL, 2 * D_MODEL, 1))
        tensors[layer + "norm1.weight"] = np.zeros(D_MODEL)
        tensors[layer + "norm1.bias"] = np.zeros(D_MODEL)
        tensors[layer + "norm2.weight"] = np.zeros(D_MODEL)
        tensors[layer + "norm2.bias"] = np.zeros(D_MODEL)
        tensors[prefix + "input_proj.weight"] = np.zeros((D_MODEL, transformer_in, 1))
        tensors[prefix + "output_proj.weight"] = np.zeros((transformer_in, D_MODEL, 1))
    write_safetensors(model_dir / "model.safetensors", tensors)

    def transformer(dimension: int) -> dict:
        return {
            "module_type": "Transformer",
            "input_dimension": dimension,
            "output_dimension": dimension,
            "d_model": D_MODEL,
            "num_heads": 2,
            "num_layers": 1,
            "max_period": 10000.0,
        }

    patch = {"module_type": "PatchedPretransform", "patch_size": OUT_DIM}
    config = {
        "sampling_rate": 24000,
        "downsample_rate": OUT_DIM,
        "quantizer_kwargs": {
            "quantizer_type": "rlfq",
            "input_dim": RVQ_DIM,
            "rvq_dim": RVQ_DIM,
            "output_dim": OUT_DIM,
            "num_quantizers": N_VQ,
            "codebook_size": CODE_SIZE,
            "codebook_dim": CODE_DIM,
        },
        "encoder_kwargs": [patch, transformer(RVQ_DIM)],
        "decoder_kwargs": [transformer(OUT_DIM), patch],
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    return model_dir


def sfx_text_encoder_tensors() -> dict[str, np.ndarray]:
    kv_dim = SFX_HEAD_DIM
    q_dim = 2 * SFX_HEAD_DIM
    layer = "model.layers.0."
    return {
        "model.embed_tokens.weight": np.ones((TEXT_VOCAB, SFX_WIDTH)),
        "model.norm.weight": np.ones(SFX_WIDTH),
        "lm_head.weight": np.ones((TEXT_VOCAB, SFX_WIDTH)),
        layer + "input_layernorm.weight": np.ones(SFX_WIDTH),
        layer + "self_attn.q_proj.weight": np.ones((q_dim, SFX_WIDTH)),
        layer + "self_attn.k_proj.weight": np.ones((kv_dim, SFX_WIDTH)),
        layer + "self_attn.v_proj.weight": np.ones((kv_dim, SFX_WIDTH)),
        layer + "self_attn.o_proj.weight": np.ones((SFX_WIDTH, q_dim)),
        layer + "self_attn.q_norm.weight": np.ones(SFX_HEAD_DIM),
        layer + "self_attn.k_norm.weight": np.ones(SFX_HEAD_DIM),
        layer + "post_attention_layernorm.weight": np.ones(SFX_WIDTH),
        layer + "mlp.gate_proj.weight": np.ones((SFX_FF, SFX_WIDTH)),
        layer + "mlp.up_proj.weight": np.ones((SFX_FF, SFX_WIDTH)),
        layer + "mlp.down_proj.weight": np.ones((SFX_WIDTH, SFX_FF)),
    }


def sfx_dit_tensors() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        "patch_embedding.weight": np.ones((SFX_WIDTH, SFX_LATENT, 1)),
        "patch_embedding.bias": np.ones(SFX_WIDTH),
        "scale_shift_table": np.ones((1, 2, SFX_WIDTH)),
        "proj_out.weight": np.ones((SFX_LATENT, SFX_WIDTH)),
        "proj_out.bias": np.ones(SFX_LATENT),
    }
    for linear, (rows, cols) in {
        "condition_embedder.text_embedder.linear_1": (SFX_WIDTH, SFX_WIDTH),
        "condition_embedder.text_embedder.linear_2": (SFX_WIDTH, SFX_WIDTH),
        "condition_embedder.time_embedder.linear_1": (SFX_WIDTH, SFX_WIDTH),
        "condition_embedder.time_embedder.linear_2": (SFX_WIDTH, SFX_WIDTH),
        "condition_embedder.time_proj": (6 * SFX_WIDTH, SFX_WIDTH),
    }.items():
        tensors[linear + ".weight"] = np.ones((rows, cols))
        tensors[linear + ".bias"] = np.ones(rows)
    block = "blocks.0."
    for attn in ("attn1", "attn2"):
        for proj in ("to_q", "to_k", "to_v", "to_out.0"):
            tensors[f"{block}{attn}.{proj}.weight"] = np.ones((SFX_WIDTH, SFX_WIDTH))
            tensors[f"{block}{attn}.{proj}.bias"] = np.ones(SFX_WIDTH)
        tensors[f"{block}{attn}.norm_q.weight"] = np.ones(SFX_WIDTH)
        tensors[f"{block}{attn}.norm_k.weight"] = np.ones(SFX_WIDTH)
    tensors[block + "norm2.weight"] = np.ones(SFX_WIDTH)
    tensors[block + "norm2.bias"] = np.ones(SFX_WIDTH)
    tensors[block + "ffn.net.0.proj.weight"] = np.ones((SFX_FF, SFX_WIDTH))
    tensors[block + "ffn.net.0.proj.bias"] = np.ones(SFX_FF)
    tensors[block + "ffn.net.2.weight"] = np.ones((SFX_WIDTH, SFX_FF))
    tensors[block + "ffn.net.2.bias"] = np.ones(SFX_WIDTH)
    tensors[block + "scale_shift_table"] = np.ones((1, 6, SFX_WIDTH))
    return tensors


def add_weight_norm(tensors: dict[str, np.ndarray], prefix: str, shape: tuple[int, ...]) -> None:
    direction = np.arange(1, int(np.prod(shape)) + 1, dtype=np.float64).reshape(shape)
    norm = np.sqrt((direction ** 2).sum(axis=tuple(range(1, len(shape))), keepdims=True))
    tensors[prefix + ".weight_v"] = direction
    tensors[prefix + ".weight_g"] = SFX_GAIN * norm


def add_residual_units(tensors: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    for unit in range(3):
        unit_prefix = f"{prefix}.{unit + 2}.block"
        tensors[unit_prefix + ".0.alpha"] = np.ones((1, channels, 1))
        add_weight_norm(tensors, unit_prefix + ".1", (channels, channels, SFX_KERNEL))
        tensors[unit_prefix + ".1.bias"] = np.ones(channels)
        tensors[unit_prefix + ".2.alpha"] = np.ones((1, channels, 1))
        add_weight_norm(tensors, unit_prefix + ".3", (channels, channels, 1))
        tensors[unit_prefix + ".3.bias"] = np.ones(channels)


def sfx_vae_tensors() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        "post_quant_conv.weight": np.ones((SFX_LATENT, SFX_LATENT, 1)),
        "post_quant_conv.bias": np.ones(SFX_LATENT),
    }
    add_weight_norm(tensors, "decoder.model.0", (SFX_DECODER_DIM, SFX_LATENT, SFX_KERNEL))
    tensors["decoder.model.0.bias"] = np.ones(SFX_DECODER_DIM)
    channels = SFX_DECODER_DIM
    for block, rate in enumerate(SFX_RATES):
        prefix = f"decoder.model.{block + 1}.block"
        tensors[prefix + ".0.alpha"] = np.full((1, channels, 1), 0.5)
        add_weight_norm(tensors, prefix + ".1", (channels, channels // 2, 2 * rate))
        tensors[prefix + ".1.bias"] = np.ones(channels // 2)
        add_residual_units(tensors, prefix, channels // 2)
        channels //= 2
    last = len(SFX_RATES) + 1
    tensors[f"decoder.model.{last}.alpha"] = np.ones((1, channels, 1))
    add_weight_norm(tensors, f"decoder.model.{last + 1}", (1, channels, SFX_KERNEL))
    tensors[f"decoder.model.{last + 1}.bias"] = np.ones(1)
    return tensors


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def make_sfx_checkpoint(root: Path) -> Path:
    model_dir = root / "sfx"
    for sub in ("text_encoder", "transformer", "vae", "tokenizer", "scheduler"):
        (model_dir / sub).mkdir(parents=True)
    write_safetensors(model_dir / "text_encoder" / "model.safetensors", sfx_text_encoder_tensors())
    write_safetensors(model_dir / "transformer" / "diffusion_pytorch_model.safetensors", sfx_dit_tensors())
    write_safetensors(model_dir / "vae" / "vae.safetensors", sfx_vae_tensors())
    write_json(model_dir / "text_encoder" / "config.json", {
        "num_hidden_layers": 1, "hidden_size": SFX_WIDTH, "intermediate_size": SFX_FF,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": SFX_HEAD_DIM,
        "rope_theta": 1000000.0, "rms_norm_eps": 1e-6})
    write_json(model_dir / "transformer" / "config.json", {
        "dim": SFX_WIDTH, "in_dim": SFX_LATENT, "out_dim": SFX_LATENT, "ffn_dim": SFX_FF,
        "text_dim": SFX_WIDTH, "freq_dim": SFX_WIDTH, "eps": 1e-6, "patch_size": [1],
        "num_heads": 2, "num_layers": 1, "has_image_input": False, "vae_type": "dac"})
    write_json(model_dir / "vae" / "config.json", {
        "latent_dim": SFX_LATENT, "decoder_dim": SFX_DECODER_DIM, "decoder_rates": SFX_RATES,
        "sample_rate": 48000})
    write_json(model_dir / "model_index.json", {"max_inference_seconds": 30})
    write_json(model_dir / "scheduler" / "scheduler_config.json", {"shift": 5.0, "num_train_timesteps": 1000})
    vocab = {f"tok{i}": i for i in range(TEXT_VOCAB)}
    write_json(model_dir / "tokenizer" / "tokenizer.json", {
        "model": {"vocab": vocab, "merges": ["a b"]},
        "added_tokens": [{"id": 0, "content": "<|endoftext|>"}]})
    write_json(model_dir / "tokenizer" / "tokenizer_config.json", {"pad_token": "<|endoftext|>"})
    return model_dir


def reader_tensor(path: Path, name: str):
    reader = gguf.GGUFReader(str(path))
    return next(t for t in reader.tensors if t.name == name)


def reader_field(path: Path, key: str) -> list:
    field = gguf.GGUFReader(str(path)).fields[key]
    return [field.parts[index][0] for index in field.data]


def check_sfx(path: Path) -> None:
    names = reader_tensors(path)
    expected = 2 + 11 + 15 + 27 + 4 + 28 * len(SFX_RATES) + 4
    assert len(names) == expected, f"sfx tensor census: {len(names)} != {expected}"
    assert "text.blk.0.attn_q.weight" in names and "dit.blk.0.cross_norm.bias" in names
    assert not any(name.startswith("lm_head") for name in names), "the unused LM head is dropped"
    assert names["dit.patch_embd.weight"] == (SFX_LATENT, SFX_WIDTH), "patch conv becomes a linear"
    assert names["dit.blk.0.modulation"] == (SFX_WIDTH, 6), "modulation drops its batch axis"
    up_width = SFX_RATES[0] * 2 * (SFX_DECODER_DIM // 2)
    assert names["vae.blk.0.up.weight"] == (SFX_DECODER_DIM, up_width), "transposed conv becomes GEMM columns"
    conv_in = reader_tensor(path, "vae.conv_in.weight")
    assert conv_in.tensor_type == gguf.GGMLQuantizationType.F16, "conv kernels stay F16 for im2col"
    direction = np.arange(1, SFX_DECODER_DIM * SFX_LATENT * SFX_KERNEL + 1, dtype=np.float32)
    assert np.allclose(np.asarray(conv_in.data, dtype=np.float32).reshape(-1), SFX_GAIN * direction,
                       rtol=1e-3), "weight norm folds to g * v / |v|"
    inv = np.asarray(reader_tensor(path, "vae.blk.0.snake.inv").data, dtype=np.float32).reshape(-1)
    assert np.allclose(inv, 2.0, rtol=1e-6), "snake stores 1 / (alpha + eps)"
    assert list(reader_field(path, "moss-sfx.vae.decoder_rates")) == SFX_RATES
    assert int(reader_field(path, "tokenizer.ggml.padding_token_id")[0]) == 0
    print("sound effect converter: PASS")


def check_sfx_half(path: Path) -> None:
    tensor = reader_tensor(path, "dit.blk.0.self_q.weight")
    assert tensor.tensor_type == gguf.GGMLQuantizationType.F16, "f16 stores linear weights as f16"
    assert reader_tensor(path, "text.output_norm.weight").tensor_type == gguf.GGMLQuantizationType.F32
    print("sound effect f16 converter: PASS")


def move_vae_to_checkpoint(model_dir: Path) -> bool:
    try:
        import torch
    except ImportError:
        print("SKIP: torch is not installed; the .pth VAE path is untested")
        return False
    vae_dir = model_dir / "vae"
    state = {name: torch.from_numpy(np.asarray(tensor, dtype=np.float32)) for name, tensor in sfx_vae_tensors().items()}
    kwargs = {"latent_dim": SFX_LATENT, "decoder_dim": SFX_DECODER_DIM, "decoder_rates": SFX_RATES,
              "sample_rate": 48000, "continuous": True}
    torch.save({"state_dict": state, "metadata": {"kwargs": kwargs}}, vae_dir / "vae.pth")
    (vae_dir / "vae.safetensors").unlink()
    write_json(vae_dir / "config.json", {"latent_dim": SFX_LATENT, "sample_rate": 48000})
    return True


def check_sfx_bfloat(path: Path) -> None:
    tensor = reader_tensor(path, "dit.blk.0.self_q.weight")
    assert tensor.tensor_type == gguf.GGMLQuantizationType.BF16, "bf16 stores linear weights as bf16"
    print("sound effect bf16 converter: PASS")


def expect_converter_failure(args: list[str], needle: str, label: str) -> None:
    result = subprocess.run([sys.executable, str(SCRIPTS / "convert-moss-sfx-to-gguf.py"), *args],
                            capture_output=True, text=True)
    assert result.returncode != 0, f"{label}: converter accepted the checkpoint"
    assert needle in result.stderr, f"{label}: wrong failure: {result.stderr[-400:]}"
    print(f"sound effect converter rejects {label}: PASS")


def add_unmapped_dit_tensor(model_dir: Path) -> None:
    tensors = sfx_dit_tensors()
    tensors["blocks.0.unknown.weight"] = np.ones(SFX_WIDTH)
    write_safetensors(model_dir / "transformer" / "diffusion_pytorch_model.safetensors", tensors)


def restore_dit_tensors(model_dir: Path) -> None:
    write_safetensors(model_dir / "transformer" / "diffusion_pytorch_model.safetensors", sfx_dit_tensors())


def write_discrete_vae_checkpoint(model_dir: Path) -> None:
    import torch

    state = {name: torch.from_numpy(np.asarray(tensor, dtype=np.float32)) for name, tensor in sfx_vae_tensors().items()}
    torch.save({"state_dict": state, "metadata": {"kwargs": {"decoder_rates": SFX_RATES, "continuous": False}}},
               model_dir / "vae" / "vae.pth")


def check_sfx_quantized(path: Path) -> None:
    tensor = reader_tensor(path, "dit.blk.0.self_q.weight")
    assert tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0, "q8_0 quantizes linear weights"
    assert reader_tensor(path, "dit.blk.0.self_q.bias").tensor_type == gguf.GGMLQuantizationType.F32
    print("sound effect q8_0 converter: PASS")


def reader_types(path: Path) -> dict[str, gguf.GGMLQuantizationType]:
    return {t.name: t.tensor_type for t in gguf.GGUFReader(str(path)).tensors}


def dequantized(path: Path, name: str) -> np.ndarray:
    tensor = reader_tensor(path, name)
    return gguf.quants.dequantize(tensor.data, tensor.tensor_type)


def check_sfx_matrix_types(path: Path, block_type: gguf.GGMLQuantizationType,
                           global_type: gguf.GGMLQuantizationType) -> None:
    types = reader_types(path)
    for name in SFX_BLOCK_MATRICES:
        assert types[name] == block_type, f"{name} is {types[name].name}, not {block_type.name}"
    for name in SFX_GLOBAL_MATRICES:
        assert types[name] == global_type, f"{name} is {types[name].name}, not {global_type.name}"
    assert types["dit.blk.0.self_q.bias"] == gguf.GGMLQuantizationType.F32, "biases stay f32"
    assert types["dit.patch_embd.weight"] == gguf.GGMLQuantizationType.F32, "the 1x1 patch projection stays f32"
    assert types["vae.conv_in.weight"] == gguf.GGMLQuantizationType.F16, "conv kernels stay f16"


def check_sfx_four_bit(path: Path) -> None:
    check_sfx(path)
    q4 = gguf.GGMLQuantizationType.Q4_0
    check_sfx_matrix_types(path, q4, q4)
    assert np.allclose(dequantized(path, "dit.blk.0.self_q.weight"), 1.0), "q4_0 blocks dequantize to the weights"
    print("sound effect q4_0 converter: PASS")


def check_sfx_four_bit_dit_blocks(path: Path) -> None:
    types = reader_types(path)
    q8, q4 = gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0
    for name in SFX_BLOCK_MATRICES:
        expected = q4 if name.startswith("dit.blk.") else q8
        assert types[name] == expected, f"{name} is {types[name].name}, not {expected.name}"
    for name in SFX_GLOBAL_MATRICES:
        assert types[name] == q8, f"{name} is {types[name].name}, not Q8_0"
    print("sound effect q4_0 DiT blocks over q8_0 converter: PASS")


def check_sfx_four_bit_text_blocks(path: Path) -> None:
    check_sfx_matrix_types(path, gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q8_0)
    print("sound effect q4_0 text and DiT blocks over q8_0 converter: PASS")


def check_sfx_default_name(sfx_dir: Path) -> None:
    run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "q8_0", "--dit-block-outtype", "q4_0"])
    named = sfx_dir / f"{sfx_dir.name}-q8_0-dit-q4_0.gguf"
    assert named.exists(), "the default name records the block override"
    named.unlink()
    print("sound effect converter default name: PASS")


def run_converter(script: str, args: list[str]) -> None:
    result = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"{script} failed:\n{result.stdout}\n{result.stderr}")


def reader_tensors(path: Path) -> dict[str, tuple[int, ...]]:
    reader = gguf.GGUFReader(str(path))
    return {t.name: tuple(int(d) for d in t.shape) for t in reader.tensors}


def check_backbone(path: Path) -> None:
    names = reader_tensors(path)
    expected = 3 + 2 * N_VQ + 11
    assert len(names) == expected, f"backbone tensor census: {len(names)} != {expected}"
    for required in ("token_embd.weight", "output.weight", "output_norm.weight",
                     "token_embd_audio.0.weight", "output_audio.1.weight",
                     "blk.0.attn_q.weight", "blk.0.ffn_down.weight"):
        assert required in names, f"backbone is missing {required}"
    print("backbone converter: PASS")


def check_codec(encoder: Path, decoder: Path) -> None:
    for path, section in ((encoder, "encoder"), (decoder, "decoder")):
        names = reader_tensors(path)
        for required in ("quantizer.quantizers.0.codebook.weight",
                         "blk.0.layer.0.attn_qkv.weight",
                         "blk.0.layer.0.ffn_up.weight"):
            assert required in names, f"{section} is missing {required}"
    print("codec converter: PASS")


def speech_layer(prefix: str) -> dict[str, np.ndarray]:
    q_dim = SPEECH_HEADS * SPEECH_HEAD_DIM
    kv_dim = SPEECH_KV_HEADS * SPEECH_HEAD_DIM
    return {
        prefix + "input_layernorm.weight": np.ones(SPEECH_EMBD),
        prefix + "self_attn.q_proj.weight": np.ones((q_dim, SPEECH_EMBD)),
        prefix + "self_attn.k_proj.weight": np.ones((kv_dim, SPEECH_EMBD)),
        prefix + "self_attn.v_proj.weight": np.ones((kv_dim, SPEECH_EMBD)),
        prefix + "self_attn.o_proj.weight": np.ones((SPEECH_EMBD, q_dim)),
        prefix + "self_attn.q_norm.weight": np.ones(SPEECH_HEAD_DIM),
        prefix + "self_attn.k_norm.weight": np.ones(SPEECH_HEAD_DIM),
        prefix + "post_attention_layernorm.weight": np.ones(SPEECH_EMBD),
        prefix + "mlp.gate_proj.weight": np.ones((SPEECH_FF, SPEECH_EMBD)),
        prefix + "mlp.up_proj.weight": np.ones((SPEECH_FF, SPEECH_EMBD)),
        prefix + "mlp.down_proj.weight": np.ones((SPEECH_EMBD, SPEECH_FF)),
    }


def speech_tensors() -> dict[str, np.ndarray]:
    tensors = {
        "model.embed_tokens.weight": np.ones((SPEECH_TEXT_VOCAB, SPEECH_EMBD)),
        "model.audio_embed.weight": np.ones((SPEECH_AUDIO_VOCAB, SPEECH_EMBD)),
        "model.text_norm.weight": np.ones(SPEECH_EMBD),
        "model.audio_norm.weight": np.ones(SPEECH_EMBD),
        "text_lm_head.weight": np.ones((SPEECH_TEXT_VOCAB, SPEECH_EMBD)),
        "audio_lm_head.weight": np.ones((SPEECH_AUDIO_VOCAB, SPEECH_EMBD)),
    }
    for block in ("shared_block", "text_block", "audio_block"):
        tensors.update(speech_layer(f"model.{block}.layers.0."))
    return tensors


def speech_config() -> dict:
    return {
        "channels": 2,
        "num_hidden_layers": 2,
        "num_shared_layers": 1,
        "num_modality_layers": 1,
        "hidden_size": SPEECH_EMBD,
        "intermediate_size": SPEECH_FF,
        "num_attention_heads": SPEECH_HEADS,
        "num_key_value_heads": SPEECH_KV_HEADS,
        "head_dim": SPEECH_HEAD_DIM,
        "max_position_embeddings": 512,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": SPEECH_TEXT_VOCAB,
        "audio_vocab_size": SPEECH_AUDIO_VOCAB,
        "modality_pad_token_id": SPEECH_PLACEHOLDER,
        "audio_pad_token_id": SPEECH_AUDIO_PAD,
        "sosp_token_id": SPEECH_START,
        "eosp_token_id": SPEECH_END,
    }


def make_speech_checkpoint(root: Path) -> Path:
    model_dir = root / "moss-speech"
    model_dir.mkdir()
    write_safetensors(model_dir / "model.safetensors", speech_tensors())
    write_json(model_dir / "config.json", speech_config())
    vocab = {f"tok{i}": i for i in range(3, SPEECH_TEXT_VOCAB)}
    tokenizer = {
        "model": {"vocab": vocab, "merges": [["t", "o"]]},
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>", "special": True},
            {"id": 1, "content": "<|im_start|>", "special": True},
            {"id": 2, "content": "<|im_end|>", "special": True},
            {"id": 3, "content": "<|empty|>", "special": False},
        ],
    }
    write_json(model_dir / "tokenizer.json", tokenizer)
    return model_dir


def check_speech(path: Path, matrix_type: gguf.GGMLQuantizationType) -> None:
    names = reader_tensors(path)
    assert len(names) == SPEECH_TENSORS, f"speech tensor census: {len(names)} != {SPEECH_TENSORS}"
    for required in ("text.token_embd.weight", "audio.output.weight", "blk.0.attn_q_norm.weight",
                     "text.blk.0.ffn_down.weight", "audio.blk.0.attn_output.weight"):
        assert required in names, f"speech model is missing {required}"
    assert reader_tensor(path, "audio.blk.0.ffn_gate.weight").tensor_type == matrix_type
    assert reader_tensor(path, "text.output_norm.weight").tensor_type == gguf.GGMLQuantizationType.F32
    assert int(reader_field(path, "moss-speech.block_count")[0]) == 1
    assert int(reader_field(path, "moss-speech.modality_block_count")[0]) == 1
    assert int(reader_field(path, "moss-speech.audio_vocab_size")[0]) == SPEECH_AUDIO_VOCAB
    assert int(reader_field(path, "moss-speech.token.speech_end")[0]) == SPEECH_END
    assert int(reader_field(path, "moss-speech.token.text_placeholder")[0]) == SPEECH_PLACEHOLDER
    assert int(reader_field(path, "moss-speech.token.im_end")[0]) == 2
    assert int(reader_field(path, "moss-speech.token.pad")[0]) == 0
    types = list(reader_field(path, "tokenizer.ggml.token_type"))
    assert types[0] == gguf.TokenType.CONTROL and types[3] == gguf.TokenType.USER_DEFINED
    assert types[4] == gguf.TokenType.NORMAL, "special tokens are CONTROL, plain added tokens USER_DEFINED"
    print(f"speech converter ({matrix_type.name}): PASS")


def expect_speech_failure(model_dir: Path, needle: str, label: str) -> None:
    result = subprocess.run([sys.executable, str(SCRIPTS / "convert-moss-speech-to-gguf.py"), str(model_dir),
                             "--outfile", str(model_dir.parent / "bad-speech.gguf")], capture_output=True, text=True)
    assert result.returncode != 0, f"{label}: converter accepted the checkpoint"
    assert needle in result.stderr, f"{label}: wrong failure: {result.stderr[-400:]}"
    print(f"speech converter rejects {label}: PASS")


def check_speech_rejections(model_dir: Path) -> None:
    config = speech_config()
    write_json(model_dir / "config.json", {**config, "num_hidden_layers": 3})
    expect_speech_failure(model_dir, "num_hidden_layers", "a layer split that does not add up")
    write_json(model_dir / "config.json", config)
    write_safetensors(model_dir / "model.safetensors", {**speech_tensors(), "model.extra.weight": np.ones(2)})
    expect_speech_failure(model_dir, "unmapped tensor", "an unmapped tensor")
    tensors = speech_tensors()
    del tensors["audio_lm_head.weight"]
    write_safetensors(model_dir / "model.safetensors", tensors)
    expect_speech_failure(model_dir, "architecture needs", "a missing output head")
    write_safetensors(model_dir / "model.safetensors", speech_tensors())


def load_codec_converter():
    try:
        import safetensors.torch  # noqa: F401
        import soundfile  # noqa: F401
        import torch  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError as error:
        print(f"SKIP: {error.name} is not installed; the speech codec helpers are untested")
        return None
    spec = importlib.util.spec_from_file_location("_speech_codec", SCRIPTS / "convert-moss-speech-codec-to-gguf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_gguf(path: Path, fill) -> None:
    writer = gguf.GGUFWriter(str(path), "moss-speech-codec")
    try:
        fill(writer)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def vq_layer(prefix: str) -> dict:
    import torch

    tensors = {prefix + "self_attn.k_proj.weight": torch.ones(VQ_EMBD, VQ_EMBD),
               prefix + "self_attn.q_proj.bias": torch.ones(VQ_EMBD),
               prefix + "self_attn_layer_norm.weight": torch.ones(VQ_EMBD),
               prefix + "self_attn.rotary.inv_freq": torch.ones(VQ_EMBD)}
    return tensors


def write_vq_checkpoint(codec_dir: Path) -> None:
    import torch
    from safetensors.torch import save_file

    tensors = {"encoder.conv1.weight": torch.ones(VQ_EMBD, VQ_MELS, VQ_KERNEL),
               "encoder.conv1.bias": torch.ones(VQ_EMBD),
               "encoder.embed_positions.weight": torch.ones(4, VQ_EMBD),
               "encoder.codebook.weight": torch.ones(VQ_CODES, VQ_EMBD),
               "decoder.layers.0.fc1.weight": torch.ones(2, 2)}
    tensors.update(vq_layer("encoder.layers.0."))
    tensors.update(vq_layer("encoder.layers.1."))
    save_file(tensors, str(codec_dir / "model.safetensors"))


def check_vq_encoder(codec, root: Path, quant: str, matrix_type: gguf.GGMLQuantizationType) -> None:
    codec_dir = root / "speech-codec"
    out = root / f"vq-{quant}.gguf"
    write_gguf(out, lambda writer: codec.emit_vq_encoder(writer, codec_dir, {"quantize_position": 1}, quant))
    names = reader_tensors(out)
    assert set(names) == {"whispervq.conv1.weight", "whispervq.conv1.bias", "whispervq.pos_embd",
                          "whispervq.codebook", "whispervq.blk.0.attn_k.weight", "whispervq.blk.0.attn_q.bias",
                          "whispervq.blk.0.attn_norm.weight"}, f"vq census: {sorted(names)}"
    assert names["whispervq.conv1.bias"] == (1, VQ_EMBD), "conv biases become columns"
    assert reader_tensor(out, "whispervq.conv1.weight").tensor_type == gguf.GGMLQuantizationType.F16
    assert reader_tensor(out, "whispervq.blk.0.attn_k.weight").tensor_type == matrix_type
    assert reader_tensor(out, "whispervq.codebook").tensor_type == gguf.GGMLQuantizationType.F32
    print(f"speech codec VQ encoder ({quant}): PASS")


def vq_config(**overrides) -> dict:
    config = {"encoder_causal_attention": True, "encoder_causal_convolution": True, "pooling_position": 16,
              "quantize_position": 16, "pooling_type": "avg"}
    config.update(overrides)
    return config


def expect_vq_metadata_failure(codec, config: dict, needle: str, label: str) -> None:
    try:
        codec.add_vq_metadata(None, config, {})
    except ValueError as error:
        assert needle in str(error), f"{label}: wrong failure: {error}"
        print(f"speech codec rejects {label}: PASS")
        return
    raise AssertionError(f"{label}: accepted")


def check_vq_metadata(codec) -> None:
    expect_vq_metadata_failure(codec, vq_config(encoder_causal_attention=False), "causal", "a bidirectional encoder")
    expect_vq_metadata_failure(codec, vq_config(pooling_type="max"), "average pool", "a max pool")
    expect_vq_metadata_failure(codec, vq_config(pooling_position=8), "average pool", "pooling before the quantizer")


def write_campplus(path: Path) -> None:
    def fill(writer: gguf.GGUFWriter) -> None:
        writer.add_uint32("campplus.embed_dim", 192)
        writer.add_float32("campplus.scale", 1.0)
        writer.add_tensor("campplus/head.weight", np.arange(6, dtype=np.float32).reshape(2, 3))
        writer.add_tensor("other/weight", np.ones(2, dtype=np.float32))

    write_gguf(path, fill)


def check_campplus_and_voice(codec, root: Path) -> None:
    import soundfile

    campplus = root / "campplus.gguf"
    write_campplus(campplus)
    voice = root / "voice.wav"
    soundfile.write(str(voice), np.stack([np.full(40, 0.5), np.full(40, -0.1)], axis=1), 22050, subtype="FLOAT")
    out = root / "codec-extras.gguf"

    def fill(writer: gguf.GGUFWriter) -> None:
        codec.copy_campplus(writer, campplus)
        codec.emit_default_voice(writer, voice)

    write_gguf(out, fill)
    names = reader_tensors(out)
    assert set(names) == {"campplus/head.weight", "voice.default_audio"}, f"extras census: {sorted(names)}"
    head = np.asarray(reader_tensor(out, "campplus/head.weight").data, dtype=np.float32).reshape(-1)
    assert np.array_equal(head, np.arange(6, dtype=np.float32)), "CAM++ weights are copied verbatim"
    assert int(reader_field(out, "campplus.embed_dim")[0]) == 192
    assert "campplus.scale" not in gguf.GGUFReader(str(out)).fields, "only uint32 CAM++ fields are copied"
    audio = np.asarray(reader_tensor(out, "voice.default_audio").data, dtype=np.float32)
    assert audio.shape == (40,) and np.allclose(audio, 0.2), "the default voice is downmixed to mono"
    assert int(reader_field(out, "moss-speech-codec.voice.sample_rate")[0]) == 22050
    print("speech codec CAM++ copy and default voice: PASS")


def check_speech_codec(root: Path) -> None:
    codec = load_codec_converter()
    if codec is None:
        return
    (root / "speech-codec").mkdir()
    write_vq_checkpoint(root / "speech-codec")
    check_vq_encoder(codec, root, "f16", gguf.GGMLQuantizationType.F16)
    check_vq_encoder(codec, root, "f32", gguf.GGMLQuantizationType.F32)
    check_vq_metadata(codec)
    check_campplus_and_voice(codec, root)


def check_speech_converters(root: Path) -> None:
    speech_dir = make_speech_checkpoint(root)
    for outtype, matrix_type in (("f32", gguf.GGMLQuantizationType.F32), ("f16", gguf.GGMLQuantizationType.F16),
                                 ("bf16", gguf.GGMLQuantizationType.BF16), ("q8_0", gguf.GGMLQuantizationType.Q8_0)):
        speech_out = root / f"speech-{outtype}.gguf"
        run_converter("convert-moss-speech-to-gguf.py",
                      [str(speech_dir), "--outtype", outtype, "--outfile", str(speech_out)])
        check_speech(speech_out, matrix_type)
    check_speech_rejections(speech_dir)
    check_speech_codec(root)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        backbone_dir = make_backbone_checkpoint(root)
        backbone_out = root / "backbone.gguf"
        run_converter("convert-moss-delay-to-gguf.py",
                      [str(backbone_dir), "--outtype", "f32", "--outfile", str(backbone_out)])
        check_backbone(backbone_out)

        codec_dir = make_codec_checkpoint(root)
        encoder_out = root / "encoder.gguf"
        decoder_out = root / "decoder.gguf"
        run_converter("convert-moss-codec-to-gguf.py",
                      [str(codec_dir), "--outtype", "f32",
                       "--encoder-outfile", str(encoder_out),
                       "--decoder-outfile", str(decoder_out)])
        check_codec(encoder_out, decoder_out)

        sfx_dir = make_sfx_checkpoint(root)
        sfx_out = root / "sfx.gguf"
        run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "f32", "--outfile", str(sfx_out)])
        check_sfx(sfx_out)
        sfx_q8 = root / "sfx-q8.gguf"
        run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "q8_0", "--outfile", str(sfx_q8)])
        check_sfx_quantized(sfx_q8)
        sfx_q4 = root / "sfx-q4.gguf"
        run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "q4_0", "--outfile", str(sfx_q4)])
        check_sfx_four_bit(sfx_q4)
        sfx_dit_q4 = root / "sfx-q8-dit-q4.gguf"
        run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "q8_0", "--dit-block-outtype", "q4_0",
                                                      "--outfile", str(sfx_dit_q4)])
        check_sfx_four_bit_dit_blocks(sfx_dit_q4)
        sfx_blk_q4 = root / "sfx-q8-blk-q4.gguf"
        run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "q8_0", "--text-block-outtype", "q4_0",
                                                      "--dit-block-outtype", "q4_0", "--outfile", str(sfx_blk_q4)])
        check_sfx_four_bit_text_blocks(sfx_blk_q4)
        check_sfx_default_name(sfx_dir)
        sfx_f16 = root / "sfx-f16.gguf"
        run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "f16", "--outfile", str(sfx_f16)])
        check_sfx_half(sfx_f16)
        sfx_bf16 = root / "sfx-bf16.gguf"
        run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "bf16", "--outfile", str(sfx_bf16)])
        check_sfx_bfloat(sfx_bf16)
        add_unmapped_dit_tensor(sfx_dir)
        expect_converter_failure([str(sfx_dir), "--outfile", str(root / "bad.gguf")], "unmapped DiT tensor",
                                 "an unmapped DiT tensor")
        restore_dit_tensors(sfx_dir)
        if move_vae_to_checkpoint(sfx_dir):
            sfx_pth = root / "sfx-pth.gguf"
            run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "f32", "--outfile", str(sfx_pth)])
            check_sfx(sfx_pth)
            print("sound effect .pth VAE converter: PASS")
            write_discrete_vae_checkpoint(sfx_dir)
            expect_converter_failure([str(sfx_dir), "--outfile", str(root / "bad.gguf")], "continuous DAC",
                                     "a quantized DAC checkpoint")
        check_speech_converters(root)
    print("moss converters: OK")


if __name__ == "__main__":
    main()
