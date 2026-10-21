#!/usr/bin/env python3
"""Export Nemotron 3 Diarization from GGUF to fixed-capacity Core ML diarizers.

Each package holds the network after mel stacking (feature projection, RoPE
encoder, subpixel speaker head and sigmoid) at one capacity in 80 ms encoder
frames, speaker cache plus chunk. The runtime picks the smallest capacity that
fits a call and falls back to ggml otherwise; see docs/backends.md. Sigmoids
are written as 1 / (1 + exp(-x)) because the Neural Engine's native sigmoid
is a coarse approximation.

Example:

  python scripts/export-nemotron-diarization-coreml.py \\
      --gguf models/Nemotron-3-Diarization.q8_0.gguf \\
      --frames 358 745 \\
      --compile-dir models
"""
import argparse
import math
import re
import shutil
from pathlib import Path

import coremltools as ct
import gguf
import numpy as np
import torch
import torch.nn.functional as F

LAYER_NORM_EPSILON = 1.0e-5
MASKED_SCORE = -1.0e4
CONV_PADDING = (0, 1)
FEATURE_INPUT = "features"
CACHE_INPUT = "cache"
MASK_INPUT = "valid_mask"
PROBABILITY_OUTPUT = "probabilities"
EMBEDDING_OUTPUT = "embeddings"
ROTATION_KEY = "rope.rotation"
SIDECAR_INFIX = "-diarizer-"
QUANT_SUFFIX = re.compile(r"[.-](f32|f16|bf16|q8_0|q5_0|q4_0|q6_k|q5_k|q4_k|mxfp4|nvfp4)$")
EXPECTED_CONTRACT = {
    "general.architecture": "sortformer",
    "sortformer.version": "v3",
    "sortformer.encoder.type": "transformer_rope",
    "sortformer.encoder.subsampling_type": "feature_stacking",
}
FLOAT_TYPES = {
    gguf.GGMLQuantizationType.F32: np.float32,
    gguf.GGMLQuantizationType.F16: np.float16,
}


def tensor_to_torch(tensor):
    shape = tuple(reversed([int(dim) for dim in tensor.shape]))
    raw = np.asarray(tensor.data)
    if tensor.tensor_type in FLOAT_TYPES:
        values = raw.view(FLOAT_TYPES[tensor.tensor_type])
    else:
        values = gguf.quants.dequantize(raw, tensor.tensor_type)
    values = np.array(values, dtype=np.float32).reshape(shape)
    return torch.from_numpy(values)


def field_value(field):
    if field.types[0] == gguf.GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode()
    if field.types[0] == gguf.GGUFValueType.ARRAY:
        return None
    return field.parts[field.data[0]][0].item()


def load_gguf(path):
    reader = gguf.GGUFReader(str(path))
    tensors = {tensor.name: tensor_to_torch(tensor) for tensor in reader.tensors}
    meta = {field.name: field_value(field) for field in reader.fields.values() if field.types}
    return tensors, meta


def validate_contract(meta):
    for key, expected in EXPECTED_CONTRACT.items():
        if meta.get(key) != expected:
            raise ValueError(f"unsupported Nemotron 3 Diarization GGUF: {key}={meta.get(key)!r}, "
                             f"expected {expected!r}")
    if float(meta.get("sortformer.encoder.rotary_fraction", 0.0)) != 1.0:
        raise ValueError("Nemotron 3 Diarization export requires full-width rotary embeddings")


def model_config(meta):
    width = int(meta["sortformer.encoder.d_model"])
    heads = int(meta["sortformer.encoder.n_heads"])
    return {
        "width": width,
        "layers": int(meta["sortformer.encoder.n_layers"]),
        "heads": heads,
        "head_width": width // heads,
        "output_width": int(meta["sortformer.transformer.hidden_size"]),
        "speakers": int(meta["sortformer.num_speakers"]),
        "stack": int(meta["sortformer.encoder.subsampling_factor"]),
        "mels": int(meta["sortformer.preprocessor.features"]),
        "position_limit": int(meta["sortformer.encoder.pos_emb_max_len"]),
        "rope_base": float(meta["sortformer.encoder.rope_base"]),
    }


def sidecar_package(gguf_path, frames):
    stem = QUANT_SUFFIX.sub("", Path(gguf_path).stem)
    return Path(gguf_path).with_name(f"{stem}{SIDECAR_INFIX}{frames}.mlpackage")


def channel_table(values, frames):
    return values.transpose(0, 1).reshape(1, -1, 1, frames)


def rope_tables(frames, config):
    half = config["head_width"] // 2
    exponents = torch.arange(half, dtype=torch.float64) * 2.0 / config["head_width"]
    frequencies = torch.pow(torch.tensor(config["rope_base"], dtype=torch.float64), -exponents)
    angles = torch.arange(frames, dtype=torch.float64).unsqueeze(1) * frequencies
    repeats = 2 * config["heads"]
    cosine = torch.cos(angles).float().repeat(1, repeats)
    sine = torch.sin(angles).float().repeat(1, repeats)
    return channel_table(cosine, frames), channel_table(sine, frames)


def rotate_half_matrix(head_width):
    half = head_width // 2
    identity = torch.eye(half)
    zeros = torch.zeros(half, half)
    return torch.cat((torch.cat((zeros, -identity), dim=1), torch.cat((identity, zeros), dim=1)))


def scaled_qkv(qkv, config):
    width = config["width"]
    scale = 1.0 / math.sqrt(config["head_width"])
    return torch.cat((qkv[:width] * scale, qkv[width:]), dim=0)


def prepare_weights(tensors, config):
    prepared = dict(tensors)
    prepared[ROTATION_KEY] = rotate_half_matrix(config["head_width"]).repeat(config["heads"], 1)
    for index in range(config["layers"]):
        key = f"encoder.layers.{index}.attn.w_qkv.weight"
        prepared[key] = scaled_qkv(tensors[key], config)
    return prepared


def logistic(values):
    return 1.0 / (1.0 + torch.exp(-values))


def pointwise(values, weight, bias=None):
    return F.conv2d(values, weight.unsqueeze(-1).unsqueeze(-1), bias)


def layer_norm(values, weights, prefix):
    centered = values - values.mean(dim=1, keepdim=True)
    variance = (centered * centered).mean(dim=1, keepdim=True)
    normalized = centered * torch.rsqrt(variance + LAYER_NORM_EPSILON)
    return (normalized * weights[f"{prefix}.weight"].reshape(1, -1, 1, 1) +
            weights[f"{prefix}.bias"].reshape(1, -1, 1, 1))


def rotate(values, weights, config, rope):
    rotation = weights[ROTATION_KEY].unsqueeze(-1).unsqueeze(-1)
    rotated = F.conv2d(values, rotation, groups=config["heads"])
    return values * rope[0] + rotated * rope[1]


def head_attention(query, key_rows, value, key_mask):
    scores = torch.einsum("bchq,bkhc->bkhq", query, key_rows) + key_mask
    return torch.einsum("bkhq,bchk->bchq", torch.softmax(scores, dim=1), value)


def heads_attention(query, key, value, config, key_mask):
    key_rows = key.transpose(1, 3)
    width = config["head_width"]
    heads = []
    for head in range(config["heads"]):
        channels = slice(head * width, (head + 1) * width)
        heads.append(head_attention(query[:, channels], key_rows[..., channels],
                                    value[:, channels], key_mask))
    return torch.cat(heads, dim=1)


def attention(values, weights, prefix, config, rope, key_mask):
    width = config["width"]
    projected = pointwise(values, weights[f"{prefix}.w_qkv.weight"])
    query = rotate(projected[:, :width], weights, config, rope)
    key = rotate(projected[:, width:2 * width], weights, config, rope)
    attended = heads_attention(query, key, projected[:, 2 * width:], config, key_mask)
    return pointwise(attended, weights[f"{prefix}.out_proj.weight"], weights[f"{prefix}.out_proj.bias"])


def feed_forward(values, weights, prefix):
    hidden = F.gelu(pointwise(values, weights[f"{prefix}.net.0.weight"], weights[f"{prefix}.net.0.bias"]))
    return pointwise(hidden, weights[f"{prefix}.net.3.weight"], weights[f"{prefix}.net.3.bias"])


def encoder_layer(values, weights, index, config, rope, key_mask):
    prefix = f"encoder.layers.{index}"
    values = values + attention(layer_norm(values, weights, f"{prefix}.norm1"),
                                weights, f"{prefix}.attn", config, rope, key_mask)
    return values + feed_forward(layer_norm(values, weights, f"{prefix}.norm2"),
                                 weights, f"{prefix}.ffn")


def encoder_layers(values, weights, config, rope, key_mask):
    for index in range(config["layers"]):
        values = encoder_layer(values, weights, index, config, rope, key_mask)
    return values


def pixel_shuffle(convolved, config):
    frames = convolved.size(-1)
    width = config["output_width"]
    stacked = convolved.reshape(config["stack"], width, frames).permute(1, 2, 0)
    return stacked.reshape(1, width, 1, frames * config["stack"])


def speaker_head(encoded, weights, config, frame_mask):
    projected = pointwise(encoded, weights["encoder_proj.weight"], weights["encoder_proj.bias"])
    convolved = F.conv2d(projected * frame_mask, weights["subpixel_upsample.weight"].unsqueeze(2),
                         weights["subpixel_upsample.bias"], padding=CONV_PADDING)
    upsampled = pixel_shuffle(convolved, config)
    hidden = pointwise(F.relu(upsampled), weights["head.first_hidden_to_hidden.weight"],
                       weights["head.first_hidden_to_hidden.bias"])
    logits = pointwise(F.relu(hidden), weights["head.single_hidden_to_spks.weight"],
                       weights["head.single_hidden_to_spks.bias"])
    return logistic(logits)


def channels_first(rows):
    return rows.transpose(0, 1).unsqueeze(0).unsqueeze(2)


def rows_of(channels):
    return channels[0, :, 0].transpose(0, 1)


def diarize(features, cache, valid_mask, weights, config, rope):
    embeddings = pointwise(channels_first(features), weights["encoder.pre_encode.proj.weight"])
    sequence = layer_norm(channels_first(cache) + embeddings, weights, "encoder.embed_norm")
    key_mask = ((1.0 - valid_mask) * MASKED_SCORE).reshape(1, -1, 1, 1)
    encoded = encoder_layers(sequence, weights, config, rope, key_mask)
    encoded = layer_norm(encoded, weights, "encoder.final_norm")
    probabilities = speaker_head(encoded, weights, config, valid_mask.reshape(1, 1, 1, -1))
    return rows_of(probabilities), rows_of(embeddings)


class Weights(dict):
    def __missing__(self, key):
        raise KeyError(f"missing Nemotron 3 Diarization tensor {key}")


class NemotronDiarizer(torch.nn.Module):
    def __init__(self, tensors, config, frames):
        super().__init__()
        self.config = config
        self.names = {}
        for index, (key, value) in enumerate(prepare_weights(tensors, config).items()):
            self.names[key] = f"w_{index}"
            self.register_buffer(self.names[key], value.contiguous())
        cosine, sine = rope_tables(frames, config)
        self.register_buffer("rope_cos", cosine)
        self.register_buffer("rope_sin", sine)

    def weights(self):
        return Weights((key, getattr(self, name)) for key, name in self.names.items())

    def forward(self, features, cache, valid_mask):
        return diarize(features, cache, valid_mask, self.weights(), self.config,
                       (self.rope_cos, self.rope_sin))


def example_inputs(config, frames):
    return (torch.zeros(frames, config["mels"] * config["stack"]),
            torch.zeros(frames, config["width"]), torch.ones(frames))


def tensor_type(name, shape=None):
    return ct.TensorType(name=name, shape=shape, dtype=np.float32)


def convert(tensors, config, frames, precision):
    inputs = example_inputs(config, frames)
    model = NemotronDiarizer(tensors, config, frames).eval()
    traced = torch.jit.trace(model, inputs, check_trace=False)
    return ct.convert(
        traced,
        inputs=[tensor_type(FEATURE_INPUT, tuple(inputs[0].shape)),
                tensor_type(CACHE_INPUT, tuple(inputs[1].shape)),
                tensor_type(MASK_INPUT, tuple(inputs[2].shape))],
        outputs=[tensor_type(PROBABILITY_OUTPUT), tensor_type(EMBEDDING_OUTPUT)],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS13,
        skip_model_load=True,
    )


def compile_package(package, compile_dir):
    compiled = Path(compile_dir) / (Path(package).stem + ".mlmodelc")
    if compiled.exists():
        shutil.rmtree(compiled)
    ct.models.utils.compile_model(str(package), str(compiled))
    return compiled


def export_capacity(tensors, config, frames, args):
    if not 0 < frames <= config["position_limit"]:
        raise SystemExit(f"--frames must be in [1, {config['position_limit']}], got {frames}")
    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    package = Path(args.out_dir) / sidecar_package(args.gguf, frames).name if args.out_dir \
        else sidecar_package(args.gguf, frames)
    convert(tensors, config, frames, precision).save(str(package))
    print(f"[export] saved {package}")
    if args.compile_dir is not None:
        print(f"[export] compiled {compile_package(package, args.compile_dir)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True,
                        help="capacities in 80 ms encoder frames, speaker cache plus chunk")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="directory for the .mlpackage files (default: beside the GGUF)")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16",
                        help="fp32 is a parity reference; only fp16 runs on the Neural Engine")
    parser.add_argument("--compile-dir", type=Path, default=None,
                        help="compile each package into this directory (macOS only)")
    return parser.parse_args()


def main():
    args = parse_args()
    tensors, meta = load_gguf(args.gguf)
    validate_contract(meta)
    config = model_config(meta)
    for frames in args.frames:
        export_capacity(tensors, config, frames, args)


if __name__ == "__main__":
    main()
