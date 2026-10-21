#!/usr/bin/env python3
"""Export Sortformer v2.1 from GGUF to fixed-capacity Core ML diarizers.

Each package holds the network after subsampling (the 17 FastConformer blocks,
the encoder projection, the 18-layer speaker transformer and the sigmoid head)
at one capacity in 80 ms encoder frames. The input is the `pre_encode`
subsampling output, speaker cache and FIFO rows first for AOSC calls; the output
is the per-frame speaker `probabilities`. See docs/backends.md.

Two rewrites keep the graph on the Neural Engine and within fp16 accuracy.
Relative-position scores use the sinusoid angle-difference identity, with the
constant key-side table applied as a 1x1 convolution: a relative shift does not
compile for the Neural Engine, and an activation-by-activation contraction over
512 channels loses too much precision there. Sigmoids are 1 / (1 + exp(-x))
because the native Neural Engine sigmoid is coarse.

Example:

  python scripts/export-sortformer-diarizer-coreml.py \\
      --gguf models/diar_streaming_sortformer_4spk-v2.1.f16.gguf \\
      --frames 410 \\
      --compile-dir models
"""
import argparse
import importlib.util
import math
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

SCRIPTS = Path(__file__).resolve().parent
PRE_ENCODE_INPUT = "pre_encode"
MASK_INPUT = "valid_mask"
PROBABILITY_OUTPUT = "probabilities"
FEED_FORWARD_SCALE = 0.5
GLU_HALVES = 2
POSITION_BASE = 10000.0


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(SCRIPTS / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = load_script("parakeet_ref_encoder", "ref-encoder-from-gguf.py")
ENCODER_EXPORT = load_script("parakeet_export_encoder", "export-encoder-coreml.py")
NEMOTRON_EXPORT = load_script("parakeet_export_nemotron_diarizer", "export-nemotron-diarization-coreml.py")
pointwise = NEMOTRON_EXPORT.pointwise
logistic = NEMOTRON_EXPORT.logistic
layer_norm = NEMOTRON_EXPORT.layer_norm
head_attention = NEMOTRON_EXPORT.head_attention
channels_first = NEMOTRON_EXPORT.channels_first


def model_config(meta):
    width = int(meta["parakeet.encoder.d_model"])
    heads = int(meta["parakeet.encoder.n_heads"])
    head_width = int(meta["parakeet.sortformer.tf_d_model"])
    head_heads = int(meta["parakeet.sortformer.tf_n_heads"])
    return {
        "width": width,
        "layers": int(meta["parakeet.encoder.n_layers"]),
        "heads": heads,
        "head_width": width // heads,
        "xscaling": bool(meta.get("parakeet.encoder.xscaling", True)),
        "speaker_width": head_width,
        "speaker_layers": int(meta["parakeet.sortformer.tf_n_layers"]),
        "speaker_heads": head_heads,
        "speaker_head_width": head_width // head_heads,
        "speakers": int(meta["parakeet.sortformer.num_spks"]),
    }


def frequencies(width):
    return torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * -(math.log(POSITION_BASE) / width))


def phase_tables(frames, width):
    angles = torch.arange(frames, dtype=torch.float32).unsqueeze(1) * frequencies(width)
    return torch.sin(angles), torch.cos(angles)


def query_phases(frames, width):
    sine, cosine = phase_tables(frames, width)
    return channels_first(sine), channels_first(cosine)


def key_phases(frames, width):
    sine, cosine = phase_tables(frames, width)
    return torch.cat((cosine, sine), dim=1).unsqueeze(-1).unsqueeze(-1)


def head_projection(weight, head, config):
    width = config["head_width"]
    by_phase = torch.cat((weight[:, 0::2], weight[:, 1::2]), dim=1)
    return by_phase[head * width:(head + 1) * width].T


def position_projections(tensors, config):
    return [torch.cat([head_projection(tensors[f"encoder.blk.{index}.attn.pos.weight"], head, config)
                       for head in range(config["heads"])])
            for index in range(config["layers"])]


def position_scores(query_v, projection, phases):
    expanded = pointwise(query_v, projection)
    sine_part, cosine_part = expanded.chunk(GLU_HALVES, dim=1)
    sine, cosine, keys = phases
    rotated = torch.cat((sine_part * sine + cosine_part * cosine,
                         cosine_part * sine - sine_part * cosine), dim=1)
    return F.conv2d(rotated, keys)


def relative_head(query_u, query_v, key_rows, value, projection, phases, key_mask):
    content = torch.einsum("bchq,bkhc->bkhq", query_u, key_rows)
    scores = content + position_scores(query_v, projection, phases) + key_mask
    return torch.einsum("bkhq,bchk->bchq", torch.softmax(scores, dim=1), value)


def position_bias(weights, name, scale):
    return (weights[name].reshape(-1) * scale).reshape(1, -1, 1, 1)


def relative_attention(values, weights, prefix, config, positions, key_mask):
    scale = 1.0 / math.sqrt(config["head_width"])
    query = pointwise(values, weights[f"{prefix}.q.weight"], weights[f"{prefix}.q.bias"]) * scale
    key_rows = pointwise(values, weights[f"{prefix}.k.weight"], weights[f"{prefix}.k.bias"]).transpose(1, 3)
    value = pointwise(values, weights[f"{prefix}.v.weight"], weights[f"{prefix}.v.bias"])
    query_u = query + position_bias(weights, f"{prefix}.pos_bias_u", scale)
    query_v = query + position_bias(weights, f"{prefix}.pos_bias_v", scale)
    projections, phases = positions
    width = config["head_width"]
    heads = []
    for head in range(config["heads"]):
        channels = slice(head * width, (head + 1) * width)
        rows = slice(head * config["width"], (head + 1) * config["width"])
        heads.append(relative_head(query_u[:, channels], query_v[:, channels], key_rows[..., channels],
                                   value[:, channels], projections[rows], phases, key_mask))
    return pointwise(torch.cat(heads, dim=1), weights[f"{prefix}.out.weight"], weights[f"{prefix}.out.bias"])


def swish(values):
    return values * logistic(values)


def feed_forward(values, weights, prefix):
    hidden = swish(pointwise(values, weights[f"{prefix}.linear1.weight"], weights[f"{prefix}.linear1.bias"]))
    return pointwise(hidden, weights[f"{prefix}.linear2.weight"], weights[f"{prefix}.linear2.bias"])


def convolution(values, weights, prefix, frame_mask):
    expanded = pointwise(values, weights[f"{prefix}.pw1.weight"].reshape(-1, values.size(1)),
                         weights[f"{prefix}.pw1.bias"])
    content, gate = expanded.chunk(GLU_HALVES, dim=1)
    gated = content * logistic(gate) * frame_mask
    depthwise = weights[f"{prefix}.dw.weight"]
    kernel = depthwise.size(-1)
    mixed = F.conv2d(gated, depthwise.reshape(depthwise.size(0), 1, 1, kernel), weights[f"{prefix}.dw.bias"],
                     padding=(0, (kernel - 1) // 2), groups=depthwise.size(0))
    normalized = (mixed * weights[f"{prefix}.bn.scale"].reshape(1, -1, 1, 1) +
                  weights[f"{prefix}.bn.shift"].reshape(1, -1, 1, 1))
    return pointwise(swish(normalized), weights[f"{prefix}.pw2.weight"].reshape(values.size(1), -1),
                     weights[f"{prefix}.pw2.bias"])


def conformer_block(values, weights, index, config, positions, masks):
    prefix = f"encoder.blk.{index}"
    frame_mask, key_mask = masks
    values = values + FEED_FORWARD_SCALE * feed_forward(
        layer_norm(values, weights, f"{prefix}.norm_ff1"), weights, f"{prefix}.ff1")
    values = values * frame_mask
    values = values + relative_attention(layer_norm(values, weights, f"{prefix}.norm_attn"),
                                         weights, f"{prefix}.attn", config, positions, key_mask)
    values = values * frame_mask
    values = values + convolution(layer_norm(values, weights, f"{prefix}.norm_conv"),
                                  weights, f"{prefix}.conv", frame_mask)
    values = values * frame_mask
    values = values + FEED_FORWARD_SCALE * feed_forward(
        layer_norm(values, weights, f"{prefix}.norm_ff2"), weights, f"{prefix}.ff2")
    return layer_norm(values, weights, f"{prefix}.norm_out") * frame_mask


def conformer_blocks(values, weights, config, positions, masks):
    for index in range(config["layers"]):
        values = conformer_block(values, weights, index, config, positions[index], masks)
    return values


def speaker_attention(values, weights, prefix, config, key_mask):
    scale = 1.0 / math.sqrt(config["speaker_head_width"])
    query = pointwise(values, weights[f"{prefix}.q.weight"], weights[f"{prefix}.q.bias"]) * scale
    key_rows = pointwise(values, weights[f"{prefix}.k.weight"], weights[f"{prefix}.k.bias"]).transpose(1, 3)
    value = pointwise(values, weights[f"{prefix}.v.weight"], weights[f"{prefix}.v.bias"])
    width = config["speaker_head_width"]
    heads = []
    for head in range(config["speaker_heads"]):
        channels = slice(head * width, (head + 1) * width)
        heads.append(head_attention(query[:, channels], key_rows[..., channels], value[:, channels], key_mask))
    return pointwise(torch.cat(heads, dim=1), weights[f"{prefix}.out.weight"], weights[f"{prefix}.out.bias"])


def speaker_block(values, weights, index, config, key_mask):
    prefix = f"sortformer.transformer.blk.{index}"
    values = layer_norm(values + speaker_attention(values, weights, f"{prefix}.attn", config, key_mask),
                        weights, f"{prefix}.ln1")
    hidden = F.relu(pointwise(values, weights[f"{prefix}.ffn.in.weight"], weights[f"{prefix}.ffn.in.bias"]))
    feed = pointwise(hidden, weights[f"{prefix}.ffn.out.weight"], weights[f"{prefix}.ffn.out.bias"])
    return layer_norm(values + feed, weights, f"{prefix}.ln2")


def speaker_blocks(values, weights, config, key_mask):
    for index in range(config["speaker_layers"]):
        values = speaker_block(values, weights, index, config, key_mask)
    return values


def speaker_head(encoded, weights, config, key_mask):
    hidden = pointwise(encoded, weights["sortformer.encoder_proj.weight"], weights["sortformer.encoder_proj.bias"])
    hidden = F.relu(speaker_blocks(hidden, weights, config, key_mask))
    hidden = F.relu(pointwise(hidden, weights["sortformer.head.first_hidden_to_hidden.weight"],
                              weights["sortformer.head.first_hidden_to_hidden.bias"]))
    logits = pointwise(hidden, weights["sortformer.head.single_hidden_to_spks.weight"],
                       weights["sortformer.head.single_hidden_to_spks.bias"])
    return logistic(logits)


def diarize(pre_encode, valid_mask, weights, config, positions):
    frame_mask = valid_mask.reshape(1, 1, 1, -1)
    key_mask = ((1.0 - valid_mask) * NEMOTRON_EXPORT.MASKED_SCORE).reshape(1, -1, 1, 1)
    values = channels_first(pre_encode)
    if config["xscaling"]:
        values = values * math.sqrt(config["width"])
    encoded = conformer_blocks(values * frame_mask, weights, config, positions, (frame_mask, key_mask))
    return NEMOTRON_EXPORT.rows_of(speaker_head(encoded, weights, config, key_mask))


class Weights(dict):
    def __missing__(self, key):
        raise KeyError(f"missing Sortformer tensor {key}")


class SortformerDiarizer(torch.nn.Module):
    def __init__(self, tensors, config, frames):
        super().__init__()
        self.config = config
        self.names = {}
        for index, (key, value) in enumerate(tensors.items()):
            self.names[key] = f"w_{index}"
            self.register_buffer(self.names[key], value.contiguous().float())
        for index, projection in enumerate(position_projections(tensors, config)):
            self.register_buffer(f"positions_{index}", projection.contiguous())
        sine, cosine = query_phases(frames, config["width"])
        self.register_buffer("phase_sine", sine)
        self.register_buffer("phase_cosine", cosine)
        self.register_buffer("phase_keys", key_phases(frames, config["width"]))

    def weights(self):
        return Weights((key, getattr(self, name)) for key, name in self.names.items())

    def positions(self):
        phases = (self.phase_sine, self.phase_cosine, self.phase_keys)
        return [(getattr(self, f"positions_{index}"), phases) for index in range(self.config["layers"])]

    def forward(self, pre_encode, valid_mask):
        return diarize(pre_encode, valid_mask, self.weights(), self.config, self.positions())


def model_tensors(tensors):
    return {key: value for key, value in tensors.items()
            if key.startswith("encoder.blk.") or key.startswith("sortformer.")}


def convert(tensors, config, frames, precision):
    inputs = (torch.zeros(frames, config["width"]), torch.ones(frames))
    model = SortformerDiarizer(model_tensors(tensors), config, frames).eval()
    traced = torch.jit.trace(model, inputs, check_trace=False)
    return ct.convert(
        traced,
        inputs=[NEMOTRON_EXPORT.tensor_type(PRE_ENCODE_INPUT, tuple(inputs[0].shape)),
                NEMOTRON_EXPORT.tensor_type(MASK_INPUT, tuple(inputs[1].shape))],
        outputs=[NEMOTRON_EXPORT.tensor_type(PROBABILITY_OUTPUT)],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS13,
        skip_model_load=True,
    )


def export_capacity(tensors, config, frames, args):
    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    package = NEMOTRON_EXPORT.sidecar_package(args.gguf, frames)
    if args.out_dir:
        package = Path(args.out_dir) / package.name
    convert(tensors, config, frames, precision).save(str(package))
    print(f"[export] saved {package}")
    if args.compile_dir is not None:
        print(f"[export] compiled {NEMOTRON_EXPORT.compile_package(package, args.compile_dir)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True,
                        help="capacities in 80 ms encoder frames, speaker cache, FIFO and chunk")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="directory for the .mlpackage files (default: beside the GGUF)")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16",
                        help="fp32 is a parity reference; only fp16 runs on the Neural Engine")
    parser.add_argument("--compile-dir", type=Path, default=None,
                        help="compile each package into this directory (macOS only)")
    return parser.parse_args()


def main():
    args = parse_args()
    tensors, meta = REFERENCE.load_gguf(args.gguf)
    ENCODER_EXPORT.validate_export_contract(meta, bypass_pre_encode=True)
    config = model_config(meta)
    for frames in args.frames:
        if frames <= 0:
            raise SystemExit(f"--frames must be positive, got {frames}")
        export_capacity(tensors, config, frames, args)


if __name__ == "__main__":
    main()
