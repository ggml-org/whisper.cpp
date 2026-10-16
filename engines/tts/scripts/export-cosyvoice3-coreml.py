#!/usr/bin/env python3
"""Export CosyVoice3 GGUF stages to Core ML models compatible with the native engine."""

import argparse
import json
import re
from pathlib import Path
import subprocess

import numpy as np
import torch

from cosyvoice_coreml_models import (CFG_BATCH, MEL_CHANNELS, TIME_CHANNELS,
                                   QwenIO, Flow, FlowFrontend, F0, HiFT, read_gguf)

STAGES = ("llm", "flow", "flow-frontend", "hift-f0", "hift")
QUANT_TAGS = ("f32", "f16", "bf16", "q8_0", "q5_0", "q4_0")


def model_stem(path, stage):
    stem = path.stem
    stem = re.sub(r"[.-](?:" + "|".join(QUANT_TAGS) + r")$", "", stem, flags=re.IGNORECASE)
    suffix = {"flow-frontend": "-frontend", "hift-f0": "-f0"}.get(stage, "")
    return stem + suffix


def zeros(*shape):
    return torch.zeros(shape, dtype=torch.float32)


def llm_model(tensors, metadata, context):
    model = QwenIO(tensors, metadata, context).eval()
    hidden = model.weight("lm/norm/weight").numel()
    inputs = {"x": zeros(1, 1, hidden), "cosine": torch.ones(1, 1, 1, model.head_dim // 2),
              "sine": zeros(1, 1, 1, model.head_dim // 2), "write_mask": zeros(1, 1, context, 1),
              "mask": zeros(1, 1, 1, context)}
    inputs["write_mask"][..., 0, :] = 1
    cache_shape = (model.depth, 1, model.kv_heads, context, model.head_dim)
    inputs["keys"], inputs["values"] = zeros(*cache_shape), zeros(*cache_shape)
    return model, inputs, ["logits", "keys_out", "values_out"]


def flow_model(tensors, metadata, frames):
    model = Flow(tensors, metadata).eval()
    inputs = {name: zeros(CFG_BATCH, frames, MEL_CHANNELS) for name in ("x", "mu", "cond")}
    inputs.update(spks=zeros(CFG_BATCH, MEL_CHANNELS), time_sin=zeros(CFG_BATCH, TIME_CHANNELS),
                  cosine=torch.ones(1, frames, model.head_dim // 2),
                  sine=zeros(1, frames, model.head_dim // 2), mask=zeros(1, 1, 1, frames))
    return model, inputs, ["velocity"]


def frontend_model(tensors, metadata, frames):
    if frames % 2:
        raise ValueError("flow frontend --frames must be even (two mel frames per token)")
    selected = {k: v for k, v in tensors.items() if k.startswith(
        ("flow/input_embedding/", "flow/pre_lookahead_layer/", "flow/spk_embed_affine_layer/"))}
    model = FlowFrontend(selected).eval()
    speaker_width = model.weight("flow/spk_embed_affine_layer/weight").shape[1]
    return model, {"token_ids": zeros(1, frames // 2), "token_mask": torch.ones(1, frames // 2, 1),
                   "speaker": torch.ones(1, speaker_width)}, ["mu", "spks"]


def hift_model(tensors, metadata, frames, f0):
    prefix = "hift/f0_predictor/"
    selected = {k: v for k, v in tensors.items() if k.startswith(prefix) == f0}
    model = (F0(selected) if f0 else HiFT(selected)).eval()
    inputs = {"mel": zeros(1, MEL_CHANNELS, frames)}
    if not f0:
        inputs["source_stft"] = zeros(1, 18, frames * 120 + 1)
    return model, inputs, ["f0" if f0 else "spec"]


def prepare_model(args):
    prefix = "lm/" if args.stage == "llm" else "flow/" if args.stage.startswith("flow") else "hift/"
    tensors, metadata = read_gguf(args.gguf, prefix)
    if args.stage == "llm":
        return llm_model(tensors, metadata, args.context)
    if args.stage == "flow":
        return flow_model(tensors, metadata, args.frames)
    if args.stage == "flow-frontend":
        return frontend_model(tensors, metadata, args.frames)
    return hift_model(tensors, metadata, args.frames, args.stage == "hift-f0")


def state_specs(model, ct):
    return [ct.StateType(wrapped_type=ct.TensorType(shape=tuple(buffer.shape), dtype=np.float16), name=name)
            for name, buffer in model.named_buffers()]


def use_half_precision(op, stage):
    if stage == "flow":
        return op.op_type in {"linear", "conv"}
    if stage == "hift-f0":
        if op.op_type != "conv":
            return False
        return "condnet__0__" not in op.inputs["weight"].name
    if stage == "hift":
        return False
    return True


def export(args):
    import coremltools as ct

    args.output.mkdir(parents=True, exist_ok=True)
    stem = model_stem(args.gguf, args.stage)
    package = args.output / (stem + ".mlpackage")
    if package.exists():
        raise FileExistsError(package)
    torch.set_num_threads(1)
    model, inputs, output_names = prepare_model(args)
    with torch.no_grad():
        traced = torch.jit.trace(model, tuple(inputs.values()), check_trace=False)
    if hasattr(model, "reset"):
        model.reset()
    converted = ct.convert(traced, convert_to="mlprogram",
                           inputs=[ct.TensorType(name=name, shape=tuple(value.shape), dtype=np.float32)
                                   for name, value in inputs.items()],
                           outputs=[ct.TensorType(name=name, dtype=np.float32) for name in output_names],
                           states=state_specs(model, ct), minimum_deployment_target=ct.target.iOS18,
                           compute_precision=ct.transform.FP16ComputePrecision(
                               op_selector=lambda op: use_half_precision(op, args.stage))
                           if args.stage in {"flow", "hift-f0", "hift"} else ct.precision.FLOAT16,
                           skip_model_load=True)
    converted.user_defined_metadata["cosyvoice3.stage"] = args.stage
    converted.user_defined_metadata["cosyvoice3.source"] = args.gguf.name
    converted.user_defined_metadata["cosyvoice3.coreml.schema"] = "1"
    converted.save(str(package))
    manifest = {"stage": args.stage, "source": args.gguf.name,
                "inputs": {name: list(value.shape) for name, value in inputs.items()},
                "outputs": output_names, "coremltools": ct.__version__, "torch": torch.__version__}
    (args.output / (stem + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    if args.compile:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(args.output)], check=True)
    print(package)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--context", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    dimension = args.context if args.stage == "llm" else args.frames
    if dimension is None or dimension < 1:
        parser.error("llm requires a positive --context; other stages require positive --frames")
    return args


if __name__ == "__main__":
    export(parse_args())
