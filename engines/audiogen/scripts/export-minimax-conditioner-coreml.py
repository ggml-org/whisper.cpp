#!/usr/bin/env python3
"""Export MiniMax's original conditioner convolution for CPU + ANE.

Only export uses Python. C++ retains the native layer mix and nearest
resampling. The fixed window includes two halo frames; it does not limit the
duration of a generation. Export from the same synth GGUF used by the engine.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from gguf import GGUFReader


class ConditionerProjection(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.register_buffer("weight", torch.from_numpy(np.array(weight, dtype=np.float32))[:, :, None, :])
        self.register_buffer("bias", torch.from_numpy(np.array(bias, dtype=np.float32)))

    def forward(self, mixed):
        return torch.nn.functional.conv2d(mixed, self.weight, self.bias, padding=(0, 1))


def load_projection(path):
    reader = GGUFReader(str(path))
    tensors = {t.name: t for t in reader.tensors}
    weight = tensors["cond.proj.weight"].data
    bias = tensors["cond.proj.bias"].data
    if weight.ndim != 3 or weight.shape[2] != 3 or bias.shape != (weight.shape[0],):
        raise ValueError("Expected cond.proj.weight [output, hidden, 3] and matching bias")
    if weight.dtype != np.float16 or bias.dtype != np.float32:
        raise ValueError("Expected the original F16 conditioner projection and F32 bias")
    return ConditionerProjection(weight, bias).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New .mlpackage path")
    parser.add_argument("--frames", type=int, required=True, help="Payload frames per window, excluding halos")
    parser.add_argument("--compile-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.frames <= 0 or args.out.suffix != ".mlpackage":
        parser.error("--frames must be positive and --out must end in .mlpackage")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    import coremltools as ct
    start = time.monotonic()
    torch.set_num_threads(8)
    projection = load_projection(args.synth)
    channels, hidden, _, _ = projection.weight.shape
    example = torch.zeros(1, hidden, 1, args.frames + 2)
    with torch.no_grad():
        traced = torch.jit.trace(projection, example)
    model = ct.convert(
        traced, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13,
        inputs=[ct.TensorType(name="mixed", shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="projected", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True,
    )
    model.user_defined_metadata["minimax.component"] = "conditioner-projection-k3-p1-v1"
    model.user_defined_metadata["minimax.source"] = args.synth.name
    model.short_description = "MiniMax-Music3 original conditioner projection; CPU + ANE"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    export_s = time.monotonic() - start
    args.compile_dir.mkdir(parents=True, exist_ok=True)
    compile_start = time.monotonic()
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(args.out), str(args.compile_dir)], check=True)
    print(json.dumps({"package": str(args.out), "compiled": str(args.compile_dir / (args.out.stem + ".mlmodelc")),
                      "payload_frames": args.frames, "hidden": hidden, "channels": channels,
                      "export_s": export_s, "compile_s": time.monotonic() - compile_start}), flush=True)


if __name__ == "__main__":
    main()
