#!/usr/bin/env python3
"""Export Chatterbox S3Gen stages from the GGUF to Core ML sidecars.

The encoder, CFM estimator and HiFT decoder are rebuilt in PyTorch from the
same GGUF tensors the ggml engine loads (scripts/chatterbox_coreml_models.py)
and written next to the GGUF under the engine's naming rule:

  chatterbox-s3gen-q8_0.gguf     -> chatterbox-s3gen-{encoder,cfm,hift}.mlmodelc
  chatterbox-s3gen-mtl-f32.gguf  -> chatterbox-s3gen-mtl-{encoder,cfm,hift}.mlmodelc

The encoder and CFM are multi-function models with one function per width
(`tokens_<T>` / `frames_<F>`, sharing one copy of the weights); the engine
takes the narrowest one that holds a call and keeps ggml for longer inputs.
HiFT is a single window the engine walks with context on both sides. Every
stage computes in float16 with float32 inputs and outputs and targets
macOS 15 / iOS 18 (multi-function models). Export from the f32 or f16 GGUF:
one sidecar serves every quantization tier of the same checkpoint.

  python3 scripts/export-chatterbox-coreml.py --gguf models/chatterbox-s3gen-f32.gguf \\
      --output models --compile
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from chatterbox_coreml_models import (EMBED, MEL, TIME_EMBED, HIFT_HOP, STFT_BINS, Encoder, Estimator, HiFT,
                                      hift_receptive_frames, read_gguf)

STAGES = ("encoder", "cfm", "hift")
QUANT_TAGS = ("f32", "f16", "bf16", "q8_0", "q5_0", "q4_0")
DEFAULT_FRAMES = (512, 640, 768, 896, 1024, 1280, 1536)
DEFAULT_WINDOW = 128
SCHEMA = "1"


def model_stem(path):
    return re.sub(r"[.-](?:" + "|".join(QUANT_TAGS) + r")$", "", Path(path).stem, flags=re.IGNORECASE)


def function_name(stage, width):
    return f"tokens_{width}" if stage == "encoder" else f"frames_{width}"


def zeros(*shape):
    return torch.zeros(shape, dtype=torch.float32)


def stage_model(stage, tensors, metadata, width):
    """(module, example inputs, output names) of one stage at one width."""
    if stage == "encoder":
        return (Encoder(tensors, metadata, width).eval(),
                {"embedding": zeros(1, width, EMBED), "valid": torch.ones(1, 1, width)}, ["mu"])
    if stage == "cfm":
        batch = 1 if metadata.get("s3gen.meanflow", True) else 2
        planes = {name: zeros(batch, MEL, width) for name in ("x", "mu", "cond")}
        return (Estimator(tensors, width, batch).eval(),
                dict(planes, spks=zeros(batch, MEL), t_emb=zeros(batch, TIME_EMBED), valid=torch.ones(1, 1, width)),
                ["velocity"])
    return (HiFT(tensors, width).eval(),
            {"mel": zeros(1, MEL, width), "source": zeros(1, STFT_BINS, width * HIFT_HOP + 1),
             "valid": torch.ones(1, 1, width)}, ["spectrum"])


def convert(model, inputs, outputs):
    import coremltools as ct

    with torch.no_grad():
        traced = torch.jit.trace(model, tuple(inputs.values()), check_trace=False)
    return ct.convert(traced, convert_to="mlprogram",
                      inputs=[ct.TensorType(name=name, shape=tuple(value.shape), dtype=np.float32)
                              for name, value in inputs.items()],
                      outputs=[ct.TensorType(name=name, dtype=np.float32) for name in outputs],
                      minimum_deployment_target=ct.target.iOS18,
                      compute_precision=ct.precision.FLOAT16)


def widths(stage, args, ratio):
    if stage == "hift":
        return [args.window]
    if any(frames % ratio for frames in args.frames):
        raise ValueError(f"--frames must be multiples of the token-mel ratio {ratio}")
    return [frames // ratio for frames in args.frames] if stage == "encoder" else list(args.frames)


def export_stage(stage, tensors, metadata, args):
    import coremltools as ct

    ratio = int(metadata.get("s3gen.encoder.token_mel_ratio", 2))
    stem = model_stem(args.gguf) + "-" + stage
    package = args.output / (stem + ".mlpackage")
    if package.exists():
        if not args.force:
            raise FileExistsError(f"{package} exists (pass --force to replace it)")
        shutil.rmtree(package)
    sizes = widths(stage, args, ratio)
    manifest = {"stage": stage, "source": args.gguf.name, "variant": metadata.get("s3gen.variant", ""),
                "functions": {}, "coremltools": ct.__version__, "torch": torch.__version__, "schema": SCHEMA}
    with tempfile.TemporaryDirectory() as scratch:
        descriptor = ct.utils.MultiFunctionDescriptor() if stage != "hift" else None
        for width in sizes:
            model, inputs, outputs = stage_model(stage, tensors, metadata, width)
            converted = convert(model, inputs, outputs)
            name = function_name(stage, width) if descriptor else "main"
            manifest["functions"][name] = {"inputs": {k: list(v.shape) for k, v in inputs.items()},
                                           "outputs": outputs}
            if descriptor is None:
                converted.user_defined_metadata["chatterbox.hift.window"] = str(width)
                converted.user_defined_metadata["chatterbox.hift.context"] = str(hift_receptive_frames())
                stamp(converted, stage, args, metadata)
                converted.save(str(package))
                break
            part = Path(scratch) / f"{name}.mlpackage"
            converted.save(str(part))
            descriptor.add_function(str(part), src_function_name="main", target_function_name=name)
            print(f"[{stage}] converted {name}")
        if descriptor is not None:
            descriptor.default_function_name = function_name(stage, sizes[0])
            # The merged package keeps no user metadata; the manifest records it.
            ct.utils.save_multifunction(descriptor, str(package))
    (args.output / (stem + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    if args.compile:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(args.output)],
                       check=True, capture_output=True, text=True)
        compiled = args.output / (stem + ".mlmodelc")
        if args.report_placement:
            report_placement(compiled, manifest["functions"])
    print(package)


def stamp(model, stage, args, metadata):
    model.user_defined_metadata["chatterbox.stage"] = stage
    model.user_defined_metadata["chatterbox.source"] = args.gguf.name
    model.user_defined_metadata["chatterbox.variant"] = str(metadata.get("s3gen.variant", ""))
    model.user_defined_metadata["chatterbox.coreml.schema"] = SCHEMA


def report_placement(compiled, functions):
    import coremltools as ct

    for name in functions:
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
        program = plan.model_structure.program
        function = program.functions[name] if name in program.functions else program.functions["main"]
        counts = {}
        for op in function.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is not None:
                device = type(usage.preferred_compute_device).__name__
                counts[device] = counts.get(device, 0) + 1
        print(f"[placement] {compiled.name} {name}: {counts}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", required=True, type=Path, help="Chatterbox S3Gen GGUF (Turbo or Multilingual)")
    parser.add_argument("--stage", default="all", choices=STAGES + ("all",))
    parser.add_argument("--frames", default=",".join(map(str, DEFAULT_FRAMES)),
                        help="comma-separated CFM frame widths (encoder widths are these / 2); default %(default)s")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="HiFT mel frames per call (default %(default)s)")
    parser.add_argument("--output", type=Path, help="output directory (default: next to the GGUF)")
    parser.add_argument("--compile", action="store_true", help="also compile to <output>/<stem>.mlmodelc")
    parser.add_argument("--report-placement", action="store_true", help="print the Neural Engine placement")
    parser.add_argument("--force", action="store_true", help="replace existing packages")
    args = parser.parse_args()
    args.output = args.output or args.gguf.parent
    try:
        args.frames = sorted({int(f) for f in args.frames.split(",") if f.strip()})
    except ValueError:
        parser.error("--frames must be comma-separated integers")
    if not args.frames or args.frames[0] < 2:
        parser.error("--frames needs at least one positive width")
    if args.window <= 2 * hift_receptive_frames():
        parser.error(f"--window must exceed twice the HiFT context ({hift_receptive_frames()} frames)")
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tensors, metadata = read_gguf(args.gguf, ("flow/", "cfm/", "hift/"))
    for stage in STAGES if args.stage == "all" else (args.stage,):
        export_stage(stage, tensors, metadata, args)


if __name__ == "__main__":
    main()
