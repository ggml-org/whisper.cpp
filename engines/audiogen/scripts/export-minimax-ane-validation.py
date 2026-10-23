#!/usr/bin/env python3
import argparse
import gc
import json
from pathlib import Path
import time

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
from gguf import GGUFReader
import numpy as np
import torch

from minimax_ane_models import Conditioner, DepthBlock, DepthInput, DiTBlock, DiTInput, DiTOutput, Feedback, Head, LMBlock, LMBlockPair, Vocoder, compact_lm_head


CONV_SCALE_FUSION_PASS = "common::fuse_conv_scale"


def conversion_pipeline(disabled_passes):
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(set(disabled_passes))
    return pipeline


def metadata(reader, prefix):
    return {name.removeprefix(prefix): field.contents() for name, field in reader.fields.items() if name.startswith(prefix)}


def subset(reader, prefix):
    return {tensor.name.removeprefix(prefix): tensor.data for tensor in reader.tensors if tensor.name.startswith(prefix)}


def reuse_compiled_export(compiled, report_path, settings):
    if not compiled.exists() or not report_path.exists():
        return False
    previous = json.loads(report_path.read_text())
    if any(previous.get(key, [] if key == "disabled_conversion_passes" else None) != value
           for key, value in settings.items()):
        raise ValueError("Existing compiled model uses different conversion settings; use a new output directory")
    return True


def export(name, module, data, outputs, directory, precision=ct.precision.FLOAT16, disabled_passes=(), artifact_settings=None):
    compiled = directory / (name + ".mlmodelc")
    report_path = directory / (name + ".json")
    settings = {"disabled_conversion_passes": list(disabled_passes), **(artifact_settings or {})}
    if reuse_compiled_export(compiled, report_path, settings):
        print(f"REUSE {name}", flush=True)
        return
    print(f"EXPORT {name}", flush=True)
    start = time.monotonic()
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, tuple(data.values()), check_trace=False)
    states = [ct.StateType(name=key, wrapped_type=ct.TensorType(shape=value.shape, dtype=np.float16))
              for key, value in module.named_buffers() if key.endswith(("key_cache", "value_cache"))]
    converted = ct.convert(traced, convert_to="mlprogram", inputs=[ct.TensorType(name=key, shape=value.shape, dtype=np.float32)
        for key, value in data.items()], outputs=[ct.TensorType(name=name, dtype=np.float32) for name in outputs],
        states=states, minimum_deployment_target=ct.target.macOS15, compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True,
        pass_pipeline=conversion_pipeline(disabled_passes))
    package = directory / (name + ".mlpackage")
    converted.save(str(package))
    print(f"COMPILE {name}", flush=True)
    ct.utils.compile_model(str(package), str(compiled))
    del converted, traced
    gc.collect()
    print(f"PLAN {name}", flush=True)
    plan = MLComputePlan.load_from_path(str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    rows = []
    for function in plan.model_structure.program.functions.values():
        for operation in function.block.operations:
            if operation.operator_name == "const":
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            rows.append({"operator": operation.operator_name,
                         "device": type(usage.preferred_compute_device).__name__ if usage else "unknown"})
    report_path.write_text(json.dumps({"conversion_and_compile_s": time.monotonic()-start,
                                      **settings, "placement": rows}, indent=2)+"\n")
    devices = {device: sum(row["device"] == device for row in rows) for device in {row["device"] for row in rows}}
    print(f"READY {name} {time.monotonic()-start:.2f}s {devices}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=["lm", "lm-pair", "lm-head", "depth", "dit", "vocoder", "aux"], required=True)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--lm-cache-positions", type=int)
    parser.add_argument("--disable-conv-scale-fusion", action="store_true")
    parser.add_argument("--stable-lm-silu", action="store_true")
    parser.add_argument("--vocoder-precision", choices=["all-fp16", "fp32-activations"])
    args = parser.parse_args()
    if args.stage == "vocoder" and args.vocoder_precision is None:
        parser.error("Vocoder validation requires an explicit --vocoder-precision")
    if (args.disable_conv_scale_fusion or args.stable_lm_silu) and args.stage not in ("lm", "lm-pair"):
        parser.error("The projection scaling control is limited to LM validation")
    disabled_passes = (CONV_SCALE_FUSION_PASS,) if args.disable_conv_scale_fusion else ()
    torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=True)
    lm = GGUFReader(str(args.models / "mm3-lm-f16.gguf"))
    sy = GGUFReader(str(args.models / "mm3-synth-f16.gguf"))
    lc, dc, tc, vc = [metadata(reader, prefix) for reader, prefix in ((lm, "qwen3."), (sy, "mm3.depth."),
                                                                      (sy, "mm3.dit."), (sy, "mm3.voc."))]
    tensors = {tensor.name: tensor.data for tensor in lm.tensors}
    full = lc["context_length"]
    capacity = args.lm_cache_positions if args.lm_cache_positions is not None else full
    if not 0 < capacity <= full:
        parser.error("LM cache profile must fit the original context")
    if args.stage == "lm-pair":
        if args.lm_cache_positions is None or args.count is None:
            parser.error("LM pair validation requires explicit --lm-cache-positions and --count")
        if args.first < 0 or args.first % 2 or args.count <= 0 or args.count % 2 or args.first+args.count > lc["block_count"]:
            parser.error("LM pairs must cover an even number of consecutive original blocks")
    manifest = {"format": 1, "context": full, "depth_context": 32, "dit_latents": tc["window_latents"],
                "condition_frames": tc["window_frames"], "vocoder_latents": 256,
                "lm_layers": lc["block_count"], "depth_layers": dc["block_count"], "dit_layers": tc["block_count"],
                "compute_units": "CPU_AND_NE", "scope": "Experimental full-route validation; no production default change"}
    if args.lm_cache_positions is not None:
        manifest["lm_profile_positions"] = capacity
    if args.stage == "lm-pair":
        manifest["lm_blocks_per_program"] = 2
    if disabled_passes:
        manifest["disabled_conversion_passes"] = list(disabled_passes)
    if args.stable_lm_silu:
        manifest["lm_activation"] = "stable_silu"
    manifest_path = args.out/"manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing compiled model profile has different dimensions")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    if args.stage in ("lm", "lm-pair"):
        pair = args.stage == "lm-pair"
        for index in range(args.first, min(lc["block_count"], args.first+(args.count or lc["block_count"])), 2 if pair else 1):
            module = (LMBlockPair([subset(lm, f"blk.{index+offset}.") for offset in range(2)], lc, capacity, args.stable_lm_silu)
                      if pair else LMBlock(subset(lm, f"blk.{index}."), lc, capacity, args.stable_lm_silu))
            name = f"lm-pair-{index//2}" if pair else f"lm-{index}"
            export(name, module, {"hidden": torch.zeros(1, 4096, 1, 2), "selector": torch.zeros(1, capacity),
                "mask": torch.zeros(1, capacity, 1, 1), "cosine": torch.ones(1, 1, 64, 1), "sine": torch.zeros(1, 1, 64, 1)},
                ["hidden_out"], args.out, disabled_passes=disabled_passes)
            del module
            gc.collect()
    elif args.stage == "depth":
        for index in range(args.first, min(dc["block_count"], args.first+(args.count or dc["block_count"]))):
            module = DepthBlock(subset(sy, f"depth.blk.{index}."), dc, 32)
            export(f"depth-{index}", module, {"hidden": torch.zeros(1, 4096, 1, 2), "selector": torch.zeros(1, 32),
                "mask": torch.zeros(1, 32, 1, 1)}, ["hidden_out"], args.out)
    elif args.stage == "dit":
        length = tc["window_latents"]+1
        for index in range(args.first, min(tc["block_count"], args.first+(args.count or tc["block_count"]))):
            module = DiTBlock(subset(sy, f"dit.blk.{index}."), tc, length)
            export(f"dit-{index}", module, {"hidden": torch.zeros(2, 2048, 1, length), "mask": torch.zeros(1, length, 1, 1)},
                ["hidden_out"], args.out)
            del module
            gc.collect()
    elif args.stage == "vocoder":
        precision = (ct.precision.FLOAT16 if args.vocoder_precision == "all-fp16" else
                     ct.transform.FP16ComputePrecision(op_selector=lambda operation: operation.op_type == "conv"))
        export("vocoder", Vocoder(subset(sy, "voc."), vc), {"latent": torch.zeros(1, 64, 1, 256),
            "valid": torch.ones(1, 1, 1, 256)}, ["waveform"], args.out,
            precision, artifact_settings={"vocoder_precision": args.vocoder_precision})
    else:
        st = {tensor.name: tensor.data for tensor in sy.tensors}
        compact = compact_lm_head(tensors, int(lm.fields["mm3.semantic_vocab_offset"].contents()),
            int(lm.fields["mm3.semantic_vocab_size"].contents()), int(lm.fields["mm3.eos_audio"].contents()))
        export("lm-head", Head({"norm": tensors["output_norm.weight"], "head.weight": compact}, 1e-6),
            {"hidden": torch.zeros(1, 4096, 1, 2)}, ["hidden_out", "logits"], args.out)
        if args.stage == "lm-head":
            return
        export("feedback", Feedback(float(lm.fields["mm3.ar.embedding_scale"].contents())),
            {"embeddings": torch.zeros(1, 4096, 8, 1)}, ["feedback"], args.out)
        export("depth-input", DepthInput({"proj.weight": st["depth.proj.weight"]}),
            {"hidden": torch.zeros(1, 4096, 1, 2), "positional": torch.zeros(1, 4096, 1, 2)}, ["hidden_out"], args.out)
        for index in range(dc["num_codebooks"]-1):
            export(f"depth-head-{index}", Head({"norm": st["depth.output_norm.weight"], "head.weight": st[f"depth.head.{index}.weight"]}, dc["rms_eps"]),
                {"hidden": torch.zeros(1, 4096, 1, 2)}, ["hidden_out", "logits"], args.out)
        export("conditioner", Conditioner(subset(sy, "cond.")), {"hidden": torch.zeros(1, 4096, 8, tc["window_frames"]+2)}, ["condition"], args.out)
        names = ["preprocess_conv.weight", "proj_in.weight", "time_embd.0.weight", "time_embd.0.bias", "time_embd.1.weight", "time_embd.1.bias"]
        export("dit-input", DiTInput({name: st["dit."+name] for name in names}),
            {"latent": torch.zeros(1, 128, 1, tc["window_latents"]), "condition": torch.zeros(1, 2048, 1, tc["window_latents"]),
             "fourier": torch.zeros(1, 256, 1, 1), "gates": torch.ones(2, 1, 1, 1)}, ["hidden_out"], args.out)
        export("dit-output", DiTOutput({name: st["dit."+name] for name in ["proj_out.weight", "postprocess_conv.weight"]}),
            {"hidden": torch.zeros(2, 2048, 1, tc["window_latents"]+1)}, ["velocity"], args.out)


if __name__ == "__main__":
    main()
