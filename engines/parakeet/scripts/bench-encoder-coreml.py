#!/usr/bin/env python3
"""Benchmark the exported Core ML FastConformer encoder across fixed input
lengths. For each length it reports Apple Neural Engine op placement, fp16
fidelity against the pure-PyTorch reference encoder, and Core ML prediction
latency. Reuses export-encoder-coreml.py so the measured graph is the one the
shipped `.mlmodelc` sidecar uses.

The ggml-Metal head-to-head RTF is measured separately through the engine
(`EngineResult.encoder_ms` with `PARAKEET_COREML_DISABLE` toggled); this script
covers the ANE-specific placement/fidelity that require coremltools.

Example:

  python scripts/bench-encoder-coreml.py \
      --gguf models/parakeet-tdt-0.6b-v3.f16.gguf \
      --mel-frames 138 826 2201
"""
import argparse
import importlib.util
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct


def load_export_module(scripts_dir):
    path = scripts_dir / "export-encoder-coreml.py"
    spec = importlib.util.spec_from_file_location("parakeet_export_coreml", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_frame_cosine(reference, actual):
    reference = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-12)
    actual = actual / (np.linalg.norm(actual, axis=1, keepdims=True) + 1e-12)
    cosines = (reference * actual).sum(axis=1)
    return float(cosines.mean()), float(cosines.min())


def convert_fixed_length(export, model, n_mels, n_mel_frames, out_path,
                         palettize_bits, palettize_group_size):
    example = torch.zeros(n_mels, n_mel_frames, dtype=torch.float32)
    examples = (example,)
    inputs = [ct.TensorType(name="mel", shape=(n_mels, n_mel_frames), dtype=np.float16)]
    if isinstance(model, export.MaskedEncoderModule):
        examples += (torch.ones(n_mel_frames),)
        inputs.append(ct.TensorType(name="mel_valid_mask", shape=(n_mel_frames,),
                                    dtype=np.float16))
    traced = torch.jit.trace(model, examples, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name="encoder_out", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=(ct.target.macOS15 if palettize_bits is not None
                                   else ct.target.macOS13),
    )
    if palettize_bits is not None:
        mlmodel = export.palettize_encoder(mlmodel, palettize_bits,
                                           palettize_group_size)
    mlmodel.save(str(out_path))
    return mlmodel


def op_placement(mlmodelc_path):
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(str(mlmodelc_path), compute_units=ct.ComputeUnit.ALL)
    counts = {}
    for function in plan.model_structure.program.functions.values():
        for operation in function.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            if usage is None:
                continue
            device = type(usage.preferred_compute_device).__name__
            counts[device] = counts.get(device, 0) + 1
    return counts


def require_ane_placement(placement):
    ane = placement.get("MLNeuralEngineComputeDevice", 0)
    if ane <= 0:
        raise RuntimeError(
            "Core ML graph has no operations assigned to the Apple Neural Engine")


def prediction_inputs(mlmodel, mel):
    inputs = {"mel": mel}
    if any(x.name == "mel_valid_mask" for x in mlmodel.get_spec().description.input):
        inputs["mel_valid_mask"] = np.ones(mel.shape[1], dtype=mel.dtype)
    return inputs


def predict_latency_ms(mlmodel, mel, iters):
    inputs = prediction_inputs(mlmodel, mel)
    start = time.perf_counter()
    for _ in range(iters):
        mlmodel.predict(inputs)
    return 1000.0 * (time.perf_counter() - start) / iters


def measure_length(export, ref, model, n_mels, n_mel_frames, iters, work_dir,
                   palettize_bits, palettize_group_size):
    mel = np.random.default_rng(0).standard_normal((n_mels, n_mel_frames)).astype(np.float32) * 2.0 - 4.0
    with torch.inference_mode():
        args = (torch.from_numpy(mel),)
        if isinstance(model, export.MaskedEncoderModule):
            args += (torch.ones(n_mel_frames),)
        reference_out = model(*args).cpu().numpy().astype(np.float32)

    package = work_dir / f"bench-{n_mel_frames}.mlpackage"
    mlmodel = convert_fixed_length(export, model, n_mels, n_mel_frames, package,
                                   palettize_bits, palettize_group_size)
    export.compile_mlmodelc(package, work_dir)
    compiled = next(work_dir.glob(f"bench-{n_mel_frames}.mlmodelc"))

    coreml_mel = mel.astype(np.float16)
    coreml_out = np.asarray(mlmodel.predict(prediction_inputs(mlmodel, coreml_mel))["encoder_out"]).astype(np.float32)
    mean_cos, min_cos = per_frame_cosine(reference_out, coreml_out)
    latency = predict_latency_ms(mlmodel, coreml_mel, iters)
    placement = op_placement(compiled)
    require_ane_placement(placement)
    return {
        "enc_frames": reference_out.shape[0],
        "mean_cos": mean_cos,
        "min_cos": min_cos,
        "predict_ms": latency,
        "placement": placement,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--scripts", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--mel-frames", type=int, nargs="+", required=True,
                    help="fixed mel-frame lengths to benchmark")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--palettize-bits", type=int, choices=[4, 6, 8], default=None,
                    help="benchmark grouped-channel palettization at this bit width")
    ap.add_argument("--palettize-group-size", type=int, default=16)
    args = ap.parse_args()
    if args.palettize_group_size <= 0:
        ap.error("--palettize-group-size must be greater than zero")

    export = load_export_module(args.scripts)
    weights, meta = export.load_reference_encoder(args.scripts).load_gguf(args.gguf)
    ref = export.load_reference_encoder(args.scripts)
    export.validate_export_contract(meta, flexible=False)
    n_mels = int(weights["preproc.mel_filterbank"].shape[0])
    module = export.MaskedEncoderModule if export.model_type(meta) == "tdt" else export.EncoderModule
    model = module(ref, weights, meta).eval()

    print(f"{'mel_frames':>10} {'enc_frames':>10} {'ANE':>6} {'GPU':>6} {'CPU':>6} "
          f"{'mean_cos':>9} {'min_cos':>8} {'predict_ms':>10}")
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for n_mel_frames in args.mel_frames:
            r = measure_length(export, ref, model, n_mels, n_mel_frames, args.iters,
                               work_dir, args.palettize_bits,
                               args.palettize_group_size)
            p = r["placement"]
            ane = p.get("MLNeuralEngineComputeDevice", 0)
            gpu = p.get("MLGPUComputeDevice", 0)
            cpu = p.get("MLCPUComputeDevice", 0)
            print(f"{n_mel_frames:>10} {r['enc_frames']:>10} {ane:>6} {gpu:>6} {cpu:>6} "
                  f"{r['mean_cos']:>9.5f} {r['min_cos']:>8.5f} {r['predict_ms']:>10.1f}")


if __name__ == "__main__":
    main()
