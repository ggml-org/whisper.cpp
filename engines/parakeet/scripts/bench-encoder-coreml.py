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


def convert_fixed_length(export, model, n_mels, n_mel_frames, out_path):
    example = torch.zeros(n_mels, n_mel_frames, dtype=torch.float32)
    traced = torch.jit.trace(model, example, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=(n_mels, n_mel_frames), dtype=np.float32)],
        outputs=[ct.TensorType(name="encoder_out", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS13,
    )
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


def predict_latency_ms(mlmodel, mel, iters):
    start = time.perf_counter()
    for _ in range(iters):
        mlmodel.predict({"mel": mel})
    return 1000.0 * (time.perf_counter() - start) / iters


def measure_length(export, ref, model, n_mels, n_mel_frames, iters, work_dir):
    mel = np.random.default_rng(0).standard_normal((n_mels, n_mel_frames)).astype(np.float32) * 2.0 - 4.0
    with torch.inference_mode():
        reference_out = model(torch.from_numpy(mel)).cpu().numpy().astype(np.float32)

    package = work_dir / f"bench-{n_mel_frames}.mlpackage"
    mlmodel = convert_fixed_length(export, model, n_mels, n_mel_frames, package)
    export.compile_mlmodelc(package, work_dir)
    compiled = next(work_dir.glob(f"bench-{n_mel_frames}.mlmodelc"))

    coreml_out = np.asarray(mlmodel.predict({"mel": mel})["encoder_out"]).astype(np.float32)
    mean_cos, min_cos = per_frame_cosine(reference_out, coreml_out)
    latency = predict_latency_ms(mlmodel, mel, iters)
    placement = op_placement(compiled)
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
    args = ap.parse_args()

    export = load_export_module(args.scripts)
    weights, meta = export.load_reference_encoder(args.scripts).load_gguf(args.gguf)
    ref = export.load_reference_encoder(args.scripts)
    n_mels = int(weights["preproc.mel_filterbank"].shape[0])
    model = export.EncoderModule(ref, weights, meta).eval()

    print(f"{'mel_frames':>10} {'enc_frames':>10} {'ANE':>6} {'GPU':>6} {'CPU':>6} "
          f"{'mean_cos':>9} {'min_cos':>8} {'predict_ms':>10}")
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for n_mel_frames in args.mel_frames:
            r = measure_length(export, ref, model, n_mels, n_mel_frames, args.iters, work_dir)
            p = r["placement"]
            ane = p.get("MLNeuralEngineComputeDevice", 0)
            gpu = p.get("MLGPUComputeDevice", 0)
            cpu = p.get("MLCPUComputeDevice", 0)
            print(f"{n_mel_frames:>10} {r['enc_frames']:>10} {ane:>6} {gpu:>6} {cpu:>6} "
                  f"{r['mean_cos']:>9.5f} {r['min_cos']:>8.5f} {r['predict_ms']:>10.1f}")


if __name__ == "__main__":
    main()
