#!/usr/bin/env python3
"""Diagnose tiny exported decoder inference independently of native cache import.

Use only the synthetic CI GGUF. The oracle uses ordinary fp32 Linear,
LayerNorm and full cached attention, independently of DecoderStep's conv and
split-current-position graph. Both paths start with identical nonzero fp16
cache values. No checkpoint download or audio generation is involved.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import traceback

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F


def load_exporter():
    path = Path(__file__).resolve().parents[2] / "scripts/parler/export-decoder-coreml.py"
    spec = importlib.util.spec_from_file_location("parler_decoder_export_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_step(tensors, hp, hidden, caches, past, cross_length):
    """One plain fp32 transformer step, with no exported graph helpers."""
    d, h = hp["d_model"], hp["n_head"]
    hd = d // h
    x = torch.from_numpy(hidden.astype(np.float32))

    def norm(x, prefix):
        return F.layer_norm(x, (d,), tensors[prefix + ".weight"],
                            tensors[prefix + ".bias"], hp["ln_eps"])

    def project(x, name):
        return F.linear(x, tensors[name + ".weight"])

    keys, values = [], []
    for layer in range(hp["n_layer"]):
        p = f"dec.blk.{layer}."
        y = norm(x, p + "attn_norm")
        q, k, v = [project(y, p + "attn_" + kind).reshape(h, 1, hd)
                   for kind in ("q", "k", "v")]
        keys.append(k)
        values.append(v)
        k_all = torch.cat((caches["self_k"][layer, :, :past], k), dim=1)
        v_all = torch.cat((caches["self_v"][layer, :, :past], v), dim=1)
        weights = torch.softmax((q @ k_all.transpose(-1, -2)) / np.sqrt(hd), dim=-1)
        x = x + project((weights @ v_all).reshape(d), p + "attn_o")
        q = project(norm(x, p + "cross_norm"), p + "cross_q").reshape(h, 1, hd)
        ck = caches["cross_k"][layer, :, :cross_length]
        cv = caches["cross_v"][layer, :, :cross_length]
        weights = torch.softmax((q @ ck.transpose(-1, -2)) / np.sqrt(hd), dim=-1)
        x = x + project((weights @ cv).reshape(d), p + "cross_o")
        x = x + project(F.gelu(project(norm(x, p + "ffn_norm"), p + "ffn_up"),
                               approximate="none"), p + "ffn_down")
    x = norm(x, "dec.output_norm")
    return {"logits": torch.stack([project(x, f"dec.lm_heads.{q}") for q in range(hp["n_q"])]).numpy(),
            "new_k": torch.stack(keys).numpy(), "new_v": torch.stack(values).numpy()}


def run_case(exporter, package, tensors, hp, units, tolerance):
    model = ct.models.MLModel(str(package), compute_units=units)
    meta = json.loads(model.user_defined_metadata["parler.decoder.contract"])
    layers, heads, hd = hp["n_layer"], hp["n_head"], hp["d_model"] // hp["n_head"]
    group, ctx, cross = meta["states"]["layers_per_state"], meta["ctx"], meta["cross_context"]
    assert meta["states"]["k_layout"] == "rows"
    assert layers == 2 and hp["d_model"] == 32, "diagnostic requires the tiny CI decoder"
    label = f"graph group={group} units={units.name}"
    rng = np.random.default_rng(813)
    initial = {}
    for kind in ("self_k", "self_v", "cross_k", "cross_v"):
        rows, used = (ctx, 3) if kind.startswith("self") else (cross, 2)
        a = np.zeros((layers, heads, rows, hd), dtype=np.float32)
        # Distinct nonzero layers, heads and positions expose slice/stride bugs.
        a[:, :, :used] = (rng.standard_normal((layers, heads, used, hd)) * 0.25).astype(np.float16)
        initial[kind] = a
    hidden_rows = [rng.standard_normal(hp["d_model"]).astype(np.float16).astype(np.float32)
                   for _ in range(3)]
    weights = {name: torch.from_numpy(value.copy()) for name, value in tensors.items()}
    passed = True
    for repeat in range(2):
        state = model.make_state()
        host = {kind: a.copy() for kind, a in initial.items()}
        reference = {kind: torch.from_numpy(a.copy()) for kind, a in initial.items()}
        for kind, a in host.items():
            for g in range(layers // group):
                state.write_state(exporter.state_name(kind, g), a[g * group:(g + 1) * group])
        for step, hidden in enumerate(hidden_rows):
            past = 3 + step
            expected = reference_step(weights, hp, hidden, reference, past, 2)
            actual = model.predict({"hidden": hidden.reshape(1, -1, 1, 1).astype(np.float16),
                                    "self_mask": exporter.self_mask(past, ctx).astype(np.float16),
                                    "cross_mask": exporter.self_mask(2, cross).astype(np.float16)}, state=state)
            errors = {}
            for name in ("logits", "new_k", "new_v"):
                got = np.asarray(actual[name], dtype=np.float32)
                assert got.shape == expected[name].shape, (name, got.shape, expected[name].shape)
                peak = float(np.max(np.abs(got - expected[name]))) if np.isfinite(got).all() else float("inf")
                errors[name] = peak
                passed &= peak < tolerance
            print(f"{label} repeat={repeat} step={step} " +
                  " ".join(f"{name}={value:.6f}" for name, value in errors.items()), flush=True)
            for kind, output in (("self_k", "new_k"), ("self_v", "new_v")):
                host[kind][:, :, past:past + 1] = actual[output]
                # Each path advances its own fp16 cache, as the native engine does.
                reference[kind][:, :, past:past + 1] = torch.from_numpy(expected[output].astype(np.float16).astype(np.float32))
                for g in range(layers // group):
                    state.write_state(exporter.state_name(kind, g), host[kind][g * group:(g + 1) * group])
    print(f"{'PASS' if passed else 'FAIL'} {label} tolerance={tolerance}", flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    args = parser.parse_args()
    assert args.gguf.stat().st_size < 2_000_000, "use only the tiny synthetic GGUF"
    exporter = load_exporter()
    tensors, hp = exporter.read_gguf(args.gguf)
    passed = True
    for units in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.ALL):
        try:
            passed &= run_case(exporter, args.package, tensors, hp, units, 0.02)
        except Exception:
            print(f"FAIL graph package={args.package} units={units.name}", flush=True)
            traceback.print_exc()
            passed = False
    print("Graph checks use Python MLState writes; native checks also cover ggml cache import, "
          "reused native input providers and native row writes.", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
