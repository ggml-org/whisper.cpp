#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import time

import coremltools as ct
from gguf import GGUFReader
import numpy as np
import torch
from torch.nn import functional as F

from minimax_ane_models import LMBlockPair


def metrics(expected, actual):
    expected, actual = [np.asarray(value, dtype=np.float64).ravel() for value in (expected, actual)]
    if expected.shape != actual.shape:
        raise ValueError("Mismatched validation output shapes")
    finite = bool(np.isfinite(expected).all() and np.isfinite(actual).all())
    norm = float(np.linalg.norm(expected))
    error = float(np.linalg.norm(actual-expected))
    product = norm*float(np.linalg.norm(actual))
    relative = error/norm if norm else (0.0 if error == 0 else float("inf"))
    cosine = float(np.dot(expected, actual))/product if product else (1.0 if error == 0 else 0.0)
    return {"finite": finite, "relative_rms": relative, "cosine": cosine,
            "passed": finite and relative <= 0.002 and cosine >= 0.9999}


def dense_block(block, hidden, key, value, position, cosine, sine):
    def norm(x, weight):
        return x*torch.rsqrt(x.square().mean(-1, keepdim=True)+block.eps)*block.w(weight)

    def project(x, name):
        return F.linear(x, block.w(name+".weight"))

    def rotate(x, weight):
        x = norm(x, weight)
        low, high = x.chunk(2, -1)
        return torch.cat([low*cosine-high*sine, high*cosine+low*sine], -1)

    x = norm(hidden, "attn_norm.weight")
    query = rotate(project(x, "attn_q").reshape(2, block.heads, block.dim), "attn_q_norm.weight")
    keys = rotate(project(x, "attn_k").reshape(2, block.kv_heads, block.dim), "attn_k_norm.weight")
    values = project(x, "attn_v").reshape_as(keys)
    key[:, :, position].copy_(keys)
    value[:, :, position].copy_(values)
    group = block.heads//block.kv_heads
    keys = key[:, :, :position+1].float().repeat_interleave(group, 1)
    values = value[:, :, :position+1].float().repeat_interleave(group, 1)
    scores = torch.einsum("bhd,bhsd->bhs", query, keys)/block.dim**0.5
    attention = torch.einsum("bhs,bhsd->bhd", scores.softmax(-1), values).reshape(2, -1)
    hidden = hidden+project(attention, "attn_output")
    x = norm(hidden, "ffn_norm.weight")
    return hidden+project(F.silu(project(x, "ffn_gate"))*project(x, "ffn_up"), "ffn_down")


def load_compiled(path):
    start = time.monotonic()
    model = ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"LOADED {path.name} {time.monotonic()-start:.3f}s", flush=True)
    return model, model.make_state()


def run(args):
    manifest = json.loads((args.compiled/"manifest.json").read_text())
    capacity = manifest["lm_profile_positions"]
    if manifest.get("lm_blocks_per_program") != 2 or not 0 < args.positions <= capacity:
        raise ValueError("Validation requires an LM pair and positions within its native cache profile")
    args.out.mkdir(parents=True, exist_ok=True)
    reader = GGUFReader(str(args.models/"mm3-lm-f16.gguf"))
    config = {name.removeprefix("qwen3."): field.contents() for name, field in reader.fields.items() if name.startswith("qwen3.")}
    tensors = [{tensor.name.removeprefix(f"blk.{index}."): tensor.data for tensor in reader.tensors
                if tensor.name.startswith(f"blk.{index}.")} for index in range(args.first, args.first+2)]
    module = LMBlockPair(tensors, config, capacity, manifest.get("lm_activation") == "stable_silu").eval()
    embeddings = next(tensor.data for tensor in reader.tensors if tensor.name == "token_embd.weight")
    tokens = np.fromfile(args.tokens, np.int32)
    if len(tokens) < args.positions:
        raise ValueError("The token fixture is shorter than the requested validation")
    pair, pair_state = load_compiled(args.compiled/f"lm-pair-{args.first//2}.mlmodelc")
    singles = [load_compiled(args.individual/f"lm-{index}.mlmodelc") for index in range(args.first, args.first+2)]
    cache_shape = (2, config["attention.head_count_kv"], capacity, config["attention.key_length"])
    caches = [(torch.zeros(cache_shape, dtype=torch.float16), torch.zeros(cache_shape, dtype=torch.float16)) for _ in range(2)]
    cases = []
    for position in range(args.positions):
        hidden = embeddings[[int(tokens[position]), args.unconditional_token]].astype(np.float32).copy()
        hidden.tofile(args.out/f"input-{position}.f32")
        phase = position*config["rope.freq_base"]**(-2*np.arange(config["attention.key_length"]//2, dtype=np.float64)/config["attention.key_length"])
        cosine, sine = [value.astype(np.float32) for value in (np.cos(phase), np.sin(phase))]
        selector = np.zeros((1, capacity), np.float32)
        selector[0, position] = 1
        mask = np.full((1, capacity, 1, 1), -np.inf, np.float32)
        mask[:, :position+1] = 0
        data = {"hidden": hidden.T[None, :, None, :].copy(), "selector": selector, "mask": mask,
                "cosine": cosine[None, None, :, None], "sine": sine[None, None, :, None]}
        with torch.no_grad():
            reference = torch.from_numpy(hidden)
            for block, (key, value) in zip(module.blocks, caches):
                reference = dense_block(block, reference, key, value, position, torch.from_numpy(cosine), torch.from_numpy(sine))
            expected = reference.numpy().T[None, :, None, :].copy()
            eager = module(*(torch.from_numpy(value) for value in data.values())).numpy()
        reference.numpy().tofile(args.out/f"reference-{position}.f32")
        start = time.monotonic()
        actual = pair.predict(data, state=pair_state)["hidden_out"]
        pair_ms = 1000*(time.monotonic()-start)
        single_data = data.copy()
        start = time.monotonic()
        for model, state in singles:
            single_data["hidden"] = model.predict(single_data, state=state)["hidden_out"]
        single_ms = 1000*(time.monotonic()-start)
        case = {"position": position, "pair_ms": pair_ms, "separate_ms": single_ms,
                "eager_vs_dense": metrics(expected, eager), "pair_vs_dense": metrics(expected, actual),
                "separate_vs_dense": metrics(expected, single_data["hidden"]),
                "pair_vs_separate": metrics(single_data["hidden"], actual)}
        cases.append(case)
        print(json.dumps(case), flush=True)
    report = {"original_context": manifest["context"], "active_cache_positions": capacity,
              "first_original_block": args.first, "compute_units": "CPU_AND_NE", "cases": cases,
              "passed": all(case[name]["passed"] for case in cases for name in ("eager_vs_dense", "pair_vs_dense")),
              "scope": "Two original LM blocks only; not a text-to-audio generation benchmark"}
    (args.out/"parity.json").write_text(json.dumps(report, indent=2)+"\n")
    return report["passed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--individual", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--positions", type=int, required=True)
    parser.add_argument("--unconditional-token", type=int, required=True)
    args = parser.parse_args()
    if args.first < 0 or args.first % 2:
        parser.error("The first original block must be a nonnegative even index")
    torch.set_num_threads(8)
    raise SystemExit(0 if run(args) else 1)


if __name__ == "__main__":
    main()
