#!/usr/bin/env python3
"""Teacher-forced decoder quality and inclusive adapter timing; numpy only.

Fixture manifests (schema_version=1) have model metadata, token_config
{bos,eos,pad,codebook_size,max_length}, and cases [{id, files: {...}}]. Files
are relative to the manifest: reference_logits [steps,codebooks,vocab],
decoder_inputs [steps,codebooks] (exact delayed frames actually fed),
delayed_tokens [codebooks,steps+1] (sampled history, before forced delay),
prompt_ids [1,P], cross_states [1,T,D]. Generation can override max_length
per case. No argmax trace substitutes for an fp32 logits reference.

  validate-decoder-coreml.py run --manifest fixtures/manifest.json \
      --adapter /absolute/path/native_adapter.py --out /tmp/candidate
  validate-decoder-coreml.py compare --manifest fixtures/manifest.json \
      --candidate /tmp/candidate/run.json --out /tmp/quality.json

Adapter module: create_runner(manifest[, config]) -> runner. Optional
prepare_case(case,arrays) runs description/T5/cross projection OUTSIDE the
decoder timer (its own prepare_ms is recorded). start_case(case,arrays)
starts a fresh self-KV session; step(index,frame)
returns RAW [codebooks,vocab] logits (index 0 performs prefill); synchronize()
waits for ALL transfers/state updates; finish_case() releases case resources.
All four methods are mandatory. Conversion/loading belongs in create_runner.
Input preparation, prediction, output materialization, and KV update must be
inside these calls; never defer KV writes to the next step or finish_case.
The harness times start/finish and synchronization and copies returned logits
within each timed step. This measures decoder host wall time, excluding
sampling, T5, DAC, disk I/O, and model load; it is not an end-to-end TTS claim.
Same runner is reused across cases/repetitions to expose reset/cache leaks.
Optional close() runs even on an exception. metadata() records native backend
and model identities; adapters must disclose any difference from HF cross states.

No built-in thresholds bless a quantization. Optional --fp16 and --q8 runs
enable a strict baseline gate: each case/codebook must match/exceed BOTH baselines in
live argmax/top-50 overlap and match/beat BOTH in mean/p95 KL; no invalid
rows, no special-row argmax disagreements, and no repeat drift are allowed.
Speed promotion additionally requires --cpu and every case's inclusive mean
AND p95 to beat fp16, q8 and CPU. Missing baselines/timing never imply promotion.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or not data.get("cases"):
        raise ValueError("expected schema_version=1 and nonempty cases")
    ids = [c["id"] for c in data["cases"]]
    if len(set(ids)) != len(ids) or any(not i or Path(i).name != i or i in (".", "..") for i in ids):
        raise ValueError("case ids must be unique filename components")
    # Include actual array bytes, not only manifest paths, in the history ID.
    h = hashlib.sha256(path.read_bytes())
    for case in data["cases"]:
        for key, filename in sorted(case["files"].items()):
            file = path.parent / filename
            actual = sha256(file)
            expected = case.get("sha256", {}).get(key)
            if expected is not None and expected != actual:
                raise ValueError(f"fixture checksum mismatch: {case['id']}/{key}")
            h.update(f"{case['id']}/{key}:{actual}".encode())
    return data, path.parent, h.hexdigest()


def row_metrics(reference, candidate, top_k=50):
    """KL(P_fp32 || Q_candidate), in nats, without probability clipping.

    Negative infinity is valid zero probability if a row has finite support.
    NaN, positive infinity and empty support invalidate a row. A missing Q
    support term with positive P gives infinite KL (never silently omitted).
    Stable sorting makes tied top-k sets deterministic by token index.
    """
    p, q = np.asarray(reference, np.float64), np.asarray(candidate, np.float64)
    if p.ndim != 1 or p.shape != q.shape or p.size == 0 or top_k < 1:
        raise ValueError("expected matching nonempty logit vectors and positive top_k")
    bad = []
    for name, a in (("reference", p), ("candidate", q)):
        if np.isnan(a).any() or np.isposinf(a).any() or not np.isfinite(a).any():
            bad.append(name)
    if bad:
        return {"valid": False, "invalid": bad, "argmax_agreement": 0.0,
                "top50_overlap": 0.0, "kl": math.inf, "max_abs_logit_error": math.inf}

    def log_softmax(a):
        shifted = a - np.max(a)
        return shifted - np.log(np.exp(shifted).sum())

    lp, lq = log_softmax(p), log_softmax(q)
    support = np.isfinite(lp)
    if np.isneginf(lq[support]).any():
        kl = math.inf
    else:
        kl = max(0.0, float(np.sum(np.exp(lp[support]) * (lp[support] - lq[support]))))
    k = min(top_k, p.size)
    ptop = np.argsort(-p, kind="stable")[:k]
    qtop = np.argsort(-q, kind="stable")[:k]
    same_mask = np.isneginf(p) & np.isneginf(q)
    delta = np.abs(p[~same_mask] - q[~same_mask])
    return {"valid": True, "invalid": [], "argmax_agreement": float(np.argmax(p) == np.argmax(q)),
            "top50_overlap": len(np.intersect1d(ptop, qtop)) / k, "kl": kl,
            "max_abs_logit_error": float(delta.max()) if delta.size else 0.0}


def classify_slots(tokens, config, steps):
    """Classify predicted column s+1; BOS/input column zero is not scored."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or tokens.shape[1] < steps + 1:
        raise ValueError("delayed_tokens must include BOS plus every predicted column")
    nq = tokens.shape[0]
    limit = config["max_length"]
    if steps + 1 > limit or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("invalid delayed token length or dtype")
    classes = np.full((steps, nq), "audio", dtype="U16")
    for k in range(nq):
        done = False
        for s in range(steps):
            col, token = s + 1, int(tokens[k, s + 1])
            forced_delay = limit >= 2 * nq - 1 and col <= k
            forced_pad = limit >= 2 * nq - 1 and col >= limit - nq + 1 + k
            if forced_delay:
                label = "delay"
            elif forced_pad or done:
                label = "padding"
            elif token == config["eos"]:
                label = "eos"
            elif token == config["pad"]:
                label = "padding"
            elif token < 0 or token >= config["codebook_size"]:
                label = "special"
            else:
                label = "audio"
            classes[s, k] = label
            # Forced delay/pad slots are model predictions overwritten before
            # feeding. They must not create a false EOS boundary in metrics.
            if not forced_delay and not forced_pad and token == config["eos"]:
                done = True
    return classes


def summarize(rows):
    if not rows:
        return {"rows": 0, "invalid_rows": 0, "argmax_agreement": None,
                "top50_overlap": None, "kl_mean": None, "kl_p95": None, "infinite_kl_rows": 0}
    kl = [r["kl"] for r in rows]
    # nearest-rank p95 handles infinite tails without inf-inf interpolation.
    return {"rows": len(rows), "invalid_rows": sum(not r["valid"] for r in rows),
            "argmax_agreement": float(np.mean([r["argmax_agreement"] for r in rows])),
            "top50_overlap": float(np.mean([r["top50_overlap"] for r in rows])),
            "kl_mean": float(np.mean(kl)),
            "kl_p95": float(np.sort(kl)[math.ceil(.95 * len(kl)) - 1]),
            "infinite_kl_rows": sum(not math.isfinite(x) for x in kl)}


def compare_case(reference, candidate, classes):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.ndim != 3 or not all(reference.shape) or reference.shape != candidate.shape or classes.shape != reference.shape[:2]:
        raise ValueError(f"logit/class shape mismatch: {reference.shape}, {candidate.shape}, {classes.shape}")
    rows, steps = [], []
    for s in range(reference.shape[0]):
        current = []
        for k in range(reference.shape[1]):
            row = dict(row_metrics(reference[s, k], candidate[s, k]), step=s, codebook=k,
                       slot_class=str(classes[s, k]))
            rows.append(row)
            current.append(row)
        steps.append({"step": s, "all": summarize(current),
                      "audio": summarize([r for r in current if r["slot_class"] == "audio"]),
                      "codebooks": current})
    return {"all": summarize(rows),
            "by_class": {c: summarize([r for r in rows if r["slot_class"] == c])
                         for c in ("audio", "delay", "padding", "eos", "special")},
            "by_codebook": {str(k): summarize([r for r in rows if r["codebook"] == k])
                            for k in range(reference.shape[1])},
            "audio_by_codebook": {str(k): summarize([r for r in rows if r["codebook"] == k and r["slot_class"] == "audio"])
                                  for k in range(reference.shape[1])}, "steps": steps}


def time_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("timings must be finite nonnegative wall-clock milliseconds")
    mean = float(values.mean())
    if not math.isfinite(mean):
        raise ValueError("timing aggregate overflow")
    return {"count": int(values.size), "mean_ms": mean,
            "p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def arrays_for(case, root):
    return {key: np.load(root / value, allow_pickle=False) for key, value in case["files"].items()}


def run_adapter(manifest_path, adapter_path, output, repeats=3, warmup=1, config=None):
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats >= 1 and warmup >= 0 required")
    manifest, root, digest = load_manifest(manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("parler_candidate_adapter", Path(adapter_path).resolve())
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    loaded = [(case, arrays_for(case, root)) for case in manifest["cases"]]
    t0 = time.perf_counter_ns()
    runner = adapter.create_runner(manifest) if config is None else adapter.create_runner(manifest, config)
    load_ms = (time.perf_counter_ns() - t0) / 1e6
    for method in ("start_case", "step", "synchronize", "finish_case"):
        if not callable(getattr(runner, method, None)):
            if callable(getattr(runner, "close", None)):
                runner.close()
            raise ValueError(f"adapter must implement {method}")
    result = {"schema_version": 1, "fixture_sha256": digest, "adapter": str(Path(adapter_path).resolve()),
              "adapter_sha256": sha256(adapter_path), "load_ms": load_ms,
              "timing_scope": "inclusive_decoder_host_wall", "warmup": warmup, "repeats": repeats,
              "environment": {"platform": platform.platform(), "machine": platform.machine(),
                              "python": sys.version, "numpy": np.__version__, "cpu_count": os.cpu_count()},
              "cases": []}
    result["adapter_config"] = config
    result["has_separate_preparation"] = callable(getattr(runner, "prepare_case", None))
    stored = {}
    try:
        for iteration in range(-warmup, repeats):
            for case, arrays in loaded:
                frames = arrays["decoder_inputs"]
                if frames.ndim != 2 or frames.shape[0] == 0 or not np.issubdtype(frames.dtype, np.integer):
                    raise ValueError("decoder_inputs must be nonempty integer [steps,codebooks]")
                prepare_ms = 0.0
                if result["has_separate_preparation"]:
                    prepare_start = time.perf_counter_ns()
                    runner.prepare_case(case, arrays)
                    runner.synchronize()
                    prepare_ms = (time.perf_counter_ns() - prepare_start) / 1e6
                total_start = time.perf_counter_ns()
                runner.start_case(case, arrays)
                runner.synchronize()
                logits, step_ms = [], []
                for index, frame in enumerate(frames):
                    start = time.perf_counter_ns()
                    value = runner.step(index, frame)
                    runner.synchronize()
                    value = np.array(value, dtype=np.float32, copy=True)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    if value.ndim != 2 or value.shape[0] != frames.shape[1]:
                        raise ValueError("adapter must return [codebooks,vocab] raw logits")
                    logits.append(value)
                    step_ms.append(elapsed)
                runner.finish_case()
                runner.synchronize()
                total_ms = (time.perf_counter_ns() - total_start) / 1e6
                if iteration < 0:
                    continue
                value = np.stack(logits)
                if case["id"] not in stored:
                    file = f"{case['id']}_logits.npy"
                    np.save(output / file, value)
                    entry = {"id": case["id"], "logits": file, "logits_sha256": sha256(output / file),
                             "repeated_logit_mismatches": 0, "timings": []}
                    stored[case["id"]] = (value, entry)
                    result["cases"].append(entry)
                first, entry = stored[case["id"]]
                if value.shape != first.shape:
                    raise ValueError("adapter output shape changed between repetitions")
                entry["repeated_logit_mismatches"] += int(np.count_nonzero(value != first))
                entry["timings"].append({"prepare_ms": prepare_ms, "total_ms": total_ms, "step_ms": step_ms})
        if callable(getattr(runner, "metadata", None)):
            result["runner_metadata"] = runner.metadata()
    finally:
        if callable(getattr(runner, "close", None)):
            runner.close()
    for _, entry in stored.values():
        entry["inclusive_cases"] = time_summary([t["total_ms"] for t in entry["timings"]])
        entry["excluded_preparation"] = time_summary([t["prepare_ms"] for t in entry["timings"]])
        entry["prefill"] = time_summary([t["step_ms"][0] for t in entry["timings"]])
        entry["decode_steps"] = time_summary([v for t in entry["timings"] for v in t["step_ms"][1:]])
    write_json(output / "run.json", result)
    return result


def evaluate_run(manifest, root, digest, run_path):
    path = Path(run_path).resolve()
    run = json.loads(path.read_text())
    if run.get("schema_version") != 1 or run.get("fixture_sha256") != digest:
        raise ValueError("candidate was not run with these exact fixtures/history")
    entries = {c["id"]: c for c in run["cases"]}
    if len(entries) != len(run["cases"]) or set(entries) != {c["id"] for c in manifest["cases"]}:
        raise ValueError("candidate case set must exactly match reference")
    results = {}
    for case in manifest["cases"]:
        arrays = arrays_for(case, root)
        if "reference_logits" not in arrays:
            raise ValueError("reference_logits missing: replay the history manifest with fp32 HF first")
        ref = arrays["reference_logits"]
        cfg = dict(manifest["token_config"], **case.get("token_config", {}))
        classes = classify_slots(arrays["delayed_tokens"], cfg, ref.shape[0])
        entry = entries[case["id"]]
        file = path.parent / entry["logits"]
        if entry.get("logits_sha256") != sha256(file):
            raise ValueError("missing or mismatched candidate logits checksum")
        metrics = compare_case(ref, np.load(file, allow_pickle=False), classes)
        if arrays["decoder_inputs"].shape != ref.shape[:2]:
            raise ValueError("reference steps must exactly match teacher-forced inputs")
        metrics["repeated_logit_mismatches"] = entry.get("repeated_logit_mismatches")
        # Recompute summaries from raw samples; do not trust a claimed mean.
        timings = entry.get("timings", [])
        if run.get("timing_scope") == "inclusive_decoder_host_wall" and timings:
            if len(timings) < 3:
                metrics["inclusive_cases"] = None
            else:
                for sample in timings:
                    if len(sample["step_ms"]) != ref.shape[0] or sample["total_ms"] < sum(sample["step_ms"]):
                        raise ValueError("invalid inclusive timing sample")
                    time_summary(sample["step_ms"])
                metrics["inclusive_cases"] = time_summary([t["total_ms"] for t in timings])
        else:
            metrics["inclusive_cases"] = None
        results[case["id"]] = metrics
    return results


def promotion_gate(candidate, baselines, cpu=None):
    reasons = []
    if set(baselines) != {"fp16", "q8"}:
        reasons.append("both fp16 and q8 baselines are required")
    if cpu is None:
        reasons.append("CPU speed baseline is required")
    for cid, result in candidate.items():
        if result["all"]["invalid_rows"] or result["all"]["infinite_kl_rows"]:
            reasons.append(f"{cid}: invalid/nonfinite distribution")
        if result["repeated_logit_mismatches"] != 0:
            reasons.append(f"{cid}: repeated-call stability not demonstrated")
        for label in ("delay", "padding", "eos", "special"):
            stats = result["by_class"][label]
            if stats["rows"] and stats["argmax_agreement"] != 1.0:
                reasons.append(f"{cid}: {label} raw argmax disagreement")
        live = result["by_class"]["audio"]
        if not live["rows"]:
            reasons.append(f"{cid}: no live audio rows")
            continue
        for name, baseline in baselines.items():
            base = baseline[cid]["by_class"]["audio"]
            if baseline[cid]["all"]["invalid_rows"] or baseline[cid]["all"]["infinite_kl_rows"]:
                reasons.append(f"{cid}: {name} baseline is invalid")
            if baseline[cid]["repeated_logit_mismatches"] != 0:
                reasons.append(f"{cid}: {name} baseline repeat stability not demonstrated")
            for metric in ("argmax_agreement", "top50_overlap"):
                if live[metric] < base[metric]:
                    reasons.append(f"{cid}: {metric} below {name}")
            for metric in ("kl_mean", "kl_p95"):
                if live[metric] > base[metric]:
                    reasons.append(f"{cid}: {metric} exceeds {name}")
            for k, stats in result["audio_by_codebook"].items():
                baseline_cb = baseline[cid]["audio_by_codebook"][k]
                if not stats["rows"]:
                    reasons.append(f"{cid}/codebook{k}: no live audio rows")
                    continue
                for metric in ("argmax_agreement", "top50_overlap"):
                    if stats[metric] < baseline_cb[metric]:
                        reasons.append(f"{cid}/codebook{k}: {metric} below {name}")
                for metric in ("kl_mean", "kl_p95"):
                    if stats[metric] > baseline_cb[metric]:
                        reasons.append(f"{cid}/codebook{k}: {metric} exceeds {name}")
            timing, bt = result["inclusive_cases"], baseline[cid]["inclusive_cases"]
            if timing is None or bt is None:
                reasons.append(f"{cid}: inclusive timing missing for candidate/{name} (need >=3 repeats)")
            elif any(timing[k] >= bt[k] for k in ("mean_ms", "p95_ms")):
                reasons.append(f"{cid}: inclusive mean/p95 does not beat {name}")
        if cpu is not None:
            if cpu[cid]["all"]["invalid_rows"] or cpu[cid]["all"]["infinite_kl_rows"] or cpu[cid]["repeated_logit_mismatches"] != 0:
                reasons.append(f"{cid}: CPU baseline is invalid or unstable")
            timing, bt = result["inclusive_cases"], cpu[cid]["inclusive_cases"]
            if timing is None or bt is None:
                reasons.append(f"{cid}: inclusive timing missing for candidate/CPU (need >=3 repeats)")
            elif any(timing[k] >= bt[k] for k in ("mean_ms", "p95_ms")):
                reasons.append(f"{cid}: inclusive mean/p95 does not beat CPU")
    return {"eligible_for_further_audio_validation": not reasons, "reasons": reasons,
            "default_enable_approved": False,
            "audio_validation": "not measured: free-running EOS/duration, listening, ASR, MOS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--adapter", required=True, type=Path)
    run.add_argument("--adapter-config", type=Path, help="JSON passed to create_runner(manifest,config)")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmup", type=int, default=1)
    compare = sub.add_parser("compare")
    compare.add_argument("--manifest", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--fp16", type=Path)
    compare.add_argument("--q8", type=Path)
    compare.add_argument("--cpu", type=Path, help="native CPU speed baseline on the same fixture manifest")
    compare.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run":
        config = json.loads(args.adapter_config.read_text()) if args.adapter_config else None
        run_adapter(args.manifest, args.adapter, args.out, args.repeats, args.warmup, config)
        print(args.out / "run.json")
    else:
        manifest, root, digest = load_manifest(args.manifest)
        candidate = evaluate_run(manifest, root, digest, args.candidate)
        baselines = {name: evaluate_run(manifest, root, digest, value)
                     for name, value in (("fp16", args.fp16), ("q8", args.q8)) if value is not None}
        cpu = evaluate_run(manifest, root, digest, args.cpu) if args.cpu else None
        gate = promotion_gate(candidate, baselines, cpu)
        write_json(args.out, {"schema_version": 1, "fixture_sha256": digest, "cases": candidate,
                              "baseline_cases": baselines, "cpu_baseline_cases": cpu, "gate": gate})
        print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
