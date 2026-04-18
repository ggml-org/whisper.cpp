#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TIMING_PATTERNS = {
    "model_load_ms": re.compile(r"load time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
    "mel_ms": re.compile(r"mel time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
    "sample_ms": re.compile(r"sample time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
    "encode_ms": re.compile(r"encode time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
    "decode_ms": re.compile(r"decode time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
    "batchd_ms": re.compile(r"batchd time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
    "prompt_ms": re.compile(r"prompt time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
    "full_runtime_ms": re.compile(r"total time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
}

TOKEN_RATE_PATTERNS = [
    re.compile(r"tokens?\s*/\s*s(?:ec(?:ond)?)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"tokens?\s+per\s+second\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
]

SEGMENT_LINE_PATTERN = re.compile(
    r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}\]\s*(.*)$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse benchmark logs and compute run statistics.")
    p.add_argument("--run-dir", required=True, help="Path to benchmark results run directory.")
    p.add_argument(
        "--refs-dir",
        default=None,
        help="Directory containing reference transcripts {short,medium,long}.txt (default: benchmark/references).",
    )
    p.add_argument("--max-wer", type=float, default=0.02, help="Max allowed median WER per audio.")
    p.add_argument("--max-cer", type=float, default=0.02, help="Max allowed median CER per audio.")
    p.add_argument(
        "--enforce-correctness",
        action="store_true",
        help="Exit non-zero if any referenced audio exceeds max WER/CER or if references are missing.",
    )
    return p.parse_args()


def to_float_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def parse_log_metrics(log_text: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in TIMING_PATTERNS.keys()}

    for key, pat in TIMING_PATTERNS.items():
        m = pat.search(log_text)
        out[key] = float(m.group(1)) if m else None

    token_rate = None
    for pat in TOKEN_RATE_PATTERNS:
        m = pat.search(log_text)
        if m:
            token_rate = float(m.group(1))
            break
    out["tokens_per_second"] = token_rate

    return out


def extract_transcript(log_text: str) -> str:
    lines: List[str] = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        m = SEGMENT_LINE_PATTERN.match(line)
        if not m:
            continue
        text = m.group(1).strip()
        if text:
            lines.append(text)
    return " ".join(lines).strip()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    prev[j] + 1,      # delete
                    curr[j - 1] + 1,  # insert
                    prev[j - 1] + cost,  # substitute
                )
            )
        prev = curr
    return prev[-1]


def error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    # Exact DP is expensive for long transcripts. Use exact distance for small/medium
    # inputs and a fast similarity-based fallback for long inputs.
    if len(reference) * max(1, len(hypothesis)) <= 2_000_000:
        dist = levenshtein_distance(reference, hypothesis)
        return dist / float(len(reference))

    ratio = SequenceMatcher(a=list(reference), b=list(hypothesis), autojunk=False).ratio()
    return max(0.0, min(1.0, 1.0 - ratio))


def compute_wer_cer(reference_text: str, hypothesis_text: str) -> Tuple[float, float]:
    ref_norm = normalize_text(reference_text)
    hyp_norm = normalize_text(hypothesis_text)

    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []
    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))

    wer = error_rate(ref_words, hyp_words)
    cer = error_rate(ref_chars, hyp_chars)

    return wer, cer


def safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return statistics.mean(vals) if vals else None


def safe_median(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return statistics.median(vals) if vals else None


def safe_min(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return min(vals) if vals else None


def safe_max(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return max(vals) if vals else None


def safe_stdev(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return statistics.stdev(vals)


def stats_block(values: Iterable[float]) -> Dict[str, Optional[float]]:
    vals = list(values)
    return {
        "mean": safe_mean(vals),
        "median": safe_median(vals),
        "min": safe_min(vals),
        "max": safe_max(vals),
        "std_dev": safe_stdev(vals),
    }


def fmt(value: Optional[float], suffix: str = "", decimals: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{decimals}f}{suffix}"


def write_runs_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "variant",
        "model",
        "audio_key",
        "audio_length_s",
        "run_kind",
        "run_index",
        "wall_clock_runtime_s",
        "tokens_per_second",
        "audio_seconds_per_second",
        "model_load_ms",
        "first_inference_latency_s",
        "full_runtime_ms",
        "mel_ms",
        "sample_ms",
        "encode_ms",
        "decode_ms",
        "batchd_ms",
        "prompt_ms",
        "transcript_word_error_rate",
        "transcript_char_error_rate",
        "reference_present",
        "reference_path",
        "transcript_path",
        "metal_kernel_runtime_ms",
        "cpu_orchestration_ms",
        "log_path",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    if args.refs_dir:
        refs_dir = Path(args.refs_dir).resolve()
    else:
        refs_dir = run_dir.parent.parent / "references"

    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Missing config file: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    variant = config["variant"]
    model = Path(config["model"]["rel_path"]).name

    run_meta_paths = sorted((run_dir / "raw").glob("*/*run_*.meta.json"))
    if not run_meta_paths:
        raise SystemExit(f"No measured run metadata files found under {(run_dir / 'raw')}")

    reference_texts: Dict[str, Optional[str]] = {}
    reference_paths: Dict[str, Path] = {}
    missing_references: List[str] = []
    for audio_key in ("short", "medium", "long"):
        ref_path = refs_dir / f"{audio_key}.txt"
        reference_paths[audio_key] = ref_path
        if ref_path.is_file():
            reference_texts[audio_key] = ref_path.read_text(encoding="utf-8", errors="replace").strip()
        else:
            reference_texts[audio_key] = None
            missing_references.append(audio_key)

    run_rows: List[Dict[str, object]] = []
    for meta_path in run_meta_paths:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("run_kind") != "measured":
            continue

        log_path = Path(meta["log_path"])
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_log_metrics(log_text)
        transcript_text = extract_transcript(log_text)
        transcript_path = log_path.with_suffix(".transcript.txt")
        transcript_path.write_text(transcript_text + "\n", encoding="utf-8")

        audio_key = str(meta["audio_key"])
        ref_text = reference_texts.get(audio_key)
        ref_present = ref_text is not None
        wer = None
        cer = None
        if ref_present:
            wer, cer = compute_wer_cer(ref_text, transcript_text)

        wall_s = float(meta["wall_clock_runtime_s"])
        audio_len_s = float(meta["audio_duration_s"])
        audio_s_per_s = audio_len_s / wall_s if wall_s > 0 else None
        full_runtime_ms = parsed["full_runtime_ms"]
        if full_runtime_ms is None:
            full_runtime_ms = wall_s * 1000.0

        run_rows.append(
            {
                "variant": variant,
                "model": model,
                "audio_key": audio_key,
                "audio_length_s": audio_len_s,
                "run_kind": meta["run_kind"],
                "run_index": int(meta["run_index"]),
                "wall_clock_runtime_s": wall_s,
                "tokens_per_second": to_float_or_none(parsed["tokens_per_second"]),
                "audio_seconds_per_second": to_float_or_none(audio_s_per_s),
                "model_load_ms": to_float_or_none(parsed["model_load_ms"]),
                "first_inference_latency_s": to_float_or_none(meta.get("first_inference_latency_s")),
                "full_runtime_ms": to_float_or_none(full_runtime_ms),
                "mel_ms": to_float_or_none(parsed["mel_ms"]),
                "sample_ms": to_float_or_none(parsed["sample_ms"]),
                "encode_ms": to_float_or_none(parsed["encode_ms"]),
                "decode_ms": to_float_or_none(parsed["decode_ms"]),
                "batchd_ms": to_float_or_none(parsed["batchd_ms"]),
                "prompt_ms": to_float_or_none(parsed["prompt_ms"]),
                "transcript_word_error_rate": to_float_or_none(wer),
                "transcript_char_error_rate": to_float_or_none(cer),
                "reference_present": ref_present,
                "reference_path": str(reference_paths[audio_key]),
                "transcript_path": str(transcript_path),
                "metal_kernel_runtime_ms": None,
                "cpu_orchestration_ms": None,
                "log_path": str(log_path),
            }
        )

    if not run_rows:
        raise SystemExit("No measured runs were parsed.")

    runs_csv_path = run_dir / "runs.csv"
    write_runs_csv(runs_csv_path, run_rows)

    by_audio: Dict[str, List[Dict[str, object]]] = {}
    for row in run_rows:
        by_audio.setdefault(str(row["audio_key"]), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    md_lines = [
        "| Variant | Model | Audio Length | Runs | Init Mean | First Inference Mean | Runtime Median | Throughput | Std Dev | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    overall_correctness_pass = True

    for audio_key in ("short", "medium", "long"):
        rows = sorted(by_audio.get(audio_key, []), key=lambda r: int(r["run_index"]))
        if not rows:
            continue

        audio_length_s = float(rows[0]["audio_length_s"])

        wall_values = [float(r["wall_clock_runtime_s"]) for r in rows if r["wall_clock_runtime_s"] is not None]
        load_values = [float(r["model_load_ms"]) for r in rows if r["model_load_ms"] is not None]
        first_values = [float(r["first_inference_latency_s"]) for r in rows if r["first_inference_latency_s"] is not None]
        throughput_values = [float(r["audio_seconds_per_second"]) for r in rows if r["audio_seconds_per_second"] is not None]
        token_values = [float(r["tokens_per_second"]) for r in rows if r["tokens_per_second"] is not None]
        full_runtime_values = [float(r["full_runtime_ms"]) for r in rows if r["full_runtime_ms"] is not None]
        encode_values = [float(r["encode_ms"]) for r in rows if r["encode_ms"] is not None]
        decode_values = [float(r["decode_ms"]) for r in rows if r["decode_ms"] is not None]
        wer_values = [float(r["transcript_word_error_rate"]) for r in rows if r["transcript_word_error_rate"] is not None]
        cer_values = [float(r["transcript_char_error_rate"]) for r in rows if r["transcript_char_error_rate"] is not None]
        reference_present = bool(rows[0]["reference_present"])

        runtime_stats = stats_block(wall_values)
        full_runtime_stats = stats_block(full_runtime_values)
        load_stats = stats_block(load_values)
        first_stats = stats_block(first_values)
        throughput_stats = stats_block(throughput_values)
        token_stats = stats_block(token_values)
        encode_stats = stats_block(encode_values)
        decode_stats = stats_block(decode_values)
        wer_stats = stats_block(wer_values)
        cer_stats = stats_block(cer_values)

        notes_parts: List[str] = []
        if token_stats["mean"] is None:
            notes_parts.append("tokens/s unavailable")
        else:
            notes_parts.append(f"tokens/s mean={token_stats['mean']:.3f}")
        if encode_stats["mean"] is not None:
            notes_parts.append(f"encode mean={encode_stats['mean']:.2f} ms")
        if decode_stats["mean"] is not None:
            notes_parts.append(f"decode mean={decode_stats['mean']:.2f} ms")
        if reference_present:
            notes_parts.append(f"wer median={fmt(wer_stats['median'], '', 4)}")
            notes_parts.append(f"cer median={fmt(cer_stats['median'], '', 4)}")
        else:
            notes_parts.append("reference missing")

        correctness_pass: Optional[bool]
        if not reference_present:
            correctness_pass = False if args.enforce_correctness else None
        else:
            correctness_pass = (
                wer_stats["median"] is not None
                and cer_stats["median"] is not None
                and wer_stats["median"] <= args.max_wer
                and cer_stats["median"] <= args.max_cer
            )
        if correctness_pass is False:
            overall_correctness_pass = False

        notes = "; ".join(notes_parts)

        summary_rows.append(
            {
                "variant": variant,
                "model": model,
                "audio_key": audio_key,
                "audio_length_s": audio_length_s,
                "runs": len(rows),
                "model_load_ms": load_stats,
                "first_inference_latency_s": first_stats,
                "wall_clock_runtime_s": runtime_stats,
                "full_runtime_ms": full_runtime_stats,
                "throughput_audio_seconds_per_second": throughput_stats,
                "tokens_per_second": token_stats,
                "encode_ms": encode_stats,
                "decode_ms": decode_stats,
                "wer": wer_stats,
                "cer": cer_stats,
                "reference_present": reference_present,
                "correctness_pass": correctness_pass,
                "notes": notes,
            }
        )

        md_lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    model,
                    f"{audio_length_s:.3f}s",
                    str(len(rows)),
                    fmt(load_stats["mean"], " ms", decimals=2),
                    fmt(first_stats["mean"], " s", decimals=3),
                    fmt(runtime_stats["median"], " s", decimals=3),
                    fmt(throughput_stats["mean"], " audio-s/s", decimals=3),
                    fmt(runtime_stats["std_dev"], " s", decimals=3),
                    notes,
                ]
            )
            + " |"
        )

    summary_csv_path = run_dir / "summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "variant",
            "model",
            "audio_key",
            "audio_length_s",
            "runs",
            "init_mean_ms",
            "first_inference_mean_s",
            "runtime_median_s",
            "throughput_mean_audio_s_per_s",
            "runtime_std_dev_s",
            "reference_present",
            "wer_median",
            "cer_median",
            "max_wer",
            "max_cer",
            "correctness_pass",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "variant": row["variant"],
                    "model": row["model"],
                    "audio_key": row["audio_key"],
                    "audio_length_s": row["audio_length_s"],
                    "runs": row["runs"],
                    "init_mean_ms": row["model_load_ms"]["mean"],
                    "first_inference_mean_s": row["first_inference_latency_s"]["mean"],
                    "runtime_median_s": row["wall_clock_runtime_s"]["median"],
                    "throughput_mean_audio_s_per_s": row["throughput_audio_seconds_per_second"]["mean"],
                    "runtime_std_dev_s": row["wall_clock_runtime_s"]["std_dev"],
                    "reference_present": row["reference_present"],
                    "wer_median": row["wer"]["median"],
                    "cer_median": row["cer"]["median"],
                    "max_wer": args.max_wer,
                    "max_cer": args.max_cer,
                    "correctness_pass": row["correctness_pass"],
                    "notes": row["notes"],
                }
            )

    summary_json_path = run_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")

    correctness_json_path = run_dir / "correctness.json"
    correctness_report = {
        "refs_dir": str(refs_dir),
        "max_wer": args.max_wer,
        "max_cer": args.max_cer,
        "enforce_correctness": args.enforce_correctness,
        "missing_references": missing_references,
        "overall_correctness_pass": overall_correctness_pass and (
            not args.enforce_correctness or len(missing_references) == 0
        ),
        "audios": [
            {
                "audio_key": row["audio_key"],
                "reference_present": row["reference_present"],
                "wer_median": row["wer"]["median"],
                "cer_median": row["cer"]["median"],
                "correctness_pass": row["correctness_pass"],
            }
            for row in summary_rows
        ],
    }
    correctness_json_path.write_text(json.dumps(correctness_report, indent=2) + "\n", encoding="utf-8")

    summary_md_path = run_dir / "summary.md"
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {runs_csv_path}")
    print(f"Wrote: {summary_csv_path}")
    print(f"Wrote: {summary_json_path}")
    print(f"Wrote: {summary_md_path}")
    print(f"Wrote: {correctness_json_path}")

    if args.enforce_correctness:
        if missing_references:
            print(
                "Correctness gate failed: missing references for "
                + ", ".join(sorted(missing_references))
            )
            return 3
        if not overall_correctness_pass:
            print(
                f"Correctness gate failed: one or more audios exceeded max WER={args.max_wer} or max CER={args.max_cer}"
            )
            return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
