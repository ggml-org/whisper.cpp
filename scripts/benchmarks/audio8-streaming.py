import argparse
import json
import math
import re
import statistics
import subprocess
import wave
from pathlib import Path

MAX_RUNS = 20
FRAME_LIMIT = 512
THREADS = 4
RUN_TIMEOUT_SECONDS = 1200
TEXT = (
    "When we build a voice assistant, the first sound should arrive quickly, even when the answer is long. "
    "The listener can begin following the explanation while the remaining sentences are still being generated. "
    "This experiment measures how quickly audio becomes available and how long the complete response takes on this computer."
)
METRICS = ("first_audio_ms", "total_ms", "audio_s", "realtime_factor", "chunks")
PERFORMANCE = re.compile(r"^\[audio8-performance\] (.+)$", re.MULTILINE)


def parse_metrics(log, mode):
    matches = PERFORMANCE.findall(log)
    if len(matches) != 1:
        raise ValueError("expected exactly one audio8-performance record")
    fields = dict(item.split("=", 1) for item in matches[0].split())
    if fields.get("mode") != mode:
        raise ValueError("performance mode mismatch")
    values = {key: float(fields[key]) for key in METRICS}
    if not all(math.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError("invalid performance values")
    if values["audio_s"] <= 0 or values["total_ms"] <= 0:
        raise ValueError("empty synthesis")
    if values["first_audio_ms"] > values["total_ms"]:
        raise ValueError("first audio arrived after completion")
    if mode == "stream" and values["chunks"] < 1:
        raise ValueError("streaming emitted no callbacks")
    return values


def command(opts, mode, name):
    args = [
        str(opts.binary), "--lm", str(opts.lm), "--codec-decoder", str(opts.decoder),
        "--text", TEXT, "--greedy", "--seed", "42", "--threads", str(THREADS),
        "--backend", opts.backend, "--n-gpu-layers", str(opts.gpu_layers),
        "--max-frames", str(FRAME_LIMIT), "--verbose",
        "--out", str(opts.out_dir / (name + ".wav")),
        "--dump-codes", str(opts.out_dir / (name + ".codes.csv")),
    ]
    return args + (["--stream"] if mode == "stream" else [])


def run_one(opts, mode, index):
    name = f"{index:02d}-{mode}"
    args = command(opts, mode, name)
    log_path = opts.out_dir / (name + ".log")
    with log_path.open("w") as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_SECONDS)
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {log_path}")
    metrics = parse_metrics(log_path.read_text(), mode)
    with wave.open(str(opts.out_dir / (name + ".wav")), "rb") as audio:
        samples = audio.getnframes()
        rate = audio.getframerate()
        if audio.getnchannels() != 1 or samples == 0:
            raise ValueError("expected nonempty mono PCM")
    if abs(samples / rate - metrics["audio_s"]) > 0.001:
        raise ValueError("reported audio duration differs from WAV")
    return {"name": name, "mode": mode, "command": args, "metrics": metrics,
            "samples": samples, "sample_rate": rate}


def validate_pair(opts, pair):
    left, right = pair
    if (left["samples"], left["sample_rate"]) != (right["samples"], right["sample_rate"]):
        raise ValueError("batch and streaming sample geometry differs")
    codes = [(opts.out_dir / (run["name"] + ".codes.csv")).read_bytes() for run in pair]
    if not codes[0] or codes[0] != codes[1]:
        raise ValueError("batch and streaming generated codes differ")


def medians(runs, mode):
    selected = [run["metrics"] for run in runs if run["mode"] == mode]
    return {key: statistics.median(row[key] for row in selected) for key in METRICS}


def save_report(opts, report):
    (opts.out_dir / "latency.json").write_text(json.dumps(report, indent=2) + "\n")


def run_pairs(opts, report):
    for index in range(opts.runs):
        order = ("batch", "stream") if index % 2 == 0 else ("stream", "batch")
        pair = []
        for mode in order:
            run = run_one(opts, mode, len(report["runs"]) + 1)
            pair.append(run)
            report["runs"].append(run)
            save_report(opts, report)
        validate_pair(opts, pair)


def benchmark(opts):
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"text": TEXT, "runs": [], "threads": THREADS, "frame_limit": FRAME_LIMIT,
              "timing_scope": "fresh engine; excludes loading and WAV writing; no inference warmup"}
    run_pairs(opts, report)
    report["medians"] = {mode: medians(report["runs"], mode) for mode in ("batch", "stream")}
    save_report(opts, report)
    return report


def print_summary(report):
    print("\n### Audio8 incremental PCM\n")
    print("Medians; model loading and WAV writing excluded. Native CLI, not SDK/duplex.\n")
    print("| Mode | First audio (s) | Total (s) | Audio (s) | RTF | Chunks |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for mode, values in report["medians"].items():
        print(f"| {mode} | {values['first_audio_ms'] / 1000:.3f} | {values['total_ms'] / 1000:.3f} "
              f"| {values['audio_s']:.3f} | {values['realtime_factor']:.3f} | {values['chunks']:.0f} |")


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--lm", required=True, type=Path)
    parser.add_argument("--decoder", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--gpu-layers", type=int, default=0)
    opts = parser.parse_args()
    if not 1 <= opts.runs <= MAX_RUNS:
        parser.error(f"--runs must be between 1 and {MAX_RUNS}")
    return opts


if __name__ == "__main__":
    print_summary(benchmark(options()))
