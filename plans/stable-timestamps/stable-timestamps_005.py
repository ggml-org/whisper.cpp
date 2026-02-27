# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.4.2",
#   "piper-onnx==1.0.6",
#   "soundfile==0.13.1",
# ]
# ///

"""
Setup
    uv pip install piper-onnx soundfile

Prepare models
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json

Run
    uv run stable-timestamps_005.py
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from piper_onnx import Piper


PHRASES = [
    "This is a stable timestamps verification sample.",
    "We are generating synthetic speech for timestamp quality checks.",
    "Whisper should align words near the actual spoken audio.",
    "This sentence is intentionally short.",
    "Now we add another line with slightly different pacing.",
    "Silence regions are useful for validating timestamp behavior.",
    "A longer pause may appear occasionally in this generated file.",
    "Please verify that boundaries snap away from silence.",
    "The quick brown fox jumps over the lazy dog.",
    "Agent based validation benefits from repeatable synthetic inputs.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic TTS audio with pauses for stable timestamp verification"
    )
    parser.add_argument("--model", default="en_US-ryan-medium.onnx", help="Path to Piper ONNX model")
    parser.add_argument("--config", default="en_US-ryan-medium.onnx.json", help="Path to Piper model config JSON")
    parser.add_argument("--output", default="stable_timestamps_5min.wav", help="Output WAV path")
    parser.add_argument(
        "--timeline-csv",
        default="",
        help="Output timeline CSV path (default: <output_stem>_timeline.csv)",
    )
    parser.add_argument(
        "--reference-txt",
        default="",
        help="Output reference transcript path (default: <output_stem>_reference.txt)",
    )
    parser.add_argument("--target-seconds", type=float, default=300.0, help="Target duration in seconds")
    parser.add_argument("--short-pause-min", type=float, default=0.5, help="Minimum normal pause in seconds")
    parser.add_argument("--short-pause-max", type=float, default=1.5, help="Maximum normal pause in seconds")
    parser.add_argument("--long-pause-seconds", type=float, default=20.0, help="Long pause duration in seconds")
    parser.add_argument(
        "--long-pause-prob",
        type=float,
        default=0.08,
        help="Probability of using long pause after an utterance",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser.parse_args()


def write_timeline_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "type",
        "start_sample",
        "end_sample",
        "start_s",
        "end_s",
        "duration_s",
        "text",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    out_path = Path(args.output)
    if args.timeline_csv:
        timeline_path = Path(args.timeline_csv)
    else:
        timeline_path = out_path.with_name(f"{out_path.stem}_timeline.csv")
    if args.reference_txt:
        reference_path = Path(args.reference_txt)
    else:
        reference_path = out_path.with_name(f"{out_path.stem}_reference.txt")

    piper = Piper(args.model, args.config)

    chunks: list[np.ndarray] = []
    timeline_rows: list[dict[str, object]] = []

    sample_rate: int | None = None
    target_samples: int | None = None
    total_samples = 0
    utterance_count = 0
    long_pause_count = 0

    while True:
        if target_samples is not None and total_samples >= target_samples:
            break

        text = rng.choice(PHRASES)
        samples, sr = piper.create(text)

        if sample_rate is None:
            sample_rate = sr
            target_samples = int(round(args.target_seconds * sample_rate))
            if target_samples <= 0:
                raise ValueError("target-seconds must be positive")
        elif sr != sample_rate:
            raise RuntimeError(f"Sample rate changed unexpectedly: {sample_rate} -> {sr}")

        remaining = target_samples - total_samples
        take = min(samples.shape[0], remaining)

        if take > 0:
            start = total_samples
            end = total_samples + take
            chunks.append(samples[:take])
            total_samples = end
            utterance_count += 1

            timeline_rows.append(
                {
                    "index": len(timeline_rows),
                    "type": "speech",
                    "start_sample": start,
                    "end_sample": end,
                    "start_s": start / sample_rate,
                    "end_s": end / sample_rate,
                    "duration_s": (end - start) / sample_rate,
                    "text": text,
                }
            )

        if total_samples >= target_samples:
            break

        if rng.random() < args.long_pause_prob:
            pause_seconds = args.long_pause_seconds
            pause_type = "long_silence"
            long_pause_count += 1
        else:
            pause_seconds = rng.uniform(args.short_pause_min, args.short_pause_max)
            pause_type = "short_silence"

        pause_len = int(round(pause_seconds * sample_rate))
        pause_len = min(pause_len, target_samples - total_samples)

        if pause_len > 0:
            start = total_samples
            end = total_samples + pause_len
            chunks.append(np.zeros(pause_len, dtype=np.float32))
            total_samples = end

            timeline_rows.append(
                {
                    "index": len(timeline_rows),
                    "type": pause_type,
                    "start_sample": start,
                    "end_sample": end,
                    "start_s": start / sample_rate,
                    "end_s": end / sample_rate,
                    "duration_s": (end - start) / sample_rate,
                    "text": "",
                }
            )

    if not chunks:
        raise RuntimeError("No audio generated")

    audio = np.concatenate(chunks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sample_rate)
    write_timeline_csv(timeline_path, timeline_rows)
    reference_text = " ".join(
        str(row["text"]).strip()
        for row in timeline_rows
        if row["type"] == "speech" and str(row["text"]).strip()
    )
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(reference_text + "\n", encoding="utf-8")

    print(
        f"Created {out_path} | duration={audio.shape[0] / sample_rate:.2f}s "
        f"| utterances={utterance_count} | long_pauses={long_pause_count}"
    )
    print(f"Created {timeline_path}")
    print(f"Created {reference_path}")


if __name__ == "__main__":
    main()
