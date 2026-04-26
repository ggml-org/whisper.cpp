#!/usr/bin/env python3
"""
Meeting transcription with per-speaker labels.

Pipeline:
  1. whisper.cpp (whisper-cli) -> timestamped transcript JSON
  2. pyannote-audio            -> speaker segments
  3. merge by timestamp overlap -> labelled output

Usage:
    python diarize.py -f meeting.wav -m large-v3 --hf-token hf_xxx

Requirements:
    pip install -r requirements-diarize.txt
    (HuggingFace token with pyannote/speaker-diarization-3.1 terms accepted)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from pyannote.audio import Pipeline


# ---------------------------------------------------------------------------
# Step 1: whisper.cpp transcription
# ---------------------------------------------------------------------------

def find_whisper_cli() -> str:
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    candidates = [
        repo_root / "build" / "bin" / "Release" / "whisper-cli.exe",  # Windows MSVC
        repo_root / "build" / "bin" / "whisper-cli.exe",               # Windows MinGW
        repo_root / "build" / "bin" / "whisper-cli",                   # Linux/Mac
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "whisper-cli not found. Build the project first:\n"
        "  cmake -B build && cmake --build build -j --config Release"
    )


def run_whisper(audio_path: str, model: str, language: str, threads: int) -> list:
    cli = find_whisper_cli()

    repo_root = Path(__file__).parent.parent.parent
    model_path = repo_root / "models" / f"ggml-{model}.bin"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Download with: bash models/download-ggml-model.sh {model}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_base = os.path.join(tmpdir, "out")
        cmd = [
            cli,
            "-m", str(model_path),
            "-f", audio_path,
            "-l", language,
            "-t", str(threads),
            "--output-json",
            "--output-file", out_base,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"whisper-cli failed:\n{result.stderr}")

        json_path = out_base + ".json"
        if not os.path.exists(json_path):
            raise RuntimeError(
                "whisper-cli did not produce a JSON file. "
                "stderr:\n" + result.stderr
            )

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

    return data.get("transcription", [])


# ---------------------------------------------------------------------------
# Step 2: pyannote-audio speaker diarization
# ---------------------------------------------------------------------------

def run_diarization(audio_path: str, hf_token: str, num_speakers: int | None) -> list:
    print("Loading pyannote speaker-diarization-3.1 (CPU) ...", file=sys.stderr)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    pipeline.to(torch.device("cpu"))

    # Use soundfile to avoid the torchcodec/FFmpeg dependency on Windows.
    # pyannote accepts a pre-loaded {'waveform': Tensor, 'sample_rate': int} dict.
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform_tensor = torch.from_numpy(waveform.T)  # (channels, time)
    audio_input = {"waveform": waveform_tensor, "sample_rate": sample_rate}

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    print("Running diarization ...", file=sys.stderr)
    result = pipeline(audio_input, **kwargs)

    # pyannote.audio 4.x returns DiarizeOutput; the Annotation is in .speaker_diarization
    annotation = result.speaker_diarization

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return segments


# ---------------------------------------------------------------------------
# Step 3: merge by timestamp overlap
# ---------------------------------------------------------------------------

def assign_speakers(transcription: list, diarization: list) -> list:
    results = []
    for seg in transcription:
        # whisper offsets are in milliseconds
        t0 = seg["offsets"]["from"] / 1000.0
        t1 = seg["offsets"]["to"] / 1000.0
        text = seg.get("text", "").strip()
        if not text:
            continue

        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for d in diarization:
            overlap = min(t1, d["end"]) - max(t0, d["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        results.append({"start": t0, "end": t1, "speaker": best_speaker, "text": text})
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_output(segments: list) -> str:
    lines = []
    for seg in segments:
        ts = f"[{_fmt_time(seg['start'])} --> {_fmt_time(seg['end'])}]"
        lines.append(f"{ts}  {seg['speaker']}: {seg['text']}")
    return "\n".join(lines)


def format_json(segments: list) -> str:
    return json.dumps(segments, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Whisper.cpp + pyannote diarization pipeline")
    p.add_argument("-f", "--file", required=True, help="Input WAV file (16 kHz, 16-bit)")
    p.add_argument("-m", "--model", default="large-v3", help="ggml model name (default: large-v3)")
    p.add_argument("-l", "--language", default="ja", help="Language code (default: ja)")
    p.add_argument("-t", "--threads", type=int, default=4, help="whisper-cli thread count")
    p.add_argument("--hf-token", required=True, help="HuggingFace access token")
    p.add_argument("--num-speakers", type=int, default=None, help="Known speaker count (optional)")
    p.add_argument("--output-json", action="store_true", help="Output JSON instead of plain text")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"Audio file not found: {args.file}")

    print("Step 1/3: Transcribing with whisper.cpp ...", file=sys.stderr)
    transcription = run_whisper(args.file, args.model, args.language, args.threads)

    print("Step 2/3: Diarizing speakers with pyannote ...", file=sys.stderr)
    diarization = run_diarization(args.file, args.hf_token, args.num_speakers)

    print("Step 3/3: Merging results ...", file=sys.stderr)
    segments = assign_speakers(transcription, diarization)

    if args.output_json:
        print(format_json(segments))
    else:
        print(format_output(segments))


if __name__ == "__main__":
    main()
