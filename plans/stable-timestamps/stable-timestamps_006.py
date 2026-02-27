# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""
Generate MP4s with soft subtitle tracks for baseline vs stable comparison.
Open in VLC/IINA and enable subtitles to view.

Usage:
    uv run plans/stable-timestamps/stable-timestamps_006.py \
      --audio plans/stable-timestamps/out/synth_5min.wav \
      --baseline-json plans/stable-timestamps/out/baseline.json \
      --stable-json plans/stable-timestamps/out/stable.json \
      --out-dir plans/stable-timestamps/out/compare_006
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MP4s with subtitles for baseline vs stable comparison")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--stable-json", required=True)
    parser.add_argument("--out-dir", default="plans/stable-timestamps/out/compare_006")
    return parser.parse_args()


def ms_to_srt(ms: int) -> str:
    ms = max(ms, 0)
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1_000
    ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_segments(path: Path) -> list[tuple[int, int, str]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    out: list[tuple[int, int, str]] = []
    for seg in obj.get("transcription", []):
        offsets = seg.get("offsets", {})
        text = str(seg.get("text", "")).replace("\r", "").strip()
        try:
            start = int(offsets.get("from", -1))
            end = int(offsets.get("to", -1))
        except (TypeError, ValueError):
            continue
        if start < 0 or end < 0 or end <= start or not text:
            continue
        out.append((start, end, text))
    return out


def write_srt(segments: list[tuple[int, int, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, (start, end, text) in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{ms_to_srt(start)} --> {ms_to_srt(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_video(audio_path: Path, srt_path: Path, output_path: Path) -> None:
    # Step 1: black video + audio
    tmp = output_path.with_suffix(".tmp.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=640x480:r=5",
        "-i", str(audio_path),
        "-shortest",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "32", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        str(tmp),
    ], check=True)
    # Step 2: mux soft subtitle track
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(tmp),
        "-i", str(srt_path),
        "-map", "0:v", "-map", "0:a", "-map", "1:0",
        "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
        str(output_path),
    ], check=True)
    tmp.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_srt = out_dir / "baseline.srt"
    stable_srt = out_dir / "stable.srt"
    write_srt(load_segments(Path(args.baseline_json)), baseline_srt)
    write_srt(load_segments(Path(args.stable_json)), stable_srt)

    baseline_mp4 = out_dir / "baseline.mp4"
    stable_mp4 = out_dir / "stable.mp4"
    make_video(Path(args.audio), baseline_srt, baseline_mp4)
    make_video(Path(args.audio), stable_srt, stable_mp4)

    print(f"\nCreated:\n  {baseline_srt}\n  {stable_srt}\n  {baseline_mp4}\n  {stable_mp4}")
    print("\nOpen in VLC/IINA and enable subtitles to compare.")


if __name__ == "__main__":
    main()
