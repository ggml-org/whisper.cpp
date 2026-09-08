#!/usr/bin/env python3
"""Download audio from YouTube and transcribe with local whisper-cli."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WHISPER_CLI = REPO_ROOT / "build" / "bin" / "whisper-cli"
MODEL = REPO_ROOT / "models" / "ggml-large-v3.bin"


def download_audio(url: str, out_path: str):
    """Download audio from YouTube URL using yt-dlp."""
    subprocess.run([
        "yt-dlp", "-x", "--audio-format", "wav",
        "-o", out_path, url,
    ], check=True)


def convert_wav(src: str, dst: str):
    """Convert audio to 16kHz mono WAV for whisper."""
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-ar", "16000", "-ac", "1", "-f", "wav", dst,
    ], check=True, capture_output=True)


def transcribe(wav_path: str) -> str:
    """Run whisper-cli on a WAV file and return text."""
    result = subprocess.run([
        str(WHISPER_CLI), "-m", str(MODEL),
        "-f", wav_path, "-nt", "-np", "-l", "auto",
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"whisper-cli error: {result.stderr[:500]}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip().replace("[BLANK_AUDIO]", "").strip()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <youtube-url>")
        sys.exit(1)

    url = sys.argv[1]

    if not WHISPER_CLI.exists():
        print(f"whisper-cli not found: {WHISPER_CLI}", file=sys.stderr)
        sys.exit(1)
    if not MODEL.exists():
        print(f"Model not found: {MODEL}\nRun: bash models/download-ggml-model.sh base", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="yt-whisper-") as tmpdir:
        raw_audio = os.path.join(tmpdir, "audio.%(ext)s")
        wav_16k = os.path.join(tmpdir, "audio_16k.wav")

        print(f"Downloading audio from {url} ...")
        download_audio(url, raw_audio)

        # find the downloaded file (yt-dlp replaces %(ext)s)
        downloaded = [f for f in os.listdir(tmpdir) if f.startswith("audio.")]
        if not downloaded:
            print("Download failed: no audio file found", file=sys.stderr)
            sys.exit(1)
        raw_path = os.path.join(tmpdir, downloaded[0])

        print("Converting to 16kHz WAV ...")
        convert_wav(raw_path, wav_16k)

        print("Transcribing ...")
        text = transcribe(wav_16k)

        print("\n=== Transcription ===")
        print(text)


if __name__ == "__main__":
    main()
