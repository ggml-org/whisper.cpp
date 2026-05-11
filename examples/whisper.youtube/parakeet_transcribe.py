#!/usr/bin/env python3
"""Transcribe audio with NVIDIA Parakeet-tdt-0.6b-v3 (multilingual)."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


def download_audio(url: str, out_path: str):
    subprocess.run([
        "yt-dlp", "-x", "--audio-format", "wav",
        "-o", out_path, url,
    ], check=True)


def convert_wav(src: str, dst: str):
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-ar", "16000", "-ac", "1", "-f", "wav", dst,
    ], check=True, capture_output=True)


def transcribe(wav_path: str, long_audio: bool = True) -> str:
    import nemo.collections.asr as nemo_asr

    print(f"Loading {MODEL_NAME} ...", flush=True)
    asr = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    asr = asr.cuda() if __import__("torch").cuda.is_available() else asr

    if long_audio:
        asr.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[256, 256],
        )

    print(f"Transcribing {wav_path} ...", flush=True)
    out = asr.transcribe([wav_path], timestamps=False)
    return out[0].text


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <wav-file-or-youtube-url>")
        sys.exit(1)

    src = sys.argv[1]

    if src.startswith(("http://", "https://", "youtu", "www.")):
        with tempfile.TemporaryDirectory(prefix="parakeet-") as tmp:
            raw = os.path.join(tmp, "audio.%(ext)s")
            wav = os.path.join(tmp, "audio_16k.wav")
            print(f"Downloading {src} ...", flush=True)
            download_audio(src, raw)
            downloaded = [f for f in os.listdir(tmp) if f.startswith("audio.")]
            print("Converting to 16kHz WAV ...", flush=True)
            convert_wav(os.path.join(tmp, downloaded[0]), wav)
            text = transcribe(wav)
    else:
        text = transcribe(src)

    print("\n=== Transcription ===")
    print(text)


if __name__ == "__main__":
    main()
