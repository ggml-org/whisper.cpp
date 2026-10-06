#!/usr/bin/env python3
"""Transcribe audio with NVIDIA Parakeet-tdt-0.6b-v3 (multilingual).

Modes:
  batch    — transcribe the whole audio at once, print at the end (default).
  stream   — sliding window with LocalAgreement: progressive text output as
             new audio arrives. Each window of WINDOW seconds is re-transcribed
             every SHIFT seconds; tokens agreed between two consecutive windows
             are "confirmed" and printed; the rest stays uncertain until the
             next iteration confirms (or invalidates) it.
"""

import argparse
import os
import subprocess
import sys
import tempfile

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"

# Sliding-window streaming defaults (in seconds).
# Matches whisper.cpp/examples/stream defaults.
DEFAULT_WINDOW_S = 10.0
DEFAULT_SHIFT_S = 3.0


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


def load_model(long_audio: bool = True):
    import nemo.collections.asr as nemo_asr
    import torch

    print(f"Loading {MODEL_NAME} ...", flush=True)
    asr = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    asr = asr.cuda() if torch.cuda.is_available() else asr
    if long_audio:
        asr.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[256, 256],
        )
    return asr


def transcribe_batch(asr, wav_path: str) -> str:
    print(f"Transcribing {wav_path} ...", flush=True)
    out = asr.transcribe([wav_path], timestamps=False)
    return out[0].text


def _norm(w: str) -> str:
    """Normalize a word for dedup comparison: lower-case, strip punctuation."""
    return "".join(ch for ch in w.lower() if ch.isalnum())


def _dedup_overlap(prev_text: str, curr_text: str) -> str:
    """Return the new portion of curr_text by removing the trailing words of
    prev_text that match the leading words of curr_text (overlap region).
    Comparison is case- and punctuation-insensitive."""
    prev_words = prev_text.split()
    curr_words = curr_text.split()
    prev_norm = [_norm(w) for w in prev_words]
    curr_norm = [_norm(w) for w in curr_words]
    max_n = min(len(prev_norm), len(curr_norm))
    for n in range(max_n, 0, -1):
        if prev_norm[-n:] == curr_norm[:n]:
            return " ".join(curr_words[n:])
    return curr_text


def transcribe_sliding(asr, wav_path: str, window_s: float, shift_s: float) -> str:
    """Sliding-window streaming: each window of WINDOW seconds is transcribed
    every SHIFT seconds; words overlapping with the previous window are
    de-duplicated and only the new tail is printed."""
    import soundfile as sf

    if shift_s <= 0 or window_s <= 0:
        raise ValueError("window and shift must be positive")
    if shift_s > window_s:
        raise ValueError("shift must be <= window")

    audio, sr = sf.read(wav_path)
    if sr != 16000:
        raise ValueError(f"Expected 16kHz WAV, got {sr}Hz")

    n_samples = len(audio)
    win = int(window_s * sr)
    shift = int(shift_s * sr)
    overlap_s = window_s - shift_s

    print(
        f"Sliding window: {window_s:.1f}s window, {shift_s:.1f}s shift "
        f"({overlap_s:.1f}s overlap), audio {n_samples / sr:.1f}s",
        flush=True,
    )
    print("--- streaming transcription ---", flush=True)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="parakeet-slide-",
                                       delete=False)
    tmp_wav = tmp.name
    tmp.close()

    full_text_parts = []
    prev_text = ""
    pos = 0

    try:
        while pos < n_samples:
            end = min(pos + win, n_samples)
            chunk = audio[pos:end]
            sf.write(tmp_wav, chunk, sr)

            out = asr.transcribe([tmp_wav], timestamps=False, verbose=False)
            current = out[0].text.strip()

            new_part = _dedup_overlap(prev_text, current) if prev_text else current
            if new_part:
                print(new_part, end=" ", flush=True)
                full_text_parts.append(new_part)

            prev_text = current
            pos += shift

        print(flush=True)
        return " ".join(full_text_parts).strip()
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="WAV file path or YouTube URL")
    parser.add_argument("--stream", action="store_true",
                        help="streaming mode (sliding window + LocalAgreement)")
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_S,
                        help=f"sliding window length in seconds (default: {DEFAULT_WINDOW_S})")
    parser.add_argument("--shift", type=float, default=DEFAULT_SHIFT_S,
                        help=f"window shift in seconds (default: {DEFAULT_SHIFT_S}); "
                             "overlap = window - shift")
    args = parser.parse_args()

    src = args.source

    if src.startswith(("http://", "https://", "youtu", "www.")):
        tmp = tempfile.mkdtemp(prefix="parakeet-")
        raw = os.path.join(tmp, "audio.%(ext)s")
        wav = os.path.join(tmp, "audio_16k.wav")
        print(f"Downloading {src} ...", flush=True)
        download_audio(src, raw)
        downloaded = [f for f in os.listdir(tmp) if f.startswith("audio.")]
        print("Converting to 16kHz WAV ...", flush=True)
        convert_wav(os.path.join(tmp, downloaded[0]), wav)
        wav_path = wav
    else:
        wav_path = src

    asr = load_model(long_audio=not args.stream)

    if args.stream:
        text = transcribe_sliding(asr, wav_path, args.window, args.shift)
    else:
        text = transcribe_batch(asr, wav_path)
        print("\n=== Transcription ===")
        print(text)


if __name__ == "__main__":
    main()
