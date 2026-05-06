#!/usr/bin/env python3
"""Dump CAMPPlus inputs and outputs for numerical validation against the
C++ port (src/campplus.cpp).

Writes:
  fbank.npy      — (T, 80) mean-subtracted Kaldi-fbank at 16 kHz.  This is
                   exactly what the CAMPPlus.forward pass receives.
  embedding.npy  — (192,) raw CAMPPlus output (pre-L2-norm).  Matches the
                   `embedding` tensor that prepare-voice.py stores.

Example:

    . path/to/chatterbox-ref/.venv/bin/activate
    python scripts/dump-campplus-reference.py REF.wav --out /tmp/camp_ref
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path, help="Reference wav (any SR; resampled to 16 kHz)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir; creates fbank.npy + embedding.npy inside it")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    tts = ChatterboxTurboTTS.from_pretrained("cpu")
    speaker_encoder = tts.s3gen.speaker_encoder
    speaker_encoder.eval()

    wav, sr = torchaudio.load(str(args.wav))
    wav = wav.mean(dim=0) if wav.ndim == 2 and wav.shape[0] > 1 else wav.squeeze(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    # Deterministic Kaldi fbank (no dither).  The Python extract_feature helper
    # uses dither=1.0 by default, which would give non-reproducible features
    # and make C++ vs. Python numerical comparison impossible.
    import torchaudio.compliance.kaldi as Kaldi
    fbank_raw = Kaldi.fbank(wav.unsqueeze(0), num_mel_bins=80, dither=0.0)  # (T, 80)
    # Also save the raw (un-mean-subtracted) fbank; C++ mean-subtract is
    # trivial.
    fbank_centered = fbank_raw - fbank_raw.mean(dim=0, keepdim=True)

    np.save(args.out / "fbank_raw.npy",      np.ascontiguousarray(fbank_raw.numpy().astype(np.float32)))
    np.save(args.out / "fbank.npy",          np.ascontiguousarray(fbank_centered.numpy().astype(np.float32)))

    with torch.no_grad():
        emb = speaker_encoder.forward(fbank_centered.unsqueeze(0).to(torch.float32))
    emb = emb[0].cpu().numpy().astype(np.float32)
    np.save(args.out / "embedding.npy", np.ascontiguousarray(emb))

    # The wav itself (post-resample) is also useful for C++ fbank parity tests.
    np.save(args.out / "wav_16k.npy", np.ascontiguousarray(wav.numpy().astype(np.float32)))

    print(f"fbank_raw.npy  shape={fbank_raw.shape}")
    print(f"fbank.npy      shape={fbank_centered.shape}")
    print(f"embedding.npy  shape={emb.shape}  norm={np.linalg.norm(emb):.4f}")
    print(f"wav_16k.npy    shape={wav.shape}")


if __name__ == "__main__":
    main()
