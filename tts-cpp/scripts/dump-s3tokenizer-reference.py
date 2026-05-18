#!/usr/bin/env python3
"""Dump S3TokenizerV2 inputs and outputs for numerical validation against
the C++ port.

Writes into --out:
  wav_16k.npy      — (L,) float32 16 kHz wav (post-resample, no trimming)
  log_mel.npy      — (128, T) log-mel spectrogram (after clamp + normalise)
  tokens.npy       — (n_tokens,) int32, same length as Python produces
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Clip tokens to this many (matches forward(max_len=...))")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from chatterbox.tts_turbo import ChatterboxTurboTTS
    tts = ChatterboxTurboTTS.from_pretrained("cpu")
    tok = tts.s3gen.tokenizer
    tok.eval()

    # Load + resample to 16 kHz.
    wav, sr = torchaudio.load(str(args.wav))
    wav = wav.mean(dim=0) if wav.ndim == 2 and wav.shape[0] > 1 else wav.squeeze(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.float()

    np.save(args.out / "wav_16k.npy",
            np.ascontiguousarray(wav.numpy().astype(np.float32)))

    with torch.no_grad():
        # log_mel is the post-clamp/normalise tensor the tokenizer consumes.
        mel = tok.log_mel_spectrogram(wav.unsqueeze(0)).squeeze(0)  # (128, T)
        tokens, lens = tok.forward([wav], max_len=args.max_tokens)

    np.save(args.out / "log_mel.npy",
            np.ascontiguousarray(mel.cpu().numpy().astype(np.float32)))
    tok_np = tokens[0, :lens[0]].cpu().numpy().astype(np.int32)
    np.save(args.out / "tokens.npy", np.ascontiguousarray(tok_np))

    print(f"wav_16k.npy  shape={wav.shape}")
    print(f"log_mel.npy  shape={mel.shape}  min={mel.min():.3f}  max={mel.max():.3f}")
    print(f"tokens.npy   n={tok_np.shape[0]}  first10={tok_np[:10].tolist()}")


if __name__ == "__main__":
    main()
