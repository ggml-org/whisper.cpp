#!/usr/bin/env python3
"""Dump per-stage reference tensors from NeMo for Parakeet-TDT numerical parity.

Produces a directory of .npy files + the NeMo reference transcript; used by
C++ tests (test-tdt-encoder-parity, test-tdt-decoder-parity) to validate encoder
output and decoder state at f16 precision:

    <out>/
        mel.npy              (n_mels, T_mel)   post-preprocessor log-mel
        encoder_out.npy      (T_enc, d_model)  NeMo encoder final output
        transcript.txt                         NeMo transcribe() greedy transcript
        pred_init_h.npy      (L, H)            initial LSTM hidden state (from blank_id=vocab)
        pred_init_c.npy      (L, H)            initial LSTM cell state
        joint_last_logits.npy (vocab+1+D,)     last joint logits (sanity check)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav", type=Path, required=True, help="Input mono 16 kHz wav")
    p.add_argument("--out", type=Path, default=Path("artifacts/tdt-ref"), help="Output directory for .npy dumps")
    p.add_argument("--nemo-model", type=Path, default=Path("models/parakeet-tdt-0.6b-v3.nemo"))
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import nemo.collections.asr as nemo_asr

    print(f"[tdt-ref] restoring from {args.nemo_model}", file=sys.stderr)
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(str(args.nemo_model), map_location=args.device)
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    import soundfile as sf
    wav, sr = sf.read(str(args.wav), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000
    print(f"[tdt-ref] wav: {len(wav)} samples @ {sr} Hz ({len(wav)/sr:.2f} s)", file=sys.stderr)

    wav_t    = torch.from_numpy(wav).unsqueeze(0).to(args.device)
    length_t = torch.tensor([len(wav)], dtype=torch.long, device=args.device)

    with torch.inference_mode():
        mel, mel_len = model.preprocessor(input_signal=wav_t, length=length_t)
        np.save(args.out / "mel.npy", mel[0].detach().cpu().numpy().astype(np.float32))
        print(f"[tdt-ref] mel: {tuple(mel.shape)} -> mel.npy", file=sys.stderr)

        enc_out, enc_len = model.encoder(audio_signal=mel, length=mel_len)
        enc_np = enc_out[0].permute(1, 0).detach().cpu().numpy().astype(np.float32)
        np.save(args.out / "encoder_out.npy", enc_np)
        print(f"[tdt-ref] encoder_out: {enc_np.shape} (T_enc, d_model) -> encoder_out.npy "
              f"(T_enc={int(enc_len[0])})", file=sys.stderr)

        blank_id = model.decoder.vocab_size
        pred_rnn_layers = model.decoder.prednet_layers if hasattr(model.decoder, 'prednet_layers') else 2

        init_token = torch.tensor([[blank_id]], dtype=torch.long, device=args.device)
        init_state = model.decoder.initialize_state(init_token)

        if isinstance(init_state, (tuple, list)) and len(init_state) == 2:
            h0, c0 = init_state
            np.save(args.out / "pred_init_h.npy", h0.detach().cpu().numpy().astype(np.float32))
            np.save(args.out / "pred_init_c.npy", c0.detach().cpu().numpy().astype(np.float32))
            print(f"[tdt-ref] LSTM init state: h={tuple(h0.shape)} c={tuple(c0.shape)}", file=sys.stderr)

        g, _, _ = model.decoder(targets=init_token, target_length=torch.tensor([1], device=args.device))
        g_np = g[0, 0].detach().cpu().numpy().astype(np.float32)
        print(f"[tdt-ref] prediction net output for blank: {g_np.shape}", file=sys.stderr)
        np.save(args.out / "pred_blank_out.npy", g_np)

    print(f"[tdt-ref] transcribing {args.wav} with NeMo TDT...", file=sys.stderr)
    hyps = model.transcribe([str(args.wav)], batch_size=1)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    h0 = hyps[0] if isinstance(hyps, list) else hyps

    text = h0.text if hasattr(h0, "text") else h0
    (args.out / "transcript.txt").write_text(text + "\n")
    print(f"[tdt-ref] transcript: {text!r}", file=sys.stderr)

    token_ids = None
    if hasattr(h0, "y_sequence"):
        ts = h0.y_sequence
        if hasattr(ts, "detach"):
            token_ids = ts.detach().cpu().numpy().astype(np.int32)
        else:
            token_ids = np.asarray(ts, dtype=np.int32)
    if token_ids is not None:
        np.save(args.out / "token_ids.npy", token_ids)
        print(f"[tdt-ref] token_ids: {token_ids.shape} -> token_ids.npy "
              f"(first 16: {token_ids[:16].tolist()})", file=sys.stderr)
    else:
        print("[tdt-ref] WARN: hypothesis has no y_sequence; skipping token_ids.npy",
              file=sys.stderr)

    print(f"[tdt-ref] done -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
