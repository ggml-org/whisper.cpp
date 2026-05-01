#!/usr/bin/env python3
"""Dump per-stage reference tensors from NeMo for Parakeet-EOU numerical parity.

EOU = ``nvidia/parakeet_realtime_eou_120m-v1`` (FastConformer-RNN-T 120M,
English, 17 encoder layers, ``att_context_size=[70, 1]`` chunked-limited,
``<EOU>`` end-of-utterance token).

Produces a directory of .npy files + the NeMo *offline* reference transcript for
C++ encoder/decoder parity tests:

    <out>/
        mel.npy              (n_mels, T_mel)   post-preprocessor log-mel (offline)
        encoder_out.npy      (T_enc, d_model)  NeMo encoder final output (offline,
                                               full attention, no streaming caches)
        transcript.txt                         NeMo transcribe() greedy transcript
                                               (offline).
        pred_init_h.npy      (L, 1, H)         initial LSTM hidden state
        pred_init_c.npy      (L, 1, H)         initial LSTM cell state
        pred_blank_out.npy   (H,)              prediction-net output for the blank
                                               token (sanity check that our embed +
                                               LSTM matches NeMo)

Uses full-utterance offline NeMo forward only. NeMo also offers ``cache_aware_stream_step``
for chunked streaming; this script does not dump that path (the C++ stack validates against
offline references).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav", type=Path, required=True, help="Input mono 16 kHz wav")
    p.add_argument("--out", type=Path, default=Path("artifacts/eou-ref"),
                   help="Output directory for .npy dumps")
    p.add_argument("--nemo-model", type=Path,
                   default=Path("models/parakeet_realtime_eou_120m-v1.nemo"))
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _load_wav(path: Path) -> np.ndarray:
    import soundfile as sf
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
    return wav


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import nemo.collections.asr as nemo_asr

    print(f"[eou-ref] restoring from {args.nemo_model}", file=sys.stderr)
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
        str(args.nemo_model), map_location=args.device)
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    wav = _load_wav(args.wav)
    print(f"[eou-ref] wav: {len(wav)} samples ({len(wav)/16000:.2f} s)", file=sys.stderr)

    wav_t    = torch.from_numpy(wav).unsqueeze(0).to(args.device)
    length_t = torch.tensor([len(wav)], dtype=torch.long, device=args.device)

    with torch.inference_mode():
        mel, mel_len = model.preprocessor(input_signal=wav_t, length=length_t)
        np.save(args.out / "mel.npy", mel[0].detach().cpu().numpy().astype(np.float32))
        print(f"[eou-ref] mel: {tuple(mel.shape)} -> mel.npy", file=sys.stderr)

        # Post-subsampler and encoder taps for C++ encoder parity tests.
        intermediates = {}

        sub_out, sub_len = model.encoder.pre_encode(x=mel.transpose(1, 2), lengths=mel_len)
        sub_np = sub_out[0].detach().cpu().numpy().astype(np.float32)
        np.save(args.out / "subsampling_out.npy", sub_np)
        intermediates['subsampling_out'] = sub_np.shape
        print(f"[eou-ref] subsampling_out: {sub_np.shape} (T_enc, d_model) -> subsampling_out.npy",
              file=sys.stderr)

        enc_out, enc_len = model.encoder(audio_signal=mel, length=mel_len)
        enc_np = enc_out[0].permute(1, 0).detach().cpu().numpy().astype(np.float32)
        np.save(args.out / "encoder_out.npy", enc_np)
        print(f"[eou-ref] encoder_out (offline): {enc_np.shape} (T_enc, d_model) -> encoder_out.npy "
              f"(T_enc={int(enc_len[0])})", file=sys.stderr)

        blank_id = model.decoder.blank_idx
        init_token = torch.tensor([[blank_id]], dtype=torch.long, device=args.device)
        init_state = model.decoder.initialize_state(init_token)

        h0, c0 = init_state
        np.save(args.out / "pred_init_h.npy",
                h0.detach().cpu().numpy().astype(np.float32))
        np.save(args.out / "pred_init_c.npy",
                c0.detach().cpu().numpy().astype(np.float32))
        print(f"[eou-ref] LSTM init state: h={tuple(h0.shape)} c={tuple(c0.shape)}",
              file=sys.stderr)

        g, _, _ = model.decoder(targets=init_token,
                                target_length=torch.tensor([1], device=args.device))
        g_np = g[0, :, 0].detach().cpu().numpy().astype(np.float32)
        np.save(args.out / "pred_blank_out.npy", g_np)
        print(f"[eou-ref] pred_blank_out: {g_np.shape} (g full shape was {tuple(g.shape)}, took [:,0])",
              file=sys.stderr)

    print(f"[eou-ref] transcribing {args.wav} with NeMo EOU (offline)...", file=sys.stderr)
    hyps = model.transcribe([str(args.wav)], batch_size=1)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    text = hyps[0] if isinstance(hyps, list) else hyps
    if hasattr(text, 'text'):
        text = text.text
    (args.out / "transcript.txt").write_text(text + "\n")
    print(f"[eou-ref] transcript: {text!r}", file=sys.stderr)
    print(f"[eou-ref] done -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
