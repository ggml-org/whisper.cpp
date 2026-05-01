#!/usr/bin/env python3
"""Dump per-stage NeMo reference tensors for Sortformer parity testing.

Produces a directory of .npy files used by test-sortformer-parity:

    <out>/
        mel.npy              (n_mels, T_mel)   post-preprocessor log-mel
        encoder_out.npy      (T_enc, fc_d_model=512)
        post_proj.npy        (T_enc, tf_d_model=192)   after encoder_proj
        post_transformer.npy (T_enc, 192)              after 18-layer Transformer
        speaker_probs.npy    (T_enc, num_spks=4)       sigmoid output

The peak-normalization NeMo applies before the preprocessor is replicated
here so the C++ port can match the same input.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/sortformer-ref"))
    p.add_argument("--nemo-model", type=Path, default=Path("models/diar_sortformer_4spk-v1.nemo"))
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import nemo.collections.asr as nemo_asr

    print(f"[sf-ref] restoring from {args.nemo_model}", file=sys.stderr)
    model = nemo_asr.models.SortformerEncLabelModel.restore_from(str(args.nemo_model), map_location=args.device)
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
    print(f"[sf-ref] wav: {len(wav)} samples @ {sr} Hz ({len(wav)/sr:.2f} s)", file=sys.stderr)

    wav_t    = torch.from_numpy(wav).unsqueeze(0).to(args.device)
    length_t = torch.tensor([len(wav)], dtype=torch.long, device=args.device)

    with torch.inference_mode():
        proc, proc_len = model.process_signal(audio_signal=wav_t, audio_signal_length=length_t)
        proc = proc[:, :, : proc_len.max()]
        np.save(args.out / "mel.npy", proc[0].detach().cpu().numpy().astype(np.float32))
        print(f"[sf-ref] mel: {tuple(proc.shape)} (after peak-normalisation)", file=sys.stderr)

        emb_seq, emb_len = model.frontend_encoder(processed_signal=proc, processed_signal_length=proc_len)
        post_proj_np = emb_seq[0].detach().cpu().numpy().astype(np.float32)
        np.save(args.out / "post_proj.npy", post_proj_np)
        print(f"[sf-ref] post_proj: {post_proj_np.shape}", file=sys.stderr)

        encoder_only_out, _ = model.encoder(audio_signal=proc, length=proc_len)
        encoder_out = encoder_only_out.transpose(1, 2)[0].detach().cpu().numpy().astype(np.float32)
        np.save(args.out / "encoder_out.npy", encoder_out)
        print(f"[sf-ref] encoder_out (pre-proj): {encoder_out.shape}", file=sys.stderr)

        encoder_mask = model.sortformer_modules.length_to_mask(emb_len, emb_seq.shape[1])
        trans_out = model.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
        np.save(args.out / "post_transformer.npy",
                trans_out[0].detach().cpu().numpy().astype(np.float32))
        print(f"[sf-ref] post_transformer: {tuple(trans_out.shape)}", file=sys.stderr)

        preds = model.sortformer_modules.forward_speaker_sigmoids(trans_out)
        preds = preds * encoder_mask.unsqueeze(-1)
        spk_np = preds[0].detach().cpu().numpy().astype(np.float32)
        np.save(args.out / "speaker_probs.npy", spk_np)
        print(f"[sf-ref] speaker_probs: {spk_np.shape}  range [{spk_np.min():.4f}, {spk_np.max():.4f}]", file=sys.stderr)

        thr = 0.5
        active = spk_np > thr
        spk_count_per_frame = active.sum(axis=1)
        print(f"[sf-ref] frames with >0 active speakers: {(spk_count_per_frame > 0).sum()}/{spk_np.shape[0]}", file=sys.stderr)
        print(f"[sf-ref] max simultaneous speakers in any frame: {spk_count_per_frame.max()}", file=sys.stderr)
        for s in range(spk_np.shape[1]):
            n_active = active[:, s].sum()
            print(f"[sf-ref]   speaker {s}: {n_active} active frames ({100.0*n_active/spk_np.shape[0]:.1f}%)", file=sys.stderr)

    print(f"[sf-ref] done -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
