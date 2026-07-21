#!/usr/bin/env python3
"""Dump per-stage reference tensors from NeMo for numerical parity testing.

Produces, next to the wav path given on the command line, a directory of .npy
files that the C++ harnesses (test-mel, test-encoder, test-ctc) compare against:

    <out>/
        mel.npy                 (80, T_mel)   post-preprocessor log-mel (f32)
        subsampling_out.npy     (T_enc, 1024) after encoder.pre_encode
        block_0_out.npy         (T_enc, 1024) output of the first conformer layer
        block_last_out.npy      (T_enc, 1024) output of the last conformer layer
        encoder_out.npy         (T_enc, 1024) final encoder output, time-first
        logits.npy              (T_enc, 1025) ConvASRDecoder log-probs
        greedy_ids.npy          (T_enc,)      frame-argmax token ids
        decoded.txt                            transcribed text (CTC collapse)

Use --text to just verify transcription without dumping.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav", type=Path, required=True, help="Input mono 16 kHz wav (or any samplerate; will be resampled).")
    p.add_argument("--out", type=Path, default=Path("artifacts/ctc-ref"), help="Output directory for .npy dumps.")
    p.add_argument("--model", default="nvidia/parakeet-ctc-0.6b", help="HF model id or local .nemo path.")
    p.add_argument("--device", default="cpu", help="torch device (cpu / cuda / mps).")
    p.add_argument("--text-only", action="store_true", help="Skip .npy dumps, just print the transcript.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import nemo.collections.asr as nemo_asr
    if args.model.endswith(".nemo") and Path(args.model).exists():
        print(f"[dump] restoring from local .nemo: {args.model}", file=sys.stderr)
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.model, map_location=args.device)
    else:
        print(f"[dump] loading pretrained: {args.model}", file=sys.stderr)
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.model, map_location=args.device)
    model.eval()

    import soundfile as sf
    wav, sr = sf.read(str(args.wav), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000
    print(f"[dump] wav: {len(wav)} samples @ {sr} Hz ({len(wav)/sr:.2f} s)", file=sys.stderr)

    wav_t = torch.from_numpy(wav).unsqueeze(0).to(args.device)
    length_t = torch.tensor([len(wav)], dtype=torch.long, device=args.device)

    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    with torch.inference_mode():
        mel, mel_len = model.preprocessor(input_signal=wav_t, length=length_t)
        np.save(args.out / "mel.npy", mel[0].detach().cpu().numpy().astype(np.float32))
        print(f"[dump] mel: {tuple(mel.shape)} (B, n_mels, T_mel)", file=sys.stderr)

        captured = {}
        def cap(name):
            def hook(m, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                captured[name] = t.detach().cpu().numpy().astype(np.float32)
            return hook

        hooks = []
        hooks.append(model.encoder.pre_encode.register_forward_hook(cap("subsampling_out")))
        n_layers = len(model.encoder.layers)
        for i, layer in enumerate(model.encoder.layers):
            hooks.append(layer.register_forward_hook(cap(f"block_{i}_out")))

        enc, enc_len = model.encoder(audio_signal=mel, length=mel_len)
        for h in hooks:
            h.remove()

        captured["block_0_out"]    = captured.get("block_0_out")
        captured["block_last_out"] = captured.get(f"block_{n_layers - 1}_out")

        for name in list(captured.keys()):
            arr = captured[name]
            if arr is None:
                continue
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.shape[0] == 1024 and arr.shape[-1] != 1024:
                arr = arr.T
            np.save(args.out / f"{name}.npy", arr)
        print(f"[dump] subsampling_out + {n_layers} block outputs dumped", file=sys.stderr)

        enc_np = enc[0].detach().cpu().numpy().astype(np.float32)
        if enc_np.shape[0] == 1024:
            enc_np = enc_np.T
        np.save(args.out / "encoder_out.npy", enc_np)
        print(f"[dump] encoder_out: {enc_np.shape} (T_enc, d_model)", file=sys.stderr)

        logp = model.decoder(encoder_output=enc)
        lp_np = logp[0].detach().cpu().numpy().astype(np.float32)
        if lp_np.shape[0] == 1025:
            lp_np = lp_np.T
        np.save(args.out / "logits.npy", lp_np)
        print(f"[dump] logits: {lp_np.shape} (T_enc, vocab+1)", file=sys.stderr)

        greedy = lp_np.argmax(axis=-1).astype(np.int32)
        np.save(args.out / "greedy_ids.npy", greedy)

        blank_id = int(lp_np.shape[-1] - 1)
        collapsed = []
        prev = -1
        for tok in greedy:
            if int(tok) != blank_id and int(tok) != prev:
                collapsed.append(int(tok))
            prev = int(tok)

    try:
        tok = model.tokenizer
        text = tok.ids_to_text(collapsed) if hasattr(tok, "ids_to_text") else tok.decode(collapsed)
    except Exception:
        text = " ".join(str(x) for x in collapsed)
    print(f"[dump] transcript: {text}")

    with open(args.out / "decoded.txt", "w") as f:
        f.write(text + "\n")
    print(f"[dump] wrote {args.out}/{{mel,subsampling_out,block_0_out,block_last_out,encoder_out,logits,greedy_ids}}.npy", file=sys.stderr)
    print(f"[dump] wrote {args.out}/decoded.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
