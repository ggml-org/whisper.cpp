#!/usr/bin/env python3
"""Python reference for chunked CTC inference with left/right context (offline weights).

For each chunk [start_s, end_s), uses audio [max(0,start-left), min(end+right,audio_end)],
runs preprocessor + encoder + CTC on that window, emits logits for frames overlapping the chunk,
and carries greedy previous-token state across chunks.

Example:

    python scripts/streaming-reference.py \\
        --wav test/samples/jfk.wav \\
        --nemo-model models/parakeet-ctc-0.6b.nemo \\
        --chunk-ms 1000 --left-ctx-ms 10000 --right-lookahead-ms 1000

Sweep mode runs a grid of configs and prints WER vs offline for each, so we
pick reasonable C++ defaults from data rather than guessing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


MEL_HOP_MS = 10
SUBSAMPLING_FACTOR = 8
ENC_FRAME_STRIDE_MS = MEL_HOP_MS * SUBSAMPLING_FACTOR  # 80 ms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--nemo-model", type=Path, default=Path("models/parakeet-ctc-0.6b.nemo"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk-ms", type=int, default=1000)
    p.add_argument("--left-ctx-ms", type=int, default=10000)
    p.add_argument("--right-lookahead-ms", type=int, default=1000)
    p.add_argument("--sweep", action="store_true",
                   help="Run a grid of (chunk, left, right) configs and report WER for each.")
    return p.parse_args()


def load_wav_16k_mono(path: Path, device: str) -> torch.Tensor:
    import soundfile as sf
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
    return torch.from_numpy(wav).to(device)


def load_model(nemo_path: Path, device: str):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import nemo.collections.asr as nemo_asr
    if nemo_path.exists():
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(str(nemo_path), map_location=device)
    else:
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b", map_location=device)
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    return model


def run_full_encoder(model, wav: torch.Tensor):
    x = wav.unsqueeze(0)
    length = torch.tensor([x.shape[1]], dtype=torch.long, device=wav.device)
    with torch.inference_mode():
        mel, mel_len = model.preprocessor(input_signal=x, length=length)
        enc_out, enc_len = model.encoder(audio_signal=mel, length=mel_len)
        logits = model.decoder(encoder_output=enc_out)
    return logits[0].detach().cpu().numpy(), int(enc_len[0])


def ctc_greedy_stateful(logits: np.ndarray, prev_token: int, blank_id: int):
    tokens = []
    prev = prev_token
    for t in range(logits.shape[0]):
        best = int(np.argmax(logits[t]))
        if best != blank_id and best != prev:
            tokens.append(best)
        prev = best
    return tokens, prev


def detokenize(vocab_pieces, ids, blank_id):
    out = []
    for i in ids:
        if i == blank_id or i < 0 or i >= len(vocab_pieces):
            continue
        piece = vocab_pieces[i]
        piece = piece.replace("\u2581", " ")
        out.append(piece)
    s = "".join(out)
    while s.startswith(" "):
        s = s[1:]
    return s


def run_chunked_with_context(model, wav: torch.Tensor,
                             chunk_ms: int, left_ctx_ms: int, right_lookahead_ms: int,
                             verbose: bool = False):
    """Chunking-with-context streaming inference using offline weights.

    Returns:
        concat_logits: (T_enc_concat, vocab) float32
        segments: list of (start_s, end_s, text) for this chunk config
        transcript: full concatenated text
    """
    sr = 16000
    audio_samples = wav.shape[0]
    chunk_samples = int(chunk_ms * sr / 1000)
    left_samples = int(left_ctx_ms * sr / 1000)
    right_samples = int(right_lookahead_ms * sr / 1000)
    frames_per_chunk = max(1, chunk_ms // ENC_FRAME_STRIDE_MS)
    left_frames = max(0, left_ctx_ms // ENC_FRAME_STRIDE_MS)
    right_frames = max(0, right_lookahead_ms // ENC_FRAME_STRIDE_MS)

    if verbose:
        print(f"[py-stream] chunk_ms={chunk_ms} left_ctx_ms={left_ctx_ms} "
              f"right_lookahead_ms={right_lookahead_ms}", file=sys.stderr)
        print(f"[py-stream] chunk_samples={chunk_samples} left={left_samples} right={right_samples}",
              file=sys.stderr)
        print(f"[py-stream] frames_per_chunk={frames_per_chunk} left_frames={left_frames} "
              f"right_frames={right_frames}", file=sys.stderr)

    blank_id = model.decoder.num_classes_with_blank - 1
    vocab_pieces = list(model.tokenizer.tokenizer.id_to_piece(i)
                        for i in range(model.tokenizer.tokenizer.get_piece_size()))

    prev_token = -1
    all_logits_pieces = []
    all_tokens = []
    segments = []
    cumulative_text = ""

    chunk_index = 0
    start_sample = 0
    while start_sample < audio_samples:
        end_sample = min(start_sample + chunk_samples, audio_samples)
        is_last_chunk = end_sample >= audio_samples

        window_start = max(0, start_sample - left_samples)
        window_end = min(audio_samples, end_sample + right_samples)

        window = wav[window_start:window_end].unsqueeze(0)
        length = torch.tensor([window.shape[1]], dtype=torch.long, device=wav.device)

        with torch.inference_mode():
            mel, mel_len = model.preprocessor(input_signal=window, length=length)
            enc_out, enc_len = model.encoder(audio_signal=mel, length=mel_len)
            logits = model.decoder(encoder_output=enc_out)
        logits = logits[0].detach().cpu().numpy()
        enc_T = int(enc_len[0])

        center_start_sample = start_sample - window_start
        center_end_sample = end_sample - window_start
        left_drop_frames = center_start_sample // (sr * ENC_FRAME_STRIDE_MS // 1000)
        center_frame_count = (center_end_sample - center_start_sample) // (sr * ENC_FRAME_STRIDE_MS // 1000)
        if is_last_chunk:
            right_drop_frames = 0
            center_frame_count = enc_T - left_drop_frames
        else:
            right_drop_frames = enc_T - left_drop_frames - center_frame_count

        left_drop_frames = max(0, min(left_drop_frames, enc_T))
        right_drop_frames = max(0, min(right_drop_frames, enc_T - left_drop_frames))
        center_logits = logits[left_drop_frames : enc_T - right_drop_frames]

        tokens, prev_token = ctc_greedy_stateful(center_logits, prev_token, blank_id)

        all_logits_pieces.append(center_logits)
        prev_cum_len = len(cumulative_text)
        all_tokens.extend(tokens)
        cumulative_text = detokenize(vocab_pieces, all_tokens, blank_id)
        win_text = cumulative_text[prev_cum_len:]

        start_s = start_sample / sr
        end_s = end_sample / sr
        segments.append((start_s, end_s, win_text))

        if verbose:
            print(f"[py-stream] chunk {chunk_index}: "
                  f"window=[{window_start/sr:.2f},{window_end/sr:.2f}] center=[{start_s:.2f},{end_s:.2f}] "
                  f"enc_T={enc_T} drop_L={left_drop_frames} drop_R={right_drop_frames} "
                  f"tokens={len(tokens)} text={win_text!r}", file=sys.stderr)

        chunk_index += 1
        start_sample = end_sample

    concat_logits = np.concatenate(all_logits_pieces, axis=0) if all_logits_pieces else np.zeros((0, logits.shape[-1]), np.float32)
    transcript = cumulative_text
    return concat_logits, segments, transcript


def wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else float("inf")
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + cost)
    return d[-1][-1] / len(ref_words)


def main():
    args = parse_args()

    print(f"[py-stream] loading model: {args.nemo_model}", file=sys.stderr)
    model = load_model(args.nemo_model, args.device)
    wav = load_wav_16k_mono(args.wav, args.device)
    print(f"[py-stream] wav: {len(wav)} samples @ 16000 Hz ({len(wav)/16000:.2f} s)", file=sys.stderr)

    print("[py-stream] offline full-encoder reference...", file=sys.stderr)
    full_logits, full_T = run_full_encoder(model, wav)
    blank_id = model.decoder.num_classes_with_blank - 1
    full_tokens, _ = ctc_greedy_stateful(full_logits, -1, blank_id)
    vocab_pieces = list(model.tokenizer.tokenizer.id_to_piece(i)
                        for i in range(model.tokenizer.tokenizer.get_piece_size()))
    full_transcript = detokenize(vocab_pieces, full_tokens, blank_id)
    print(f"[py-stream] offline: {full_T} encoder frames, {len(full_tokens)} tokens", file=sys.stderr)
    print(f"[py-stream] offline: {full_transcript}", file=sys.stderr)

    if args.sweep:
        configs = [
            (chunk, left, right)
            for chunk in (500, 1000, 2000, 4000)
            for left in (0, 2000, 5000, 10000)
            for right in (0, 500, 1000, 2000)
        ]
    else:
        configs = [(args.chunk_ms, args.left_ctx_ms, args.right_lookahead_ms)]

    print()
    print(f"{'chunk_ms':>8} {'left_ms':>7} {'right_ms':>8} {'chunks':>6} {'tokens':>6} {'WER_rel':>8}  transcript")
    print("-" * 100)
    for chunk_ms, left_ms, right_ms in configs:
        _, segments, transcript = run_chunked_with_context(
            model, wav, chunk_ms, left_ms, right_ms, verbose=(not args.sweep and args.chunk_ms == chunk_ms))
        w = wer(full_transcript, transcript) * 100.0
        preview = transcript[:80] + ("..." if len(transcript) > 80 else "")
        print(f"{chunk_ms:>8} {left_ms:>7} {right_ms:>8} {len(segments):>6} "
              f"{sum(1 for _ in transcript.split()):>6} {w:>7.2f}%  {preview}")


if __name__ == "__main__":
    main()
