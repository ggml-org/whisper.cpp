#!/usr/bin/env python3
"""Golden fixtures for the FULL LavaSR denoiser pipeline.

The spec-level fixtures (dump-lavasr-denoiser-fixtures.py) only exercise the
UL-UNAS neural core on a pre-computed [1,2,63,257] spectrogram. This script
produces a *waveform-in -> waveform-out* golden that also covers the parts the
core parity can't see: STFT (center-padded periodic Hann), the fixed-63 / hop-21
squared-Hann chunk overlap-add stitch, and the ISTFT normalization.

It mirrors, line for line:
  - tts-cpp/src/lavasr/dsp/stft_processor.cpp  (StftProcessor, center pad)
  - @qvac/tts-onnx LavaSRDenoiser::denoise + buildChunkWeights  (chunk OLA)
running each fixed-63 chunk through onnxruntime (the ground-truth core).

The signal is >63 frames so it exercises multiple overlap-add seams. Output:
  <out-dir>/pcm_in.npy   float32 [N]   (16 kHz mono)
  <out-dir>/pcm_out.npy  float32 [N]   (denoised, same length)

The C++ test (test_lavasr_denoiser_gguf.cpp) calls Denoiser::denoise(pcm_in,
16000) — work rate == input rate, so the resampler is identity and only the
STFT/OLA/ISTFT math is compared against this golden.
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort

# Must match convert-lavasr-denoiser-to-gguf.py / StftProcessor / LavaSRDenoiser.
N_FFT = 512
HOP = 256
WIN = 512
SPEC_BINS = N_FFT // 2 + 1  # 257
CHUNK_FRAMES = 63
CHUNK_HOP = 21


def hann_periodic(length: int) -> np.ndarray:
    i = np.arange(length)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * i / length))


WINDOW = hann_periodic(WIN).astype(np.float64)


def stft(signal: np.ndarray) -> np.ndarray:
    """Mirror StftProcessor::stft (center pad = n_fft/2, reflect). -> [T,257] cplx."""
    pad = N_FFT // 2
    xpad = np.pad(signal.astype(np.float64), (pad, pad), mode="reflect")
    if xpad.size < WIN:
        xpad = np.concatenate([xpad, np.zeros(WIN - xpad.size)])
    num_frames = (xpad.size - WIN) // HOP + 1
    spec = np.empty((num_frames, SPEC_BINS), dtype=np.complex128)
    for t in range(num_frames):
        seg = xpad[t * HOP : t * HOP + WIN] * WINDOW
        spec[t] = np.fft.fft(seg, N_FFT)[:SPEC_BINS]
    return spec


def istft(spec: np.ndarray, target_len: int) -> np.ndarray:
    """Mirror StftProcessor::istft (hermitian, ifft, wenv norm, trim pad)."""
    pad = N_FFT // 2
    T = spec.shape[0]
    output_size = (T - 1) * HOP + WIN
    y = np.zeros(output_size, dtype=np.float64)
    wenv = np.zeros(output_size, dtype=np.float64)
    for t in range(T):
        full = np.zeros(N_FFT, dtype=np.complex128)
        full[0:SPEC_BINS] = spec[t]
        full[SPEC_BINS:N_FFT] = np.conj(spec[t][1 : N_FFT // 2][::-1])
        frame = np.fft.ifft(full).real  # ifft applies 1/N, matches C++
        y[t * HOP : t * HOP + WIN] += frame[:WIN] * WINDOW
        wenv[t * HOP : t * HOP + WIN] += WINDOW * WINDOW
    out = y[pad : output_size - pad] / np.maximum(wenv[pad : output_size - pad], 1e-8)
    if target_len > 0:
        out = out[:target_len]
        if out.size < target_len:
            out = np.concatenate([out, np.zeros(target_len - out.size)])
    return out


def chunk_weights(L: int) -> np.ndarray:
    """Squared symmetric Hann + 1e-4 floor — LavaSRDenoiser::buildChunkWeights."""
    i = np.arange(L)
    h = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / (L - 1)))
    return np.maximum(h * h, 1e-4)


def denoise(sess: ort.InferenceSession, in_name: str, out_name: str,
            wav: np.ndarray) -> np.ndarray:
    spec = stft(wav)  # [T,257] complex
    T = spec.shape[0]
    L, H = CHUNK_FRAMES, CHUNK_HOP
    flat = np.stack([spec.real, spec.imag], axis=0).astype(np.float32)  # [2,T,257]

    def run(chunk):  # chunk [2,L,257] -> [2,L,257]
        inp = chunk[np.newaxis].astype(np.float32)  # [1,2,L,257]
        return sess.run([out_name], {in_name: inp})[0][0]

    if T <= L:
        chunk = np.zeros((2, L, SPEC_BINS), dtype=np.float32)
        chunk[:, :T, :] = flat
        out = run(chunk)
        flat = out[:, :T, :]
    else:
        starts = list(range(0, T - L + 1, H))
        if starts[-1] != T - L:
            starts.append(T - L)
        cw = chunk_weights(L).astype(np.float32)
        acc = np.zeros((2, T, SPEC_BINS), dtype=np.float64)
        wacc = np.zeros(T, dtype=np.float64)
        for s in starts:
            out = run(flat[:, s : s + L, :])
            acc[:, s : s + L, :] += out * cw[np.newaxis, :, np.newaxis]
            wacc[s : s + L] += cw
        flat = (acc / np.maximum(wacc, 1e-6)[np.newaxis, :, np.newaxis]).astype(np.float32)

    spec_enh = flat[0].astype(np.complex128) + 1j * flat[1].astype(np.complex128)
    return istft(spec_enh, wav.size).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--denoiser", required=True, help="UL-UNAS denoiser ONNX")
    ap.add_argument("--out-dir", required=True, help="fixtures output dir")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    n = int(args.seconds * 16000)
    rng = np.random.default_rng(args.seed)
    t = np.arange(n) / 16000.0
    # Deterministic non-trivial signal: a few tones + low-level noise. (Parity
    # test — content need not be speech, only reproducible.)
    wav = (0.35 * np.sin(2 * np.pi * 180.0 * t)
           + 0.20 * np.sin(2 * np.pi * 900.0 * t)
           + 0.10 * np.sin(2 * np.pi * 3400.0 * t)
           + 0.03 * rng.standard_normal(n)).astype(np.float32)

    sess = ort.InferenceSession(args.denoiser, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    spec = stft(wav)
    print(f"signal: {n} samples ({args.seconds}s @16k) -> T={spec.shape[0]} frames "
          f"(chunks exercise overlap-add: T>{CHUNK_FRAMES})")
    pcm_out = denoise(sess, in_name, out_name, wav)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "pcm_in.npy"), wav)
    np.save(os.path.join(args.out_dir, "pcm_out.npy"), pcm_out)
    print(f"wrote pcm_in.npy / pcm_out.npy [{n}] to {args.out_dir}")
    print(f"pcm_out range [{pcm_out.min():.4f}, {pcm_out.max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
