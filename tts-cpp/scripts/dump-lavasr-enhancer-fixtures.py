#!/usr/bin/env python3
"""Dump LavaSR enhancer fixtures for the C++ parity test (test-lavasr-enhancer-core).

Emits, into <out-dir>, one <canonical-name>.npy per enhancer weight tensor (in
the same orientation the GGUF / scalar core use) plus two goldens:

Neural-core golden (checks the backbone + spec head):
  mel.npy   [80, T]     deterministic random log-mel input (seed 0)
  real.npy  [1025, T]   spec-head real output from the original ONNX graphs
  imag.npy  [1025, T]   spec-head imag output

End-to-end golden (checks the full DSP pipeline + neural core):
  pcm_in.npy        deterministic 24 kHz mono input signal
  enhanced_48k.npy  full numpy reference pipeline output (resample -> mel ->
                    backbone/spec-head -> ISTFT -> FastLR), 48 kHz

The numpy DSP below is an independent reimplementation of
tts-cpp/src/lavasr/dsp (FFT via numpy = the same DFT the radix-2 C++ computes),
so the resampler / mel / ISTFT / FastLR stages are compared against a reference
rather than only smoke-tested.

Requires: numpy, onnx, onnxruntime.  Inputs are the public LavaSRcpp release
ONNX files (enhancer_backbone.onnx{,.data}, enhancer_spec_head.onnx{,.data}).

Usage:
  python dump-lavasr-enhancer-fixtures.py \
      --onnx-dir /path/to/lavasr/onnx \
      --out-dir  /path/to/lavasr/fixtures \
      --frames   50
"""
import argparse
import math
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


# --- Numpy reference DSP (independent reimplementation of
# --- tts-cpp/src/lavasr/dsp/*; FFT via numpy == the DFT the radix-2 C++ computes).

_LANCZOS_A = 5


def resample(x, sr_in, sr_out):
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out or x.size == 0:
        return x
    ratio = sr_out / sr_in
    out_len = int(round(x.size * ratio))
    scale = min(1.0, ratio)
    n = x.size
    out = np.zeros(out_len, dtype=np.float32)
    for i in range(out_len):
        center = i / ratio
        left = int(max(0.0, math.floor(center - _LANCZOS_A / scale)))
        right = int(min(n - 1, math.floor(center + _LANCZOS_A / scale)))
        s = 0.0
        ws = 0.0
        for j in range(left, right + 1):
            xv = (center - j) * scale
            if xv != 0.0:
                pix = math.pi * xv
                w = math.sin(pix) * math.sin(pix / _LANCZOS_A) / (pix * pix / _LANCZOS_A)
            else:
                w = 1.0
            s += float(x[j]) * w
            ws += w
        out[i] = (s / ws) if ws > 0.0 else 0.0
    return out


def _hann(L):
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(L) / L))


def stft(signal, n_fft, hop, win, center):
    pad = n_fft // 2 if center else (win - hop) // 2
    if signal.size > 1:
        xpad = np.pad(signal, (pad, pad), mode="reflect")
    else:
        xpad = np.zeros(signal.size + 2 * pad)
    if xpad.size < win:
        xpad = np.concatenate([xpad, np.zeros(win - xpad.size)])
    num_frames = (xpad.size - win) // hop + 1
    window = _hann(win)
    bins = n_fft // 2 + 1
    spec = np.zeros((num_frames, bins), dtype=np.complex128)
    for t in range(num_frames):
        frame = np.zeros(n_fft)
        frame[:win] = xpad[t * hop: t * hop + win] * window
        spec[t] = np.fft.fft(frame)[:bins]
    return spec


def istft(spec, n_fft, hop, win, center, target_len):
    pad = n_fft // 2 if center else (win - hop) // 2
    T = spec.shape[0]
    out_size = (T - 1) * hop + win
    y = np.zeros(out_size)
    wenv = np.zeros(out_size)
    window = _hann(win)
    half = n_fft // 2
    for t in range(T):
        frame = np.zeros(n_fft, dtype=np.complex128)
        frame[:half + 1] = spec[t][:half + 1]
        frame[half + 1:] = np.conj(spec[t][1:half][::-1])
        time = np.fft.ifft(frame).real
        y[t * hop: t * hop + win] += time[:win] * window
        wenv[t * hop: t * hop + win] += window * window
    out = y[pad:out_size - pad] / np.maximum(wenv[pad:out_size - pad], 1e-8)
    if target_len > 0:
        out = out[:target_len] if out.size >= target_len else \
            np.concatenate([out, np.zeros(target_len - out.size)])
    return out.astype(np.float32)


def _hz_to_mel(f):
    return 15.0 + math.log(f / 1000.0) / (math.log(6.4) / 27.0) if f >= 1000.0 else f / (200.0 / 3.0)


def _mel_to_hz(m):
    return 1000.0 * math.exp((math.log(6.4) / 27.0) * (m - 15.0)) if m >= 15.0 else (200.0 / 3.0) * m


def mel_filters(sr, n_fft, n_mels, fmin, fmax):
    n_freqs = n_fft // 2 + 1
    fftfreqs = np.arange(n_freqs) * sr / n_fft
    mmin, mmax = _hz_to_mel(fmin), _hz_to_mel(fmax)
    fpts = np.array([_mel_to_hz(mmin + i * (mmax - mmin) / (n_mels + 1)) for i in range(n_mels + 2)])
    filt = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        fl = max(fpts[i + 1] - fpts[i], 1e-12)
        fr = max(fpts[i + 2] - fpts[i + 1], 1e-12)
        enorm = 2.0 / max(fpts[i + 2] - fpts[i], 1e-12)
        for j in range(n_freqs):
            lo = (fftfreqs[j] - fpts[i]) / fl
            hi = (fpts[i + 2] - fftfreqs[j]) / fr
            filt[i, j] = max(0.0, min(lo, hi)) * enorm
    return filt


def log_mel(wav, sr_ref, n_fft, hop, n_mels):
    spec = stft(wav, n_fft, hop, n_fft, False)         # [T][bins]
    mag = np.abs(spec)
    filt = mel_filters(sr_ref, n_fft, n_mels, 0.0, 8000.0)
    mel = np.log(np.maximum(mag @ filt.T, 1e-5))       # [T][n_mels]
    return mel.T.astype(np.float32)                    # [n_mels][T]


def fastlr_merge(enhanced, original, sr, cutoff_hz, transition_bins=256):
    N, M = len(enhanced), len(original)
    if N == 0:
        return np.zeros(0, dtype=np.float32)
    if M == 0:
        return np.asarray(enhanced, dtype=np.float32)
    npow2 = 1
    while npow2 < max(N, M):
        npow2 <<= 1
    a = np.zeros(npow2); a[:N] = enhanced
    b = np.zeros(npow2); b[:M] = original
    A, B = np.fft.fft(a), np.fft.fft(b)
    nbins = npow2 // 2 + 1
    cutoff_bin = int(cutoff_hz / (sr / 2.0) * nbins)
    half = transition_bins // 2
    start, end = max(0, cutoff_bin - half), min(nbins, cutoff_bin + half)
    mask = np.ones(nbins)
    mask[:start] = 0.0
    if end - start > 1:
        for i in range(start, end):
            x = -1.0 + 2.0 * (i - start) / (end - start - 1)
            tt = (x + 1.0) / 2.0
            mask[i] = 3 * tt * tt - 2 * tt * tt * tt
    elif end == start + 1:
        mask[start] = 0.5
    out = B.astype(np.complex128).copy()
    out[:nbins] = B[:nbins] + (A[:nbins] - B[:nbins]) * mask
    for i in range(1, npow2 // 2):
        out[npow2 - i] = np.conj(out[i])
    out[npow2 // 2] = complex(out[npow2 // 2].real, 0.0)
    return np.fft.ifft(out).real[:N].astype(np.float32)


def enhance_ref(pcm_in, sr_in, bb_sess, sh_sess, sr_ref=44100, work_sr=48000,
                n_fft=2048, hop=512, win=2048, spec_bins=1025, n_mels=80):
    wav = resample(pcm_in, sr_in, work_sr)
    mel = log_mel(wav, sr_ref, n_fft, hop, n_mels)                     # [n_mels][T]
    hid = bb_sess.run(None, {bb_sess.get_inputs()[0].name: mel[None].astype(np.float32)})[0]
    real, imag = sh_sess.run(None, {sh_sess.get_inputs()[0].name: hid})
    spec = (real[0] + 1j * imag[0]).T                                 # [T][bins]
    enhanced = istft(spec, n_fft, hop, win, False, len(wav))
    return fastlr_merge(enhanced, wav, work_sr, sr_in // 2)


def by_out(g):
    d = {}
    for n in g.node:
        for o in n.output:
            d[o] = n
    return d


def matmul_weight(g, inits, bo, bias_name):
    for n in g.node:
        if n.op_type == "Add" and bias_name in n.input:
            other = [i for i in n.input if i != bias_name][0]
            for i in bo[other].input:
                if i in inits:
                    return inits[i]
    raise RuntimeError(f"no MatMul weight for {bias_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    bb_path = os.path.join(args.onnx_dir, "enhancer_backbone.onnx")
    sh_path = os.path.join(args.onnx_dir, "enhancer_spec_head.onnx")
    bb = onnx.load(bb_path, load_external_data=True).graph
    sh = onnx.load(sh_path, load_external_data=True).graph
    bi = {t.name: numpy_helper.to_array(t) for t in bb.initializer}
    si = {t.name: numpy_helper.to_array(t) for t in sh.initializer}
    bb_bo, sh_bo = by_out(bb), by_out(sh)

    def save(name, arr):
        np.save(os.path.join(args.out_dir, name + ".npy"),
                np.ascontiguousarray(arr, dtype=np.float32))

    save("enhancer.embed.weight", bi["backbone.embed.weight"])
    save("enhancer.embed.bias", bi["backbone.embed.bias"])
    save("enhancer.norm.weight", bi["backbone.norm.weight"])
    save("enhancer.norm.bias", bi["backbone.norm.bias"])
    for i in range(8):
        p = f"backbone.convnext.{i}"
        save(f"enhancer.block.{i}.dwconv.weight", bi[f"{p}.dwconv.weight"])
        save(f"enhancer.block.{i}.dwconv.bias", bi[f"{p}.dwconv.bias"])
        save(f"enhancer.block.{i}.norm.weight", bi[f"{p}.norm.weight"])
        save(f"enhancer.block.{i}.norm.bias", bi[f"{p}.norm.bias"])
        # Linear weights are transposed to [out, in] to match the GGUF/core.
        save(f"enhancer.block.{i}.pwconv1.weight",
             matmul_weight(bb, bi, bb_bo, f"{p}.pwconv1.bias").T)
        save(f"enhancer.block.{i}.pwconv1.bias", bi[f"{p}.pwconv1.bias"])
        save(f"enhancer.block.{i}.pwconv2.weight",
             matmul_weight(bb, bi, bb_bo, f"{p}.pwconv2.bias").T)
        save(f"enhancer.block.{i}.pwconv2.bias", bi[f"{p}.pwconv2.bias"])
        save(f"enhancer.block.{i}.gamma", bi[f"{p}.gamma"])
    save("enhancer.final_norm.weight", bi["backbone.final_layer_norm.weight"])
    save("enhancer.final_norm.bias", bi["backbone.final_layer_norm.bias"])
    save("spec_head.out.weight", matmul_weight(sh, si, sh_bo, "out.bias").T)
    save("spec_head.out.bias", si["out.bias"])

    rng = np.random.RandomState(args.seed)
    T = args.frames
    mel = (rng.randn(80, T).astype(np.float32)) * 2.0 - 4.0
    bb_sess = ort.InferenceSession(bb_path, providers=["CPUExecutionProvider"])
    sh_sess = ort.InferenceSession(sh_path, providers=["CPUExecutionProvider"])
    hidden = bb_sess.run(None, {bb_sess.get_inputs()[0].name: mel[None]})[0]
    real, imag = sh_sess.run(None, {sh_sess.get_inputs()[0].name: hidden})
    save("mel", mel)
    save("real", real[0])
    save("imag", imag[0])

    # End-to-end golden: a deterministic 24 kHz signal (low tone + a high tone
    # above the engine Nyquist so the FastLR high band is exercised) run through
    # the full numpy reference pipeline. The C++ test compares enhance(pcm_in)
    # against this, covering resample + mel + ISTFT + FastLR, not just the core.
    sr_in = 24000
    n = sr_in // 4  # 0.25 s
    ts = np.arange(n) / sr_in
    pcm_in = (0.3 * np.sin(2 * np.pi * 220.0 * ts)
              + 0.15 * np.sin(2 * np.pi * 3500.0 * ts)).astype(np.float32)
    enhanced_48k = enhance_ref(pcm_in, sr_in, bb_sess, sh_sess)
    save("pcm_in", pcm_in)
    save("enhanced_48k", enhanced_48k)

    print(f"Wrote enhancer fixtures to {args.out_dir} "
          f"(core golden T={T}; e2e golden pcm_in={pcm_in.size} -> "
          f"enhanced_48k={enhanced_48k.size})")


if __name__ == "__main__":
    main()
