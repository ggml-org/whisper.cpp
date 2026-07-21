#!/usr/bin/env python3
"""Benchmark the Supertonic 2 ONNX Runtime pipeline.

Mirrors `dump-supertonic-reference.py` but instruments per-stage wall time so
the C++ GGML port (`build/supertonic-bench`) can be compared apples-to-apples.

Stages:
  preprocess      - text normalization + unicode_indexer lookup
  duration        - duration_predictor.onnx
  text_encoder    - text_encoder.onnx
  vector_estimator- N x vector_estimator.onnx
  vocoder         - vocoder.onnx
  total           - sum of the above

Usage:
  python scripts/bench-supertonic-onnx.py \
      --onnx-dir /path/to/onnx_models/onnx \
      --text "..." [--lang en] [--steps 5] [--speed 1.05] [--seed 42] \
      [--noise-npy noise.npy] [--runs 5] [--warmup 1] \
      [--providers CPUExecutionProvider] [--language-wrap-mode open_close] [--json-out result.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from unicodedata import normalize

import numpy as np
import onnxruntime as ort


AVAILABLE_LANGS = ("en", "ko", "es", "pt", "fr")
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Supertonic 2 ONNX pipeline.")
    p.add_argument("--onnx-dir", type=Path, required=True)
    p.add_argument("--assets-dir", type=Path, default=None)
    p.add_argument("--voice-style", type=Path, default=None)
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--lang", default="en", choices=AVAILABLE_LANGS)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--speed", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-npy", type=Path, default=None,
                   help="Optional fixed noise tensor [1, latent_dim, L] in float32.")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--providers", default="CPUExecutionProvider")
    p.add_argument("--threads", type=int, default=None,
                   help="Set intra/inter op thread counts on the ONNX Runtime session.")
    p.add_argument("--language-wrap-mode", default="open_close",
                   choices=("none", "prefix", "open_close"),
                   help="Text language wrapping mode. Use open_close for Supertonic 2 quality parity.")
    p.add_argument("--json-out", type=Path, default=None,
                   help="Optional path to write structured benchmark metrics.")
    return p.parse_args()


def preprocess_text(text: str, lang: str, language_wrap_mode: str = "open_close") -> str:
    if lang not in AVAILABLE_LANGS:
        raise ValueError(f"invalid language: {lang}")
    text = normalize("NFKD", text)
    emoji_pattern = re.compile(
        "[\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f700-\U0001f77f"
        "\U0001f780-\U0001f7ff"
        "\U0001f800-\U0001f8ff"
        "\U0001f900-\U0001f9ff"
        "\U0001fa00-\U0001fa6f"
        "\U0001fa70-\U0001faff"
        "\u2600-\u26ff"
        "\u2700-\u27bf"
        "\U0001f1e6-\U0001f1ff]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    replacements = {
        "–": "-", "‑": "-", "—": "-", "_": " ",
        "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "´": "'", "`": "'",
        "[": " ", "]": " ", "|": " ", "/": " ", "#": " ",
        "→": " ", "←": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"[♥☆♡©\\]", "", text)
    for src, dst in {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}.items():
        text = text.replace(src, dst)
    text = re.sub(r" ,", ",", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r" !", "!", text)
    text = re.sub(r" \?", "?", text)
    text = re.sub(r" ;", ";", text)
    text = re.sub(r" :", ":", text)
    text = re.sub(r" '", "'", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
        text += "."
    if language_wrap_mode == "none":
        return text
    if language_wrap_mode == "prefix":
        return f"<{lang}>{text} "
    if language_wrap_mode == "open_close":
        return f"<{lang}>{text}</{lang}>"
    raise ValueError(f"invalid language wrap mode: {language_wrap_mode}")


def text_to_ids(text: str, lang: str, unicode_indexer: list[int], language_wrap_mode: str):
    normalized = preprocess_text(text, lang, language_wrap_mode=language_wrap_mode)
    ids = []
    for ch in normalized:
        cp = ord(ch)
        if cp >= len(unicode_indexer) or unicode_indexer[cp] < 0:
            raise ValueError(f"unsupported character: {ch!r} U+{cp:04X}")
        ids.append(unicode_indexer[cp])
    text_ids = np.asarray(ids, dtype=np.int64)[None, :]
    text_mask = np.ones((1, 1, len(ids)), dtype=np.float32)
    return text_ids, text_mask, normalized


def load_voice_style(path: Path):
    data = json.loads(path.read_text())
    style_ttl = np.asarray(data["style_ttl"]["data"], dtype=np.float32)
    style_dp = np.asarray(data["style_dp"]["data"], dtype=np.float32)
    return style_ttl, style_dp


def stats(name: str, samples: list[float]) -> str:
    if not samples:
        return f"  {name:<26s} n=0"
    s = sorted(samples)
    n = len(s)
    def pct(p):
        idx = p * (n - 1)
        lo = int(idx); hi = min(lo + 1, n - 1)
        return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)
    mean = sum(s) / n
    return (f"  {name:<26s} n={n}  min={s[0]*1000:7.2f}  med={pct(0.5)*1000:7.2f}  "
            f"mean={mean*1000:7.2f}  p95={pct(0.95)*1000:7.2f}  max={s[-1]*1000:7.2f}  ms")


def metric_dict(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        return {"n": 0, "min_ms": 0.0, "median_ms": 0.0, "mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    s = sorted(samples)
    n = len(s)
    def pct(p: float) -> float:
        idx = p * (n - 1)
        lo = int(idx); hi = min(lo + 1, n - 1)
        return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)
    return {
        "n": n,
        "min_ms": s[0] * 1000.0,
        "median_ms": pct(0.5) * 1000.0,
        "mean_ms": (sum(s) / n) * 1000.0,
        "p95_ms": pct(0.95) * 1000.0,
        "max_ms": s[-1] * 1000.0,
    }


def main() -> None:
    args = parse_args()

    cfg = json.loads((args.onnx_dir / "tts.json").read_text())
    sample_rate = int(cfg["ae"]["sample_rate"])
    base_chunk_size = int(cfg["ae"]["base_chunk_size"])
    chunk_compress_factor = int(cfg["ttl"]["chunk_compress_factor"])
    latent_dim = int(cfg["ttl"]["latent_dim"]) * chunk_compress_factor

    if args.assets_dir is not None:
        assets_dir = args.assets_dir
    elif (args.onnx_dir / "unicode_indexer.json").exists():
        assets_dir = args.onnx_dir
    else:
        assets_dir = args.onnx_dir.parent.parent / "assets"

    voice_style_path = args.voice_style or (assets_dir / "voice_styles" / "M1.json")
    unicode_indexer = json.loads((assets_dir / "unicode_indexer.json").read_text())
    style_ttl, style_dp = load_voice_style(voice_style_path)

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    sess_opts = ort.SessionOptions()
    if args.threads is not None:
        sess_opts.intra_op_num_threads = args.threads
        sess_opts.inter_op_num_threads = args.threads

    duration_sess = ort.InferenceSession(str(args.onnx_dir / "duration_predictor.onnx"),
                                         sess_options=sess_opts, providers=providers)
    text_sess = ort.InferenceSession(str(args.onnx_dir / "text_encoder.onnx"),
                                     sess_options=sess_opts, providers=providers)
    vector_sess = ort.InferenceSession(str(args.onnx_dir / "vector_estimator.onnx"),
                                       sess_options=sess_opts, providers=providers)
    vocoder_sess = ort.InferenceSession(str(args.onnx_dir / "vocoder.onnx"),
                                        sess_options=sess_opts, providers=providers)

    fixed_noise = None
    if args.noise_npy is not None:
        fixed_noise = np.load(args.noise_npy).astype(np.float32)

    pre_t = []
    dur_t = []
    txt_t = []
    vec_t = []
    voc_t = []
    tot_t = []
    audio_s_last = 0.0

    total_runs = args.runs + args.warmup
    for r in range(total_runs):
        record = r >= args.warmup
        t0 = time.perf_counter()
        text_ids, text_mask, normalized = text_to_ids(
            args.text, args.lang, unicode_indexer, args.language_wrap_mode)
        t1 = time.perf_counter()

        duration_raw = duration_sess.run(None, {
            "text_ids": text_ids,
            "style_dp": style_dp,
            "text_mask": text_mask,
        })[0].astype(np.float32)
        duration = (duration_raw / np.float32(args.speed)).astype(np.float32)
        t2 = time.perf_counter()

        text_emb = text_sess.run(None, {
            "text_ids": text_ids,
            "style_ttl": style_ttl,
            "text_mask": text_mask,
        })[0].astype(np.float32)
        t3 = time.perf_counter()

        wav_lengths = (duration * sample_rate).astype(np.int64)
        chunk_size = base_chunk_size * chunk_compress_factor
        latent_lengths = (wav_lengths + chunk_size - 1) // chunk_size
        latent_len = int(latent_lengths.max())
        ids = np.arange(latent_len)
        latent_mask = (ids < latent_lengths[:, None]).astype(np.float32).reshape(-1, 1, latent_len)

        if fixed_noise is not None:
            xt = fixed_noise * latent_mask
            latent_len = fixed_noise.shape[2]
        else:
            np.random.seed(args.seed + r)
            noise = np.random.randn(1, latent_dim, latent_len).astype(np.float32)
            xt = noise * latent_mask

        total_step = np.asarray([args.steps], dtype=np.float32)
        for step in range(args.steps):
            current_step = np.asarray([step], dtype=np.float32)
            xt = vector_sess.run(None, {
                "noisy_latent": xt,
                "text_emb": text_emb,
                "style_ttl": style_ttl,
                "latent_mask": latent_mask,
                "text_mask": text_mask,
                "current_step": current_step,
                "total_step": total_step,
            })[0].astype(np.float32)
        t4 = time.perf_counter()

        wav = vocoder_sess.run(None, {"latent": xt * latent_mask})[0].astype(np.float32)
        t5 = time.perf_counter()

        audio_s = wav.size / sample_rate
        if record:
            pre_t.append(t1 - t0)
            dur_t.append(t2 - t1)
            txt_t.append(t3 - t2)
            vec_t.append(t4 - t3)
            voc_t.append(t5 - t4)
            tot_t.append(t5 - t0)
            audio_s_last = audio_s

        sys.stderr.write(
            f"[run {r+1}/{total_runs}]{' (warmup)' if not record else ''} "
            f"total={(t5-t0)*1000:.1f}ms audio={audio_s:.2f}s RTF={(t5-t0)/audio_s:.3f}\n")

    print()
    print("Supertonic 2 ONNX Runtime benchmark")
    print(f"  text length: {len(args.text)} chars")
    print(f"  voice style: {voice_style_path}")
    print(f"  language: {args.lang}, steps: {args.steps}, speed: {args.speed:.2f}")
    print(f"  language wrap: {args.language_wrap_mode}")
    print(f"  audio per run: {audio_s_last:.3f}s @ {sample_rate} Hz")
    print(f"  providers: {providers}, threads: {args.threads}")
    print(f"  runs: {args.runs} (warmup discarded: {args.warmup})")
    print()
    print(stats("preprocess", pre_t))
    print(stats("duration", dur_t))
    print(stats("text_encoder", txt_t))
    print(stats(f"vector_estimator ({args.steps} step)", vec_t))
    print(stats("vocoder", voc_t))
    print(stats("total", tot_t))
    if tot_t:
        rtfs = [t / audio_s_last for t in tot_t]
        rtfs_sorted = sorted(rtfs)
        med = rtfs_sorted[len(rtfs_sorted) // 2]
        print()
        print(f"  RTF (total / audio):    min={min(rtfs):.3f}  med={med:.3f}  "
              f"mean={sum(rtfs)/len(rtfs):.3f}  max={max(rtfs):.3f}")
        print(f"  Real-time multiplier:   med={1/med:.2f}x "
              f"(1 second of audio per {sorted(tot_t)[len(tot_t)//2]/audio_s_last*1000:.2f} ms)")
        if args.json_out is not None:
            args.json_out.write_text(json.dumps({
                "runtime": "onnxruntime",
                "onnx_dir": str(args.onnx_dir),
                "text_length": len(args.text),
                "voice_style": str(voice_style_path),
                "language": args.lang,
                "language_wrap_mode": args.language_wrap_mode,
                "steps": args.steps,
                "speed": args.speed,
                "providers": providers,
                "threads": args.threads,
                "audio_s": audio_s_last,
                "runs": args.runs,
                "warmup": args.warmup,
                "rtf": {
                    "min": min(rtfs),
                    "median": med,
                    "mean": sum(rtfs) / len(rtfs),
                    "max": max(rtfs),
                },
                "stages": {
                    "preprocess": metric_dict(pre_t),
                    "duration": metric_dict(dur_t),
                    "text_encoder": metric_dict(txt_t),
                    f"vector_estimator ({args.steps} step)": metric_dict(vec_t),
                    "vocoder": metric_dict(voc_t),
                    "total": metric_dict(tot_t),
                },
            }, indent=2) + "\n")


if __name__ == "__main__":
    main()
