#!/usr/bin/env python3
"""Dump Supertonic 2 ONNX Runtime reference tensors for ggml parity tests.

The C++ port should treat this script as the source of truth for stage inputs
and outputs.  It mirrors the official Supertone Python helper semantics:
text normalization, language tags, speed-adjusted duration, latent masking,
and the vector-estimator loop where ONNX returns the next latent directly.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from unicodedata import normalize
import wave

import numpy as np
import onnxruntime as ort


AVAILABLE_LANGS = ("en", "ko", "es", "pt", "fr")
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump Supertonic 2 ONNX reference tensors.")
    p.add_argument("--onnx-dir", type=Path, required=True,
                   help="Directory containing duration_predictor.onnx, text_encoder.onnx, "
                        "vector_estimator.onnx, vocoder.onnx, and tts.json.")
    p.add_argument("--assets-dir", type=Path, default=None,
                   help="Directory containing unicode_indexer.json and voice_styles/. "
                        "Defaults to --onnx-dir when present, otherwise ../../assets "
                        "relative to --onnx-dir.")
    p.add_argument("--voice-style", type=Path, default=None,
                   help="Voice style JSON. Defaults to <onnx-dir>/voice_styles/M1.json.")
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--lang", default="en", choices=AVAILABLE_LANGS)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--speed", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-language-wrap", action="store_true",
                   help="Do not wrap text as <lang>... . Use for English-only Supertone/supertonic.")
    p.add_argument("--language-wrap-mode", choices=("none", "prefix", "open_close"), default="open_close",
                   help="How to wrap text for multilingual models. Default matches latest QVAC supertonic-2.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--providers", default="CPUExecutionProvider",
                   help="Comma-separated ONNX Runtime providers.")
    p.add_argument("--write-wav", action="store_true",
                   help="Also write reference.wav scaled to int16 for listening.")
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
        "–": "-",
        "‑": "-",
        "—": "-",
        "_": " ",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "´": "'",
        "`": "'",
        "[": " ",
        "]": " ",
        "|": " ",
        "/": " ",
        "#": " ",
        "→": " ",
        "←": " ",
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


def text_to_ids(text: str, lang: str, unicode_indexer: list[int],
                language_wrap_mode: str = "open_close") -> tuple[np.ndarray, np.ndarray, str]:
    normalized = preprocess_text(text, lang, language_wrap_mode=language_wrap_mode)
    ids = []
    for ch in normalized:
        cp = ord(ch)
        if cp >= len(unicode_indexer) or unicode_indexer[cp] < 0:
            raise ValueError(f"unsupported character after preprocessing: {ch!r} U+{cp:04X}")
        ids.append(unicode_indexer[cp])
    text_ids = np.asarray(ids, dtype=np.int64)[None, :]
    text_mask = np.ones((1, 1, len(ids)), dtype=np.float32)
    return text_ids, text_mask, normalized


def load_voice_style(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    style_ttl = np.asarray(data["style_ttl"]["data"], dtype=np.float32)
    style_dp = np.asarray(data["style_dp"]["data"], dtype=np.float32)
    return style_ttl, style_dp


def length_to_mask(lengths: np.ndarray, max_len: int | None = None) -> np.ndarray:
    max_len = int(max_len or lengths.max())
    ids = np.arange(max_len)
    return (ids < lengths[:, None]).astype(np.float32).reshape(-1, 1, max_len)


def latent_mask_for_duration(duration: np.ndarray, sample_rate: int,
                             base_chunk_size: int, chunk_compress_factor: int) -> tuple[int, np.ndarray]:
    wav_lengths = (duration * sample_rate).astype(np.int64)
    chunk_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + chunk_size - 1) // chunk_size
    latent_len = int(latent_lengths.max())
    return latent_len, length_to_mask(latent_lengths, latent_len)


def save_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    peak = max(float(np.max(np.abs(wav))), 1e-6)
    pcm = (wav / peak * 0.95 * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

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
    if not voice_style_path.exists() and (args.onnx_dir.parent / "voice_styles" / "F1.json").exists():
        voice_style_path = args.onnx_dir.parent / "voice_styles" / "F1.json"
    unicode_path = assets_dir / "unicode_indexer.json"
    if not unicode_path.exists() and (args.onnx_dir / "unicode_indexer.json").exists():
        unicode_path = args.onnx_dir / "unicode_indexer.json"
    unicode_indexer = json.loads(unicode_path.read_text())
    wrap_mode = "none" if args.no_language_wrap else args.language_wrap_mode
    text_ids, text_mask, normalized_text = text_to_ids(
        args.text, args.lang, unicode_indexer, language_wrap_mode=wrap_mode)
    style_ttl, style_dp = load_voice_style(voice_style_path)

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    sess_opts = ort.SessionOptions()
    duration_sess = ort.InferenceSession(str(args.onnx_dir / "duration_predictor.onnx"),
                                         sess_options=sess_opts, providers=providers)
    text_sess = ort.InferenceSession(str(args.onnx_dir / "text_encoder.onnx"),
                                     sess_options=sess_opts, providers=providers)
    vector_sess = ort.InferenceSession(str(args.onnx_dir / "vector_estimator.onnx"),
                                       sess_options=sess_opts, providers=providers)
    vocoder_sess = ort.InferenceSession(str(args.onnx_dir / "vocoder.onnx"),
                                        sess_options=sess_opts, providers=providers)

    duration_raw = duration_sess.run(None, {
        "text_ids": text_ids,
        "style_dp": style_dp,
        "text_mask": text_mask,
    })[0].astype(np.float32)
    duration = (duration_raw / np.float32(args.speed)).astype(np.float32)

    text_emb = text_sess.run(None, {
        "text_ids": text_ids,
        "style_ttl": style_ttl,
        "text_mask": text_mask,
    })[0].astype(np.float32)

    latent_len, latent_mask = latent_mask_for_duration(
        duration, sample_rate, base_chunk_size, chunk_compress_factor)
    # Use MT19937-compatible RandomState so C++ can reproduce legacy NumPy if needed.
    np.random.seed(args.seed)
    noise = np.random.randn(1, latent_dim, latent_len).astype(np.float32)
    xt = noise * latent_mask

    step_outputs = []
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
        step_outputs.append(xt.copy())

    wav_full = vocoder_sess.run(None, {"latent": xt * latent_mask})[0].astype(np.float32)
    wav_samples = int(float(duration[0]) * sample_rate)
    wav_trimmed = wav_full.reshape(-1)[:wav_samples].astype(np.float32)

    np.save(args.out / "text_ids.npy", np.ascontiguousarray(text_ids))
    np.save(args.out / "text_mask.npy", np.ascontiguousarray(text_mask))
    np.save(args.out / "style_ttl.npy", np.ascontiguousarray(style_ttl))
    np.save(args.out / "style_dp.npy", np.ascontiguousarray(style_dp))
    np.save(args.out / "duration_raw.npy", np.ascontiguousarray(duration_raw))
    np.save(args.out / "duration.npy", np.ascontiguousarray(duration))
    np.save(args.out / "text_emb.npy", np.ascontiguousarray(text_emb))
    np.save(args.out / "noise.npy", np.ascontiguousarray(noise))
    np.save(args.out / "latent_mask.npy", np.ascontiguousarray(latent_mask))
    for i, value in enumerate(step_outputs):
        np.save(args.out / f"vector_step_{i:02d}.npy", np.ascontiguousarray(value))
    np.save(args.out / "final_latent.npy", np.ascontiguousarray(xt))
    np.save(args.out / "wav_full.npy", np.ascontiguousarray(wav_full))
    np.save(args.out / "wav.npy", np.ascontiguousarray(wav_trimmed))

    meta = {
        "text": args.text,
        "normalized_text": normalized_text,
        "language": args.lang,
        "voice_style": str(voice_style_path),
        "steps": args.steps,
        "speed": args.speed,
        "seed": args.seed,
        "sample_rate": sample_rate,
        "base_chunk_size": base_chunk_size,
        "chunk_compress_factor": chunk_compress_factor,
        "latent_dim": latent_dim,
        "latent_len": latent_len,
        "duration_s": float(duration[0]),
        "wav_samples": int(wav_trimmed.shape[0]),
        "providers": providers,
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    if args.write_wav:
        save_wav(args.out / "reference.wav", wav_trimmed, sample_rate)

    print(f"Wrote Supertonic reference to {args.out}")
    print(f"  text ids: {text_ids.shape}, latent: {xt.shape}, wav: {wav_trimmed.shape}")
    print(f"  duration: {float(duration[0]):.3f}s, sample_rate: {sample_rate}")


if __name__ == "__main__":
    main()
