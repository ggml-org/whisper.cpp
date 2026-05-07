#!/usr/bin/env python3
"""Convert official Supertonic 2 ONNX/assets into a single GGUF file.

This is intentionally model-specific.  The GGUF stores every ONNX initializer
and tensor-valued Constant under short ggml-safe names, plus metadata arrays
mapping those short names back to their source ONNX names.  The C++ runtime can
therefore ask for a tensor by its original ONNX source name without relying on
long ggml tensor names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import numpy_helper

try:
    import gguf
except ImportError as exc:  # pragma: no cover - user environment guard
    raise SystemExit("error: Python package 'gguf' is required; install with `pip install gguf`.") from exc


STAGES = (
    ("duration", "duration_predictor.onnx"),
    ("text_encoder", "text_encoder.onnx"),
    ("vector_estimator", "vector_estimator.onnx"),
    ("vocoder", "vocoder.onnx"),
)
REQUIRED_ONNX = tuple(filename for _, filename in STAGES)
HF_ALLOW_PATTERNS = (
    "*.onnx",
    "*.json",
    "*.bin",
    "*.data",
    "**/*.onnx",
    "**/*.json",
    "**/*.bin",
    "**/*.data",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Supertonic 2 ONNX/assets to GGUF.")
    p.add_argument("--onnx-dir", type=Path, default=None,
                   help="Directory containing the four Supertonic ONNX files and tts.json. "
                        "If omitted, downloads --repo-id from Hugging Face first.")
    p.add_argument("--assets-dir", type=Path, default=None,
                   help="Directory containing unicode_indexer.json and voice_styles/. "
                        "Defaults to --onnx-dir if present, otherwise ../../assets relative to --onnx-dir.")
    p.add_argument("--out", type=Path, default=Path("models/supertonic2.gguf"))
    p.add_argument("--arch", default="supertonic2", choices=("supertonic", "supertonic2"),
                   help="Model family metadata. Use 'supertonic' for the English-only HF bundle.")
    p.add_argument("--repo-id", default=None,
                   help="Hugging Face repo to download when --onnx-dir is omitted. "
                        "Defaults to Supertone/supertonic-2 or Supertone/supertonic based on --arch.")
    p.add_argument("--download-dir", type=Path, default=None,
                   help="Optional local directory for the Hugging Face snapshot download.")
    p.add_argument("--hf-token", default=None, help="Optional Hugging Face token.")
    p.add_argument("--local-files-only", action="store_true",
                   help="Use only the local Hugging Face cache when downloading.")
    p.add_argument("--reference-repo", default=None,
                   help="HF repo/source metadata. Defaults from --arch.")
    p.add_argument("--default-voice", default=None,
                   help="Default voice metadata. Defaults to F1 when present, otherwise first voice.")
    p.add_argument("--default-steps", type=int, default=None,
                   help="Default denoising steps metadata. Defaults to 5 to match reference dumps and examples.")
    p.add_argument("--default-speed", type=float, default=1.05,
                   help="Default speed metadata.")
    p.add_argument("--ftype", choices=("f32", "f16", "q8_0"), default="f32",
                   help="Weight storage type. f32 is required by the current scalar reference backend; "
                        "f16/q8_0 are intended for the GGML graph backend.")
    p.add_argument("--language-wrap-mode", choices=("none", "prefix", "open_close"), default=None,
                   help="Text wrapping metadata. Defaults to none for --arch supertonic and open_close for supertonic2.")
    p.add_argument("--no-language-wrap", action="store_true",
                   help="Store metadata telling runtimes not to wrap text as <lang>... . "
                        "Use for the English-only Supertone/supertonic bundle.")
    p.add_argument("--validate", action="store_true",
                   help="Re-open the written GGUF and validate tensor count + metadata.")
    return p.parse_args()


def default_repo_for_arch(arch: str) -> str:
    return "Supertone/supertonic" if arch == "supertonic" else "Supertone/supertonic-2"


def download_hf_snapshot(repo_id: str,
                         token: str | None,
                         download_dir: Path | None,
                         local_files_only: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - user environment guard
        raise SystemExit(
            "error: Python package 'huggingface_hub' is required for automatic download; "
            "install with `pip install huggingface_hub` or pass --onnx-dir."
        ) from exc

    kwargs = {
        "repo_id": repo_id,
        "token": token,
        "allow_patterns": list(HF_ALLOW_PATTERNS),
        "local_files_only": local_files_only,
    }
    if download_dir is not None:
        kwargs["local_dir"] = str(download_dir)
    return Path(snapshot_download(**kwargs))


def contains_required_onnx(path: Path) -> bool:
    return all((path / filename).exists() for filename in REQUIRED_ONNX)


def resolve_onnx_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root / "onnx_models" / "onnx",
        repo_root / "onnx",
        repo_root / "onnx_models",
        repo_root,
    ]
    for candidate in candidates:
        if contains_required_onnx(candidate):
            return candidate

    for duration_path in repo_root.rglob("duration_predictor.onnx"):
        candidate = duration_path.parent
        if contains_required_onnx(candidate):
            return candidate

    required = ", ".join(REQUIRED_ONNX)
    raise FileNotFoundError(f"could not find Supertonic ONNX directory under {repo_root}; required: {required}")


def resolve_tts_json(onnx_dir: Path, repo_root: Path | None) -> Path:
    candidates = [onnx_dir / "tts.json"]
    if repo_root is not None:
        candidates.extend([
            repo_root / "tts.json",
            repo_root / "onnx_models" / "onnx" / "tts.json",
            repo_root / "onnx" / "tts.json",
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"tts.json not found near {onnx_dir}")


def resolve_assets_dir(onnx_dir: Path, assets_dir: Path | None, repo_root: Path | None = None) -> Path:
    if assets_dir is not None:
        return assets_dir
    if (onnx_dir / "unicode_indexer.json").exists():
        return onnx_dir
    if repo_root is not None and (repo_root / "assets").exists():
        return repo_root / "assets"
    if (onnx_dir.parent / "assets").exists():
        return onnx_dir.parent / "assets"
    return onnx_dir.parent.parent / "assets"


def resolve_unicode_indexer(onnx_dir: Path, assets_dir: Path, repo_root: Path | None = None) -> Path:
    candidates = [assets_dir / "unicode_indexer.json", onnx_dir / "unicode_indexer.json"]
    if repo_root is not None:
        candidates.extend([repo_root / "unicode_indexer.json", repo_root / "assets" / "unicode_indexer.json"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"unicode_indexer.json not found under {assets_dir} or {onnx_dir}")


def resolve_voice_styles_dir(onnx_dir: Path, assets_dir: Path, repo_root: Path | None = None) -> Path:
    candidates = [assets_dir / "voice_styles", onnx_dir / "voice_styles", onnx_dir.parent / "voice_styles"]
    if repo_root is not None:
        candidates.extend([repo_root / "voice_styles", repo_root / "assets" / "voice_styles"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"voice_styles/ not found under {assets_dir}, {onnx_dir}, or {onnx_dir.parent}")


def as_contiguous(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    # GGUF stores int64 tensors, but int32 is easier for ggml consumers when
    # values are small ids/shapes.  Leave true int64 if narrowing would change data.
    if arr.dtype == np.int64:
        narrowed = arr.astype(np.int32)
        if np.array_equal(arr, narrowed.astype(np.int64)):
            arr = narrowed
    return np.ascontiguousarray(arr)


def tensor_sha256(arr: np.ndarray) -> str:
    data = np.ascontiguousarray(arr).view(np.uint8)
    return hashlib.sha256(data).hexdigest()


def prepare_weight_tensor(arr: np.ndarray, ftype: str) -> tuple[np.ndarray, tuple[int, ...] | None, "gguf.GGMLQuantizationType | None"]:
    if ftype == "f32" or not np.issubdtype(arr.dtype, np.floating):
        return arr, None, None
    if ftype == "f16":
        return np.ascontiguousarray(arr.astype(np.float16)), None, None
    if ftype == "q8_0":
        # Keep small/vector tensors in F32. Quantizing bias/norm/scalar tensors
        # hurts parity and gives little size/speed benefit.
        if arr.ndim < 2 or arr.size < 256:
            return arr, None, None
        qtype = gguf.GGMLQuantizationType.Q8_0
        try:
            q = gguf.quantize(np.ascontiguousarray(arr.astype(np.float32)), qtype)
        except gguf.QuantError:
            return arr, None, None
        return q, None, qtype
    raise ValueError(f"unsupported ftype: {ftype}")


def tensor_from_attribute(attr: onnx.AttributeProto) -> np.ndarray | None:
    if attr.type == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    if attr.type == onnx.AttributeProto.FLOAT:
        return np.asarray([attr.f], dtype=np.float32)
    if attr.type == onnx.AttributeProto.FLOATS:
        return np.asarray(attr.floats, dtype=np.float32)
    if attr.type == onnx.AttributeProto.INT:
        return np.asarray([attr.i], dtype=np.int32)
    if attr.type == onnx.AttributeProto.INTS:
        return np.asarray(attr.ints, dtype=np.int32)
    return None


def iter_onnx_tensors(model_path: Path) -> Iterable[tuple[str, np.ndarray]]:
    model = onnx.load(str(model_path), load_external_data=True)
    seen: set[str] = set()

    for init in model.graph.initializer:
        name = init.name
        if not name:
            continue
        arr = numpy_helper.to_array(init)
        seen.add(name)
        yield name, as_contiguous(arr)

    for node_idx, node in enumerate(model.graph.node):
        if node.op_type != "Constant":
            continue
        if not node.output:
            continue
        out_name = node.output[0]
        if not out_name or out_name in seen:
            continue
        for attr in node.attribute:
            arr = tensor_from_attribute(attr)
            if arr is None:
                continue
            seen.add(out_name)
            yield out_name, as_contiguous(arr)
            break


def add_json_metadata(writer: "gguf.GGUFWriter", prefix: str, data: dict) -> None:
    writer.add_string(prefix, json.dumps(data, ensure_ascii=False, separators=(",", ":")))


def main() -> int:
    args = parse_args()
    repo_root: Path | None = None
    repo_id = args.repo_id or default_repo_for_arch(args.arch)
    if args.onnx_dir is None:
        print(f"Downloading {repo_id} from Hugging Face (cached by huggingface_hub)")
        repo_root = download_hf_snapshot(repo_id, args.hf_token, args.download_dir, args.local_files_only)
        args.onnx_dir = resolve_onnx_dir(repo_root)
    else:
        args.onnx_dir = args.onnx_dir.resolve()

    assets_dir = resolve_assets_dir(args.onnx_dir, args.assets_dir, repo_root)
    unicode_path = resolve_unicode_indexer(args.onnx_dir, assets_dir, repo_root)
    voice_styles_dir = resolve_voice_styles_dir(args.onnx_dir, assets_dir, repo_root)
    tts_json_path = resolve_tts_json(args.onnx_dir, repo_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using ONNX directory: {args.onnx_dir}")
    print(f"Using assets directory: {assets_dir}")
    cfg = json.loads(tts_json_path.read_text())
    unicode_indexer = np.asarray(json.loads(unicode_path.read_text()), dtype=np.int32)

    reference_repo = args.reference_repo or repo_id
    writer = gguf.GGUFWriter(str(args.out), args.arch)
    writer.add_name("Supertonic" if args.arch == "supertonic" else "Supertonic 2")
    writer.add_description(f"{reference_repo} ONNX weights/assets converted for a model-specific ggml runtime.")
    writer.add_string("supertonic.arch", args.arch)
    writer.add_string("supertonic.reference_repo", reference_repo)
    writer.add_string("supertonic.ftype", args.ftype)
    writer.add_string("supertonic.tts_version", str(cfg.get("tts_version", "")))
    writer.add_string("supertonic.split", str(cfg.get("split", "")))
    writer.add_uint32("supertonic.sample_rate", int(cfg["ae"]["sample_rate"]))
    writer.add_uint32("supertonic.base_chunk_size", int(cfg["ae"]["base_chunk_size"]))
    writer.add_uint32("supertonic.ttl_chunk_compress_factor", int(cfg["ttl"]["chunk_compress_factor"]))
    writer.add_uint32("supertonic.latent_dim", int(cfg["ttl"]["latent_dim"]))
    writer.add_uint32(
        "supertonic.latent_channels",
        int(cfg["ttl"]["latent_dim"]) * int(cfg["ttl"]["chunk_compress_factor"]),
    )
    wrap_mode = "none" if args.no_language_wrap else (args.language_wrap_mode or ("none" if args.arch == "supertonic" else "open_close"))
    default_steps = args.default_steps if args.default_steps is not None else 5

    writer.add_uint32("supertonic.default_steps", default_steps)
    writer.add_float32("supertonic.default_speed", args.default_speed)
    writer.add_uint32("supertonic.language_wrap", 0 if wrap_mode == "none" else 1)
    writer.add_string("supertonic.language_wrap_mode", wrap_mode)
    writer.add_array("supertonic.languages", ["en", "ko", "es", "pt", "fr"])
    add_json_metadata(writer, "supertonic.tts_json", cfg)

    writer.add_tensor("supertonic/unicode_indexer", unicode_indexer)

    voice_names: list[str] = []
    for voice_path in sorted(voice_styles_dir.glob("*.json")):
        voice_name = voice_path.stem
        voice = json.loads(voice_path.read_text())
        ttl = as_contiguous(np.asarray(voice["style_ttl"]["data"], dtype=np.float32))
        dp = as_contiguous(np.asarray(voice["style_dp"]["data"], dtype=np.float32))
        writer.add_tensor(f"supertonic/voices/{voice_name}/ttl", ttl)
        writer.add_tensor(f"supertonic/voices/{voice_name}/dp", dp)
        writer.add_string(f"supertonic.voice.{voice_name}.metadata",
                          json.dumps(voice.get("metadata", {}), ensure_ascii=False, separators=(",", ":")))
        voice_names.append(voice_name)
    writer.add_array("supertonic.voice_names", voice_names)
    default_voice = args.default_voice or ("F1" if "F1" in voice_names else (voice_names[0] if voice_names else ""))
    writer.add_string("supertonic.default_voice", default_voice)

    tensor_names: list[str] = []
    source_names: list[str] = []
    tensor_shapes: list[str] = []
    tensor_dtypes: list[str] = []
    tensor_hashes: list[str] = []
    per_stage_counts: dict[str, int] = {}
    total_bytes = 0

    for stage, filename in STAGES:
        count = 0
        for source_name, arr in iter_onnx_tensors(args.onnx_dir / filename):
            short_name = f"supertonic/{stage}/t{count:04d}"
            source_key = f"{stage}:{source_name}"
            stored, raw_shape, raw_dtype = prepare_weight_tensor(arr, args.ftype)
            writer.add_tensor(short_name, stored, raw_shape=raw_shape, raw_dtype=raw_dtype)
            tensor_names.append(short_name)
            source_names.append(source_key)
            tensor_shapes.append(json.dumps(list(arr.shape), separators=(",", ":")))
            tensor_dtypes.append(str(raw_dtype.name if raw_dtype is not None else stored.dtype))
            tensor_hashes.append(tensor_sha256(stored))
            total_bytes += stored.nbytes
            count += 1
        per_stage_counts[stage] = count
        print(f"{stage:16s} {count:5d} tensors")

    writer.add_array("supertonic.tensor_names", tensor_names)
    writer.add_array("supertonic.source_names", source_names)
    writer.add_array("supertonic.tensor_shapes", tensor_shapes)
    writer.add_array("supertonic.tensor_dtypes", tensor_dtypes)
    writer.add_array("supertonic.tensor_sha256", tensor_hashes)
    for stage, count in per_stage_counts.items():
        writer.add_uint32(f"supertonic.{stage}.tensor_count", count)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Wrote {len(tensor_names)} ONNX tensors + {1 + 2 * len(voice_names)} asset tensors")
    print(f"  output: {args.out}")
    print(f"  source tensor bytes: {total_bytes / 1e6:.1f} MB")

    if args.validate:
        reader = gguf.GGUFReader(args.out, "r")
        if len(reader.tensors) != len(tensor_names) + 1 + 2 * len(voice_names):
            raise RuntimeError(
                f"tensor count mismatch: got {len(reader.tensors)}, "
                f"expected {len(tensor_names) + 1 + 2 * len(voice_names)}"
            )
        print("Validation: tensor count OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
