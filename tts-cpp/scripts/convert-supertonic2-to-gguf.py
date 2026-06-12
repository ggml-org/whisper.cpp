#!/usr/bin/env python3
"""Convert official Supertonic (v1/v2/v3) ONNX/assets into a single GGUF file.

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
    p.add_argument("--arch", default="supertonic2", choices=("supertonic", "supertonic2", "supertonic3"),
                   help="Model family metadata. Use 'supertonic' for the English-only HF bundle, "
                        "'supertonic2' for the 5-language bundle, 'supertonic3' for the 31-language bundle.")
    p.add_argument("--repo-id", default=None,
                   help="Hugging Face repo to download when --onnx-dir is omitted. "
                        "Defaults to Supertone/supertonic, -2, or -3 based on --arch.")
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
    return {
        "supertonic": "Supertone/supertonic",
        "supertonic2": "Supertone/supertonic-2",
        "supertonic3": "Supertone/supertonic-3",
    }[arch]


# Supertonic language coverage per family.  The runtime validates user-supplied
# language codes against the GGUF `supertonic.languages` array, so this list is
# the source of truth for which `<lang>` wrappers are accepted.  v3 adds the
# language-agnostic `na` code (README: pass lang="na" when the input language is
# unknown) alongside the 31 supported languages.
ARCH_LANGUAGES = {
    "supertonic": ["en"],
    "supertonic2": ["en", "ko", "es", "pt", "fr"],
    "supertonic3": [
        "ar", "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el",
        "hi", "hu", "id", "it", "ja", "ko", "lv", "lt", "pl", "pt", "ro", "ru",
        "sk", "sl", "es", "sv", "tr", "uk", "vi", "na",
    ],
}


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


def iter_onnx_tensors_from_model(model: "onnx.ModelProto") -> Iterable[tuple[str, np.ndarray]]:
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


def iter_onnx_tensors(model_path: Path) -> Iterable[tuple[str, np.ndarray]]:
    yield from iter_onnx_tensors_from_model(onnx.load(str(model_path), load_external_data=True))


# ---------------------------------------------------------------------------
# Cross-version stable naming (Supertonic 3 support)
# ---------------------------------------------------------------------------
#
# ONNX export auto-numbers the weights that have no PyTorch parameter name —
# linear-layer weights surface as `onnx::MatMul_<NNNN>`, conv/PReLU weights as
# `onnx::Conv_<NNNN>` / `onnx::PRelu_<NNNN>`.  Those numeric ids are assigned by
# the exporter and are completely renumbered between Supertonic 2 and 3, so a
# runtime that hard-codes `onnx::MatMul_3095` cannot bind the same logical
# weight across families.  v3 additionally prefixes every node/initializer with
# the top-level module name (`vector_estimator.tts...`, `/vector_estimator/...`).
#
# To make the runtime version-independent we emit, for every tensor, a stable
# *canonical* alias derived from structure that is identical across families:
#   - dotted PyTorch paths     -> strip everything before `tts.`
#   - auto MatMul/Gemm weights -> the adjacent bias' PyTorch path with
#                                 `.bias` -> `.weight` (MatMul -> Add(bias))
#   - auto Conv/PRelu weights  -> the consuming node's (stable) name + input idx
# The GGUF stores these as `supertonic.source_aliases` (canonical key) ->
# `supertonic.source_alias_targets` (the primary `<stage>:<onnx-name>` source);
# the loader registers both keys against the same tensor.  See the C++ loader.


def _norm_tts(name: str) -> str:
    i = name.find("tts.")
    return name[i:] if i >= 0 else name


def build_graph_index(model: "onnx.ModelProto") -> tuple[set[str], dict[str, list[tuple["onnx.NodeProto", int]]]]:
    init_names = {i.name for i in model.graph.initializer}
    consumers: dict[str, list[tuple["onnx.NodeProto", int]]] = {}
    for node in model.graph.node:
        for idx, inp in enumerate(node.input):
            if inp:
                consumers.setdefault(inp, []).append((node, idx))
    return init_names, consumers


def canonical_source_name(name: str,
                          init_names: set[str],
                          consumers: dict[str, list[tuple["onnx.NodeProto", int]]]) -> str | None:
    """Stable, cross-family alias for `name`, or None when no alias is needed."""
    if "tts." in name:
        canon = _norm_tts(name)
        return canon if canon != name else None
    if "onnx::" in name and ("MatMul_" in name or "Gemm_" in name):
        for node, _ in consumers.get(name, []):
            if node.op_type not in ("MatMul", "Gemm"):
                continue
            out = node.output[0] if node.output else ""
            for cons, _ in consumers.get(out, []):
                if cons.op_type == "Add":
                    for inp in cons.input:
                        if inp in init_names and inp.endswith(".bias") and "onnx::" not in inp:
                            return _norm_tts(inp[: -len(".bias")] + ".weight")
            return f"node:{node.name}#W"
        return None
    if "onnx::" in name and ("Conv_" in name or "PRelu_" in name):
        for node, idx in consumers.get(name, []):
            return f"node:{node.name}#{idx}"
        return None
    return None


# Runtime "logical id" contract (bridge aliases).
#
# The C++ vector-estimator path references its ~36 linear weights by the
# Supertonic-2 ONNX auto-id (`vector_estimator:onnx::MatMul_<id>`).  Those ids
# are renumbered in v3, so for any non-v2 bundle we additionally emit an alias
# from the *v2 logical id* to the actual weight, keyed off the stable canonical
# (PyTorch) name.  This keeps the delicate vector-estimator scalar + graph
# paths byte-identical across families (only the genuine head-count change is
# threaded in code) while still deriving everything from stable structure.
def _build_ve_canonical_to_logical() -> dict[str, str]:
    pre = "tts.ttl.vector_field.main_blocks."
    t_linear = [3095, 3140, 3185, 3230]
    attn_q = [3101, 3146, 3191, 3236]
    attn_out = [3110, 3155, 3200, 3245]
    style_q = [3116, 3161, 3206, 3251]
    style_out = [3119, 3164, 3209, 3254]
    table: dict[str, str] = {}
    for g in range(4):
        table[f"{pre}{1 + 6 * g}.linear.linear.weight"] = f"onnx::MatMul_{t_linear[g]}"
        ab = 3 + 6 * g
        for off, role in enumerate(("W_query", "W_key", "W_value")):
            table[f"{pre}{ab}.attn.{role}.linear.weight"] = f"onnx::MatMul_{attn_q[g] + off}"
        table[f"{pre}{ab}.attn.out_fc.linear.weight"] = f"onnx::MatMul_{attn_out[g]}"
        sb = 5 + 6 * g
        for off, role in enumerate(("W_query", "W_key", "W_value")):
            table[f"{pre}{sb}.attention.{role}.linear.weight"] = f"onnx::MatMul_{style_q[g] + off}"
        table[f"{pre}{sb}.attention.out_fc.linear.weight"] = f"onnx::MatMul_{style_out[g]}"
    return table


CANONICAL_TO_LOGICAL: dict[str, dict[str, str]] = {
    "vector_estimator": _build_ve_canonical_to_logical(),
}


# Input-independent intermediate constants that the runtime fetches by their
# ONNX value name.  In Supertonic 2 the ONNX exporter constant-folded these
# into initializers / Constant nodes (so `iter_onnx_tensors` already emits
# them); in v3 they remain *live* graph outputs and must be materialised by
# running the input-independent subgraph once at convert time.  Validated
# input-independent (identical for arbitrary dummy inputs) — see QVAC-19305.
MATERIALIZE_INTERMEDIATES: dict[str, list[str]] = {
    "text_encoder": [
        "/speech_prompted_text_encoder/attention1/tanh/Tanh_output_0",
        "/speech_prompted_text_encoder/attention2/tanh/Tanh_output_0",
    ],
}

# Intermediate constants whose ONNX name carries the v3 module prefix.  Each
# entry maps a canonical name (the one the runtime asks for) to the candidate
# source names that may carry the data across families; the converter aliases
# whichever candidate it actually emitted.
INTERMEDIATE_ALIASES: dict[str, list[tuple[str, list[str]]]] = {
    "vector_estimator": [
        ("/Expand_output_0", ["/Expand_output_0", "/vector_estimator/Expand_output_0"]),
    ],
}


def make_dummy_feeds(model: "onnx.ModelProto") -> dict[str, np.ndarray]:
    """Build zero/one dummy inputs for every graph input (symbolic dims -> small
    constants).  Only used to evaluate input-independent subgraphs, so the
    concrete values are irrelevant."""
    elem_to_np = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.BOOL: np.bool_,
    }
    feeds: dict[str, np.ndarray] = {}
    for inp in model.graph.input:
        tt = inp.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else 4 for d in tt.shape.dim]
        npdt = elem_to_np.get(tt.elem_type, np.float32)
        feeds[inp.name] = np.ones(tuple(dims) if dims else (1,), dtype=npdt)
    return feeds


def materialize_intermediates(model_path: Path, names: list[str]) -> dict[str, np.ndarray]:
    """Run `model_path` once with dummy inputs, capturing `names` (assumed
    input-independent).  Returns {name: array}.  Requires onnxruntime."""
    try:
        import onnxruntime as ort  # noqa: WPS433 (lazy: only needed for v3)
    except ImportError as exc:  # pragma: no cover - user environment guard
        raise SystemExit(
            "error: materialising Supertonic 3 intermediate constants needs "
            "'onnxruntime'; install with `pip install onnxruntime`."
        ) from exc
    model = onnx.load(str(model_path), load_external_data=True)
    existing = {o.name for o in model.graph.output}
    for nm in names:
        if nm not in existing:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(nm))
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = sess.run(names, make_dummy_feeds(model))
    return {nm: as_contiguous(np.asarray(arr)) for nm, arr in zip(names, outs)}


def extract_text_convnext_dilations(onnx_path: Path) -> list[int]:
    """Per-block dwconv dilations of the text-encoder ConvNeXt stack.

    Supertonic 3 introduced a *dilated* text-encoder ConvNeXt (dilation grows
    per block, e.g. 1,1,2,2,4,4) whereas v1/v2 used dilation 1 everywhere.  The
    dilation is a structural parameter (not a weight), so the runtime cannot
    recover it from the tensors alone; we read it from the ONNX `Conv` attrs at
    convert time and stash it in metadata.  Blocks are ordered by the numeric
    index in `/text_encoder/convnext/convnext.<i>/dwconv/Conv`.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    prefix = "/text_encoder/convnext/convnext."
    suffix = "/dwconv/Conv"
    found: dict[int, int] = {}
    for node in model.graph.node:
        if node.op_type != "Conv" or not node.name.startswith(prefix) or not node.name.endswith(suffix):
            continue
        try:
            idx = int(node.name[len(prefix):-len(suffix)])
        except ValueError:
            continue
        dil = 1
        for attr in node.attribute:
            if attr.name == "dilations" and len(attr.ints) >= 1:
                dil = int(attr.ints[0])
        found[idx] = dil
    if not found:
        return []
    return [found[i] for i in sorted(found)]


def extract_cfg_scales(onnx_path: Path) -> tuple[float, float] | None:
    """Classifier-free-guidance scales baked into the vector estimator.

    Supertonic 3's vector_estimator runs the field on a batch-2 input
    (conditional + unconditional, the latter fed learned `uncond_masker`
    null tokens) and forms the velocity as
    `v = cond_scale * v_cond - uncond_scale * v_uncond`
    (e.g. 4*cond - 3*uncond, i.e. guidance scale 4).  v1/v2 have no such
    combination.  We locate the two scalar `Mul` constants that scale the
    conditional/unconditional `Slice`s of `proj_out` and return them so the
    runtime can replicate the guidance.  Returns None when the graph has no
    CFG (no `uncond_masker`), so the runtime falls back to a single pass.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    has_uncond = any("uncond_masker" in i.name for i in model.graph.initializer)
    if not has_uncond:
        return None
    prod = {o: n for n in model.graph.node for o in n.output}

    def scalar_const(name: str):
        n = prod.get(name)
        if n is not None and n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    arr = onnx.numpy_helper.to_array(a.t)
                    if arr.size == 1:
                        return float(arr.reshape(-1)[0])
        for i in model.graph.initializer:
            if i.name == name:
                arr = onnx.numpy_helper.to_array(i)
                if arr.size == 1:
                    return float(arr.reshape(-1)[0])
        return None

    # Find Sub = Mul_cond - Mul_uncond where each Mul scales a Slice by a scalar.
    for n in model.graph.node:
        if n.op_type != "Sub":
            continue
        muls = [prod.get(i) for i in n.input]
        if len(muls) != 2 or any(m is None or m.op_type != "Mul" for m in muls):
            continue
        scales = []
        slice_starts = []
        ok = True
        for m in muls:
            scalar = None
            slice_node = None
            for i in m.input:
                v = scalar_const(i)
                if v is not None:
                    scalar = v
                p = prod.get(i)
                if p is not None and p.op_type == "Slice":
                    slice_node = p
            if scalar is None or slice_node is None:
                ok = False
                break
            scales.append(scalar)
            # Slice start (input[1]) tells cond (start 0) from uncond (start B).
            start = scalar_const(slice_node.input[1]) if len(slice_node.input) > 1 else None
            slice_starts.append(start)
        if not ok:
            continue
        # cond = the Mul whose Slice starts at 0; uncond = the other.
        if slice_starts[0] == 0 or (slice_starts[1] not in (0, None)):
            cond_scale, uncond_scale = scales[0], scales[1]
        else:
            cond_scale, uncond_scale = scales[1], scales[0]
        return (cond_scale, uncond_scale)
    return None


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
    arch_display = {
        "supertonic": "Supertonic",
        "supertonic2": "Supertonic 2",
        "supertonic3": "Supertonic 3",
    }[args.arch]
    writer = gguf.GGUFWriter(str(args.out), args.arch)
    writer.add_name(arch_display)
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
    # Vector-estimator text cross-attention head count.  This is the only
    # internal topology dimension that differs across families (v1/v2: 4,
    # v3: 8 with the same head_dim=64), so the runtime reads it from metadata
    # instead of branching on arch.  Falls back to 4 for older bundles whose
    # tts.json omits the key.
    text_cond = cfg.get("ttl", {}).get("vector_field", {}).get("main_blocks", {}).get("text_cond_layer", {})
    writer.add_uint32("supertonic.vector_text_attn_heads", int(text_cond.get("n_heads", 4)))
    # Per-block dwconv dilations of the text-encoder ConvNeXt stack.  v1/v2 use
    # dilation 1 everywhere; Supertonic 3 dilates it (e.g. 1,1,2,2,4,4).  The
    # runtime reads this array and applies it per block; bundles without the key
    # (older conversions) default to dilation 1, preserving v1/v2 behaviour.
    text_convnext_dilations = extract_text_convnext_dilations(args.onnx_dir / "text_encoder.onnx")
    if text_convnext_dilations:
        writer.add_array(
            "supertonic.text_convnext_dilations",
            [int(d) for d in text_convnext_dilations],
        )
        print(f"text-encoder ConvNeXt dilations: {text_convnext_dilations}")
    # Classifier-free guidance scales baked into the v3 vector estimator
    # (`v = cond_scale*v_cond - uncond_scale*v_uncond`).  Absent on v1/v2,
    # where the runtime defaults to cond_scale=1, uncond_scale=0 (no guidance).
    cfg_scales = extract_cfg_scales(args.onnx_dir / "vector_estimator.onnx")
    if cfg_scales is not None:
        cond_scale, uncond_scale = cfg_scales
        writer.add_float32("supertonic.cfg_cond_scale", float(cond_scale))
        writer.add_float32("supertonic.cfg_uncond_scale", float(uncond_scale))
        print(f"vector-estimator CFG scales: cond={cond_scale} uncond={uncond_scale}")
    wrap_mode = "none" if args.no_language_wrap else (args.language_wrap_mode or ("none" if args.arch == "supertonic" else "open_close"))
    default_steps = args.default_steps if args.default_steps is not None else 5

    writer.add_uint32("supertonic.default_steps", default_steps)
    writer.add_float32("supertonic.default_speed", args.default_speed)
    writer.add_uint32("supertonic.language_wrap", 0 if wrap_mode == "none" else 1)
    writer.add_string("supertonic.language_wrap_mode", wrap_mode)
    writer.add_array("supertonic.languages", ARCH_LANGUAGES[args.arch])
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

    # Stable cross-family aliases: `<stage>:<canonical>` -> `<stage>:<onnx-name>`.
    source_aliases: list[str] = []
    source_alias_targets: list[str] = []

    def emit_tensor(stage: str, source_name: str, arr: np.ndarray, count: int) -> None:
        nonlocal total_bytes
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

    for stage, filename in STAGES:
        model = onnx.load(str(args.onnx_dir / filename), load_external_data=True)
        init_names, consumers = build_graph_index(model)
        # canonical alias -> primary source key (for collision detection).
        stage_aliases: dict[str, str] = {}
        emitted_sources: set[str] = set()
        count = 0
        for source_name, arr in iter_onnx_tensors_from_model(model):
            emit_tensor(stage, source_name, arr, count)
            emitted_sources.add(source_name)
            canon = canonical_source_name(source_name, init_names, consumers)
            target_key = f"{stage}:{source_name}"

            def _record_alias(alias_name: str) -> None:
                if alias_name == source_name:
                    return
                alias_key = f"{stage}:{alias_name}"
                if alias_key in stage_aliases and stage_aliases[alias_key] != target_key:
                    print(f"  WARNING: alias collision {alias_key}: "
                          f"{stage_aliases[alias_key]} vs {target_key} (keeping first)")
                elif alias_key not in stage_aliases:
                    stage_aliases[alias_key] = target_key

            if canon is not None:
                _record_alias(canon)
                # Bridge alias: map the v2 logical id to this weight so the
                # runtime's hard-coded `onnx::MatMul_<v2id>` references resolve
                # against a renumbered (v3) export.
                logical = CANONICAL_TO_LOGICAL.get(stage, {}).get(canon)
                if logical is not None:
                    _record_alias(logical)
            count += 1

        # Materialise input-independent intermediates that v3 leaves live.
        for nm in MATERIALIZE_INTERMEDIATES.get(stage, []):
            if nm in emitted_sources:
                continue
            print(f"  materialising input-independent constant {stage}:{nm}")
            for mat_name, mat_arr in materialize_intermediates(args.onnx_dir / filename, [nm]).items():
                emit_tensor(stage, mat_name, mat_arr, count)
                emitted_sources.add(mat_name)
                count += 1

        # Module-prefixed intermediates: alias the canonical name the runtime
        # asks for to whichever candidate source we actually emitted.
        for canon_name, candidates in INTERMEDIATE_ALIASES.get(stage, []):
            if canon_name in emitted_sources:
                continue  # already present under the canonical name (v2)
            for cand in candidates:
                if cand in emitted_sources:
                    stage_aliases[f"{stage}:{canon_name}"] = f"{stage}:{cand}"
                    break

        # QVAC-19305 — convert-time guarantee that every vector-estimator
        # bridge the runtime depends on is resolvable.  The bridge maps each
        # v2 logical id (`onnx::MatMul_<v2id>`) to a weight via its stable
        # canonical (PyTorch) name, and that canonical name is recovered from
        # the MatMul's adjacent bias-`Add` (see `canonical_source_name`).  If a
        # v3 export drops the adjacent bias for a projection (e.g. a bias-free
        # `W_key`), the canonical lookup misses, no bridge alias is emitted,
        # and the runtime would later fail to bind `onnx::MatMul_<v2id>`.  A
        # logical id is resolvable when it is either emitted as a primary
        # source (the v2 case, where the id *is* the weight's name) or
        # recorded as a bridge alias (the v3 case).  Fail loudly here — naming
        # the offending ids — rather than shipping an unloadable GGUF.
        missing_bridges = sorted(
            logical
            for canon, logical in CANONICAL_TO_LOGICAL.get(stage, {}).items()
            if logical not in emitted_sources
            and f"{stage}:{logical}" not in stage_aliases
        )
        if missing_bridges:
            raise RuntimeError(
                f"{stage}: {len(missing_bridges)} vector-estimator bridge "
                f"alias(es) could not be derived: {missing_bridges}.  These "
                f"weights are bound by their v2 logical id at runtime; the "
                f"likely cause is a projection whose canonical name could not "
                f"be recovered (e.g. a MatMul with no adjacent bias-Add in "
                f"this export).  Extend canonical_source_name / "
                f"CANONICAL_TO_LOGICAL to cover it before converting."
            )

        for alias_key, target_key in stage_aliases.items():
            source_aliases.append(alias_key)
            source_alias_targets.append(target_key)

        per_stage_counts[stage] = count
        print(f"{stage:16s} {count:5d} tensors  ({len(stage_aliases)} stable aliases)")

    writer.add_array("supertonic.tensor_names", tensor_names)
    writer.add_array("supertonic.source_names", source_names)
    writer.add_array("supertonic.tensor_shapes", tensor_shapes)
    writer.add_array("supertonic.tensor_dtypes", tensor_dtypes)
    writer.add_array("supertonic.tensor_sha256", tensor_hashes)
    if source_aliases:
        writer.add_array("supertonic.source_aliases", source_aliases)
        writer.add_array("supertonic.source_alias_targets", source_alias_targets)
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
