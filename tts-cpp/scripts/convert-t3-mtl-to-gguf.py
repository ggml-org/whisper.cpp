#!/usr/bin/env python3
"""
Convert the multilingual Chatterbox T3 (t3_mtl23ls_v2) weights to GGUF.

Parallels scripts/convert-t3-turbo-to-gguf.py, adapted to:
 - Llama 520M backbone (30 layers, RoPE llama3 scaling) instead of GPT-2 medium
 - Tokenizer: grapheme_mtl_merged_expanded_v1 (embedded as raw JSON blob)
 - T3 cond enc with perceiver resampler + emotion_adv projection
 - VoiceEncoder weights from ve.pt (torch state_dict)
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def _load_requantize_policy():
    """Load should_quantize + _QUANT_TYPE from requantize-gguf.py (single
    source of truth shared with convert-s3gen-to-gguf.py and the offline
    requantize tool).  Keeps the deny-list in one place so adding a new
    tensor name to T3 doesn't accidentally leak into a quantised slot."""
    path = Path(__file__).resolve().parent / "requantize-gguf.py"
    spec = importlib.util.spec_from_file_location("_chatterbox_requantize_policy", path)
    if spec is None or spec.loader is None:
        print(f"error: could not load quant policy from {path}", file=sys.stderr)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.should_quantize, mod._QUANT_TYPE


_SHOULD_QUANTIZE, _RQ_QUANT_TYPE = _load_requantize_policy()


REPO_ID = "ResembleAI/chatterbox"
ALLOW_PATTERNS = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]

# All language codes the *Python reference tokenizer* accepts. The C++
# tokenizer in src/mtl_tokenizer.cpp only honors the tier-1 subset (18
# of these); the other 5 (ja, he, ru, zh, hi) need external preprocessing
# (pykakasi / dicta / russian_text_stresser / Cangjie) and hard-error at
# runtime. See mtl_tokenizer::supported_languages() for the runtime list.
ALL_KNOWN_LANGUAGES = [
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh",
]

N_EMBD = 1024
N_HEAD = 16
N_KV_HEAD = 16
HEAD_DIM = 64
N_LAYER = 30
INTERMEDIATE_SIZE = 4096
TEXT_VOCAB_SIZE = 2454
SPEECH_VOCAB_SIZE = 8194
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
START_TEXT_TOKEN = 255
STOP_TEXT_TOKEN = 0
MAX_TEXT_TOKENS = 2048
MAX_SPEECH_TOKENS = 4096
SPEECH_COND_PROMPT_LEN = 150
SPEAKER_EMBED_SIZE = 256
PERCEIVER_QUERY_TOKENS = 32
PERCEIVER_QUERY_SIZE = 1024
PERCEIVER_NUM_HEADS = 4
RMS_NORM_EPS = 1e-5
ROPE_THETA = 500000.0
ROPE_SCALING_FACTOR = 8.0
ROPE_LOW_FREQ_FACTOR = 1.0
ROPE_HIGH_FREQ_FACTOR = 4.0
ROPE_ORIGINAL_MAX_POS = 8192

N_CTX = MAX_TEXT_TOKENS + MAX_SPEECH_TOKENS + 4

LAYER_RE = re.compile(r"^tfmr\.layers\.(\d+)\.(.+)$")

QUANT_CHOICES = ["f16", "q8_0", "q5_0", "q4_0"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Chatterbox multilingual T3 weights to GGUF.")
    p.add_argument("--ckpt-dir", type=Path, help="Local checkpoint dir (downloads from HF if omitted).")
    p.add_argument("--out", type=Path, default=Path("models/chatterbox-t3-mtl.gguf"))
    p.add_argument("--hf-token", default=None)
    p.add_argument("--quant", choices=QUANT_CHOICES, default="f16",
                   help="Weight dtype for the big 2-D matmul weights.  f16 keeps "
                        "the GGUF byte-identical to the legacy default.  q8_0/q5_0/q4_0 "
                        "block-quantise eligible tensors per scripts/requantize-gguf.py "
                        "(same deny-list as the S3Gen converter and the offline "
                        "requantize tool: keeps embeddings, position embeddings, "
                        "norms/biases, voice-encoder weights, and built-in voice "
                        "conditioning at full precision).")
    return p.parse_args()


def as_numpy(tensor: torch.Tensor, *, dtype=None, transpose: bool = False) -> np.ndarray:
    if dtype is not None:
        tensor = tensor.to(dtype)
    arr = tensor.detach().cpu().numpy()
    if transpose:
        arr = arr.T
    return np.ascontiguousarray(arr)


def add_maybe_quantized(writer: "gguf.GGUFWriter", name: str, array: np.ndarray, quant: str) -> str:
    """Write a tensor; quantise eligible big 2-D float weights when
    quant != f16.  Eligibility is decided by `should_quantize()` in
    scripts/requantize-gguf.py — single source of truth shared with
    convert-s3gen-to-gguf.py and the offline requantize tool.

    Concretely, for the T3 MTL tensor set this means q8_0/q5_0/q4_0
    quantises:
      - model/h{i}/attn/{q,k,v,o}/w
      - model/h{i}/mlp/{gate,up,down}/w
      - chatterbox/{text,speech}_head
      - chatterbox/cond_spkr/w
      - chatterbox/perceiver/pre_attention_query
      - chatterbox/perceiver/attn/{to_q,to_k,to_v,proj_out}/w
    and keeps full precision on:
      - all norms / biases (matched by `/g`, `/b`, `/norm/`, `/ln_`)
      - text/speech token embedding tables (`text_emb`, `speech_emb`)
      - text/speech positional embedding tables (`pos_emb`)
      - voice_encoder/* and chatterbox/builtin/* (whole subtrees)
      - chatterbox/emotion_adv_fc/w (fails the rank/alignment gate; ne[0]=1)

    Returns the storage dtype as a short string for the BENCH log.
    """
    if quant == "f16":
        writer.add_tensor(name, array)
        return str(array.dtype)
    if array.dtype.kind in ("i", "u") or np.issubdtype(array.dtype, np.integer):
        writer.add_tensor(name, array)
        return str(array.dtype)
    qtype = _RQ_QUANT_TYPE[quant]
    if not _SHOULD_QUANTIZE(name, array.shape, qtype):
        writer.add_tensor(name, array)
        return str(array.dtype)
    qdata = gguf.quants.quantize(np.ascontiguousarray(array.astype(np.float32)), qtype)
    writer.add_tensor(name, qdata, raw_shape=qdata.shape, raw_dtype=qtype)
    return qtype.name


def map_llama_layer(name: str):
    """Return (gguf_name, dtype, transpose) for a Llama backbone tensor, or None."""
    m = LAYER_RE.match(name)
    if not m:
        return None
    idx = int(m.group(1))
    suffix = m.group(2)
    # Llama uses nn.Linear everywhere. PyTorch stores those as (out, in);
    # ggml's axis reversal (numpy.shape[0] <-> ne[1]) already gives us (in, out)
    # for free, so no explicit transpose is needed and the assertion
    # ggml_can_mul_mat(weight, x) (ne[0] must match) lines up correctly.
    table = {
        "input_layernorm.weight":          ("model/h{}/ln_attn/g", torch.float32, False),
        "post_attention_layernorm.weight": ("model/h{}/ln_mlp/g",  torch.float32, False),
        "self_attn.q_proj.weight":         ("model/h{}/attn/q/w",  torch.float16, False),
        "self_attn.k_proj.weight":         ("model/h{}/attn/k/w",  torch.float16, False),
        "self_attn.v_proj.weight":         ("model/h{}/attn/v/w",  torch.float16, False),
        "self_attn.o_proj.weight":         ("model/h{}/attn/o/w",  torch.float16, False),
        "mlp.gate_proj.weight":            ("model/h{}/mlp/gate/w", torch.float16, False),
        "mlp.up_proj.weight":              ("model/h{}/mlp/up/w",   torch.float16, False),
        "mlp.down_proj.weight":            ("model/h{}/mlp/down/w", torch.float16, False),
    }
    if suffix not in table:
        return None
    fmt, dtype, transpose = table[suffix]
    return fmt.format(idx), dtype, transpose


def map_tensor(name: str):
    """Map any T3 state dict key to (gguf_name, dtype, transpose) or None to skip."""
    mapped = map_llama_layer(name)
    if mapped is not None:
        return mapped

    if name == "tfmr.norm.weight":
        return "model/norm/g", torch.float32, False
    if name == "text_emb.weight":
        return "chatterbox/text_emb", torch.float16, False
    if name == "speech_emb.weight":
        return "chatterbox/speech_emb", torch.float16, False
    if name == "text_head.weight":
        return "chatterbox/text_head", torch.float16, False
    if name == "speech_head.weight":
        return "chatterbox/speech_head", torch.float16, False
    if name == "text_pos_emb.emb.weight":
        return "chatterbox/text_pos_emb", torch.float32, False
    if name == "speech_pos_emb.emb.weight":
        return "chatterbox/speech_pos_emb", torch.float32, False

    if name == "cond_enc.spkr_enc.weight":
        return "chatterbox/cond_spkr/w", torch.float32, False
    if name == "cond_enc.spkr_enc.bias":
        return "chatterbox/cond_spkr/b", torch.float32, False

    if name == "cond_enc.emotion_adv_fc.weight":
        return "chatterbox/emotion_adv_fc/w", torch.float32, False

    if name.startswith("cond_enc.perceiver."):
        rest = name[len("cond_enc.perceiver."):]
        if rest == "pre_attention_query":
            return "chatterbox/perceiver/pre_attention_query", torch.float32, False
        if rest == "attn.norm.weight":
            return "chatterbox/perceiver/attn/norm/g", torch.float32, False
        if rest == "attn.norm.bias":
            return "chatterbox/perceiver/attn/norm/b", torch.float32, False
        for proj in ("to_q", "to_k", "to_v", "proj_out"):
            if rest == f"attn.{proj}.weight":
                return f"chatterbox/perceiver/attn/{proj}/w", torch.float32, False
            if rest == f"attn.{proj}.bias":
                return f"chatterbox/perceiver/attn/{proj}/b", torch.float32, False

    return None


def write_metadata(writer: gguf.GGUFWriter, quant: str) -> None:
    writer.add_name("Chatterbox Multilingual T3")
    writer.add_description("Chatterbox multilingual text-to-speech token generator (23 languages) for ggml.")
    writer.add_context_length(N_CTX)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_head_count(N_HEAD)
    # Note: vocab size goes through `chatterbox.text_vocab_size` only (read
    # by the C++ loader as KEY_TEXT_VOCAB_SIZE).  Skipping the GGUF-standard
    # `general.vocab_size` keeps a single canonical source so a future
    # converter can't have the two metadata entries drift.

    writer.add_string("chatterbox.variant", "t3_mtl")
    writer.add_string("chatterbox.backbone", "llama_520m")
    writer.add_uint32("chatterbox.n_ctx", N_CTX)
    writer.add_uint32("chatterbox.n_embd", N_EMBD)
    writer.add_uint32("chatterbox.n_head", N_HEAD)
    writer.add_uint32("chatterbox.n_kv_head", N_KV_HEAD)
    writer.add_uint32("chatterbox.head_dim", HEAD_DIM)
    writer.add_uint32("chatterbox.n_layer", N_LAYER)
    writer.add_uint32("chatterbox.intermediate_size", INTERMEDIATE_SIZE)
    writer.add_uint32("chatterbox.text_vocab_size", TEXT_VOCAB_SIZE)
    writer.add_uint32("chatterbox.speech_vocab_size", SPEECH_VOCAB_SIZE)
    writer.add_uint32("chatterbox.start_speech_token", START_SPEECH_TOKEN)
    writer.add_uint32("chatterbox.stop_speech_token", STOP_SPEECH_TOKEN)
    writer.add_uint32("chatterbox.start_text_token", START_TEXT_TOKEN)
    writer.add_uint32("chatterbox.stop_text_token", STOP_TEXT_TOKEN)
    writer.add_uint32("chatterbox.max_text_tokens", MAX_TEXT_TOKENS)
    writer.add_uint32("chatterbox.max_speech_tokens", MAX_SPEECH_TOKENS)
    writer.add_uint32("chatterbox.speech_cond_prompt_len", SPEECH_COND_PROMPT_LEN)
    writer.add_uint32("chatterbox.speaker_embed_size", SPEAKER_EMBED_SIZE)
    writer.add_uint32("chatterbox.perceiver_query_tokens", PERCEIVER_QUERY_TOKENS)
    writer.add_uint32("chatterbox.perceiver_query_size", PERCEIVER_QUERY_SIZE)
    writer.add_uint32("chatterbox.perceiver_num_heads", PERCEIVER_NUM_HEADS)
    writer.add_bool("chatterbox.emotion_adv", True)
    writer.add_float32("chatterbox.rms_norm_eps", RMS_NORM_EPS)
    writer.add_float32("chatterbox.rope_theta", ROPE_THETA)
    writer.add_string("chatterbox.rope.scaling_type", "llama3")
    writer.add_float32("chatterbox.rope.scaling_factor", ROPE_SCALING_FACTOR)
    writer.add_float32("chatterbox.rope.low_freq_factor", ROPE_LOW_FREQ_FACTOR)
    writer.add_float32("chatterbox.rope.high_freq_factor", ROPE_HIGH_FREQ_FACTOR)
    writer.add_uint32("chatterbox.rope.original_max_position", ROPE_ORIGINAL_MAX_POS)
    writer.add_string("chatterbox.reference_repo", REPO_ID)
    writer.add_string("chatterbox.quantization", quant)


def write_tokenizer(writer: gguf.GGUFWriter, ckpt_dir: Path) -> None:
    tok_path = ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"
    text = tok_path.read_text(encoding="utf-8")
    writer.add_string("tokenizer.ggml.model", "mtl_grapheme")
    writer.add_string("tokenizer.ggml.mtl_json", text)
    writer.add_array("tokenizer.ggml.mtl_languages", ALL_KNOWN_LANGUAGES)
    print(f"Embedded tokenizer JSON ({len(text)} bytes), {len(ALL_KNOWN_LANGUAGES)} languages")


def write_voice_encoder(writer: gguf.GGUFWriter, ckpt_dir: Path) -> None:
    ve_path = ckpt_dir / "ve.pt"
    if not ve_path.exists():
        print(f"warning: no ve.pt at {ve_path}, skipping VoiceEncoder weights")
        return

    ve_state = torch.load(ve_path, map_location="cpu", weights_only=True)
    VE_HIDDEN = 256
    VE_INPUT = 40
    writer.add_uint32("voice_encoder.n_mels",        VE_INPUT)
    writer.add_uint32("voice_encoder.hidden_size",   VE_HIDDEN)
    writer.add_uint32("voice_encoder.num_layers",    3)
    writer.add_uint32("voice_encoder.embedding_size", VE_HIDDEN)
    writer.add_uint32("voice_encoder.partial_frames", 160)
    writer.add_uint32("voice_encoder.sample_rate",   16000)
    writer.add_uint32("voice_encoder.n_fft",         400)
    writer.add_uint32("voice_encoder.hop_size",      160)
    writer.add_uint32("voice_encoder.win_size",      400)
    writer.add_float32("voice_encoder.overlap",      0.5)
    writer.add_float32("voice_encoder.rate",         1.3)
    writer.add_float32("voice_encoder.min_coverage", 0.8)

    n = 0
    for k, t in ve_state.items():
        if k.startswith("similarity_"):
            continue
        writer.add_tensor(f"voice_encoder/{k.replace('.', '/')}",
                          as_numpy(t, dtype=torch.float32))
        n += 1

    import librosa
    ve_mel_fb = librosa.filters.mel(
        sr=16000, n_fft=400, n_mels=40, fmin=0, fmax=8000,
    ).astype(np.float32)
    writer.add_tensor("voice_encoder/mel_fb", np.ascontiguousarray(ve_mel_fb))
    print(f"Embedded VoiceEncoder: {n} tensors + mel_fb {ve_mel_fb.shape}")


def main() -> None:
    args = parse_args()
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
    else:
        ckpt_dir = Path(snapshot_download(
            repo_id=REPO_ID,
            token=args.hf_token or os.getenv("HF_TOKEN"),
            allow_patterns=ALLOW_PATTERNS,
        ))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {ckpt_dir}")
    state = load_file(ckpt_dir / "t3_mtl23ls_v2.safetensors")
    if "model" in state and not torch.is_tensor(state["model"]):
        state = state["model"][0]
    conds = torch.load(ckpt_dir / "conds.pt", map_location="cpu", weights_only=True)

    writer = gguf.GGUFWriter(str(args.out), "chatterbox")
    write_metadata(writer, args.quant)
    write_tokenizer(writer, ckpt_dir)

    exported = 0
    quantized = 0
    ignored = []
    for name, tensor in state.items():
        mapped = map_tensor(name)
        if mapped is None:
            ignored.append(name)
            continue
        gguf_name, dtype, transpose = mapped
        arr = as_numpy(tensor, dtype=dtype, transpose=transpose)
        written = add_maybe_quantized(writer, gguf_name, arr, args.quant)
        exported += 1
        if written not in ("float32", "float16"):
            quantized += 1
        print(f"{gguf_name:46s} {str(tuple(arr.shape)):22s} {written}")

    builtin_speaker = conds["t3"]["speaker_emb"].reshape(1, SPEAKER_EMBED_SIZE)
    builtin_tokens = conds["t3"]["cond_prompt_speech_tokens"].reshape(-1).to(torch.int32)
    writer.add_uint32("chatterbox.cond_prompt_length", int(builtin_tokens.numel()))
    writer.add_tensor("chatterbox/builtin/speaker_emb", as_numpy(builtin_speaker, dtype=torch.float32))
    writer.add_tensor("chatterbox/builtin/cond_prompt_speech_tokens", as_numpy(builtin_tokens))

    write_voice_encoder(writer, ckpt_dir)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = args.out.stat().st_size
    print(f"\nWrote {exported + 2} mapped tensors to {args.out} ({out_size / 1e6:.1f} MB)")
    print(f"  --quant {args.quant}: {quantized}/{exported} weight tensors quantized")
    if ignored:
        print("\nIgnored tensors (first 20):")
        for n in ignored[:20]:
            print(f"  {n}")
        if len(ignored) > 20:
            print(f"  ... and {len(ignored) - 20} more")


if __name__ == "__main__":
    main()
