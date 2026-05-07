#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import gguf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


REPO_ID = "ResembleAI/chatterbox-turbo"
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]

TEXT_VOCAB_SIZE = 50276
SPEECH_VOCAB_SIZE = 6563
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SPEAKER_EMBED_SIZE = 256
N_CTX = 8196
N_EMBD = 1024
N_HEAD = 16
N_LAYER = 24
LAYER_NORM_EPS = 1e-5

LAYER_RE = re.compile(r"^tfmr\.h\.(\d+)\.(.+)$")


QUANT_CHOICES = ["f16", "q8_0", "q5_0", "q4_0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Chatterbox Turbo T3 weights to GGUF.")
    parser.add_argument("--ckpt-dir", type=Path, help="Local checkpoint dir (downloads from HF if omitted).")
    parser.add_argument("--out", type=Path, default=Path("models/chatterbox-t3-turbo.gguf"), help="Output GGUF path.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--quant", choices=QUANT_CHOICES, default="f16",
                        help=("Weight dtype for attention + MLP + speech_head projections. "
                              "f16 (default, ~730 MB), q8_0 (~385 MB), q5_0 (~250 MB), "
                              "q4_0 (~205 MB). Biases, layer norms, embeddings and "
                              "positional embeddings always stay at their original dtype. "
                              "For K-quants (q4_k / q5_k / q6_k), run the resulting f16 "
                              "GGUF through llama.cpp's llama-quantize instead — the "
                              "Python gguf package doesn't implement them yet."))
    return parser.parse_args()


def as_numpy(tensor: torch.Tensor, *, dtype=None, transpose: bool = False) -> np.ndarray:
    if dtype is not None:
        tensor = tensor.to(dtype)
    array = tensor.detach().cpu().numpy()
    if transpose:
        array = array.T
    return np.ascontiguousarray(array)


# Which exported tensor names hold "big" 2-D projection weights that are
# worth quantizing. These are the ones ggml_mul_mat will consume; their
# inner (reduction) dimension is always a multiple of 256 for GPT-2 Medium
# (n_embd = 1024, inner_ffn = 4096), which is the block size requirement
# for Q4_K / Q5_K.
def _is_quantizable_weight(gguf_name: str) -> bool:
    if gguf_name == "chatterbox/speech_head":
        return True
    # Per-layer: model/h{i}/attn/c_attn/w, c_proj/w, mlp/c_fc/w, mlp/c_proj/w
    if gguf_name.startswith("model/h") and (
        gguf_name.endswith("/attn/c_attn/w") or
        gguf_name.endswith("/attn/c_proj/w") or
        gguf_name.endswith("/mlp/c_fc/w") or
        gguf_name.endswith("/mlp/c_proj/w")
    ):
        return True
    return False


# NOTE: the Python gguf 0.18 package only implements the "legacy" block
# types (Q4_0/1, Q5_0/1, Q8_0). The K-quants (Q4_K, Q5_K, Q6_K) are
# declared but NotImplementedError at runtime — use llama.cpp's
# llama-quantize tool on the F16 GGUF if you need those.
_QUANT_TYPE = {
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
}


def add_maybe_quantized(writer: "gguf.GGUFWriter", name: str, array: np.ndarray, quant: str):
    """Pass F32/F16 arrays straight through; quantize the "big" projection
    weights when --quant is not f16.
    """
    if quant == "f16" or not _is_quantizable_weight(name):
        writer.add_tensor(name, array)
        return str(array.dtype)

    qtype = _QUANT_TYPE[quant]
    # Block-quantized kernels consume F32 input.
    qdata = gguf.quants.quantize(array.astype(np.float32), qtype)
    # GGUF writer wants the BYTE shape as raw_shape (the qdata.shape);
    # it converts back to element shape using the quant type's block size.
    writer.add_tensor(name, qdata, raw_shape=qdata.shape, raw_dtype=qtype)
    return qtype.name


def load_tokenizer_assets(ckpt_dir: Path):
    """Read vocab.json + merges.txt + added_tokens.json and return arrays
    ready to embed as GGUF metadata.

    Returns (tokens, types, merges):
      tokens:  list[str], token text indexed by token id
      types:   list[int], gguf TokenType (1=NORMAL, 4=USER_DEFINED for added tokens)
      merges:  list[str], BPE merge rules in "left right" format (header skipped)
    """
    vocab_path  = ckpt_dir / "vocab.json"
    merges_path = ckpt_dir / "merges.txt"
    added_path  = ckpt_dir / "added_tokens.json"

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))  # {token: id}
    added = {}
    if added_path.exists():
        added = json.loads(added_path.read_text(encoding="utf-8"))

    id_to_tok = {int(idx): tok for tok, idx in vocab.items()}
    for tok, idx in added.items():
        id_to_tok[int(idx)] = tok

    max_id = max(id_to_tok) if id_to_tok else -1
    tokens = []
    types  = []
    for i in range(max_id + 1):
        tok = id_to_tok.get(i, "")
        tokens.append(tok)
        types.append(int(gguf.TokenType.USER_DEFINED) if tok in added else int(gguf.TokenType.NORMAL))

    merges = []
    for line in merges_path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        merges.append(line)

    return tokens, types, merges


def map_tensor_name(name: str):
    if name == "tfmr.wte.weight":
        return None
    if name == "tfmr.wpe.weight":
        return "model/wpe", torch.float32, False
    if name == "tfmr.ln_f.weight":
        return "model/ln_f/g", torch.float32, False
    if name == "tfmr.ln_f.bias":
        return "model/ln_f/b", torch.float32, False
    if name == "text_emb.weight":
        return "chatterbox/text_emb", torch.float16, False
    if name == "speech_emb.weight":
        return "chatterbox/speech_emb", torch.float16, False
    if name == "speech_head.weight":
        return "chatterbox/speech_head", torch.float16, False
    if name == "speech_head.bias":
        return "chatterbox/speech_head_bias", torch.float32, False
    if name == "cond_enc.spkr_enc.weight":
        return "chatterbox/cond_spkr/w", torch.float32, False
    if name == "cond_enc.spkr_enc.bias":
        return "chatterbox/cond_spkr/b", torch.float32, False

    match = LAYER_RE.match(name)
    if not match:
        return None

    layer_idx = int(match.group(1))
    suffix = match.group(2)

    # GPT-2 Conv1D weights need transposing; biases and LayerNorm do not
    table = {
        "ln_1.weight": ("model/h{}/ln_1/g", torch.float32, False),
        "ln_1.bias": ("model/h{}/ln_1/b", torch.float32, False),
        "ln_2.weight": ("model/h{}/ln_2/g", torch.float32, False),
        "ln_2.bias": ("model/h{}/ln_2/b", torch.float32, False),
        "attn.c_attn.weight": ("model/h{}/attn/c_attn/w", torch.float16, True),
        "attn.c_attn.bias": ("model/h{}/attn/c_attn/b", torch.float32, False),
        "attn.c_proj.weight": ("model/h{}/attn/c_proj/w", torch.float16, True),
        "attn.c_proj.bias": ("model/h{}/attn/c_proj/b", torch.float32, False),
        "mlp.c_fc.weight": ("model/h{}/mlp/c_fc/w", torch.float16, True),
        "mlp.c_fc.bias": ("model/h{}/mlp/c_fc/b", torch.float32, False),
        "mlp.c_proj.weight": ("model/h{}/mlp/c_proj/w", torch.float16, True),
        "mlp.c_proj.bias": ("model/h{}/mlp/c_proj/b", torch.float32, False),
    }
    if suffix not in table:
        return None
    fmt, dtype, transpose = table[suffix]
    return fmt.format(layer_idx), dtype, transpose


def main() -> None:
    args = parse_args()
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
    else:
        ckpt_dir = Path(snapshot_download(repo_id=REPO_ID, token=args.hf_token, allow_patterns=ALLOW_PATTERNS))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {ckpt_dir}")
    state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
    conds = torch.load(ckpt_dir / "conds.pt", map_location="cpu", weights_only=True)

    writer = gguf.GGUFWriter(str(args.out), "chatterbox")
    writer.add_name("Chatterbox Turbo T3")
    writer.add_description("Chatterbox Turbo text-to-speech token generator for ggml.")
    writer.add_context_length(N_CTX)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_head_count(N_HEAD)
    writer.add_vocab_size(TEXT_VOCAB_SIZE)
    writer.add_uint32("chatterbox.n_ctx", N_CTX)
    writer.add_uint32("chatterbox.n_embd", N_EMBD)
    writer.add_uint32("chatterbox.n_head", N_HEAD)
    writer.add_uint32("chatterbox.n_layer", N_LAYER)
    writer.add_uint32("chatterbox.text_vocab_size", TEXT_VOCAB_SIZE)
    writer.add_uint32("chatterbox.speech_vocab_size", SPEECH_VOCAB_SIZE)
    writer.add_uint32("chatterbox.start_speech_token", START_SPEECH_TOKEN)
    writer.add_uint32("chatterbox.stop_speech_token", STOP_SPEECH_TOKEN)
    writer.add_uint32("chatterbox.speaker_embed_size", SPEAKER_EMBED_SIZE)
    writer.add_float32("chatterbox.layer_norm_eps", LAYER_NORM_EPS)
    writer.add_string("chatterbox.variant", "t3_turbo")
    writer.add_string("chatterbox.reference_repo", REPO_ID)

    # Embed the GPT-2 BPE tokenizer so the C++ binary has no runtime dependency
    # on vocab.json / merges.txt / added_tokens.json on disk.
    tok_tokens, tok_types, tok_merges = load_tokenizer_assets(ckpt_dir)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tok_tokens)
    writer.add_token_types(tok_types)
    writer.add_token_merges(tok_merges)
    print(f"Embedded tokenizer: {len(tok_tokens)} tokens, "
          f"{sum(1 for t in tok_types if t == int(gguf.TokenType.USER_DEFINED))} added, "
          f"{len(tok_merges)} merges")

    writer.add_string("chatterbox.quantization", args.quant)

    exported = 0
    quantized = 0
    ignored = []
    for name, tensor in state.items():
        mapped = map_tensor_name(name)
        if mapped is None:
            ignored.append(name)
            continue
        gguf_name, dtype, transpose = mapped
        array = as_numpy(tensor, dtype=dtype, transpose=transpose)
        written_type = add_maybe_quantized(writer, gguf_name, array, args.quant)
        exported += 1
        if written_type not in ("float32", "float16"):
            quantized += 1
        print(f"{gguf_name:32s} {str(tuple(array.shape)):18s} {written_type}")

    builtin_speaker = conds["t3"]["speaker_emb"].reshape(1, SPEAKER_EMBED_SIZE)
    builtin_tokens = conds["t3"]["cond_prompt_speech_tokens"].reshape(-1).to(torch.int32)

    writer.add_uint32("chatterbox.cond_prompt_length", int(builtin_tokens.numel()))
    writer.add_tensor("chatterbox/builtin/speaker_emb", as_numpy(builtin_speaker, dtype=torch.float32))
    writer.add_tensor("chatterbox/builtin/cond_prompt_speech_tokens", as_numpy(builtin_tokens))

    # VoiceEncoder weights (3-layer unidirectional LSTM + Linear projection).
    # Used by main.cpp to compute speaker_emb natively when --reference-audio
    # is given, so no Python helper is needed at inference time.  LSTM layout
    # is PyTorch's default: each weight_i{h,h}_l* is (4*hidden, ...) with the
    # [i, f, g, o] gate rows stacked.
    ve_path = ckpt_dir / "ve.safetensors"
    if ve_path.exists():
        ve_state = load_file(ve_path)
        VE_HIDDEN = 256
        VE_INPUT  = 40
        writer.add_uint32("voice_encoder.n_mels",        VE_INPUT)
        writer.add_uint32("voice_encoder.hidden_size",   VE_HIDDEN)
        writer.add_uint32("voice_encoder.num_layers",    3)
        writer.add_uint32("voice_encoder.embedding_size", VE_HIDDEN)  # proj is (256, 256)
        writer.add_uint32("voice_encoder.partial_frames", 160)
        writer.add_uint32("voice_encoder.sample_rate",   16000)
        writer.add_uint32("voice_encoder.n_fft",         400)
        writer.add_uint32("voice_encoder.hop_size",      160)
        writer.add_uint32("voice_encoder.win_size",      400)
        writer.add_float32("voice_encoder.overlap",      0.5)
        writer.add_float32("voice_encoder.rate",         1.3)
        writer.add_float32("voice_encoder.min_coverage", 0.8)

        for k, t in ve_state.items():
            # Skip the cosine-similarity scaling parameters; they're only used
            # for training/CFG and don't affect embedding extraction.
            if k.startswith("similarity_"):
                continue
            writer.add_tensor(
                f"voice_encoder/{k.replace('.', '/')}",
                as_numpy(t, dtype=torch.float32),
            )

        # Precomputed mel filterbank for the VE mel (40 channels @ 16 kHz,
        # n_fft=400). Matches librosa.filters.mel with fmin=0, fmax=8000.
        import librosa
        import numpy as np
        ve_mel_fb = librosa.filters.mel(
            sr=16000, n_fft=400, n_mels=40, fmin=0, fmax=8000,
        ).astype(np.float32)  # (40, 201)
        writer.add_tensor("voice_encoder/mel_fb",
                          np.ascontiguousarray(ve_mel_fb))
        print(f"Embedded VoiceEncoder: 14 tensors, mel_fb {ve_mel_fb.shape}")
    else:
        print(f"warning: no ve.safetensors at {ve_path}, skipping VoiceEncoder weights")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nWrote {exported + 2} tensors to {args.out}")
    print(f"  --quant {args.quant}: {quantized}/{exported} weight tensors quantized "
          f"({'f16/f32' if args.quant == 'f16' else args.quant.upper()} for quantized; "
          f"embeddings + biases + layer-norms unchanged)")
    if ignored:
        print("\nIgnored tensors:")
        for n in ignored:
            print(f"  {n}")


if __name__ == "__main__":
    main()
