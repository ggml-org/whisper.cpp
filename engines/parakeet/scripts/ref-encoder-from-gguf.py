#!/usr/bin/env python3
"""Pure-PyTorch FastConformer encoder with weights loaded from a parakeet GGUF.

Compare tensor outputs to ``dump-ctc-reference.py`` artifacts to validate GGUF layout
and the forward pass against NeMo references.

Usage:

  python scripts/dump-ctc-reference.py \
      --wav test/samples/jfk.wav \
      --model models/parakeet-ctc-0.6b.nemo \
      --out artifacts/ctc-ref

  python scripts/ref-encoder-from-gguf.py \
      --gguf models/parakeet-ctc-0.6b.gguf \
      --ref  artifacts/ctc-ref
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import gguf


def ggml_to_torch(t):
    raw = np.asarray(t.data)
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        arr = raw.view(np.float32).reshape(tuple(reversed(t.shape)))
    elif t.tensor_type == gguf.GGMLQuantizationType.F16:
        arr = raw.view(np.float16).reshape(tuple(reversed(t.shape))).astype(np.float32)
    else:
        raise RuntimeError(f"unexpected tensor type {t.tensor_type}")
    return torch.from_numpy(np.ascontiguousarray(arr))


def load_gguf(gguf_path):
    r = gguf.GGUFReader(str(gguf_path))
    tensors = {t.name: ggml_to_torch(t) for t in r.tensors}
    meta = {}
    for field in r.fields.values():
        name = field.name
        if not field.types:
            continue
        tp = field.types[0]
        if tp in (gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32,
                  gguf.GGUFValueType.UINT64, gguf.GGUFValueType.INT64,
                  gguf.GGUFValueType.UINT16, gguf.GGUFValueType.INT16,
                  gguf.GGUFValueType.UINT8,  gguf.GGUFValueType.INT8):
            meta[name] = int(field.parts[field.data[0]][0])
        elif tp == gguf.GGUFValueType.BOOL:
            meta[name] = bool(field.parts[field.data[0]][0])
        elif tp == gguf.GGUFValueType.FLOAT32:
            meta[name] = float(field.parts[field.data[0]][0])
        elif tp == gguf.GGUFValueType.STRING:
            meta[name] = bytes(field.parts[field.data[0]]).decode()
        elif tp == gguf.GGUFValueType.ARRAY:
            elt = field.types[1] if len(field.types) > 1 else None
            if elt == gguf.GGUFValueType.UINT8:
                meta[name] = bytes(
                    int(field.parts[di][0]) for di in field.data
                )
            else:
                pass
    return tensors, meta


def sinusoidal_rel_pe(length: int, d_model: int, dtype=torch.float32):
    positions = torch.arange(length - 1, -length, -1, dtype=torch.float32).unsqueeze(1)
    pe = torch.zeros(2 * length - 1, d_model)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe.unsqueeze(0).to(dtype)


def _mask_time(x, valid_len):
    B, C, T, Fq = x.shape
    idx = torch.arange(T, device=x.device)
    m = (idx < valid_len).view(1, 1, T, 1).to(x.dtype)
    return x * m


def _conv_out_len(L, k, s, p):
    return (L + 2 * p - k) // s + 1


def subsampling(mel, W, valid_len=None):
    x = mel.unsqueeze(0).transpose(1, 2).unsqueeze(1)
    if valid_len is None:
        valid_len = x.size(2)

    L = valid_len

    x = _mask_time(x, L)
    x = F.conv2d(x, W["encoder.subsampling.conv0.weight"],
                    bias=W["encoder.subsampling.conv0.bias"],
                    stride=2, padding=1)
    L = _conv_out_len(L, 3, 2, 1)
    x = _mask_time(x, L)
    x = F.relu(x)

    x = _mask_time(x, L)
    x = F.conv2d(x, W["encoder.subsampling.conv1_dw.weight"],
                    bias=W["encoder.subsampling.conv1_dw.bias"],
                    stride=2, padding=1, groups=x.size(1))
    L = _conv_out_len(L, 3, 2, 1)
    x = _mask_time(x, L)
    x = F.conv2d(x, W["encoder.subsampling.conv1_pw.weight"],
                    bias=W["encoder.subsampling.conv1_pw.bias"],
                    stride=1, padding=0)
    x = _mask_time(x, L)
    x = F.relu(x)

    x = _mask_time(x, L)
    x = F.conv2d(x, W["encoder.subsampling.conv2_dw.weight"],
                    bias=W["encoder.subsampling.conv2_dw.bias"],
                    stride=2, padding=1, groups=x.size(1))
    L = _conv_out_len(L, 3, 2, 1)
    x = _mask_time(x, L)
    x = F.conv2d(x, W["encoder.subsampling.conv2_pw.weight"],
                    bias=W["encoder.subsampling.conv2_pw.bias"],
                    stride=1, padding=0)
    x = _mask_time(x, L)
    x = F.relu(x)

    B, C, T, Fq = x.shape
    x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * Fq)
    x = F.linear(x,
                 W["encoder.subsampling.out.weight"],
                 W["encoder.subsampling.out.bias"])
    return x, L


def layer_norm(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, (x.size(-1),), weight=weight, bias=bias, eps=eps)


def rel_shift(x):
    b, h, qlen, pos_len = x.size()
    x = F.pad(x, (1, 0))
    x = x.view(b, h, -1, qlen)
    x = x[:, :, 1:].reshape(b, h, qlen, pos_len)
    return x


def rel_pos_mha(x, pos_emb, W, prefix, n_heads):
    d_model = x.size(-1)
    head_dim = d_model // n_heads
    s_d_k = math.sqrt(head_dim)
    B, T, _ = x.shape

    q = F.linear(x, W[f"{prefix}.q.weight"], W[f"{prefix}.q.bias"]).view(B, T, n_heads, head_dim)
    k = F.linear(x, W[f"{prefix}.k.weight"], W[f"{prefix}.k.bias"]).view(B, T, n_heads, head_dim)
    v = F.linear(x, W[f"{prefix}.v.weight"], W[f"{prefix}.v.bias"]).view(B, T, n_heads, head_dim)

    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    p = F.linear(pos_emb, W[f"{prefix}.pos.weight"]).view(1, -1, n_heads, head_dim).transpose(1, 2)

    pos_bias_u = W[f"{prefix}.pos_bias_u"]
    pos_bias_v = W[f"{prefix}.pos_bias_v"]

    q_u = (q + pos_bias_u).transpose(1, 2)
    q_v = (q + pos_bias_v).transpose(1, 2)

    matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
    matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))
    matrix_bd = rel_shift(matrix_bd)
    matrix_bd = matrix_bd[:, :, :, :matrix_ac.size(-1)]

    scores = (matrix_ac + matrix_bd) / s_d_k
    attn = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(attn, v)
    ctx = ctx.transpose(1, 2).reshape(B, T, d_model)

    return F.linear(ctx, W[f"{prefix}.out.weight"], W[f"{prefix}.out.bias"])


def conformer_ff(x, W, prefix):
    x = F.linear(x, W[f"{prefix}.linear1.weight"], W[f"{prefix}.linear1.bias"])
    x = F.silu(x)
    x = F.linear(x, W[f"{prefix}.linear2.weight"], W[f"{prefix}.linear2.bias"])
    return x


def conformer_conv(x, W, prefix):
    x_t = x.transpose(1, 2)

    x_t = F.conv1d(x_t,
                   W[f"{prefix}.pw1.weight"].squeeze(-1).unsqueeze(-1),
                   W[f"{prefix}.pw1.bias"])
    x_t = F.glu(x_t, dim=1)

    C = x_t.size(1)
    x_t = F.conv1d(x_t,
                   W[f"{prefix}.dw.weight"],
                   W[f"{prefix}.dw.bias"],
                   padding=(W[f"{prefix}.dw.weight"].size(-1) - 1) // 2,
                   groups=C)

    scale = W[f"{prefix}.bn.scale"].view(1, -1, 1)
    shift = W[f"{prefix}.bn.shift"].view(1, -1, 1)
    x_t = x_t * scale + shift

    x_t = F.silu(x_t)
    x_t = F.conv1d(x_t,
                   W[f"{prefix}.pw2.weight"].squeeze(-1).unsqueeze(-1),
                   W[f"{prefix}.pw2.bias"])
    return x_t.transpose(1, 2)


def conformer_block(x, pos_emb, W, i, n_heads):
    p = f"encoder.blk.{i}"

    residual = x
    y = layer_norm(x, W[f"{p}.norm_ff1.weight"], W[f"{p}.norm_ff1.bias"])
    y = conformer_ff(y, W, f"{p}.ff1")
    x = residual + 0.5 * y

    residual = x
    y = layer_norm(x, W[f"{p}.norm_attn.weight"], W[f"{p}.norm_attn.bias"])
    y = rel_pos_mha(y, pos_emb, W, f"{p}.attn", n_heads)
    x = residual + y

    residual = x
    y = layer_norm(x, W[f"{p}.norm_conv.weight"], W[f"{p}.norm_conv.bias"])
    y = conformer_conv(y, W, f"{p}.conv")
    x = residual + y

    residual = x
    y = layer_norm(x, W[f"{p}.norm_ff2.weight"], W[f"{p}.norm_ff2.bias"])
    y = conformer_ff(y, W, f"{p}.ff2")
    x = residual + 0.5 * y

    x = layer_norm(x, W[f"{p}.norm_out.weight"], W[f"{p}.norm_out.bias"])
    return x


def encoder_forward(mel, W, meta, mel_valid_len=None, captures=None):
    d_model  = meta["parakeet.encoder.d_model"]
    n_layers = meta["parakeet.encoder.n_layers"]
    n_heads  = meta["parakeet.encoder.n_heads"]
    xscaling = meta.get("parakeet.encoder.xscaling", True)

    if mel_valid_len is None:
        mel_valid_len = mel.size(-1)

    x, enc_valid_len = subsampling(mel, W, valid_len=mel_valid_len)
    if captures is not None:
        captures["subsampling_out"] = x[0].clone()
        captures["enc_valid_len"] = int(enc_valid_len)

    if xscaling:
        x = x * math.sqrt(d_model)

    T = x.size(1)
    pe_full = sinusoidal_rel_pe(max(T, meta.get("parakeet.encoder.pos_emb_max_len", 5000)),
                                d_model, dtype=x.dtype)
    center = pe_full.size(1) // 2 + 1
    pos_emb = pe_full[:, center - T : center + T - 1]

    for i in range(n_layers):
        x = conformer_block(x, pos_emb, W, i, n_heads)
        if captures is not None and i == 0:
            captures["block_0_out"] = x[0].clone()
        if captures is not None and i == n_layers - 1:
            captures["block_last_out"] = x[0].clone()

    if captures is not None:
        captures["encoder_out"] = x[0].clone()
    return x


def ctc_head(x, W):
    logits = F.linear(x, W["ctc.decoder.weight"], W["ctc.decoder.bias"])
    return F.log_softmax(logits, dim=-1)


def ctc_greedy_decode(logits, blank_id):
    ids = logits.argmax(dim=-1).cpu().numpy()
    out = []
    prev = -1
    for tok in ids[0]:
        if tok != blank_id and tok != prev:
            out.append(int(tok))
        prev = int(tok)
    return out


def detokenize_with_sentencepiece(ids, spm_bytes):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load_from_serialized_proto(spm_bytes)
    return sp.decode(ids)


def parity(a, b, label):
    a = a.detach().cpu().float().numpy() if torch.is_tensor(a) else a
    b = b if isinstance(b, np.ndarray) else np.asarray(b)
    if a.shape != b.shape:
        print(f"[parity] {label}: shape mismatch a={a.shape} b={b.shape}")
        return False
    diff = (a - b).astype(np.float64)
    max_abs = float(np.abs(diff).max())
    rel = float(np.sqrt((diff * diff).sum() / max((b * b).sum(), 1e-30)))
    print(f"[parity] {label:<18}  rel = {rel:.3e}   max_abs = {max_abs:.3e}   shape={a.shape}")
    return rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, default=Path("models/parakeet-ctc-0.6b.gguf"))
    ap.add_argument("--ref",  type=Path, default=Path("artifacts/ctc-ref"))
    args = ap.parse_args()

    print(f"[shadow] loading {args.gguf}", file=sys.stderr)
    W, meta = load_gguf(args.gguf)
    print(f"[shadow] tensors={len(W)}  layers={meta['parakeet.encoder.n_layers']}  "
          f"d_model={meta['parakeet.encoder.d_model']}  heads={meta['parakeet.encoder.n_heads']}",
          file=sys.stderr)

    mel_np = np.load(args.ref / "mel.npy")
    mel = torch.from_numpy(mel_np)
    print(f"[shadow] mel: {tuple(mel.shape)}", file=sys.stderr)

    mel_valid_len = int((mel != 0).any(dim=0).sum().item())
    print(f"[shadow] mel valid frames (non-zero): {mel_valid_len}", file=sys.stderr)

    captures = {}
    with torch.inference_mode():
        enc = encoder_forward(mel, W, meta, mel_valid_len=mel_valid_len, captures=captures)
        logits = ctc_head(enc, W)
    print(f"[shadow] encoder valid frames: {captures['enc_valid_len']}", file=sys.stderr)

    # parity vs NeMo
    ok = True
    for name in ("subsampling_out", "block_0_out", "block_last_out", "encoder_out"):
        ref_path = args.ref / f"{name}.npy"
        if not ref_path.exists():
            print(f"[shadow] {ref_path} missing, skipping {name}")
            continue
        rel = parity(captures[name], np.load(ref_path), name)
        ok = ok and (rel is not False and rel < 1e-3)

    if (args.ref / "logits.npy").exists():
        rel = parity(logits[0], np.load(args.ref / "logits.npy"), "logits")
        ok = ok and (rel is not False and rel < 2e-3)

    ids = ctc_greedy_decode(logits, meta["parakeet.ctc.blank_id"])
    print(f"[shadow] greedy ids  (len={len(ids)}): {ids}")

    spm_bytes = meta.get("tokenizer.ggml.sentencepiece_model")
    if spm_bytes:
        txt = detokenize_with_sentencepiece(ids, spm_bytes)
        print(f"[shadow] transcript: {txt}")
        ref_txt_path = args.ref / "decoded.txt"
        if ref_txt_path.exists():
            ref_txt = ref_txt_path.read_text().strip()
            print(f"[shadow] reference : {ref_txt}")
            print(f"[shadow] match     : {txt.strip() == ref_txt}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
