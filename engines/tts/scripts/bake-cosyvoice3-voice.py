#!/usr/bin/env python3
"""Bake a CosyVoice3 default voice into a small GGUF (voice.gguf).

CosyVoice3 zero-shot conditioning needs four prompt tensors derived from a
reference utterance (via the S3 tokenizer + CAM++ speaker encoder + prompt
mel).  Until those front-ends are ported to native C++, we precompute them once
from a reference clip and ship the result as a compact GGUF so the Engine can
synthesize speech with a fixed default voice and **no PyTorch dependency**.

Tensors written (names consumed by cosyvoice_engine.cpp):
    voice/prompt_stok   i32 [T_ptok]      LM prompt speech tokens (S3)
    voice/prompt_token  i32 [T_ptok]      flow prompt speech tokens (S3)
    voice/prompt_feat   f32 [mel_len1,80] prompt mel (mel-fastest per frame)
    voice/embedding     f32 [192]         CAM++ speaker embedding

    python3 bake-cosyvoice3-voice.py \\
        --prompt-stok  llm-ref/prompt_stok.npy \\
        --prompt-token e2e/prompt_token.npy \\
        --prompt-feat  e2e/prompt_feat.npy \\
        --embedding    e2e/embedding.npy \\
        --outfile      voice.gguf

Requires: pip install gguf numpy
"""
import argparse

import numpy as np
import gguf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-stok", required=True)
    ap.add_argument("--prompt-token", required=True)
    ap.add_argument("--prompt-feat", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--outfile", default="voice.gguf")
    ap.add_argument("--name", default="default", help="voice label (metadata only)")
    ap.add_argument("--prompt-text", default="",
                    help="transcript of the reference clip; baked in so the LM "
                         "prompt is self-contained (voice.prompt_text metadata)")
    ap.add_argument("--language", default="", help="voice language hint (metadata)")
    args = ap.parse_args()

    pstok = np.load(args.prompt_stok).astype(np.int32).reshape(-1)
    ptok = np.load(args.prompt_token).astype(np.int32).reshape(-1)
    pfeat = np.load(args.prompt_feat).astype(np.float32)
    emb = np.load(args.embedding).astype(np.float32).reshape(-1)

    assert pfeat.ndim == 2 and pfeat.shape[1] == 80, f"prompt_feat must be [mel_len,80], got {pfeat.shape}"
    assert emb.shape[0] == 192, f"embedding must be [192], got {emb.shape}"

    w = gguf.GGUFWriter(args.outfile, "cosyvoice3-voice")
    w.add_string("voice.name", args.name)
    w.add_string("voice.prompt_text", args.prompt_text)
    w.add_string("voice.language", args.language)
    w.add_uint32("voice.prompt_len", int(pstok.shape[0]))
    w.add_uint32("voice.mel_len", int(pfeat.shape[0]))
    w.add_tensor("voice/prompt_stok", pstok)
    w.add_tensor("voice/prompt_token", ptok)
    w.add_tensor("voice/prompt_feat", pfeat)
    w.add_tensor("voice/embedding", emb)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"wrote {args.outfile}: prompt_len={pstok.shape[0]} mel_len={pfeat.shape[0]} "
          f"emb={emb.shape[0]} (voice='{args.name}', lang='{args.language}', "
          f"prompt_text={args.prompt_text[:40]!r})")


if __name__ == "__main__":
    main()
