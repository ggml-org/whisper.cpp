#!/usr/bin/env python3
"""Dump multilingual T3 reference intermediates for C++ parity validation.

Mirrors scripts/reference-t3-turbo.py + dump-s3gen-reference.py style:
runs ChatterboxMultilingualTTS in PyTorch with a fixed seed, captures the
key tensors at each stage of the T3 forward path (cond_emb, text/speech
embeddings with learned pos, selected hidden states, logits, sampled
tokens) as .npy files under --out.

Used by src/t3_mtl.cpp parity stages M1..M7.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ref_src = os.environ.get(
    "CHATTERBOX_REF_SRC",
    str(Path(__file__).resolve().parent.parent.parent / "chatterbox-ref" / "src"),
)
sys.path.insert(0, ref_src)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="Hello there.")
    p.add_argument("--language", default="en")
    p.add_argument("--out", type=Path, default=Path("artifacts/t3-mtl-ref"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg-weight", type=float, default=0.5)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--min-p", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--repetition-penalty", type=float, default=2.0)
    p.add_argument("--n-predict", type=int, default=60,
                   help="Also run greedy decoding for this many tokens for bit-exact check.")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def save(path: Path, arr, **kv):
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch.is_tensor(arr):
        arr = arr.detach().to("cpu")
        if arr.dtype in (torch.bfloat16, torch.float16):
            arr = arr.float()
        arr = arr.numpy()
    arr = np.ascontiguousarray(arr)
    np.save(path, arr)
    shape_s = "x".join(str(d) for d in arr.shape)
    dtype_s = str(arr.dtype)
    print(f"  save {path.name:38s} shape={shape_s:18s} dtype={dtype_s}")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    print(f"Loading ChatterboxMultilingualTTS (device={args.device})")
    tts = ChatterboxMultilingualTTS.from_pretrained(args.device)

    hooks = []
    captures = {}

    def remember(name, multi_call=False):
        def hook(_mod, _in, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if multi_call:
                captures.setdefault(name, []).append(t)
            else:
                captures[name] = t
        return hook

    hooks.append(tts.t3.cond_enc.spkr_enc.register_forward_hook(remember("spkr_enc_out")))
    hooks.append(tts.t3.cond_enc.perceiver.register_forward_hook(remember("perceiver_out")))
    hooks.append(tts.t3.cond_enc.emotion_adv_fc.register_forward_hook(remember("emotion_adv_out")))
    hooks.append(tts.t3.cond_enc.register_forward_hook(remember("cond_emb")))
    hooks.append(tts.t3.text_emb.register_forward_hook(remember("text_emb_raw", multi_call=True)))
    hooks.append(tts.t3.text_pos_emb.register_forward_hook(remember("text_pos_emb_out")))
    hooks.append(tts.t3.speech_emb.register_forward_hook(remember("speech_emb_raw", multi_call=True)))
    hooks.append(tts.t3.speech_pos_emb.register_forward_hook(remember("speech_pos_emb_out", multi_call=True)))
    for idx in (0, 1, 2, 14, 29):
        hooks.append(tts.t3.tfmr.layers[idx].register_forward_hook(
            remember(f"layer{idx}_out", multi_call=True)))
    hooks.append(tts.t3.tfmr.norm.register_forward_hook(remember("final_norm_out", multi_call=True)))
    hooks.append(tts.t3.speech_head.register_forward_hook(remember("speech_logits", multi_call=True)))

    # Capture the full `inputs_embeds` tensor that goes into the first T3
    # forward call by wrapping patched_model.forward.
    orig_forward = None
    captured_inputs_embeds = []

    def capture_forward(*a, **kw):
        ie = kw.get("inputs_embeds")
        if ie is None and len(a) > 0:
            ie = a[0]
        if ie is not None:
            captured_inputs_embeds.append(ie.detach().cpu().clone())
        return orig_forward(*a, **kw)
    # Wrap after patched_model is created (on first call). Done in a monkey-patch
    # below by replacing the inference() attribute access via a hook on tfmr.
    orig_tfmr_forward = tts.t3.tfmr.forward

    def tfmr_forward_spy(*a, **kw):
        ie = kw.get("inputs_embeds")
        if ie is None and len(a) > 0:
            ie = a[0]
        if ie is not None:
            captured_inputs_embeds.append(ie.detach().cpu().clone())
        return orig_tfmr_forward(*a, **kw)
    tts.t3.tfmr.forward = tfmr_forward_spy

    text_tokens_saved = []
    orig_encode = tts.tokenizer.text_to_tokens

    def encode_spy(text, language_id=None, **kw):
        out = orig_encode(text, language_id=language_id, **kw)
        text_tokens_saved.append(out.clone())
        return out
    tts.tokenizer.text_to_tokens = encode_spy

    # text_emb forward hook captures the padded token tensor that actually
    # reaches the embedding layer (post [START]/[STOP] padding + CFG batch
    # duplication).  We keep both for completeness.
    text_tokens_padded_captured = []
    orig_text_emb = tts.t3.text_emb

    class TextEmbSpy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, tokens):
            text_tokens_padded_captured.append(tokens.detach().cpu().clone())
            return self.inner(tokens)
    tts.t3.text_emb = TextEmbSpy(orig_text_emb)

    print(f"Generating greedy (top_k=1, cfg_weight={args.cfg_weight}, lang={args.language})")
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        wav_greedy = tts.generate(
            text=args.text, language_id=args.language,
            cfg_weight=args.cfg_weight, temperature=1.0,
            repetition_penalty=1.0, min_p=0.0, top_p=1.0,
        )

    save(args.out / "text_tokens_raw.npy", text_tokens_saved[-1][0])
    if text_tokens_padded_captured:
        # First call is the padded (SOT + raw + EOT, batch-2 for CFG) one
        # that feeds text_emb in prepare_input_embeds. Save batch index 0.
        save(args.out / "text_tokens.npy", text_tokens_padded_captured[0][0].to(torch.int32))
    save(args.out / "spkr_enc_out.npy", captures["spkr_enc_out"])
    save(args.out / "perceiver_out.npy", captures["perceiver_out"])
    save(args.out / "emotion_adv_out.npy", captures["emotion_adv_out"])
    save(args.out / "cond_emb.npy", captures["cond_emb"])
    save(args.out / "text_pos_emb_out.npy", captures["text_pos_emb_out"])

    if "text_emb_raw" in captures and captures["text_emb_raw"]:
        save(args.out / "text_emb_raw.npy", captures["text_emb_raw"][0])
    if "speech_emb_raw" in captures and captures["speech_emb_raw"]:
        save(args.out / "speech_emb_raw_call0.npy", captures["speech_emb_raw"][0])
    if "speech_pos_emb_out" in captures and captures["speech_pos_emb_out"]:
        save(args.out / "speech_pos_emb_step0.npy", captures["speech_pos_emb_out"][0])
    if captured_inputs_embeds:
        save(args.out / "inputs_embeds_initial.npy", captured_inputs_embeds[0])

    for idx in (0, 1, 2, 14, 29):
        calls = captures.get(f"layer{idx}_out", [])
        if calls:
            save(args.out / f"layer{idx}_out_call0.npy", calls[0])
            if len(calls) > 1:
                save(args.out / f"layer{idx}_out_call1.npy", calls[1])
    for name in ("final_norm_out", "speech_logits"):
        calls = captures.get(name, [])
        if calls:
            save(args.out / f"{name}_call0.npy", calls[0])
            if len(calls) > 1:
                save(args.out / f"{name}_call1.npy", calls[1])

    save(args.out / "wav_greedy.npy", wav_greedy.squeeze(0).float().numpy())

    meta = {
        "text": args.text,
        "language": args.language,
        "seed": args.seed,
        "cfg_weight": args.cfg_weight,
        "device": args.device,
        "n_text_tokens": int(text_tokens_saved[-1].numel()),
        "captured_stages": list(captures.keys()),
        "repo": "ResembleAI/chatterbox",
    }
    (args.out / "summary.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out}/summary.json")

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
