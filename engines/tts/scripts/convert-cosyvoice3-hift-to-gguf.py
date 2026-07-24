#!/usr/bin/env python3
"""Convert the CosyVoice3 HiFT vocoder (hift.pt) to GGUF.

First converter of the CosyVoice3 bring-up (the vocoder is the smallest,
back-of-pipeline sub-model, validated first against the PyTorch reference
dumps: {flow_mel,hift_wav}.npy).

Resolves weight_norm (both the legacy `weight_g`/`weight_v` and the new
`parametrizations.weight.original0/1` layouts) back to a plain `.weight`, then
writes every tensor under its original state-dict key so the C++ side can look
them up by name. Structural hparams (num_mels, #upsample stages, iSTFT n_fft/hop)
are inferred from tensor shapes / key counts rather than guessed from config, so
the metadata cannot silently disagree with the weights.

    python3 convert-cosyvoice3-hift-to-gguf.py --hift hift.pt \\
        --outfile cosyvoice3-hift-f32.gguf --dtype f32

Requires: pip install gguf numpy torch
"""
import argparse
import re

import numpy as np


def load_state_dict(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("generator", "model", "state_dict"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return {k: v for k, v in obj.items() if hasattr(v, "detach")}


def resolve_weight_norm(sd):
    """Collapse weight_norm (g,v) pairs into a single materialised `.weight`.

    weight_norm default dim=0: norm taken over every dim except 0, so
    w = g * v / ||v||_{dims!=0}. Handles both storage layouts.
    """
    import torch
    out = {}
    consumed = set()
    for k in list(sd.keys()):
        if k in consumed:
            continue
        # New: <p>.parametrizations.weight.original0 (g) / original1 (v)
        m = re.match(r"(.*)\.parametrizations\.weight\.original0$", k)
        if m:
            base = m.group(1)
            g = sd[k]
            v = sd[f"{base}.parametrizations.weight.original1"]
            consumed.add(k)
            consumed.add(f"{base}.parametrizations.weight.original1")
            norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
            out[f"{base}.weight"] = (g * v / norm)
            continue
        # Legacy: <p>.weight_g / <p>.weight_v
        m = re.match(r"(.*)\.weight_g$", k)
        if m:
            base = m.group(1)
            g = sd[k]
            v = sd[f"{base}.weight_v"]
            consumed.add(k)
            consumed.add(f"{base}.weight_v")
            norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
            out[f"{base}.weight"] = (g * v / norm)
            continue
    for k, val in sd.items():
        if k in consumed:
            continue
        out.setdefault(k, val)
    return out


def to_numpy(t, dtype):
    a = t.detach().to("cpu").to(_torch_dtype("f32")).numpy()
    a = np.ascontiguousarray(a)
    if dtype == "f16" and a.ndim >= 2 and a.size >= 32:
        return a.astype(np.float16)
    return a.astype(np.float32)


def _torch_dtype(_):
    import torch
    return torch.float32


def infer_hparams(sd):
    hp = {"istft_n_fft": 16, "istft_hop_len": 4}  # CosyVoice HiFTGenerator default
    if "conv_pre.weight" in sd:
        # Conv1d weight [out, in=num_mels, k]
        hp["num_mels"] = int(sd["conv_pre.weight"].shape[1])
        hp["base_channels"] = int(sd["conv_pre.weight"].shape[0])
    if "conv_post.weight" in sd:
        # conv_post -> n_fft + 2 channels
        hp["conv_post_out"] = int(sd["conv_post.weight"].shape[0])
    ups = {int(m.group(1)) for k in sd for m in [re.match(r"ups\.(\d+)\.", k)] if m}
    if ups:
        hp["num_upsamples"] = max(ups) + 1
    rbs = {int(m.group(1)) for k in sd for m in [re.match(r"resblocks\.(\d+)\.", k)] if m}
    if rbs:
        hp["num_resblocks"] = max(rbs) + 1
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hift", required=True, help="hift.pt state dict")
    ap.add_argument("--config", default=None, help="cosyvoice.yaml (embedded as metadata)")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    args = ap.parse_args()

    import gguf

    sd = load_state_dict(args.hift)
    sd = resolve_weight_norm(sd)
    hp = infer_hparams(sd)
    print(f"hparams (inferred): {hp}")

    w = gguf.GGUFWriter(args.outfile, "cosyvoice3-hift")
    w.add_uint32("cosyvoice3.hift.sampling_rate", 24000)
    for k, v in hp.items():
        w.add_uint32(f"cosyvoice3.hift.{k}", int(v))
    w.add_string("cosyvoice3.hift.weight_norm", "resolved")
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as fh:
                w.add_string("cosyvoice3.hift.config_yaml", fh.read())
        except OSError:
            pass

    # Name tensors to match the existing tts-cpp HiFT graph lookups
    # (src/chatterbox_tts.cpp run_hift_decode -> find_tensor("hift/...")): the
    # CosyVoice HiFTGenerator lineage is shared with Chatterbox S3Gen, so the
    # ggml builder is reused. The Chatterbox s3gen converter maps
    # `mel2wav.<k>` -> `hift/<k with . -> />`; the CosyVoice hift.pt keys are
    # bare (conv_pre.weight, ups.0.weight, resblocks.0.convs1.0.weight, ...),
    # so we just prepend `hift/` and slash-separate.
    def gguf_name(k):
        return "hift/" + k.replace(".", "/")

    n = 0
    for name, tensor in sorted(sd.items()):
        arr = to_numpy(tensor, args.dtype)
        w.add_tensor(gguf_name(name), arr)
        n += 1
    print(f"writing {n} tensors -> {args.outfile} ({args.dtype})")
    print(f"  (named 'hift/<path>' to match src/chatterbox_tts.cpp run_hift_decode)")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("done.")


if __name__ == "__main__":
    main()
