#!/usr/bin/env python3
"""Convert the CosyVoice3 flow model (flow.pt) to GGUF.

Second converter of the CosyVoice3 bring-up (stage 4). The flow model is
`CausalMaskedDiffWithDiT`:

    speech tokens --(embed 6561x80)--> PreLookahead(2 convs + residual)
        --> repeat_interleave x2 --> CausalConditionalCFM (10-step Euler,
        cosine schedule, CFG rate 0.7) wrapping a 22-layer DiT estimator
        --> mel [80, T]

All weights are plain Linear/Conv1d/LayerNorm/Embedding (no weight_norm), so we
just write every tensor under `flow/<slash-separated key>` -- the same naming
convention the HiFT converter uses -- and record the architecture hparams as
metadata so the C++ graph cannot silently disagree with the weights.

    python3 convert-cosyvoice3-flow-to-gguf.py --flow flow.pt \\
        --config cosyvoice3.yaml --outfile cosyvoice3-flow-f32.gguf --dtype f32

Requires: pip install gguf numpy torch
"""
import argparse

import numpy as np


def load_state_dict(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict", "generator"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return {k: v for k, v in obj.items() if hasattr(v, "detach")}


def to_numpy(t, dtype):
    import torch
    a = t.detach().to(torch.float32).cpu().numpy()
    if dtype == "f16":
        a = a.astype(np.float16)
    else:
        a = a.astype(np.float32)
    return np.ascontiguousarray(a)


def infer_hparams(sd):
    """Derive architecture hparams from tensor shapes / key counts, so the
    metadata is a fact about the weights rather than a guessed config."""
    import re
    depth = 1 + max(
        int(m.group(1))
        for k in sd
        if (m := re.search(r"transformer_blocks\.(\d+)\.", k))
    )
    dim = sd["decoder.estimator.input_embed.proj.bias"].shape[0]          # 1024
    proj_in = sd["decoder.estimator.input_embed.proj.weight"].shape[1]    # 320 = 80*3 + spk80
    ff_inner = sd["decoder.estimator.transformer_blocks.0.ff.ff.0.0.bias"].shape[0]  # 2048
    inv_freq = sd["decoder.estimator.rotary_embed.inv_freq"].shape[0]     # 32 -> dim_head 64
    vocab, in_size = sd["input_embedding.weight"].shape                   # 6561, 80
    spk_in = sd["spk_embed_affine_layer.weight"].shape[1]                 # 192
    mel_dim = sd["decoder.estimator.proj_out.weight"].shape[0]            # 80
    return {
        "depth": depth,
        "dim": dim,
        "dim_head": inv_freq * 2,
        "heads": dim // (inv_freq * 2),
        "ff_inner": ff_inner,
        "proj_in": proj_in,
        "vocab_size": vocab,
        "in_size": in_size,
        "spk_embed_dim": spk_in,
        "mel_dim": mel_dim,
        "token_mel_ratio": 2,
        "pre_lookahead_len": 3,
        "n_timesteps": 10,
        "conv_pos_kernel": sd["decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight"].shape[2],
        "conv_pos_groups": 16,
    }


def main():
    import gguf
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True, help="path to flow.pt")
    ap.add_argument("--config", default=None, help="cosyvoice3.yaml (embedded as metadata)")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    args = ap.parse_args()

    sd = load_state_dict(args.flow)
    hp = infer_hparams(sd)
    print(f"hparams (inferred from weights): {hp}")

    w = gguf.GGUFWriter(args.outfile, "cosyvoice3-flow")
    w.add_float32("cosyvoice3.flow.inference_cfg_rate", 0.7)
    w.add_float32("cosyvoice3.flow.sigma_min", 1e-06)
    w.add_string("cosyvoice3.flow.t_scheduler", "cosine")
    w.add_string("cosyvoice3.flow.solver", "euler")
    for k, v in hp.items():
        w.add_uint32(f"cosyvoice3.flow.{k}", int(v))
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as fh:
                w.add_string("cosyvoice3.flow.config_yaml", fh.read())
        except OSError:
            pass

    # flow/<slash-separated key>, matching the HiFT converter (hift/<...>), but
    # abbreviated so names stay under ggml's GGML_MAX_NAME (64) limit -- the full
    # "decoder.estimator.transformer_blocks.21..." path overflows it and the C++
    # gguf reader then fails ("failed to read tensor info"). The C++ side
    # (cosyvoice_flow.cpp) uses the same short names.
    def gguf_name(k):
        k = k.replace("decoder.estimator.", "")     # DiT lives at flow/<...>
        k = k.replace("transformer_blocks.", "blk.")  # flow/blk/<i>/<...>
        return "flow/" + k.replace(".", "/")

    n = 0
    for name, tensor in sorted(sd.items()):
        w.add_tensor(gguf_name(name), to_numpy(tensor, args.dtype))
        n += 1

    # CausalConditionalCFM.__init__ does set_all_random_seed(0); rand_noise =
    # torch.randn([1,80,50*300]). This fixed noise is the Euler start point (not a
    # state_dict param), so bake it in for a self-contained C++ flow.
    import torch
    torch.manual_seed(0)
    rand_noise = torch.randn([1, 80, 50 * 300])[0]   # [80, 15000]
    w.add_tensor("flow/rand_noise", to_numpy(rand_noise, args.dtype))
    n += 1
    print(f"writing {n} tensors -> {args.outfile} ({args.dtype})")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("done.")


if __name__ == "__main__":
    main()
