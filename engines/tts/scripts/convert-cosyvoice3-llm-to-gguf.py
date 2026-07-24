#!/usr/bin/env python3
"""Convert the CosyVoice3 LLM (llm.pt) to GGUF -- stage 5 of the bring-up.

CosyVoice3LM = a Qwen2.5-0.5B causal LM (24 layers, hidden 896, 14 attn heads /
2 KV heads GQA, head_dim 64, SwiGLU intermediate 4864, RoPE theta 1e6, QKV bias,
RMSNorm eps 1e-6) driven by input embeddings, plus:
  - speech_embedding : Embedding(6761, 896)  (speech tokens + sos/eos/task/fill)
  - llm_decoder      : Linear(896, 6761, bias=False)  (speech-token logits head)

We emit every weight under a short `lm/<...>` name (< ggml's 64-char limit) so
the C++ graph can look them up. lm_head (tied text head) is skipped -- the speech
path uses llm_decoder. Weights are materialised to f32 for CPU parity.

    python3 convert-cosyvoice3-llm-to-gguf.py --llm llm.pt \\
        --outfile cosyvoice3-llm-f32.gguf --dtype f32
"""
import argparse
import re

import numpy as np


def load_state_dict(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return {k: v for k, v in obj.items() if hasattr(v, "detach")}


def to_f32(t):
    import torch
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())


# Quantize only the transformer-body Linear weights (q/k/v/o_proj, gate/up/down).
# The engine's ggml_mul_mat consumes quantized weights against f32 activations
# with no code change.  embed_tokens / speech_embedding are gathered via a CPU
# f32 pointer in the engine, so they (and all norms/biases + the llm_decoder
# head) stay f32 for sampling fidelity.  Block size 32 -> the in-dim (shape[1])
# must be a multiple of 32 (all Qwen2 body weights are).
def should_quant(gn, a):
    return (a.ndim == 2 and gn.startswith("lm/blk/") and gn.endswith("/weight")
            and a.shape[1] % 32 == 0)


# map long HF/CosyVoice keys -> short lm/<...> names
def gguf_name(k):
    k = k.replace("llm.model.model.", "")
    k = k.replace("layers.", "blk.")
    k = k.replace("self_attn.", "")
    k = k.replace("input_layernorm", "in_ln")
    k = k.replace("post_attention_layernorm", "post_ln")
    k = k.replace("mlp.gate_proj", "gate")
    k = k.replace("mlp.up_proj", "up")
    k = k.replace("mlp.down_proj", "down")
    return "lm/" + k.replace(".", "/")


def main():
    import gguf
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16", "q8_0", "q4_0"], default="f32")
    args = ap.parse_args()
    QMAP = {"q8_0": gguf.GGMLQuantizationType.Q8_0, "q4_0": gguf.GGMLQuantizationType.Q4_0}

    sd = load_state_dict(args.llm)
    depth = 1 + max((int(m.group(1)) for k in sd if (m := re.search(r"layers\.(\d+)\.", k))), default=-1)
    hidden = sd["llm.model.model.norm.weight"].shape[0]
    kv_dim = sd["llm.model.model.layers.0.self_attn.k_proj.bias"].shape[0]
    inter = sd["llm.model.model.layers.0.mlp.gate_proj.weight"].shape[0]
    n_head = 14
    head_dim = hidden // n_head
    n_kv = kv_dim // head_dim
    print(f"Qwen2: depth={depth} hidden={hidden} n_head={n_head} n_kv={n_kv} "
          f"head_dim={head_dim} inter={inter}")

    w = gguf.GGUFWriter(args.outfile, "cosyvoice3-llm")
    for k, v in dict(depth=depth, hidden=hidden, n_head=n_head, n_kv=n_kv,
                     head_dim=head_dim, inter=inter,
                     speech_token_size=6561, sos=6561, eos=6562, task_id=6563,
                     vocab=sd["llm.model.model.embed_tokens.weight"].shape[0],
                     endofprompt=151646).items():
        w.add_uint32(f"cosyvoice3.llm.{k}", int(v))
    w.add_float32("cosyvoice3.llm.rope_theta", 1000000.0)
    w.add_float32("cosyvoice3.llm.rms_eps", 1e-06)

    n = 0
    nq = 0
    for name, tensor in sorted(sd.items()):
        if name == "llm.model.lm_head.weight":
            continue  # tied text head, unused on the speech path
        gn = gguf_name(name)
        assert len(gn) < 64, f"name too long: {gn}"
        a = to_f32(tensor)
        if args.dtype in QMAP and should_quant(gn, a):
            qt = QMAP[args.dtype]
            w.add_tensor(gn, gguf.quants.quantize(a, qt), raw_dtype=qt)
            nq += 1
        else:
            # f16 only for the f16 build; quantized builds keep the rest f32.
            w.add_tensor(gn, a.astype(np.float16) if args.dtype == "f16" else a)
        n += 1
    print(f"writing {n} tensors ({nq} quantized to {args.dtype}) -> {args.outfile}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("done.")


if __name__ == "__main__":
    main()
