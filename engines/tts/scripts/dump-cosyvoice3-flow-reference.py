#!/usr/bin/env python3
"""Dump intermediate tensors of the CosyVoice3 flow for stage-4 ggml parity.

Calls `model.flow.inference` directly on the saved reference inputs (speech
tokens + prompt tensors from a prior dump_cosyvoice3_reference.py run), so no LLM
is needed and the run is fully reproducible. Captures, as C-contiguous .npy:

  front-end:
    spks.npy            [80]        after F.normalize + spk_embed_affine_layer
    mu.npy              [80, T]     encoder output h (decoder `mu` input)
    cond.npy            [80, T]     prompt-feat conditioning (zeros after mel_len1)
    z.npy               [80, T]     fixed seed-0 Euler noise (rand_noise[:, :, :T])
  DiT estimator, FIRST Euler step (the parity gate for the whole 22-layer DiT):
    dit_in_x.npy        [2, 80, T]
    dit_in_mu.npy       [2, 80, T]
    dit_in_cond.npy     [2, 80, T]
    dit_in_t.npy        [2]
    dit_in_spks.npy     [2, 80]
    dit_out.npy         [2, 80, T]  estimator output (dphi_dt, pre-CFG-combine)
  full output:
    flow_feat.npy       [80, T]     decoder output before prompt trim
    flow_mel_out.npy    [80, T-mel_len1]  final flow mel (matches flow_mel.npy)

Usage (from models_evaluation/CosyVoice, venv active):
  PYTHONPATH=CosyVoice:CosyVoice/third_party/Matcha-TTS \\
  python3 dump_flow_intermediates.py --model-dir models/Fun-CosyVoice3-0.5B \\
      --ref-dir artifacts/local-smoke --out-dir artifacts/flow-ref
"""
import argparse
import functools
import os

import numpy as np


def npy(out_dir, name, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    np.save(os.path.join(out_dir, name), arr)
    print(f"  wrote {name}  {tuple(arr.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ref-dir", required=True, help="dir with speech_tokens/prompt_token/embedding/prompt_feat .npy")
    ap.add_argument("--out-dir", default="artifacts/flow-ref")
    args = ap.parse_args()

    import torch
    from cosyvoice.cli.cosyvoice import CosyVoice3
    os.makedirs(args.out_dir, exist_ok=True)

    cv = CosyVoice3(args.model_dir, load_trt=False, load_vllm=False, fp16=False)
    flow = cv.model.flow
    dev = torch.device("cpu")

    def load(name):
        return np.load(os.path.join(args.ref_dir, name))

    token = torch.tensor(load("speech_tokens.npy"), dtype=torch.int32).unsqueeze(0)
    prompt_token = torch.tensor(load("prompt_token.npy"), dtype=torch.int32).unsqueeze(0)
    embedding = torch.tensor(load("embedding.npy"), dtype=torch.float32).unsqueeze(0)
    prompt_feat = torch.tensor(load("prompt_feat.npy"), dtype=torch.float32).unsqueeze(0)
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32)
    prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32)
    prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.int32)

    cap = {}

    # Capture the FIRST estimator call (x, mask, mu, t, spks, cond) -> out.
    est = flow.decoder.estimator
    orig_est = est.forward

    @functools.wraps(orig_est)
    def est_wrap(x, mask, mu, t, spks=None, cond=None, streaming=False):
        out = orig_est(x, mask, mu, t, spks=spks, cond=cond, streaming=streaming)
        if "dit_out" not in cap:
            # IMPORTANT: .clone() before numpy -- the Euler loop reuses the x_in /
            # mu_in / ... buffers in place across all 10 steps, so a bare
            # .cpu().numpy() shares memory and would be overwritten by later steps
            # (the captured "step 1" tensor would silently become step 10).
            cap["dit_in_x"] = x.detach().cpu().clone().numpy()
            cap["dit_in_mu"] = mu.detach().cpu().clone().numpy()
            cap["dit_in_cond"] = cond.detach().cpu().clone().numpy()
            cap["dit_in_t"] = t.detach().cpu().clone().numpy()
            cap["dit_in_spks"] = spks.detach().cpu().clone().numpy()
            cap["dit_out"] = out.detach().cpu().clone().numpy()
        return out
    est.forward = est_wrap

    # Capture solve_euler's z / mu / spks / cond by wrapping it.
    orig_solve = flow.decoder.solve_euler

    @functools.wraps(orig_solve)
    def solve_wrap(x, t_span, mu, mask, spks, cond, streaming=False):
        cap["z"] = x.detach().cpu().numpy()
        cap["mu"] = mu.detach().cpu().numpy()
        cap["spks"] = spks.detach().cpu().numpy()
        cap["cond"] = cond.detach().cpu().numpy()
        return orig_solve(x, t_span, mu, mask, spks, cond, streaming=streaming)
    flow.decoder.solve_euler = solve_wrap

    with torch.inference_mode():
        feat, _ = flow.inference(
            token=token, token_len=token_len,
            prompt_token=prompt_token, prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat, prompt_feat_len=prompt_feat_len,
            embedding=embedding, streaming=False, finalize=True,
        )

    mel_len1 = prompt_feat.shape[1]
    # front-end + noise
    npy(args.out_dir, "spks.npy", cap["spks"][0])          # [80]
    npy(args.out_dir, "mu.npy", cap["mu"][0])              # [80, T]
    npy(args.out_dir, "cond.npy", cap["cond"][0])          # [80, T]
    npy(args.out_dir, "z.npy", cap["z"][0])                # [80, T]
    # DiT estimator gate (first step)
    for k in ("dit_in_x", "dit_in_mu", "dit_in_cond", "dit_in_t", "dit_in_spks", "dit_out"):
        npy(args.out_dir, k + ".npy", cap[k])
    # full output
    npy(args.out_dir, "flow_feat.npy", feat[0].cpu().numpy())      # [80, T] (before... actually already trimmed)
    print(f"mel_len1(prompt)={mel_len1}  feat shape={tuple(feat.shape)}")
    print(f"done. intermediates in {args.out_dir}")


if __name__ == "__main__":
    main()
