#!/usr/bin/env python3
"""Dump CosyVoice3 LLM (stage 5) references for the ggml port.

Runs the real zero-shot LLM path but forces GREEDY decoding (argmax) so the
token stream is deterministic and the C++ graph can be validated bit-for-bit
(the shipped RAS sampling is stochastic and can't be matched exactly). Captures:

  lm_input.npy      [L, 896]   the constructed input embeds fed to the Qwen2 LM
                               ([sos_emb, text_emb, task_id_emb, prompt_speech_emb])
  logits.npy        [S, 6761]  per-step llm_decoder log-softmax logits (greedy)
  gen_tokens.npy    [S] int32  the greedy speech tokens produced
  text_ids.npy      [T] int32  the Qwen2 BPE token ids (prompt_text + text)
  prompt_stok.npy   [P] int32  prompt speech tokens (may be empty)

Usage (venv active, from models_evaluation/CosyVoice):
  PYTHONPATH=CosyVoice:CosyVoice/third_party/Matcha-TTS \\
  python3 dump_llm_reference.py --model-dir models/Fun-CosyVoice3-0.5B \\
      --tts-text "..." --prompt-audio CosyVoice/asset/zero_shot_prompt.wav \\
      --prompt-text "..." --out-dir artifacts/llm-ref --max-steps 40
"""
import argparse
import functools
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tts-text", default="收到好友从远方寄来的生日礼物，那份意外的惊喜让我心中充满了甜蜜的快乐。")
    ap.add_argument("--prompt-audio", required=True)
    ap.add_argument("--prompt-text", default="希望你以后能够做的比我还好呦。")
    ap.add_argument("--out-dir", default="artifacts/llm-ref")
    ap.add_argument("--max-steps", type=int, default=40)
    args = ap.parse_args()

    import torch
    from cosyvoice.cli.cosyvoice import CosyVoice3
    os.makedirs(args.out_dir, exist_ok=True)
    cv = CosyVoice3(args.model_dir, load_trt=False, load_vllm=False, fp16=False)
    lm = cv.model.llm
    # The Qwen pretrain loads as bf16; force f32 so the reference matches an f32
    # ggml port (bf16 vs f32 diverges over 24 layers).
    lm.llm.model = lm.llm.model.float()
    lm.float()

    cap = {"logits": [], "tokens": []}

    # Force greedy + capture the log-softmax logits at each step.
    orig_sample = lm.sampling_ids
    def greedy(weighted_scores, decoded_tokens, sampling, ignore_eos=False):
        ws = weighted_scores.detach().cpu().numpy().astype(np.float32)  # [6761]
        cap["logits"].append(ws.copy())
        if ignore_eos:
            ws = ws.copy(); ws[lm.speech_token_size:] = -1e30
        tid = int(np.argmax(ws))
        cap["tokens"].append(tid)
        return tid
    lm.sampling_ids = greedy

    # Capture the first forward_one_step input (lm_input).
    orig_fos = lm.llm.forward_one_step
    @functools.wraps(orig_fos)
    def fos(xs, masks, cache=None):
        first = "lm_input" not in cap
        if first:
            cap["lm_input"] = xs.detach().cpu().clone().numpy()[0]  # [L, 896]
        out = orig_fos(xs, masks, cache=cache)
        if first:
            cap["hidden0"] = out[0].detach().cpu().clone().numpy()[0]  # Qwen hidden [L, 896]
        return out
    lm.llm.forward_one_step = fos

    # Also capture text_ids + prompt speech tokens from the frontend.
    from cosyvoice.utils.file_utils import load_wav
    prompt_text = "You are a helpful assistant.<|endofprompt|>" + args.prompt_text
    model_input = cv.frontend.frontend_zero_shot(args.tts_text, prompt_text,
                                                 args.prompt_audio, cv.sample_rate, "")
    # full text ids the LM embeds = concat(prompt_text, text) -- prompt_text holds <|endofprompt|>
    full_text = torch.concat([model_input["prompt_text"], model_input["text"]], dim=1)
    np.save(os.path.join(args.out_dir, "text_ids.npy"),
            np.ascontiguousarray(full_text[0].cpu().numpy().astype(np.int32)))
    pstok = model_input["llm_prompt_speech_token"][0].cpu().numpy().astype(np.int32)
    np.save(os.path.join(args.out_dir, "prompt_stok.npy"), np.ascontiguousarray(pstok))

    # Run the LLM inference (greedy) for up to max-steps. inference_wrapper is
    # already @torch.inference_mode(); don't nest another context here.
    gen = lm.inference(
        text=model_input["text"], text_len=model_input["text_len"],
        prompt_text=model_input["prompt_text"],
        prompt_text_len=model_input["prompt_text_len"],
        prompt_speech_token=model_input["llm_prompt_speech_token"],
        prompt_speech_token_len=model_input["llm_prompt_speech_token_len"],
        embedding=model_input["llm_embedding"], sampling=25)
    toks = []
    try:  # breaking a torch inference_mode generator can raise at teardown; captures already fired
        for t in gen:
            toks.append(t)
            if len(toks) >= args.max_steps:
                break
    except Exception as e:
        print(f"(generator stopped: {type(e).__name__})")

    np.save(os.path.join(args.out_dir, "hidden0.npy"), np.ascontiguousarray(cap["hidden0"].astype(np.float32)))
    np.save(os.path.join(args.out_dir, "lm_input.npy"),
            np.ascontiguousarray(cap["lm_input"].astype(np.float32)))
    np.save(os.path.join(args.out_dir, "logits.npy"),
            np.ascontiguousarray(np.stack(cap["logits"]).astype(np.float32)))
    np.save(os.path.join(args.out_dir, "gen_tokens.npy"),
            np.ascontiguousarray(np.array(cap["tokens"], dtype=np.int32)))
    print(f"lm_input {cap['lm_input'].shape}  logits {np.stack(cap['logits']).shape} "
          f"tokens {len(cap['tokens'])}  first10={cap['tokens'][:10]}")
    print(f"done. refs in {args.out_dir}")


if __name__ == "__main__":
    main()
