#!/usr/bin/env python3
"""Dump per-stage CosyVoice3 reference tensors for the tts-cpp bring-up.

Runs one synthesis on the upstream PyTorch model and captures the inputs/outputs
of the flow (DiT) and hift (vocoder) stages as C-contiguous float32/int32 .npy
files, plus the reference waveform. Those tensors are the numerical source of
truth the ggml graphs are validated against (see README.md).

We wrap `model.flow.inference` and `model.hift.inference` rather than using
forward hooks, because CosyVoice calls custom `.inference()` methods (not
`forward()`), so nn.Module hooks would not fire.

Two modes:
  * zero-shot  (default)      -> inference_zero_shot(tts_text, prompt_text, ...)
  * instruct / emotion        -> inference_instruct2(tts_text, instruct, ...)
    when --emotion or --instruct-text is given.

Emotion control note: the model is conditioned by the *exact* natural-language
instructions it was trained on (see EMOTION_PRESETS, copied verbatim from
CosyVoice/cosyvoice/utils/common.py). Ad-hoc English like "Speak happily" is
out-of-distribution and does not reliably steer prosody. Use --emotion to pick a
canonical preset. --seed pins all RNG so a sweep over emotions varies ONLY the
instruction (isolating its effect from LLM/CFM sampling noise).

Run where torch + the CosyVoice repo + weights are available:
    PYTHONPATH=CosyVoice:CosyVoice/third_party/Matcha-TTS \\
    python3 dump_cosyvoice3_reference.py --model-dir models/Fun-CosyVoice3-0.5B \\
        --prompt-audio CosyVoice/asset/zero_shot_prompt.wav \\
        --prompt-text "..." --tts-text "..." --out-dir artifacts/cv3-ref
"""
import argparse
import functools
import inspect
import os
import sys

import numpy as np

# Canonical style/emotion instructions, copied verbatim from
# CosyVoice/cosyvoice/utils/common.py `instruct_list` -- the strings the model
# was actually trained on. The "You are a helpful assistant. " wrapper and the
# Chinese phrasing are part of that distribution; English paraphrases are not.
EMOTION_PRESETS = {
    "happy":   "You are a helpful assistant. 请非常开心地说一句话。",
    "sad":     "You are a helpful assistant. 请非常伤心地说一句话。",
    "angry":   "You are a helpful assistant. 请非常生气地说一句话。",
    "loud":    "You are a helpful assistant. Please say a sentence as loudly as possible.",
    "soft":    "You are a helpful assistant. Please say a sentence in a very soft voice.",
    "slow":    "You are a helpful assistant. 请用尽可能慢地语速说一句话。",
    "fast":    "You are a helpful assistant. 请用尽可能快地语速说一句话。",
    "sichuan": "You are a helpful assistant. 请用四川话说这句话。",
    "cantonese": "You are a helpful assistant. 请用广东话表达。",
    # neutral: valid instruct-mode control with no emotive content (baseline).
    "neutral": "You are a helpful assistant. 请用正常平淡的语气说一句话。",
}
ENDOFPROMPT = "<|endofprompt|>"


def _import_cosyvoice(model_class):
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2  # noqa: F401
    classes = {"CosyVoice": CosyVoice, "CosyVoice2": CosyVoice2}
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3  # type: ignore
        classes["CosyVoice3"] = CosyVoice3
    except Exception:
        pass
    if model_class:
        return classes[model_class]
    # Fun-CosyVoice3 is the target here; prefer it, then fall back.
    for name in ("CosyVoice3", "CosyVoice2", "CosyVoice"):
        if name in classes:
            return classes[name]
    raise RuntimeError("no CosyVoice class found on PYTHONPATH")


def build_model(CV, model_dir):
    """Construct with only the kwargs this class actually accepts.

    CosyVoice3 dropped `load_jit` (its __init__ is
    (model_dir, load_trt, load_vllm, fp16, trt_concurrent)), so passing
    load_jit=False -- as CosyVoice/CosyVoice2 want -- raises TypeError. Filter by
    signature. All acceleration off: CPU, f32, deterministic.
    """
    wanted = dict(load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    accepted = inspect.signature(CV.__init__).parameters
    kwargs = {k: v for k, v in wanted.items() if k in accepted}
    print(f"  constructor kwargs: {kwargs}")
    return CV(model_dir, **kwargs)


def set_seed(seed):
    """Pin every RNG the LLM sampler and CFM noise draw from."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_np(x):
    """Detach a torch tensor to a C-contiguous numpy array (f32 / i32)."""
    import torch
    if isinstance(x, torch.Tensor):
        t = x.detach().to("cpu")
        if t.dtype in (torch.int32, torch.int64):
            return np.ascontiguousarray(t.to(torch.int32).numpy())
        return np.ascontiguousarray(t.to(torch.float32).numpy())
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def save(out_dir, name, arr):
    path = os.path.join(out_dir, name)
    np.save(path, arr)
    print(f"  wrote {name}  shape={tuple(arr.shape)} dtype={arr.dtype}")


def squeeze_batch(arr):
    return arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 else arr


def resolve_instruction(args):
    """Return the instruction string (with <|endofprompt|>) or '' for zero-shot."""
    if args.instruct_text:
        instruct = args.instruct_text
    elif args.emotion:
        # argparse `choices` already guarantees a valid preset here.
        instruct = EMOTION_PRESETS[args.emotion]
    else:
        return ""
    if ENDOFPROMPT not in instruct:
        instruct += ENDOFPROMPT  # CosyVoice3 LLM asserts token 151646 is present
    return instruct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tts-text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--prompt-audio", required=True,
                    help="reference wav PATH (CosyVoice3 frontend re-loads it via "
                         "load_wav, so a path -- not a preloaded tensor -- is required)")
    ap.add_argument("--prompt-text", default="", help="transcript of --prompt-audio (zero-shot mode)")
    ap.add_argument("--emotion", default=None, choices=sorted(EMOTION_PRESETS),
                    help="canonical style preset (validated against EMOTION_PRESETS). "
                         "Uses inference_instruct2. Prefer these over --instruct-text.")
    ap.add_argument("--instruct-text", default="",
                    help="raw instruction string (overrides --emotion). "
                         "<|endofprompt|> is appended automatically if absent.")
    ap.add_argument("--seed", type=int, default=1986,
                    help="pin all RNG so an emotion sweep varies only the instruction")
    ap.add_argument("--out-dir", default="artifacts/cv3-ref")
    ap.add_argument("--model-class", default=None,
                    help="CosyVoice / CosyVoice2 / CosyVoice3 (auto: prefers CosyVoice3)")
    args = ap.parse_args()

    import torch  # noqa: F401
    import torchaudio

    os.makedirs(args.out_dir, exist_ok=True)
    CV = _import_cosyvoice(args.model_class)
    print(f"loading {CV.__name__} from {args.model_dir}")
    cosyvoice = build_model(CV, args.model_dir)

    instruct = resolve_instruction(args)
    is_cv3 = CV.__name__ == "CosyVoice3"
    # CosyVoice3's frontend loads the prompt wav itself (load_wav), so pass a PATH.
    prompt_path = args.prompt_audio

    # CosyVoice3's LLM asserts <|endofprompt|> (token 151646) is present in the
    # concatenated (prompt_text + text). For zero-shot the marker lives in
    # prompt_text, prefixed by a system string -- see official example.py:76,
    #   prompt_text = "You are a helpful assistant.<|endofprompt|>" + transcript
    # Without it the LLM emits ZERO speech tokens -> empty mel -> HiFT crash.
    prompt_text = args.prompt_text
    if is_cv3 and not instruct and ENDOFPROMPT not in prompt_text:
        prompt_text = "You are a helpful assistant." + ENDOFPROMPT + prompt_text
        print(f"  zero-shot prompt_text -> {prompt_text!r}")

    captured = {}

    def wrap(module, method_name, tag):
        orig = getattr(module, method_name)

        @functools.wraps(orig)
        def wrapped(*a, **kw):
            captured.setdefault(tag, []).append(dict(kw))
            out = orig(*a, **kw)
            captured[tag][-1]["__out__"] = out
            return out
        setattr(module, method_name, wrapped)
        return orig

    model = cosyvoice.model
    wrap(model.flow, "inference", "flow")
    wrap(model.hift, "inference", "hift")

    set_seed(args.seed)  # pin RNG right before the (lazy) generator is consumed
    if instruct:
        print(f"instruct2 synthesis  seed={args.seed}  instruct={instruct!r}")
        gen = cosyvoice.inference_instruct2(args.tts_text, instruct, prompt_path, stream=False)
    else:
        print(f"zero-shot synthesis  seed={args.seed}")
        gen = cosyvoice.inference_zero_shot(args.tts_text, prompt_text, prompt_path, stream=False)
    speech_chunks = [out["tts_speech"] for out in gen]
    tts_speech = torch.concat(speech_chunks, dim=1)

    # ---- Flow stage (tokens -> mel) ----------------------------------------
    if "flow" in captured:
        fk = captured["flow"][0]
        for src, dst in (("token", "speech_tokens"), ("prompt_token", "prompt_token"),
                         ("embedding", "embedding"), ("prompt_feat", "prompt_feat")):
            if src in fk and fk[src] is not None:
                save(args.out_dir, f"{dst}.npy", squeeze_batch(to_np(fk[src])))
        out = fk.get("__out__")
        mel = out[0] if isinstance(out, (tuple, list)) else out
        save(args.out_dir, "flow_mel.npy", squeeze_batch(to_np(mel)))  # [80, T_mel]
    else:
        print("WARNING: flow.inference was not observed", file=sys.stderr)

    # ---- HiFT stage (mel -> wav) -------------------------------------------
    if "hift" in captured:
        hk = captured["hift"][-1]
        if "speech_feat" in hk and hk["speech_feat"] is not None:
            save(args.out_dir, "hift_mel_in.npy", squeeze_batch(to_np(hk["speech_feat"])))
        out = hk.get("__out__")
        wav = out[0] if isinstance(out, (tuple, list)) else out
        save(args.out_dir, "hift_wav.npy", squeeze_batch(to_np(wav)))
    else:
        print("WARNING: hift.inference was not observed", file=sys.stderr)

    save(args.out_dir, "tts_speech.npy", squeeze_batch(to_np(tts_speech)))
    torchaudio.save(os.path.join(args.out_dir, "reference.wav"),
                    tts_speech, cosyvoice.sample_rate)
    # Record exactly what produced this run, next to the audio.
    with open(os.path.join(args.out_dir, "run_meta.txt"), "w") as f:
        f.write(f"model_class={CV.__name__}\nseed={args.seed}\n"
                f"tts_text={args.tts_text!r}\ninstruct={instruct!r}\n"
                f"emotion={args.emotion!r}\nprompt_audio={args.prompt_audio!r}\n")
    print(f"done. references in {args.out_dir}")


if __name__ == "__main__":
    main()
