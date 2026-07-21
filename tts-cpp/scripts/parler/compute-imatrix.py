#!/usr/bin/env python3
"""Compute an importance matrix (imatrix) for Parler-TTS quantization.

Runs the HF reference model (fp32, CPU, greedy) over a small calibration set
and accumulates, for every nn.Linear weight, the mean squared activation per
input column — the `quant_weights` vector ggml_quantize_chunk uses to weight
per-column quantization error. Output: an .npz keyed by HF state-dict weight
name (e.g. "decoder.model.decoder.layers.0.self_attn.q_proj.weight"),
consumed by convert-to-gguf.py --imatrix (which maps HF names to GGUF
tensor names itself).

The calibration texts deliberately EXCLUDE the dump-reference.py
fixture prompts/descriptions, so fixture-based argmax agreement stays a
held-out metric. Numbers are written as words, matching what the engine's
prompt normalizer feeds the model.

The transformers sharded-load weight-norm repair (see dump script) is NOT
needed here: it only affects DAC/quantizer conv weights, which are never
hooked (not nn.Linear) and never quantized, and DAC runs after the decoder
so it cannot influence any hooked activation.

Usage:
  python3 compute-imatrix.py --model-id parler-tts/parler-tts-mini-v1 \
      --out tts-cpp/models/parler-mini-v1-imatrix.npz
"""

import argparse

import numpy as np

# (prompt, description) pairs; varied speakers, pacing, rooms and content
CALIBRATION = [
    ("Hello there! It is such a pleasure to finally meet you in person.",
     "A female speaker with a calm, clear voice, close up, in a quiet room."),
    ("The weather forecast says light rain in the afternoon, so bring an umbrella.",
     "A male speaker with a deep voice speaks quickly with slight background noise."),
    ("Once upon a time, a small fox lived at the edge of a quiet forest.",
     "An expressive female voice with high pitch delivers an animated speech in a small room."),
    ("Could you please turn down the music? I am trying to focus on my homework.",
     "A male speaker with a low-pitched, monotone voice, very close up, studio quality."),
    ("Breaking news: the city council approved the new library budget this morning.",
     "A female speaker talks slowly and softly, slightly distant, with mild reverberation."),
    ("I counted twenty-three ships in the harbor before the fog rolled in.",
     "An energetic male voice with moderate pitch and very high recording quality."),
    ("Thank you for calling customer support; your call may be recorded for quality purposes.",
     "A female speaker with a calm, clear voice, close up, in a quiet room."),
    ("Wow, that was absolutely incredible! Do it again, do it again!",
     "An expressive female voice with high pitch delivers an animated speech in a small room."),
    ("Slowly and carefully, she lowered the fragile vase onto the marble table.",
     "A male speaker with a low-pitched, monotone voice, very close up, studio quality."),
    ("The meeting is scheduled for Tuesday at half past nine in the main conference room.",
     "A male speaker with a deep voice speaks quickly with slight background noise."),
    ("He whispered the secret so quietly that nobody else in the room could hear it.",
     "A female speaker talks slowly and softly, slightly distant, with mild reverberation."),
    ("Please remember to water the plants, feed the cat, and lock the front door.",
     "An energetic male voice with moderate pitch and very high recording quality."),
    ("Three hundred forty-two kilometers separate the two mountain villages.",
     "A male speaker with a deep voice speaks quickly with slight background noise."),
    ("Is this the right platform for the express train to the airport?",
     "A female speaker with a calm, clear voice, close up, in a quiet room."),
    ("Every single morning she jogs around the lake before the sun comes up.",
     "An expressive female voice with high pitch delivers an animated speech in a small room."),
    ("The recipe calls for two cups of flour, one egg, and a pinch of salt.",
     "A male speaker with a low-pitched, monotone voice, very close up, studio quality."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="parler-tts/parler-tts-mini-v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-frames", type=int, default=200,
                    help="cap greedy generation length (delayed steps) per calibration pair")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from parler_tts import ParlerTTSForConditionalGeneration

    print(f"loading {args.model_id} (fp32, cpu, eager)")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    gen_cfg = model.generation_config

    sums, counts, hooks = {}, {}, []

    def add_hook(name, mod):
        def hook(_mod, inputs, _out):
            x = inputs[0]
            if x is None or x.numel() == 0:
                return
            x = x.detach().reshape(-1, x.shape[-1]).to(torch.float64)
            if name not in sums:
                sums[name] = torch.zeros(x.shape[-1], dtype=torch.float64)
                counts[name] = 0
            sums[name] += x.square().sum(dim=0)
            counts[name] += x.shape[0]
        hooks.append(mod.register_forward_hook(hook))

    n_linear = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            add_hook(name, mod)
            n_linear += 1
    print(f"hooked {n_linear} nn.Linear modules")

    torch.manual_seed(args.seed)
    for idx, (prompt, description) in enumerate(CALIBRATION):
        desc_ids = tok(description, return_tensors="pt").input_ids
        prompt_ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            model.generate(
                input_ids=desc_ids,
                prompt_input_ids=prompt_ids,
                do_sample=False,
                num_beams=1,
                max_length=min(gen_cfg.max_length, args.max_new_frames),
            )
        print(f"  [{idx + 1}/{len(CALIBRATION)}] done: {prompt[:48]}...", flush=True)

    for h in hooks:
        h.remove()

    out, silent = {}, []
    for name in sorted(sums):
        if counts[name] == 0:
            silent.append(name)
            continue
        out[name + ".weight"] = (sums[name] / counts[name]).to(torch.float32).numpy()
    if silent:
        print(f"WARNING: {len(silent)} hooked Linear(s) never ran: {silent}")

    np.savez(args.out, **out)
    rows = {n: counts[n] for n in list(counts)[:1]}
    print(f"wrote {args.out}: {len(out)} importance vectors "
          f"(~{next(iter(rows.values())) if rows else 0} activation rows each)")


if __name__ == "__main__":
    main()
