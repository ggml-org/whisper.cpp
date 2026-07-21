#!/usr/bin/env python3
"""Dump Parler-TTS reference fixtures for tts-cpp parity tests.

Runs the HF parler-tts pipeline deterministically (CPU, fp32, eager attention,
greedy decoding) and dumps per-stage .npy fixtures consumed by the
test_parler_* C++ tests, plus reference wavs for by-ear verification.

Usage:
  python3 dump-reference.py --model-id parler-tts/parler-tts-mini-v1 \
      --out tts-cpp/artifacts/parler-ref
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch


CASES = [
    {
        "name": "case0",
        "description": "A female speaker delivers a slightly expressive and animated speech "
                       "with a moderate speed and pitch. The recording is of very high quality, "
                       "with the speaker's voice sounding clear and very close up.",
        "prompt": "Hey, how are you doing today?",
    },
    {
        "name": "case1",
        "description": "A male speaker with a low-pitched voice speaks slowly in a large room "
                       "with a lot of reverberation.",
        "prompt": "The quick brown fox jumps over 12 lazy dogs, doesn't it?",
    },
]

# corpus for tokenizer parity (encoded standalone, ids dumped)
TOKENIZER_CORPUS = [
    "Hey, how are you doing today?",
    "The quick brown fox jumps over 12 lazy dogs, doesn't it?",
    "  multiple   spaces\tand\nnewlines  ",
    "Numbers: 3.14159, 42 and 1,000,000!",
    "Quotes “like this” and 'this' — dashes…",
    "Café naïve résumé über Straße",
    "emoji \U0001f600 and CJK 你好世界 mixed",
    "A",
    "",
    "hello",
    "HELLO WORLD",
    "don't can't won't it's",
]

# corpus for the separate BPE PROMPT tokenizer (indic-class models); exercises
# Metaspace prepend, byte fallback, danda/ZWJ clusters, digits, mixed scripts.
# Must not contain literal <s>/</s>/<unk> (special-token matching is out of
# scope in the C++ encoder).
PROMPT_TOKENIZER_CORPUS = [
    "मेरा नाम प्रतीक है। नमस्ते।",
    "મારું નામ પ્રતિક છે. કેમ છો?",
    "Hey, how are you doing today?",
    "आज १२ तारीख़ है और 12 बज रहे हैं।",
    "કુલ ૨૫ રૂપિયા થયા, બરાબર 25.",
    "हिन्दी में क्षत्रिय और ज्ञान जैसे संयुक्ताक्षर।",
    "  multiple   spaces\tand\nnewlines  ",
    " leading space",
    "▁starts with the metaspace marker",
    "emoji \U0001f600 and CJK 你好世界 mixed",
    "English वाक्य के बीच में Hindi mixed sentence.",
    "A",
    "",
    "\t",
    "3.14159, 42 and 1,000,000!",
    "தமிழ் வணக்கம், తెలుగు నమస్కారం, ಕನ್ನಡ ನಮಸ್ಕಾರ",
    "বাংলা বাক্য এবং ওড়িয়া ମିଶ୍ରଣ",
    "پاکستان اور اردو زبان کا جملہ",
    "don't can't won't it's",
    "श्री२। ॐ नमः शिवाय॥",
]


def npy_save(out_dir, name, arr):
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    path = os.path.join(out_dir, name + ".npy")
    np.save(path, np.ascontiguousarray(arr))
    print(f"  wrote {name}.npy shape={tuple(np.asarray(arr).shape)} dtype={np.asarray(arr).dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="parler-tts/parler-tts-mini-v1",
                    help="HF repo id or a local snapshot directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-step-logits", type=int, default=20)
    ap.add_argument("--max-new-frames", type=int, default=430,
                    help="cap greedy generation length (delayed steps) to keep fixtures small; "
                         "430 steps ~= 5s audio")
    ap.add_argument("--cases-json", default=None,
                    help="JSON list of {name, description, prompt} overriding the built-in cases")
    args = ap.parse_args()

    global CASES
    if args.cases_json:
        with open(args.cases_json) as f:
            CASES = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    from transformers import AutoTokenizer
    import transformers
    import parler_tts
    from parler_tts import ParlerTTSForConditionalGeneration
    from parler_tts.logits_processors import ParlerTTSLogitsProcessor
    from parler_tts.modeling_parler_tts import build_delay_pattern_mask

    print(f"loading {args.model_id} (fp32, cpu, eager)")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    # repo tokenizer serves the PROMPT; T5-family repo tokenizers (mini/large)
    # also serve the description, otherwise (indic-class BPE) the description
    # tokenizer comes from the text encoder's own repo — mirrors the model card.
    tok = AutoTokenizer.from_pretrained(args.model_id)
    shared_tok = "T5" in type(tok).__name__
    desc_tok = tok if shared_tok else AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path)
    if not shared_tok:
        print(f"separate prompt tokenizer: {type(tok).__name__} "
              f"(descriptions via {model.config.text_encoder._name_or_path})")

    # Sharded checkpoints (large-v1) hit a transformers low-mem loading bug:
    # weight-norm parametrizations (parametrizations.weight.original0/1) are
    # left at their init values instead of the checkpoint's weight_g/weight_v.
    # Detect by comparing one probe tensor and repair ALL of them from the
    # safetensors ground truth so the fixtures reflect the real codec.
    def _repair_weight_norm(model, model_id):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        snap = model_id if os.path.isdir(model_id) else snapshot_download(model_id)
        handles = {}
        if os.path.exists(os.path.join(snap, "model.safetensors.index.json")):
            wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
            def get(name):
                f = handles.setdefault(wmap[name], safe_open(os.path.join(snap, wmap[name]), framework="pt"))
                return f.get_tensor(name)
            names = set(wmap)
        else:
            f = safe_open(os.path.join(snap, "model.safetensors"), framework="pt")
            def get(name):
                return f.get_tensor(name)
            names = set(f.keys())
        sd = dict(model.named_parameters())
        fixed = 0
        with torch.no_grad():
            for pname, param in sd.items():
                if pname.endswith("parametrizations.weight.original0"):
                    base = pname[: -len("parametrizations.weight.original0")]
                    g_name = base + "weight_g"
                    v_name = base + "weight_v"
                    if g_name in names and v_name in names:
                        g_ref = get(g_name).to(param.dtype)
                        if not torch.equal(param.cpu(), g_ref):
                            param.copy_(g_ref)
                            sd[base + "parametrizations.weight.original1"].copy_(
                                get(v_name).to(param.dtype))
                            fixed += 1
        if fixed:
            print(f"REPAIRED {fixed} weight-norm parametrizations from safetensors "
                  f"(transformers sharded-load bug)")
        return fixed

    _repair_weight_norm(model, args.model_id)

    cfg = model.config
    dec_cfg = cfg.decoder
    t5_cfg = model.text_encoder.config
    dac = model.audio_encoder
    gen_cfg = model.generation_config

    num_codebooks = model.decoder.num_codebooks
    bos_id = gen_cfg.bos_token_id
    eos_id = gen_cfg.eos_token_id
    pad_id = gen_cfg.pad_token_id

    # ---------------- risk-register answers ----------------
    sd_keys = list(model.state_dict().keys())
    risk = {}
    risk["R1_embed_positions_in_state_dict"] = any("embed_positions" in k for k in sd_keys)
    ep = model.decoder.model.decoder.embed_positions.weights
    risk["R1_embed_positions_live_buffer_shape"] = list(ep.shape)
    ln = model.decoder.model.decoder.layers[0].self_attn_layer_norm
    risk["R3_decoder_ln_eps"] = ln.eps
    risk["R3_decoder_attn_bias_keys"] = [k for k in sd_keys if ".self_attn.q_proj.bias" in k][:2]
    risk["R3_t5_rms_eps"] = t5_cfg.layer_norm_epsilon
    desc_ids_specials = desc_tok(CASES[0]["description"]).input_ids
    desc_ids_plain = desc_tok(CASES[0]["description"], add_special_tokens=False).input_ids
    prompt_ids_specials = tok(CASES[0]["prompt"]).input_ids
    prompt_ids_plain = tok(CASES[0]["prompt"], add_special_tokens=False).input_ids
    risk["R8_description_appends_eos"] = desc_ids_specials[-1] == t5_cfg.eos_token_id and \
        len(desc_ids_specials) == len(desc_ids_plain) + 1
    risk["R8_prompt_appends_eos"] = prompt_ids_specials[-1] == t5_cfg.eos_token_id and \
        len(prompt_ids_specials) == len(prompt_ids_plain) + 1
    risk["R8_prompt_prepends_bos"] = len(prompt_ids_specials) == len(prompt_ids_plain) + 1 and \
        prompt_ids_specials[0] != prompt_ids_plain[0]
    dac_core = dac.model if hasattr(dac, "model") else dac  # descript wrapper vs DacModel
    ru_conv1 = (dac_core.decoder.model[1].block[2].block[1] if hasattr(dac_core.decoder, "model")
                else dac_core.decoder.block[0].res_unit1.conv1)
    risk["R11_dac_residual_conv1_weight_shape"] = list(ru_conv1.weight.shape)
    risk["enc_to_dec_proj_applied"] = (
        t5_cfg.hidden_size != dec_cfg.hidden_size and dec_cfg.cross_attention_hidden_size is None
    )
    risk["prompt_cross_attention"] = model.prompt_cross_attention
    risk["dec_num_layers"] = dec_cfg.num_hidden_layers
    print("RISK ANSWERS:", json.dumps(risk, indent=2, default=str))

    # ---------------- tokenizer corpora ----------------
    # tokenizer_corpus.json always reflects the DESCRIPTION (T5) tokenizer —
    # that is what tokenizer.ggml.* in the GGUF encodes.
    tok_ids = [desc_tok(t).input_ids for t in TOKENIZER_CORPUS]
    with open(os.path.join(args.out, "tokenizer_corpus.json"), "w") as f:
        json.dump({"texts": TOKENIZER_CORPUS, "ids": tok_ids}, f, indent=1)
    print(f"  wrote tokenizer_corpus.json ({len(TOKENIZER_CORPUS)} cases)")
    if not shared_tok:
        p_ids = [tok(t).input_ids for t in PROMPT_TOKENIZER_CORPUS]
        with open(os.path.join(args.out, "prompt_tokenizer_corpus.json"), "w") as f:
            json.dump({"texts": PROMPT_TOKENIZER_CORPUS, "ids": p_ids}, f, indent=1)
        print(f"  wrote prompt_tokenizer_corpus.json ({len(PROMPT_TOKENIZER_CORPUS)} cases)")

    # ---------------- delay-pattern fixture (compact, model-free) ----------------
    start = torch.full((num_codebooks, 1), bos_id, dtype=torch.long)
    ids0, mask = build_delay_pattern_mask(start, bos_id, pad_id, 24, num_codebooks)
    npy_save(args.out, "delay_mask_len24", mask.numpy().astype(np.int64))
    npy_save(args.out, "delay_start_ids", ids0.numpy().astype(np.int64))

    # ---------------- logits-processor trace (synthetic, real HF class) ----------------
    # scenario: codebook rows emit EOS staggered; record masked rows per step.
    proc = ParlerTTSLogitsProcessor(eos_id, num_codebooks, batch_size=1, device="cpu")
    trace = []
    hist = torch.full((num_codebooks, 1), bos_id, dtype=torch.long)
    for step in range(1, 16):
        scores = torch.zeros((num_codebooks, dec_cfg.vocab_size))
        out = proc(hist, scores.clone())
        masked = (out[:, eos_id] == -math.inf).tolist()
        trace.append({
            "step": step,
            "history": hist.tolist(),
            "eos_masked_rows": masked,
            "pointer": int(proc.first_codebooks_unfinished[0]),
        })
        # codebook k samples EOS at step 4+k (staggered finish), else token 100+step
        new = []
        for k in range(num_codebooks):
            new.append(eos_id if step >= 4 + k else 100 + step)
        hist = torch.cat([hist, torch.tensor(new, dtype=torch.long).unsqueeze(1)], dim=1)
    with open(os.path.join(args.out, "logits_proc_trace.json"), "w") as f:
        json.dump({"eos_id": eos_id, "num_codebooks": num_codebooks, "trace": trace}, f, indent=1)
    print("  wrote logits_proc_trace.json")

    # ---------------- conv-transpose micro fixture ----------------
    g = torch.Generator().manual_seed(1234)
    convt = torch.nn.ConvTranspose1d(4, 3, kernel_size=16, stride=8, padding=4, bias=True)
    with torch.no_grad():
        convt.weight.copy_(torch.randn(convt.weight.shape, generator=g) * 0.2)
        convt.bias.copy_(torch.randn(convt.bias.shape, generator=g) * 0.1)
    xin = torch.randn(1, 4, 11, generator=g)
    with torch.no_grad():
        yout = convt(xin)
    npy_save(args.out, "convt_unit_w", convt.weight)      # [in_ch, out_ch, k]
    npy_save(args.out, "convt_unit_b", convt.bias)
    npy_save(args.out, "convt_unit_in", xin)
    npy_save(args.out, "convt_unit_out", yout)

    # ---------------- per-case pipeline fixtures ----------------
    meta = {
        "model_id": args.model_id,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "parler_tts": getattr(parler_tts, "__version__", "unknown"),
        "seed": args.seed,
        "attn_implementation": "eager",
        "risk": risk,
        "config": {
            "t5": {"d_model": t5_cfg.d_model, "d_ff": t5_cfg.d_ff, "num_layers": t5_cfg.num_layers,
                   "num_heads": t5_cfg.num_heads, "d_kv": t5_cfg.d_kv,
                   "rel_buckets": t5_cfg.relative_attention_num_buckets,
                   "rel_max_dist": t5_cfg.relative_attention_max_distance,
                   "rms_eps": t5_cfg.layer_norm_epsilon, "vocab": t5_cfg.vocab_size},
            "dec": {"hidden": dec_cfg.hidden_size, "layers": dec_cfg.num_hidden_layers,
                    "heads": dec_cfg.num_attention_heads, "ffn": dec_cfg.ffn_dim,
                    "codebooks": num_codebooks, "vocab": dec_cfg.vocab_size,
                    "ln_eps": ln.eps, "bos": bos_id, "eos": eos_id, "pad": pad_id,
                    "max_position": dec_cfg.max_position_embeddings},
            "gen": {"max_length": gen_cfg.max_length,
                    "min_new_tokens": getattr(gen_cfg, "min_new_tokens", None),
                    "do_sample": gen_cfg.do_sample},
            "dac": {"sr": dac.config.sampling_rate, "hop": int(np.prod([8, 8, 4, 2])),
                    "n_q": getattr(dac.config, "num_codebooks", None) or dac.config.n_codebooks,
                    "codebook_size": dac.config.codebook_size,
                    "latent": getattr(dac.config, "latent_dim", None) or dac.config.hidden_size},
        },
        "cases": [],
    }

    def dac_decode(z_q):
        out = dac_core.decode(z_q)
        return out if torch.is_tensor(out) else out.audio_values

    for case in CASES:
        name = case["name"]
        print(f"case {name}: '{case['prompt'][:40]}...'")
        desc_ids = desc_tok(case["description"], return_tensors="pt").input_ids
        prompt_ids = tok(case["prompt"], return_tensors="pt").input_ids
        npy_save(args.out, f"{name}_desc_ids", desc_ids.numpy().astype(np.int64))
        npy_save(args.out, f"{name}_prompt_ids", prompt_ids.numpy().astype(np.int64))

        with torch.no_grad():
            t5_out = model.text_encoder(input_ids=desc_ids).last_hidden_state
        npy_save(args.out, f"{name}_t5_encoder_out", t5_out.numpy())

        cross_states = t5_out
        if risk["enc_to_dec_proj_applied"]:
            with torch.no_grad():
                cross_states = model.enc_to_dec_proj(cross_states)
        npy_save(args.out, f"{name}_cross_states", cross_states.numpy())

        with torch.no_grad():
            prompt_embeds = model.embed_prompts(prompt_ids)
        npy_save(args.out, f"{name}_prompt_embeds", prompt_embeds.numpy())

        # capture decoder final-norm output at prefill via hook
        prefill_capture = {}
        def hook(_m, _i, out):
            if "h" not in prefill_capture:
                prefill_capture["h"] = out.detach().clone()
        h = model.decoder.model.decoder.layer_norm.register_forward_hook(hook)

        # The ParlerTTSLogitsProcessor mutates scores IN PLACE; when it is the
        # first processor (large-v1 has no min_new_tokens, so nothing clones
        # before it) the "raw" logits captured by output_logits get the -inf
        # EOS mask stamped into them. Prepend a semantics-neutral cloning
        # processor so the dumped logits stay truly raw.
        from transformers import LogitsProcessor, LogitsProcessorList

        class _CloneScores(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return scores.clone()

        proc_list = LogitsProcessorList([
            _CloneScores(),
            ParlerTTSLogitsProcessor(eos_id, num_codebooks, batch_size=1,
                                     device=desc_ids.device),
        ])

        torch.manual_seed(args.seed)
        with torch.no_grad():
            gen = model.generate(
                input_ids=desc_ids,
                prompt_input_ids=prompt_ids,
                do_sample=False,
                num_beams=1,
                max_length=min(gen_cfg.max_length, args.max_new_frames),
                return_dict_in_generate=True,
                output_logits=True,
                output_scores=True,
                logits_processor=proc_list,
            )
        h.remove()

        npy_save(args.out, f"{name}_dec_prefill_hidden", prefill_capture["h"].numpy())
        step_logits = torch.stack(gen.logits[: args.max_step_logits])  # [S, 9, vocab] raw
        npy_save(args.out, f"{name}_step_logits", step_logits.numpy())
        step_scores = torch.stack(gen.scores[: args.max_step_logits])  # [S, 9, vocab] processed
        npy_save(args.out, f"{name}_step_scores", step_scores.numpy())

        # NOTE: composite generate() replaces .sequences with the decoded WAVEFORM
        # (modeling_parler_tts.py:3650). Reconstruct the delayed token sequences from
        # the processed scores: greedy = argmax of processed scores at each step.
        steps = torch.stack([s.argmax(dim=-1) for s in gen.scores], dim=1)  # [9, S]
        seqs = torch.cat([torch.full((num_codebooks, 1), bos_id, dtype=torch.long), steps], dim=1)
        npy_save(args.out, f"{name}_greedy_delayed", seqs.numpy().astype(np.int64))

        # un-delay exactly like generate() tail (modeling:3585-3597)
        L = seqs.shape[1]
        mask2 = model.decoder.build_delay_pattern_mask(
            torch.full((num_codebooks, 1), bos_id, dtype=torch.long),
            bos_token_id=bos_id, pad_token_id=pad_id, max_length=L)[1]
        output_ids = model.decoder.apply_delay_pattern_mask(seqs, mask2)
        keep = (mask2 != bos_id) & (mask2 != pad_id)
        codes = output_ids[keep].reshape(1, num_codebooks, -1)
        # drop frames containing any special token (decode_sequentially rule, modeling:3615-3636)
        frame_ok = (codes < dac.config.codebook_size).all(dim=1).squeeze(0)
        codes = codes[:, :, frame_ok].to(torch.long)
        npy_save(args.out, f"{name}_codes", codes.numpy().astype(np.int64))

        with torch.no_grad():
            z_q = dac_core.quantizer.from_codes(codes)[0]
            npy_save(args.out, f"{name}_dac_latent", z_q.numpy())
            wav = dac_decode(z_q)
        wav_np = wav.squeeze().numpy()
        # the composite generate's own waveform must match our reconstruction
        gen_wav = gen.sequences.squeeze().numpy()
        assert gen_wav.shape == wav_np.shape and np.allclose(gen_wav, wav_np, atol=1e-5), \
            f"reconstructed wav mismatch: {gen_wav.shape} vs {wav_np.shape}"
        npy_save(args.out, f"{name}_wav_greedy", wav_np)
        import soundfile as sf
        sf.write(os.path.join(args.out, f"{name}_wav_greedy.wav"), wav_np, dac.config.sampling_rate)
        print(f"  wav: {len(wav_np) / dac.config.sampling_rate:.2f}s")

        meta["cases"].append({
            "name": name, "description": case["description"], "prompt": case["prompt"],
            "desc_len": int(desc_ids.shape[1]), "prompt_len": int(prompt_ids.shape[1]),
            "delayed_len": int(seqs.shape[1]), "frames": int(codes.shape[2]),
            "wav_samples": int(len(wav_np)),
        })

    # small random-codes DAC-only case (decouples codec parity from LM parity)
    gr = torch.Generator().manual_seed(777)
    rnd_codes = torch.randint(0, dac.config.codebook_size, (1, num_codebooks, 40), generator=gr)
    with torch.no_grad():
        z_q = dac_core.quantizer.from_codes(rnd_codes)[0]
        wav = dac_decode(z_q)
    npy_save(args.out, "dacrand_codes", rnd_codes.numpy().astype(np.int64))
    npy_save(args.out, "dacrand_latent", z_q.numpy())
    npy_save(args.out, "dacrand_wav", wav.squeeze().numpy())

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print("meta.json written. DONE")


if __name__ == "__main__":
    sys.exit(main())
