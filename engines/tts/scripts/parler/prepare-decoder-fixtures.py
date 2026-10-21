#!/usr/bin/env python3
"""Bounded CPU/fp32 HF sampled fixtures and independent teacher-forced replay.

No ML packages are imported for --help, --plan-only or legacy-manifest.
Outputs must be new directories. Existing greedy fixtures are never renamed
or overwritten. Each generated case records exact sampled history, actual
delayed decoder inputs, raw (pre-processor) logits, DAC codes and waveform.

  prepare-decoder-fixtures.py sample --model /models/hf-mini-v1 \
      --cases-json /models/parler-ref/meta.json --out /tmp/mini-sampled \
      --max-length 256 --seed 42 --plan-only
  prepare-decoder-fixtures.py sample --model /models/hf-indic \
      --cases-json scripts/parler/indic-cases.json --out /tmp/indic-sampled

For a 200-step fp32 reference on the OLD greedy history (not a twin with
fp16 KV), first create a manifest and then replay it with HF's decoder:
  prepare-decoder-fixtures.py legacy-manifest --model /models/hf-mini-v1 \
      --reference-dir /models/parler-ref --out /tmp/mini-history --steps 200
  prepare-decoder-fixtures.py replay --manifest /tmp/mini-history/manifest.json \
      --out /tmp/mini-fp32

For --steps 20 (or fewer), legacy-manifest can reuse existing raw HF logits
when EVERY requested step is present. A 200-step plan never pads/repeats the
first 20 logits or treats greedy token argmax as a logits reference.

HF/model/DAC runs are deliberately separate from validation and timing. A
sample max_length is a delayed-token bound, not a promise of complete speech.
Listen/check EOS and duration before treating bounded audio as a speech gate.
"""
import argparse
from contextlib import ExitStack
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shutil

import numpy as np


def validation():
    spec = importlib.util.spec_from_file_location("parler_decoder_validation", Path(__file__).with_name("validate-decoder-coreml.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_cases(path):
    value = json.loads(Path(path).read_text())
    cases = value["cases"] if isinstance(value, dict) else value
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        raise ValueError("provide 1..8 cases per bounded fixture run")
    seen = set()
    for case in cases:
        cid = case.get("id", case.get("name"))
        if not isinstance(cid, str) or not cid or Path(cid).name != cid or cid in (".", "..") or cid in seen:
            raise ValueError("cases need unique filename-safe id/name")
        if not case.get("prompt") or not case.get("description"):
            raise ValueError("each case needs prompt and description captions")
        seen.add(cid)
    return cases


def save_arrays(directory, prefix, arrays):
    helper = validation()
    files, hashes = {}, {}
    for name, value in arrays.items():
        file = directory / f"{prefix}_{name}.npy"
        if file.exists():
            raise ValueError(f"refusing to overwrite {file}")
        np.save(file, np.ascontiguousarray(value))
        files[name] = file.name
        hashes[name] = helper.sha256(file)
    return files, hashes


def checkpoint_identity(model_path, hash_weights=True):
    helper = validation()
    path = Path(model_path).resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("--model must be a local HF checkpoint directory")
    files = sorted(path.glob("*.json"))
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise ValueError("local safetensors checkpoint required")
    if hash_weights:
        files += weights
    return {"path": str(path), "files_sha256": {p.name: helper.sha256(p) for p in files},
            "weight_hashes_included": hash_weights}


def load_hf(model_path, threads, description_tokenizer=None, tokenizers=True):
    import torch
    from transformers import AutoTokenizer
    from parler_tts import ParlerTTSForConditionalGeneration
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype=torch.float32, attn_implementation="eager",
        local_files_only=True).cpu().eval()
    # parler-tts 0.2.x decoder configs omit audio_channels (mono implied).
    if model.prompt_cross_attention or getattr(model.config.decoder, "audio_channels", 1) != 1:
        raise ValueError("fixture schema currently supports mono/prompt-prepend Parler only")
    repaired = repair_weight_norm(model, Path(model_path))
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "parler-tts", "numpy")}
    runtime = {"versions": versions, "device": "cpu", "dtype": "float32",
               "attention": "eager", "threads": threads, "weight_norm_repairs": repaired,
               "platform": platform.platform(), "machine": platform.machine()}
    prompt_tok = desc_tok = None
    if tokenizers:
        prompt_tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        shared = "T5" in type(prompt_tok).__name__
        desc_tok = prompt_tok if shared and description_tokenizer is None else AutoTokenizer.from_pretrained(
            str(description_tokenizer or model.config.text_encoder._name_or_path), local_files_only=True)
        runtime.update(description_tokenizer=str(description_tokenizer or desc_tok.name_or_path),
                       prompt_tokenizer=str(prompt_tok.name_or_path))
    return torch, model, prompt_tok, desc_tok, runtime


def repair_weight_norm(model, path):
    """Restore checkpoint weight-norm tensors after HF sharded-load init bugs.

    Same checkpoint mapping as dump-reference.py, but check BOTH original
    tensors so a matching gain never hides a mismatched direction tensor.
    """
    import torch
    from safetensors import safe_open
    index = path / "model.safetensors.index.json"
    with ExitStack() as stack:
        handles = {}
        if index.exists():
            mapping = json.loads(index.read_text())["weight_map"]
        else:
            handle = stack.enter_context(safe_open(str(path / "model.safetensors"), framework="pt"))
            handles["model.safetensors"] = handle
            mapping = {key: "model.safetensors" for key in handle.keys()}

        def tensor(name):
            file = mapping[name]
            if file not in handles:
                handles[file] = stack.enter_context(safe_open(str(path / file), framework="pt"))
            return handles[file].get_tensor(name)

        params, repaired = dict(model.named_parameters()), 0
        with torch.no_grad():
            for name, gain in params.items():
                suffix = "parametrizations.weight.original0"
                if not name.endswith(suffix):
                    continue
                base = name[:-len(suffix)]
                gain_key, direction_key = base + "weight_g", base + "weight_v"
                if gain_key not in mapping or direction_key not in mapping:
                    continue
                direction = params[base + "parametrizations.weight.original1"]
                g, v = tensor(gain_key), tensor(direction_key)
                if not torch.equal(gain, g) or not torch.equal(direction, v):
                    gain.copy_(g)
                    direction.copy_(v)
                    repaired += 1
        return repaired


def seed_all(torch, seed):
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_sample(torch, model, kwargs):
    """Clone inside forward/_sample; sampling is never reconstructed by argmax."""
    raw, inputs, sampled, codes, first = [], [], [], [], {}

    def pre(_module, args, kw):
        frame = kw.get("input_ids", args[0] if args else None)
        if frame is None or frame.shape != (model.decoder.num_codebooks, 1):
            raise ValueError("expected one mono delayed input frame per decoder call")
        inputs.append(frame[:, 0].detach().cpu().numpy().copy())
        if not first:
            for key in ("encoder_hidden_states", "prompt_hidden_states"):
                if kw.get(key) is None:
                    raise ValueError(f"missing first decoder {key}")
                first[key] = kw[key].detach().cpu().numpy().copy()

    def post(_module, _args, output):
        # Clone now: HF processors may later mutate the returned logits.
        raw.append(output.logits[:, -1, :].detach().float().cpu().numpy().copy())

    original_sample, original_decode = model._sample, model.audio_encoder.decode

    def sample(*args, **kw):
        output = original_sample(*args, **kw)
        sampled.append(output.sequences.detach().cpu().numpy().copy())
        return output

    def decode(*args, **kw):
        value = kw.get("audio_codes", args[0] if args else None)
        if value is None:
            raise ValueError("DAC decode did not receive codes")
        value = value.detach().cpu().numpy().copy()
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError("expected one DAC sample")
        codes.append(value)
        return original_decode(*args, **kw)

    pre_hook = model.decoder.register_forward_pre_hook(pre, with_kwargs=True)
    post_hook = model.decoder.register_forward_hook(post)
    model._sample, model.audio_encoder.decode = sample, decode
    try:
        with torch.inference_mode():
            generated = model.generate(**kwargs, return_dict_in_generate=True, output_logits=False, output_scores=False)
    finally:
        pre_hook.remove()
        post_hook.remove()
        model._sample, model.audio_encoder.decode = original_sample, original_decode
    if len(sampled) != 1 or len(codes) != 1 or len(raw) != len(inputs):
        raise ValueError("expected one sampled sequence and one nonempty DAC decode")
    if sampled[0].shape[1] != len(raw) + 1:
        raise ValueError("sample history and raw logits are misaligned")
    waveform = generated.sequences.detach().float().cpu().numpy().reshape(-1).copy()
    return {"reference_logits": np.stack(raw), "decoder_inputs": np.stack(inputs),
            "delayed_tokens": sampled[0], "codes": codes[0], "waveform": waveform,
            "cross_states": first["encoder_hidden_states"], "prompt_embeds": first["prompt_hidden_states"]}


def sample_fixtures(args):
    cases = read_cases(args.cases_json)
    if not 1 <= args.threads <= 16 or not 1 <= args.max_length <= 512:
        raise ValueError("bounded fixtures require threads 1..16 and max_length 1..512")
    if args.top_k < 1 or not 0 < args.top_p <= 1 or not np.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("invalid sampling parameters")
    if not 0 <= args.seed < 2**32 or args.seed + len(cases) - 1 >= 2**32:
        raise ValueError("seed range exceeds uint32")
    identity = checkpoint_identity(args.model, hash_weights=not args.plan_only)
    plan = {"model": identity, "cases": cases, "seed": args.seed, "max_length": args.max_length,
            "top_k": args.top_k, "top_p": args.top_p, "temperature": args.temperature, "out": str(args.out)}
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    torch, model, prompt_tok, desc_tok, runtime = load_hf(args.model, args.threads, args.description_tokenizer)
    cfg = model.generation_config
    nq = model.decoder.num_codebooks
    if args.max_length < 2 * nq or args.max_length > cfg.max_length:
        raise ValueError("max_length must allow at least one audio frame and fit model capacity")
    manifest = {"schema_version": 1, "kind": "sampled-hf-fp32", "model": identity, "runtime": runtime,
                "token_config": {"bos": cfg.bos_token_id, "eos": cfg.eos_token_id, "pad": cfg.pad_token_id,
                                 "codebook_size": model.audio_encoder.config.codebook_size, "max_length": args.max_length},
                "generation": {**model.generation_config.to_dict(), "do_sample": True, "num_beams": 1,
                               "max_length": args.max_length, "top_k": args.top_k, "top_p": args.top_p,
                               "temperature": args.temperature}, "cases": []}
    import soundfile as sf
    for i, case in enumerate(cases):
        cid = case.get("id", case.get("name"))
        seed = int(case.get("seed", args.seed + i))
        seed_all(torch, seed)
        desc = desc_tok(case["description"], return_tensors="pt")
        prompt = prompt_tok(case["prompt"], return_tensors="pt")
        if max(desc.input_ids.shape[1], prompt.input_ids.shape[1]) > 256:
            raise ValueError("bounded fixtures limit description/prompt to 256 tokens each")
        arrays = capture_sample(torch, model, {"input_ids": desc.input_ids, "attention_mask": desc.attention_mask,
                   "prompt_input_ids": prompt.input_ids, "prompt_attention_mask": prompt.attention_mask,
                   "do_sample": True, "num_beams": 1, "max_length": args.max_length,
                   "top_k": args.top_k, "top_p": args.top_p, "temperature": args.temperature})
        arrays.update(prompt_ids=prompt.input_ids.numpy(), description_ids=desc.input_ids.numpy())
        if not np.isfinite(arrays["reference_logits"]).all() or not np.isfinite(arrays["waveform"]).all():
            raise ValueError("nonfinite raw HF logits/waveform")
        files, hashes = save_arrays(args.out, cid + "_sampled", arrays)
        sr = model.audio_encoder.config.sampling_rate
        wav_file = args.out / f"{cid}_sampled.wav"
        sf.write(str(wav_file), arrays["waveform"], sr, subtype="FLOAT")
        hit_bound = arrays["delayed_tokens"].shape[1] >= args.max_length
        manifest["cases"].append({"id": cid, "prompt": case["prompt"], "description": case["description"],
            "seed": seed, "files": files, "sha256": hashes, "wav": wav_file.name,
            "wav_sha256": validation().sha256(wav_file), "sample_rate": sr,
            "frames": int(arrays["codes"].shape[-1]), "duration_s": arrays["waveform"].size / sr,
            "hit_length_bound": hit_bound, "speech_completion_verified": False})
        print(f"{cid}: {arrays['decoder_inputs'].shape[0]} steps, {arrays['codes'].shape[-1]} DAC frames; bound={hit_bound}", flush=True)
    validation().write_json(args.out / "manifest.json", manifest)


def legacy_manifest(args):
    if not 1 <= args.steps <= 512:
        raise ValueError("steps must be 1..512")
    meta = json.loads((args.reference_dir / "meta.json").read_text())
    dec = meta["config"]["dec"]
    histories = {case["name"]: np.load(args.reference_dir / f"{case['name']}_greedy_delayed.npy",
                                       allow_pickle=False) for case in meta["cases"]}
    for cid, history in histories.items():
        # Old dumps were generated with max_length=min(gen max, --max-new-frames),
        # so the capture length (not gen max_length) bounds the forced-PAD tail:
        # columns >= length-nq+1+k were overwritten. Stop before that boundary.
        nq, captured = history.shape
        if args.steps > captured - nq:
            raise ValueError(f"{cid}: {captured}-column bounded history allows at most "
                             f"{captured - nq} steps before the forced-PAD boundary")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "kind": "teacher-history-needs-fp32-replay",
                "model": checkpoint_identity(args.model, hash_weights=False),
                "token_config": {"bos": dec["bos"], "eos": dec["eos"], "pad": dec["pad"],
                    "codebook_size": meta["config"]["dac"]["codebook_size"],
                    "max_length": meta["config"]["gen"]["max_length"]}, "cases": []}
    for case in meta["cases"]:
        cid = case["name"]
        history = histories[cid]
        captured = history.shape[1]
        frames = history[:, :args.steps].T.copy()
        # Recreate original GENERATION input mask, not the shorter final
        # undelay mask used to package the bounded waveform in old fixtures.
        maximum = min(manifest["token_config"]["max_length"], captured)
        if maximum >= 2 * frames.shape[1] - 1:
            for k in range(frames.shape[1]):
                cols = np.arange(args.steps)
                frames[cols <= k, k] = dec["bos"]
                frames[cols >= maximum - frames.shape[1] + 1 + k, k] = dec["pad"]
        arrays = {"decoder_inputs": frames, "delayed_tokens": history[:, :args.steps + 1],
                  "prompt_ids": np.load(args.reference_dir / f"{cid}_prompt_ids.npy", allow_pickle=False),
                  "description_ids": np.load(args.reference_dir / f"{cid}_desc_ids.npy", allow_pickle=False),
                  "cross_states": np.load(args.reference_dir / f"{cid}_cross_states.npy", allow_pickle=False),
                  "prompt_embeds": np.load(args.reference_dir / f"{cid}_prompt_embeds.npy", allow_pickle=False),
                  "anchor_logits": np.load(args.reference_dir / f"{cid}_step_logits.npy", allow_pickle=False)[:args.steps]}
        if len(arrays["anchor_logits"]) == args.steps:
            arrays["reference_logits"] = arrays["anchor_logits"].copy()
        files, hashes = save_arrays(args.out, cid + "_teacher", arrays)
        manifest["cases"].append({"id": cid, "prompt": case["prompt"], "description": case["description"],
                                  "files": files, "sha256": hashes, "token_config": {"max_length": maximum}})
    if all("reference_logits" in c["files"] for c in manifest["cases"]):
        manifest["kind"] = "existing-hf-fp32-raw-reference"
    validation().write_json(args.out / "manifest.json", manifest)


def replay_logits(torch, decoder, frames, cross, prompt):
    """Independent HF reference with fresh fp32 KV, reset for every case."""
    past, logits = None, []

    def cache_tensors(value):
        if hasattr(value, "to_legacy_cache"):
            value = value.to_legacy_cache()
        if torch.is_tensor(value):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from cache_tensors(item)

    with torch.inference_mode():
        for i, frame in enumerate(frames):
            result = decoder(input_ids=torch.from_numpy(frame.astype(np.int64))[:, None],
                encoder_hidden_states=cross, prompt_hidden_states=prompt if i == 0 else None,
                past_key_values=past, use_cache=True, return_dict=True)
            past = result.past_key_values
            tensors = list(cache_tensors(past))
            if not tensors or any(t.dtype != torch.float32 for t in tensors):
                raise ValueError("reference requires real fp32 HF KV, not fp16/quantized cache")
            logits.append(result.logits[:, -1, :].detach().cpu().numpy().copy())
    return np.stack(logits)


def replay_fixtures(args):
    helper = validation()
    manifest, root, digest = helper.load_manifest(args.manifest)
    if args.plan_only:
        print(json.dumps({"model": manifest["model"], "cases": [c["id"] for c in manifest["cases"]],
                          "out": str(args.out), "source_fixture_sha256": digest}, indent=2))
        return
    if not 1 <= args.threads <= 16:
        raise ValueError("threads must be 1..16")
    args.out.mkdir(parents=True, exist_ok=False)
    model_path = Path(manifest["model"]["path"])
    identity = checkpoint_identity(model_path)
    for name, expected in manifest["model"]["files_sha256"].items():
        if identity["files_sha256"].get(name) != expected:
            raise ValueError(f"source model identity changed: {name}")
    torch, model, _, _, runtime = load_hf(model_path, args.threads, tokenizers=False)
    output = {**manifest, "model": identity, "runtime": runtime, "kind": "hf-fp32-teacher-forced",
              "source_fixture_sha256": digest, "cases": []}
    for case in manifest["cases"]:
        arrays = helper.arrays_for(case, root)
        frames = arrays["decoder_inputs"]
        if not 1 <= len(frames) <= 512:
            raise ValueError("replay supports 1..512 steps per case")
        cross = torch.from_numpy(arrays["cross_states"]).float()
        prompt = torch.from_numpy(arrays["prompt_embeds"]).float()
        # Real HF decoder with fp32 weights AND fp32 self/cross KV. No twin,
        # processor, teacher-score argmax, or fp16 cache defines the reference.
        reference = replay_logits(torch, model.decoder, frames, cross, prompt)
        if not np.isfinite(reference).all():
            raise ValueError("nonfinite fp32 reference")
        anchor = arrays.get("anchor_logits", arrays.get("reference_logits"))
        anchor_result = None
        if anchor is not None:
            n = min(len(anchor), len(reference))
            agreement = float(np.mean(anchor[:n].argmax(-1) == reference[:n].argmax(-1)))
            max_error = float(np.max(np.abs(anchor[:n] - reference[:n])))
            anchor_result = {"steps": n, "argmax_agreement": agreement, "max_abs_logit_error": max_error}
            if agreement != 1.0 or not np.allclose(anchor[:n], reference[:n], atol=1e-4, rtol=1e-4):
                raise ValueError(f"{case['id']}: fp32 HF anchor mismatch: {anchor_result}")
        arrays["reference_logits"] = reference
        files, hashes = save_arrays(args.out, case["id"] + "_fp32", arrays)
        saved_case = {**case, "files": files, "sha256": hashes, "anchor_check": anchor_result}
        if case.get("wav"):
            source_wav = root / case["wav"]
            if helper.sha256(source_wav) != case["wav_sha256"]:
                raise ValueError("sampled WAV checksum mismatch")
            target_wav = args.out / Path(case["wav"]).name
            if target_wav.exists():
                raise ValueError(f"refusing to overwrite {target_wav}")
            shutil.copyfile(source_wav, target_wav)
            saved_case["wav"] = target_wav.name
        output["cases"].append(saved_case)
        print(f"{case['id']}: replayed {len(reference)} steps; anchor={anchor_result}", flush=True)
    helper.write_json(args.out / "manifest.json", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample")
    sample.add_argument("--model", required=True, type=Path)
    sample.add_argument("--cases-json", required=True, type=Path)
    sample.add_argument("--out", required=True, type=Path)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--max-length", type=int, default=256)
    sample.add_argument("--top-k", type=int, default=50)
    sample.add_argument("--top-p", type=float, default=1.0)
    sample.add_argument("--temperature", type=float, default=1.0)
    sample.add_argument("--threads", type=int, default=2)
    sample.add_argument("--description-tokenizer", type=Path)
    sample.add_argument("--plan-only", action="store_true")
    legacy = sub.add_parser("legacy-manifest")
    legacy.add_argument("--model", required=True, type=Path)
    legacy.add_argument("--reference-dir", required=True, type=Path)
    legacy.add_argument("--out", required=True, type=Path)
    legacy.add_argument("--steps", type=int, default=200)
    replay = sub.add_parser("replay")
    replay.add_argument("--manifest", required=True, type=Path)
    replay.add_argument("--out", required=True, type=Path)
    replay.add_argument("--threads", type=int, default=2)
    replay.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    {"sample": sample_fixtures, "legacy-manifest": legacy_manifest, "replay": replay_fixtures}[args.command](args)


if __name__ == "__main__":
    main()
