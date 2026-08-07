#!/usr/bin/env python3
"""Dump Audio8 DualAR references for the ggml port.

Runs the real generation path with greedy decoding so the trajectory is
deterministic and the C++ graph can be compared step by step. The shipped
sampler draws Gumbel noise and applies repetition-aware resampling, neither of
which can be reproduced across runtimes; what *can* be compared is everything
feeding the draw, so the filtered score vectors are dumped alongside the tokens.

Written to --out-dir:

  prompt.npy         [11, T] int32   packed semantic/codebook prompt rows
  prompt_mask.npy    [T]     int32   1 for real positions, 0 for left padding
  embed.npy          [T, 896]        the summed token + codebook embedding
  slow_hidden.npy    [S, 896]        per-step fast-AR input (normalised hidden)
  sem_logits.npy     [S, 4097]       raw logits over the selectable rows,
                                     ordered semantic_begin..semantic_end, EOS
  filtered.npy       [S, 4097]       the same rows after masking, top-k/top-p
                                     and temperature, as the sampler sees them
  semantic.npy       [S]     int32   chosen semantic token ids (absolute)
  fast_logits.npy    [S, 9, 4096]    fast-AR logits for codebooks 1..9
  codes.npy          [10, S] int32   the emitted codec frames
  wav.npy            [N]             with --dump-wav, those frames decoded, which
                                     is what the end-to-end engine test compares
  meta.json                          text, sampling settings and shapes

    python3 dump-audio8-lm-reference.py \\
        --model-dir models/Audio8-TTS-Preview-0.6b \\
        --out-dir artifacts/audio8-ref --max-new-tokens 32
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio8_reference import (  # noqa: E402
    load_generation_model,
    save_arrays,
    to_numpy,
    write_meta,
)

FAST_PRIMING_POSITION = 0


class Recorder:
    def __init__(self, model):
        self.model = model
        self.embed = None
        self.slow_hidden = []
        self.sem_logits = []
        self.filtered = []
        self.fast_logits = []
        self._original = {}

    def _selectable_rows(self, logits):
        config = self.model.config
        semantic = logits[0, config.semantic_begin_id : config.semantic_end_id + 1]
        return np.append(to_numpy(semantic), to_numpy(logits[0, config.eos_token_id]))

    def _wrap_embed(self, original):
        def wrapped(input_ids):
            output = original(input_ids)
            if self.embed is None:
                self.embed = to_numpy(output[0])
            return output
        return wrapped

    def _wrap_slow_step(self, original):
        def wrapped(input_ids, cache_position, position_ids, attention_mask):
            logits, hidden = original(
                input_ids, cache_position, position_ids, attention_mask
            )
            self.slow_hidden.append(to_numpy(hidden[0, -1]))
            self.sem_logits.append(self._selectable_rows(logits))
            return logits, hidden
        return wrapped

    def _wrap_fast_step(self, original):
        def wrapped(hidden, position):
            scores = original(hidden, position)
            if position != FAST_PRIMING_POSITION:
                self.fast_logits.append(to_numpy(scores[0]))
            return scores
        return wrapped

    def _wrap_processed_scores(self, original):
        def wrapped(input_ids, scores, processors, top_k, top_p, temperature):
            output = original(input_ids, scores, processors, top_k, top_p, temperature)
            if output.shape[-1] == self.model.config.vocab_size:
                self.filtered.append(self._selectable_rows(output))
            return output
        return wrapped

    def attach(self):
        wrappers = {
            "_embed": self._wrap_embed,
            "_slow_step": self._wrap_slow_step,
            "_fast_step": self._wrap_fast_step,
            "_processed_scores": self._wrap_processed_scores,
        }
        for name, factory in wrappers.items():
            original = getattr(self.model, name)
            self._original[name] = original
            setattr(self.model, name, factory(original))

    def detach(self):
        for name, original in self._original.items():
            setattr(self.model, name, original)


def build_inputs(processor, text, reference_codes, reference_text):
    if reference_codes is None:
        return processor(text=text)
    return processor(
        text=text, reference_text=reference_text, reference_codes=reference_codes
    )


def load_reference_codes(path):
    return None if path is None else np.load(path)


def stack_steps(values, name):
    if not values:
        raise ValueError(f"no {name} were recorded")
    return np.stack(values).astype(np.float32)


def reshape_fast_logits(values, steps, num_codebooks):
    per_step = num_codebooks - 1
    array = stack_steps(values, "fast logits")
    return array.reshape(steps, per_step, -1)


def semantic_from_codes(codes, semantic_begin):
    return (codes[0].astype(np.int64) + semantic_begin).astype(np.int32)


def waveform_arrays(model, codes, wanted):
    if not wanted:
        return {}
    waveform, _ = model.decode_audio(codes)
    return {"wav.npy": to_numpy(waveform[0]).astype(np.float32)}


def step_arrays(recorder, codes, steps, config):
    return {
        "embed.npy": recorder.embed,
        "slow_hidden.npy": stack_steps(recorder.slow_hidden[:steps], "hidden states"),
        "sem_logits.npy": stack_steps(recorder.sem_logits[:steps], "semantic logits"),
        "filtered.npy": stack_steps(recorder.filtered[:steps], "filtered scores"),
        "fast_logits.npy": reshape_fast_logits(
            recorder.fast_logits[: steps * (config.num_codebooks - 1)],
            steps, config.num_codebooks,
        ),
        "codes.npy": codes,
        "semantic.npy": semantic_from_codes(codes, config.semantic_begin_id),
    }


def sampling_settings(model):
    config = model.generation_config
    return {
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "top_k": int(config.top_k),
        "do_sample": False,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", default="Audio eight speaks on the CPU.")
    parser.add_argument("--reference-codes", help="[10, T] .npy from the codec encoder")
    parser.add_argument("--reference-text")
    parser.add_argument("--out-dir", default="artifacts/audio8-ref")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dump-wav", action="store_true",
                        help="also decode the frames, which loads the codec")
    return parser.parse_args()


def main():
    import torch

    args = parse_args()
    torch.manual_seed(0)
    processor, model = load_generation_model(args.model_dir)
    inputs = build_inputs(
        processor, args.text, load_reference_codes(args.reference_codes),
        args.reference_text,
    )

    recorder = Recorder(model)
    recorder.attach()
    try:
        codes = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    finally:
        recorder.detach()

    prompt, prompt_mask = model._prepare_prompt(**inputs)
    steps = int(codes.shape[-1])
    config = model.config
    print(f"dumping {steps} generated frames to {args.out_dir}")

    arrays = {
        "prompt.npy": to_numpy(prompt[0]).astype(np.int32),
        "prompt_mask.npy": to_numpy(prompt_mask[0]).astype(np.int32),
    }
    arrays.update(step_arrays(
        recorder, to_numpy(codes[0]).astype(np.int32), steps, config
    ))
    arrays.update(waveform_arrays(model, codes, args.dump_wav))
    save_arrays(args.out_dir, arrays)

    write_meta(args.out_dir, {
        "text": args.text,
        "reference_text": args.reference_text,
        "has_reference": args.reference_codes is not None,
        "max_new_tokens": args.max_new_tokens,
        "generated_frames": steps,
        "prompt_length": int(prompt.shape[-1]),
        "num_codebooks": int(config.num_codebooks),
        "codebook_size": int(config.codebook_size),
        "semantic_begin_id": int(config.semantic_begin_id),
        "eos_token_id": int(config.eos_token_id),
        "sampling": sampling_settings(model),
    })


if __name__ == "__main__":
    main()
