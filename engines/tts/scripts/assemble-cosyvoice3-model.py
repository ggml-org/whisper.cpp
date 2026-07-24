#!/usr/bin/env python3
"""Assemble a CosyVoice3 model directory for tts_cpp::cosyvoice::Engine.

The Engine resolves its components by name under `EngineOptions.model_dir`
(see resolve_component in src/cosyvoice_engine.cpp).  This script copies the
GGUFs + tokenizer + baked voice into one folder with the expected names:

    <model_dir>/
        cosyvoice3-llm-f32.gguf     Qwen2.5-0.5B speech LM
        cosyvoice3-flow-f32.gguf    DiT conditional-flow-matching estimator
        cosyvoice3-hift-f32.gguf    CausalHiFT vocoder
        voice.gguf                  baked default voice (bake-cosyvoice3-voice.py)
        vocab.json                  Qwen2 BPE vocab
        merges.txt                  Qwen2 BPE merges

    python3 assemble-cosyvoice3-model.py \\
        --llm    cosyvoice3-llm-f32.gguf \\
        --flow   cosyvoice3-flow-f32.gguf \\
        --hift   cosyvoice3-hift-f32.gguf \\
        --voice  voice.gguf \\
        --vocab  CosyVoice-BlankEN/vocab.json \\
        --merges CosyVoice-BlankEN/merges.txt \\
        --out    models/cosyvoice3-0.5b
"""
import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--hift", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--merges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symlink", action="store_true",
                    help="symlink instead of copy (saves ~3.4 GB of duplication)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    targets = {
        args.llm: "cosyvoice3-llm-f32.gguf",
        args.flow: "cosyvoice3-flow-f32.gguf",
        args.hift: "cosyvoice3-hift-f32.gguf",
        args.voice: "voice.gguf",
        args.vocab: "vocab.json",
        args.merges: "merges.txt",
    }
    for src, name in targets.items():
        if not os.path.isfile(src):
            raise SystemExit(f"missing input: {src}")
        dst = os.path.join(args.out, name)
        if os.path.lexists(dst):
            os.remove(dst)
        if args.symlink:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
        print(f"{'link' if args.symlink else 'copy'}  {name:28s} <- {src}")

    print(f"\nmodel_dir ready: {args.out}")
    print("Engine usage: EngineOptions opts; opts.model_dir = \"%s\";" % args.out)


if __name__ == "__main__":
    main()
