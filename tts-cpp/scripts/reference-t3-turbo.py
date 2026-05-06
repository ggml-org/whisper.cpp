#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official Chatterbox Turbo T3 and compare to ggml port.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/chatterbox"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--cpp-bin", type=Path, help="Optional ggml binary for comparison.")
    parser.add_argument("--cpp-model", type=Path, help="GGUF model path for --cpp-bin.")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def write_tokens(path: Path, tokens: list[int]) -> None:
    path.write_text(",".join(str(t) for t in tokens) + "\n")


def read_tokens(path: Path) -> list[int]:
    raw = path.read_text().strip()
    if not raw:
        return []
    return [int(t.strip()) for t in raw.replace("\n", ",").split(",") if t.strip()]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = ChatterboxTurboTTS.from_pretrained(device=args.device)
    assert model.conds is not None, "Built-in voice conditionals required"

    text = punc_norm(args.text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
    speech_tokens = model.t3.inference_turbo(
        t3_cond=model.conds.t3, text_tokens=text_tokens,
        temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repeat_penalty, max_gen_len=args.n_predict,
    )

    text_list = text_tokens.squeeze(0).detach().cpu().tolist()
    ref_list = speech_tokens.squeeze(0).detach().cpu().tolist()

    (args.out_dir / "normalized.txt").write_text(text + "\n")
    write_tokens(args.out_dir / "text_tokens.txt", text_list)
    write_tokens(args.out_dir / "ref_speech_tokens.txt", ref_list)
    print(f"Normalized text: {args.out_dir / 'normalized.txt'}")
    print(f"Text tokens:     {args.out_dir / 'text_tokens.txt'}")
    print(f"Ref tokens:      {args.out_dir / 'ref_speech_tokens.txt'}")

    if args.cpp_bin:
        assert args.cpp_model, "--cpp-model required with --cpp-bin"
        cpp_out = args.out_dir / "ggml_speech_tokens.txt"
        cmd = [
            str(args.cpp_bin), "--model", str(args.cpp_model),
            "--tokens-file", str(args.out_dir / "text_tokens.txt"),
            "--output", str(cpp_out),
            "--n-predict", str(args.n_predict),
            "--top-k", str(args.top_k), "--top-p", str(args.top_p),
            "--temp", str(args.temp), "--repeat-penalty", str(args.repeat_penalty),
            "--seed", str(args.seed), "--threads", str(args.threads),
        ]
        print("\nRunning ggml port:\n" + " ".join(cmd))
        subprocess.run(cmd, check=True)

        ggml_tokens = read_tokens(cpp_out)
        prefix = 0
        for r, g in zip(ref_list, ggml_tokens):
            if r != g:
                break
            prefix += 1
        exact = ref_list == ggml_tokens
        print(f"\nExact match: {'yes' if exact else 'no'}")
        print(f"Reference length: {len(ref_list)}")
        print(f"GGML length:      {len(ggml_tokens)}")
        print(f"Matching prefix:  {prefix}")


if __name__ == "__main__":
    main()
