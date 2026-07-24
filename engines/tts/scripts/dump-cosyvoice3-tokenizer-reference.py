#!/usr/bin/env python3
"""Dump HF-tokenizer reference ids for the qwen_tokenizer parity test.

The C++ qwen_tokenizer is a hand-rolled Qwen2 byte-level BPE. This script emits
the ground-truth token ids from the real HF AutoTokenizer for a fixed set of
strings so test/test_cosyvoice_tokenizer.cpp can assert encode() reproduces
them exactly. The strings deliberately cover:
  * plain ASCII with punctuation
  * the <|endofprompt|> special token (must be split out, not BPE'd)
  * the "You are a helpful assistant." + <|endofprompt|> prompt preamble
  * a CJK (Chinese) sample — multi-byte, no ASCII whitespace splits

Output (tokenizer_ref.txt), one case per line as:
    <id,id,...>\t<exact text>
plus '#'-prefixed comment lines. `add_special_tokens=False` matches how the
CosyVoice frontend feeds text (the special marker is already inline).

Run where transformers + the model's tokenizer files are available:
    python3 dump-cosyvoice3-tokenizer-reference.py \\
        --model-dir models/Fun-CosyVoice3-0.5B --out-dir artifacts/cosyvoice3-ref
"""
import argparse
import os

ENDOFPROMPT = "<|endofprompt|>"

# (description, text) — keep single-line strings (the ref format is line-based).
CASES = [
    ("ascii + punctuation",   "The quick brown fox jumps over the lazy dog."),
    ("special: endofprompt",  ENDOFPROMPT),
    ("prompt preamble",       "You are a helpful assistant." + ENDOFPROMPT),
    ("cjk (chinese)",         "我们的愿景是让每个人都能使用人工智能。"),
    ("mixed cjk + ascii",     "CosyVoice3 是一个 TTS 模型。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="dir with the Qwen2 tokenizer files (vocab.json/merges.txt/tokenizer_config.json)")
    ap.add_argument("--out-dir", default="artifacts/cosyvoice3-ref")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "tokenizer_ref.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# qwen_tokenizer parity fixture — HF AutoTokenizer ground truth\n")
        f.write("# format: <id,id,...>\\t<exact text>   (add_special_tokens=False)\n")
        for desc, text in CASES:
            ids = tok.encode(text, add_special_tokens=False)
            f.write(f"# {desc}\n")
            f.write(",".join(str(i) for i in ids) + "\t" + text + "\n")
            print(f"  {desc}: {len(ids)} ids")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
