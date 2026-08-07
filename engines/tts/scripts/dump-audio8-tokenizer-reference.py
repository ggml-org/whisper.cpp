#!/usr/bin/env python3
"""Dump HF-tokenizer ground truth for the Audio8 tokenizer parity test.

Audio8 uses the Qwen2 byte-level BPE with 4131 added tokens on top: the ChatML
control tokens, the `<|voice|>` marker that opens every assistant turn, and one
`<|semantic:N|>` token per entry of the first codec codebook. The C++ tokenizer
must split those out verbatim instead of running BPE over them, and must apply
NFC normalisation before pre-tokenising, so the cases below cover both.

Output (tokenizer_ref.txt), one case per line as:
    <id,id,...>\t<exact text>
with '#'-prefixed comments. `add_special_tokens=False` matches how the Audio8
processor feeds text: the control tokens are already inline.

    python3 dump-audio8-tokenizer-reference.py \\
        --model-dir models/Audio8-TTS-Preview-0.6b \\
        --out-dir artifacts/audio8-ref
"""
import argparse
import os

SYSTEM_PROMPT = "<|im_start|>system\nconvert the provided text to speech<|im_end|>\n"
ASSISTANT_OPEN = "<|im_start|>assistant\n<|voice|>"

CASES = (
    ("ascii + punctuation", "The quick brown fox jumps over the lazy dog."),
    ("chatml system turn", SYSTEM_PROMPT),
    ("assistant turn with voice marker", ASSISTANT_OPEN),
    ("semantic tokens", "<|semantic:0|><|semantic:2047|><|semantic:4095|>"),
    ("speaker tag", "<|speaker:0|>Reference transcript."),
    ("chinese", "我们的愿景是让每个人都能使用人工智能。"),
    ("japanese", "音声合成のテストです。"),
    ("korean", "음성 합성 테스트입니다."),
    ("mixed cjk + ascii", "Audio8 是一个 TTS 模型。"),
    ("accented latin", "Übermäßig schöne Grüße, ça va très bien."),
    ("digits and symbols", "1234567890 +-*/=%$#@!"),
    ("nfc normalisation", "e\u0301gal"),
    ("repeated whitespace collapse", "spaced    out   text"),
    ("case-insensitive contractions", "IT'S theirs, They'VE gone, don'T"),
    ("emoji and astral plane", "ok \U0001f600 \U0001f9e0"),
)


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_dir, use_fast=True, trust_remote_code=True, fix_mistral_regex=False
    )


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def escape(text):
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def write_cases(handle, tokenizer):
    handle.write("# Audio8 tokenizer parity fixture -- HF AutoTokenizer ground truth\n")
    handle.write("# format: <id,id,...>\\t<escaped text>   (add_special_tokens=False)\n")
    handle.write("# escapes: \\\\ \\n \\t\n")
    for description, text in CASES:
        ids = encode(tokenizer, text)
        handle.write(f"# {description}\n")
        handle.write(",".join(str(i) for i in ids) + "\t" + escape(text) + "\n")
        print(f"  {description}: {len(ids)} ids")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", default="artifacts/audio8-ref")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.model_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "tokenizer_ref.txt")
    with open(path, "w", encoding="utf-8") as handle:
        write_cases(handle, tokenizer)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
