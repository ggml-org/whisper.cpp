#!/usr/bin/env python3
# Generates a *minimal* legacy-format whisper ggml `.bin` for tests.
#
# The layout mirrors models/convert-pt-to-ggml.py (magic, hparams, mel filters,
# tokenizer, tensors). With --bad-n-dims N the first tensor record declares
# n_dims=N; whisper_model_load reads n_dims values into a fixed 4-element stack
# array, so N>4 is a stack buffer overflow (CWE-121). Used to verify the guard
# added for issue #3944.

import argparse
import struct

GGML_FILE_MAGIC = 0x67676D6C  # "ggml"
GGML_TYPE_F32 = 0

# Minimal, self-consistent hparams: n_text_state must equal n_audio_state, and
# ftype=0 maps to F32. Everything is 1 so model construction stays tiny.
HPARAMS = [
    ("n_vocab", 1),
    ("n_audio_ctx", 1),
    ("n_audio_state", 1),
    ("n_audio_head", 1),
    ("n_audio_layer", 1),
    ("n_text_ctx", 1),
    ("n_text_state", 1),
    ("n_text_head", 1),
    ("n_text_layer", 1),
    ("n_mels", 1),
    ("ftype", 0),
]


def generate(output_path, n_dims):
    with open(output_path, "wb") as f:
        f.write(struct.pack("i", GGML_FILE_MAGIC))
        for _, v in HPARAMS:
            f.write(struct.pack("i", v))

        # mel filters: n_mel, n_fft, then n_mel*n_fft float32
        n_mel, n_fft = 1, 1
        f.write(struct.pack("i", n_mel))
        f.write(struct.pack("i", n_fft))
        for _ in range(n_mel * n_fft):
            f.write(struct.pack("f", 0.0))

        # tokenizer vocab: n_vocab, then per token: len + bytes
        f.write(struct.pack("i", 1))
        token = b"x"
        f.write(struct.pack("i", len(token)))
        f.write(token)

        # first tensor record: n_dims (attacker-controlled), name len, ttype,
        # then n_dims shape ints, then the name bytes.
        name = b"encoder.positional_embedding"
        f.write(struct.pack("iii", n_dims, len(name), GGML_TYPE_F32))
        for _ in range(n_dims):
            f.write(struct.pack("i", 1))
        f.write(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--bad-n-dims", type=int, default=64,
                    help="n_dims to declare for the first tensor (>4 overflows)")
    args = ap.parse_args()
    generate(args.output, args.bad_n_dims)
    print(f"wrote {args.output} (n_dims={args.bad_n_dims})")


if __name__ == "__main__":
    main()
