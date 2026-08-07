#!/usr/bin/env python3
"""Dump Audio8 neural-codec references for the ggml port.

Runs the reference codec in both directions and captures the boundary between
every stage, so a failing C++ stage can be located without bisecting the whole
network. The decode path is what synthesis needs; the encode path is what voice
enrolment needs.

Decode (from --codes, or from a fresh reference-free generation):
  codec_codes.npy      [10, T] int32  the frames fed to the decoder
  codec_semantic.npy   [1024, T]      semantic quantiser output
  codec_residual.npy   [1024, T]      summed residual quantiser output
  codec_post.npy       [1024, T]      windowed post-module output
  codec_latent.npy     [1024, 4T]     after the two upsampling stages
  codec_wav.npy        [N]            decoded waveform at 44.1 kHz

Encode (only with --audio):
  codec_audio.npy      [N]            the padded mono input at 44.1 kHz
  codec_encoded.npy    [1024, E]      convolutional encoder output
  codec_downsampled.npy[1024, T]      after downsampling and the pre-module
  codec_enc_codes.npy  [10, T] int32  quantised frames

    python3 dump-audio8-codec-reference.py \\
        --model-dir models/Audio8-TTS-Preview-0.6b \\
        --codes artifacts/audio8-ref/codes.npy \\
        --audio reference.wav --out-dir artifacts/audio8-ref
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio8_reference import (  # noqa: E402
    load_codec_module,
    load_config,
    save_arrays,
    to_numpy,
    write_meta,
)


class StageCapture:
    def __init__(self, module):
        self.module = module
        self.outputs = {}
        self.handles = []

    def _record(self, name):
        def hook(_module, _inputs, output):
            self.outputs[name] = to_numpy(output[0])
        return hook

    def watch(self, name, submodule):
        self.handles.append(submodule.register_forward_hook(self._record(name)))

    def release(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def watch_decode_stages(capture, quantizer):
    capture.watch("post", quantizer.post_module)
    capture.watch("latent", quantizer.upsample)


def watch_encode_stages(capture, codec):
    capture.watch("encoded", codec.encoder)
    capture.watch("downsampled", codec.quantizer.pre_module)


def quantizer_parts(quantizer, codes):
    import torch

    indices = torch.as_tensor(codes, dtype=torch.long).unsqueeze(0)
    semantic = quantizer.semantic_quantizer.from_codes(indices[:, :1])
    residual = quantizer.quantizer.from_codes(indices[:, 1:])
    return to_numpy(semantic[0]), to_numpy(residual[0])


def decode_reference(codec, codes):
    import torch

    capture = StageCapture(codec)
    watch_decode_stages(capture, codec.quantizer)
    try:
        indices = torch.as_tensor(codes, dtype=torch.long).unsqueeze(0)
        waveform = codec.decode(indices)
    finally:
        capture.release()
    semantic, residual = quantizer_parts(codec.quantizer, codes)
    return {
        "codec_codes.npy": np.asarray(codes, dtype=np.int32),
        "codec_semantic.npy": semantic,
        "codec_residual.npy": residual,
        "codec_post.npy": capture.outputs["post"],
        "codec_latent.npy": capture.outputs["latent"],
        "codec_wav.npy": to_numpy(waveform[0, 0]),
    }


def encode_reference(codec, audio):
    import torch

    capture = StageCapture(codec)
    watch_encode_stages(capture, codec)
    try:
        values = torch.as_tensor(audio, dtype=torch.float32).reshape(1, 1, -1)
        codes, lengths = codec.encode(values)
    finally:
        capture.release()
    valid = int(lengths[0])
    return {
        "codec_audio.npy": np.asarray(audio, dtype=np.float32),
        "codec_encoded.npy": capture.outputs["encoded"],
        "codec_downsampled.npy": capture.outputs["downsampled"],
        "codec_enc_codes.npy": to_numpy(codes[0, :, :valid]).astype(np.int32),
    }


def read_audio(path, sample_rate):
    import soundfile as sf

    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if source_rate != sample_rate:
        raise ValueError(
            f"{path} is {source_rate} Hz; resample it to {sample_rate} Hz first"
        )
    return mono


def generate_codes(model_dir, text, max_new_tokens):
    import torch

    from audio8_reference import load_generation_model

    torch.manual_seed(0)
    processor, model = load_generation_model(model_dir)
    inputs = processor(text=text)
    codes = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return to_numpy(codes[0]).astype(np.int32)


def resolve_codes(args):
    if args.codes:
        return np.load(args.codes).astype(np.int32)
    return generate_codes(args.model_dir, args.text, args.max_new_tokens)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--codes", help="[10, T] int .npy; generated when omitted")
    parser.add_argument("--audio", help="mono reference WAV at the codec sample rate")
    parser.add_argument("--text", default="Audio eight speaks on the CPU.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out-dir", default="artifacts/audio8-ref")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.model_dir)
    sample_rate = int(config["codec_sample_rate"])
    codec = load_codec_module(args.model_dir)

    codes = resolve_codes(args)
    print(f"decoding {codes.shape[1]} frames")
    arrays = decode_reference(codec, codes)

    meta = {
        "sample_rate": sample_rate,
        "frame_size": int(config["codec_frame_size"]),
        "frames": int(codes.shape[1]),
        "decoded_samples": int(arrays["codec_wav.npy"].shape[0]),
    }
    if args.audio:
        audio = read_audio(args.audio, sample_rate)
        print(f"encoding {audio.shape[0]} samples")
        arrays.update(encode_reference(codec, audio))
        meta["encoded_samples"] = int(audio.shape[0])
        meta["encoded_frames"] = int(arrays["codec_enc_codes.npy"].shape[1])

    save_arrays(args.out_dir, arrays)
    write_meta(args.out_dir, meta, name="codec_meta.json")


if __name__ == "__main__":
    main()
