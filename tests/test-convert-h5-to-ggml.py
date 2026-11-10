"""Run with python tests/test-convert-h5-to-ggml.py (requires torch and numpy)."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


CONVERTER = Path(__file__).resolve().parents[1] / "models/convert-h5-to-ggml.py"

# Source name, output name, input shape, output shape, default storage type.
TENSORS = (
    ("model.encoder.conv1.weight", "encoder.conv1.weight", (2, 2, 3), (2, 2, 3), 1),
    ("model.encoder.conv1.bias", "encoder.conv1.bias", (4,), (4, 1), 0),
    ("model.encoder.conv2.bias", "encoder.conv2.bias", (4,), (4, 1), 0),
    ("model.encoder.layer_norm.weight", "encoder.ln_post.weight", (4,), (4,), 0),
    ("model.encoder.embed_positions.weight", "encoder.positional_embedding", (2, 2), (2, 2), 0),
    ("model.decoder.embed_positions.weight", "decoder.positional_embedding", (2, 2), (2, 2), 0),
    ("model.decoder.layers.0.fc1.weight", "decoder.blocks.0.mlp.0.weight", (2, 2), (2, 2), 1),
)


def make_state(dtypes, values):
    state = {}
    for i, (source, _, shape, _, _) in enumerate(TENSORS):
        tensor = torch.tensor(values, dtype=dtypes[i % len(dtypes)])
        state[source] = tensor.repeat(int(np.prod(shape)) // len(values)).reshape(shape)
    state["proj_out.weight"] = torch.ones((2, 2), dtype=dtypes[0])
    return state


def convert(state, use_f32):
    # Only model loading is replaced; execute the actual converter and writer.
    model = types.SimpleNamespace(state_dict=lambda: state)
    transformers = types.ModuleType("transformers")
    transformers.WhisperForConditionalGeneration = types.SimpleNamespace(from_pretrained=lambda _: model)
    with tempfile.TemporaryDirectory(prefix="whisper-convert-test-") as tmp:
        root = Path(tmp)
        (root / "whisper/assets").mkdir(parents=True)
        config = dict(vocab_size=1, max_source_positions=2, max_length=2, d_model=4,
                      encoder_attention_heads=1, encoder_layers=1,
                      decoder_attention_heads=1, decoder_layers=1, num_mel_bins=2)
        for name, data in (("vocab.json", {"a": 0}), ("added_tokens.json", {}), ("config.json", config)):
            (root / name).write_text(json.dumps(data), encoding="utf-8")
        np.savez(root / "whisper/assets/mel_filters.npz", mel_2=np.ones((2, 2), dtype=np.float32))
        argv = [str(CONVERTER), str(root), str(root), str(root)] + (["use-f32"] if use_f32 else [])
        # ExitStack closes output handles even when testing an unfixed converter.
        with contextlib.ExitStack() as stack:
            original_open = open

            def tracked_open(*args, **kwargs):
                return stack.enter_context(original_open(*args, **kwargs))

            with patch.dict(sys.modules, {"transformers": transformers}), patch.object(sys, "argv", argv), \
                    patch("builtins.open", tracked_open), patch("io.open", tracked_open), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    np.errstate(over="ignore", under="ignore"):
                runpy.run_path(str(CONVERTER), run_name="__main__")
        return (root / ("ggml-model-f32.bin" if use_f32 else "ggml-model.bin")).read_bytes()


def read_model(raw):
    stream = io.BytesIO(raw)

    def ints(count):
        return struct.unpack("i" * count, stream.read(4 * count))

    header = ints(12)
    rows, cols = ints(2)
    filters = np.frombuffer(stream.read(rows * cols * 4), dtype=np.float32)
    tokens = [stream.read(ints(1)[0]) for _ in range(ints(1)[0])]
    tensors = {}
    while stream.tell() < len(raw):
        ndim, length, ftype = ints(3)
        shape = ints(ndim)[::-1]
        name = stream.read(length).decode("utf-8")
        dtype = {0: np.float32, 1: np.float16}[ftype]
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        tensors[name] = (ftype, np.frombuffer(stream.read(size), dtype=dtype).reshape(shape))
    return header, filters, tokens, tensors


class TestConvertH5(unittest.TestCase):
    def check_conversion(self, state, use_f32):
        raw = convert(state, use_f32)
        header, filters, tokens, tensors = read_model(raw)
        self.assertEqual(header, (0x67676d6c, 1, 2, 4, 1, 1, 2, 4, 1, 1, 2, int(not use_f32)))
        np.testing.assert_array_equal(filters, np.ones(4, dtype=np.float32))
        self.assertEqual(tokens, [b"a"])
        self.assertEqual(set(tensors), {row[1] for row in TENSORS})
        for source, name, _, shape, default_type in TENSORS:
            expected_type = 0 if use_f32 else default_type
            ftype, data = tensors[name]
            self.assertEqual(ftype, expected_type, name)
            self.assertEqual(data.shape, shape, name)
            dtype = np.float32 if expected_type == 0 else np.float16
            with np.errstate(over="ignore", under="ignore"):
                expected = state[source].float().numpy().reshape(shape).astype(dtype)
            np.testing.assert_array_equal(data, expected, err_msg=name)
        return raw

    def test_input_and_output_types(self):
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            for use_f32 in (False, True):
                with self.subTest(dtype=dtype, use_f32=use_f32):
                    self.check_conversion(make_state([dtype], [1.0001, -2.125, 0.25, 0]), use_f32)

    def test_values_outside_fp16_range(self):
        for dtype in (torch.float32, torch.bfloat16):
            for use_f32 in (False, True):
                with self.subTest(dtype=dtype, use_f32=use_f32):
                    self.check_conversion(make_state([dtype], [65536, -65536, 1e-8, -1e-8]), use_f32)

    def test_mixed_types(self):
        state = make_state([torch.bfloat16, torch.float16, torch.float32], [1, -2, 0.25, 0])
        for use_f32 in (False, True):
            with self.subTest(use_f32=use_f32):
                self.check_conversion(state, use_f32)

    def test_storage_does_not_depend_on_input_dtype(self):
        for use_f32 in (False, True):
            with self.subTest(use_f32=use_f32):
                outputs = [self.check_conversion(make_state([dtype], [1, -2, 0.25, 0]), use_f32)
                           for dtype in (torch.float16, torch.float32, torch.bfloat16)]
                self.assertEqual(outputs[0], outputs[1])
                self.assertEqual(outputs[0], outputs[2])


if __name__ == "__main__":
    unittest.main()
