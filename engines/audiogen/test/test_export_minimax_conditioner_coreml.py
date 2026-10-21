#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import torch
    from gguf import GGUFWriter
    HAVE_DEPENDENCIES = True
except ImportError:
    HAVE_DEPENDENCIES = False


def load_exporter():
    path = Path(__file__).resolve().parents[1] / "scripts/export-minimax-conditioner-coreml.py"
    spec = importlib.util.spec_from_file_location("minimax_conditioner_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_projection(path, weight, bias):
    writer = GGUFWriter(str(path), "mm3-synth")
    writer.add_tensor("cond.proj.weight", weight)
    writer.add_tensor("cond.proj.bias", bias)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@unittest.skipUnless(HAVE_DEPENDENCIES, "requires torch, numpy and gguf")
class ConditionerExporterTests(unittest.TestCase):
    def test_original_convolution_layout_and_bias(self):
        exporter = load_exporter()
        weight = (np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 32).astype(np.float16)
        bias = np.array([0.25, -0.5], dtype=np.float32)
        x = np.arange(21, dtype=np.float32).reshape(1, 3, 1, 7) / 16
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synth.gguf"
            write_projection(path, weight, bias)
            with torch.no_grad():
                actual = exporter.load_projection(path)(torch.from_numpy(x)).numpy()
        expected = np.zeros((1, 2, 1, 7), dtype=np.float32)
        for d in range(2):
            for t in range(7):
                expected[0, d, 0, t] = bias[d]
                for c in range(3):
                    for k in range(3):
                        if 0 <= t + k - 1 < 7:
                            expected[0, d, 0, t] += weight[d, c, k] * x[0, c, 0, t + k - 1]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_rejects_incompatible_kernel_bias_and_precision(self):
        exporter = load_exporter()
        cases = [
            (np.ones((2, 3, 1), dtype=np.float16), np.zeros(2, dtype=np.float32)),
            (np.ones((2, 3, 3), dtype=np.float16), np.zeros(3, dtype=np.float32)),
            (np.ones((2, 3, 3), dtype=np.float32), np.zeros(2, dtype=np.float32)),
        ]
        for weight, bias in cases:
            with self.subTest(shape=weight.shape, dtype=weight.dtype, bias=bias.shape):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "synth.gguf"
                    write_projection(path, weight, bias)
                    with self.assertRaises(ValueError):
                        exporter.load_projection(path)


if __name__ == "__main__":
    unittest.main()
