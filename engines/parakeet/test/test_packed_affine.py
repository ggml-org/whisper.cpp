#!/usr/bin/env python3
"""Carrier transport must be exact, including FP32 affine metadata."""
import sys
import importlib.util
import types
from unittest.mock import patch
import unittest
from pathlib import Path
try:
    import torch
    import numpy as np
    import gguf
except ImportError:
    print('SKIP: torch, numpy and gguf are required')
    raise SystemExit(0)
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from packed_affine import encode, decode, carrier


class PackedAffineTests(unittest.TestCase):
    def test_exact_codes_and_effective_weights(self):
        for bits in (4, 8):
            with self.subTest(bits=bits):
                source = torch.randn(19, 96, generator=torch.Generator().manual_seed(17)).half()
                source[0] = 0
                source[1] = -0.375
                source[2] = 0.375
                record = encode(source, bits)
                data, dtype = carrier(record)
                codes = gguf.quants.dequantize(data, dtype) + (8 if bits == 4 else 128)
                self.assertTrue(np.array_equal(codes, record['codes']))
                expected = decode(record)
                scaled = codes * record['step'][:, None]
                actual = (scaled + record['minimum'][:, None]).astype(np.float16)
                self.assertTrue(np.array_equal(expected.view(np.uint16), actual.view(np.uint16)))
                self.assertEqual(record['minimum'].dtype, np.float32)
                self.assertEqual(record['step'].dtype, np.float32)

    def test_reference_loader_refuses_carriers_as_real_weights(self):
        script = Path(__file__).parents[1] / 'scripts/ref-encoder-from-gguf.py'
        spec = importlib.util.spec_from_file_location('carrier_reference', script)
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        fake = types.SimpleNamespace(fields={'parakeet.affine.version': None})
        with patch.object(gguf, 'GGUFReader', return_value=fake):
            with self.assertRaisesRegex(ValueError, 'dense-control'):
                ref.load_gguf('not-read.gguf')

    def test_rejects_nonfinite_and_unaligned_sources(self):
        with self.assertRaises(ValueError):
            encode(torch.full((3, 32), float('nan')), 4)
        with self.assertRaises(ValueError):
            carrier(encode(torch.ones(3, 33), 4))


if __name__ == '__main__':
    unittest.main()
