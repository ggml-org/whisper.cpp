#!/usr/bin/env python3
"""Regressions for shared Compact GGUF preparation and Core ML weight loading."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import gguf
    import numpy as np
    import torch
except ImportError:
    print('SKIP: Compact sidecar tests require gguf, numpy and torch')
    raise SystemExit(0)

ENGINE = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ENGINE / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load('prepare_mosaic', 'prepare-mosaic.py')
REFERENCE = load('mosaic_reference', 'ref-encoder-from-gguf.py')


class CompactSidecarTests(unittest.TestCase):
    def test_q3_k_and_iq3_s_encoder_weights_load_without_requantization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'compact.gguf'
            writer = gguf.GGUFWriter(str(path), 'parakeet-ctc')
            expected = {}
            for kind in (gguf.GGMLQuantizationType.Q3_K, gguf.GGMLQuantizationType.IQ3_S):
                _, size = gguf.GGML_QUANT_SIZES[kind]
                raw = np.random.default_rng(17).integers(0, 256, (2, size), dtype=np.uint8)
                # Q3_K stores d at the end of its block; IQ3_S at the start.
                scale = slice(-2, None) if kind == gguf.GGMLQuantizationType.Q3_K else slice(0, 2)
                raw[:, scale] = np.frombuffer(np.float16(1).tobytes(), dtype=np.uint8)
                name = f'encoder.{kind.name}.weight'
                expected[name] = gguf.quants.dequantize(raw, kind).astype(np.float32)
                writer.add_tensor(name, raw, raw_dtype=kind)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            actual, _ = REFERENCE.load_gguf(path)
            for name, values in expected.items():
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue(np.any(values != 0))
                self.assertEqual(actual[name].dtype, torch.float32)
                np.testing.assert_array_equal(actual[name].numpy(), values)

    def test_native_preparation_keeps_variant_identity_and_names(self):
        for variant, stem in [('mosaic', 'parakeet-mosaic'), ('compact', 'parakeet-mosaic-compact')]:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, output = root / 'source.gguf', root / 'bundle'
                source.write_bytes(b'unchanged test payload')
                identity = {'bytes': source.stat().st_size,
                            'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
                with patch.object(sys, 'argv', ['prepare-mosaic', '--gguf', str(source),
                        '--variant', variant, '--output-dir', str(output)]), \
                        patch.object(PREPARE.json, 'loads', return_value=identity):
                    PREPARE.main()
                self.assertEqual((output / f'{stem}.gguf').read_bytes(), source.read_bytes())
                complete = json.loads((output / 'complete.json').read_text())
                self.assertEqual(complete['gguf_sha256'], identity['sha256'])
                self.assertFalse(complete['sidecar_generated'])
                self.assertEqual(json.loads((output / 'plan.json').read_text())['name'], stem)

    def test_wrong_artifact_and_archive_compact_requests_leave_no_output(self):
        for source_flag in ('--gguf', '--archive'):
            with self.subTest(source=source_flag), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, output = root / 'source', root / 'bundle'
                source.write_bytes(b'not the pinned Compact model')
                with patch.object(sys, 'argv', ['prepare-mosaic', source_flag, str(source),
                        '--variant', 'compact', '--output-dir', str(output)]):
                    with self.assertRaises(SystemExit) as error:
                        PREPARE.main()
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
