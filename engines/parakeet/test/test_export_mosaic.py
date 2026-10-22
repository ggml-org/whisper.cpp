#!/usr/bin/env python3
"""Check decoder-only export, metadata preservation and failure guards."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

try:
    import gguf
    import numpy as np
except ImportError:
    print('SKIP: Mosaic exporter tests require numpy and gguf')
    raise SystemExit(0)

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('mosaic_export', SCRIPTS / 'export-mosaic.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def scalar_q8(values):
    """Independent scalar rounding, including zero blocks and signed half ties."""
    out = bytearray()
    for block in np.asarray(values, dtype=np.float32).reshape(-1, 32):
        scale = np.float32(max(abs(float(x)) for x in block) / 127)
        inverse = np.float32(0 if scale == 0 else 1 / scale)
        out += struct.pack('<e', float(scale))
        for x in block:
            value = float(np.float32(x * inverse))
            rounded = int(np.copysign(np.floor(abs(value) + .5), value))
            out += struct.pack('b', rounded)
    return np.frombuffer(out, dtype=np.uint8).reshape(values.shape[0], -1).copy()


class MosaicExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.gguf'
        self.output = self.root / 'mosaic.gguf'
        self.recipe = json.loads((SCRIPTS.parent / 'configs/parakeet-mosaic.decoder.recipe.json').read_text())
        self.targets = self.recipe['targets']
        self.arrays = {}
        for i, name in enumerate(self.targets):
            a = np.linspace(-2-i, 2+i, 64, dtype=np.float32).astype(np.float16).reshape(2, 32)
            if i == 0:
                a[0] = 0
            self.arrays[name] = a
        self._write(self.source, converted=False)
        expected = self.root / 'expected.gguf'
        self._write(expected, converted=True)
        self.recipe['source'] = {'bytes': self.source.stat().st_size, 'sha256': exporter.sha(self.source), 'tensors': 653}
        self.recipe['output'] = {'bytes': expected.stat().st_size, 'sha256': exporter.sha(expected)}

    def _write(self, path, converted):
        w = gguf.GGUFWriter(str(path), 'parakeet-ctc')
        w.add_name('preserved encoder provenance')
        w.add_string('mosaic.recipe', 'group32_attention8')
        w.add_array('test.metadata', [1, 7, 17])
        for name, a in self.arrays.items():
            if converted:
                w.add_tensor(name, scalar_q8(a), raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            else:
                w.add_tensor(name, a)
        for i in range(645):
            w.add_tensor(f'encoder.unchanged.{i}', np.array([i / 1000], dtype=np.float32))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    def test_exact_decoder_blocks_and_unchanged_payloads(self):
        receipt = exporter.export_decoder(self.source, self.output, self.recipe)
        before, after = gguf.GGUFReader(str(self.source)), gguf.GGUFReader(str(self.output))
        old = {t.name: t for t in before.tensors}
        for t in after.tensors:
            if t.name in self.targets:
                self.assertEqual(t.tensor_type, gguf.GGMLQuantizationType.Q8_0)
                np.testing.assert_array_equal(t.data, scalar_q8(self.arrays[t.name]))
            else:
                self.assertEqual(t.tensor_type, old[t.name].tensor_type)
                self.assertEqual(t.data.tobytes(), old[t.name].data.tobytes())
        for name, field in before.fields.items():
            if not name.startswith('GGUF.'):
                self.assertEqual(field.contents(), after.get_field(name).contents())
        self.assertEqual(receipt['unchanged_payloads'], 645)
        self.assertEqual(receipt['version'], '1')
        self.assertFalse(receipt['requires_affine_fused_kernel'])

    def test_wrong_source_is_rejected_without_output(self):
        self.recipe['source']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            exporter.export_decoder(self.source, self.output, self.recipe)
        self.assertFalse(self.output.exists())

    def test_existing_output_and_receipt_are_preserved(self):
        for path in [self.output, self.output.with_suffix('.receipt.json')]:
            path.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                exporter.export_decoder(self.source, self.output, self.recipe)
            self.assertEqual(path.read_bytes(), b'keep')
            path.unlink()

    def test_wrong_output_identity_removes_partial_artifact(self):
        self.recipe['output']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            exporter.export_decoder(self.source, self.output, self.recipe)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix('.receipt.json').exists())

    def test_wrong_decoder_precision_is_rejected(self):
        self.arrays[self.targets[0]] = self.arrays[self.targets[0]].astype(np.float32)
        self._write(self.source, converted=False)
        self.recipe['source'].update(bytes=self.source.stat().st_size, sha256=exporter.sha(self.source))
        with self.assertRaisesRegex(ValueError, 'decoder source type/shape'):
            exporter.export_decoder(self.source, self.output, self.recipe)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
