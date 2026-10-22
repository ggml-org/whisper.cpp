#!/usr/bin/env python3
"""Apple integration with generated models: no checkpoint or ggml build needed.

Requires Torch, coremltools, and Xcode command-line tools. FP32 conversions
check padding semantics; FP16/FP32 native I/O uses matched compiled-model and
compute-unit references, independently of FP16-versus-PyTorch numerical drift.
"""
import os
import platform
import shutil
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_coreml_padding import EXPORTER, fixture, torch
import numpy as np

try:
    import coremltools as ct
    HAVE_COREML = hasattr(ct, 'convert')
except ImportError:
    HAVE_COREML = False

# Exercise the first frame-count region that FP16 cannot represent exactly.
CAPACITY = 2057


def make_cases(root, compiled, mode, precision, units, ref, weights, meta, mels):
    dim = meta['parakeet.encoder.d_model']
    dtype = np.float32 if precision == 'fp32' else np.float16
    reference = ct.models.CompiledMLModel(str(compiled), compute_units=(
        ct.ComputeUnit.CPU_ONLY if units == 'cpu_only' else ct.ComputeUnit.ALL))
    module = (EXPORTER.BypassEncoderModule if mode == 'bypass' else
              EXPORTER.EncoderModule)(ref, weights, meta).eval()
    lengths = ((2, 5, 1, 4, 2) if mode == 'bypass' else
               (9, CAPACITY, 1, 8, 1023, 1024, 1025, 1501, 2048, 2049, 2056, 9))
    cases = []
    for index, valid in enumerate(lengths):
        features, capacity = (dim, 5) if mode == 'bypass' else (mels, CAPACITY)
        frames = capacity if mode == 'masked' and index % 2 == 0 else valid
        data = np.random.default_rng(17).standard_normal((features, frames)).astype(dtype)
        if frames > valid:
            data[:, valid:] = 50  # Invalid values must not leak inward.
        padded = np.zeros((features, capacity), dtype=dtype)
        padded[:, :frames] = data
        tensor = torch.from_numpy(padded.astype(np.float32))
        mask = (torch.arange(capacity) < valid).float()
        with torch.inference_mode():
            if mode == 'masked':
                rows = (valid + 7) // 8
                expected = module(tensor[:, :valid])  # Unpadded reference math.
            elif mode == 'legacy':
                rows = (frames + 7) // 8
                expected = module(tensor)[:rows]
            else:
                rows = valid
                expected = module(tensor, mask)[:rows]
        values = ({'pre_encode': padded, 'valid_mask': mask.numpy().astype(dtype)}
                  if mode == 'bypass' else {'mel': padded})
        if mode == 'masked':
            values['mel_valid_mask'] = mask.numpy().astype(dtype)
        full_result = np.asarray(reference.predict(values)['encoder_out'])
        actual = full_result[:rows]
        if mode == 'masked':
            # A rounded FP16 length (e.g. 2049 -> 2048) would erase this row.
            assert np.linalg.norm(actual[-1]) > 1e-3
            assert np.count_nonzero(full_result[rows:]) == 0
        if precision == 'fp32':
            np.testing.assert_allclose(actual, expected.numpy(), atol=1e-4, rtol=1e-3)
        source, target = root / f'{mode}-{index}.f32', root / f'{mode}-{index}-ref.f32'
        data.T.astype('<f4').tofile(source)
        actual.astype('<f4').tofile(target)
        cases.append(f'{source} {target} {frames} {valid} {features} {rows} {dim}')
    manifest = root / f'{mode}-{precision}-{units}.txt'
    manifest.write_text('\n'.join(cases) + '\n')
    return manifest


@unittest.skipUnless(platform.system() == 'Darwin' and HAVE_COREML,
                     'requires Apple Core ML runtime and coremltools')
class RuntimeTests(unittest.TestCase):
    def test_masked_legacy_and_bypass_io(self):
        torch.set_num_threads(1)
        ref, weights, meta, mels = fixture(dim=32, mels=16, channels=4)
        dim = meta['parakeet.encoder.d_model']
        engine = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory(prefix='parakeet-mask-test-') as tmp:
            root = Path(tmp)
            binary = root / 'coreml-padding-io'
            subprocess.run(['clang++', '-std=c++17', '-fobjc-arc',
                            '-I' + str(engine / 'src'),
                            str(engine / 'src/coreml/parakeet-encoder.mm'),
                            str(engine / 'src/coreml/parakeet_coreml_shape.cpp'),
                            str(engine / 'test/coreml_padding_io.cpp'),
                            '-framework', 'CoreML', '-framework', 'Foundation',
                            '-o', str(binary)], check=True)
            for precision in ('fp32', 'fp16'):
                dtype = np.float32 if precision == 'fp32' else np.float16
                compute = ct.precision.FLOAT32 if precision == 'fp32' else ct.precision.FLOAT16
                for mode in ('masked', 'legacy', 'bypass'):
                    with self.subTest(mode=mode, precision=precision):
                        if mode == 'masked':
                            model = EXPORTER.convert_fixed(
                                ref, weights, meta, torch.zeros(mels, CAPACITY), mels, CAPACITY, dim,
                                compute, dtype, ct.target.macOS13)
                        elif mode == 'legacy':
                            module = EXPORTER.EncoderModule(ref, weights, meta).eval()
                            traced = torch.jit.trace(module, torch.zeros(mels, CAPACITY), check_trace=False)
                            model = ct.convert(traced,
                                inputs=[ct.TensorType(name='mel', shape=(mels, CAPACITY), dtype=dtype)],
                                outputs=[ct.TensorType(name='encoder_out', dtype=dtype)],
                                compute_precision=compute,
                                minimum_deployment_target=ct.target.macOS13)
                        else:
                            model = EXPORTER.convert_bypass_fixed(
                                ref, weights, {**meta, 'parakeet.model.type': 'sortformer'},
                                torch.zeros(dim, 5), dim, 5, compute, dtype, ct.target.macOS13)
                        package = root / f'{mode}-{precision}.mlpackage'
                        model.save(str(package))
                        compiled = Path(ct.models.utils.compile_model(str(package)))
                        try:
                            for units in ('cpu_only', 'all'):
                                manifest = make_cases(root, compiled, mode, precision, units,
                                                      ref, weights, meta, mels)
                                env = {**os.environ, 'PARAKEET_COREML_COMPUTE_UNITS': units}
                                subprocess.run([str(binary), str(compiled), mode, str(manifest)],
                                               env=env, check=True, timeout=120)
                        finally:
                            shutil.rmtree(compiled)


if __name__ == '__main__':
    result = unittest.main(exit=False).result
    # CTest must not count an unavailable Apple runtime as executed coverage.
    # Failures take precedence if this suite grows additional cases later.
    if not result.wasSuccessful():
        sys.exit(1)
    sys.exit(77 if result.testsRun == len(result.skipped) else 0)
