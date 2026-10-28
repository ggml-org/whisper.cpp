import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    import numpy as np
    import torch
    from lavasr_coreml_models import Enhancer, LogMagnitude, gru
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def tensor(rng, *shape):
    return (rng.standard_normal(shape) * 0.2).astype(np.float32)


def tiny_enhancer():
    rng = np.random.default_rng(29)
    width, hidden, bins, mels, blocks, kernel = 4, 8, 5, 3, 2, 3
    values = {
        'enhancer.embed.weight': tensor(rng, width, mels, kernel),
        'enhancer.embed.bias': tensor(rng, width),
        'spec_head.out.weight': tensor(rng, 2 * bins, width),
        'spec_head.out.bias': tensor(rng, 2 * bins),
    }
    for prefix in ['enhancer.norm', 'enhancer.final_norm']:
        values[prefix + '.weight'] = 1 + tensor(rng, width)
        values[prefix + '.bias'] = tensor(rng, width)
    for i in range(blocks):
        prefix = f'enhancer.block.{i}'
        values.update({
            prefix + '.dwconv.weight': tensor(rng, width, 1, kernel),
            prefix + '.dwconv.bias': tensor(rng, width),
            prefix + '.norm.weight': 1 + tensor(rng, width),
            prefix + '.norm.bias': tensor(rng, width),
            prefix + '.pwconv1.weight': tensor(rng, hidden, width),
            prefix + '.pwconv1.bias': tensor(rng, hidden),
            prefix + '.pwconv2.weight': tensor(rng, width, hidden),
            prefix + '.pwconv2.bias': tensor(rng, width),
            prefix + '.gamma': tensor(rng, width),
        })
    metadata = {'n_mels': mels, 'n_blocks': blocks, 'kernel': kernel,
                'layernorm_eps': 1e-6, 'clip_max': 1000.0}
    return Enhancer(values, metadata).eval()


def windowed(model, mel, capacity):
    frames = mel.shape[-1]
    results = [[], []]
    offset = 0
    while offset < frames:
        start = min(max(0, offset - model.context), max(0, frames - capacity))
        filled = min(capacity, frames - start)
        right_context = 0 if start + filled == frames else model.context
        count = start + filled - right_context - offset
        x = torch.zeros((1, mel.shape[1], capacity))
        valid = torch.zeros((1, 1, capacity))
        x[..., :filled] = mel[..., start:start + filled]
        valid[..., :filled] = 1
        for target, output in zip(results, model(x, valid)):
            target.append(output[..., offset - start:offset - start + count])
        offset += count
    return tuple(torch.cat(chunks, dim=-1) for chunks in results)


@unittest.skipUnless(HAVE_TORCH, 'numpy and torch are required')
class EnhancerTests(unittest.TestCase):
    def test_window_boundaries_match_full_network(self):
        model = tiny_enhancer()
        capacity = 16
        torch.manual_seed(71)
        for frames in [1, 2, capacity - 1, capacity, capacity + 1, 3 * capacity + 3]:
            with self.subTest(frames=frames), torch.inference_mode():
                mel = torch.randn((1, 3, frames))
                expected = model(mel, torch.ones((1, 1, frames)))
                actual = windowed(model, mel, capacity)
                for output, reference in zip(actual, expected):
                    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)

    def test_padded_frames_are_zero_in_both_output_planes(self):
        model = tiny_enhancer()
        mel = torch.randn((1, 3, 16))
        valid = torch.zeros((1, 1, 16))
        valid[..., :5] = 1
        with torch.inference_mode():
            for output in model(mel, valid):
                self.assertEqual(torch.count_nonzero(output[..., 5:]).item(), 0)

    def test_clip_that_fits_a_window_predicts_once(self):
        from unittest.mock import Mock

        model = tiny_enhancer()
        predict = Mock(wraps=model)
        predict.context = model.context
        with torch.inference_mode():
            windowed(predict, torch.randn((1, 3, 16)), 16)
        self.assertEqual(predict.call_count, 1)


@unittest.skipUnless(HAVE_TORCH, 'numpy and torch are required')
class GRUTests(unittest.TestCase):
    def test_gguf_gate_order_and_zero_state_match_pytorch(self):
        rng = np.random.default_rng(6)
        reference = torch.nn.GRU(3, 4, batch_first=True, bidirectional=True).eval()
        tensors = {'rnn.' + name: tensor(rng, *value.shape)
                   for name, value in reference.named_parameters()}
        reference.load_state_dict({name: torch.from_numpy(tensors['rnn.' + name])
                                   for name, _ in reference.named_parameters()})
        rebuilt = gru(tensors, 'rnn', True).eval()
        sequence = torch.from_numpy(tensor(rng, 2, 7, 3))
        with torch.inference_mode():
            expected, _ = reference(sequence)
            actual, _ = rebuilt(sequence)
            repeated, _ = rebuilt(sequence)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(repeated, expected)


@unittest.skipUnless(HAVE_TORCH and sys.platform == 'darwin', 'Apple Core ML runtime is required')
class PrecisionTests(unittest.TestCase):
    def test_float16_export_preserves_log_magnitude_range(self):
        try:
            import coremltools as ct
        except ImportError:
            self.skipTest('coremltools is required')
        spec = importlib.util.spec_from_file_location('export_lavasr_coreml', SCRIPTS / 'export-lavasr-coreml.py')
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.log_magnitude = LogMagnitude()

            def forward(self, real, imag):
                return self.log_magnitude(real, imag)

        model = Probe().eval()
        shape = (1, 1, 3, 5)
        samples = (torch.ones(shape), torch.ones(shape))
        traced = torch.jit.trace(model, samples)
        converted = ct.convert(traced, convert_to='mlprogram',
            inputs=[ct.TensorType(name='real', shape=shape, dtype=np.float32),
                    ct.TensorType(name='imag', shape=shape, dtype=np.float32)],
            outputs=[ct.TensorType(name='log_magnitude', dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=exporter.compute_precision('float16'),
            compute_units=ct.ComputeUnit.CPU_ONLY)
        for value in (0.0, 1e-9, 256.0, 512.0):
            with self.subTest(value=value), torch.inference_mode():
                real = np.full(shape, value, np.float32)
                imag = np.zeros(shape, np.float32)
                expected = model(torch.from_numpy(real), torch.from_numpy(imag)).numpy()
                actual = converted.predict({'real': real, 'imag': imag})['log_magnitude']
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    if not HAVE_TORCH:
        print('SKIP: numpy and torch are required')
    unittest.main()
