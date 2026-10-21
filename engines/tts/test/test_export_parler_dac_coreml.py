#!/usr/bin/env python3
"""Model-free unit tests for scripts/parler/export-dac-coreml.py.

A tiny DAC decoder is built from random tensors in the Parler GGUF's names and
layouts (nonzero biases, Snake alphas like the checkpoint's) and checked
against an independent plain-PyTorch decoder written from the definitions:
"same"-padded Conv1d, ConvTranspose1d trimmed by ceil(stride / 2), and
x + sin(alpha x)^2 / alpha. The rebuild must match it flat and folded, and a
masked fixed window must equal the unpadded decode of the shorter sequence --
the property that lets the engine serve every length and the last window with
one fixed-shape graph. Requires torch; skips without it.
"""
import importlib.util
import pathlib
import unittest

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "parler" / "export-dac-coreml.py"

LATENT = 8
DECODER_DIM = 16
RATES = [2, 4]
HOP = 8
N_Q = 3
CODEBOOK = 24
CB_DIM = 4
DILATIONS = [1, 3, 9]


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_parler_dac_coreml", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rand(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32) * 0.3


def kernel(rng, out, inp, taps):
    """A conv kernel scaled by its fan-in, so activations stay O(1) through the
    stack (as a trained decoder's do) and fp32 rounding stays far below atol."""
    return (rng.standard_normal((out, inp, taps)) / np.sqrt(inp * taps)).astype(np.float32)


def alpha(rng, channels):
    return rng.uniform(0.15, 2.3, size=(1, channels, 1)).astype(np.float32)


def tiny_tensors(rng):
    """numpy arrays in the GGUF's names, torch layouts (gguf-py reverses ggml ne):
    Conv1d [out, in, k], ConvTranspose1d [in, out, 2 * stride]."""
    t = {}
    for k in range(N_Q):
        t[f"dac.quant.{k}.codebook.weight"] = rand(rng, CODEBOOK, CB_DIM)
        t[f"dac.quant.{k}.out_proj.weight"] = rand(rng, LATENT, CB_DIM, 1)
        t[f"dac.quant.{k}.out_proj.bias"] = rand(rng, LATENT)
    t["dac.dec.conv_in.weight"] = kernel(rng, DECODER_DIM, LATENT, 7)
    t["dac.dec.conv_in.bias"] = rand(rng, DECODER_DIM)
    channels = DECODER_DIM
    for b, stride in enumerate(RATES):
        out = channels // 2
        pre = f"dac.dec.blk.{b}."
        t[pre + "snake.alpha"] = alpha(rng, channels)
        t[pre + "convt.weight"] = kernel(rng, channels, out, 2 * stride) * np.float32(np.sqrt(stride))
        t[pre + "convt.bias"] = rand(rng, out)
        for j in range(3):
            unit = f"{pre}res.{j}."
            t[unit + "snake1.alpha"] = alpha(rng, out)
            t[unit + "conv1.weight"] = kernel(rng, out, out, 7)
            t[unit + "conv1.bias"] = rand(rng, out)
            t[unit + "snake2.alpha"] = alpha(rng, out)
            t[unit + "conv2.weight"] = kernel(rng, out, out, 1)
            t[unit + "conv2.bias"] = rand(rng, out)
        channels = out
    t["dac.dec.snake_out.alpha"] = alpha(rng, channels)
    t["dac.dec.conv_out.weight"] = kernel(rng, 1, channels, 7)
    t["dac.dec.conv_out.bias"] = rand(rng, 1)
    return t


def tiny_hparams():
    return {"hop": HOP, "latent_dim": LATENT, "decoder_dim": DECODER_DIM, "rates": RATES,
            "n_q": N_Q, "codebook_size": CODEBOOK, "sample_rate": 44100}


def reference_decode(t, latent):
    """The DAC decoder from its definition, [1, LATENT, n] -> [1, 1, n * HOP]."""
    p = {k: torch.from_numpy(v) for k, v in t.items()}

    def snake(x, a):
        return x + (1.0 / (a + 1e-9)) * torch.sin(a * x) ** 2

    def conv(x, name, dilation=1):
        w = p[name + ".weight"]
        return F.conv1d(x, w, p[name + ".bias"], padding=(w.shape[-1] - 1) * dilation // 2, dilation=dilation)

    x = conv(latent, "dac.dec.conv_in")
    for b, stride in enumerate(RATES):
        pre = f"dac.dec.blk.{b}."
        x = snake(x, p[pre + "snake.alpha"])
        x = F.conv_transpose1d(x, p[pre + "convt.weight"], p[pre + "convt.bias"], stride=stride,
                               padding=(stride + 1) // 2)
        for j, d in enumerate(DILATIONS):
            unit = f"{pre}res.{j}."
            y = conv(snake(x, p[unit + "snake1.alpha"]), unit + "conv1", d)
            x = x + conv(snake(y, p[unit + "snake2.alpha"]), unit + "conv2")
    return torch.tanh(conv(snake(x, p["dac.dec.snake_out.alpha"]), "dac.dec.conv_out"))


@unittest.skipUnless(HAVE_TORCH, "SKIP: torch not installed")
class DacRebuildTest(unittest.TestCase):
    def setUp(self):
        self.exp = load_exporter()
        self.rng = np.random.default_rng(7)
        self.t = tiny_tensors(self.rng)
        self.hp = tiny_hparams()
        self.saved = (self.exp.FOLD_ROW, self.exp.Snake.mode)

    def tearDown(self):
        self.exp.FOLD_ROW, self.exp.Snake.mode = self.saved

    def decode(self, latent, window, valid_frames=None, fold_row=0, mode="sin"):
        self.exp.FOLD_ROW = fold_row
        self.exp.Snake.mode = mode
        model = self.exp.DacDecoder(self.t, self.hp, window).eval()
        lat = np.zeros((1, LATENT, window), dtype=np.float32)
        lat[0, :, :latent.shape[1]] = latent
        valid = np.zeros((1, 1, window), dtype=np.float32)
        valid[0, 0, :latent.shape[1] if valid_frames is None else valid_frames] = 1.0
        with torch.no_grad():
            return model(torch.from_numpy(lat), torch.from_numpy(valid)).numpy().reshape(-1)

    def reference(self, latent):
        with torch.no_grad():
            return reference_decode(self.t, torch.from_numpy(latent[None])).numpy().reshape(-1)

    def test_flat_rebuild_matches_definition(self):
        latent = rand(self.rng, LATENT, 32)
        np.testing.assert_allclose(self.decode(latent, 32), self.reference(latent), atol=5e-5)

    def test_fold_is_exact(self):
        # 32 frames -> 64 and 256 samples after the two blocks; rows of 32
        # samples put most of a dilation-9 halo (27 samples) in the next row
        latent = rand(self.rng, LATENT, 32)
        flat = self.decode(latent, 32)
        for fold_row in (32, 64):
            with self.subTest(fold_row=fold_row):
                np.testing.assert_allclose(self.decode(latent, 32, fold_row=fold_row), flat, atol=5e-5)

    def test_masked_window_equals_shorter_decode(self):
        # the frames past `n` hold garbage the mask must hide, folded or not
        for n in (1, 5, 17, 31):
            for fold_row in (0, 32):
                with self.subTest(n=n, fold_row=fold_row):
                    latent = rand(self.rng, LATENT, 32)
                    got = self.decode(latent, 32, valid_frames=n, fold_row=fold_row)[:n * HOP]
                    np.testing.assert_allclose(got, self.reference(latent[:, :n]), atol=5e-5)

    def test_windowed_walk_equals_whole_decode(self):
        rf = self.exp.receptive_field_frames(self.t, self.hp)
        window = 2 * rf + 8
        self.exp.FOLD_ROW = 0
        self.exp.Snake.mode = "sin"
        model = self.exp.DacDecoder(self.t, self.hp, window).eval()
        for frames in (window - 3, 3 * window + 5):
            with self.subTest(frames=frames):
                latent = rand(self.rng, LATENT, frames)
                got = self.exp.decode_windowed(self.exp.torch_runner(model), latent, window, rf, HOP)
                np.testing.assert_allclose(got, self.reference(latent), atol=5e-5)

    def test_receptive_field_bounds_influence(self):
        # perturbing one frame leaves every output more than rf frames away unchanged
        rf = self.exp.receptive_field_frames(self.t, self.hp)
        frames, hit = 4 * rf, 2 * rf
        latent = rand(self.rng, LATENT, frames)
        moved = latent.copy()
        moved[:, hit] += 5.0
        diff = np.abs(self.reference(moved) - self.reference(latent)).reshape(frames, HOP).max(axis=1)
        self.assertGreater(diff[hit], 0.0)
        self.assertEqual(float(diff[:hit - rf].max()), 0.0)
        self.assertEqual(float(diff[hit + rf + 1:].max()), 0.0)

    def test_fold_rows_narrower_than_a_halo_are_rejected(self):
        self.exp.FOLD_ROW = 16
        with self.assertRaises(ValueError):
            self.exp.DacDecoder(self.t, self.hp, 32)

    def test_windows_that_cannot_fold_are_rejected(self):
        # stages of W, 2W and 8W samples against rows of 4096
        self.exp.FOLD_ROW = 4096
        with self.assertRaises(ValueError):
            self.exp.DacDecoder(self.t, self.hp, 3000)   # 24000 samples: > 16384, not a row multiple
        self.exp.DacDecoder(self.t, self.hp, 1000)       # every stage <= 16384
        self.exp.DacDecoder(self.t, self.hp, 4096)       # every long stage folds

    def test_polynomial_snake_tracks_sine(self):
        latent = rand(self.rng, LATENT, 32)
        np.testing.assert_allclose(self.decode(latent, 32, mode="poly"), self.reference(latent), atol=2e-3)

    def test_rvq_latent(self):
        codes = self.rng.integers(0, CODEBOOK, size=(N_Q, 9))
        want = np.zeros((LATENT, 9), dtype=np.float32)
        for k in range(N_Q):
            vectors = self.t[f"dac.quant.{k}.codebook.weight"][codes[k]].T
            want += self.t[f"dac.quant.{k}.out_proj.weight"][:, :, 0] @ vectors
            want += self.t[f"dac.quant.{k}.out_proj.bias"][:, None]
        np.testing.assert_allclose(self.exp.rvq_latent(self.t, self.hp, codes), want, atol=1e-5)


@unittest.skipUnless(HAVE_TORCH, "SKIP: torch not installed")
class SidecarStemTest(unittest.TestCase):
    def test_stem_drops_quant_tag(self):
        exp = load_exporter()
        self.assertEqual(exp.sidecar_stem("models/parler-mini-v1-q8_0.gguf"), "parler-mini-v1-dac")
        self.assertEqual(exp.sidecar_stem("parler-indic-f32.gguf"), "parler-indic-dac")
        self.assertEqual(exp.sidecar_stem("parler-mini-v1.gguf"), "parler-mini-v1-dac")


if __name__ == "__main__":
    unittest.main()
