#!/usr/bin/env python3
"""Small, deterministic full-encoder regressions; no checkpoint required."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_export_encoder_coreml import EXPORTER, torch


def fixture(dim=4, mels=8, channels=2):
    path = Path(__file__).parents[1] / 'scripts' / 'ref-encoder-from-gguf.py'
    spec = importlib.util.spec_from_file_location('padding_reference', path)
    ref = importlib.util.module_from_spec(spec)
    # GGUF is used only by the checkpoint loader, not the reference math.
    with patch.dict(sys.modules, {'gguf': types.ModuleType('gguf')}):
        spec.loader.exec_module(ref)
    gen = torch.Generator().manual_seed(17)
    weights = {}

    def rand(name, shape):
        weights[name] = torch.randn(shape, generator=gen) * 0.15

    def linear(prefix, out_size, in_size):
        rand(prefix + '.weight', (out_size, in_size))
        rand(prefix + '.bias', (out_size,))

    for name, shape in (
        ('conv0', (channels, 1, 3, 3)),
        ('conv1_dw', (channels, 1, 3, 3)),
        ('conv1_pw', (channels, channels, 1, 1)),
        ('conv2_dw', (channels, 1, 3, 3)),
        ('conv2_pw', (channels, channels, 1, 1)),
    ):
        rand('encoder.subsampling.' + name + '.weight', shape)
        rand('encoder.subsampling.' + name + '.bias', (channels,))
    linear('encoder.subsampling.out', dim, channels * ((mels + 7) // 8))
    for index in range(2):
        prefix = f'encoder.blk.{index}'
        for norm in ('norm_ff1', 'norm_attn', 'norm_conv', 'norm_ff2', 'norm_out'):
            weights[f'{prefix}.{norm}.weight'] = torch.ones(dim)
            rand(f'{prefix}.{norm}.bias', (dim,))
        for ff in ('ff1', 'ff2'):
            linear(f'{prefix}.{ff}.linear1', dim * 2, dim)
            linear(f'{prefix}.{ff}.linear2', dim, dim * 2)
        for proj in ('q', 'k', 'v', 'out'):
            linear(f'{prefix}.attn.{proj}', dim, dim)
        rand(f'{prefix}.attn.pos.weight', (dim, dim))
        for bias in ('pos_bias_u', 'pos_bias_v'):
            rand(f'{prefix}.attn.{bias}', (2, dim // 2))
        for name, shape in (('pw1', (dim * 2, dim, 1)),
                            ('dw', (dim, 1, 5)), ('pw2', (dim, dim, 1))):
            rand(f'{prefix}.conv.{name}.weight', shape)
            rand(f'{prefix}.conv.{name}.bias', (shape[0],))
        weights[f'{prefix}.conv.bn.scale'] = torch.ones(dim)
        rand(f'{prefix}.conv.bn.shift', (dim,))
    meta = {'parakeet.model.type': 'tdt', 'parakeet.encoder.d_model': dim,
            'parakeet.encoder.n_layers': 2, 'parakeet.encoder.n_heads': 2,
            'parakeet.encoder.pos_emb_max_len': 64}
    return ref, weights, meta, mels


class PaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ref, cls.weights, cls.meta, cls.mels = fixture()
        cls.masked = EXPORTER.MaskedEncoderModule(cls.ref, cls.weights, cls.meta).eval()
        cls.legacy = EXPORTER.EncoderModule(cls.ref, cls.weights, cls.meta).eval()
        cls.mel = torch.randn(cls.mels, 33, generator=torch.Generator().manual_seed(29))
        cls.traced = torch.jit.trace(cls.masked, (cls.mel, torch.ones(33)), check_trace=False)

    def test_traced_valid_length_is_dynamic_and_padding_is_invariant(self):
        # Covers all stride residues, one frame, and exact capacity. In particular
        # N=9, L=8 must expose one valid encoder row, not ceil(N/8)=2 rows.
        for length in (*range(1, 18), 24, 31, 32, 33):
            with self.subTest(length=length), torch.inference_mode():
                mask = (torch.arange(33) < length).float()
                natural = self.legacy(self.mel[:, :length])
                padded = self.mel.clone()
                padded[:, length:] = 100.0  # invalid values must not leak inward
                result = self.traced(padded, mask)
                rows = (length + 7) // 8
                torch.testing.assert_close(result[:rows], natural, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(result[rows:], torch.zeros_like(result[rows:]))
                self.assertTrue(torch.isfinite(result).all())

    def test_fixture_detects_old_unmasked_padding_bug(self):
        with torch.inference_mode():
            natural = self.legacy(self.mel[:, :9])
            padded = self.mel.clone()
            padded[:, 9:] = 0
            unmasked = self.legacy(padded)[:2]
        self.assertGreater((natural - unmasked).abs().max().item(), 1e-3)

    def test_rejects_tdt_contexts_not_supported_by_masked_export(self):
        for key, value in (
            ('parakeet.encoder.causal_downsampling', True),
            ('parakeet.encoder.conv_context_size', 'causal'),
            ('parakeet.encoder.conv_norm_type', 'layer_norm'),
            ('parakeet.encoder.att_context_size_right', 1),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                EXPORTER.validate_export_contract({**self.meta, key: value})

    def test_full_valid_matches_existing_encoder(self):
        with torch.inference_mode():
            torch.testing.assert_close(self.masked(self.mel, torch.ones(33)),
                                       self.legacy(self.mel))


if __name__ == '__main__':
    unittest.main()
