#!/usr/bin/env python3
"""Model-free checks for the Neural Engine Core ML exporters.

Covers the pure helpers of export-encoder-coreml-ane.py (frame arithmetic and
the channel-axis LayerNorm the ANE graph uses) and the TDT decoder graph of
export-decoder-coreml-ane.py against its scalar reference on random weights.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None

if torch is None:
    print("SKIP: torch is not installed")
    raise SystemExit(0)

if "coremltools" not in sys.modules:
    try:
        import coremltools  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["coremltools"] = types.ModuleType("coremltools")

try:
    import gguf  # noqa: F401
except ModuleNotFoundError:
    print("SKIP: gguf is not installed")
    raise SystemExit(0)

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load(name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENCODER = load("export-encoder-coreml-ane.py", "export_encoder_coreml_ane")
DECODER = load("export-decoder-coreml-ane.py", "export_decoder_coreml_ane")
REFERENCE = load("ref-encoder-from-gguf.py", "ref_encoder_from_gguf")


def toy_encoder_weights(d_model=8, n_layers=2, n_heads=2, n_mels=8, channels=4, ff_dim=16,
                        conv_kernel=5):
    """Random FastConformer weights in the GGUF layout both scripts read."""
    torch.manual_seed(2)
    w = {}

    def rand(*shape, scale=0.3):
        return torch.randn(*shape) * scale

    w["preproc.mel_filterbank"] = torch.ones(n_mels, 9)
    w["encoder.subsampling.conv0.weight"] = rand(channels, 1, 3, 3)
    w["encoder.subsampling.conv0.bias"] = rand(channels, scale=0.1)
    for stage in ("conv1", "conv2"):
        w[f"encoder.subsampling.{stage}_dw.weight"] = rand(channels, 1, 3, 3)
        w[f"encoder.subsampling.{stage}_dw.bias"] = rand(channels, scale=0.1)
        w[f"encoder.subsampling.{stage}_pw.weight"] = rand(channels, channels, 1, 1)
        w[f"encoder.subsampling.{stage}_pw.bias"] = rand(channels, scale=0.1)
    freq_out = ENCODER.encoder_frames_for_mel(n_mels, causal=False)
    w["encoder.subsampling.out.weight"] = rand(d_model, channels * freq_out)
    w["encoder.subsampling.out.bias"] = rand(d_model, scale=0.1)
    for i in range(n_layers):
        p = f"encoder.blk.{i}"
        for norm in ("norm_ff1", "norm_attn", "norm_conv", "norm_ff2", "norm_out", "conv.norm"):
            w[f"{p}.{norm}.weight"] = 1.0 + rand(d_model, scale=0.1)
            w[f"{p}.{norm}.bias"] = rand(d_model, scale=0.1)
        for ff in ("ff1", "ff2"):
            w[f"{p}.{ff}.linear1.weight"] = rand(ff_dim, d_model)
            w[f"{p}.{ff}.linear1.bias"] = rand(ff_dim, scale=0.1)
            w[f"{p}.{ff}.linear2.weight"] = rand(d_model, ff_dim)
            w[f"{p}.{ff}.linear2.bias"] = rand(d_model, scale=0.1)
        for proj in ("q", "k", "v", "out", "pos"):
            w[f"{p}.attn.{proj}.weight"] = rand(d_model, d_model)
            if proj != "pos":
                w[f"{p}.attn.{proj}.bias"] = rand(d_model, scale=0.1)
        w[f"{p}.attn.pos_bias_u"] = rand(n_heads, d_model // n_heads, scale=0.1)
        w[f"{p}.attn.pos_bias_v"] = rand(n_heads, d_model // n_heads, scale=0.1)
        w[f"{p}.conv.pw1.weight"] = rand(2 * d_model, d_model)
        w[f"{p}.conv.pw1.bias"] = rand(2 * d_model, scale=0.1)
        w[f"{p}.conv.dw.weight"] = rand(d_model, 1, conv_kernel)
        w[f"{p}.conv.dw.bias"] = rand(d_model, scale=0.1)
        w[f"{p}.conv.pw2.weight"] = rand(d_model, d_model)
        w[f"{p}.conv.pw2.bias"] = rand(d_model, scale=0.1)
    meta = {
        "parakeet.encoder.d_model": d_model,
        "parakeet.encoder.n_layers": n_layers,
        "parakeet.encoder.n_heads": n_heads,
        "parakeet.encoder.xscaling": True,
        "parakeet.encoder.conv_norm_type": "layer_norm",
        "parakeet.encoder.causal_downsampling": False,
        "parakeet.encoder.pos_emb_max_len": 64,
    }
    return w, meta


class EncoderHelperTests(unittest.TestCase):
    def test_default_window_is_fifteen_seconds(self):
        self.assertEqual(ENCODER.DEFAULT_WINDOW_MEL_FRAMES, 1500)

    def test_encoder_frames_for_offline_window(self):
        # Three stride-2 convs, kernel 3, pad 1: 1500 -> 750 -> 375 -> 188.
        self.assertEqual(ENCODER.encoder_frames_for_mel(1500, causal=False), 188)
        self.assertEqual(ENCODER.encoder_frames_for_mel(1501, causal=False), 188)
        self.assertEqual(ENCODER.encoder_frames_for_mel(326, causal=False), 41)

    def test_encoder_frames_for_causal_window(self):
        self.assertEqual(ENCODER.encoder_frames_for_mel(1500, causal=True), 189)

    def test_channel_norm_matches_layer_norm_over_channels(self):
        torch.manual_seed(0)
        x = torch.randn(1, 6, 1, 5)
        gamma = torch.randn(1, 6, 1, 1)
        beta = torch.randn(1, 6, 1, 1)
        got = ENCODER.channel_norm(x, gamma, beta, eps=1e-5)
        want = F.layer_norm(x.permute(0, 2, 3, 1), (6,), gamma.reshape(6), beta.reshape(6),
                            eps=1e-5).permute(0, 3, 1, 2)
        self.assertTrue(torch.allclose(got, want, atol=1e-5))

    def test_key_bias_pad_is_finite(self):
        self.assertTrue(np.isfinite(np.float16(ENCODER.KEY_BIAS_PAD)))

    def test_halved_mask_keeps_exactly_the_valid_frames(self):
        # Every valid length v: the conv keeps floor((v - 1) / 2) + 1 frames,
        # and the even-index slice of a v-frame mask has that many ones.
        for valid in range(1, 40):
            mask = torch.zeros(1, 1, 40, 1)
            mask[:, :, :valid] = 1.0
            halved = ENCODER.AneEncoder.halve_mask(mask)
            self.assertEqual(int(halved.sum()), ENCODER.conv_out_len(valid), valid)
            self.assertEqual(halved.shape[2], ENCODER.conv_out_len(40))


class EncoderGraphTests(unittest.TestCase):
    WINDOW = 64

    def setUp(self):
        self.weights, self.meta = toy_encoder_weights()
        self.model = ENCODER.AneEncoder(REFERENCE, self.weights, self.meta, self.WINDOW).eval()

    def test_full_window_matches_reference(self):
        torch.manual_seed(0)
        mel = torch.randn(self.model.n_mels, self.WINDOW)
        self.assertTrue(ENCODER.compare_with_reference(
            self.model, REFERENCE, self.weights, self.meta, mel, "full"))

    def test_partial_window_valid_frames_match_short_input(self):
        # The valid frames of a padded, masked window must equal the reference
        # run on the short input alone; without masks at every subsampling
        # resolution the last valid frames absorb the padding.
        torch.manual_seed(1)
        for valid in (13, 22, 37, 51):
            mel = torch.randn(self.model.n_mels, valid)
            self.assertTrue(ENCODER.compare_with_reference(
                self.model, REFERENCE, self.weights, self.meta, mel, f"partial {valid}"), valid)

    def test_trailing_padding_frame_is_masked_at_the_valid_length(self):
        # The mel front-end pads the last frame with zeros; the engine passes
        # the audio length to the wrapper, which must mask from there. Marking
        # the padding frame valid lets its conv activations reach the last
        # audio frames.
        torch.manual_seed(3)
        padded = torch.randn(self.model.n_mels, 37)
        padded[:, -1] = 0.0
        want = ENCODER.reference_frames(REFERENCE, self.weights, self.meta, padded, valid_mel=36)
        masked_at_audio = ENCODER.ane_encoder_frames(self.model, padded, valid_mel=36)
        padding_marked_valid = ENCODER.ane_encoder_frames(self.model, padded, valid_mel=37)
        self.assertLess(ENCODER.relative_error(masked_at_audio, want), ENCODER.PARITY_REL_TOLERANCE)
        self.assertGreater(ENCODER.relative_error(padding_marked_valid[: want.shape[0]], want),
                           ENCODER.PARITY_REL_TOLERANCE)

    def test_check_parity_covers_full_and_partial_windows(self):
        self.assertTrue(ENCODER.check_parity(self.model, REFERENCE, self.weights, self.meta))


def toy_tdt_weights(h_pred=4, h_joint=3, d_enc=5, vocab=6, n_durations=3, layers=2):
    torch.manual_seed(1)
    weights = {}
    for layer in range(layers):
        p = f"tdt.predict.lstm.{layer}"
        weights[f"{p}.w_ih"] = torch.randn(4 * h_pred, h_pred) * 0.3
        weights[f"{p}.w_hh"] = torch.randn(4 * h_pred, h_pred) * 0.3
        weights[f"{p}.b_ih"] = torch.randn(4 * h_pred) * 0.1
        weights[f"{p}.b_hh"] = torch.randn(4 * h_pred) * 0.1
    weights["tdt.predict.embed.weight"] = torch.randn(vocab + 1, h_pred)
    weights["tdt.joint.enc.weight"] = torch.randn(h_joint, d_enc) * 0.3
    weights["tdt.joint.enc.bias"] = torch.randn(h_joint) * 0.1
    weights["tdt.joint.pred.weight"] = torch.randn(h_joint, h_pred) * 0.3
    weights["tdt.joint.pred.bias"] = torch.randn(h_joint) * 0.1
    weights["tdt.joint.out.weight"] = torch.randn(vocab + 1 + n_durations, h_joint) * 0.3
    weights["tdt.joint.out.bias"] = torch.randn(vocab + 1 + n_durations) * 0.1
    meta = {
        "parakeet.tdt.pred_rnn_layers": layers,
        "parakeet.tdt.pred_hidden": h_pred,
        "parakeet.tdt.joint_hidden": h_joint,
        "parakeet.tdt.vocab_size": vocab,
        "parakeet.tdt.blank_id": vocab,
        "parakeet.tdt.num_durations": n_durations,
    }
    return weights, meta


class DecoderExporterTests(unittest.TestCase):
    def test_geometry_reads_the_checkpoint(self):
        weights, meta = toy_tdt_weights()
        geom = DECODER.TdtGeometry(weights, meta)
        self.assertEqual((geom.n_layers, geom.h_pred, geom.d_enc, geom.v_out), (2, 4, 5, 10))
        self.assertEqual(geom.blank_id, 6)

    def test_geometry_rejects_wrong_joint_output(self):
        weights, meta = toy_tdt_weights()
        meta["parakeet.tdt.num_durations"] = 2
        with self.assertRaises(RuntimeError):
            DECODER.TdtGeometry(weights, meta)

    def test_geometry_rejects_missing_lstm_layer(self):
        weights, meta = toy_tdt_weights()
        meta["parakeet.tdt.pred_rnn_layers"] = 3
        with self.assertRaises(RuntimeError):
            DECODER.TdtGeometry(weights, meta)

    def test_graph_matches_scalar_reference(self):
        weights, meta = toy_tdt_weights()
        geom = DECODER.TdtGeometry(weights, meta)
        model = DECODER.AneTdtDecoder(weights, geom, lanes=3, width=1).eval()
        self.assertTrue(DECODER.check_parity(model, weights, geom))

    def test_default_lanes_and_width(self):
        self.assertEqual(DECODER.DEFAULT_LANES, 16)
        self.assertEqual(DECODER.DEFAULT_WIDTH, 1)


if __name__ == "__main__":
    unittest.main()
