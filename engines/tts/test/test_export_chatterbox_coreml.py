#!/usr/bin/env python3
"""Chatterbox S3Gen Core ML rebuild checks without model downloads.

Random weights with the checkpoint's shapes (fewer blocks where the block
count is a module constant): padding/masking equivalence of the encoder and
CFM estimator, the HiFT window walk against a whole-utterance decode, the
Neural Engine spellings (phase-form transposed convs, polyphase dilated and
blocked strided convs, polynomial Snake) against their direct forms, the
sidecar naming rule, and a Core ML conversion of a reduced encoder."""

import importlib.util
import math
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

HAVE_DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "torch"))
HAVE_COREML = HAVE_DEPS and importlib.util.find_spec("coremltools") is not None and sys.platform == "darwin"
if HAVE_DEPS:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import chatterbox_coreml_models as cm
    SPEC = importlib.util.spec_from_file_location("export_chatterbox_coreml", ROOT / "scripts/export-chatterbox-coreml.py")
    EXPORTER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(EXPORTER)

RNG = np.random.default_rng(20261027) if HAVE_DEPS else None


def tensor(*shape, scale=0.05):
    return RNG.normal(0, scale, shape).astype(np.float32)


def norm(weights, name, width, w="w", b="b"):
    weights[f"{name}/{w}"] = (1 + tensor(width)).astype(np.float32)
    weights[f"{name}/{b}"] = tensor(width)


def encoder_weights(blocks=1, up_blocks=1):
    t = {}
    e = "flow/encoder"
    for name in ("embed", "up_embed"):
        t[f"{e}/{name}/linear/w"], t[f"{e}/{name}/linear/b"] = tensor(cm.EMBED, cm.EMBED), tensor(cm.EMBED)
        norm(t, f"{e}/{name}/norm", cm.EMBED)
    for name, taps in (("pre_lookahead/conv1", 4), ("pre_lookahead/conv2", 3), ("up_layer/conv", 5)):
        t[f"{e}/{name}/w"], t[f"{e}/{name}/b"] = tensor(cm.EMBED, cm.EMBED, taps, scale=0.02), tensor(cm.EMBED)
    for prefix in [f"block{i}" for i in range(blocks)] + [f"up_block{i}" for i in range(up_blocks)]:
        p = f"{e}/{prefix}"
        norm(t, p + "/norm_mha", cm.EMBED)
        norm(t, p + "/norm_ff", cm.EMBED)
        for proj in "qkvo":
            t[f"{p}/attn/{proj}/w"], t[f"{p}/attn/{proj}/b"] = tensor(cm.EMBED, cm.EMBED), tensor(cm.EMBED)
        t[f"{p}/attn/pos/w"] = tensor(cm.EMBED, cm.EMBED)
        t[f"{p}/attn/pos_bias_u"] = tensor(cm.HEADS, cm.HEAD_DIM)
        t[f"{p}/attn/pos_bias_v"] = tensor(cm.HEADS, cm.HEAD_DIM)
        t[f"{p}/ff/w1/w"], t[f"{p}/ff/w1/b"] = tensor(2048, cm.EMBED), tensor(2048)
        t[f"{p}/ff/w2/w"], t[f"{p}/ff/w2/b"] = tensor(cm.EMBED, 2048), tensor(cm.EMBED)
    norm(t, f"{e}/after_norm", cm.EMBED)
    t["flow/encoder_proj/w"], t["flow/encoder_proj/b"] = tensor(cm.MEL, cm.EMBED), tensor(cm.MEL)
    return t


def estimator_weights(mid_blocks=1, tfm_blocks=1, width=256):
    t = {}

    def resnet(prefix, inputs):
        for block, ins in (("block1", inputs), ("block2", width)):
            t[f"{prefix}/{block}/block/0/weight"] = tensor(width, ins, 3)
            t[f"{prefix}/{block}/block/0/bias"] = tensor(width)
            norm(t, f"{prefix}/{block}/block/2", width, "weight", "bias")
        t[f"{prefix}/mlp/1/weight"], t[f"{prefix}/mlp/1/bias"] = tensor(width, cm.TIME_EMBED), tensor(width)
        t[f"{prefix}/res_conv/weight"], t[f"{prefix}/res_conv/bias"] = tensor(width, inputs, 1), tensor(width)

    def stage(prefix, inputs):
        resnet(prefix + "/0", inputs)
        for i in range(tfm_blocks):
            p = f"{prefix}/1/{i}"
            norm(t, p + "/norm1", width, "weight", "bias")
            norm(t, p + "/norm3", width, "weight", "bias")
            for proj in ("to_q", "to_k", "to_v"):
                t[f"{p}/attn1/{proj}/weight"] = tensor(cm.HEADS * cm.HEAD_DIM, width)
            t[f"{p}/attn1/to_out/0/weight"] = tensor(width, cm.HEADS * cm.HEAD_DIM)
            t[f"{p}/attn1/to_out/0/bias"] = tensor(width)
            t[f"{p}/ff/net/0/proj/weight"], t[f"{p}/ff/net/0/proj/bias"] = tensor(4 * width, width), tensor(4 * width)
            t[f"{p}/ff/net/2/weight"], t[f"{p}/ff/net/2/bias"] = tensor(width, 4 * width), tensor(width)

    stage("cfm/down_blocks/0", 4 * cm.MEL)
    for i in range(mid_blocks):
        stage(f"cfm/mid_blocks/{i}", width)
    stage("cfm/up_blocks/0", 2 * width)
    for name in ("cfm/down_blocks/0/2", "cfm/up_blocks/0/2", "cfm/final_block/block/0"):
        t[name + "/weight"], t[name + "/bias"] = tensor(width, width, 3), tensor(width)
    norm(t, "cfm/final_block/block/2", width, "weight", "bias")
    t["cfm/final_proj/weight"], t["cfm/final_proj/bias"] = tensor(cm.MEL, width, 1), tensor(cm.MEL)
    return t


def hift_weights():
    t = {"hift/conv_pre/weight": tensor(512, cm.MEL, 7), "hift/conv_pre/bias": tensor(512),
         "hift/conv_post/weight": tensor(cm.STFT_BINS, 64, 7), "hift/conv_post/bias": tensor(cm.STFT_BINS)}
    channels = (512, 256, 128, 64)
    for i, (stride, kernel) in enumerate(cm.HIFT_UPSAMPLE):
        t[f"hift/ups/{i}/weight"] = tensor(channels[i], channels[i + 1], kernel)
        t[f"hift/ups/{i}/bias"] = tensor(channels[i + 1])
        source_kernel = {0: 30, 1: 6, 2: 1}[i]
        t[f"hift/source_downs/{i}/weight"] = tensor(channels[i + 1], cm.STFT_BINS, source_kernel)
        t[f"hift/source_downs/{i}/bias"] = tensor(channels[i + 1])
        blocks = [(f"hift/source_resblocks/{i}", cm.SOURCE_KERNELS[i])]
        blocks += [(f"hift/resblocks/{i * 3 + j}", k) for j, k in enumerate(cm.RESBLOCK_KERNELS)]
        for prefix, kernel_size in blocks:
            width = channels[i + 1]
            for j in range(3):
                for act in ("activations1", "activations2"):
                    t[f"{prefix}/{act}/{j}/alpha"] = (0.05 + RNG.uniform(0, 2.5, width)).astype(np.float32)
                for conv in ("convs1", "convs2"):
                    t[f"{prefix}/{conv}/{j}/weight"] = tensor(width, width, kernel_size)
                    t[f"{prefix}/{conv}/{j}/bias"] = tensor(width)
    return t


def assert_close(test, got, want, atol):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    test.assertEqual(got.shape, want.shape)
    err = float(np.abs(got - want).max())
    test.assertLessEqual(err, atol, f"max abs error {err:.3e} > {atol:.1e}")


def hift_windows(frames, window, context):
    """coreml_hift_windows (src/chatterbox_coreml.cpp)."""
    plan, begin = [], 0
    while begin < frames:
        start = max(0, begin - context)
        end = frames if start + window >= frames else start + window - context
        plan.append((start, begin, end))
        begin = end
    return plan


class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.saved = {k: getattr(cm, k) for k in ("ENCODER_BLOCKS", "UP_ENCODER_BLOCKS", "CFM_MID_BLOCKS",
                                                  "CFM_TFM_BLOCKS")}
        cm.ENCODER_BLOCKS = cm.UP_ENCODER_BLOCKS = cm.CFM_MID_BLOCKS = cm.CFM_TFM_BLOCKS = 1

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.saved.items():
            setattr(cm, k, v)

    def test_context_matches_runtime(self):
        header = (ROOT / "src/chatterbox_coreml.h").read_text()
        runtime = int(re.search(r"HIFT_CONTEXT_FRAMES\s*=\s*(\d+)", header).group(1))
        self.assertEqual(cm.hift_receptive_frames(), runtime)

    def test_sidecar_naming(self):
        self.assertEqual(EXPORTER.model_stem(Path("m/chatterbox-s3gen-q8_0.gguf")), "chatterbox-s3gen")
        self.assertEqual(EXPORTER.model_stem(Path("chatterbox-s3gen-mtl-f32.gguf")), "chatterbox-s3gen-mtl")
        self.assertEqual(EXPORTER.model_stem(Path("chatterbox-s3gen.gguf")), "chatterbox-s3gen")
        self.assertEqual(EXPORTER.function_name("encoder", 320), "tokens_320")
        self.assertEqual(EXPORTER.function_name("cfm", 640), "frames_640")

    def test_relative_shift(self):
        length = 6
        scores = torch.randn(1, cm.HEADS, length, 2 * length - 1)
        shifted = cm.relative_shift(scores, length)
        for i in range(length):
            for j in range(length):
                # column m of `scores` is relative position length - 1 - m
                torch.testing.assert_close(shifted[0, :, i, j], scores[0, :, i, length - 1 - (i - j)])

    def test_encoder_padding(self):
        tensors, meta = encoder_weights(), {"s3gen.encoder.token_mel_ratio": 2}
        tokens = 7
        embedding = torch.from_numpy(tensor(1, tokens, cm.EMBED, scale=1.0))
        with torch.no_grad():
            exact = cm.Encoder(tensors, meta, tokens).eval()(embedding, torch.ones(1, 1, tokens))
            padded_in = F.pad(embedding, (0, 0, 0, 5))
            valid = torch.zeros(1, 1, tokens + 5)
            valid[..., :tokens] = 1
            padded = cm.Encoder(tensors, meta, tokens + 5).eval()(padded_in, valid)[:, :2 * tokens]
        assert_close(self, padded, exact, 1e-4)

    def test_estimator_padding(self):
        tensors = estimator_weights()
        frames, batch, extra = 9, 2, 7
        planes = [torch.from_numpy(tensor(batch, cm.MEL, frames, scale=1.0)) for _ in range(3)]
        spks, t_emb = torch.from_numpy(tensor(batch, cm.MEL, scale=1.0)), torch.from_numpy(tensor(batch, 1024))
        with torch.no_grad():
            exact = cm.Estimator(tensors, frames, batch).eval()(*planes, spks, t_emb, torch.ones(1, 1, frames))
            valid = torch.zeros(1, 1, frames + extra)
            valid[..., :frames] = 1
            padded = cm.Estimator(tensors, frames + extra, batch).eval()(
                *[F.pad(p, (0, extra)) for p in planes], spks, t_emb, valid)[..., :frames]
        assert_close(self, padded, exact, 1e-4)

    def test_upsample_phase_form(self):
        for stride, kernel in cm.HIFT_UPSAMPLE:
            weight, bias = tensor(4, 3, kernel, scale=1.0), tensor(3)
            x = torch.from_numpy(tensor(1, 4, 10, scale=1.0))
            up = cm.Upsample(weight, bias, stride, (kernel - stride) // 2, 10)
            want = F.conv_transpose1d(x, torch.from_numpy(weight), torch.from_numpy(bias), stride=stride,
                                      padding=(kernel - stride) // 2)
            assert_close(self, up(x), want, 1e-5)

    def test_hift_conv_forms(self):
        hift = cm.HiFT(hift_weights(), 34)
        x = torch.from_numpy(tensor(1, 128, 41, scale=1.0))
        for name, kernel, dilation in (("hift/resblocks/4/convs1/2", 7, 5), ("hift/resblocks/5/convs1/1", 11, 3)):
            pad = (kernel * dilation - dilation) // 2
            want = F.conv1d(F.pad(x, (pad, pad)), hift.weight(name + "/weight"), hift.weight(name + "/bias"),
                            dilation=dilation)
            assert_close(self, hift.conv(x, name, pad, dilation, length=41), want, 1e-4)
        source = torch.from_numpy(tensor(1, cm.STFT_BINS, 4 * cm.HIFT_HOP + 1, scale=1.0))
        name = "hift/source_downs/0"
        want = F.conv1d(F.pad(source, (7, 7)), hift.weight(name + "/weight"), hift.weight(name + "/bias"), stride=15)
        assert_close(self, hift.conv(source, name, 7, stride=15, length=4 * 8), want, 1e-4)

    def test_snake_polynomial(self):
        alpha = np.array([-0.03, 0.0009, 0.05, 0.5, 1.3, 2.6], dtype=np.float32)
        x = torch.linspace(-12, 12, 4001).reshape(1, 1, -1).expand(1, len(alpha), -1)
        exact, poly = cm.Snake(alpha, exact=True), cm.Snake(alpha)
        with torch.no_grad():
            err = (poly(x) - exact(x)).abs() / (1 + exact(x).abs())
        self.assertLess(float(err.max()), 2e-3)

    def test_hift_window_walk(self):
        tensors, window, frames = hift_weights(), 34, 75
        context = cm.hift_receptive_frames()
        mel = torch.from_numpy(tensor(1, cm.MEL, frames, scale=1.0))
        source = torch.from_numpy(tensor(1, cm.STFT_BINS, frames * cm.HIFT_HOP + 1, scale=1.0))
        with torch.no_grad():
            whole = cm.HiFT(tensors, frames, exact_snake=True).eval()(mel, source, torch.ones(1, 1, frames))
            model = cm.HiFT(tensors, window, exact_snake=True).eval()
            stitched = torch.zeros_like(whole)
            for start, keep_begin, keep_end in hift_windows(frames, window, context):
                count = min(window, frames - start)
                mel_w = torch.zeros(1, cm.MEL, window)
                mel_w[..., :count] = mel[..., start:start + count]
                src_w = torch.zeros(1, cm.STFT_BINS, window * cm.HIFT_HOP + 1)
                src_w[..., :count * cm.HIFT_HOP + 1] = source[..., start * cm.HIFT_HOP:(start + count) * cm.HIFT_HOP + 1]
                valid = torch.zeros(1, 1, window)
                valid[..., :count] = 1
                out = model(mel_w, src_w, valid)
                a = (keep_begin - start) * cm.HIFT_HOP
                n = (keep_end - keep_begin) * cm.HIFT_HOP + (1 if keep_end == frames else 0)
                stitched[..., keep_begin * cm.HIFT_HOP:keep_begin * cm.HIFT_HOP + n] = out[..., a:a + n]
        assert_close(self, stitched, whole, 2e-4 * max(1.0, float(whole.abs().max())))

    @unittest.skipUnless(HAVE_COREML, "coremltools on macOS")
    def test_coreml_encoder_conversion(self):
        import coremltools as ct

        tensors, meta = encoder_weights(), {"s3gen.encoder.token_mel_ratio": 2}
        model, inputs, outputs = EXPORTER.stage_model("encoder", tensors, meta, 8)
        inputs = {"embedding": torch.from_numpy(tensor(1, 8, cm.EMBED, scale=1.0)), "valid": torch.ones(1, 1, 8)}
        inputs["valid"][..., 6:] = 0
        converted = EXPORTER.convert(model, inputs, outputs)
        got = converted.predict({k: v.numpy() for k, v in inputs.items()})["mu"]
        with torch.no_grad():
            want = model(*inputs.values()).numpy()
        cosine = float((got * want).sum() / (np.linalg.norm(got) * np.linalg.norm(want)))
        self.assertGreater(cosine, 0.999)
        self.assertEqual(ct.target.iOS18, ct.target.iOS18)


if __name__ == "__main__":
    if not HAVE_DEPS:
        print("SKIP: numpy/torch unavailable")
        sys.exit(0)
    unittest.main()
