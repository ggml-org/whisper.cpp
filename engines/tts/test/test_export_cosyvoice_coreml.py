#!/usr/bin/env python3
"""CosyVoice Core ML state and padding equivalence checks without model downloads."""

from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

HAVE_DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "torch"))
if HAVE_DEPS:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from cosyvoice_coreml_models import F0, Flow, FlowFrontend, Qwen, QwenIO
    EXPORT_SPEC = importlib.util.spec_from_file_location(
        "export_cosyvoice_coreml", Path(__file__).resolve().parents[1] / "scripts/export-cosyvoice3-coreml.py")
    EXPORTER = importlib.util.module_from_spec(EXPORT_SPEC)
    EXPORT_SPEC.loader.exec_module(EXPORTER)

GENERATOR = np.random.default_rng(14832) if HAVE_DEPS else None
DIM = 16
HEAD_DIM = 8
FRAMES = 8
VALID_FRAMES = 3


def tensor(*shape):
    return GENERATOR.normal(0, 0.03, shape).astype(np.float32)


def linear(weights, name, output, width, bias=True):
    weights[name + "/weight"] = tensor(output, width)
    if bias:
        weights[name + "/bias"] = tensor(output)


def conv(weights, name, output, width, kernel):
    weights[name + "/weight"] = tensor(output, width, kernel)
    weights[name + "/bias"] = tensor(output)


def flow_weights():
    weights = {}
    linear(weights, "flow/input_embed/proj", DIM, 320)
    linear(weights, "flow/time_embed/time_mlp/0", DIM, 256)
    linear(weights, "flow/time_embed/time_mlp/2", DIM, DIM)
    linear(weights, "flow/norm_out/linear", DIM * 2, DIM)
    linear(weights, "flow/proj_out", 80, DIM)
    for index in range(2):
        conv(weights, f"flow/input_embed/conv_pos_embed/conv{index + 1}/0", DIM, DIM // 4, 3)
        prefix = f"flow/blk/{index}"
        linear(weights, prefix + "/attn_norm/linear", DIM * 6, DIM)
        linear(weights, prefix + "/attn/to_qkv", DIM * 3, DIM)
        linear(weights, prefix + "/attn/to_out/0", DIM, DIM)
        linear(weights, prefix + "/ff/ff/0/0", DIM * 2, DIM)
        linear(weights, prefix + "/ff/ff/2", DIM, DIM * 2)
    return weights


def qwen_weights():
    weights = {"lm/blk/0/in_ln/weight": np.ones(DIM, np.float32),
               "lm/blk/0/post_ln/weight": np.ones(DIM, np.float32),
               "lm/norm/weight": np.ones(DIM, np.float32)}
    for name, output in (("q_proj", DIM), ("k_proj", HEAD_DIM), ("v_proj", HEAD_DIM)):
        linear(weights, "lm/blk/0/" + name, output, DIM)
    for name, output, width in (("o_proj", DIM, DIM), ("gate", DIM * 2, DIM),
                                ("up", DIM * 2, DIM), ("down", DIM, DIM * 2)):
        linear(weights, "lm/blk/0/" + name, output, width, False)
    linear(weights, "lm/llm_decoder", 11, DIM, False)
    return weights


@unittest.skipUnless(HAVE_DEPS, "numpy/torch unavailable")
class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(193)

    def test_hift_preserves_fp32_for_waveform_parity(self):
        for operation in ("conv", "linear", "exp", "mul", "sin"):
            op = SimpleNamespace(op_type=operation, inputs={
                "weight": SimpleNamespace(name="conv_post_magnitude__weight")})
            self.assertFalse(EXPORTER.use_half_precision(op, "hift"))

    def test_f0_padding_preserves_real_frames(self):
        weights = {}
        for index in range(5):
            conv(weights, f"hift/f0_predictor/condnet/{index * 2}", DIM, 80 if index == 0 else DIM, 3)
        linear(weights, "hift/f0_predictor/classifier", 1, DIM)
        model = F0(weights).eval()
        mel = torch.randn(1, 80, VALID_FRAMES)
        with torch.no_grad():
            original = model(mel)
            padded = model(F.pad(mel, (0, FRAMES - VALID_FRAMES)))[:, :VALID_FRAMES]
        torch.testing.assert_close(original, padded, atol=1e-6, rtol=1e-5)

    def test_flow_padding_masks_attention_at_every_layer(self):
        model = Flow(flow_weights(), {"cosyvoice3.flow.depth": 2, "cosyvoice3.flow.heads": 2,
                                     "cosyvoice3.flow.dim_head": HEAD_DIM,
                                     "cosyvoice3.flow.conv_groups": 4}).eval()
        inputs = [torch.randn(2, VALID_FRAMES, 80) for _ in range(3)]
        speaker, time = torch.randn(2, 80), torch.randn(2, 256)
        cosine, sine = torch.ones(1, FRAMES, HEAD_DIM // 2), torch.zeros(1, FRAMES, HEAD_DIM // 2)
        mask = torch.full((1, 1, 1, FRAMES), -1e4)
        mask[..., :VALID_FRAMES] = 0
        with torch.no_grad():
            original = model(*inputs, speaker, time, cosine[:, :VALID_FRAMES], sine[:, :VALID_FRAMES],
                             torch.zeros(1, 1, 1, VALID_FRAMES))
            padded = model(*(F.pad(x, (0, 0, 0, FRAMES - VALID_FRAMES)) for x in inputs),
                           speaker, time, cosine, sine, mask)[:, :VALID_FRAMES]
        torch.testing.assert_close(original, padded, atol=1e-6, rtol=1e-5)

    def test_frontend_padding_does_not_embed_padding_tokens(self):
        weights = {"flow/input_embedding/weight": tensor(10, 80)}
        conv(weights, "flow/pre_lookahead_layer/conv1", DIM, 80, 4)
        conv(weights, "flow/pre_lookahead_layer/conv2", 80, DIM, 3)
        linear(weights, "flow/spk_embed_affine_layer", 80, DIM)
        model = FlowFrontend(weights).eval()
        ids = torch.tensor([[1.0, 3.0, 2.0]])
        speaker = torch.randn(1, DIM)
        with torch.no_grad():
            original, _ = model(ids, torch.ones(1, VALID_FRAMES, 1), speaker)
            padded, _ = model(F.pad(ids, (0, FRAMES - VALID_FRAMES)),
                              F.pad(torch.ones(1, VALID_FRAMES, 1), (0, 0, 0, FRAMES - VALID_FRAMES)), speaker)
        torch.testing.assert_close(original, padded[:, :VALID_FRAMES * 2], atol=1e-6, rtol=1e-5)

    def test_lm_explicit_cache_matches_stateful_sequence(self):
        metadata = {"cosyvoice3.llm.depth": 1, "cosyvoice3.llm.n_head": 2,
                    "cosyvoice3.llm.n_kv": 1, "cosyvoice3.llm.head_dim": HEAD_DIM}
        weights = qwen_weights()
        stateful, explicit = Qwen(weights, metadata, FRAMES), QwenIO(weights, metadata, FRAMES)
        keys = torch.zeros(1, 1, 1, FRAMES, HEAD_DIM)
        values = torch.zeros_like(keys)
        cosine, sine = torch.ones(1, 1, 1, HEAD_DIM // 2), torch.zeros(1, 1, 1, HEAD_DIM // 2)
        for position in range(3):
            x = torch.randn(1, 1, DIM)
            write = torch.zeros(1, 1, FRAMES, 1)
            write[..., position, :] = 1
            mask = torch.full((1, 1, 1, FRAMES), -1e4)
            mask[..., :position + 1] = 0
            with torch.no_grad():
                expected = stateful(x, cosine, sine, write, mask)
                actual, keys, values = explicit(x, cosine, sine, write, mask, keys, values)
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            torch.testing.assert_close(stateful.key_0.float(), keys[0], atol=0, rtol=0)
            torch.testing.assert_close(stateful.value_0.float(), values[0], atol=0, rtol=0)

    def test_flow_large_attention_scores_remain_finite_in_half_precision(self):
        weights = flow_weights()
        for index in range(2):
            prefix = f"flow/blk/{index}/attn/to_qkv"
            weights[prefix + "/weight"].fill(0)
            weights[prefix + "/bias"].fill(300)
        model = Flow(weights, {"cosyvoice3.flow.depth": 2, "cosyvoice3.flow.heads": 2,
                               "cosyvoice3.flow.dim_head": HEAD_DIM,
                               "cosyvoice3.flow.conv_groups": 4}).eval().half()
        inputs = [torch.randn(2, FRAMES, 80).half() for _ in range(3)]
        with torch.no_grad():
            output = model(*inputs, torch.randn(2, 80).half(), torch.randn(2, 256).half(),
                           torch.ones(1, FRAMES, HEAD_DIM // 2).half(),
                           torch.zeros(1, FRAMES, HEAD_DIM // 2).half(),
                           torch.zeros(1, 1, 1, FRAMES).half())
        self.assertTrue(torch.isfinite(output).all().item())

    @unittest.skipUnless(sys.platform == "darwin" and importlib.util.find_spec("coremltools"),
                         "Core ML prediction requires macOS and coremltools")
    def test_coreml_lm_cache_roundtrip(self):
        import coremltools as ct
        metadata = {"cosyvoice3.llm.depth": 1, "cosyvoice3.llm.n_head": 2,
                    "cosyvoice3.llm.n_kv": 1, "cosyvoice3.llm.head_dim": HEAD_DIM}
        model = QwenIO(qwen_weights(), metadata, FRAMES).eval()
        x = torch.randn(1, 1, DIM)
        cosine, sine = torch.ones(1, 1, 1, HEAD_DIM // 2), torch.zeros(1, 1, 1, HEAD_DIM // 2)
        write, mask = torch.zeros(1, 1, FRAMES, 1), torch.full((1, 1, 1, FRAMES), -1e4)
        write[..., 0, :] = 1
        mask[..., 0] = 0
        keys = torch.zeros(1, 1, 1, FRAMES, HEAD_DIM)
        inputs = dict(x=x, cosine=cosine, sine=sine, write_mask=write, mask=mask,
                      keys=keys, values=keys.clone())
        with torch.no_grad():
            traced = torch.jit.trace(model, tuple(inputs.values()), check_trace=False)
            expected = model(*inputs.values())
        converted = ct.convert(traced, convert_to="mlprogram",
                               inputs=[ct.TensorType(name=k, shape=v.shape, dtype=np.float32)
                                       for k, v in inputs.items()],
                               outputs=[ct.TensorType(name=k, dtype=np.float32)
                                        for k in ("logits", "keys_out", "values_out")],
                               minimum_deployment_target=ct.target.iOS18,
                               compute_precision=ct.precision.FLOAT16,
                               compute_units=ct.ComputeUnit.CPU_AND_NE)
        actual = converted.predict({k: v.numpy() for k, v in inputs.items()})
        for name, reference in zip(("logits", "keys_out", "values_out"), expected):
            np.testing.assert_allclose(reference.numpy(), actual[name], atol=2e-3, rtol=2e-3)
        self.assertGreater(np.abs(actual["keys_out"]).sum(), 0)
        inputs["keys"] = torch.from_numpy(actual["keys_out"])
        inputs["values"] = torch.from_numpy(actual["values_out"])
        inputs["x"] = torch.randn(1, 1, DIM)
        angle = torch.pow(1000000.0, -2 * torch.arange(HEAD_DIM // 2) / HEAD_DIM)
        inputs["cosine"] = angle.cos().reshape(1, 1, 1, -1)
        inputs["sine"] = angle.sin().reshape(1, 1, 1, -1)
        inputs["write_mask"].zero_()
        inputs["write_mask"][..., 1, :] = 1
        inputs["mask"][..., 1] = 0
        with torch.no_grad():
            expected = model(*inputs.values())
        repeated = converted.predict({k: v.numpy() for k, v in inputs.items()})
        for name, reference in zip(("logits", "keys_out", "values_out"), expected):
            np.testing.assert_allclose(reference.numpy(), repeated[name], atol=2e-3, rtol=2e-3)


    def test_lm_state_reset_reproduces_the_same_prefix(self):
        model = Qwen(qwen_weights(), {"cosyvoice3.llm.depth": 1, "cosyvoice3.llm.n_head": 2,
                                     "cosyvoice3.llm.n_kv": 1, "cosyvoice3.llm.head_dim": HEAD_DIM}, FRAMES)
        x = torch.randn(1, 1, DIM)
        cosine, sine = torch.ones(1, 1, 1, HEAD_DIM // 2), torch.zeros(1, 1, 1, HEAD_DIM // 2)
        write_mask = torch.zeros(1, 1, FRAMES, 1)
        write_mask[..., 0, :] = 1
        mask = torch.full((1, 1, 1, FRAMES), -1e4)
        mask[..., 0] = 0
        with torch.no_grad():
            first = model(x, cosine, sine, write_mask, mask)
            self.assertGreater(model.key_0.abs().sum().item(), 0)
            model.reset()
            self.assertEqual(model.key_0.abs().sum().item(), 0)
            repeated = model(x, cosine, sine, write_mask, mask)
        torch.testing.assert_close(first, repeated, atol=0, rtol=0)


if __name__ == "__main__":
    if not HAVE_DEPS:
        print("SKIP: numpy/torch unavailable")
    unittest.main()
