import sys
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from minimax_ane_models import LMBlock, LMBlockPair, Vocoder, Weights, phase_weight, compact_lm_head, stable_silu


class StableActivationTests(unittest.TestCase):
    def test_matches_original_silu_across_negative_and_positive_values(self):
        value = torch.linspace(-100, 100, 10001)
        torch.testing.assert_close(stable_silu(value), F.silu(value), rtol=1e-6, atol=1e-6)

    def test_fp16_negative_tail_remains_finite_and_matches_float_reference(self):
        value = torch.tensor([-65504, -100, -31, -20, -16, -12, -11.5, -11, -8, -1, 0, 1, 12, 100, 65504], dtype=torch.float16)
        actual = stable_silu(value)
        expected = F.silu(value.float()).half()
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=0.002, atol=2e-6)
        self.assertLess(actual[5].item(), 0)
        self.assertEqual((value/(1+torch.exp(-value)))[5].item(), 0)


class StableProjectionTests(unittest.TestCase):
    def test_qk_normalization_preserves_large_fp16_activations(self):
        config = {"attention.layer_norm_rms_epsilon": 1e-6, "attention.key_length": 8,
                  "attention.head_count": 1, "attention.head_count_kv": 1}
        module = LMBlock({"norm": np.ones(8, dtype=np.float32)}, config, 8).half()
        value = torch.ones(1, 8, 1, 2, dtype=torch.float16)
        value[:, 0] = 10000
        actual = module.qk_norm_rope(value, "norm", torch.ones(1, 1, 4, 1, dtype=value.dtype),
                                    torch.zeros(1, 1, 4, 1, dtype=value.dtype))
        expected = value.float()*torch.rsqrt(value.float().square().mean(1, keepdim=True)+1e-6)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float(), expected, rtol=.002, atol=2e-6)

    def test_rms_preserves_large_small_and_zero_fp16_activations(self):
        module = Weights({"norm": np.ones(4096, dtype=np.float32)}).half()
        for magnitude in [10000, 1e-4, 1e-7, 0]:
            value = torch.full((1, 4096, 1, 2), magnitude/100, dtype=torch.float16)
            value[:, 0] = magnitude
            actual = module.rms(value, "norm", 1e-6)
            expected = value.float()*torch.rsqrt(value.float().square().mean(1, keepdim=True)+1e-6)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual.float(), expected, rtol=.002, atol=2e-6)

    def test_mlp_down_projection_preserves_finite_fp16_outlier(self):
        config = {"attention.layer_norm_rms_epsilon": 1e-6, "attention.key_length": 1,
                  "attention.head_count": 1, "attention.head_count_kv": 1}
        tensors = {"ffn_gate.weight": np.full((2, 1), 94, dtype=np.float32),
                   "ffn_up.weight": np.full((2, 1), 100, dtype=np.float32),
                   "ffn_down.weight": np.full((1, 2), .25, dtype=np.float32)}
        module = LMBlock(tensors, config, 8, stable_activation=True).half()
        value = torch.ones(1, 1, 1, 2, dtype=torch.float16)
        actual = module.swiglu(value)
        gate = F.conv2d(value.float(), torch.from_numpy(tensors["ffn_gate.weight"])[:, :, None, None])
        up = F.conv2d(value.float(), torch.from_numpy(tensors["ffn_up.weight"])[:, :, None, None])
        expected = F.conv2d(F.silu(gate)*up, torch.from_numpy(tensors["ffn_down.weight"])[:, :, None, None]).half()
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=.002, atol=0)


class LMBlockPairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.config = {"attention.layer_norm_rms_epsilon": 1e-6, "attention.key_length": 8,
                       "attention.head_count": 4, "attention.head_count_kv": 2}
        self.tensors = [self.weights() for _ in range(2)]

    def weights(self):
        shapes = {"attn_q": (32, 32), "attn_k": (16, 32), "attn_v": (16, 32),
                  "attn_output": (32, 32), "ffn_gate": (48, 32), "ffn_up": (48, 32), "ffn_down": (32, 48)}
        weights = {name+".weight": (torch.randn(shape)*0.1).numpy() for name, shape in shapes.items()}
        for name, length in (("attn_norm", 32), ("ffn_norm", 32), ("attn_q_norm", 8), ("attn_k_norm", 8)):
            weights[name+".weight"] = torch.ones(length).numpy()
        return weights

    def inputs(self, position):
        selector = torch.zeros(1, 8)
        selector[0, position] = 1
        mask = torch.full((1, 8, 1, 1), -torch.inf)
        mask[:, :position+1] = 0
        phase = position*torch.pow(1000000.0, -torch.arange(4)/4)
        return (torch.randn(1, 32, 1, 2), selector, mask, phase.cos()[None, None, :, None],
                phase.sin()[None, None, :, None])

    def test_consecutive_causal_steps_match_separate_blocks(self):
        pair = LMBlockPair(self.tensors, self.config, 8)
        blocks = [LMBlock(weights, self.config, 8) for weights in self.tensors]
        with torch.no_grad():
            for position in range(4):
                data = self.inputs(position)
                expected = blocks[1](blocks[0](*data), *data[1:])
                torch.testing.assert_close(pair(*data), expected, rtol=0, atol=0)
                for grouped, separate in zip(pair.blocks, blocks):
                    torch.testing.assert_close(grouped.key_cache, separate.key_cache, rtol=0, atol=0)
                    torch.testing.assert_close(grouped.value_cache, separate.value_cache, rtol=0, atol=0)

    def test_stable_activation_preserves_float32_causal_outputs(self):
        original = LMBlockPair(self.tensors, self.config, 8)
        stable = LMBlockPair(self.tensors, self.config, 8, stable_activation=True)
        with torch.no_grad():
            for position in range(4):
                data = self.inputs(position)
                actual, expected = stable(*data), original(*data)
                relative = torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected)
                self.assertLessEqual(relative.item(), 32*torch.finfo(torch.float32).eps)

    def test_cfg_branch_and_block_states_are_independent(self):
        first = LMBlockPair(self.tensors, self.config, 8)
        second = LMBlockPair(self.tensors, self.config, 8)
        with torch.no_grad():
            for position in range(3):
                data = self.inputs(position)
                changed = data[0].clone()
                changed[:, :, :, 1] += 3
                expected = first(*data)
                actual = second(changed, *data[1:])
                torch.testing.assert_close(actual[:, :, :, 0], expected[:, :, :, 0], rtol=0, atol=0)
                self.assertFalse(torch.allclose(actual[:, :, :, 1], expected[:, :, :, 1]))
        states = [value for name, value in first.named_buffers() if name.endswith(("key_cache", "value_cache"))]
        self.assertEqual(len({value.data_ptr() for value in states}), 4)

    def test_rejects_incomplete_or_oversized_pair(self):
        for tensors in ([], self.tensors[:1], self.tensors+self.tensors[:1]):
            with self.assertRaises(ValueError):
                LMBlockPair(tensors, self.config, 8)


class ConvolutionLayoutTests(unittest.TestCase):
    def test_lm_uses_untied_output_weights_and_eos_last(self):
        output = np.arange(40, dtype=np.float32).reshape(10, 4)
        tensors = {"output.weight": output, "token_embd.weight": -output}
        actual = compact_lm_head(tensors, 3, 4, 1)
        np.testing.assert_array_equal(actual, output[[3, 4, 5, 6, 1]])

    def test_lm_rejects_missing_output_weights(self):
        with self.assertRaises(KeyError):
            compact_lm_head({"token_embd.weight": np.zeros((10, 4))}, 3, 4, 1)

    def test_folded_convolution_preserves_adjacent_tiles(self):
        torch.manual_seed(27)
        convolve = Vocoder.convolve
        for length in (1024, 2048):
            for dilation in (1, 3, 9):
                value = torch.randn(1, 3, 1, length)
                value[:, :, :, 255:258] += 7
                weight, bias = torch.randn(4, 3, 7), torch.randn(4)
                expected = F.conv2d(value, weight[:, :, None, :], bias, padding=(0, 3*dilation), dilation=(1, dilation))
                actual = convolve(None, value, weight, bias, dilation)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_folded_pointwise_convolution_preserves_order(self):
        value = torch.arange(2048).reshape(1, 1, 1, 2048).float()
        weight, bias = torch.tensor([[[2.0]]]), torch.tensor([3.0])
        actual = Vocoder.convolve(None, value, weight, bias)
        torch.testing.assert_close(actual, value*2+3)

    def test_phase_convolution_matches_transposed_convolution(self):
        torch.manual_seed(44)
        for stride in (2, 4, 8):
            value = torch.randn(2, 3, 19)
            weight, bias = torch.randn(3, 4, 2*stride), torch.randn(4)
            expected = F.conv_transpose1d(value, weight, bias, stride=stride, padding=stride//2)
            projected = F.conv1d(value, phase_weight(weight, stride), bias.repeat(stride), padding=1)
            actual = projected.reshape(2, stride, 4, 19).permute(0, 2, 3, 1).reshape_as(expected)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_padded_vocoder_matches_actual_short_input(self):
        torch.manual_seed(38)
        tensors = {}
        def conv(name, output, inputs, kernel):
            tensors[name+".weight"] = (torch.randn(output, inputs, kernel)*0.025).numpy()
            tensors[name+".bias"] = (torch.randn(output)*0.025).numpy()
        def snake(name, channels):
            tensors[name] = torch.full((1, channels, 1), 1.7).numpy()
        conv("dec_in", 4, 2, 1)
        conv("conv_in", 16, 4, 7)
        channels = 16
        for index, stride in enumerate((8, 8, 4, 2)):
            prefix = f"blk.{index}."
            snake(prefix+"snake.alpha", channels)
            tensors[prefix+"convt.weight"] = (torch.randn(channels, channels//2, 2*stride)*0.025).numpy()
            tensors[prefix+"convt.bias"] = (torch.randn(channels//2)*0.025).numpy()
            channels //= 2
            for residual in range(3):
                unit = prefix+f"res.{residual}."
                snake(unit+"snake1.alpha", channels)
                snake(unit+"snake2.alpha", channels)
                conv(unit+"conv1", channels, channels, 7)
                conv(unit+"conv2", channels, channels, 1)
        snake("snake_out.alpha", channels)
        conv("conv_out", 1, channels, 7)
        model = Vocoder(tensors, {"upsample_rates": [8, 8, 4, 2], "res_dilations": [1, 3, 9], "snake_eps": 1e-9})
        value = torch.randn(1, 2, 1, 4)
        with torch.no_grad():
            short = model(value, torch.ones(1, 1, 1, 4))
            padded = model(F.pad(value, (0, 4)), F.pad(torch.ones(1, 1, 1, 4), (0, 4)))
        torch.testing.assert_close(short, padded[:, :, :, :2048], rtol=1e-5, atol=1e-5)
        self.assertEqual(torch.count_nonzero(padded[:, :, :, 2048:]).item(), 0)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
