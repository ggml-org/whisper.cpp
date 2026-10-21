#!/usr/bin/env python3
"""Model-free checks for the Sortformer v2.1 Core ML diarizer exporter."""

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    print("SKIP: torch is not installed")
    raise SystemExit(0)


def stub_module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module


def ensure_optional_modules():
    try:
        import coremltools  # noqa: F401
    except ModuleNotFoundError:
        stub_module("coremltools")
    try:
        import gguf  # noqa: F401
    except ModuleNotFoundError:
        stub_module("gguf", GGMLQuantizationType=types.SimpleNamespace(F32="F32", F16="F16"))


ensure_optional_modules()
SCRIPT = Path(__file__).parents[1] / "scripts" / "export-sortformer-diarizer-coreml.py"
SPEC = importlib.util.spec_from_file_location("export_sortformer_diarizer_coreml", SCRIPT)
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)

SEED = 11
TOLERANCE = 1.0e-4
WIDTH = 16
HEADS = 2
LAYERS = 2
FEED_FORWARD = 32
KERNEL = 5
SPEAKER_WIDTH = 12
SPEAKER_HEADS = 3
SPEAKER_LAYERS = 2
SPEAKER_INNER = 20
SPEAKERS = 4
PRE_ENCODE_SCALE = 20.0
META = {
    "parakeet.model.type": "sortformer",
    "parakeet.encoder.d_model": WIDTH,
    "parakeet.encoder.n_layers": LAYERS,
    "parakeet.encoder.n_heads": HEADS,
    "parakeet.encoder.xscaling": True,
    "parakeet.encoder.pos_emb_max_len": 64,
    "parakeet.sortformer.tf_d_model": SPEAKER_WIDTH,
    "parakeet.sortformer.tf_n_layers": SPEAKER_LAYERS,
    "parakeet.sortformer.tf_n_heads": SPEAKER_HEADS,
    "parakeet.sortformer.num_spks": SPEAKERS,
}


class TensorFactory:
    def __init__(self):
        self.generator = torch.Generator().manual_seed(SEED)

    def __call__(self, *shape, offset=0.0):
        return offset + torch.randn(*shape, generator=self.generator) * 0.2


def norm(tensors, make, name, width):
    tensors[name + ".weight"] = make(width, offset=1.0)
    tensors[name + ".bias"] = make(width)


def linear(tensors, make, name, outputs, inputs):
    tensors[name + ".weight"] = make(outputs, inputs)
    tensors[name + ".bias"] = make(outputs)


def encoder_block(tensors, make, index):
    prefix = f"encoder.blk.{index}."
    for name in ("norm_ff1", "norm_attn", "norm_conv", "norm_ff2", "norm_out"):
        norm(tensors, make, prefix + name, WIDTH)
    for name in ("ff1", "ff2"):
        linear(tensors, make, prefix + name + ".linear1", FEED_FORWARD, WIDTH)
        linear(tensors, make, prefix + name + ".linear2", WIDTH, FEED_FORWARD)
    for name in ("q", "k", "v", "out"):
        linear(tensors, make, prefix + "attn." + name, WIDTH, WIDTH)
    tensors[prefix + "attn.pos.weight"] = make(WIDTH, WIDTH)
    tensors[prefix + "attn.pos_bias_u"] = make(HEADS, WIDTH // HEADS)
    tensors[prefix + "attn.pos_bias_v"] = make(HEADS, WIDTH // HEADS)
    linear(tensors, make, prefix + "conv.pw1", 2 * WIDTH, WIDTH)
    tensors[prefix + "conv.pw1.weight"] = tensors[prefix + "conv.pw1.weight"].unsqueeze(-1)
    tensors[prefix + "conv.dw.weight"] = make(WIDTH, 1, KERNEL)
    tensors[prefix + "conv.dw.bias"] = make(WIDTH)
    tensors[prefix + "conv.bn.scale"] = make(WIDTH, offset=1.0)
    tensors[prefix + "conv.bn.shift"] = make(WIDTH)
    linear(tensors, make, prefix + "conv.pw2", WIDTH, WIDTH)
    tensors[prefix + "conv.pw2.weight"] = tensors[prefix + "conv.pw2.weight"].unsqueeze(-1)


def speaker_block(tensors, make, index):
    prefix = f"sortformer.transformer.blk.{index}."
    for name in ("q", "k", "v", "out"):
        linear(tensors, make, prefix + "attn." + name, SPEAKER_WIDTH, SPEAKER_WIDTH)
    norm(tensors, make, prefix + "ln1", SPEAKER_WIDTH)
    norm(tensors, make, prefix + "ln2", SPEAKER_WIDTH)
    linear(tensors, make, prefix + "ffn.in", SPEAKER_INNER, SPEAKER_WIDTH)
    linear(tensors, make, prefix + "ffn.out", SPEAKER_WIDTH, SPEAKER_INNER)


def random_tensors():
    make = TensorFactory()
    tensors = {}
    for index in range(LAYERS):
        encoder_block(tensors, make, index)
    for index in range(SPEAKER_LAYERS):
        speaker_block(tensors, make, index)
    linear(tensors, make, "sortformer.encoder_proj", SPEAKER_WIDTH, WIDTH)
    linear(tensors, make, "sortformer.head.first_hidden_to_hidden", SPEAKER_WIDTH, SPEAKER_WIDTH)
    linear(tensors, make, "sortformer.head.single_hidden_to_spks", SPEAKERS, SPEAKER_WIDTH)
    return tensors


def reference_speaker_block(values, tensors, index):
    prefix = f"sortformer.transformer.blk.{index}."
    frames = values.size(0)

    def project(name, inputs):
        return inputs @ tensors[prefix + name + ".weight"].T + tensors[prefix + name + ".bias"]

    split = [project("attn." + name, values).reshape(frames, SPEAKER_HEADS, -1).transpose(0, 1)
             for name in ("q", "k", "v")]
    weights = torch.softmax(split[0] @ split[1].transpose(1, 2) / math.sqrt(SPEAKER_WIDTH // SPEAKER_HEADS), dim=-1)
    attended = (weights @ split[2]).transpose(0, 1).reshape(frames, SPEAKER_WIDTH)
    values = F.layer_norm(values + project("attn.out", attended), (SPEAKER_WIDTH,),
                          tensors[prefix + "ln1.weight"], tensors[prefix + "ln1.bias"])
    feed = project("ffn.out", F.relu(project("ffn.in", values)))
    return F.layer_norm(values + feed, (SPEAKER_WIDTH,), tensors[prefix + "ln2.weight"], tensors[prefix + "ln2.bias"])


def reference_diarize(pre_encode, tensors):
    module = EXPORTER.ENCODER_EXPORT.BypassEncoderModule(EXPORTER.REFERENCE, tensors, META).eval()
    with torch.inference_mode():
        encoded = module(pre_encode.T, torch.ones(pre_encode.size(0)))
    values = encoded @ tensors["sortformer.encoder_proj.weight"].T + tensors["sortformer.encoder_proj.bias"]
    for index in range(SPEAKER_LAYERS):
        values = reference_speaker_block(values, tensors, index)
    hidden = F.relu(F.relu(values) @ tensors["sortformer.head.first_hidden_to_hidden.weight"].T +
                    tensors["sortformer.head.first_hidden_to_hidden.bias"])
    logits = hidden @ tensors["sortformer.head.single_hidden_to_spks.weight"].T + tensors["sortformer.head.single_hidden_to_spks.bias"]
    return torch.sigmoid(logits)


def run_export_graph(pre_encode, capacity, tensors):
    config = EXPORTER.model_config(META)
    padded = torch.zeros(capacity, WIDTH)
    padded[:pre_encode.size(0)] = pre_encode
    mask = torch.zeros(capacity)
    mask[:pre_encode.size(0)] = 1.0
    model = EXPORTER.SortformerDiarizer(tensors, config, capacity).eval()
    with torch.inference_mode():
        return model(padded, mask)


class SortformerDiarizerExporterTest(unittest.TestCase):
    def setUp(self):
        self.tensors = random_tensors()
        self.pre_encode = PRE_ENCODE_SCALE * torch.randn(9, WIDTH, generator=torch.Generator().manual_seed(SEED + 1))

    def test_position_scores_match_relative_shift(self):
        frames, width = 6, 8
        generator = torch.Generator().manual_seed(SEED + 2)
        query = torch.randn(1, width, 1, frames, generator=generator)
        weight = torch.randn(width, width, generator=generator)
        table = EXPORTER.REFERENCE.sinusoidal_rel_pe(frames, width)
        center = table.size(1) // 2 + 1
        projected = F.linear(table[0, center - frames:center + frames - 1], weight)
        raw = (query[0, :, 0].T @ projected.T).reshape(1, 1, frames, -1)
        expected = EXPORTER.REFERENCE.rel_shift(raw)[..., :frames][0, 0].T
        config = {"head_width": width, "heads": 1, "width": width}
        projection = EXPORTER.head_projection(weight, 0, config)
        phases = (*EXPORTER.query_phases(frames, width), EXPORTER.key_phases(frames, width))
        scores = EXPORTER.position_scores(query, projection, phases)[0, :, 0]
        torch.testing.assert_close(scores, expected, atol=TOLERANCE, rtol=0)

    def test_export_graph_matches_reference(self):
        expected = reference_diarize(self.pre_encode, self.tensors)
        probabilities = run_export_graph(self.pre_encode, self.pre_encode.size(0), self.tensors)
        torch.testing.assert_close(probabilities, expected, atol=TOLERANCE, rtol=0)

    def test_padding_is_exact(self):
        frames = self.pre_encode.size(0)
        expected = run_export_graph(self.pre_encode, frames, self.tensors)
        padded = run_export_graph(self.pre_encode, frames + 6, self.tensors)
        torch.testing.assert_close(padded[:frames], expected, atol=TOLERANCE, rtol=0)

    def test_model_tensors_drop_frontend(self):
        tensors = {"encoder.subsampling.conv0.weight": torch.zeros(1), "preproc.mel_filterbank": torch.zeros(1),
                   "encoder.blk.0.norm_out.weight": torch.zeros(1), "sortformer.encoder_proj.weight": torch.zeros(1)}
        self.assertEqual(sorted(EXPORTER.model_tensors(tensors)),
                         ["encoder.blk.0.norm_out.weight", "sortformer.encoder_proj.weight"])


if __name__ == "__main__":
    unittest.main()
