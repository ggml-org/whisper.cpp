#!/usr/bin/env python3
"""Model-free checks for the Nemotron 3 Diarization Core ML exporter."""

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
        quantization = types.SimpleNamespace(F32="F32", F16="F16")
        stub_module("gguf", GGMLQuantizationType=quantization)


ensure_optional_modules()
SCRIPT = Path(__file__).parents[1] / "scripts" / "export-nemotron-diarization-coreml.py"
SPEC = importlib.util.spec_from_file_location("export_nemotron_diarization_coreml", SCRIPT)
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)

SEED = 7
TOLERANCE = 1.0e-4
CONFIG = {
    "width": 16, "layers": 2, "heads": 2, "head_width": 8, "output_width": 6,
    "speakers": 3, "stack": 2, "mels": 4, "position_limit": 64, "rope_base": 10000.0,
}
FEED_FORWARD_WIDTH = 24
CONV_KERNEL = 3


def random_tensors():
    generator = torch.Generator().manual_seed(SEED)

    def tensor(*shape):
        return torch.randn(*shape, generator=generator) * 0.3

    width, output = CONFIG["width"], CONFIG["output_width"]
    tensors = {
        "encoder.pre_encode.proj.weight": tensor(width, CONFIG["mels"] * CONFIG["stack"]),
        "encoder.embed_norm.weight": 1.0 + tensor(width), "encoder.embed_norm.bias": tensor(width),
        "encoder.final_norm.weight": 1.0 + tensor(width), "encoder.final_norm.bias": tensor(width),
        "encoder_proj.weight": tensor(output, width), "encoder_proj.bias": tensor(output),
        "subpixel_upsample.weight": tensor(output * CONFIG["stack"], output, CONV_KERNEL),
        "subpixel_upsample.bias": tensor(output * CONFIG["stack"]),
        "head.first_hidden_to_hidden.weight": tensor(output, output),
        "head.first_hidden_to_hidden.bias": tensor(output),
        "head.single_hidden_to_spks.weight": tensor(CONFIG["speakers"], output),
        "head.single_hidden_to_spks.bias": tensor(CONFIG["speakers"]),
    }
    for index in range(CONFIG["layers"]):
        prefix = f"encoder.layers.{index}."
        tensors.update({
            prefix + "norm1.weight": 1.0 + tensor(width), prefix + "norm1.bias": tensor(width),
            prefix + "attn.w_qkv.weight": tensor(3 * width, width),
            prefix + "attn.out_proj.weight": tensor(width, width),
            prefix + "attn.out_proj.bias": tensor(width),
            prefix + "norm2.weight": 1.0 + tensor(width), prefix + "norm2.bias": tensor(width),
            prefix + "ffn.net.0.weight": tensor(FEED_FORWARD_WIDTH, width),
            prefix + "ffn.net.0.bias": tensor(FEED_FORWARD_WIDTH),
            prefix + "ffn.net.3.weight": tensor(width, FEED_FORWARD_WIDTH),
            prefix + "ffn.net.3.bias": tensor(width),
        })
    return tensors


def neox_rotate(heads, positions):
    half = CONFIG["head_width"] // 2
    exponents = torch.arange(half, dtype=torch.float64) * 2.0 / CONFIG["head_width"]
    angles = positions.double().unsqueeze(1) * torch.pow(torch.tensor(CONFIG["rope_base"]), -exponents)
    cosine, sine = torch.cos(angles).float(), torch.sin(angles).float()
    first, second = heads[..., :half], heads[..., half:]
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)


def reference_layer(values, tensors, index):
    prefix = f"encoder.layers.{index}."
    width, heads = CONFIG["width"], CONFIG["heads"]
    frames = values.size(0)
    normalized = F.layer_norm(values, (width,), tensors[prefix + "norm1.weight"], tensors[prefix + "norm1.bias"])
    qkv = normalized @ tensors[prefix + "attn.w_qkv.weight"].T
    split = [qkv[:, i * width:(i + 1) * width].reshape(frames, heads, -1).transpose(0, 1) for i in range(3)]
    positions = torch.arange(frames)
    query, key = neox_rotate(split[0], positions), neox_rotate(split[1], positions)
    weights = torch.softmax(query @ key.transpose(1, 2) / math.sqrt(CONFIG["head_width"]), dim=-1)
    merged = (weights @ split[2]).transpose(0, 1).reshape(frames, width)
    values = values + merged @ tensors[prefix + "attn.out_proj.weight"].T + tensors[prefix + "attn.out_proj.bias"]
    normalized = F.layer_norm(values, (width,), tensors[prefix + "norm2.weight"], tensors[prefix + "norm2.bias"])
    hidden = F.gelu(normalized @ tensors[prefix + "ffn.net.0.weight"].T + tensors[prefix + "ffn.net.0.bias"])
    return values + hidden @ tensors[prefix + "ffn.net.3.weight"].T + tensors[prefix + "ffn.net.3.bias"]


def reference_diarize(features, cache, tensors):
    embeddings = features @ tensors["encoder.pre_encode.proj.weight"].T
    values = torch.cat((cache, embeddings), dim=0)
    width = CONFIG["width"]
    values = F.layer_norm(values, (width,), tensors["encoder.embed_norm.weight"], tensors["encoder.embed_norm.bias"])
    for index in range(CONFIG["layers"]):
        values = reference_layer(values, tensors, index)
    values = F.layer_norm(values, (width,), tensors["encoder.final_norm.weight"], tensors["encoder.final_norm.bias"])
    projected = values @ tensors["encoder_proj.weight"].T + tensors["encoder_proj.bias"]
    convolved = F.conv1d(projected.T.unsqueeze(0), tensors["subpixel_upsample.weight"],
                         tensors["subpixel_upsample.bias"], padding=1)[0].T
    upsampled = convolved.reshape(-1, CONFIG["output_width"])
    hidden = F.relu(upsampled) @ tensors["head.first_hidden_to_hidden.weight"].T + tensors["head.first_hidden_to_hidden.bias"]
    logits = F.relu(hidden) @ tensors["head.single_hidden_to_spks.weight"].T + tensors["head.single_hidden_to_spks.bias"]
    return torch.sigmoid(logits), embeddings


def padded_inputs(features, cache, capacity):
    cache_rows, chunk_rows = cache.size(0), features.size(0)
    padded_features = torch.zeros(capacity, features.size(1))
    padded_features[cache_rows:cache_rows + chunk_rows] = features
    padded_cache = torch.zeros(capacity, cache.size(1))
    padded_cache[:cache_rows] = cache
    mask = torch.zeros(capacity)
    mask[:cache_rows + chunk_rows] = 1.0
    return padded_features, padded_cache, mask


def run_export_graph(features, cache, capacity, tensors):
    model = EXPORTER.NemotronDiarizer(tensors, CONFIG, capacity).eval()
    with torch.inference_mode():
        return model(*padded_inputs(features, cache, capacity))


class NemotronDiarizationExporterTest(unittest.TestCase):
    def setUp(self):
        self.tensors = random_tensors()
        generator = torch.Generator().manual_seed(SEED + 1)
        self.cache = torch.randn(5, CONFIG["width"], generator=generator)
        self.features = torch.randn(7, CONFIG["mels"] * CONFIG["stack"], generator=generator)

    def test_sidecar_name_strips_quantization(self):
        cases = {
            "models/Nemotron-3-Diarization.q8_0.gguf": "models/Nemotron-3-Diarization-diarizer-358.mlpackage",
            "Nemotron-3-Diarization.converted.q4_k.gguf": "Nemotron-3-Diarization.converted-diarizer-358.mlpackage",
            "models/model-nvfp4.gguf": "models/model-diarizer-358.mlpackage",
            "models/plain.gguf": "models/plain-diarizer-358.mlpackage",
        }
        for gguf_path, expected in cases.items():
            self.assertEqual(str(EXPORTER.sidecar_package(gguf_path, 358)), expected)

    def test_rotation_matrix_rotates_half(self):
        values = torch.arange(CONFIG["head_width"], dtype=torch.float32)
        half = CONFIG["head_width"] // 2
        expected = torch.cat((-values[half:], values[:half]))
        self.assertTrue(torch.equal(EXPORTER.rotate_half_matrix(CONFIG["head_width"]) @ values, expected))

    def test_export_graph_matches_row_major_reference(self):
        expected_probabilities, expected_embeddings = reference_diarize(self.features, self.cache, self.tensors)
        frames = self.cache.size(0) + self.features.size(0)
        probabilities, embeddings = run_export_graph(self.features, self.cache, frames, self.tensors)
        torch.testing.assert_close(probabilities, expected_probabilities, atol=TOLERANCE, rtol=0)
        torch.testing.assert_close(embeddings[self.cache.size(0):], expected_embeddings, atol=TOLERANCE, rtol=0)

    def test_padding_is_exact(self):
        frames = self.cache.size(0) + self.features.size(0)
        exact, _ = run_export_graph(self.features, self.cache, frames, self.tensors)
        padded, _ = run_export_graph(self.features, self.cache, frames + 9, self.tensors)
        torch.testing.assert_close(padded[:frames * CONFIG["stack"]], exact, atol=TOLERANCE, rtol=0)

    def test_offline_call_without_cache(self):
        empty = torch.zeros(0, CONFIG["width"])
        expected, _ = reference_diarize(self.features, empty, self.tensors)
        probabilities, _ = run_export_graph(self.features, empty, self.features.size(0) + 3, self.tensors)
        torch.testing.assert_close(probabilities[:expected.size(0)], expected, atol=TOLERANCE, rtol=0)

    def test_contract_rejects_other_architectures(self):
        meta = dict(EXPORTER.EXPECTED_CONTRACT, **{"sortformer.encoder.rotary_fraction": 1.0})
        EXPORTER.validate_contract(meta)
        with self.assertRaises(ValueError):
            EXPORTER.validate_contract(dict(meta, **{"sortformer.version": "v2.1"}))
        with self.assertRaises(ValueError):
            EXPORTER.validate_contract(dict(meta, **{"sortformer.encoder.rotary_fraction": 0.5}))


if __name__ == "__main__":
    unittest.main()
