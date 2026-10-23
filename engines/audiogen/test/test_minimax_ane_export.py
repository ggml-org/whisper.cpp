import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import coremltools as ct
import numpy as np
import torch
from torch.nn import functional as F


SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("minimax_exporter", SCRIPTS/"export-minimax-ane-validation.py")
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class ScaledProjection(torch.nn.Module):
    SCALE = 64.0

    def __init__(self):
        super().__init__()
        weight = torch.arange(-16, 16).reshape(4, 8, 1, 1)/1024.0
        self.register_buffer("weight", weight.half().float())

    def forward(self, hidden):
        return F.conv2d(hidden*self.SCALE, self.weight)/self.SCALE


class ProjectionConversionTests(unittest.TestCase):
    def converted_weight(self, disabled_passes):
        module = ScaledProjection().eval()
        value = torch.ones(1, 8, 1, 2)
        traced = torch.jit.trace(module, value)
        program = ct.convert(traced, convert_to="milinternal", inputs=[ct.TensorType(shape=value.shape)],
                             minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                             pass_pipeline=EXPORTER.conversion_pipeline(disabled_passes))
        convolutions = [operation for operation in program.functions["main"].operations if operation.op_type == "conv"]
        self.assertEqual(len(convolutions), 1)
        return module.weight.numpy(), convolutions[0].weight.val

    def test_disabling_scale_fusion_preserves_original_projection_weights(self):
        expected, actual = self.converted_weight((EXPORTER.CONV_SCALE_FUSION_PASS,))
        np.testing.assert_array_equal(actual, expected.astype(np.float16))

    def test_default_conversion_reproduces_weight_scale_folding(self):
        expected, actual = self.converted_weight(())
        np.testing.assert_array_equal(actual, (expected/ScaledProjection.SCALE).astype(np.float16))

    def test_control_does_not_change_the_global_default_pipeline(self):
        before = ct.PassPipeline.DEFAULT.passes.copy()
        pipeline = EXPORTER.conversion_pipeline((EXPORTER.CONV_SCALE_FUSION_PASS,))
        self.assertNotIn(EXPORTER.CONV_SCALE_FUSION_PASS, pipeline.passes)
        self.assertEqual(ct.PassPipeline.DEFAULT.passes, before)


class CompiledExportReuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.compiled = self.root/"vocoder.mlmodelc"
        self.compiled.mkdir()
        self.report = self.root/"vocoder.json"

    def write_report(self, precision):
        self.report.write_text(json.dumps({"disabled_conversion_passes": [], "vocoder_precision": precision}))

    def test_reuses_matching_vocoder_precision(self):
        self.write_report("fp32-activations")
        self.assertTrue(EXPORTER.reuse_compiled_export(self.compiled, self.report,
            {"disabled_conversion_passes": [], "vocoder_precision": "fp32-activations"}))

    def test_rejects_changed_vocoder_precision_in_both_directions(self):
        for previous, requested in [("all-fp16", "fp32-activations"), ("fp32-activations", "all-fp16")]:
            with self.subTest(previous=previous, requested=requested):
                self.write_report(previous)
                with self.assertRaisesRegex(ValueError, "different conversion settings"):
                    EXPORTER.export("vocoder", None, {}, [], self.root,
                        artifact_settings={"vocoder_precision": requested})

    def test_rejects_unknown_vocoder_precision(self):
        self.report.write_text(json.dumps({"disabled_conversion_passes": []}))
        with self.assertRaisesRegex(ValueError, "different conversion settings"):
            EXPORTER.reuse_compiled_export(self.compiled, self.report,
                {"disabled_conversion_passes": [], "vocoder_precision": "fp32-activations"})

    def test_preserves_conversion_pipeline_check(self):
        self.report.write_text(json.dumps({"disabled_conversion_passes": []}))
        with self.assertRaisesRegex(ValueError, "different conversion settings"):
            EXPORTER.reuse_compiled_export(self.compiled, self.report,
                {"disabled_conversion_passes": [EXPORTER.CONV_SCALE_FUSION_PASS]})

    def test_missing_compiled_artifact_is_not_reused(self):
        self.write_report("all-fp16")
        self.compiled.rmdir()
        self.assertFalse(EXPORTER.reuse_compiled_export(self.compiled, self.report,
            {"vocoder_precision": "all-fp16"}))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
