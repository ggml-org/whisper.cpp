#!/usr/bin/env python3
"""Model-free tests and tiny random-weight HF capture; no checkpoint or ANE load.

Set PARLER_REQUIRE_HF_CAPTURE=1 to fail instead of skipping missing dependencies.
"""
import contextlib
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "parler"


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V = load("validate-decoder-coreml")
F = load("prepare-decoder-fixtures")


class MetricsTest(unittest.TestCase):
    def test_identity_and_additive_logit_shift(self):
        p = np.array([1.0, -2, 8, 0])
        result = V.row_metrics(p, p + 10000)
        self.assertTrue(result["valid"])
        self.assertEqual(result["argmax_agreement"], 1)
        self.assertEqual(result["top50_overlap"], 1)
        self.assertLess(result["kl"], 1e-12)

    def test_distribution_mismatch_even_with_same_argmax_and_top50(self):
        result = V.row_metrics([10, 0, 0], [.01, 0, 0])
        self.assertEqual(result["argmax_agreement"], 1)
        self.assertEqual(result["top50_overlap"], 1)
        self.assertGreater(result["kl"], 1.0)

    def test_known_kl_direction(self):
        p, q = np.array([.9, .1]), np.array([.5, .5])
        forward = V.row_metrics(np.log(p), np.log(q))["kl"]
        reverse = V.row_metrics(np.log(q), np.log(p))["kl"]
        self.assertAlmostEqual(forward, float(np.sum(p * np.log(p / q))), places=13)
        self.assertNotAlmostEqual(forward, reverse, places=3)

    def test_extreme_logits_stable(self):
        result = V.row_metrics([1e6, -1e6, 0], [-1e6, 1e6, 0])
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["kl"], 2e6)
        self.assertEqual(result["argmax_agreement"], 0)

    def test_negative_infinity_support(self):
        result = V.row_metrics([0, -math.inf], [0, -math.inf])
        self.assertEqual(result["kl"], 0)
        self.assertEqual(result["max_abs_logit_error"], 0)
        self.assertTrue(result["valid"])

    def test_missing_candidate_support_not_clipped(self):
        result = V.row_metrics([0, 0], [0, -math.inf])
        self.assertTrue(result["valid"])
        self.assertTrue(math.isinf(result["kl"]))

    def test_reference_zero_probability_does_not_produce_nan(self):
        result = V.row_metrics([0, -math.inf], [0, 0])
        self.assertAlmostEqual(result["kl"], math.log(2))

    def test_nan_positive_inf_and_empty_support_fail_closed(self):
        for bad in ([math.nan, 1], [math.inf, 0], [-math.inf, -math.inf]):
            for p, q in ((bad, [0, 1]), ([0, 1], bad)):
                with self.subTest(p=p, q=q):
                    result = V.row_metrics(p, q)
                    self.assertFalse(result["valid"])
                    self.assertEqual(result["argmax_agreement"], 0)
                    self.assertTrue(math.isinf(result["kl"]))

    def test_top50_overlap_and_ties(self):
        p = np.arange(100, dtype=float)
        result = V.row_metrics(p, -p)
        self.assertEqual(result["top50_overlap"], 0)
        self.assertEqual(V.row_metrics(np.zeros(100), np.zeros(100))["top50_overlap"], 1)
        q = p.copy()
        q[0] = 1000
        self.assertAlmostEqual(V.row_metrics(p, q)["top50_overlap"], 49 / 50)

    def test_shape_validation(self):
        for p, q in (([], []), ([0], [0, 1]), ([[0]], [[0]])):
            with self.assertRaises(ValueError):
                V.row_metrics(p, q)

    def test_infinite_tail_p95_never_nan(self):
        rows = [V.row_metrics([0, 0], [0, -math.inf])]
        result = V.summarize(rows)
        self.assertTrue(math.isinf(result["kl_mean"]))
        self.assertTrue(math.isinf(result["kl_p95"]))
        self.assertEqual(result["infinite_kl_rows"], 1)
        encoded = json.dumps(V.json_safe(result), allow_nan=False)
        self.assertIn('"+inf"', encoded)

    def test_invalid_rows_not_omitted_from_aggregate(self):
        result = V.summarize([V.row_metrics([0, 1], [0, 1]), V.row_metrics([0, 1], [math.nan, 0])])
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["invalid_rows"], 1)
        self.assertEqual(result["argmax_agreement"], .5)


class ClassificationTest(unittest.TestCase):
    CFG = dict(bos=11, eos=10, pad=10, codebook_size=10, max_length=8)

    def test_delay_eos_padding_and_live_slots(self):
        tokens = np.array([[11, 2, 10, 10, 10, 10, 10, 10],
                           [11, 4, 3, 5, 6, 7, 8, 10],
                           [11, 4, 5, 3, 4, 5, 6, 7]])
        classes = V.classify_slots(tokens, self.CFG, 7)
        self.assertEqual(classes[:, 0].tolist(), ["audio", "eos", "padding", "padding", "padding", "padding", "padding"])
        self.assertEqual(classes[:, 1].tolist(), ["delay", "audio", "audio", "audio", "audio", "audio", "padding"])
        self.assertEqual(classes[:, 2].tolist(), ["delay", "delay", "audio", "audio", "audio", "audio", "audio"])

    def test_forced_delay_raw_eos_does_not_end_live_slots(self):
        tokens = np.array([[11, 1, 2, 3], [11, 10, 2, 3]])
        classes = V.classify_slots(tokens, self.CFG, 3)
        self.assertEqual(classes[:, 1].tolist(), ["delay", "audio", "audio"])

    def test_short_generation_has_no_forced_pattern(self):
        tokens = np.array([[11, 1, 2], [11, 1, 2], [11, 1, 11]])
        classes = V.classify_slots(tokens, dict(self.CFG, max_length=3), 2)
        self.assertEqual(classes[0].tolist(), ["audio"] * 3)
        self.assertEqual(classes[1, 2], "special")

    def test_metrics_report_each_step_codebook_and_class(self):
        reference = np.zeros((2, 2, 4))
        candidate = reference.copy()
        candidate[1, 1, 3] = 10
        result = V.compare_case(reference, candidate, np.array([["delay", "audio"], ["eos", "audio"]]))
        self.assertEqual(result["by_class"]["audio"]["argmax_agreement"], .5)
        self.assertEqual(result["by_codebook"]["1"]["argmax_agreement"], .5)
        self.assertEqual(result["steps"][1]["codebooks"][1]["slot_class"], "audio")


ADAPTER = '''import numpy as np
class Runner:
    def __init__(self):
        self.state = 0
    def start_case(self, case, arrays):
        self.state = 0
    def step(self, index, frame):
        self.buffer = np.zeros((len(frame), 4), np.float32)
        self.pending = int(frame.sum()) + self.state
        return self.buffer
    def synchronize(self):
        if hasattr(self, "pending"):
            self.state = self.pending
            self.buffer[:, self.state % 4] = 3
            del self.pending
    def finish_case(self):
        pass
def create_runner(manifest):
    return Runner()
'''


class HarnessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "manifest.json"
        cases = []
        for cid, frame in (("one", [1, 2, 3]), ("two", [2, 3, 2])):
            frames = np.array(frame, dtype=np.int64)[:, None]
            ref = np.zeros((3, 1, 4), np.float32)
            for s, tok in enumerate(np.cumsum(frame) % 4):
                ref[s, 0, tok] = 3
            files, hashes = F.save_arrays(self.root, cid, {"decoder_inputs": frames,
                "delayed_tokens": np.array([[5, 1, 2, 3]]), "reference_logits": ref,
                "prompt_ids": np.array([[1]]), "cross_states": np.ones((1, 1, 2))})
            cases.append({"id": cid, "files": files, "sha256": hashes})
        V.write_json(self.manifest, {"schema_version": 1, "cases": cases,
              "token_config": dict(bos=5, eos=4, pad=4, codebook_size=4, max_length=8)})
        self.adapter = self.root / "adapter.py"
        self.adapter.write_text(ADAPTER)

    def tearDown(self):
        self.tmp.cleanup()

    def run_and_evaluate(self, adapter=None, repeats=3):
        output = self.root / "run"
        result = V.run_adapter(self.manifest, adapter or self.adapter, output, repeats=repeats, warmup=1)
        manifest, root, digest = V.load_manifest(self.manifest)
        metrics = V.evaluate_run(manifest, root, digest, output / "run.json")
        return result, metrics

    def test_adapter_sync_materializes_logits_and_reuses_reset_state(self):
        run, metrics = self.run_and_evaluate()
        for case in metrics.values():
            self.assertEqual(case["all"]["argmax_agreement"], 1)
            self.assertEqual(case["repeated_logit_mismatches"], 0)
            self.assertEqual(case["inclusive_cases"]["count"], 3)
        self.assertGreaterEqual(run["load_ms"], 0)
        for case in run["cases"]:
            for timing in case["timings"]:
                self.assertGreaterEqual(timing["total_ms"], sum(timing["step_ms"]))
        self.assertFalse(V.promotion_gate(metrics, {})["eligible_for_further_audio_validation"])

    def test_state_leak_detected_across_cases_and_repetitions(self):
        self.adapter.write_text(ADAPTER.replace("self.state = 0\n    def step", "pass\n    def step"))
        _, metrics = self.run_and_evaluate()
        self.assertTrue(any(c["repeated_logit_mismatches"] > 0 for c in metrics.values()))
        self.assertFalse(V.promotion_gate(metrics, {"fp16": metrics, "q8": metrics})["eligible_for_further_audio_validation"])

    def test_one_repeat_does_not_establish_inclusive_speed(self):
        _, metrics = self.run_and_evaluate(repeats=1)
        self.assertTrue(all(c["inclusive_cases"] is None for c in metrics.values()))

    def test_mutated_fixture_or_candidate_rejected(self):
        self.run_and_evaluate()
        manifest, root, digest = V.load_manifest(self.manifest)
        file = self.root / "run" / "one_logits.npy"
        np.save(file, np.zeros((3, 1, 4)))
        with self.assertRaisesRegex(ValueError, "checksum"):
            V.evaluate_run(manifest, root, digest, self.root / "run" / "run.json")
        np.save(self.root / "one_decoder_inputs.npy", np.zeros((3, 1), dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "checksum"):
            V.load_manifest(self.manifest)

    def test_different_history_digest_rejected(self):
        self.run_and_evaluate()
        manifest, root, _ = V.load_manifest(self.manifest)
        with self.assertRaisesRegex(ValueError, "exact fixtures"):
            V.evaluate_run(manifest, root, "different", self.root / "run" / "run.json")

    def test_strict_promotion_gate_and_no_audio_approval(self):
        _, metrics = self.run_and_evaluate()
        baselines = {name: copy.deepcopy(metrics) for name in ("fp16", "q8")}
        for result in metrics.values():
            result["inclusive_cases"] = dict(mean_ms=1, p95_ms=2)
        for baseline in baselines.values():
            for result in baseline.values():
                result["inclusive_cases"] = dict(mean_ms=2, p95_ms=3)
        gate = V.promotion_gate(metrics, baselines, baselines["q8"])
        self.assertTrue(gate["eligible_for_further_audio_validation"])
        self.assertFalse(gate["default_enable_approved"])
        # Distribution degradation must fail even if argmax remains perfect.
        metrics["one"]["by_class"]["audio"]["kl_mean"] = .001
        gate = V.promotion_gate(metrics, baselines, baselines["q8"])
        self.assertFalse(gate["eligible_for_further_audio_validation"])
        self.assertTrue(any("kl_mean" in reason for reason in gate["reasons"]))

    def test_refuse_overwrite_or_missing_synchronization(self):
        V.run_adapter(self.manifest, self.adapter, self.root / "run", repeats=1, warmup=0)
        with self.assertRaises(FileExistsError):
            V.run_adapter(self.manifest, self.adapter, self.root / "run")
        self.adapter.write_text("def create_runner(manifest): return object()")
        with self.assertRaisesRegex(ValueError, "start_case"):
            V.run_adapter(self.manifest, self.adapter, self.root / "other")

    def test_prepare_hook_recorded_separately_and_close_runs(self):
        marker = self.root / "closed"
        extra = f'''
    def prepare_case(self, case, arrays):
        self.prepared = case["id"]
    def close(self):
        from pathlib import Path
        Path({str(marker)!r}).write_text("closed")
    def metadata(self):
        return {{"separate_t5": True}}
'''
        source = ADAPTER.replace("def create_runner(manifest):", extra + "\ndef create_runner(manifest):")
        source = source.replace("self.state = 0\n    def step", "self.state = 0\n        assert self.prepared == case['id']\n    def step")
        self.adapter.write_text(source)
        run, _ = self.run_and_evaluate()
        self.assertTrue(run["has_separate_preparation"])
        self.assertTrue(run["runner_metadata"]["separate_t5"])
        self.assertTrue(marker.exists())
        self.assertTrue(all(c["excluded_preparation"]["count"] == 3 for c in run["cases"]))

    def test_adapter_closed_on_failed_step(self):
        marker = self.root / "closed"
        self.adapter.write_text(f'''
class Runner:
    def start_case(self, *args): pass
    def step(self, *args): raise RuntimeError("deliberate failure")
    def synchronize(self): pass
    def finish_case(self): pass
    def close(self):
        from pathlib import Path
        Path({str(marker)!r}).write_text("closed")
def create_runner(manifest): return Runner()
''')
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            V.run_adapter(self.manifest, self.adapter, self.root / "failed")
        self.assertTrue(marker.exists())
        self.assertFalse((self.root / "failed/run.json").exists())

    def test_invalid_timing_cannot_pass(self):
        for values in ([math.nan], [math.inf], [-1]):
            with self.assertRaises(ValueError):
                V.time_summary(values)


class FixturePlanningTest(unittest.TestCase):
    def test_cases_accept_legacy_meta_and_indic_list(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "cases.json"
            expected = [{"name": "case0", "prompt": "नमस्ते", "description": "Warm voice"}]
            for obj in (expected, {"cases": expected}):
                file.write_text(json.dumps(obj))
                self.assertEqual(F.read_cases(file), expected)
            file.write_text(json.dumps(expected * 2))
            with self.assertRaises(ValueError):
                F.read_cases(file)
            expected[0]["name"] = "../case0"
            file.write_text(json.dumps(expected))
            with self.assertRaises(ValueError):
                F.read_cases(file)

    def test_plan_only_does_not_load_hf_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"not actually loaded")
            cases = root / "cases.json"
            cases.write_text(json.dumps([dict(name="c", prompt="hello", description="warm")]))
            args = types.SimpleNamespace(model=root, cases_json=cases, out=root / "out", seed=42,
                threads=2, max_length=256, temperature=1.0, top_k=50, top_p=1.0, plan_only=True)
            old = F.load_hf
            F.load_hf = lambda *a, **k: self.fail("HF must not be loaded by planning")
            try:
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    F.sample_fixtures(args)
                self.assertFalse(args.out.exists())
                self.assertFalse(json.loads(stdout.getvalue())["model"]["weight_hashes_included"])
            finally:
                F.load_hf = old

    def test_legacy_manifest_replays_original_not_final_delay_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"plan only")
            meta = {"config": {"dec": dict(bos=11, eos=10, pad=10),
                    "dac": dict(codebook_size=10), "gen": dict(max_length=100)},
                    "cases": [dict(name="c", prompt="hello", description="warm")]}
            (root / "meta.json").write_text(json.dumps(meta))
            history = np.array([[11, 1, 2, 3, 4, 5, 6, 10], [11, 4, 3, 2, 1, 0, 9, 8],
                                [11, 7, 6, 5, 4, 3, 2, 1]])
            for name, arr in {"greedy_delayed": history, "prompt_ids": [[1]], "desc_ids": [[2, 1]],
                              "cross_states": np.ones((1, 2, 4)), "prompt_embeds": np.ones((1, 1, 4)),
                              "step_logits": np.ones((2, 3, 12))}.items():
                np.save(root / f"c_{name}.npy", arr)
            args = types.SimpleNamespace(model=root, reference_dir=root, out=root / "out", steps=5)
            F.legacy_manifest(args)
            manifest, out, _ = V.load_manifest(args.out / "manifest.json")
            arrays = V.arrays_for(manifest["cases"][0], out)
            np.testing.assert_array_equal(arrays["decoder_inputs"][:3], [[11, 11, 11], [1, 11, 11], [2, 3, 11]])
            np.testing.assert_array_equal(arrays["decoder_inputs"][-1], history[:, 4])
            self.assertNotIn("reference_logits", arrays)
            self.assertEqual(arrays["anchor_logits"].shape[0], 2)
            np.testing.assert_array_equal(np.load(root / "c_greedy_delayed.npy"), history)
            # 8-column capture with 3 codebooks: gen max_length=100 is not the
            # bound; the captured length is, so step 5 is the last scored column
            # before the forced-PAD tail (columns >= 8-3+1+k).
            self.assertEqual(manifest["cases"][0]["token_config"]["max_length"], 8)
            cfg = dict(manifest["token_config"], **manifest["cases"][0]["token_config"])
            classes = V.classify_slots(arrays["delayed_tokens"], cfg, 5)
            self.assertNotIn("padding", classes)
            self.assertEqual(classes[-1].tolist(), ["audio"] * 3)
            forced = V.classify_slots(history, cfg, 7)
            self.assertEqual(forced[5:].T.tolist(), [["padding", "padding"], ["audio", "padding"],
                                                     ["audio", "audio"]])
            args.steps, args.out = 6, root / "past-forced-pad"
            with self.assertRaisesRegex(ValueError, "at most 5 steps before the forced-PAD"):
                F.legacy_manifest(args)
            self.assertFalse(args.out.exists())
            args.steps, args.out = 2, root / "short-existing"
            F.legacy_manifest(args)
            short, short_root, _ = V.load_manifest(args.out / "manifest.json")
            self.assertEqual(short["kind"], "existing-hf-fp32-raw-reference")
            short_arrays = V.arrays_for(short["cases"][0], short_root)
            np.testing.assert_array_equal(short_arrays["reference_logits"], short_arrays["anchor_logits"])


REQUIRE_HF_CAPTURE = os.environ.get("PARLER_REQUIRE_HF_CAPTURE") == "1"

try:
    import torch
except ImportError:
    if REQUIRE_HF_CAPTURE:
        raise
    torch = None


@unittest.skipIf(torch is None, "tiny capture test requires torch")
class CaptureTest(unittest.TestCase):
    def test_tiny_real_hf_replay_matches_uncached_prefix_and_resets(self):
        if importlib.util.find_spec("parler_tts") is None:
            if REQUIRE_HF_CAPTURE:
                self.fail("PARLER_REQUIRE_HF_CAPTURE=1 requires parler_tts")
            self.skipTest("optional tiny HF decoder test requires parler_tts")
        from parler_tts.configuration_parler_tts import ParlerTTSDecoderConfig
        from parler_tts.modeling_parler_tts import ParlerTTSForCausalLM
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            torch.manual_seed(15)
            cfg = ParlerTTSDecoderConfig(hidden_size=16, ffn_dim=24, num_hidden_layers=2,
                num_attention_heads=2, num_codebooks=3, vocab_size=16, max_position_embeddings=64,
                dropout=0, attention_dropout=0, activation_dropout=0)
            cfg._attn_implementation = "eager"
            decoder = ParlerTTSForCausalLM(cfg).eval()
            prompt, cross = torch.randn(1, 2, 16), torch.randn(1, 3, 16)
            frames = torch.randint(0, 16, (6, 3)).numpy()
            cached = F.replay_logits(torch, decoder, frames, cross, prompt)
            repeated = F.replay_logits(torch, decoder, frames, cross, prompt)
            np.testing.assert_array_equal(cached, repeated)
            with torch.inference_mode():
                for i in range(len(frames)):
                    full = decoder(input_ids=torch.from_numpy(frames[:i + 1].T.copy()),
                        encoder_hidden_states=cross, prompt_hidden_states=prompt, use_cache=False, return_dict=True)
                    np.testing.assert_allclose(cached[i], full.logits[:, -1, :].numpy(), atol=1e-6)
        finally:
            torch.set_num_threads(threads)

    def test_exact_sampled_codes_and_pre_processor_logits(self):
        class Decoder(torch.nn.Module):
            num_codebooks = 2

            def forward(self, input_ids, **kwargs):
                return types.SimpleNamespace(logits=torch.arange(12, dtype=torch.float32).reshape(2, 1, 6).clone())

        class Audio(torch.nn.Module):
            def decode(self, audio_codes):
                return types.SimpleNamespace(audio_values=audio_codes.float().sum(1))

        class Model:
            def __init__(self):
                self.decoder, self.audio_encoder = Decoder(), Audio()

            def _sample(self):
                for i in range(3):
                    kwargs = dict(input_ids=torch.ones((2, 1), dtype=torch.long),
                                  encoder_hidden_states=torch.ones((1, 2, 4)))
                    if i == 0:
                        kwargs["prompt_hidden_states"] = torch.ones((1, 1, 4))
                    output = self.decoder(**kwargs)
                    output.logits.fill_(-math.inf)  # processor mutates raw tensor
                return types.SimpleNamespace(sequences=torch.tensor([[5, 1, 2, 3], [5, 3, 2, 1]]))

            def generate(self, **kwargs):
                result = self._sample()
                # Exact DAC inputs differ from argmax and sampled delayed ids.
                codes = torch.tensor([[[1, 2], [3, 1]]])
                result.sequences = self.audio_encoder.decode(audio_codes=codes).audio_values
                return result

        model = Model()
        sample_before, decode_before = model._sample, model.audio_encoder.decode
        arrays = F.capture_sample(torch, model, {})
        np.testing.assert_array_equal(arrays["delayed_tokens"], [[5, 1, 2, 3], [5, 3, 2, 1]])
        np.testing.assert_array_equal(arrays["codes"], [[[1, 2], [3, 1]]])
        np.testing.assert_array_equal(arrays["waveform"], [4, 3])
        self.assertTrue(np.isfinite(arrays["reference_logits"]).all())
        self.assertEqual(arrays["reference_logits"].shape, (3, 2, 6))
        self.assertEqual(model._sample, sample_before)
        self.assertEqual(model.audio_encoder.decode, decode_before)
        self.assertFalse(model.decoder._forward_hooks)
        self.assertFalse(model.decoder._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
