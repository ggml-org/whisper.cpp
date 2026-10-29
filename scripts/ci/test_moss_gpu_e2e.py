"""Admission and intelligibility gates must not report false passes."""
import importlib.util
import io
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid
import wave

spec = importlib.util.spec_from_file_location("moss_e2e", Path(__file__).with_name("moss-gpu-e2e.py"))
e2e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2e)


class AdmissionTests(unittest.TestCase):
    def test_requires_consecutive_headroom_samples(self):
        memory = [{"uuid": "GPU-a", "total_mib": 20475, "free_mib": free}
                  for free in [17067, 20000, 17000, 20000, 20000]]
        with patch.object(e2e, "gpu_memory", side_effect=memory) as query, \
             patch.object(e2e.time, "monotonic", return_value=0), \
             patch.object(e2e.time, "sleep"):
            e2e.wait_for_gpu(100, io.StringIO(), "GPU-a")
        self.assertEqual(query.call_count, 5)

    def test_busy_gpu_times_out_before_inference(self):
        with patch.object(e2e, "gpu_memory", return_value={
                "uuid": "GPU-a", "total_mib": 20475, "free_mib": 17067}), \
             patch.object(e2e.time, "monotonic", side_effect=[0, 10]), \
             patch.object(e2e.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "inference was not started"):
                e2e.wait_for_gpu(5, io.StringIO(), "GPU-a")

    def test_lock_excludes_other_process_and_releases_after_failure(self):
        gpu = "GPU-" + str(uuid.uuid4())
        path = Path("/tmp") / f"qvac-moss-e2e-{gpu}.lock"
        probe = """import fcntl, sys
with open(sys.argv[1]) as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(7)
"""
        try:
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(e2e, "gpu_memory", return_value={"uuid": gpu}), \
                 patch.object(e2e, "wait_for_gpu"):
                with self.assertRaisesRegex(RuntimeError, "synthesis failed"):
                    with e2e.gpu_lease(Path(directory), "test"):
                        self.assertEqual(subprocess.run([sys.executable, "-c", probe, str(path)]).returncode, 7)
                        raise RuntimeError("synthesis failed")
                self.assertEqual(subprocess.run([sys.executable, "-c", probe, str(path)]).returncode, 0)
        finally:
            path.unlink(missing_ok=True)


class RunnerTests(unittest.TestCase):
    def test_gpu_invocation_requires_the_selected_backend(self):
        for backend, selected, other in (("cuda", "CUDA0", "Vulkan0"),
                                         ("vulkan", "Vulkan0", "CUDA0"),
                                         ("webgpu", "WebGPU: WebGPU", "Vulkan0")):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runner = e2e.CaseRunner(backend, root, root, root)
                for reported, accepted in (("CPU", False), (other, False), (selected, True)):
                    def execute(args, **kwargs):
                        self.assertEqual(kwargs["env"]["TTS_CPP_GPU_BACKEND"], backend)
                        kwargs["stdout"].write(f"[moss-cli] backend: {reported}\n")
                    with patch.object(e2e, "gpu_lease") as lease, \
                         patch.object(e2e, "gpu_snapshot"), \
                         patch.object(e2e.subprocess, "run", side_effect=execute):
                        if accepted:
                            runner.invoke("test", ["moss-cli"], wav=False)
                        else:
                            with self.assertRaisesRegex(RuntimeError, "did not select"):
                                runner.invoke("test", ["moss-cli"], wav=False)
                        lease.assert_called_once_with(root, "test")

    def test_inference_oom_remains_failure_and_captures_after_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = e2e.CaseRunner("cuda", root, root, root)
            with patch.object(e2e, "gpu_lease"), \
                 patch.object(e2e, "gpu_snapshot") as snapshot, \
                 patch.object(e2e.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "moss-cli")), \
                 patch.object(sys, "stderr", io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.invoke("oom", ["moss-cli"], wav=False)
                self.assertEqual([call.args for call in snapshot.call_args_list],
                                 [(root, "oom", "before"), (root, "oom", "after")])
                self.assertNotIn("oom", runner.checks)

    def test_family_flows_preserve_streaming_and_dialogue(self):
        root = Path("/test")
        for family in ("tts", "ttsd", "sfx", "speech", "transcribe"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / "transcript.json").write_text(json.dumps({"segments": [
                    {"speaker": "S0", "start": 0, "end": 1, "text": "ask your country"}]}))
                runner = e2e.CaseRunner("vulkan", root, root, output)
                def invoke(name, args, **kwargs):
                    runner.checks[name] = {}
                with patch.object(runner, "invoke", side_effect=invoke) as calls, \
                     patch.object(e2e, "download", side_effect=lambda name, *_: name), \
                     patch.object(e2e, "reference"), \
                     patch.object(e2e, "audio_stats", return_value={}):
                    e2e.FAMILY_RUNNERS[family](runner, "model.gguf")
                by_name = {call.args[0]: call for call in calls.call_args_list}
                if family == "tts":
                    self.assertEqual(set(by_name), {"batch", "stream", "clone", "cpu-reference",
                                     "stream-agreement", "cpu-stream-agreement"})
                    self.assertNotIn("--gpu", by_name["cpu-reference"].args[1])
                    self.assertTrue(by_name["cpu-reference"].kwargs["cpu"])
                    self.assertEqual(by_name["stream-agreement"].args[1][0],
                                     "/test/engines/tts/test-moss-tts-stream-e2e")
                    self.assertEqual(by_name["cpu-stream-agreement"].args[1][1], "cpu")
                    self.assertTrue(by_name["cpu-stream-agreement"].kwargs["cpu"])
                elif family == "ttsd":
                    self.assertEqual(by_name["dialogue"].args[1].count("--dialogue-ref"), 2)
                elif family == "speech":
                    self.assertEqual(set(by_name), {"codec-agreement", "reply"})
                    self.assertIn("reply-repeat", runner.checks)
                elif family == "transcribe":
                    self.assertEqual(runner.checks["transcribe"]["segments"], 1)
                else:
                    self.assertEqual(set(by_name), {"sound"})


class CheckpointTests(unittest.TestCase):
    def test_quantized_checkpoint_is_hashed_before_source_removal(self):
        for case in ("tts-q8_0", "ttsd-q8_0"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.gguf"
                source.write_bytes(b"GGUF" + b"source" * 100)
                generated = b"GGUFquantized"
                def quantize(args, **kwargs):
                    self.assertEqual(args[-3:], ["q8_0", "--name-filter", "blk."])
                    self.assertEqual(kwargs["timeout"], e2e.QUANTIZATION_TIMEOUT_SECONDS)
                    Path(args[3]).write_bytes(generated)
                with patch.object(e2e, "download", return_value=str(source)), \
                     patch.object(e2e.subprocess, "run", side_effect=quantize):
                    result = Path(e2e.prepare_checkpoint(case, root, root))
                self.assertEqual(result.read_bytes(), generated)
                self.assertFalse(source.exists())
                self.assertEqual((root / "models.sha256").read_text(),
                                 f"{hashlib.sha256(generated).hexdigest()}  generated/{result.name}\n")

    def test_invalid_or_failed_quantization_preserves_source(self):
        for result in (b"bad", b"GGUF" + b"x" * 100, None):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.gguf"
                source.write_bytes(b"GGUFsource")
                def quantize(args, **kwargs):
                    if result is None:
                        raise subprocess.CalledProcessError(1, args)
                    Path(args[3]).write_bytes(result)
                with patch.object(e2e, "download", return_value=str(source)), \
                     patch.object(e2e.subprocess, "run", side_effect=quantize):
                    with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                        e2e.prepare_checkpoint("tts-q8_0", root, root)
                self.assertEqual(source.read_bytes(), b"GGUFsource")
                self.assertFalse((root / "models.sha256").exists())


class TtsOutputScoringTests(unittest.TestCase):
    # Explicit artifact contract, independent of the scorer's enumeration loop.
    SHORT_OUTPUTS = (
        "batch", "stream", "clone", "cpu-reference",
        "tts-c7-batch", "tts-c7-stream", "tts-c7-after-callback-cancel", "tts-c7-after-explicit-cancel",
        "tts-c25-batch", "tts-c25-stream", "tts-c25-after-callback-cancel", "tts-c25-after-explicit-cancel",
        "unconditioned/tts-c7-batch", "unconditioned/tts-c7-stream",
        "unconditioned/tts-c7-after-callback-cancel", "unconditioned/tts-c7-after-explicit-cancel",
        "unconditioned/tts-c25-batch", "unconditioned/tts-c25-stream",
        "unconditioned/tts-c25-after-callback-cancel", "unconditioned/tts-c25-after-explicit-cancel",
        "cpu/tts-c7-batch", "cpu/tts-c7-stream",
        "cpu/tts-c7-after-callback-cancel", "cpu/tts-c7-after-explicit-cancel",
        "cpu/tts-c25-batch", "cpu/tts-c25-stream",
        "cpu/tts-c25-after-callback-cancel", "cpu/tts-c25-after-explicit-cancel",
        "cpu/unconditioned/tts-c7-batch", "cpu/unconditioned/tts-c7-stream",
        "cpu/unconditioned/tts-c7-after-callback-cancel", "cpu/unconditioned/tts-c7-after-explicit-cancel",
        "cpu/unconditioned/tts-c25-batch", "cpu/unconditioned/tts-c25-stream",
        "cpu/unconditioned/tts-c25-after-callback-cancel", "cpu/unconditioned/tts-c25-after-explicit-cancel",
    )
    LONG_OUTPUTS = (
        "tts-long-batch", "tts-long-stream",
        "unconditioned/tts-long-batch", "unconditioned/tts-long-stream",
        "cpu/tts-long-batch", "cpu/tts-long-stream",
        "cpu/unconditioned/tts-long-batch", "cpu/unconditioned/tts-long-stream",
    )

    def score_outputs(self, output, missing=None):
        expected = {name: "short" for name in self.SHORT_OUTPUTS}
        expected.update({name: "long" for name in self.LONG_OUTPUTS})
        self.assertEqual(len(expected), 44)
        for name in expected:
            if name != missing:
                audio = output / f"{name}.wav"
                audio.parent.mkdir(parents=True, exist_ok=True)
                audio.touch()
        for length, text in (("short", e2e.TTS_TEXT), ("long", e2e.TTS_LONG_TEXT)):
            (output / f"reference-{length}.txt").write_text(text)
        asr_binary, model = output / "whisper-cli", output / "asr-model.bin"
        prepare = Mock()
        prepare.prepare.return_value = model
        scored = []

        def score(args, **kwargs):
            def argument(flag):
                return Path(args[args.index(flag) + 1])
            name = argument("--audio").relative_to(output).with_suffix("").as_posix()
            self.assertIn(name, expected)
            self.assertEqual(argument("--reference"), output / f"reference-{expected[name]}.txt")
            self.assertEqual(argument("--asr-binary"), asr_binary)
            self.assertEqual(argument("--asr-model"), model)
            self.assertEqual(kwargs, {"check": False, "timeout": e2e.ASR_TIMEOUT_SECONDS})
            argument("--json-out").write_text(json.dumps({
                "status": "ok", "wer": 0.0, "n_ref_words": 8,
            }))
            scored.append(name)
            return subprocess.CompletedProcess(args, 0)

        # Exercise the real gate, replacing only model acquisition and ASR.
        with patch.object(e2e.importlib.util, "spec_from_file_location"), \
             patch.object(e2e.importlib.util, "module_from_spec", return_value=prepare), \
             patch.object(e2e.subprocess, "run", side_effect=score):
            result = e2e.score_tts_outputs(asr_binary, output / "models", output)
        prepare.prepare.assert_called_once_with(e2e.ASR_SPEC, (output / "models").resolve())
        self.assertCountEqual(scored, [name for name in expected if name != missing])
        self.assertEqual(set(result["scores"]), set(expected))
        return result

    def test_scores_all_44_outputs_with_the_matching_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.score_outputs(Path(directory))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["quality_status"], "passed")
        self.assertEqual(result["failed_outputs"], [])

    def test_missing_cpu_and_unconditioned_outputs_fail_the_gate(self):
        for missing in ("cpu/tts-c7-batch", "unconditioned/tts-long-stream",
                        "cpu/unconditioned/tts-c25-after-explicit-cancel"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                result = self.score_outputs(Path(directory), missing)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["quality_status"], "failed")
                self.assertEqual(result["failed_outputs"], [missing])
                self.assertEqual(result["scores"][missing], {"status": "missing", "wer": None})


class IntelligibilityTests(unittest.TestCase):
    def test_tts_quality_gate_covers_quantized_cases_on_both_backends(self):
        for case in ("tts-f16", "tts-q8_0"):
            for backend in ("cuda", "vulkan"):
                with self.subTest(case=case, backend=backend), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "output"
                    args = ["moss-gpu-e2e.py", "--case", case, "--backend", backend,
                            "--models", str(Path(directory) / "models"), "--output", str(output)]
                    quality = {"status": "failed", "failed_outputs": ["clone"]}
                    with patch.object(sys, "argv", args), \
                         patch.object(e2e, "run_case", return_value={}) as run, \
                         patch.object(e2e, "score_tts_outputs", return_value=quality) as score, \
                         patch.object(e2e.subprocess, "check_output", return_value="commit"), \
                         patch.object(sys, "stderr", io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            e2e.main()
                    self.assertEqual(raised.exception.code, 1)
                    self.assertEqual(run.call_args.args[:2], (case, backend))
                    score.assert_called_once()
                    self.assertEqual((output / "reference-short.txt").read_text().strip(), e2e.TTS_TEXT)
                    result = json.loads((output / "result.json").read_text())
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(result["backend"], backend)

    def test_wrong_text_blocks_every_output(self):
        bad = {"status": "ok", "wer": 1.0, "n_ref_words": 8}
        scores = {name: bad for name in ("batch", "stream", "cpu-reference", "clone",
                                         "unconditioned/tts-c7-after-callback-cancel",
                                         "cpu/unconditioned/tts-long-stream")}
        self.assertEqual(e2e.classify_tts_scores(scores), list(scores))
        self.assertEqual(e2e.classify_tts_scores({"batch": {**bad, "wer": 0.0}}), [])

    def test_wrong_or_empty_speech_fails_using_shared_word_error_scorer(self):
        spec = importlib.util.spec_from_file_location("wer", Path(__file__).parents[1] / "benchmarks/compute-wer.py")
        wer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wer)
        for transcript, expected in [(e2e.TTS_TEXT, True), ("No more...", False), ("", False)]:
            with self.subTest(transcript=transcript):
                score = {"status": "ok", **wer.compute_wer(transcript, e2e.TTS_TEXT)}
                self.assertEqual(e2e.intelligibility_passes(score), expected)

    def test_unavailable_missing_or_nonfinite_scores_cannot_pass(self):
        for score in [{}, {"status": "error", "wer": 0, "n_ref_words": 8},
                      {"status": "missing", "wer": None},
                      {"status": "ok", "wer": None, "n_ref_words": 8},
                      {"status": "ok", "wer": float("nan"), "n_ref_words": 8},
                      {"status": "ok", "wer": float("inf"), "n_ref_words": 8},
                      {"status": "ok", "wer": -0.1, "n_ref_words": 8},
                      {"status": "ok", "wer": 0, "n_ref_words": 0}]:
            with self.subTest(score=score):
                self.assertFalse(e2e.intelligibility_passes(score))
                self.assertEqual(e2e.classify_tts_scores({"batch": score}), ["batch"])


if __name__ == "__main__":
    unittest.main()
