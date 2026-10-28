from array import array
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "moss-validated-cloning.py"
SPEC = importlib.util.spec_from_file_location("validated_cloning", SCRIPT)
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
TARGET = "[S1] The train leaves this afternoon. [S2] Please bring your ticket."
TRANSCRIPT = "[0.1][S01] The train leaves this afternoon.[S02] Please bring your ticket."


def write_audio(path, value=1000):
    samples = array("h", [value] * 2400)
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        output.writeframes(samples.tobytes())


class ValidatedCloningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(
            moss_cli="moss-cli", asr_cli="moss-transcribe", backbone="backbone.gguf",
            encoder="encoder.gguf", decoder="decoder.gguf", asr_model="asr.gguf",
            reference=[("reference.wav", "An unrelated reference transcript.")],
            target=TARGET, output=str(self.root / "accepted.wav"), seed=5, attempts=3,
            threads=4, max_new_tokens=1024, timeout=30, max_error_rate=0.25, gpu=False, backend="cuda")
        self.seeds = []
        self.asr_calls = 0
        self.transcripts = {}
        self.frames = {}
        self.marker = True
        self.silent = False

    def fake_command(self, command, directory, name, timeout):
        if name == "synthesis":
            seed = int(command[command.index("--seed") + 1])
            self.seeds.append(seed)
            write_audio(directory / "audio.wav", 0 if self.silent else seed * 100)
            frames = self.frames.get(seed, 100)
            text = f"[moss-cli] wrote output: 8s @ 24000 Hz ({frames} frames, generate 100 ms)"
            (directory / "synthesis.log").write_text(text if self.marker else "unknown format")
        else:
            self.asr_calls += 1
            payload = self.transcripts.get(self.seeds[-1], {"text": TRANSCRIPT})
            if isinstance(payload, dict):
                payload = {"generated_tokens": 100, **payload}
            (directory / "transcript.json").write_text(json.dumps(payload))

    def run_guard(self):
        with patch.object(GUARD, "run_command", side_effect=self.fake_command):
            return GUARD.run_guard(self.args)

    def test_cpu_commands_do_not_request_a_gpu(self):
        synthesis = GUARD.synthesis_command(self.args, 7, self.root / "audio.wav")
        asr = GUARD.asr_command(self.args, self.root / "audio.wav", self.root / "text.json")
        self.assertNotIn("--gpu", synthesis)
        self.assertNotIn("--backend", synthesis)
        self.assertNotIn("--gpu", asr)
        self.assertNotIn("--backend", asr)

    def check_gpu_commands(self, backend, device):
        self.args.gpu = True
        self.args.backend = backend
        synthesis = GUARD.synthesis_command(self.args, 7, self.root / "audio.wav")
        asr = GUARD.asr_command(self.args, self.root / "audio.wav", self.root / "text.json")
        self.assertEqual(synthesis[-3:], ["--gpu", "--backend", device])
        self.assertEqual(asr[-3:], ["--gpu", "--backend", backend])

    def test_backend_argument_defaults_to_cuda_and_accepts_vulkan(self):
        required = ["--moss-cli", "moss-cli", "--asr-cli", "asr", "--backbone", "b",
                    "--encoder", "e", "--decoder", "d", "--asr-model", "a",
                    "--text-file", "text", "--output", "out", "--reference", "ref", "words"]
        self.assertEqual(GUARD.parse_args(required + ["--gpu"]).backend, "cuda")
        args = GUARD.parse_args(required + ["--gpu", "--backend", "vulkan"])
        self.assertTrue(args.gpu)
        self.assertEqual(args.backend, "vulkan")

    def test_cuda_commands_preserve_device_selection(self):
        self.check_gpu_commands("cuda", "CUDA0")

    def test_vulkan_commands_select_both_engines(self):
        self.check_gpu_commands("vulkan", "Vulkan0")

    def test_missing_last_two_dialogue_turns_retries_before_publishing(self):
        self.args.target = (
            "[S1] The train leaves at four in the afternoon. "
            "[S2] Thanks, I will bring my ticket. "
            "[S1] Meet me beside the large clock. "
            "[S2] All right, I will see you there.")
        self.transcripts[5] = {"text": (
            "The train leaves at four in the afternoon. Thanks, I will bring my ticket.")}
        self.transcripts[6] = {"text": self.args.target}
        status, report = self.run_guard()
        self.assertEqual(status, 0)
        self.assertEqual(self.seeds, [5, 6])
        self.assertEqual(report["attempts"][0]["reason"], "text_mismatch")
        self.assertGreater(report["attempts"][0]["ending_error_rate"], self.args.max_error_rate)
        accepted = Path(report["attempts"][1]["directory"]) / "audio.wav"
        self.assertEqual(Path(self.args.output).read_bytes(), accepted.read_bytes())

    def test_accepts_and_publishes_only_validated_audio(self):
        status, report = self.run_guard()
        self.assertEqual(status, 0)
        self.assertEqual(self.seeds, [5])
        accepted = Path(report["attempts"][0]["directory"]) / "audio.wav"
        self.assertEqual(Path(self.args.output).read_bytes(), accepted.read_bytes())
        self.assertEqual(json.loads((Path(report["evidence"]) / "report.json").read_text())["status"], "accepted")

    def test_retries_reference_repetition(self):
        self.transcripts[5] = {"text": "An unrelated reference transcript with many extra words. " + TRANSCRIPT}
        status, report = self.run_guard()
        self.assertEqual(status, 0)
        self.assertEqual(self.seeds, [5, 6])
        self.assertEqual(report["attempts"][0]["reason"], "text_mismatch")
        accepted = Path(report["attempts"][1]["directory"]) / "audio.wav"
        self.assertEqual(Path(self.args.output).read_bytes(), accepted.read_bytes())

    def test_cap_is_rejected_without_asr(self):
        self.frames[5] = self.args.max_new_tokens
        status, report = self.run_guard()
        self.assertEqual(status, 0)
        self.assertEqual(self.asr_calls, 1)
        self.assertEqual(report["attempts"][0]["reason"], "generation_limit")

    def test_matching_transcript_at_asr_limit_is_rejected_before_scoring(self):
        self.args.attempts = 1
        self.transcripts[5] = {"text": TRANSCRIPT, "generated_tokens": 1024}
        with patch.object(GUARD, "text_scores", side_effect=AssertionError("Must not score truncated ASR")):
            status, report = self.run_guard()
        self.assertEqual(status, 2)
        self.assertEqual(report["attempts"][0]["reason"], "asr_generation_limit")
        self.assertFalse(Path(self.args.output).exists())

    def test_asr_limit_retries_next_seed(self):
        self.transcripts[5] = {"text": TRANSCRIPT, "generated_tokens": 1025}
        status, report = self.run_guard()
        self.assertEqual(status, 0)
        self.assertEqual(self.seeds, [5, 6])
        self.assertEqual(report["attempts"][0]["reason"], "asr_generation_limit")

    def test_invalid_asr_token_count_fails_closed(self):
        for count in (None, True, -1, "100", 100.5):
            with self.subTest(count=count):
                self.transcripts[5] = {"text": TRANSCRIPT, "generated_tokens": count}
                status, _ = self.run_guard()
                self.assertEqual(status, 1)
                self.assertFalse(Path(self.args.output).exists())

    def test_long_target_does_not_hide_reference_repetition(self):
        reference = "Yesterday we watched the sunset beside the quiet lake."
        self.args.reference = [("reference.wav", reference)]
        self.args.target = (
            "[S1] The museum opens at nine tomorrow morning. Shall we take the bus or walk there? "
            "Let us walk if the weather stays dry. I will bring an umbrella just in case. "
            "We can have lunch near the river afterward. That sounds good. "
            "[S2] I will meet you outside the hotel.")
        self.transcripts[5] = {"text": reference + " " + self.args.target}
        self.transcripts[6] = {"text": self.args.target}
        scores = GUARD.text_scores(self.args.target, self.transcripts[5]["text"])
        self.assertLess(scores["word_error_rate"], 0.25)
        status, report = self.run_guard()
        self.assertEqual(status, 0)
        self.assertEqual(report["attempts"][0]["reason"], "reference_repetition")
        self.assertEqual(report["attempts"][0]["unexpected_references"], [1])
        self.assertEqual(self.seeds, [5, 6])

    def test_requested_reference_phrase_is_allowed(self):
        reference = "Yesterday we watched the sunset beside the quiet lake."
        self.args.reference = [("reference.wav", reference)]
        self.args.target = "[S1] " + reference + " [S2] Please bring your ticket."
        self.transcripts[5] = {"text": self.args.target}
        status, report = self.run_guard()
        self.assertEqual(status, 0)
        self.assertEqual(report["attempts"][0]["unexpected_references"], [])

    def test_empty_transcript_retries(self):
        self.transcripts[5] = {"text": ""}
        self.args.max_error_rate = 1.0
        status, _ = self.run_guard()
        self.assertEqual(status, 0)
        self.assertEqual(self.seeds, [5, 6])

    def test_exhaustion_leaves_no_output(self):
        self.frames = {seed: self.args.max_new_tokens for seed in (5, 6, 7)}
        status, report = self.run_guard()
        self.assertEqual(status, 2)
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(self.seeds, [5, 6, 7])
        self.assertFalse(Path(self.args.output).exists())

    def test_invalid_retry_budget_is_rejected(self):
        self.args.attempts = GUARD.MAX_ATTEMPTS + 1
        with self.assertRaises(ValueError):
            GUARD.validate(self.args)

    def test_seed_overflow_is_rejected(self):
        self.args.seed = GUARD.MAX_SEED
        with self.assertRaises(ValueError):
            GUARD.validate(self.args)

    def test_nonfinite_threshold_is_rejected(self):
        self.args.max_error_rate = float("nan")
        with self.assertRaises(ValueError):
            GUARD.validate(self.args)

    def test_existing_output_is_preserved(self):
        Path(self.args.output).write_bytes(b"existing")
        with self.assertRaises(ValueError):
            self.run_guard()
        self.assertEqual(Path(self.args.output).read_bytes(), b"existing")
        self.assertEqual(self.seeds, [])

    def test_unknown_frame_count_fails_closed(self):
        self.marker = False
        status, report = self.run_guard()
        self.assertEqual(status, 1)
        self.assertEqual(report["status"], "error")
        self.assertEqual(self.asr_calls, 0)
        self.assertFalse(Path(self.args.output).exists())

    def test_malformed_asr_payload_fails_closed(self):
        self.transcripts[5] = []
        status, _ = self.run_guard()
        self.assertEqual(status, 1)
        self.assertEqual(self.seeds, [5])
        self.assertFalse(Path(self.args.output).exists())

    def test_command_failure_does_not_retry(self):
        with patch.object(GUARD, "run_command", side_effect=subprocess.CalledProcessError(1, ["moss-cli"])):
            status, report = GUARD.run_guard(self.args)
        self.assertEqual(status, 1)
        self.assertEqual(len(report["attempts"]), 1)
        self.assertFalse(Path(self.args.output).exists())

    def test_timeout_does_not_publish(self):
        with patch.object(GUARD, "run_command", side_effect=subprocess.TimeoutExpired(["moss-cli"], 30)):
            status, _ = GUARD.run_guard(self.args)
        self.assertEqual(status, 1)
        self.assertFalse(Path(self.args.output).exists())

    def test_silent_output_is_rejected(self):
        self.silent = True
        status, report = self.run_guard()
        self.assertEqual(status, 2)
        self.assertEqual(report["attempts"][0]["reason"], "invalid_audio")
        self.assertEqual(self.asr_calls, 0)

    def test_missing_middle_turn_retries_despite_low_overall_error(self):
        self.args.target = (
            "[S1] The museum opens at nine tomorrow morning. "
            "[S2] Shall we take the bus or walk there? "
            "[S1] Let us walk if the weather stays dry. "
            "[S2] I will bring an umbrella just in case. "
            "[S1] We can have lunch near the river afterward. "
            "[S2] That sounds good. I will meet you outside the hotel.")
        incomplete = self.args.target.replace("[S1] Let us walk if the weather stays dry. ", "")
        scores = GUARD.text_scores(self.args.target, incomplete)
        self.assertLess(scores["word_error_rate"], self.args.max_error_rate)
        self.assertEqual(scores["ending_error_rate"], 0.0)
        self.assertGreater(scores["worst_turn_error_rate"], self.args.max_error_rate)
        self.transcripts[5] = {"text": incomplete}
        self.transcripts[6] = {"text": self.args.target}
        status, report = self.run_guard()
        self.assertEqual(status, 0)
        self.assertEqual(self.seeds, [5, 6])
        self.assertEqual(report["attempts"][0]["reason"], "text_mismatch")
        accepted = Path(report["attempts"][1]["directory"]) / "audio.wav"
        self.assertEqual(Path(self.args.output).read_bytes(), accepted.read_bytes())

    def test_turn_check_allows_small_transcription_errors(self):
        scores = GUARD.text_scores(TARGET, "A train leaves this afternoon. Please bring your ticket.")
        self.assertLessEqual(max(scores.values()), self.args.max_error_rate)

    def test_missing_short_final_turn_is_rejected(self):
        expected = "[S1] We will visit the museum near the river tomorrow morning. [S2] Bring tickets."
        scores = GUARD.text_scores(expected, "We will visit the museum near the river tomorrow morning.")
        self.assertLess(scores["word_error_rate"], 0.25)
        self.assertGreater(scores["ending_error_rate"], 0.25)

    def test_resampling_preserves_rate_duration_and_constant_signal(self):
        source, output = self.root / "source.wav", self.root / "asr.wav"
        write_audio(source)
        self.assertAlmostEqual(GUARD.prepare_asr_audio(source, output), 0.1)
        with wave.open(str(output), "rb") as audio:
            self.assertEqual(audio.getframerate(), 16000)
            self.assertEqual(audio.getnframes(), 1600)
        samples, _ = GUARD.read_audio(output)
        self.assertEqual(set(samples), {1000})

    def test_publish_does_not_replace_racing_output(self):
        source, output = self.root / "candidate.wav", self.root / "output.wav"
        source.write_bytes(b"validated")
        output.write_bytes(b"other writer")
        with self.assertRaises(FileExistsError):
            GUARD.publish(source, output)
        self.assertEqual(output.read_bytes(), b"other writer")


if __name__ == "__main__":
    unittest.main()
