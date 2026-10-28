#!/usr/bin/env python3
"""Model-free measurement-consumer contracts; not model/HTP correctness tests.

The real installed consumer checks malformed CLI/configuration and constructor
errors. A CMake-built second executable links its exact source with a public-
Engine test double to exercise failed streams, output validity, phases and receipts.
No production library is replaced in a device/model validation run.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import wave


def reject_constant(value):
    raise ValueError("non-JSON number: " + value)


class DriverContracts(unittest.TestCase):
    real = None
    stub = None
    temporary = None

    def setUp(self):
        self.case_dir = Path(tempfile.mkdtemp(dir=self.temporary))

    def command(self, binary=None, text="contract audio", extra=(), out=None, model="contract-model"):
        return [str(binary or self.stub), "--model", model, "--s3gen-gguf", "contract-s3gen",
                "--backend", "cpu", "--text", text, "--out", str(out or self.case_dir / "output.wav"),
                "--warmup", "0", "--runs", "1", *extra]

    def invoke(self, command, status):
        proc = subprocess.run(command, capture_output=True, encoding="utf-8", timeout=30)
        detail = repr(command) + "\n" + proc.stdout + "\n" + proc.stderr
        self.assertEqual(proc.returncode, status, detail)
        records = [json.loads(line, parse_constant=reject_constant) for line in proc.stdout.splitlines()]
        self.assertTrue(all(isinstance(r, dict) for r in records), detail)
        if status:
            self.assertEqual(records[-1]["kind"], "error", detail)
            self.assertEqual(records[-1]["exit_code"], status, detail)
            self.assertFalse(records[-1]["complete"], detail)
            self.assertFalse(any(r["kind"] == "end" for r in records), detail)
        return records

    def test_invalid_arguments_never_load_a_model(self):
        cases = [[], ["--unknown"], ["--model"],
                 self.command(self.real) [1:] + ["--threads", "0"],
                 self.command(self.real) [1:] + ["--n-ctx", "2147483648"],
                 self.command(self.real) [1:] + ["--duration-seconds", "nan"],
                 self.command(self.real) [1:] + ["--duration-seconds", "inf"],
                 self.command(self.real) [1:] + ["--duration-seconds", "-1"],
                 self.command(self.real) [1:] + ["--duration-seconds", " 1"],
                 self.command(self.real) [1:] + ["--stream-cfm-steps", "1"],
                 self.command(self.real) [1:] + ["--language", "auto"],
                 self.command(self.real) [1:] + ["--kv-cache-type", "unknown"],
                 self.command(self.real) [1:] + ["--text", "duplicate"]]
        for args in cases:
            with self.subTest(args=args):
                records = self.invoke([str(self.real), *args], 2)
                self.assertEqual(len(records), 1)

    def test_real_missing_model_has_no_success_receipt(self):
        records = self.invoke(self.command(self.real, model=str(self.case_dir / "missing.gguf")), 5)
        self.assertEqual([r["kind"] for r in records], ["config", "error"])
        self.assertFalse((self.case_dir / "output.wav").exists())

    def test_complete_batch_and_stream_receipts(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                out = self.case_dir / ("stream.wav" if stream else "batch.wav")
                records = self.invoke(self.command(extra=("--stream-chunk-tokens", "25") if stream else (), out=out), 0)
                self.assertEqual([r["kind"] for r in records], ["config", "load", "run", "run", "end"])
                self.assertFalse(records[0]["backend_execution_confirmed"])
                self.assertTrue(records[1]["primary_backend_matches_request"])
                self.assertFalse(records[1]["constructor_includes_all_model_loading"])
                self.assertIsNone(records[1]["preload_s"])
                runs = [r for r in records if r["kind"] == "run"]
                self.assertEqual([r["phase"] for r in runs], ["first_call", "measure"])
                for run in runs:
                    self.assertEqual(run["status"], "ok")
                    self.assertGreater(run["native_s"], 0)
                    self.assertAlmostEqual(run["phase_end_s"] - run["phase_begin_s"], run["native_s"])
                    self.assertTrue(run["audio_contract_ok"])
                    self.assertTrue(run["stream_contract_ok"])
                    self.assertEqual(run["non_finite_samples"], 0)
                    self.assertEqual(run["clipped_samples"], 2)
                    self.assertEqual(run["peak_abs"], 1.25)
                    self.assertAlmostEqual(run["rms"], math.sqrt(4.3125 / 8))
                    self.assertEqual(run["termination_status"], "not_exposed_by_public_api")
                    if stream:
                        self.assertGreater(run["first_audio_s"], 0)
                        self.assertLessEqual(run["first_audio_s"], run["native_s"])
                        self.assertEqual(run["stream_chunks"], 2)
                        self.assertEqual(run["last_chunks"], 1)
                    else:
                        self.assertIsNone(run["first_audio_s"])
                        self.assertEqual(run["stream_chunks"], 0)
                    with wave.open(run["output_path"]) as wav:
                        self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()), (1, 2, 24000, 8))
                self.assertTrue(records[-1]["complete"])
                self.assertEqual(out.read_bytes(), Path(runs[-1]["output_path"]).read_bytes())

    def test_constructor_and_backend_failures(self):
        self.invoke(self.command(model="throw-constructor"), 5)
        command = self.command(model="wrong-primary")
        command[command.index("--backend") + 1] = "hexagon"
        records = self.invoke(command, 4)
        self.assertFalse(records[1]["primary_backend_matches_request"])
        records = self.invoke(self.command(model="unknown-primary"), 4)
        self.assertFalse(records[1]["primary_backend_matches_request"])

    def test_inference_and_audio_failures(self):
        for text, status in (("throw-synthesis", 7), ("empty", 6), ("non-finite", 6),
                             ("bad-sample-count", 6), ("bad-stage-time", 6)):
            with self.subTest(text=text):
                out = self.case_dir / (text + ".wav")
                self.invoke(self.command(text=text, out=out), status)
                self.assertFalse(out.exists())
                self.assertFalse(list(self.case_dir.glob(text + ".wav.*.wav")))

    def test_stream_contract_failures(self):
        for text in ("bad-index", "early-last", "missing-last", "mismatched-pcm", "missing-callback", "null-callback"):
            with self.subTest(text=text):
                out = self.case_dir / (text + ".wav")
                self.invoke(self.command(text=text, out=out, extra=("--stream-chunk-tokens", "25")),
                            7 if text == "null-callback" else 6)
                self.assertFalse(out.exists())

    def test_warmups_and_native_sustained_budget(self):
        command = self.command(extra=("--duration-seconds", "0.012"))
        command[command.index("--warmup") + 1] = "2"
        command[command.index("--runs") + 1] = "2"
        records = self.invoke(command, 0)
        runs = [r for r in records if r["kind"] == "run"]
        self.assertEqual([r["phase"] for r in runs[:5]], ["first_call", "warmup", "warmup", "measure", "measure"])
        sustained = [r for r in runs if r["phase"] == "sustained"]
        self.assertTrue(sustained)
        self.assertGreaterEqual(sum(r["native_s"] for r in sustained), 0.012)
        self.assertAlmostEqual(records[-1]["sustained_native_s"], sum(r["native_s"] for r in sustained))
        self.assertEqual(records[-1]["total_calls"], len(runs))
        self.assertEqual(records[-1]["sustained_calls"], len(sustained))

    def test_output_refusal_and_no_stale_success(self):
        out = self.case_dir / "existing.wav"
        out.write_bytes(b"old-output")
        self.invoke(self.command(out=out), 6)
        self.assertEqual(out.read_bytes(), b"old-output")
        self.invoke(self.command(out=self.case_dir / "missing-parent" / "out.wav"), 6)
        partial = self.case_dir / "partial.wav.partial"
        partial.write_bytes(b"incomplete")
        self.invoke(self.command(out=self.case_dir / "partial.wav"), 6)

    @unittest.skipIf(os.name == "nt", "Windows symlink creation may require additional privileges")
    def test_dangling_output_symlinks_are_not_followed(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                out = self.case_dir / ("partial-link.wav" if partial else "link.wav")
                link = Path(str(out) + ".partial") if partial else out
                target = self.case_dir / ("partial-target" if partial else "target")
                link.symlink_to(target)
                self.invoke(self.command(out=out), 6)
                self.assertTrue(link.is_symlink())
                self.assertFalse(target.exists())

    @unittest.skipIf(os.name == "nt", "Windows narrow-argv filename encoding is not an exercised contract")
    def test_unicode_text_file_and_paths(self):
        text = self.case_dir / "voix française.txt"
        text.write_text("Bonjour, le café est prêt.\n", encoding="utf-8")
        command = self.command(out=self.case_dir / "été.wav")
        at = command.index("--text")
        command[at:at + 2] = ["--text-file", str(text)]
        records = self.invoke(command, 0)
        self.assertEqual(records[0]["text"], text.read_text(encoding="utf-8"))
        self.assertEqual(records[0]["text_file"], str(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--stub-binary", required=True, type=Path)
    args = parser.parse_args()
    if not args.binary.is_file() or not args.stub_binary.is_file():
        parser.error("the native real and contract-stub binaries must exist")
    with tempfile.TemporaryDirectory(prefix="chatterbox-driver-contracts-") as temporary:
        DriverContracts.real = args.binary.resolve()
        DriverContracts.stub = args.stub_binary.resolve()
        DriverContracts.temporary = temporary
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DriverContracts))
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
