"""Exercise the production benchmark process with a generated CPU decoder."""
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import tempfile
import unittest
import wave


BENCH, FIXTURE = sys.argv[1:3]
del sys.argv[1:3]


class CodecBenchRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audio8-codec-bench-runner-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {**os.environ, "AUDIO8_COREML": "0"}
        result = subprocess.run([FIXTURE, str(self.root)], env=self.env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.reference = [float.fromhex(line) for line in
                          (self.root / "reference.txt").read_text().splitlines()]
        self.assertEqual(len(self.reference), 12)
        self.assertTrue(all(math.isfinite(x) for x in self.reference))
        self.assertGreater(max(self.reference) - min(self.reference), 1e-6)

    def command(self, *extra):
        return [BENCH, "--codec-decoder", str(self.root / "decoder.gguf"),
                "--codes", str(self.root / "codes.csv"), "--codec-backend", "cpu",
                "--frames", "3", "--threads", "1", *extra]

    def report(self, stdout, prefix):
        lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
        self.assertEqual(len(lines), 1, stdout)
        return dict(re.findall(r"(\w+)=([^\s]+)", lines[0]))

    def check_outputs(self, wav, f32):
        self.assertEqual(f32.read_bytes(), struct.pack("<12f", *self.reference))
        self.assertEqual(wav.stat().st_size, 44 + 12 * 2)
        data = wav.read_bytes()
        self.assertEqual(data[:4], b"RIFF")
        self.assertEqual(struct.unpack_from("<I", data, 4)[0], len(data) - 8)
        self.assertEqual(data[8:12], b"WAVE")
        with wave.open(str(wav), "rb") as file:
            self.assertEqual((file.getnchannels(), file.getsampwidth(), file.getframerate(),
                              file.getnframes()), (1, 2, 8000, 12))
            want = [round(max(-1.0, min(1.0, sample)) * 32767.0) for sample in self.reference]
            self.assertEqual(file.readframes(12), struct.pack("<12h", *want))

    def test_iterations_statistics_and_outputs(self):
        # Single, odd and even repetition counts exercise both median branches.
        # Warmups do not contribute iteration lines or summary statistics.
        for warmup, repeat in ((0, 1), (0, 3), (0, 4), (2, 2)):
            with self.subTest(warmup=warmup, repeat=repeat):
                wav = self.root / f"out-{warmup}-{repeat}.wav"
                f32 = self.root / f"out-{warmup}-{repeat}.f32"
                result = subprocess.run(self.command("--warmup", str(warmup), "--repeat", str(repeat),
                                        "--out", str(wav), "--out-f32", str(f32)),
                                        env=self.env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("decoder backend: CPU (requested=cpu)", result.stdout)
                iterations = re.findall(r"^decode iteration: repeat=(\d+) decoder_ms=([\d.]+) "
                                        r"synthesis_backend=ggml scheduler_initialized=false "
                                        r"cpu_fallback_available=false$", result.stdout, re.MULTILINE)
                self.assertEqual([int(index) for index, _ in iterations], list(range(1, repeat + 1)),
                                 result.stdout)
                times = [float(value) for _, value in iterations]
                self.assertTrue(all(math.isfinite(t) and t >= 0 for t in times))
                timing = self.report(result.stdout, "decoder timing:")
                expected = dict(min_ms=min(times), mean_ms=statistics.mean(times),
                                median_ms=statistics.median(times), max_ms=max(times),
                                stddev_ms=statistics.pstdev(times))
                self.assertEqual(set(timing), set(expected))
                for key, value in expected.items():
                    self.assertAlmostEqual(float(timing[key]), value, delta=2e-6, msg=key)
                info = self.report(result.stdout, "codec decode:")
                for key, value in dict(frames=3, codebooks=2, codebook_dim=4, latent_dim=4,
                                       samples=12, sample_rate=8000, warmup=warmup,
                                       repeats=repeat, threads=1).items():
                    self.assertEqual(int(info[key]), value, key)
                self.assertEqual(info["finite"], "true")
                self.assertAlmostEqual(float(info["audio_seconds"]), 12 / 8000, delta=1e-6)
                for key, value in dict(pcm_min=min(self.reference), pcm_max=max(self.reference),
                                       pcm_rms=math.sqrt(sum(x*x for x in self.reference) / 12)).items():
                    self.assertAlmostEqual(float(info[key]), value, delta=1e-8, msg=key)
                stability = self.report(result.stdout, "decoder repetitions:")
                self.assertEqual(stability["pcm_bitwise_stable"], "true")
                self.assertEqual(float(stability["pcm_max_abs_diff"]), 0.0)
                self.check_outputs(wav, f32)

    def test_nonexistent_case_alias_outputs_are_rejected_before_writing(self):
        lower, upper = self.root / "out.wav", self.root / "OUT.WAV"
        self.assertFalse(lower.exists())
        self.assertFalse(upper.exists())
        result = subprocess.run(self.command("--warmup", "0", "--repeat", "1",
                                "--out", str(lower), "--out-f32", str(upper)),
                                env=self.env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("WAV and F32 outputs alias the same file", result.stderr)
        self.assertNotIn("decode iteration:", result.stdout)
        self.assertFalse(lower.exists())
        self.assertFalse(upper.exists())


if __name__ == "__main__":
    unittest.main()
