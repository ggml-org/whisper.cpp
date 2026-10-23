import importlib.util
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).with_name("audio8-streaming.py")
SPEC = importlib.util.spec_from_file_location("audio8_streaming", SOURCE)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
RATE = 8000
SAMPLES = 8000


def record(mode, first="500", total="1000", chunks="2"):
    return (f"[audio8-performance] mode={mode} first_audio_ms={first} total_ms={total} "
            f"audio_s=1.000 realtime_factor=1.000 chunks={chunks}\n")


def fake_run(args, stdout, stderr, timeout):
    mode = "stream" if "--stream" in args else "batch"
    stdout.write(record(mode))
    with wave.open(args[args.index("--out") + 1], "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(RATE)
        audio.writeframes(bytes(SAMPLES * 2))
    Path(args[args.index("--dump-codes") + 1]).write_text("1,2\n")
    return subprocess.CompletedProcess(args, 0)


class StreamingBenchmarkTests(unittest.TestCase):
    def test_parse_rejects_invalid_measurements(self):
        for text in ("", record("batch"), record("stream", first="nan"),
                     record("stream", first="-1"), record("stream", first="1001"),
                     record("stream", chunks="0"), record("stream") * 2):
            with self.subTest(text=text), self.assertRaises(ValueError):
                BENCH.parse_metrics(text, "stream")

    def test_pairs_preserve_evidence_and_alternate_order(self):
        with tempfile.TemporaryDirectory() as directory:
            opts = SimpleNamespace(binary=Path("audio8-cli"), lm=Path("lm.gguf"),
                                   decoder=Path("decoder.gguf"), out_dir=Path(directory),
                                   runs=2, backend="cpu", gpu_layers=0)
            with patch.object(BENCH.subprocess, "run", side_effect=fake_run) as run:
                report = BENCH.benchmark(opts)
            self.assertEqual([item["mode"] for item in report["runs"]],
                             ["batch", "stream", "stream", "batch"])
            self.assertEqual(report["medians"]["stream"]["first_audio_ms"], 500)
            self.assertTrue((opts.out_dir / "latency.json").is_file())
            self.assertEqual(run.call_count, 4)
            pair = report["runs"][:2]
            (opts.out_dir / (pair[1]["name"] + ".codes.csv")).write_text("9,9\n")
            with self.assertRaisesRegex(ValueError, "codes differ"):
                BENCH.validate_pair(opts, pair)

    def test_failed_process_keeps_log(self):
        with tempfile.TemporaryDirectory() as directory:
            opts = SimpleNamespace(binary=Path("audio8-cli"), lm=Path("lm.gguf"),
                                   decoder=Path("decoder.gguf"), out_dir=Path(directory),
                                   runs=1, backend="cpu", gpu_layers=0)
            with patch.object(BENCH.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    BENCH.run_one(opts, "stream", 1)
            self.assertTrue((opts.out_dir / "01-stream.log").is_file())


if __name__ == "__main__":
    unittest.main()
