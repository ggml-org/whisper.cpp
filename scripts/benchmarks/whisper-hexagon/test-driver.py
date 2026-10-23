#!/usr/bin/env python3
"""Model-free regression checks for the installed native measurement driver.

Usage: python3 test-driver.py --binary /path/to/whisper-hexagon

All audio fixtures are temporary. Valid WAVs deliberately use a missing model:
exit 5 proves audio/configuration validation reached the loader, not inference
correctness. These tests require neither a model nor a GPU/HTP device.
"""
import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import wave


@dataclass
class Case:
    name: str
    args: list
    status: int
    message: str
    config: dict = field(default_factory=dict)


def chunk(name, payload):
    """Encode a RIFF chunk, including its required even-byte padding."""
    return name + struct.pack("<I", len(payload)) + payload + b"\0" * (len(payload) % 2)


def riff(*chunks):
    payload = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(payload)) + payload


def fixtures(root):
    paths = {}

    def put(name, content):
        paths[name] = root / (name + ".wav")
        paths[name].write_bytes(content)

    # Use the standard-library writer for ordinary PCM, independently of the
    # driver's decoder. The handcrafted cases below express RIFF fixtures only.
    for name, channels, rate, width in (("pcm", 1, 16000, 2),
                                         ("stereo", 2, 16000, 2),
                                         ("rate", 1, 8000, 2),
                                         ("bits", 1, 16000, 1)):
        paths[name] = root / (name + ".wav")
        with wave.open(str(paths[name]), "wb") as output:
            output.setparams((channels, width, rate, 0, "NONE", "not compressed"))
            output.writeframes(b"\0" * (160 * channels * width))

    pcm = paths["pcm"].read_bytes()
    pcm_fmt = chunk(b"fmt ", struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16))
    float_fmt = chunk(b"fmt ", struct.pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32))
    pcm_data = chunk(b"data", struct.pack("<160h", *([-32768, -123, 0, 123, 32767] * 32)))
    float_data = chunk(b"data", struct.pack("<160f", *([-1.0, -0.25, 0.0, 0.25, 1.0] * 32)))
    put("float", riff(float_fmt, float_data))
    put("junk", riff(chunk(b"JUNK", b"hello"), pcm_fmt, pcm_data))
    put("not-wav", b"RIFX" + pcm[4:])
    put("short", b"RIFF")
    put("format", riff(chunk(b"fmt ", struct.pack("<HHIIHH", 7, 1, 16000, 32000, 2, 16)), pcm_data))
    put("nan", riff(float_fmt, chunk(b"data", struct.pack("<f", float("nan")))))
    put("inf", riff(float_fmt, chunk(b"data", struct.pack("<f", float("inf")))))
    put("duplicate-fmt", riff(pcm_fmt, pcm_fmt, pcm_data))
    put("duplicate-data", riff(pcm_fmt, pcm_data, pcm_data))
    put("empty", riff(pcm_fmt, chunk(b"data", b"")))
    put("misaligned", riff(pcm_fmt, chunk(b"data", b"\0")))
    put("truncated", pcm[:-1])
    put("huge-chunk", riff(pcm_fmt, b"data" + struct.pack("<I", 0xFFFFFFFF) + b"\0\0"))
    put("small-riff", pcm[:4] + struct.pack("<I", 3) + pcm[8:])
    return paths


def cases(root):
    audio = fixtures(root)
    missing_model = str(root / "missing model caf\u00e9.bin")
    common = ["--model", missing_model, "--audio", str(audio["pcm"])]
    cpu = common + ["--no-gpu"]
    result = [
        Case("no-arguments", [], 2, "--model and --audio are required"),
        Case("missing-value", ["--model"], 2, "--model requires a value"),
        Case("unknown-flag", ["--unknown-option"], 2, "unknown argument: --unknown-option"),
        Case("missing-backend", common, 2, "choose exactly one of --device N and --no-gpu"),
        Case("contradictory-backend", cpu + ["--device", "0"], 2,
             "choose exactly one of --device N and --no-gpu"),
        Case("negative-device", common + ["--device", "-1"], 2, "--device: invalid integer '-1'"),
        Case("partial-device", common + ["--device", "0tail"], 2, "--device: invalid integer '0tail'"),
        Case("overflow-device", common + ["--device", "2147483648"], 2,
             "--device: invalid integer '2147483648'"),
        Case("zero-threads", cpu + ["--threads", "0"], 2, "--threads: value is below 1"),
        Case("negative-warmup", cpu + ["--warmup", "-1"], 2, "--warmup: invalid integer '-1'"),
        Case("zero-runs", cpu + ["--runs", "0"], 2, "--runs: value is below 1"),
    ]
    for name, value in (("nan-duration", "nan"), ("infinite-duration", "inf"),
                        ("negative-duration", "-1"), ("underflow-duration", "1e-9999"),
                        ("whitespace-duration", " 1")):
        result.append(Case(name, cpu + ["--duration-seconds", value], 2,
                           "--duration-seconds requires a finite non-negative number"))
    result.append(Case("auto-language", cpu + ["--language", "auto"], 2,
                       "--language must be a fixed supported language code"))

    # Successful audio validation emits a configuration record, then the
    # deliberately missing model fails. Non-default options pin CLI-to-JSON
    # propagation without running or allocating a real model.
    for name in ("pcm", "float", "junk"):
        config = {"model": missing_model, "audio": str(audio[name]), "samples": 160,
                  "threads": 4, "language": "en", "translate": False, "flash_attn": True,
                  "warmup_calls": 1, "measurement_calls": 3, "sustained_seconds": 0}
        args = ["--model", missing_model, "--audio", str(audio[name]), "--no-gpu"]
        if name == "float":
            args += ["--threads", "2", "--language", "fr", "--translate", "--no-flash-attn",
                     "--warmup", "0", "--runs", "1", "--duration-seconds", "0.5"]
            config.update(threads=2, language="fr", translate=True, flash_attn=False,
                          warmup_calls=0, measurement_calls=1, sustained_seconds=0.5)
        result.append(Case(name, args, 5, "Whisper model/context loading failed", config))

    wav_errors = {
        "not-wav": "only little-endian RIFF/WAVE audio is supported",
        "short": "WAV is shorter than its RIFF header",
        "stereo": "WAV must be mono 16000 Hz PCM16 (format 1) or float32 (format 3)",
        "rate": "WAV must be mono 16000 Hz PCM16 (format 1) or float32 (format 3)",
        "bits": "WAV must be mono 16000 Hz PCM16 (format 1) or float32 (format 3)",
        "format": "WAV must be mono 16000 Hz PCM16 (format 1) or float32 (format 3)",
        "nan": "WAV contains a non-finite float sample",
        "inf": "WAV contains a non-finite float sample",
        "duplicate-fmt": "invalid or duplicate WAV fmt chunk",
        "duplicate-data": "multiple WAV data chunks are unsupported",
        "empty": "missing, empty or misaligned WAV audio",
        "misaligned": "missing, empty or misaligned WAV audio",
        "truncated": "invalid RIFF size",
        "huge-chunk": "WAV chunk exceeds RIFF size",
        "small-riff": "invalid RIFF size",
    }
    for name, message in wav_errors.items():
        result.append(Case(name, ["--model", missing_model, "--audio", str(audio[name]), "--no-gpu"],
                           3, message))
    # INT_MAX is a valid numeric argument and an unavailable ordinal even on
    # GPU-equipped CI machines; this must fail at backend selection, not parsing.
    result.append(Case("unavailable-gpu", common + ["--device", "2147483647"], 4,
                       "requested GPU/IGPU ordinal is unavailable; check whisper-cli --list-devices"))
    return result


def reject_non_json_number(value):
    raise ValueError("non-JSON numeric literal: " + value)


class DriverCase(unittest.TestCase):
    def __init__(self, binary, case):
        super().__init__("runTest")
        self.binary = binary
        self.case = case

    def shortDescription(self):
        return self.case.name

    def runTest(self):
        case = self.case
        command = [self.binary] + case.args
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=15)
        detail = "command: {!r}\nexit: {}\nstdout:\n{}\nstderr:\n{}".format(
            command, completed.returncode, completed.stdout, completed.stderr)
        self.assertEqual(completed.returncode, case.status, detail)
        try:
            records = [json.loads(line, parse_constant=reject_non_json_number)
                       for line in completed.stdout.splitlines()]
        except (ValueError, TypeError) as error:
            self.fail("stdout must contain only valid JSONL: {}\n{}".format(error, detail))
        expected_kinds = ["config", "error"] if case.config else ["error"]
        self.assertTrue(all(isinstance(record, dict) for record in records), detail)
        self.assertEqual([record.get("kind") for record in records], expected_kinds, detail)
        self.assertEqual(records[-1], {"kind": "error", "exit_code": case.status, "message": case.message}, detail)
        self.assertIn("whisper-hexagon: " + case.message, completed.stderr, detail)
        if not case.config:
            return

        config = records[0]
        for key, expected in case.config.items():
            self.assertIn(key, config, detail)
            self.assertIs(type(config[key]), type(expected), detail)
            self.assertEqual(config[key], expected, detail)
        self.assertEqual(config.get("schema_version"), 1, detail)
        self.assertTrue(isinstance(config.get("whisper_version"), str) and config["whisper_version"], detail)
        self.assertTrue(math.isclose(config.get("audio_s", -1), 0.01, abs_tol=1e-12), detail)
        self.assertEqual(config.get("requested_backend"), "CPU", detail)
        self.assertIsNone(config.get("requested_gpu_ordinal"), detail)
        self.assertIs(config.get("no_context"), True, detail)
        self.assertIs(config.get("backend_execution_confirmed"), False, detail)
        for key, expected in (("temperature", 0), ("temperature_inc", 0), ("seed", 0), ("greedy_best_of", 1)):
            self.assertEqual(config.get(key), expected, detail)
        expected_env = {key: os.environ.get(key) for key in
                        ("GGML_HEXAGON_NDEV", "GGML_HEXAGON_HOSTBUF", "GGML_HEXAGON_BF16",
                         "GGML_HEXAGON_PROFILE", "GGML_HEXAGON_OPTRACE", "GGML_SCHED_DEBUG")}
        self.assertEqual(config.get("environment"), expected_env, detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="installed-consumer whisper-hexagon executable")
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error("--binary is not a file: " + str(args.binary))
    binary = str(args.binary.resolve())
    with tempfile.TemporaryDirectory(prefix="whisper-driver-") as temporary:
        suite = unittest.TestSuite(DriverCase(binary, case) for case in cases(Path(temporary)))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
