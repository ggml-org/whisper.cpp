#!/usr/bin/env python3
"""Model-free measurement-integrity contracts; no engine or DSP math is tested."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock
import wave

HERE = Path(__file__).resolve().parent
module_spec = importlib.util.spec_from_file_location(
    "chatterbox_collection", HERE / "chatterbox-hexagon-validation.py")
collector = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(collector)
FIT_BINARY = None

# An actual child process emits files and receipts. The process shim below only
# inserts Python's executable, making this fixture work on Windows too.
FAKE_DRIVER = r'''
import json, os, shutil, struct, sys, wave
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
mode = os.environ.get("CHATTERBOX_COLLECTION_TEST_MODE", "")
out = Path(args["--out"])
settings = {}
for key in ("seed", "threads", "n_ctx", "n_predict", "max_sentence_chars", "cfm_steps",
            "stream_chunk_tokens", "stream_first_chunk_tokens", "stream_left_context_tokens",
            "stream_cfm_steps"):
    settings[key] = int(args.get("--" + key.replace("_", "-"), "0"))
settings.update(n_gpu_layers=0 if args["--backend"] == "cpu" else 99,
    kv_cache_type=args["--kv-cache-type"], language=args["--language"],
    warmup_calls=int(args["--warmup"]), measurement_calls=int(args["--runs"]),
    first_call_calls=1, sustained_seconds=float(args["--duration-seconds"]))
settings.update(top_k=1000, top_p=0.95, temperature=0.8, repeat_penalty=1.2,
    cfg_weight=0.5, exaggeration=0.5, s3gen_cfg_rate=-1.0, crossfade_ms=30)
if mode == "wrong_settings":
    settings["seed"] += 1
environment = {key:os.environ.get(key) for key in (
    "GGML_HEXAGON_NDEV", "GGML_HEXAGON_HOSTBUF", "GGML_HEXAGON_BF16",
    "GGML_HEXAGON_OPPOLL", "GGML_HEXAGON_OPSTAGE", "GGML_HEXAGON_PROFILE",
    "GGML_HEXAGON_OPTRACE", "GGML_SCHED_DEBUG", "TTS_CPP_FORCE_SCHED", "TTS_CPP_GPU_BACKEND")}
events = [dict(kind="config", schema_version=1, requested_backend=args["--backend"],
    model=args["--model"], s3gen_gguf=args["--s3gen-gguf"], output_path=str(out),
    reference_audio=args.get("--reference-audio", ""), text=args["--text"],
    settings=settings, environment=environment),
    dict(kind="load", primary_backend_matches_request=mode != "wrong_primary",
         actual_primary_backend={"cpu":"CPU","opencl":"OpenCL0","hexagon":"HTP0"}[args["--backend"]])]
phases = [("first_call", 1), ("warmup", int(args["--warmup"])),
          ("measure", int(args["--runs"]))]
duration = float(args["--duration-seconds"])
if duration:
    import math
    phases.append(("sustained", math.ceil(duration / 0.05)))
call = 0
last_path = None
stream = settings["stream_chunk_tokens"] > 0
for phase, count in phases:
    for index in range(count):
        call += 1
        path = out.with_name(out.stem + "." + phase + "-" + str(index) + ".wav")
        if mode == "reuse_path":
            path = out.with_name("reused.wav")
        if mode == "escape_path":
            path = out.parent.parent / ("escaped-" + str(call) + ".wav")
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(struct.pack("<160h", *[(i % 7) * 100 for i in range(160)]))
        record = dict(kind="run", phase=phase, index=index, call=call, status="ok",
            native_s=0.05, phase_begin_s=index * 0.05, phase_end_s=(index + 1) * 0.05,
            samples=160, sample_rate=16000, audio_s=0.01, t3_s=0.02, s3gen_s=0.03,
            peak_abs=0.02, rms=0.01, non_finite_samples=0, clipped_samples=0,
            audio_contract_ok=True, stream_contract_ok=True, stream_order_ok=True,
            stream_pcm_matches=True, stream_chunks=2 if stream else 0,
            last_chunks=1 if stream else 0, first_audio_s=0.02 if stream else None,
            output_path=str(path))
        if mode == "wrong_frames":
            record["samples"] += 1
        if mode == "bad_timing":
            record["phase_end_s"] += 1
        if mode == "bad_stream":
            record["stream_order_ok"] = False
        if mode == "nan":
            record["native_s"] = float("nan")
        if mode == "zero_audio":
            record["rms"] = 0
        events.append(record)
        last_path = path
shutil.copyfile(last_path, out)
end = dict(kind="end", complete=True, exit_code=0, output_path=str(out),
    total_calls=call, measurement_calls=int(args["--runs"]),
    sustained_calls=dict(phases).get("sustained", 0),
    sustained_native_s=dict(phases).get("sustained", 0) * 0.05)
if mode != "missing_end":
    events.append(end)
if mode == "missing_first":
    events = [e for e in events if e.get("phase") != "first_call"]
if mode == "duplicate_run":
    events.insert(-1, events[-2])
if mode == "missing_wav":
    last_path.unlink()
if mode == "truncated_wav":
    out.write_bytes(out.read_bytes()[:-2])
if mode == "wrong_final":
    data = bytearray(out.read_bytes())
    data[-1] ^= 1
    out.write_bytes(data)
for event in events:
    print(json.dumps(event), flush=True)
if mode == "malformed":
    print("{broken-json")
sys.exit(7 if mode == "native7" else 0)
'''


def write_wav(path, *, channels=1, frames=160):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((channels, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(struct.pack("<" + str(frames * channels) + "h",
                                   *([100] * frames * channels)))


class CollectionContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="chatterbox collector ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = collector.read_json(collector.DEFAULT_SPEC)
        self.spec_path = self.root / "matrix.json"
        collector.write_json(self.spec_path, self.spec)
        self.models = self.root / "models"
        self.models.mkdir()
        for family in self.spec["model_files"].values():
            for pair in family.values():
                for name in pair.values():
                    (self.models / name).write_bytes(("fake fixture " + name).encode())
        self.driver = self.root / "fake-driver.py"
        self.driver.write_text(FAKE_DRIVER, encoding="utf-8")
        self.runtime = self.root / "runtime.so"
        self.runtime.write_bytes(b"fake runtime fixture")
        self.voice = self.root / "voice.wav"
        write_wav(self.voice)
        self.device_path = self.root / "phone.json"
        collector.write_json(self.device_path,
            dict(model="test-model", soc="test-soc", htp_architecture="test-v79"))
        self.counter = 0
        self.default_ids = ["builtin-turbo-q4_0-en-" + backend + "-batch"
                            for backend in ("cpu", "opencl", "hexagon")]
        self.frozen_path = self.freeze()

    def freeze(self, ids=None, **overrides):
        self.counter += 1
        out = self.root / ("frozen-" + str(self.counter) + ".json")
        values = dict(spec=self.spec_path, models_root=self.models, asset_map=None,
            reference_audio=self.voice, driver_artifact=self.driver,
            runtime_artifact=[self.runtime], speech_sha="a" * 40, ggml_sha="b" * 40,
            case=self.default_ids if ids is None else ids, out=out)
        values.update(overrides)
        collector.freeze(argparse.Namespace(**values))
        return out

    def args(self, ids=None, **overrides):
        values = dict(frozen=self.frozen_path, case=[self.default_ids[0]] if ids is None else ids,
            driver=None, out_dir=self.root / "attempts", transport="local",
            device_info=None, env=[], timeout=10, warmup=0, runs=1, duration_seconds=0,
            continue_on_error=False, adb=Path("fake-adb"), adb_port=15045,
            serial="test-phone:1234", remote_root="/data/local/tmp/collection-test",
            remote_models_root=None, remote_driver=None)
        values.update(overrides)
        return argparse.Namespace(**values)

    def local_process(self, command, stdout, stderr, timeout, *, env=None):
        # macOS may spell the same temporary path as /var or /private/var.
        # The collector canonicalizes paths; compare real file identities.
        self.assertEqual(self.driver.resolve(), Path(command[0]).resolve())
        return self.real_process([sys.executable, str(self.driver), *command[1:]],
                                 stdout, stderr, timeout, env=env)

    def collect(self, mode="", ids=None, **overrides):
        additional_env = overrides.pop("env", [])
        args = self.args(ids, env=["CHATTERBOX_COLLECTION_TEST_MODE=" + mode] + additional_env,
                         **overrides)
        self.real_process = collector.process
        with mock.patch.object(collector, "process", side_effect=self.local_process):
            result = collector.collect(args)
        directory = Path(result["collection"]).parent
        collection = collector.read_json(result["collection"])
        cases = [collector.read_json(c["result"]) for c in collection["cases"]]
        return result, directory, collection, cases

    def summary(self, *directories):
        self.counter += 1
        out = self.root / ("summary-" + str(self.counter) + ".json")
        collector.summarize(argparse.Namespace(attempt_dir=list(directories), out=out))
        return collector.read_json(out)

    def test_frozen_matrix_has_exact_coverage_and_languages(self):
        cases = collector.expand_cases(self.spec)
        self.assertEqual(84, len(cases))
        self.assertEqual(Counter(builtin=48, clone=24, paragraph=12),
                         Counter(c["group"] for c in cases))
        self.assertEqual({"fr", "es"}, {c["language"] for c in cases if c["language"] != "en"})
        self.assertTrue(all(c["family"] == "mtl" for c in cases if c["language"] != "en"))
        self.assertEqual(84, len({c["id"] for c in cases}))

    def test_collection_accepts_alias_of_frozen_driver(self):
        alias = self.root / "driver-alias.py"
        try:
            alias.symlink_to(self.driver)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("symbolic links unavailable: " + str(exc))
        self.driver = alias
        result, directory, collection, cases = self.collect(driver=alias)
        self.assertEqual("execution_validated", cases[0]["status"])
        self.assertEqual("complete", result["status"])
        self.assertEqual(0, cases[0]["native_exit_code"])

    def test_unknown_or_duplicate_case_is_refused(self):
        cases = collector.expand_cases(self.spec)
        for ids in (["unknown"], [cases[0]["id"]] * 2):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                collector.choose_cases(cases, ids)

    def test_freeze_requires_revision_and_exclusive_destination(self):
        with self.assertRaises(ValueError):
            self.freeze(speech_sha="short")
        frozen = collector.load_frozen(self.frozen_path)
        args = argparse.Namespace(spec=self.spec_path, models_root=self.models,
            asset_map=None, reference_audio=self.voice, driver_artifact=self.driver,
            runtime_artifact=[self.runtime], speech_sha="a" * 40, ggml_sha="b" * 40,
            case=self.default_ids, out=self.frozen_path)
        with self.assertRaises(FileExistsError):
            collector.freeze(args)
        self.assertEqual(frozen, collector.load_frozen(self.frozen_path))

    def test_manifest_or_frozen_case_mutation_is_refused(self):
        original = collector.read_json(self.frozen_path)
        for resign in (False, True):
            with self.subTest(resign=resign):
                data = copy.deepcopy(original)
                data["cases"][0]["text"] = "changed prompt"
                if resign:
                    data["frozen_sha256"] = collector.digest(
                        {k: v for k, v in data.items() if k != "frozen_sha256"})
                collector.write_json(self.frozen_path, data)
                with self.assertRaises(ValueError):
                    collector.load_frozen(self.frozen_path)

    def test_changed_models_driver_or_runtime_prevent_start(self):
        frozen = collector.load_frozen(self.frozen_path)
        for path in (self.models / "chatterbox-t3-turbo-q4_0.gguf", self.driver, self.runtime):
            with self.subTest(path=path.name):
                old = path.read_bytes()
                path.write_bytes(old + b"changed")
                with self.assertRaises(ValueError):
                    collector.verify_inputs(frozen, frozen["cases"], self.driver)
                path.write_bytes(old)

    def test_nonfinite_or_malformed_json_is_refused(self):
        path = self.root / "events.jsonl"
        for data in ('{"kind":"run","native_s":NaN}\n', '{"broken"\n',
                     '[]\n', '{"kind":"unrecognized"}\n'):
            with self.subTest(data=data):
                path.write_text(data)
                with self.assertRaises(ValueError):
                    collector.parse_events(path)

    def test_wav_structure_rejects_truncated_stereo_and_empty_audio(self):
        path = self.root / "invalid.wav"
        for channels, frames, truncate in ((1, 160, True), (2, 160, False), (1, 0, False)):
            with self.subTest(channels=channels, frames=frames, truncate=truncate):
                write_wav(path, channels=channels, frames=frames)
                if truncate:
                    path.write_bytes(path.read_bytes()[:-2])
                with self.assertRaises((ValueError, EOFError, wave.Error)):
                    collector.validate_wav(path)

    def test_real_child_process_accepts_complete_output_without_quality_claim(self):
        result, directory, collection, cases = self.collect()
        self.assertEqual("complete", result["status"])
        self.assertEqual(1, result["validated_cells"])
        self.assertEqual(0, cases[0]["native_exit_code"])
        self.assertEqual({"first_call": 1, "warmup": 0, "measure": 1, "sustained": 0},
                         cases[0]["validation"]["phase_counts"])
        summary = self.summary(directory)
        self.assertEqual("not_evaluated", summary["quality_gate"])
        self.assertEqual("not_confirmed", summary["htp_execution"])
        self.assertFalse(summary["primary_matrix_coverage"][0]["complete_primary_matrix"])
        self.assertEqual(83, summary["primary_matrix_coverage"][0]["missing_primary_cells"])

    def test_nonenglish_streaming_case_keeps_language_and_callback_contract(self):
        case = "builtin-mtl-q8_0-fr-hexagon-stream"
        self.frozen_path = self.freeze([case])
        result, _, _, cases = self.collect(ids=[case])
        self.assertEqual("complete", result["status"])
        self.assertEqual("fr", cases[0]["effective_driver_settings"]["language"])
        self.assertEqual(2, cases[0]["validation"]["runs"][-1]["stream_chunks"])
        self.assertFalse(cases[0]["validation"]["backend_execution_confirmed"])

    def test_clone_requires_frozen_reference_voice(self):
        case = "clone-turbo-q4_0-en-cpu-batch"
        with self.assertRaisesRegex(ValueError, "reference-audio"):
            self.freeze([case], reference_audio=None)
        self.frozen_path = self.freeze([case])
        result, _, _, cases = self.collect(ids=[case])
        self.assertEqual("complete", result["status"])
        self.assertEqual("reference", cases[0]["case"]["voice"])
        self.voice.write_bytes(self.voice.read_bytes() + b"changed")
        frozen = collector.load_frozen(self.frozen_path)
        with self.assertRaisesRegex(ValueError, "frozen input changed"):
            collector.verify_inputs(frozen, frozen["cases"], self.driver)

    def test_partial_bad_or_failed_native_run_never_validates(self):
        modes = ("native7", "missing_end", "missing_first", "duplicate_run", "wrong_settings",
                 "wrong_primary", "missing_wav", "truncated_wav", "wrong_final", "wrong_frames",
                 "bad_timing", "nan", "malformed", "zero_audio", "reuse_path", "escape_path")
        for mode in modes:
            with self.subTest(mode=mode):
                result, _, _, cases = self.collect(mode)
                self.assertNotEqual("complete", result["status"])
                self.assertEqual(0, result["validated_cells"])
                self.assertNotEqual("execution_validated", cases[0]["status"])
                if mode == "native7":
                    self.assertEqual(7, cases[0]["native_exit_code"])

    def test_bad_stream_order_never_validates(self):
        case = "builtin-turbo-q4_0-en-cpu-stream"
        self.frozen_path = self.freeze([case])
        result, _, _, _ = self.collect("bad_stream", ids=[case])
        self.assertEqual(0, result["validated_cells"])

    def test_sustained_duration_is_native_and_ends_at_requested_time(self):
        result, _, _, cases = self.collect(duration_seconds=0.12)
        self.assertEqual("complete", result["status"])
        record = cases[0]["validation"]
        self.assertEqual(3, record["phase_counts"]["sustained"])
        self.assertAlmostEqual(0.15, sum(r["native_s"] for r in record["runs"]
                                      if r["phase"] == "sustained"))

    def test_attempts_are_isolated_and_duplicates_do_not_fill_matrix(self):
        one, directory1, _, _ = self.collect()
        two, directory2, _, _ = self.collect()
        self.assertNotEqual(one["collection"], two["collection"])
        summary = self.summary(directory1, directory2)
        self.assertTrue(summary["comparisons"][0]["ambiguous_repeated_backend"])
        coverage = summary["primary_matrix_coverage"][0]
        self.assertEqual([self.default_ids[0]], coverage["duplicate_case_ids"])
        self.assertEqual(83, coverage["missing_primary_cells"])
        self.assertFalse(coverage["complete_primary_matrix"])

    def test_saved_wav_log_or_case_mutation_cannot_keep_cached_success(self):
        for artifact in ("wav", "log", "case"):
            with self.subTest(artifact=artifact):
                _, directory, collection, cases = self.collect()
                record = cases[0]
                if artifact == "wav":
                    path = Path(record["validation"]["final_audio"]["path"])
                    data = bytearray(path.read_bytes())
                    data[-1] ^= 1
                    path.write_bytes(data)
                elif artifact == "log":
                    Path(record["logs"][0]["path"]).write_bytes(b"changed log")
                else:
                    path = Path(collection["cases"][0]["result"])
                    path.write_bytes(path.read_bytes() + b"\n")
                if artifact == "case":
                    with self.assertRaises(ValueError):
                        self.summary(directory)
                else:
                    self.assertEqual(0, self.summary(directory)["counts"]["execution_validated"])

    def test_missing_or_changed_input_hash_bracket_invalidates_summary(self):
        for action in ("missing", "changed"):
            with self.subTest(action=action):
                _, directory, data, _ = self.collect()
                if action == "missing":
                    data.pop("input_hashes_after", None)
                else:
                    data["input_hashes_after"] = {"different": "0" * 64}
                collector.write_json(directory / "collection.json", data)
                summary = self.summary(directory)
                self.assertEqual(0, summary["counts"]["execution_validated"])
                self.assertEqual(1, summary["counts"]["errors"])

    def test_input_changed_during_collection_invalidates_all_native_successes(self):
        real_process = collector.process
        count = 0

        def changing_process(command, stdout, stderr, timeout, *, env=None):
            nonlocal count
            count += 1
            result = real_process([sys.executable, str(self.driver), *command[1:]],
                                  stdout, stderr, timeout, env=env)
            if count == 1:
                self.runtime.write_bytes(b"changed during collection")
            return result

        with mock.patch.object(collector, "process", side_effect=changing_process):
            result = collector.collect(self.args(ids=self.default_ids))
        data = collector.read_json(result["collection"])
        cases = [collector.read_json(c["result"]) for c in data["cases"]]
        self.assertEqual(3, len(cases))
        self.assertEqual(0, result["validated_cells"])
        self.assertEqual({0}, {c["native_exit_code"] for c in cases})
        self.assertEqual({"input_hash_error"}, {c["status"] for c in cases})
        self.assertFalse(data["input_hash_brackets_valid"])

    def test_comparisons_require_same_source_models_and_settings(self):
        _, all_backends, _, records = self.collect(ids=self.default_ids)
        self.assertEqual({"cpu": 0, "opencl": 99, "hexagon": 99},
            {r["case"]["backend"]: r["effective_driver_settings"]["n_gpu_layers"] for r in records})
        summary = self.summary(all_backends)
        self.assertEqual(1, len(summary["comparisons"]))
        self.assertTrue(summary["comparisons"][0]["all_three_backends_present"])
        _, cpu, _, _ = self.collect()
        for change in ("source", "model", "settings", "context", "runtime", "device", "environment"):
            with self.subTest(change=change):
                model_path = self.models / "chatterbox-t3-turbo-q4_0.gguf"
                previous = model_path.read_bytes()
                runtime_previous = self.runtime.read_bytes()
                spec_previous = self.spec_path.read_bytes()
                collect_options = {}
                if change == "model":
                    model_path.write_bytes(previous + b"new model")
                if change in ("settings", "context"):
                    new_spec = copy.deepcopy(self.spec)
                    new_spec["settings"]["seed" if change == "settings" else "n_ctx"] = (
                        43 if change == "settings" else 2048)
                    collector.write_json(self.spec_path, new_spec)
                if change == "runtime":
                    self.runtime.write_bytes(runtime_previous + b"new runtime")
                if change == "device":
                    collect_options["device_info"] = self.device_path
                if change == "environment":
                    collect_options["env"] = ["GGML_HEXAGON_HOSTBUF=0"]
                self.frozen_path = self.freeze(
                    speech_sha="c" * 40 if change == "source" else "a" * 40)
                _, opencl, _, _ = self.collect(ids=[self.default_ids[1]], **collect_options)
                compared = self.summary(cpu, opencl)
                self.assertEqual(2, len(compared["comparisons"]))
                self.assertFalse(any(c["all_three_backends_present"] for c in compared["comparisons"]))
                model_path.write_bytes(previous)
                self.runtime.write_bytes(runtime_previous)
                self.spec_path.write_bytes(spec_previous)

    def adb_collect(self, mode="ok", ids=None):
        args = self.args(ids=ids, transport="adb", device_info=self.device_path)
        real_process = collector.process
        self.counter += 1
        staging = self.root / ("remote-" + str(self.counter))
        staging.mkdir()
        calls = Counter()
        frozen = collector.load_frozen(self.frozen_path)
        by_name = {a["name"]: a["sha256"] for a in
                   list(frozen["assets"].values()) + frozen["runtime_artifacts"] + [frozen["driver"]]}
        remote_driver = args.remote_root + "/" + self.driver.name

        def adb_process(command, stdout, stderr, timeout, *, env=None):
            command = [str(x) for x in command]
            Path(stdout).write_bytes(b"")
            Path(stderr).write_bytes(b"")
            receipt = dict(command=command, returncode=0, timed_out=False)
            if "pull" in command:
                calls["pull"] += 1
                shutil.copytree(staging, Path(command[-1]), dirs_exist_ok=True)
            elif command[-1].startswith("sha256sum "):
                calls["hash"] += 1
                names = shlex.split(command[-1])[1:]
                lines = [by_name[Path(p).name] + "  " + p for p in names]
                if mode == "bad_after_hash" and calls["hash"] >= 2:
                    lines[0] = "0" * 64 + lines[0][64:]
                Path(stdout).write_text("\n".join(lines) + "\n", encoding="utf-8")
                if mode == "missing_after_hash" and calls["hash"] >= 2:
                    receipt["returncode"] = 1
            else:
                calls["native"] += 1
                tokens = shlex.split(command[-1])
                start = tokens.index(remote_driver)
                end = next(i for i in range(start, len(tokens)) if tokens[i].startswith(">native.stdout"))
                native = tokens[start:end]
                output_index = native.index("--out") + 1
                remote_output = native[output_index]
                native[output_index] = str(staging / "audio.wav")
                native[0] = str(self.driver)
                local_env = dict(os.environ, CHATTERBOX_COLLECTION_TEST_MODE="native7" if mode == "native7" else "")
                child = real_process([sys.executable, *native], staging / "native.stdout",
                    staging / "native.stderr", timeout, env=local_env)
                events = [json.loads(line) for line in
                          (staging / "native.stdout").read_text(encoding="utf-8").splitlines()]
                for event in events:
                    if "output_path" in event:
                        event["output_path"] = str(PurePosixPath(remote_output).parent /
                                                  Path(event["output_path"]).name)
                (staging / "native.stdout").write_text(
                    "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
                if mode != "missing_status":
                    (staging / "native.status").write_text(str(child["returncode"]) + "\n")
                if mode == "transport_failure":
                    receipt["returncode"] = 1
            return receipt

        with mock.patch.object(collector, "process", side_effect=adb_process):
            result = collector.collect(args)
        collection = collector.read_json(result["collection"])
        cases = [collector.read_json(c["result"]) for c in collection["cases"]]
        return result, collection, cases, calls

    def test_adb_native_receipt_is_separate_from_transport_exit(self):
        result, _, cases, calls = self.adb_collect()
        self.assertEqual("complete", result["status"])
        self.assertEqual(0, cases[0]["native_exit_code"])
        self.assertEqual(0, cases[0]["transport_process"]["returncode"])
        self.assertEqual(2, calls["hash"])
        self.assertEqual(1, calls["pull"])

    def test_adb_transport_failure_missing_status_or_nonzero_native_never_pass(self):
        for mode in ("transport_failure", "missing_status", "native7"):
            with self.subTest(mode=mode):
                result, _, cases, _ = self.adb_collect(mode)
                self.assertEqual(0, result["validated_cells"])
                self.assertNotEqual("complete", result["status"])
                if mode == "native7":
                    self.assertEqual(7, cases[0]["native_exit_code"])
                    self.assertEqual(0, cases[0]["transport_process"]["returncode"])
                else:
                    self.assertIsNone(cases[0]["native_exit_code"])

    def test_bad_remote_after_bracket_invalidates_native_success(self):
        for mode in ("bad_after_hash", "missing_after_hash"):
            with self.subTest(mode=mode):
                result, data, cases, calls = self.adb_collect(mode, ids=self.default_ids)
                self.assertEqual(0, result["validated_cells"])
                self.assertNotEqual("complete", result["status"])
                self.assertEqual(3, len(cases))
                self.assertEqual({0}, {c["native_exit_code"] for c in cases})
                self.assertEqual({"input_hash_error"}, {c["status"] for c in cases})
                self.assertEqual(2, calls["hash"])
                self.assertFalse(data["input_hash_brackets_valid"])

    def test_changed_remote_hash_receipt_invalidates_saved_success(self):
        result, data, _, _ = self.adb_collect()
        path = Path(data["hash_logs"][0]["path"])
        path.write_bytes(path.read_bytes() + b"changed receipt")
        summary = self.summary(Path(result["collection"]).parent)
        self.assertEqual(0, summary["counts"]["execution_validated"])

    def test_adb_requires_labeled_device_identity(self):
        args = self.args(transport="adb", device_info=None)
        with self.assertRaisesRegex(ValueError, "device-info"):
            collector.collect(args)
        collector.write_json(self.device_path, {"model": "phone"})
        with self.assertRaisesRegex(ValueError, "soc"):
            collector.collect(self.args(transport="adb", device_info=self.device_path))

    def test_environment_override_names_and_duplicates_are_refused(self):
        for values in (["NOT AN ENV=x"], ["KEY=one", "KEY=two"], ["NO_EQUALS"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                collector.parse_env(values)

    def test_fit_harness_malformed_backend_context_arguments_exit_two(self):
        if FIT_BINARY is None:
            self.skipTest("native fit-test binary not supplied by standalone consumer build")
        import subprocess
        for args in (["--backend"], ["--backend", "--n-ctx"], ["--n-ctx", "-1"],
                     ["--n-ctx", "2147483648"], ["--unknown"]):
            with self.subTest(args=args):
                result = subprocess.run([str(FIT_BINARY), "missing-t3.gguf", "missing-s3.gguf", *args],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(2, result.returncode, result.stderr)
                self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fit-binary", type=Path)
    known, remaining = parser.parse_known_args()
    FIT_BINARY = known.fit_binary
    unittest.main(argv=[sys.argv[0], *remaining])
