#!/usr/bin/env python3
"""Freeze and collect native Chatterbox runs without turning partial output into a pass.

This tool checks measurement/artifact integrity. Numerical, intelligibility,
listening and actual-DSP-execution evidence remain separate requirements.
Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import statistics
import subprocess
import sys
import uuid
import wave

HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "fixtures" / "chatterbox-hexagon.json"
PHASES = ("first_call", "warmup", "measure", "sustained")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reject_constant(value):
    raise ValueError("non-finite JSON number: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, name, *, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), name + " must be a finite number")
    require(value > 0 if positive else value >= 0, name + " is out of range")
    return value


def integer(value, name, *, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            name + " must be an integer >= " + str(minimum))
    return value


def expand_cases(spec):
    require(spec.get("schema_version") == 1, "unsupported matrix schema")
    cases = []
    for group in spec["matrix"]:
        for family in group["families"]:
            languages = spec["families"][family] if group["languages"] == "family" else group["languages"]
            for language in languages:
                require(language in spec["families"][family], "unsupported family language")
                prompt_id = language + "_" + group["prompt"]
                prompt = spec["prompts"][prompt_id]
                require(prompt["language"] == language and prompt["text"].strip(), "invalid prompt")
                for tier in spec["tiers"]:
                    for backend in spec["backends"]:
                        for mode in group["modes"]:
                            case_id = "-".join([group["group"], family, tier, language, backend, mode])
                            require(re.fullmatch(r"[a-z0-9][a-z0-9_-]+", case_id), "invalid case id")
                            cases.append({"id": case_id, "group": group["group"], "family": family,
                                          "tier": tier, "language": language, "backend": backend,
                                          "mode": mode, "voice": group["voice"], "prompt_id": prompt_id,
                                          "text": prompt["text"], "models": spec["model_files"][family][tier]})
    require(len(cases) == spec["expected_primary_cells"], "matrix size differs from expected cells")
    require(len({c["id"] for c in cases}) == len(cases), "duplicate matrix case id")
    require(all(c["backend"] in ("cpu", "opencl", "hexagon") for c in cases), "invalid backend")
    return cases


def choose_cases(cases, ids):
    if not ids:
        return list(cases)
    require(len(ids) == len(set(ids)), "duplicate requested case id")
    by_id = {c["id"]: c for c in cases}
    missing = sorted(set(ids) - set(by_id))
    require(not missing, "unknown case id(s): " + ", ".join(missing))
    return [by_id[key] for key in ids]


def file_identity(path, *, allow_empty=False):
    path = Path(path).resolve()
    require(path.is_file() and (allow_empty or path.stat().st_size > 0), "missing or empty input: " + str(path))
    return {"path": str(path), "name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def freeze(args):
    spec = read_json(args.spec)
    all_cases = expand_cases(spec)
    cases = choose_cases(all_cases, args.case)
    model_names = sorted({v for c in cases for v in c["models"].values()})
    overrides = read_json(args.asset_map) if args.asset_map else {}
    assets = {name: file_identity(overrides.get(name, str(args.models_root / name))) for name in model_names}
    require(all(Path(a["path"]).name == key for key, a in assets.items()),
            "asset map must retain standard basenames (copy/link to the recorded model cache first)")
    reference = None
    if any(c["voice"] == "reference" for c in cases):
        require(args.reference_audio is not None, "reference-audio is required for cloning cases")
        reference = file_identity(args.reference_audio)
        validate_wav(args.reference_audio)
    for name, value in (("speech", args.speech_sha), ("ggml", args.ggml_sha)):
        require(re.fullmatch(r"[0-9a-fA-F]{40}", value), name + " revision must be a full Git SHA")
    payload = {"schema_version": 1, "kind": "chatterbox_hexagon_frozen", "created_utc": utc_now(),
               "matrix_spec": spec, "matrix_sha256": digest(spec), "primary_matrix_size": len(all_cases),
               "cases": cases, "sources": {"speech": args.speech_sha.lower(), "ggml": args.ggml_sha.lower()},
               "assets": assets, "reference_audio": reference, "driver": file_identity(args.driver_artifact),
               "runtime_artifacts": [file_identity(p) for p in args.runtime_artifact]}
    names = [x["name"] for x in payload["runtime_artifacts"]]
    require(len(names) == len(set(names)), "runtime artifact basenames must be unique")
    payload["frozen_sha256"] = digest(payload)
    write_json(args.out, payload, exclusive=True)
    return {"frozen": str(args.out), "frozen_sha256": payload["frozen_sha256"], "selected_cells": len(cases),
            "primary_matrix_size": len(all_cases), "measurements_started": False}


def load_frozen(path):
    frozen = read_json(path)
    require(frozen.get("schema_version") == 1 and frozen.get("kind") == "chatterbox_hexagon_frozen",
            "unsupported frozen manifest")
    expected = frozen.get("frozen_sha256")
    actual = digest({k: v for k, v in frozen.items() if k != "frozen_sha256"})
    require(expected == actual, "frozen manifest content changed")
    spec = frozen["matrix_spec"]
    require(digest(spec) == frozen["matrix_sha256"], "matrix hash mismatch")
    legal = {c["id"]: c for c in expand_cases(spec)}
    require(frozen["cases"] and len({c["id"] for c in frozen["cases"]}) == len(frozen["cases"]),
            "empty or duplicate frozen cases")
    require(all(legal.get(c["id"]) == c for c in frozen["cases"]), "frozen cases differ from matrix")
    return frozen


def verify_inputs(frozen, cases, driver):
    wanted = {name for c in cases for name in c["models"].values()}
    identities = [frozen["assets"][key] for key in sorted(wanted)] + frozen["runtime_artifacts"]
    if any(c["voice"] == "reference" for c in cases):
        require(frozen["reference_audio"] is not None, "missing frozen reference voice")
        identities.append(frozen["reference_audio"])
    identities.append(dict(frozen["driver"], path=str(driver)))
    for identity in identities:
        current = file_identity(identity["path"])
        require(current["sha256"] == identity["sha256"] and current["bytes"] == identity["bytes"],
                "frozen input changed: " + identity["path"])
    return {i["path"]: i["sha256"] for i in identities}


def validate_wav(path):
    with wave.open(str(path), "rb") as f:
        channels, width, rate, frames, compression, _ = f.getparams()
        require(compression == "NONE" and width in (1, 2, 3, 4), "WAV must be integer PCM")
        require(channels == 1 and rate > 0 and frames > 0, "WAV must contain mono audio")
        remaining = frames
        while remaining:
            n = min(remaining, 65536)
            require(len(f.readframes(n)) == n * channels * width, "truncated WAV payload")
            remaining -= n
    return {"frames": frames, "sample_rate": rate, "channels": channels,
            "sample_width_bytes": width, "audio_s": frames / rate, "sha256": sha256(path)}


def parse_events(path):
    events = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line, parse_constant=reject_constant)
        require(isinstance(event, dict), f"non-object event at line {number}")
        require(event.get("kind") in ("config", "load", "run", "end", "error"),
                f"unknown event kind at line {number}")
        events.append(event)
    return events


def validate_events(events, *, runs, warmup, duration_seconds, streaming, expected_output,
                    artifact_dir, native_root=None):
    for kind in ("config", "load", "end"):
        require(sum(e["kind"] == kind for e in events) == 1, "need exactly one " + kind + " event")
    require(len(events) >= 4 and events[0]["kind"] == "config" and events[1]["kind"] == "load"
            and events[-1]["kind"] == "end" and all(e["kind"] == "run" for e in events[2:-1]),
            "partial or unordered event stream")
    require(not any(e["kind"] == "error" for e in events), "driver emitted an error event")
    config = next(e for e in events if e["kind"] == "config")
    load = next(e for e in events if e["kind"] == "load")
    end = events[-1]
    require(config.get("schema_version") == 1, "unsupported driver schema")
    require(load.get("primary_backend_matches_request") is True, "requested primary backend was not confirmed")
    require(isinstance(load.get("actual_primary_backend"), str) and load["actual_primary_backend"],
            "missing primary backend identity")
    require(end.get("complete") is True and type(end.get("exit_code")) is int and end["exit_code"] == 0,
            "driver did not report successful completion")
    require(end.get("output_path") == str(expected_output), "final output path differs from requested path")
    records = [e for e in events if e["kind"] == "run"]
    require(records, "missing native run records")
    phase_counts = {phase: sum(e.get("phase") == phase for e in records) for phase in PHASES}
    require(phase_counts["first_call"] == 1 and phase_counts["warmup"] == warmup
            and phase_counts["measure"] == runs, "native phase counts differ from request")
    require(sum(phase_counts.values()) == len(records), "unknown native run phase")
    require([PHASES.index(e["phase"]) for e in records] == sorted(PHASES.index(e["phase"]) for e in records),
            "native phases are out of order")
    require(integer(end.get("total_calls"), "total_calls") == len(records), "native call count mismatch")
    require(integer(end.get("measurement_calls"), "measurement_calls") == runs, "measured call count mismatch")
    require(integer(end.get("sustained_calls"), "sustained_calls") == phase_counts["sustained"], "sustained count mismatch")
    seen = set()
    artifacts = []
    phase_time = {phase: 0.0 for phase in PHASES}
    for call, record in enumerate(records, 1):
        require(record.get("status") == "ok" and record.get("audio_contract_ok") is True
                and record.get("stream_contract_ok") is True, "native audio/stream contract did not pass")
        require(integer(record.get("call"), "call", minimum=1) == call, "native call sequence is incomplete")
        pair = (record["phase"], integer(record.get("index"), "run index"))
        require(pair not in seen, "duplicate native run record")
        seen.add(pair)
        native = finite_number(record.get("native_s"), "native_s", positive=True)
        begin = finite_number(record.get("phase_begin_s"), "phase_begin_s")
        end_time = finite_number(record.get("phase_end_s"), "phase_end_s")
        require(abs(begin - phase_time[record["phase"]]) <= 1e-5 * max(1, begin)
                and abs(end_time - begin - native) <= 1e-5 * max(1, end_time),
                "native phase timing is inconsistent")
        phase_time[record["phase"]] = end_time
        samples = integer(record.get("samples"), "samples", minimum=1)
        rate = integer(record.get("sample_rate"), "sample_rate", minimum=1)
        audio = finite_number(record.get("audio_s"), "audio_s", positive=True)
        require(abs(audio - samples / rate) <= 1 / rate + 1e-8, "audio duration differs from sample count")
        for key in ("t3_s", "s3gen_s", "peak_abs", "rms"):
            finite_number(record.get(key), key)
        require(record["rms"] > 0, "generated speech is silent")
        require(integer(record.get("non_finite_samples"), "non_finite_samples") == 0,
                "non-finite native PCM samples")
        require(integer(record.get("clipped_samples"), "clipped_samples") <= samples,
                "clipped sample count exceeds PCM length")
        require(record.get("stream_order_ok") is True and record.get("stream_pcm_matches") is True,
                "stream ordering or PCM concatenation check failed")
        chunks = integer(record.get("stream_chunks"), "stream_chunks")
        last = integer(record.get("last_chunks"), "last_chunks")
        if streaming:
            require(chunks > 0 and last == 1, "missing or invalid final stream chunk")
            require(finite_number(record.get("first_audio_s"), "first_audio_s") <= native + 1e-6,
                    "first audio occurs after native call")
        else:
            require(chunks == 0 and last == 0 and record.get("first_audio_s") is None,
                    "batch call unexpectedly reports streaming chunks")
        output = record.get("output_path")
        require(isinstance(output, str), "missing per-call output path")
        if native_root:
            remote = PurePosixPath(output)
            require(remote.parent == PurePosixPath(native_root), "native WAV escapes attempt directory")
            local = Path(artifact_dir) / remote.name
        else:
            local = Path(output)
            require(local.parent.resolve() == Path(artifact_dir).resolve(), "native WAV escapes attempt directory")
        metadata = validate_wav(local)
        require(metadata["frames"] == samples and metadata["sample_rate"] == rate,
                "WAV metadata differs from native run record")
        artifacts.append({"path": str(local), "phase": record["phase"], "index": record["index"], **metadata})
    require(len({a["path"] for a in artifacts}) == len(artifacts), "native runs reused WAV output")
    sustained = sum(e["native_s"] for e in records if e["phase"] == "sustained")
    require(abs(finite_number(end.get("sustained_native_s"), "sustained_native_s") - sustained)
            <= 1e-5 * max(1, sustained), "sustained time total mismatch")
    require(sustained + 1e-6 >= duration_seconds, "sustained collection ended before requested native duration")
    final = Path(artifact_dir) / PurePosixPath(str(expected_output)).name
    final_metadata = validate_wav(final)
    require(final_metadata["sha256"] == artifacts[-1]["sha256"], "final WAV differs from last native call")
    return {"phase_counts": phase_counts, "load": load, "runs": records, "artifacts": artifacts,
            "final_audio": {"path": str(final), **final_metadata}, "backend_execution_confirmed": False}


def validate_config(config, case, settings, command, warmup, runs, duration_seconds):
    values = {command[i]: command[i + 1] for i in range(1, len(command), 2)}
    require(config.get("requested_backend") == case["backend"], "driver backend request differs from case")
    for key, flag in (("model", "--model"), ("s3gen_gguf", "--s3gen-gguf"), ("output_path", "--out")):
        require(config.get(key) == values[flag], "driver configuration differs for " + key)
    require(config.get("text") == case["text"], "driver prompt differs from frozen case")
    require(config.get("reference_audio") == values.get("--reference-audio", ""), "driver reference voice differs")
    actual = config.get("settings")
    require(isinstance(actual, dict), "missing effective driver settings")
    expected = {key: settings[key] for key in ("seed", "threads", "n_ctx", "n_predict", "kv_cache_type", "max_sentence_chars", "cfm_steps")}
    for key in ("stream_chunk_tokens", "stream_first_chunk_tokens", "stream_left_context_tokens", "stream_cfm_steps"):
        expected[key] = settings[key] if case["mode"] == "stream" else 0
    expected.update(language=case["language"], warmup_calls=warmup, measurement_calls=runs,
                    first_call_calls=1, sustained_seconds=duration_seconds)
    for key, value in expected.items():
        require(actual.get(key) == value, "driver setting differs from frozen request: " + key)
    require(isinstance(config.get("environment"), dict), "missing driver environment snapshot")
    return actual


def process(command, stdout, stderr, timeout, *, env=None):
    receipt = {"command": command, "timeout_s": timeout, "started_utc": utc_now(), "returncode": None}
    try:
        with Path(stdout).open("wb") as out, Path(stderr).open("wb") as err:
            completed = subprocess.run(command, stdout=out, stderr=err, timeout=timeout, env=env, check=False)
        receipt["returncode"] = completed.returncode
        receipt["timed_out"] = False
    except subprocess.TimeoutExpired:
        receipt["timed_out"] = True
    except OSError as exc:
        receipt["error"] = str(exc)
    receipt["ended_utc"] = utc_now()
    return receipt


def adb_prefix(args):
    command = [str(args.adb)]
    if args.adb_port is not None:
        command += ["-P", str(args.adb_port)]
    return command + ["-s", args.serial]


def remote_paths(args, identities):
    return {name: str(PurePosixPath(args.remote_models_root or (args.remote_root + "/models")) / identity["name"])
            for name, identity in identities.items()}


def verify_remote(args, identities, directory, label):
    command = adb_prefix(args) + ["shell", "sha256sum " + " ".join(shlex.quote(p) for p in identities)]
    out, err = directory / (label + ".stdout"), directory / (label + ".stderr")
    receipt = process(command, out, err, 120)
    require(receipt["returncode"] == 0, "remote input hash command failed: " + str(receipt))
    actual = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.*)", line)
        require(match is not None, "invalid remote hash receipt")
        require(match[2] not in actual, "duplicate remote hash receipt")
        actual[match[2]] = match[1].lower()
    require(actual == identities, "deployed input hashes differ from frozen inputs")
    receipt["actual_sha256"] = actual
    return receipt


def remote_inputs(args, frozen, cases):
    deployed = remote_paths(args, frozen["assets"])
    driver = args.remote_driver or str(PurePosixPath(args.remote_root) / frozen["driver"]["name"])
    names = {name for case in cases for name in case["models"].values()}
    identities = {deployed[name]: frozen["assets"][name]["sha256"] for name in names}
    identities[driver] = frozen["driver"]["sha256"]
    identities.update({str(PurePosixPath(args.remote_root) / "lib" / a["name"]): a["sha256"]
                       for a in frozen["runtime_artifacts"]})
    reference = None
    if any(c["voice"] == "reference" for c in cases):
        require(frozen["reference_audio"] is not None, "missing frozen reference voice")
        reference = str(PurePosixPath(args.remote_models_root or args.remote_root + "/models")
                        / frozen["reference_audio"]["name"])
        identities[reference] = frozen["reference_audio"]["sha256"]
    return deployed, driver, reference, identities


def make_command(driver, case, settings, paths, output, reference, *, warmup, runs, duration_seconds):
    command = [str(driver), "--model", paths[case["models"]["t3"]], "--s3gen-gguf", paths[case["models"]["s3gen"]],
               "--backend", case["backend"], "--text", case["text"], "--language", case["language"], "--out", str(output),
               "--warmup", str(warmup), "--runs", str(runs), "--duration-seconds", str(duration_seconds)]
    for key in ("seed", "threads", "n_ctx", "n_predict", "kv_cache_type", "max_sentence_chars", "cfm_steps"):
        command += ["--" + key.replace("_", "-"), str(settings[key])]
    if case["mode"] == "stream":
        for key in ("stream_chunk_tokens", "stream_first_chunk_tokens", "stream_left_context_tokens", "stream_cfm_steps"):
            command += ["--" + key.replace("_", "-"), str(settings[key])]
    if case["voice"] == "reference":
        require(reference is not None, "reference voice path is missing")
        command += ["--reference-audio", str(reference)]
    return command


def parse_env(values):
    result = {}
    for item in values:
        name, separator, value = item.partition("=")
        require(separator and ENV_NAME.fullmatch(name), "environment override must be NAME=VALUE")
        require(name not in result, "duplicate environment override: " + name)
        result[name] = value
    return result


def relevant_environment(env):
    return {k: v for k, v in sorted(env.items())
            if k.startswith(("GGML_", "TTS_CPP_")) or k in ("LD_LIBRARY_PATH", "ADSP_LIBRARY_PATH", "DSP_LIBRARY_PATH")}


def collect(args):
    frozen = load_frozen(args.frozen)
    cases = choose_cases(frozen["cases"], args.case)
    for key in ("timeout", "duration_seconds"):
        finite_number(getattr(args, key), key, positive=key == "timeout")
    integer(args.runs, "runs", minimum=1)
    integer(args.warmup, "warmup")
    overrides = parse_env(args.env)
    driver = Path(args.driver or frozen["driver"]["path"]).resolve()
    before = verify_inputs(frozen, cases, driver)
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
    directory = (args.out_dir / attempt_id).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    device = read_json(args.device_info) if args.device_info else {
        "kind": "host" if args.transport == "local" else "unrecorded_adb_device",
        "system": platform.system(), "machine": platform.machine(), "node": platform.node()}
    require(isinstance(device, dict) and device, "device-info must be a nonempty JSON object")
    if args.transport == "adb":
        require(args.serial, "ADB transport requires --serial")
        require(args.device_info is not None, "ADB collection requires --device-info with recorded model/SoC/HTP architecture")
        for key in ("model", "soc", "htp_architecture"):
            require(isinstance(device.get(key), str) and device[key].strip(), "device-info needs a recorded " + key)
        require(args.remote_root and PurePosixPath(args.remote_root).is_absolute()
                and ".." not in PurePosixPath(args.remote_root).parts, "remote-root must be an absolute deployment directory")
    metadata = {"schema_version": 1, "kind": "chatterbox_hexagon_collection", "created_utc": utc_now(),
                "frozen_sha256": frozen["frozen_sha256"], "frozen": str(args.frozen.resolve()),
                "sources": frozen["sources"], "matrix_sha256": frozen["matrix_sha256"],
                "expected_primary_cells": frozen["primary_matrix_size"], "requested_cases": [c["id"] for c in cases],
                "device": device, "transport": args.transport, "status": "incomplete", "cases": [],
                "input_hashes_before": before, "environment_overrides": overrides,
                "hash_scope": "collection_bracketed_not_per_case",
                "meaning": "artifact/measurement integrity only; numerical, ASR/listening quality and DSP execution require separate evidence"}
    write_json(directory / "collection.json", metadata, exclusive=True)
    env = dict(os.environ, **overrides)
    remote_context = None
    try:
        if args.transport == "adb":
            remote_context = remote_inputs(args, frozen, cases)
            metadata["deployed_input_sha256"] = remote_context[3]
            metadata["deployed_hashes_before"] = verify_remote(args, remote_context[3], directory, "deployed-inputs-before")
            write_json(directory / "collection.json", metadata)
        for case in cases:
            case_dir = directory / case["id"]
            case_dir.mkdir()
            record = {"schema_version": 1, "case": case, "status": "error", "native_exit_code": None,
                      "frozen_sha256": frozen["frozen_sha256"], "sources": frozen["sources"], "device": device,
                      "settings": frozen["matrix_spec"]["settings"], "environment": overrides,
                      "model_sha256": {k: frozen["assets"][v]["sha256"] for k, v in case["models"].items()},
                      "protocol": {"runs": args.runs, "warmup": args.warmup, "duration_seconds": args.duration_seconds},
                      "directory": str(case_dir), "quality_gate": "not_evaluated", "htp_execution": "not_confirmed"}
            record["hash_scope"] = "collection_bracketed_not_per_case"
            write_json(case_dir / "case.json", record, exclusive=True)
            try:
                local_assets = {k: v["path"] for k, v in frozen["assets"].items()}
                reference = frozen["reference_audio"]["path"] if frozen["reference_audio"] else None
                native_root = None
                artifact_dir = case_dir
                if args.transport == "local":
                    output = case_dir / "audio.wav"
                    command = make_command(driver, case, frozen["matrix_spec"]["settings"], local_assets, output,
                                           reference, warmup=args.warmup, runs=args.runs, duration_seconds=args.duration_seconds)
                    record["environment"] = relevant_environment(env)
                    receipt = process(command, case_dir / "native.stdout", case_dir / "native.stderr", args.timeout, env=env)
                    record["native_process"] = receipt
                    record["native_exit_code"] = receipt["returncode"]
                    require(receipt["returncode"] == 0 and not receipt.get("timed_out"), "native process failed or timed out")
                else:
                    native_root = str(PurePosixPath(args.remote_root) / "results" / attempt_id / case["id"])
                    output = str(PurePosixPath(native_root) / "audio.wav")
                    deployed, remote_driver, remote_reference, _ = remote_context
                    reference = remote_reference if case["voice"] == "reference" else None
                    command = make_command(remote_driver, case, frozen["matrix_spec"]["settings"], deployed, output,
                                           reference, warmup=args.warmup, runs=args.runs, duration_seconds=args.duration_seconds)
                    record["native_command"] = command
                    shell = "umask 077; mkdir -p " + shlex.quote(native_root) + " && cd " + shlex.quote(native_root)
                    shell += " && { timeout -s TERM " + str(math.ceil(args.timeout)) + " "
                    shell += shlex.join(["env"] + [k + "=" + v for k, v in sorted(overrides.items())] + command)
                    shell += " >native.stdout 2>native.stderr; native_status=$?; printf '%s\\n' \"$native_status\" >native.status; };"
                    receipt = process(adb_prefix(args) + ["shell", shell], case_dir / "transport.stdout", case_dir / "transport.stderr", args.timeout + 30)
                    record["transport_process"] = receipt
                    require(receipt["returncode"] == 0 and not receipt.get("timed_out"), "ADB transport failed; remote outcome may be unknown")
                    artifact_dir = case_dir / "artifacts"
                    artifact_dir.mkdir()
                    pull = process(adb_prefix(args) + ["pull", native_root + "/.", str(artifact_dir)],
                                   case_dir / "pull.stdout", case_dir / "pull.stderr", 180)
                    record["pull_process"] = pull
                    require(pull["returncode"] == 0, "ADB artifact transfer failed")
                    status_text = (artifact_dir / "native.status").read_text(encoding="ascii").strip()
                    require(re.fullmatch(r"[0-9]{1,3}", status_text) and int(status_text) <= 255, "invalid native exit receipt")
                    record["native_exit_code"] = int(status_text)
                    require(record["native_exit_code"] == 0, "remote native process returned nonzero")
                events = parse_events(artifact_dir / "native.stdout")
                require(events, "driver emitted no JSON events")
                record["effective_driver_settings"] = validate_config(events[0], case,
                    frozen["matrix_spec"]["settings"], command, args.warmup, args.runs, args.duration_seconds)
                record["driver_environment"] = events[0]["environment"]
                for key, value in overrides.items():
                    if key in record["driver_environment"]:
                        require(record["driver_environment"][key] == value, "driver environment override differs: " + key)
                record["validation"] = validate_events(events, runs=args.runs, warmup=args.warmup,
                    duration_seconds=args.duration_seconds, streaming=case["mode"] == "stream",
                    expected_output=output, artifact_dir=artifact_dir, native_root=native_root)
                group = {k: v for k, v in case.items() if k not in ("id", "backend")}
                # Explicit CPU selection uses zero offload layers, whereas
                # OpenCL/HTP use 99. This is the backend under comparison,
                # not a workload difference. Preserve the full settings in
                # the case receipt while keeping all other settings strict.
                comparison_settings = {k: v for k, v in record["effective_driver_settings"].items()
                                       if k != "n_gpu_layers"}
                record["comparison_key"] = digest({"case": group, "sources": frozen["sources"],
                    "models": record["model_sha256"], "settings": comparison_settings, "device": device,
                    "driver": frozen["driver"]["sha256"], "runtime": frozen["runtime_artifacts"],
                    "reference": frozen["reference_audio"]["sha256"] if case["voice"] == "reference" else None,
                    "environment": record["environment"], "driver_environment": record["driver_environment"],
                    "runs": args.runs, "warmup": args.warmup,
                    "duration_seconds": args.duration_seconds})
                # A cell cannot become validated until the collection-wide
                # after hashes confirm its immutable input bracket.
                record["status"] = "execution_provisional"
            except (OSError, ValueError, KeyError, EOFError, wave.Error) as exc:
                record["reason"] = str(exc)
            finally:
                record["logs"] = [file_identity(p, allow_empty=True) for parent in (case_dir, case_dir / "artifacts")
                                  if parent.is_dir() for p in sorted(parent.iterdir())
                                  if p.is_file() and p.suffix in (".stdout", ".stderr", ".status")]
                write_json(case_dir / "case.json", record)
                metadata["cases"].append({"id": case["id"], "status": record["status"],
                                          "result": str(case_dir / "case.json"),
                                          "result_sha256": sha256(case_dir / "case.json")})
                write_json(directory / "collection.json", metadata)
            if record["status"] != "execution_provisional" and (not args.continue_on_error or record["native_exit_code"] is None):
                break
        metadata["status"] = "complete" if len(metadata["cases"]) == len(cases) and all(c["status"] == "execution_provisional" for c in metadata["cases"]) else "incomplete_or_failed"
    except (OSError, ValueError, KeyboardInterrupt) as exc:
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        metadata["reason"] = str(exc)
    finally:
        metadata["native_collection_status"] = metadata["status"]
        try:
            metadata["input_hashes_after"] = verify_inputs(frozen, cases, driver)
        except (OSError, ValueError) as exc:
            metadata["input_hashes_after_error"] = str(exc)
        if remote_context is not None:
            try:
                metadata["deployed_hashes_after"] = verify_remote(args, remote_context[3], directory, "deployed-inputs-after")
            except (OSError, ValueError) as exc:
                metadata["deployed_hashes_after_error"] = str(exc)
        brackets_valid = metadata.get("input_hashes_after") == metadata["input_hashes_before"]
        if args.transport == "adb":
            brackets_valid = brackets_valid and bool(metadata.get("deployed_hashes_before")) and bool(metadata.get("deployed_hashes_after"))
        metadata["input_hash_brackets_valid"] = brackets_valid
        if not brackets_valid:
            metadata["status"] = "unverified_inputs"
        for entry in metadata["cases"]:
            record = read_json(entry["result"])
            if record["status"] == "execution_provisional":
                record["status"] = "execution_validated" if brackets_valid else "input_hash_error"
                if not brackets_valid:
                    record["reason"] = "missing or mismatched collection-wide input hash bracket"
                write_json(entry["result"], record)
                entry.update(status=record["status"], result_sha256=sha256(entry["result"]))
        metadata["hash_logs"] = [file_identity(p, allow_empty=True) for p in sorted(directory.iterdir())
                                 if p.is_file() and p.suffix in (".stdout", ".stderr")]
        metadata["ended_utc"] = utc_now()
        write_json(directory / "collection.json", metadata)
    return {"collection": str(directory / "collection.json"), "status": metadata["status"],
            "validated_cells": sum(c["status"] == "execution_validated" for c in metadata["cases"]),
            "requested_cells": len(cases), "primary_matrix_size": frozen["primary_matrix_size"]}


def summarize(args):
    collections = [read_json(p / "collection.json") for p in args.attempt_dir]
    results = []
    for collection in collections:
        brackets_valid = (collection.get("input_hash_brackets_valid") is True
                          and bool(collection.get("input_hashes_after"))
                          and collection["input_hashes_before"] == collection["input_hashes_after"])
        if collection["transport"] == "adb":
            wanted = collection.get("deployed_input_sha256")
            brackets_valid = brackets_valid and bool(wanted) and all(
                collection.get(key, {}).get("returncode") == 0
                and collection.get(key, {}).get("actual_sha256") == wanted
                for key in ("deployed_hashes_before", "deployed_hashes_after"))
        try:
            for log in collection.get("hash_logs", []):
                require(sha256(log["path"]) == log["sha256"], "input hash log changed")
        except (OSError, ValueError):
            brackets_valid = False
        for entry in collection["cases"]:
            require(sha256(entry["result"]) == entry["result_sha256"], "saved case result changed")
            result = read_json(entry["result"])
            require(result["case"]["id"] == entry["id"], "case result identity mismatch")
            if not brackets_valid and result["status"] == "execution_validated":
                result = dict(result, status="input_hash_error", reason="missing or changed input hash bracket")
            # Recheck retained audio, so a deleted/modified artifact cannot keep a cached success.
            if result["status"] == "execution_validated":
                try:
                    for log in result["logs"]:
                        require(sha256(log["path"]) == log["sha256"], "saved native/transport log changed")
                    for artifact in result["validation"]["artifacts"] + [result["validation"]["final_audio"]]:
                        require(validate_wav(artifact["path"])["sha256"] == artifact["sha256"], "saved WAV changed")
                except (OSError, ValueError, EOFError, wave.Error) as exc:
                    result = dict(result, status="artifact_error", reason=str(exc))
            results.append(result)
    groups = {}
    for result in results:
        if result["status"] != "execution_validated":
            continue
        groups.setdefault(result["comparison_key"], []).append(result)
    comparisons = []
    for key, members in groups.items():
        backends = [r["case"]["backend"] for r in members]
        duplicate = len(backends) != len(set(backends))
        rows = []
        for result in members:
            for phase in ("measure", "sustained"):
                runs = [r for r in result["validation"]["runs"] if r["phase"] == phase]
                if not runs:
                    continue
                native = [r["native_s"] for r in runs]
                audio = sum(r["audio_s"] for r in runs)
                rows.append({"backend": result["case"]["backend"], "phase": phase, "calls": len(runs),
                    "native_mean_s": statistics.mean(native), "native_median_s": statistics.median(native),
                    "native_min_s": min(native), "native_max_s": max(native), "audio_s": audio,
                    "rtf": sum(native) / audio, "audio_seconds_per_native_second": audio / sum(native),
                    "first_audio_s": [r["first_audio_s"] for r in runs], "result": result["directory"]})
        comparisons.append({"comparison_key": key, "case": {k: v for k, v in members[0]["case"].items() if k not in ("id", "backend")},
                            "ambiguous_repeated_backend": duplicate, "all_three_backends_present": set(backends) == {"cpu", "opencl", "hexagon"} and not duplicate, "rows": rows})
    coverage = {}
    for result in results:
        key = digest({"frozen": result["frozen_sha256"], "device": result["device"],
                      "environment": result["environment"], "protocol": result["protocol"]})
        entry = coverage.setdefault(key, {"key": key, "case_ids": [], "validated_case_ids": [], "errors": 0})
        entry["case_ids"].append(result["case"]["id"])
        if result["status"] == "execution_validated":
            entry["validated_case_ids"].append(result["case"]["id"])
        else:
            entry["errors"] += 1
    primary_size = max((c["expected_primary_cells"] for c in collections), default=84)
    for entry in coverage.values():
        unique = set(entry["case_ids"])
        entry["primary_matrix_size"] = primary_size
        entry["unique_cases"] = len(unique)
        entry["missing_primary_cells"] = max(0, primary_size - len(unique))
        entry["duplicate_case_ids"] = sorted(k for k in unique if entry["case_ids"].count(k) > 1)
        entry["complete_primary_matrix"] = (len(set(entry["validated_case_ids"])) == primary_size
                                             and not entry["duplicate_case_ids"] and not entry["errors"])
    summary = {"schema_version": 1, "kind": "chatterbox_hexagon_summary", "created_utc": utc_now(),
               "collections": [str(p / "collection.json") for p in args.attempt_dir],
               "counts": {"execution_validated": sum(r["status"] == "execution_validated" for r in results),
                          "errors": sum(r["status"] != "execution_validated" for r in results), "records": len(results)},
               "collection_statuses": [c["status"] for c in collections], "comparisons": comparisons,
               "primary_matrix_coverage": list(coverage.values()),
               "failures": [{"id": r["case"]["id"], "status": r["status"], "reason": r.get("reason")} for r in results if r["status"] != "execution_validated"],
               "quality_gate": "not_evaluated", "htp_execution": "not_confirmed",
               "meaning": "Structural collection validation only. Missing/duplicate cells, ASR/WER, listening, numerical gates and actual DSP execution require separate review."}
    write_json(args.out, summary, exclusive=True)
    return {"summary": str(args.out), "counts": summary["counts"], "quality_gate": "not_evaluated"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    listing = commands.add_parser("list", help="list fixed primary matrix cells")
    listing.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    freezing = commands.add_parser("freeze", help="hash inputs before evaluation; refuse overwriting a manifest")
    freezing.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    freezing.add_argument("--models-root", type=Path, required=True)
    freezing.add_argument("--asset-map", type=Path)
    freezing.add_argument("--reference-audio", type=Path)
    freezing.add_argument("--driver-artifact", type=Path, required=True)
    freezing.add_argument("--runtime-artifact", type=Path, action="append", default=[])
    freezing.add_argument("--speech-sha", required=True)
    freezing.add_argument("--ggml-sha", required=True)
    freezing.add_argument("--case", action="append", default=[])
    freezing.add_argument("--out", type=Path, required=True)
    collecting = commands.add_parser("collect", help="run cells into a new, isolated attempt directory")
    collecting.add_argument("--frozen", type=Path, required=True)
    collecting.add_argument("--case", action="append", default=[])
    collecting.add_argument("--driver", type=Path)
    collecting.add_argument("--out-dir", type=Path, required=True)
    collecting.add_argument("--transport", choices=("local", "adb"), default="local")
    collecting.add_argument("--device-info", type=Path)
    collecting.add_argument("--env", action="append", default=[])
    collecting.add_argument("--timeout", type=float, default=1800)
    collecting.add_argument("--warmup", type=int, default=0)
    collecting.add_argument("--runs", type=int, default=1)
    collecting.add_argument("--duration-seconds", type=float, default=0)
    collecting.add_argument("--continue-on-error", action="store_true")
    collecting.add_argument("--adb", type=Path, default=Path("adb"))
    collecting.add_argument("--adb-port", type=int)
    collecting.add_argument("--serial")
    collecting.add_argument("--remote-root")
    collecting.add_argument("--remote-models-root")
    collecting.add_argument("--remote-driver")
    summary = commands.add_parser("summarize", help="report retained results without inventing a quality gate")
    summary.add_argument("--attempt-dir", type=Path, action="append", required=True)
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "list":
            cases = expand_cases(read_json(args.spec))
            for case in cases:
                print(json.dumps(case, ensure_ascii=False, sort_keys=True))
            return 0
        result = {"freeze": freeze, "collect": collect, "summarize": summarize}[args.action](args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 1 if args.action == "collect" and result["status"] != "complete" else 0
    except (OSError, ValueError, KeyError, TypeError, EOFError, wave.Error) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
