#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUDIO_DIR="${SCRIPT_DIR}/audio"
REFS_DIR="${SCRIPT_DIR}/references"
RESULTS_ROOT="${SCRIPT_DIR}/results"
LOCK_FILE="${AUDIO_DIR}/lock.json"

# Fixed benchmark configuration for reproducibility.
WARMUP_RUNS=1
MEASURED_RUNS=5
THREADS=8
PROCESSORS=1
VARIANT="metal-baseline"
MODEL_REL_DEFAULT="models/ggml-small.en.bin"
MAX_WER=0.02
MAX_CER=0.02
ENFORCE_CORRECTNESS=1
AUDIO_KEYS=()
CLI_BIN="${REPO_ROOT}/build/bin/whisper-cli"
CLI_ARGS=(
  "-l" "en"
  "-tp" "0"
  "-tpi" "0"
  "-nf"
  "-bs" "1"
  "-bo" "1"
  "-fa"
)

usage() {
  cat <<'EOF'
Usage:
  benchmark/bench.sh --create-lock
  benchmark/bench.sh [--variant <name>] [--audio <short|medium|long>]...
  benchmark/bench.sh [--variant <name>] --all-audio

Description:
  --create-lock  Validates benchmark/audio/{short,medium,long}.wav and writes benchmark/audio/lock.json
                 with file hashes + durations. This lock is required for benchmark runs.

  --variant      Logical variant label in output tables (default: metal-baseline).
  --audio        Add one audio key to run: short, medium, or long.
                 If omitted, only short is run (development default).
  --all-audio    Run short + medium + long.

Notes:
  - This harness runs 1 warm-up + 5 measured runs per audio sample.
  - Runs are always sequential and use fixed model + fixed CLI flags.
  - Audio files must be 16 kHz, mono, 16-bit WAV.
  - Correctness is computed via WER/CER against benchmark/references/{short,medium,long}.txt.
EOF
}

MODE="run"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --create-lock)
      MODE="create-lock"
      shift
      ;;
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --audio)
      case "$2" in
        short|medium|long)
          AUDIO_KEYS+=( "$2" )
          ;;
        *)
          echo "Invalid --audio value: $2 (expected short|medium|long)" >&2
          exit 2
          ;;
      esac
      shift 2
      ;;
    --all-audio)
      AUDIO_KEYS=( "short" "medium" "long" )
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

create_lock_file() {
  mkdir -p "${AUDIO_DIR}"
  require_file "${REPO_ROOT}/${MODEL_REL_DEFAULT}"
  require_file "${AUDIO_DIR}/short.wav"
  require_file "${AUDIO_DIR}/medium.wav"
  require_file "${AUDIO_DIR}/long.wav"

  python3 - "${LOCK_FILE}" "${REPO_ROOT}" "${MODEL_REL_DEFAULT}" "${AUDIO_DIR}" <<'PY'
import contextlib
import hashlib
import json
import os
import sys
import wave
from pathlib import Path

lock_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
model_rel = sys.argv[3]
audio_dir = Path(sys.argv[4])

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def inspect_wav(path: Path):
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
    if channels != 1:
        raise SystemExit(f"{path}: expected mono (1 channel), got {channels}")
    if sample_rate != 16000:
        raise SystemExit(f"{path}: expected 16000 Hz, got {sample_rate}")
    if sample_width != 2:
        raise SystemExit(f"{path}: expected 16-bit PCM (2 bytes), got {sample_width}")
    duration_s = frames / float(sample_rate)
    return {
        "sha256": sha256(path),
        "duration_s": duration_s,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
    }

model_path = repo_root / model_rel
if not model_path.is_file():
    raise SystemExit(f"Model not found: {model_path}")

lock = {
    "schema_version": 1,
    "model_rel": model_rel,
    "model_sha256": sha256(model_path),
    "audio": {
        "short": inspect_wav(audio_dir / "short.wav"),
        "medium": inspect_wav(audio_dir / "medium.wav"),
        "long": inspect_wav(audio_dir / "long.wav"),
    },
}

tmp = lock_path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, lock_path)
print(f"Wrote lock file: {lock_path}")
PY
}

validate_inputs_against_lock() {
  require_file "${LOCK_FILE}"
  require_file "${CLI_BIN}"

  local validation_json="$1"

  python3 - "${LOCK_FILE}" "${REPO_ROOT}" "${AUDIO_DIR}" "${MODEL_REL_DEFAULT}" > "${validation_json}" <<'PY'
import contextlib
import hashlib
import json
import sys
import wave
from pathlib import Path

lock_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
audio_dir = Path(sys.argv[3])
configured_model_rel = sys.argv[4]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def inspect_wav(path: Path):
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
    duration_s = frames / float(sample_rate)
    return channels, sample_rate, sample_width, duration_s

lock = json.loads(lock_path.read_text(encoding="utf-8"))
if lock.get("model_rel") != configured_model_rel:
    raise SystemExit(
        f"Lock model_rel mismatch: lock={lock.get('model_rel')} script={configured_model_rel}"
    )

model_rel = lock["model_rel"]
model_path = repo_root / model_rel
if not model_path.is_file():
    raise SystemExit(f"Model file not found: {model_path}")

actual_model_sha = sha256(model_path)
if actual_model_sha != lock["model_sha256"]:
    raise SystemExit(
        f"Model checksum mismatch for {model_path}. expected={lock['model_sha256']} actual={actual_model_sha}"
    )

validated_audio = {}
for key in ("short", "medium", "long"):
    wav_path = audio_dir / f"{key}.wav"
    if not wav_path.is_file():
        raise SystemExit(f"Audio file missing: {wav_path}")

    expected = lock["audio"][key]
    actual_sha = sha256(wav_path)
    if actual_sha != expected["sha256"]:
        raise SystemExit(
            f"Checksum mismatch for {wav_path}. expected={expected['sha256']} actual={actual_sha}"
        )

    channels, sample_rate, sample_width, duration_s = inspect_wav(wav_path)
    if channels != 1 or sample_rate != 16000 or sample_width != 2:
        raise SystemExit(
            f"Format mismatch for {wav_path}: channels={channels}, sample_rate={sample_rate}, sample_width={sample_width}"
        )

    validated_audio[key] = {
        "path": str(wav_path),
        "duration_s": duration_s,
        "sha256": actual_sha,
    }

print(
    json.dumps(
        {
            "model_rel": model_rel,
            "model_abs": str(model_path),
            "model_sha256": actual_model_sha,
            "audio": validated_audio,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
}

write_config_file() {
  local config_path="$1"
  local validation_json="$2"

  python3 - "${config_path}" "${validation_json}" "${VARIANT}" "${THREADS}" "${PROCESSORS}" "${WARMUP_RUNS}" "${MEASURED_RUNS}" "${REPO_ROOT}" "${CLI_BIN}" "${CLI_ARGS[@]}" <<'PY'
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
validation_path = Path(sys.argv[2])
variant = sys.argv[3]
threads = int(sys.argv[4])
processors = int(sys.argv[5])
warmup_runs = int(sys.argv[6])
measured_runs = int(sys.argv[7])
repo_root = sys.argv[8]
cli_bin = sys.argv[9]
cli_args = sys.argv[10:]

validation = json.loads(validation_path.read_text(encoding="utf-8"))

def run_cmd(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""

env = {
    "git_commit": run_cmd(["git", "-C", repo_root, "rev-parse", "HEAD"]),
    "git_short_commit": run_cmd(["git", "-C", repo_root, "rev-parse", "--short", "HEAD"]),
    "sw_vers": run_cmd(["sw_vers"]),
    "uname": run_cmd(["uname", "-a"]),
    "hw_memsize": run_cmd(["sysctl", "-n", "hw.memsize"]),
    "cpu_brand_string": run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]),
    "xcodebuild_version": run_cmd(["xcodebuild", "-version"]),
    "clang_version": run_cmd(["clang", "--version"]),
    "cmake_version": run_cmd(["cmake", "--version"]),
    "python_version": platform.python_version(),
}

cfg = {
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "variant": variant,
    "run_policy": {
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "sequential_execution": True,
    },
    "model": {
        "rel_path": validation["model_rel"],
        "abs_path": validation["model_abs"],
        "sha256": validation["model_sha256"],
    },
    "audio": validation["audio"],
    "cli": {
        "binary": cli_bin,
        "args": cli_args,
        "threads": threads,
        "processors": processors,
    },
    "environment": env,
}

config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_one() {
  local log_path="$1"
  local meta_path="$2"
  local audio_key="$3"
  local audio_duration="$4"
  local run_kind="$5"
  local run_index="$6"
  shift 6
  local cmd=( "$@" )

  python3 - "${log_path}" "${meta_path}" "${audio_key}" "${audio_duration}" "${run_kind}" "${run_index}" "${cmd[@]}" <<'PY'
import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

log_path = Path(sys.argv[1])
meta_path = Path(sys.argv[2])
audio_key = sys.argv[3]
audio_duration_s = float(sys.argv[4])
run_kind = sys.argv[5]
run_index = int(sys.argv[6])
cmd = sys.argv[7:]

segment_re = re.compile(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}\]")

start_utc = dt.datetime.now(dt.timezone.utc).isoformat()
start = time.perf_counter()
first_inference_s = None

with log_path.open("w", encoding="utf-8") as logf:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    for line in proc.stdout:
        logf.write(line)
        if first_inference_s is None and segment_re.search(line.strip()):
            first_inference_s = time.perf_counter() - start

    rc = proc.wait()

end = time.perf_counter()
end_utc = dt.datetime.now(dt.timezone.utc).isoformat()

meta = {
    "audio_key": audio_key,
    "audio_duration_s": audio_duration_s,
    "run_kind": run_kind,
    "run_index": run_index,
    "command": cmd,
    "log_path": str(log_path),
    "start_utc": start_utc,
    "end_utc": end_utc,
    "wall_clock_runtime_s": end - start,
    "first_inference_latency_s": first_inference_s,
    "exit_code": rc,
}

meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
sys.exit(rc)
PY
}

if [[ "${MODE}" == "create-lock" ]]; then
  create_lock_file
  exit 0
fi

if [[ "${#AUDIO_KEYS[@]}" -eq 0 ]]; then
  AUDIO_KEYS=( "short" )
fi

mkdir -p "${RESULTS_ROOT}"

RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}_${VARIANT}"
RAW_DIR="${RUN_DIR}/raw"
mkdir -p "${RAW_DIR}"

VALIDATION_JSON="${RUN_DIR}/validated_inputs.json"
validate_inputs_against_lock "${VALIDATION_JSON}"
write_config_file "${RUN_DIR}/config.json" "${VALIDATION_JSON}"

MODEL_PATH="$(python3 - "${VALIDATION_JSON}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(data["model_abs"])
PY
)"

echo "Benchmark run directory: ${RUN_DIR}"
echo "Variant: ${VARIANT}"
echo "Model: ${MODEL_PATH}"
echo "Warm-up runs: ${WARMUP_RUNS}, measured runs: ${MEASURED_RUNS}"
echo "Threads: ${THREADS}, processors: ${PROCESSORS}"
echo "Fixed CLI flags: ${CLI_ARGS[*]}"
echo "Audio set: ${AUDIO_KEYS[*]}"

for audio_key in "${AUDIO_KEYS[@]}"; do
  audio_path="${AUDIO_DIR}/${audio_key}.wav"
  audio_duration="$(python3 - "${VALIDATION_JSON}" "${audio_key}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(data["audio"][sys.argv[2]]["duration_s"])
PY
)"

  audio_run_dir="${RAW_DIR}/${audio_key}"
  mkdir -p "${audio_run_dir}"

  for i in $(seq 1 "${WARMUP_RUNS}"); do
    printf -v idx "%02d" "${i}"
    echo "[${audio_key}] warm-up ${i}/${WARMUP_RUNS}"
    run_one \
      "${audio_run_dir}/warmup_${idx}.log" \
      "${audio_run_dir}/warmup_${idx}.meta.json" \
      "${audio_key}" \
      "${audio_duration}" \
      "warmup" \
      "${i}" \
      "${CLI_BIN}" -m "${MODEL_PATH}" -f "${audio_path}" -t "${THREADS}" -p "${PROCESSORS}" "${CLI_ARGS[@]}"
  done

  for i in $(seq 1 "${MEASURED_RUNS}"); do
    printf -v idx "%02d" "${i}"
    echo "[${audio_key}] measured ${i}/${MEASURED_RUNS}"
    run_one \
      "${audio_run_dir}/run_${idx}.log" \
      "${audio_run_dir}/run_${idx}.meta.json" \
      "${audio_key}" \
      "${audio_duration}" \
      "measured" \
      "${i}" \
      "${CLI_BIN}" -m "${MODEL_PATH}" -f "${audio_path}" -t "${THREADS}" -p "${PROCESSORS}" "${CLI_ARGS[@]}"
  done
done

PARSE_ARGS=(
  "--run-dir" "${RUN_DIR}"
  "--refs-dir" "${REFS_DIR}"
  "--max-wer" "${MAX_WER}"
  "--max-cer" "${MAX_CER}"
)
if [[ "${ENFORCE_CORRECTNESS}" == "1" ]]; then
  PARSE_ARGS+=( "--enforce-correctness" )
fi
python3 "${SCRIPT_DIR}/parse_results.py" "${PARSE_ARGS[@]}"

echo "Completed benchmark parsing:"
echo "  ${RUN_DIR}/runs.csv"
echo "  ${RUN_DIR}/summary.csv"
echo "  ${RUN_DIR}/summary.md"
echo "  ${RUN_DIR}/correctness.json"
