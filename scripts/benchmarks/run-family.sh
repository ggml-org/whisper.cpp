#!/usr/bin/env bash
# Driver invoked by .github/workflows/speech-benchmark-desktop.yml for one
# (family, runner) matrix cell. Also runnable locally:
#
#   scripts/benchmarks/run-family.sh \
#       --family whisper --runs 3 --build-dir build \
#       --models-root $PWD/bench-models --out result.json
#
# Contract: emits one JSON file (--out) with the shape
#
#   {
#     "family":         "whisper",
#     "model":          "whisper-tiny",
#     "runner":         "linux",
#     "os":             "Linux",
#     "backend":        "CUDA" | "Metal" | "Vulkan" | "OpenCL" | "CPU" | "unknown"
#                       ("unknown" only on failed runs — a green run with no
#                        GPU-engagement evidence is reported as "CPU"),
#     "encoder_backend":"coreml-all" | "ggml-metal" | ... | "unknown",
#     "encoder_coreml_all_runs": true | false | null,
#     "encoder_ms_median": 32.1,       # native engines that report it
#     "baseline_encoder_ms_median": 121.3, # Darwin Core ML comparisons
#     "encoder_speedup": 3.78,         # baseline / Core ML
#     "wall_ms_median": 1234.5,
#     "wall_ms_min":    1210.0,
#     "wall_ms_max":    1301.2,
#     "rtf_median":     0.612,        # null when the family's rtf can't be measured
#     "peak_rss_mib":   1024.5,       # null on runners without /usr/bin/time -v/-l
#     "runs":           3,
#     "status":         "ok" | "not-in-registry" | "missing-model"
#                       | "build-failed" | "run-failed" | "fetch-failed"
#                       | "infrastructure-failed" (only written by the
#                         workflow's pre-flight seed, never by this driver),
#     "notes":          "..."
#   }
#
# The summarizer (scripts/benchmarks/summarize.py) reads these and produces
# the single markdown table for $GITHUB_STEP_SUMMARY.

set -euo pipefail

FAMILY=""
RUNS=3
WARMUP=1
BUILD_DIR="build"
MODELS_ROOT="$PWD/bench-models"
OUT="result.json"
RUNNER_LABEL="${RUNNER_OS:-unknown}"
AUDIO_DIR="engines/parakeet/test/samples"
WHISPER_SIZE="tiny"

usage() {
  cat <<EOF >&2
usage: $0 --family FAMILY [--runs N] [--warmup N]
          [--build-dir DIR] [--models-root DIR] [--audio-dir DIR]
          [--runner LABEL] [--whisper-size tiny|base|small]
          --out result.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family)         FAMILY="$2"; shift 2 ;;
    --runs)           RUNS="$2"; shift 2 ;;
    --warmup)         WARMUP="$2"; shift 2 ;;
    --build-dir)      BUILD_DIR="$2"; shift 2 ;;
    --models-root)    MODELS_ROOT="$2"; shift 2 ;;
    --audio-dir)      AUDIO_DIR="$2"; shift 2 ;;
    --runner)         RUNNER_LABEL="$2"; shift 2 ;;
    --whisper-size)   WHISPER_SIZE="$2"; shift 2 ;;
    --out)            OUT="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$FAMILY" ]]; then usage; exit 2; fi

# ---- read families.json -----------------------------------------------------
# BENCH_FAMILIES_JSON overrides the spec path so the driver's own test suite
# (test-run-family.sh) can exercise it against synthetic families and stub
# binaries without touching the real spec.
FAMILIES_JSON="${BENCH_FAMILIES_JSON:-$(dirname "$0")/families.json}"
if ! [[ -f "$FAMILIES_JSON" ]]; then
  echo "families.json not found at $FAMILIES_JSON" >&2; exit 1
fi
if ! command -v jq >/dev/null; then
  echo "jq is required" >&2; exit 1
fi

# Local-run guard: on pre-macOS-26 the system date's %N returns the literal
# "N" rather than nanoseconds, awk parses only the leading digits, and every
# time-wrapped measurement silently reports ~0.0 ms. Check once at start
# with the absolute /bin/date path (matches the invocation in now_ns()
# below) and fail loudly instead — the runners are known-good, but the
# driver is documented as locally runnable.
if ! [[ "$(/bin/date +%N 2>/dev/null)" =~ ^[0-9]+$ ]]; then
  echo "run-family.sh: /bin/date +%N does not return nanoseconds on this host" >&2
  echo "  (got: '$(/bin/date +%N 2>/dev/null)')." >&2
  echo "  Time-wrapped bench timings would be silently zero. Bailing." >&2
  exit 1
fi

spec_field() {
  jq -r --arg family "$FAMILY" --arg field "$1" \
    'if .[$family][$field] == null then "" else .[$family][$field] end' \
    "$FAMILIES_JSON"
}

benchmark_platform() {
  if [[ -n "${BENCH_TEST_PLATFORM:-}" ]]; then
    printf '%s\n' "$BENCH_TEST_PLATFORM"
  else
    uname -s
  fi
}
spec_field_raw() {
  # For fields that may be null (audio_duration_seconds) — return jq's raw
  # "null" instead of an empty string so downstream can distinguish.
  jq -r --arg family "$FAMILY" --arg field "$1" '.[$family][$field]' "$FAMILIES_JSON"
}

BENCH_KIND="$(spec_field bench_kind)"
BINARY_REL="$(spec_field binary)"
S3_PREFIX="$(spec_field s3_prefix)"
NOTES="$(spec_field notes)"
AUDIO_DURATION_S="$(spec_field_raw audio_duration_seconds)"   # "null" or a number

if [[ -z "$BENCH_KIND" ]]; then
  echo "family '$FAMILY' not found in families.json" >&2; exit 1
fi

# Config sanity: correctness is honored for both bench kinds — native families
# feed compute-wer.py their .transcript field from --json-out; time-wrapped
# families feed the last run's captured stdout. Anything else (TTS-style
# families with no transcript at all, or a hypothetical third bench kind)
# is flagged but not fatal — the perf portion still works.
if jq -e --arg family "$FAMILY" '.[$family].correctness // empty' \
     "$FAMILIES_JSON" > /dev/null 2>&1; then
  case "$BENCH_KIND" in
    native|time-wrapped) : ;;
    *)
      echo "warning: family '$FAMILY' declares correctness but bench_kind='$BENCH_KIND'" >&2
      echo "  correctness scoring requires a hypothesis source; block will be ignored." >&2
      ;;
  esac
fi

MODEL_DIR="$MODELS_ROOT/$FAMILY"
mkdir -p "$MODEL_DIR"

# Music diagnostics are explicitly opt-in; their longer, WAV-writing workload
# has a separate timing baseline from the default performance dispatch.
MUSIC_ENABLED=0
MUSIC_RESULT=null
MODEL_PREPARATION=null
MUSIC_OUTPUT_DIR=""
MUSIC_AUDIO_OUT=""
if [[ "${MUSIC_ALIGNMENT:-0}" == "1" ]] && jq -e --arg f "$FAMILY" '.[$f].music_alignment' "$FAMILIES_JSON" >/dev/null; then
  MUSIC_ENABLED=1
  MUSIC_RESULT="$(jq -n --argjson runs "$RUNS" '{status:"unavailable",score:null,reason:"generation has not completed",coverage:{scored:0,expected:$runs}}')"
  music_field() { jq -r --arg f "$FAMILY" --arg k "$1" '.[$f].music_alignment[$k]' "$FAMILIES_JSON"; }
  MUSIC_CAPTION="${MUSIC_CAPTION-$(music_field caption)}"
  MUSIC_SEED="${MUSIC_SEED-$(music_field seed)}"
  MUSIC_DURATION="${MUSIC_DURATION-$(music_field duration_seconds)}"
  MUSIC_MAX_FRAMES="${MUSIC_MAX_FRAMES-$(music_field max_frames)}"
  MUSIC_LYRICS="${MUSIC_LYRICS-$(music_field lyrics)}"
  MUSIC_DEVICE="${MUSIC_DEVICE:-cpu}"
  case "$MUSIC_DEVICE" in cpu|gpu) ;; *) echo "MUSIC_DEVICE must be cpu or gpu" >&2; exit 2 ;; esac
  MUSIC_PYTHON="${MUSIC_ALIGNMENT_PYTHON:-python3}"
  MUSIC_ARTIFACTS="${OUT%.json}.music"
  mkdir -p "$(dirname "$OUT")"
  # A unique directory prevents stale WAV reuse even after a failed rerun.
  MUSIC_ARTIFACTS="$(mktemp -d "${MUSIC_ARTIFACTS}.XXXXXX")"
  if [[ "$FAMILY" == "acestep" ]]; then AUDIO_DURATION_S="$MUSIC_DURATION"; fi
  NOTES="Music diagnostic workload: WAV writing included in timed generation; CLAP postpass excluded. Compare only matching generation settings; inspect retained generation.json."
fi

# ---- output-JSON emitter (used from every exit path) ------------------------
BACKEND="unknown"     # populated from JSON or stderr scrape on a successful run
ENCODER_BACKEND="unknown"
ENCODER_COREML_ALL_RUNS="null"
ENCODER_MS_MEDIAN="null"
BASELINE_BACKEND=""
BASELINE_ENCODER_BACKEND=""
BASELINE_ENCODER_MS_MEDIAN="null"
BASELINE_INFERENCE_MS_MEDIAN="null"
BASELINE_RTF_MEDIAN="null"
BASELINE_WER_MEDIAN="null"
ENCODER_SPEEDUP="null"
INFERENCE_SPEEDUP="null"
PEAK_RSS_MIB="null"   # tracked across runs; max seen
RTF_MEDIAN="null"     # from bench JSON (native) or computed (time-wrapped w/ audio_duration_seconds)
# Correctness fields — populated only when the family carries a `correctness`
# block AND the run produced enough output to score against a reference. All
# three stay null on families without a correctness spec so the result.json
# schema is stable (summarize.py renders "—" for null).
WER_MEDIAN="null"          # numeric WER in [0,1] or null (kind='wer' only)
DER_MEDIAN="null"          # numeric DER in [0,1] or null (kind='der' only)
F1_MEDIAN="null"           # numeric F1  in [0,1] or null (kind='f1'  only)
CORRECTNESS_KIND="null"    # "wer" | "der" | "f1" | null
AUDIO_QUALITY="null"
CORRECTNESS_REF="null"     # repo-relative path to the reference file or null
TTS_INTELLIGIBILITY="null"
TTS_KIND="$(jq -r --arg f "$FAMILY" '.[$f].correctness.kind // ""' "$FAMILIES_JSON")"
TTS_TEXT=""
TTS_AUDIO_OUT=""
if [[ "$TTS_KIND" == "tts_intelligibility" ]]; then
  CORRECTNESS_KIND='"tts_intelligibility"'
  CORRECTNESS_REF="$(jq --arg f "$FAMILY" '.[$f].correctness.reference' "$FAMILIES_JSON")"
  TTS_INTELLIGIBILITY='{"status":"unavailable","wer":null,"reason":"synthesis has not completed"}'
  # Clear only artifacts owned by this result, including early-failure paths.
  python3 - "$OUT" <<'PY'
import pathlib, sys
base = pathlib.Path(sys.argv[1])
stem = str(base)[:-5] if str(base).endswith('.json') else str(base)
for suffix in ('.tts.wav', '.tts-reference.txt', '.tts-transcript.txt', '.tts-asr.log', '.tts-intelligibility.json'):
    pathlib.Path(stem + suffix).unlink(missing_ok=True)
PY
fi
# Audio artifacts belong to this invocation, including when a later run fails.
case "$(jq -r --arg f "$FAMILY" '.[$f].correctness.kind // ""' "$FAMILIES_JSON")" in
  audio|sisdr|stoi)
    for suffix in reference.wav input.wav hypothesis.wav audio-quality.json input-preparation.json; do
      rm -f -- "${OUT%.json}.$suffix"
    done
    ;;
esac

emit_json() {
  local status="$1" median="${2:-null}" wmin="${3:-null}" wmax="${4:-null}" extra="${5:-}"
  local uname_s; uname_s="$(uname -s)"
  if [[ "$MUSIC_ENABLED" == 1 && "$status" != "ok" && "$status" != "preparation-failed" ]]; then
    MUSIC_RESULT="$(printf '%s' "$MUSIC_RESULT" | jq --arg status "$status" '.reason = ("generation unavailable: " + $status) | .stage = "generation" | .status = (if $status == "missing-model" then "model-missing" else "unavailable" end)')"
  fi
  jq -n \
    --arg  family "$FAMILY" \
    --arg  model  "$MODEL_LABEL" \
    --arg  runner "$RUNNER_LABEL" \
    --arg  os     "$uname_s" \
    --arg  backend "$BACKEND" \
    --arg  encoder_backend "$ENCODER_BACKEND" \
    --argjson encoder_coreml_all_runs "$ENCODER_COREML_ALL_RUNS" \
    --argjson encoder_ms_median "$ENCODER_MS_MEDIAN" \
    --arg  baseline_backend "$BASELINE_BACKEND" \
    --arg  baseline_encoder_backend "$BASELINE_ENCODER_BACKEND" \
    --argjson baseline_encoder_ms_median "$BASELINE_ENCODER_MS_MEDIAN" \
    --argjson baseline_inference_ms_median "$BASELINE_INFERENCE_MS_MEDIAN" \
    --argjson baseline_rtf_median "$BASELINE_RTF_MEDIAN" \
    --argjson baseline_wer_median "$BASELINE_WER_MEDIAN" \
    --argjson encoder_speedup "$ENCODER_SPEEDUP" \
    --argjson inference_speedup "$INFERENCE_SPEEDUP" \
    --argjson wall_median "$median" \
    --argjson wall_min    "$wmin" \
    --argjson wall_max    "$wmax" \
    --argjson rtf_median  "$RTF_MEDIAN" \
    --argjson peak_rss    "$PEAK_RSS_MIB" \
    --argjson runs        "$RUNS" \
    --argjson wer_median  "$WER_MEDIAN" \
    --argjson der_median  "$DER_MEDIAN" \
    --argjson f1_median   "$F1_MEDIAN" \
    --argjson audio_quality "$AUDIO_QUALITY" \
    --argjson correctness_kind "$CORRECTNESS_KIND" \
    --argjson correctness_reference "$CORRECTNESS_REF" \
    --argjson tts_intelligibility "$TTS_INTELLIGIBILITY" \
    --argjson music_alignment "$MUSIC_RESULT" \
    --argjson model_preparation "$MODEL_PREPARATION" \
    --arg  status "$status" \
    --arg  notes  "$NOTES$extra" \
    '{family:$family, model:$model, runner:$runner, os:$os, backend:$backend,
      encoder_backend:$encoder_backend,
      encoder_coreml_all_runs:$encoder_coreml_all_runs,
      encoder_ms_median:$encoder_ms_median,
      baseline_backend:(if $baseline_backend == "" then null else $baseline_backend end),
      baseline_encoder_backend:(if $baseline_encoder_backend == "" then null else $baseline_encoder_backend end),
      baseline_encoder_ms_median:$baseline_encoder_ms_median,
      baseline_inference_ms_median:$baseline_inference_ms_median,
      baseline_rtf_median:$baseline_rtf_median,
      baseline_wer_median:$baseline_wer_median,
      encoder_speedup:$encoder_speedup, inference_speedup:$inference_speedup,
      wall_ms_median:$wall_median, wall_ms_min:$wall_min, wall_ms_max:$wall_max,
      rtf_median:$rtf_median, peak_rss_mib:$peak_rss,
      wer_median:$wer_median, der_median:$der_median, f1_median:$f1_median,
      audio_quality:$audio_quality,
      correctness_kind:$correctness_kind,
      correctness_reference:$correctness_reference,
      tts_intelligibility:$tts_intelligibility, music_alignment:$music_alignment,
      model_preparation:$model_preparation,
      runs:$runs, status:$status, notes:$notes}' > "$OUT"
  echo "wrote $OUT"
  cat "$OUT" >&2
}

# The `source` field per family (default "s3"): "s3" for the tetherto S3
# registry via aws s3 cp, "huggingface" for `curl` against a pinned HF
# revision. Vanilla whisper GGMLs (tiny/base/small) live on HuggingFace,
# not S3 — the whisper family sets source="huggingface" and provides
# hf_repo / hf_ref / models_by_size[<size>] = {name, sha256?} entries.
SOURCE="$(spec_field source)"
[[ -n "$SOURCE" ]] || SOURCE="s3"

# ---- resolve MODEL_LABEL + placeholders -------------------------------------
# models[] entries may be strings or objects ({s3, as} for renamed S3 keys,
# {name, sha256?} for HuggingFace files) — the label wants the filename the
# bench actually loads.
if [[ "$FAMILY" == "whisper" ]]; then
  MODEL_LABEL="whisper-$WHISPER_SIZE"
else
  first_model="$(jq -r --arg f "$FAMILY" \
    '.[$f].models[0] // "" | if type == "object" then (.name // .as // .s3 // "") else . end' \
    "$FAMILIES_JSON")"
  MODEL_LABEL="$FAMILY: ${first_model##*/}"
fi

# ---- fetch models from HuggingFace (curl, optional sha256 verification) ----
# whisper picks one size-keyed entry from models_by_size; every other HF
# family lists its files in models[] as {name, sha256?} objects (or plain
# name strings), resolved against the family's pinned hf_repo@hf_ref.
fetch_from_hf() {
  local repo ref entries
  repo="$(spec_field hf_repo)"
  ref="$(spec_field hf_ref)"
  if [[ -z "$repo" || -z "$ref" ]]; then
    echo "$FAMILY: missing hf_repo / hf_ref in families.json" >&2
    return 1
  fi

  if [[ "$FAMILY" == "whisper" ]]; then
    local wname wsha
    wname="$(jq -r --arg size "$WHISPER_SIZE" '.whisper.models_by_size[$size].name // ""' "$FAMILIES_JSON")"
    wsha="$(jq -r --arg size "$WHISPER_SIZE" '.whisper.models_by_size[$size].sha256 // ""' "$FAMILIES_JSON")"
    if [[ -z "$wname" || "$wname" == "null" ]]; then
      echo "$FAMILY-$WHISPER_SIZE: no HF model name in registry (not-in-registry)"
      return 66
    fi
    entries="$wname"$'\t'"$wsha"
  else
    entries="$(jq -r --arg family "$FAMILY" \
      '.[$family].models[]? | if type == "object" then "\(.name // "")\t\(.sha256 // "")" else "\(.)\t" end' \
      "$FAMILIES_JSON")"
    if [[ -z "$entries" ]]; then
      return 65      # same shape as the S3 path: no files pinned yet
    fi
  fi

  local line name sha256 dest url
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    name="${line%%$'\t'*}"
    sha256="${line#*$'\t'}"
    if [[ -z "$name" ]]; then
      echo "$FAMILY: HF model entry without a name in families.json" >&2
      return 1
    fi
    dest="$MODEL_DIR/$name"
    if [[ -f "$dest" ]]; then continue; fi
    url="https://huggingface.co/${repo}/resolve/${ref}/${name}"
    echo "fetch $url -> $dest"
    # curl -f: fail on HTTP >= 400 rather than writing an HTML error page and
    # returning success. -L: follow redirects (HF uses them). --retry rides
    # out transient network resets. Explicit || return $? so a fetch failure
    # propagates instead of the loop swallowing it (bash's `set -e` is
    # disabled inside a function called with `||`).
    curl -fL -o "$dest" "$url" --silent --show-error \
      --retry 3 --retry-delay 5 || return $?

    if [[ -n "$sha256" && "$sha256" != "null" ]]; then
      if command -v sha256sum >/dev/null; then
        echo "$sha256  $dest" | sha256sum -c - || return $?
      elif command -v shasum >/dev/null; then
        echo "$sha256  $dest" | shasum -a 256 -c - || return $?
      fi
    fi
  done <<< "$entries"
  return 0
}

# aws s3 cp retries API-level throttles internally, but a mid-stream
# connection reset (acestep's ~5 GB shard has hit ConnectionResetError(104))
# aborts the transfer and leaves a partial file. Retry the whole copy,
# removing the partial first so a truncated file can never satisfy the
# already-downloaded check on the next attempt or the next dispatch.
s3_cp_retry() {
  local s3url="$1" dest="$2"
  local attempt rc=1
  for attempt in 1 2 3; do
    if aws s3 cp "$s3url" "$dest" --no-progress; then
      return 0
    else
      rc=$?
    fi
    rm -f "$dest"
    if [[ $attempt -lt 3 ]]; then
      echo "aws s3 cp $s3url failed (exit $rc, attempt $attempt/3) — retrying" >&2
      # BENCH_RETRY_SLEEP_S: test hook (test-run-family.sh sets 0).
      sleep $(( attempt * ${BENCH_RETRY_SLEEP_S:-5} ))
    fi
  done
  return $rc
}

# ---- fetch models from S3 (skipped when models[] empty / no bucket) --------
fetch_from_s3() {
  local bucket="${MODEL_S3_BUCKET:-}"
  if [[ -z "$bucket" ]]; then
    echo "MODEL_S3_BUCKET not set — skipping remote fetch (assuming local models present)" >&2
    return 0
  fi

  # `mapfile -t` is bash 4+, so the macOS self-hosted runner would break if
  # /usr/bin/env bash resolved to the system 3.2. Use a portable read loop.
  # Each row is "s3-key<TAB>rename-to" — rename-to is empty for string
  # entries and the engine-expected filename for object entries like
  # {"s3": "2026-07-23/voice-en.gguf", "as": "voice.gguf"} (cosyvoice's
  # baked voice is stored language-suffixed in S3 but the engine looks
  # for the canonical `voice.gguf`).
  local -a keys=()
  local line
  while IFS= read -r line; do keys+=("$line"); done < <(jq -r --arg family "$FAMILY" \
    '.[$family].models[]? | if type == "string" then "\(.)\t" else "\(.s3)\t\(.as // "")" end' \
    "$FAMILIES_JSON")
  if [[ ${#keys[@]} -eq 0 ]]; then
    return 65      # minimax shape: no S3 path
  fi

  for row in "${keys[@]}"; do
    local key="${row%%$'\t'*}"
    local rename_to="${row#*$'\t'}"
    local basename="${rename_to:-${key##*/}}"
    local dest="$MODEL_DIR/$basename"
    if [[ -f "$dest" ]]; then continue; fi
    local s3url="s3://$bucket/qvac_models_compiled/ggml/$S3_PREFIX/$key"
    echo "fetch $s3url -> $dest"
    # `|| return $?` so an S3 failure (404, denied, network) propagates as a
    # non-zero exit from the function — otherwise `set -e` is disabled by
    # the caller's `||` guard and the loop continues past a missing model,
    # eventually running the bench binary against nothing and mis-reporting
    # `run-failed` for what's actually a `fetch-failed`.
    s3_cp_retry "$s3url" "$dest" || return $?
  done

  return 0
}

fetch_models() {
  case "$SOURCE" in
    huggingface) fetch_from_hf ;;
    s3|"")       fetch_from_s3 ;;
    *) echo "unknown source '$SOURCE' for family '$FAMILY'" >&2; return 1 ;;
  esac
}

# ---- pick the model path for whisper (single -m arg) ------------------------
MODEL_PATH=""
if [[ "$FAMILY" == "whisper" ]]; then
  wname="$(jq -r --arg size "$WHISPER_SIZE" '.whisper.models_by_size[$size].name // ""' "$FAMILIES_JSON")"
  if [[ -n "$wname" && "$wname" != "null" ]]; then
    MODEL_PATH="$MODEL_DIR/$wname"
  fi
fi

fetch_status=0
if [[ "$FAMILY" == "minimax" && "$MUSIC_ENABLED" != 1 ]]; then
  emit_json "missing-model" null null null " (MiniMax performance-only invocation is not configured; use MUSIC_ALIGNMENT=1 with a prepared GGUF pair)"; exit 0
fi
if [[ "$MUSIC_ENABLED" == 1 && "$FAMILY" == "minimax" && -n "${MINIMAX_PREPARATION_REPORT:-}" ]]; then
  if ! jq -e 'type == "object" and (.status == "ok" or .status == "preparation-failed")' "$MINIMAX_PREPARATION_REPORT" >/dev/null 2>&1; then
    MODEL_PREPARATION='{"status":"preparation-failed","stage":"report","reason":"MiniMax preparation did not produce a valid report"}'
  else
    MODEL_PREPARATION="$(cat "$MINIMAX_PREPARATION_REPORT")"
  fi
  if [[ "$(printf '%s' "$MODEL_PREPARATION" | jq -r '.status')" == "ok" ]]; then
    prepared_dir="$(printf '%s' "$MODEL_PREPARATION" | jq -r '.model_dir // ""')"
    if [[ -n "$prepared_dir" && -d "$prepared_dir" ]]; then
      MUSIC_ALIGNMENT_MODEL_DIR="$prepared_dir"
      MODEL_LABEL="minimax: $(printf '%s' "$MODEL_PREPARATION" | jq -r '.quant // "unknown"')"
    else
      MODEL_PREPARATION='{"status":"preparation-failed","stage":"report","reason":"Prepared MiniMax model directory is missing"}'
    fi
  fi
  if [[ "$(printf '%s' "$MODEL_PREPARATION" | jq -r '.status')" != "ok" ]]; then
    MUSIC_RESULT="$(printf '%s' "$MUSIC_RESULT" | jq --arg reason "$(printf '%s' "$MODEL_PREPARATION" | jq -r '.reason')" '.stage = "preparation" | .reason = $reason | .status = "unavailable"')"
    emit_json "preparation-failed" null null null " (MiniMax preparation failed; see model_preparation)"; exit 0
  fi
fi
if [[ "$MUSIC_ENABLED" == 1 && -n "${MUSIC_ALIGNMENT_MODEL_DIR:-}" ]]; then
  MODEL_DIR="$MUSIC_ALIGNMENT_MODEL_DIR"
  if ! python3 "$(dirname "$0")/validate-music-models.py" "$MODEL_DIR" "$FAMILY"; then fetch_status=65; fi
elif [[ "$MUSIC_ENABLED" == 1 && "$FAMILY" == "minimax" ]]; then
  # S3 is optional; accept a provisioned, matching local GGUF pair.
  if ! python3 "$(dirname "$0")/validate-music-models.py" "$MODEL_DIR" "$FAMILY"; then fetch_status=65; fi
else
  fetch_models || fetch_status=$?
fi
if   [[ $fetch_status -eq 66 ]]; then
  emit_json "not-in-registry" null null null " (whisper $WHISPER_SIZE size not in the model registry)"; exit 0
elif [[ $fetch_status -eq 65 ]]; then
  emit_json "missing-model" null null null " (no usable model files provisioned; MiniMax music diagnostics accept a local matching GGUF pair)"; exit 0
elif [[ $fetch_status -ne 0 ]]; then
  emit_json "fetch-failed"    null null null " (model fetch failed with status $fetch_status)"; exit 0
fi

# Prepare CLAP assets before any timed generation. Failures remain diagnostic.
MUSIC_CLAP_DIR="${MUSIC_ALIGNMENT_MODEL:-${MUSIC_CLAP_MODEL_DIR:-}}"
MUSIC_MANIFEST="${MUSIC_ALIGNMENT_MANIFEST:-$(dirname "$0")/music-alignment-model.json}"
if [[ "$MUSIC_ENABLED" == 1 && -z "$MUSIC_CLAP_DIR" ]]; then
  music_preparation="$(python3 "$(dirname "$0")/prepare-music-alignment.py" --models-root "$MODELS_ROOT" --manifest "$MUSIC_MANIFEST" || true)"
  MUSIC_CLAP_DIR="$(printf '%s' "$music_preparation" | jq -r '.model_dir // ""' 2>/dev/null || true)"
  printf '%s\n' "$music_preparation" > "$MUSIC_ARTIFACTS/preparation.json"
fi

# ---- native-mode bench JSON path (resolved before args expansion) ----------
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
JSON_OUT="$tmp_dir/native.json"
AUDIO_OUT="$tmp_dir/unused.wav"
AUDIO_INPUT=""
AUDIO_REFERENCE=""
AUDIO_KIND="$(jq -r --arg f "$FAMILY" '.[$f].correctness.kind // ""' "$FAMILIES_JSON")"
AUDIO_PYTHON="${AUDIO_QUALITY_PYTHON:-python3}"
case "$AUDIO_KIND" in
  audio|sisdr|stoi)
    AUDIO_REFERENCE="$(jq -r --arg f "$FAMILY" '.[$f].correctness.reference // ""' "$FAMILIES_JSON")"
    if [[ "$AUDIO_REFERENCE" != /* ]]; then
      AUDIO_REFERENCE="$(cd "$(dirname "$0")/../.." && pwd)/$AUDIO_REFERENCE"
    fi
    AUDIO_INPUT="$AUDIO_REFERENCE"
    if jq -e --arg f "$FAMILY" '.[$f].correctness.degradation != null' "$FAMILIES_JSON" >/dev/null; then
      AUDIO_INPUT="$tmp_dir/degraded.wav"
      snr="$(jq -r --arg f "$FAMILY" '.[$f].correctness.degradation.snr_db' "$FAMILIES_JSON")"
      seed="$(jq -r --arg f "$FAMILY" '.[$f].correctness.degradation.seed' "$FAMILIES_JSON")"
      if ! "$AUDIO_PYTHON" "$(dirname "$0")/prepare-audio-quality.py" \
          --reference "$AUDIO_REFERENCE" --reference-output "$tmp_dir/reference.wav" --output "$AUDIO_INPUT" --snr-db "$snr" --seed "$seed" \
          > "${OUT%.json}.input-preparation.json"; then
        emit_json "run-failed" null null null " (audio input preparation failed)"
        exit 0
      fi
      AUDIO_REFERENCE="$tmp_dir/reference.wav"
    fi
    ;;
esac

if [[ "$TTS_KIND" == "tts_intelligibility" ]]; then
  tts_reference="$(jq -r --arg f "$FAMILY" '.[$f].correctness.reference' "$FAMILIES_JSON")"
  if [[ "$tts_reference" != /* ]]; then
    tts_reference="$(cd "$(dirname "$0")/../.." && pwd)/$tts_reference"
  fi
  if [[ ! -s "$tts_reference" ]]; then
    emit_json "run-failed" null null null " (TTS prompt fixture missing or empty)"; exit 0
  fi
  TTS_TEXT="$(cat "$tts_reference"; printf '.')"
  TTS_TEXT="${TTS_TEXT%.}"
  TTS_AUDIO_OUT="$tmp_dir/native.wav"
  tts_asr_preparation="$(python3 "$(dirname "$0")/prepare-tts-asr.py" \
    --spec "$FAMILIES_JSON" --family "$FAMILY" --models-root "$MODELS_ROOT" || true)"
  tts_asr_model="$(printf '%s' "$tts_asr_preparation" | jq -r '.model // ""')"
fi

# The Core ML bundle is derived from the family's configured source GGUF. A restored
# Actions cache makes this a no-op on subsequent runs; the first run exports and
# compiles locally on the Apple Silicon benchmark host.
if [[ "$(benchmark_platform)" == "Darwin" ]] &&
   [[ "$(spec_field coreml_compare_on_darwin)" == "true" ]]; then
  coreml_basename="$(spec_field coreml_model_basename)"
  coreml_source_gguf="$(spec_field coreml_source_gguf)"
  coreml_frames="$(spec_field coreml_export_mel_frames)"
  if [[ -z "$coreml_basename" || "$coreml_basename" == "null" ||
        -z "$coreml_frames" || "$coreml_frames" == "null" ]]; then
    emit_json "run-failed" null null null " (Core ML benchmark metadata is incomplete)"
    exit 0
  fi
  if [[ -z "$coreml_source_gguf" || "$coreml_source_gguf" == "null" ]]; then
    coreml_source_gguf="${coreml_basename}.f16.gguf"
  fi
  if [[ ! -f "$MODEL_DIR/$coreml_source_gguf" ]]; then
    emit_json "run-failed" null null null \
      " (Core ML source GGUF is missing: $MODEL_DIR/$coreml_source_gguf)"
    exit 0
  fi
  coreml_sidecar="$MODEL_DIR/${coreml_basename}-encoder.mlmodelc"
  if [[ ! -d "$coreml_sidecar" ]]; then
    coreml_python="${PARAKEET_COREML_PYTHON:-python3}"
    if [[ ! -x "$coreml_python" ]] && ! command -v "$coreml_python" >/dev/null 2>&1; then
      emit_json "run-failed" null null null " (Core ML exporter Python not found: $coreml_python)"
      exit 0
    fi
    coreml_package="$tmp_dir/${coreml_basename}-encoder.mlpackage"
    echo "generating Core ML encoder sidecar from the benchmark GGUF" >&2
    if ! "$coreml_python" engines/parakeet/scripts/export-encoder-coreml.py \
        --gguf "$MODEL_DIR/$coreml_source_gguf" \
        --n-mel-frames "$coreml_frames" \
        --palettize-bits 6 \
        --palettize-group-size 16 \
        --out "$coreml_package" \
        --compile-dir "$MODEL_DIR"; then
      emit_json "run-failed" null null null " (Core ML sidecar generation failed)"
      exit 0
    fi
  fi
  if [[ ! -d "$coreml_sidecar" ]]; then
    emit_json "run-failed" null null null " (Core ML exporter did not produce $coreml_sidecar)"
    exit 0
  fi
fi

# ---- build the argv array from the JSON-array `args` field -----------------
# families.json stores `args` as a JSON array so multi-word values (e.g.
# "benchmark run") stay a single argv token — a plain space-separated string
# would be word-split by the shell into ['benchmark], [run'] on
# expansion. Every element is passed through the same placeholder
# substitution table.
expand_placeholder() {
  local s="$1"
  case "$s" in
    '${MUSIC_CAPTION}') printf '%s' "$MUSIC_CAPTION"; return ;;
    '${MUSIC_LYRICS}') printf '%s' "$MUSIC_LYRICS"; return ;;
  esac
  s="${s//\$\{MUSIC_SEED\}/${MUSIC_SEED:-}}"
  s="${s//\$\{MUSIC_DURATION\}/${MUSIC_DURATION:-}}"
  s="${s//\$\{MUSIC_MAX_FRAMES\}/${MUSIC_MAX_FRAMES:-}}"
  s="${s//\$\{MUSIC_AUDIO_OUT\}/$MUSIC_AUDIO_OUT}"
  s="${s//\$\{MUSIC_OUTPUT_DIR\}/$MUSIC_OUTPUT_DIR}"
  if [[ "$s" == '${TTS_TEXT}' ]]; then printf '%s' "$TTS_TEXT"; return; fi
  s="${s//\$\{MODEL_DIR\}/$MODEL_DIR}"
  s="${s//\$\{MODELS_ROOT\}/$MODELS_ROOT}"
  s="${s//\$\{AUDIO_DIR\}/$AUDIO_DIR}"
  s="${s//\$\{AUDIO_INPUT\}/$AUDIO_INPUT}"
  s="${s//\$\{AUDIO_OUT\}/$AUDIO_OUT}"
  s="${s//\$\{MODEL_PATH\}/${MODEL_PATH:-}}"
  s="${s//\$\{RUNS\}/$RUNS}"
  s="${s//\$\{WARMUP\}/$WARMUP}"
  s="${s//\$\{JSON_OUT\}/$JSON_OUT}"
  s="${s//\$\{TTS_AUDIO_OUT\}/$TTS_AUDIO_OUT}"
  printf '%s' "$s"
}

BENCH_ARGS=()
build_bench_args() {
  BENCH_ARGS=()
  local line="" expanded=""
  while IFS= read -r line; do
    expanded="$(expand_placeholder "$line"; printf '.')"
    BENCH_ARGS+=("${expanded%.}")
  done < <(jq -r --arg family "$FAMILY" --argjson music "$MUSIC_ENABLED" ' .[$family] | if $music == 1 then .music_alignment.args[]? else .args[]? end' "$FAMILIES_JSON")
  if [[ "$MUSIC_ENABLED" == 1 ]]; then
    if [[ "$FAMILY" == "minimax" ]]; then BENCH_ARGS+=(--device "$MUSIC_DEVICE");
    elif [[ "$MUSIC_DEVICE" == gpu ]]; then BENCH_ARGS+=(--gpu); fi
  fi
}
build_bench_args

BINARY="$BUILD_DIR/$BINARY_REL"
if ! [[ -x "$BINARY" ]]; then
  echo "binary not found or not executable: $BINARY" >&2
  emit_json "build-failed" null null null " (binary '$BINARY' missing)"; exit 0
fi

if [[ "$MUSIC_ENABLED" == 1 ]]; then
  if ! python3 "$(dirname "$0")/music-generation-provenance.py" "$MODEL_DIR" "$BINARY" "$MUSIC_DEVICE" > "$MUSIC_ARTIFACTS/generation-provenance.json"; then
    emit_json "run-failed" null null null " (could not fingerprint music generation inputs)"; exit 0
  fi
fi

# ---- /usr/bin/time wrapper (portable RSS capture) --------------------------
# Linux GNU time: --verbose prints 'Maximum resident set size (kbytes): NNN'.
# macOS BSD time: -l prints '  NNN  maximum resident set size' in BYTES.
# Both write to stderr. We capture stderr into a file and parse after the run.
have_gnu_time() {
  /usr/bin/time --version >/dev/null 2>&1
}

wrap_time() {
  # Prepend the correct time invocation. When neither is available, skip.
  if [[ "$(uname -s)" == "Darwin" ]]; then
    /usr/bin/time -l "$@"
  elif have_gnu_time; then
    /usr/bin/time -v "$@"
  else
    "$@"
  fi
}

parse_rss_from_time_stderr() {
  # $1: path to a stderr log produced by wrap_time.
  # Emits the peak RSS in MiB (or empty when none can be parsed).
  local file="$1"
  local kb bytes
  # GNU time: kbytes
  kb="$(grep -oE 'Maximum resident set size \(kbytes\): [0-9]+' "$file" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)"
  if [[ -n "$kb" ]]; then
    awk -v k="$kb" 'BEGIN { printf "%.1f", k / 1024.0 }'
    return
  fi
  # BSD time: bytes on a line ending 'maximum resident set size'
  bytes="$(grep -oE '[0-9]+  *maximum resident set size' "$file" 2>/dev/null | grep -oE '^[0-9]+' | head -1 || true)"
  if [[ -n "$bytes" ]]; then
    awk -v b="$bytes" 'BEGIN { printf "%.1f", b / (1024.0 * 1024.0) }'
    return
  fi
  echo ""
}

# ---- backend extraction ----------------------------------------------------
# Attribution chain over the run's stdout AND stderr, most explicit first:
#   1. `using <NAME> backend`  — parakeet-cli / whisper / tts-cli style line.
#   2. `backend: <NAME>`       — the *-bench binaries print this on stdout
#      (matches lavasr's "denoiser backend: <NAME>" too).
#   3. GPU-engagement evidence — ggml's own compute-init logs, independent of
#      engine verbosity: creating a Metal compute context always logs
#      ggml_metal_library_init, Vulkan compiles shaders, OpenCL compiles its
#      program. Device *enumeration* alone (which also happens on CPU runs of
#      GPU-enabled builds) does not print these, so enumeration cannot
#      mis-label a CPU run.
# When every pattern misses, the caller labels a green run "CPU": with no
# compute-init evidence, nothing ran on a GPU. The residual risk is an engine
# that installs a silent ggml log handler AND prints no backend line of its
# own; none of the benched binaries do. Failed runs keep "unknown" — their
# logs may be truncated mid-init.
parse_backend_from_logs() {
  local hit f
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    hit="$(grep -oE 'Using GPU backend [A-Za-z0-9_]+' "$f" 2>/dev/null | head -1 | awk '{print $4}' || true)"
    if [[ -n "$hit" ]]; then echo "$hit"; return; fi
  done
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    hit="$(grep -oE 'using [A-Za-z0-9_]+ backend' "$f" 2>/dev/null | head -1 | awk '{print $2}' || true)"
    if [[ -n "$hit" ]]; then echo "$hit"; return; fi
  done
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    hit="$(grep -oE 'backend: [A-Za-z0-9_]+' "$f" 2>/dev/null | head -1 | awk '{print $2}' || true)"
    if [[ -n "$hit" ]]; then echo "$hit"; return; fi
  done
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    if grep -q  'ggml_metal_library_init'            "$f" 2>/dev/null; then echo "Metal";  return; fi
    if grep -q  'ggml_vulkan: Compiling shaders'     "$f" 2>/dev/null; then echo "Vulkan"; return; fi
    if grep -qiE 'ggml_opencl:.*(compil|program)'    "$f" 2>/dev/null; then echo "OpenCL"; return; fi
  done
  echo ""
}

# ---- correctness scoring (WER / DER / F1) ----------------------------------
# When a family in families.json carries a `correctness` block, score the
# hypothesis against the checked-in reference and echo "score|kind|reference"
# (or empty when scoring is skipped). Callers route the score into WER_MEDIAN
# (kind='wer') or DER_MEDIAN (kind='der') and set CORRECTNESS_* from kind/ref.
#
# Two hypothesis sources, one per bench kind:
#   mode="bench-json" <path>  — native families whose --json-out carries
#                               the hypothesis field (parakeet WER, via
#                               .transcript). DER has no native source today.
#   mode="text-file"  <path>  — time-wrapped families whose CLI prints its
#                               hypothesis on stdout (whisper -nt for WER;
#                               moss-transcribe's segment lines for WER with
#                               correctness.hypothesis_format='moss-segments';
#                               parakeet-cli --emit jsonl for sortformer DER).
#
# All error paths are non-fatal: a missing reference file, missing hypothesis
# field, missing stdout capture, or a compute-*.py failure logs to stderr and
# returns empty so the perf portion of the run still succeeds — a green
# benchmark with correctness=null tells the reader "perf is fine, correctness
# didn't run" instead of dragging the whole cell to run-failed.
score_correctness() {
  local mode="$1" src_path="$2"
  local spec_kind spec_ref spec_norm spec_format spec_collar spec_tolerance
  spec_kind="$(jq      -r --arg family "$FAMILY" \
    '.[$family].correctness.kind // ""'              "$FAMILIES_JSON")"
  spec_ref="$(jq       -r --arg family "$FAMILY" \
    '.[$family].correctness.reference // ""'         "$FAMILIES_JSON")"
  spec_norm="$(jq      -r --arg family "$FAMILY" \
    '.[$family].correctness.normalizer // "english"' "$FAMILIES_JSON")"
  spec_format="$(jq    -r --arg family "$FAMILY" \
    '.[$family].correctness.hypothesis_format // "plain"' "$FAMILIES_JSON")"
  spec_collar="$(jq    -r --arg family "$FAMILY" \
    '.[$family].correctness.collar_ms // 250'        "$FAMILIES_JSON")"
  spec_tolerance="$(jq -r --arg family "$FAMILY" \
    '.[$family].correctness.tolerance_ms // 100'     "$FAMILIES_JSON")"

  # No correctness block — silent no-op (the common case).
  if [[ -z "$spec_kind" ]]; then return 0; fi
  # The audio-based TTS path runs separately in the parent shell.
  if [[ "$spec_kind" == "tts_intelligibility" ]]; then return 0; fi

  case "$spec_kind" in
    audio|sisdr|stoi) return 0 ;;
    wer|der|f1) : ;;
    *)
      echo "$FAMILY: unsupported correctness.kind '$spec_kind' (known: wer, der, f1)" >&2
      return 0
      ;;
  esac

  # Resolve the reference path against the repo root (two levels up from
  # $(dirname "$0")/../..), or accept it as absolute. This mirrors how the
  # workflow invokes us with the repo checked out at $PWD.
  local ref_path="$spec_ref"
  if [[ "$ref_path" != /* ]]; then
    ref_path="$(cd "$(dirname "$0")/../.." && pwd)/$spec_ref"
  fi
  if ! [[ -f "$ref_path" ]]; then
    echo "$FAMILY: correctness reference file not found: $ref_path" >&2
    return 0
  fi

  # ---- dispatch by kind ----------------------------------------------------
  # Each kind builds its own compute-*.py argv and knows how to source the
  # hypothesis from the mode/src_path pair. Bench-json is WER-only today
  # (no native diarize schema); text-file works for both.
  local script_out score script_name
  case "$spec_kind" in
    wer)
      # Build --hypothesis-* args for compute-wer.py. Both modes preserve the
      # "present-but-empty transcript => WER 1.0" invariant: bench-json checks
      # field presence separately from truthiness so `.transcript=""` reaches
      # compute-wer.py rather than being conflated with field-absent; text-file
      # passes even an empty stdout capture through, which compute-wer.py
      # scores as a full miss when the reference is non-empty.
      local -a hyp_args=()
      case "$mode" in
        bench-json)
          local hyp
          if ! jq -e 'has("transcript") and (.transcript | type == "string")' \
                 "$src_path" > /dev/null 2>&1; then
            echo "$FAMILY: bench JSON has no .transcript field — correctness skipped" >&2
            return 0
          fi
          hyp="$(jq -r '.transcript' "$src_path" 2>/dev/null || true)"
          hyp_args=(--hypothesis-text "$hyp")
          ;;
        text-file)
          if ! [[ -f "$src_path" ]]; then
            echo "$FAMILY: hypothesis stdout capture missing: $src_path — correctness skipped" >&2
            return 0
          fi
          hyp_args=(--hypothesis-file "$src_path")
          ;;
        *)
          echo "$FAMILY: internal error: unknown score_correctness mode '$mode'" >&2
          return 0
          ;;
      esac
      # Route stderr through a temp file so per-line diagnostics from
      # compute-*.py (e.g. skipped non-JSON hypothesis lines) flow to the
      # workflow log for debuggability WITHOUT contaminating stdout, which
      # jq needs to parse cleanly as the tool's JSON result. Merging with
      # `2>&1` would break jq the moment compute-*.py prints a warning.
      local script_err
      script_err="$tmp_dir/${spec_kind}-score.err"
      script_name="compute-wer.py"
      if ! script_out="$(python3 "$(dirname "$0")/compute-wer.py" \
           "${hyp_args[@]}" \
           --reference   "$ref_path" \
           --normalizer  "$spec_norm" \
           --hypothesis-format "$spec_format" 2> "$script_err")"; then
        echo "$FAMILY: compute-wer.py failed: $(cat "$script_err" 2>/dev/null)" >&2
        return 0
      fi
      cat "$script_err" >&2 2>/dev/null || true
      score="$(echo "$script_out" | jq -r '.wer // empty' 2>/dev/null || true)"
      ;;

    der)
      # DER needs a JSONL hypothesis source: parakeet-cli's --emit jsonl on
      # stdout. Bench-json mode has no diarize-equivalent schema today, so
      # a native family declaring DER is a config error we surface.
      if [[ "$mode" != "text-file" ]]; then
        echo "$FAMILY: correctness.kind='der' requires text-file mode; got mode='$mode' — skipped" >&2
        return 0
      fi
      if ! [[ -f "$src_path" ]]; then
        echo "$FAMILY: hypothesis stdout capture missing: $src_path — correctness skipped" >&2
        return 0
      fi
      # See the WER branch above for the stderr-vs-stdout separation rationale.
      local script_err
      script_err="$tmp_dir/${spec_kind}-score.err"
      script_name="compute-der.py"
      if ! script_out="$(python3 "$(dirname "$0")/compute-der.py" \
           --hypothesis-jsonl "$src_path" \
           --reference        "$ref_path" \
           --collar-ms        "$spec_collar" 2> "$script_err")"; then
        echo "$FAMILY: compute-der.py failed: $(cat "$script_err" 2>/dev/null)" >&2
        return 0
      fi
      cat "$script_err" >&2 2>/dev/null || true
      score="$(echo "$script_out" | jq -r '.der // empty' 2>/dev/null || true)"
      ;;

    f1)
      # F1 (VAD) needs a text-file hypothesis source: whisper-vad-speech-segments
      # stdout with --no-prints (or JSON-array format for local tools).
      # Bench-json mode has no VAD-equivalent schema, mirroring DER's constraint.
      if [[ "$mode" != "text-file" ]]; then
        echo "$FAMILY: correctness.kind='f1' requires text-file mode; got mode='$mode' — skipped" >&2
        return 0
      fi
      if ! [[ -f "$src_path" ]]; then
        echo "$FAMILY: hypothesis stdout capture missing: $src_path — correctness skipped" >&2
        return 0
      fi
      # See the WER branch above for the stderr-vs-stdout separation rationale.
      local script_err
      script_err="$tmp_dir/${spec_kind}-score.err"
      script_name="compute-f1.py"
      if ! script_out="$(python3 "$(dirname "$0")/compute-f1.py" \
           --hypothesis-file "$src_path" \
           --reference       "$ref_path" \
           --collar-ms       "$spec_tolerance" 2> "$script_err")"; then
        echo "$FAMILY: compute-f1.py failed: $(cat "$script_err" 2>/dev/null)" >&2
        return 0
      fi
      cat "$script_err" >&2 2>/dev/null || true
      score="$(echo "$script_out" | jq -r '.f1 // empty' 2>/dev/null || true)"
      ;;
  esac

  if [[ -z "$score" ]]; then
    echo "$FAMILY: could not parse score from $script_name output: $script_out" >&2
    return 0
  fi

  # Emit the tuple the caller expects. Reference is echoed as the
  # families.json-declared (repo-relative) path so the summary table stays
  # stable across runners with different absolute checkout paths.
  echo "$score|$spec_kind|$spec_ref"
}

# Nanosecond timestamp. macOS 26 / arm64's /bin/date does support %N despite
# being a GNU extension; the script-start guard above bails loudly if we
# land on an older date that would silently return zeros. Invokes /bin/date
# by absolute path (matching the guard) so a stray coreutils `date` earlier
# on PATH can't diverge from what the guard actually verified.
now_ns() {
  /bin/date +%s%N
}

# ---- native bench: one invocation, parse the emitted JSON ------------------
# Native JSON schema tolerance — different bench binaries put the same
# information in different keys. We fall back through the known layouts:
#   * .stages.tot|.total|.e2e     -- supertonic/parler/cosyvoice
#   * .inference_ms               -- parakeet-cli --bench
#   * .end_to_end_ms              -- alternate parakeet naming (defensive)
# RTF likewise: .rtf.median (supertonic-shape) vs .rtf_median (parakeet).
# Backend likewise: .backend at top level (parakeet) — otherwise fall back
# to stderr parsing.
run_native() {
  local native_json="$1"; shift
  local stderr_log="$1"; shift
  # Ensure stderr_log exists even if wrap_time fails to write it, so the
  # caller can safely `parse_backend_from_logs "$stderr_log" ...` on either
  # success or failure paths.
  : > "$stderr_log"

  if wrap_time "$BINARY" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} \
      > "$stderr_log.stdout" 2>> "$stderr_log"; then
    :
  else
    # Surface the child's actual failure to the step log so debugging a
    # run-failed cell doesn't require local repro.
    { echo "--- native bench failed; last 30 lines of stdout ---"
      tail -n 30 "$stderr_log.stdout" 2>/dev/null || true
      echo "--- last 30 lines of stderr ---"
      tail -n 30 "$stderr_log"        2>/dev/null || true
      echo "--- end ---"
    } >&2
    return 1
  fi

  if ! [[ -s "$native_json" ]]; then
    # Exit 0 without the JSON means the binary took a code path that never
    # reaches its bench writer (this is how the sortformer diarization cells
    # failed invisibly) — dump the child's output so the step log shows what
    # actually ran, exactly like the non-zero-exit branch above.
    { echo "native bench exited 0 but did not emit the --json-out file"
      echo "--- last 30 lines of stdout ---"
      tail -n 30 "$stderr_log.stdout" 2>/dev/null || true
      echo "--- last 30 lines of stderr ---"
      tail -n 30 "$stderr_log"        2>/dev/null || true
      echo "--- end ---"
    } >&2
    return 1
  fi

  local median wmin wmax rtf backend
  median="$(jq -r '(.stages.tot.median_ms // .stages.total.median_ms // .stages.e2e.median_ms // .inference_ms.median // .end_to_end_ms.median // empty)' "$native_json" 2>/dev/null || true)"
  wmin="$(jq   -r '(.stages.tot.min_ms    // .stages.total.min_ms    // .stages.e2e.min_ms    // .inference_ms.min    // .end_to_end_ms.min    // empty)' "$native_json" 2>/dev/null || true)"
  wmax="$(jq   -r '(.stages.tot.max_ms    // .stages.total.max_ms    // .stages.e2e.max_ms    // .inference_ms.max    // .end_to_end_ms.max    // empty)' "$native_json" 2>/dev/null || true)"
  rtf="$(jq    -r '(.rtf.median // .rtf_median // empty)' "$native_json" 2>/dev/null || true)"
  backend="$(jq -r '(.backend // empty)' "$native_json" 2>/dev/null || true)"

  # Echo "median|min|max|rtf|backend" for the caller.
  echo "${median}|${wmin}|${wmax}|${rtf}|${backend}"
}

# Retain the exact logs used for backend attribution, including failed attempts.
# Copies happen outside the timed interval and survive temporary-dir cleanup.
preserve_music_generation_logs() {
  [[ "$MUSIC_ENABLED" == 1 ]] || return 0
  local stderr_log="$1"
  cp "$stderr_log.stdout" "$MUSIC_OUTPUT_DIR/generation.stdout.log" || \
    echo "warning: could not retain music generation stdout" >&2
  cp "$stderr_log" "$MUSIC_OUTPUT_DIR/generation.stderr.log" || \
    echo "warning: could not retain music generation stderr" >&2
}

# ---- time-wrapped bench: N invocations, we take the median ------------------
run_one_time_wrapped() {
  local iter="$1" stderr_log="$2" rss_log="$3"
  if [[ "$MUSIC_ENABLED" == 1 ]]; then
    MUSIC_OUTPUT_DIR="$MUSIC_ARTIFACTS/$(basename "$stderr_log" .err)"
    mkdir -p "$MUSIC_OUTPUT_DIR"
    MUSIC_AUDIO_OUT="$MUSIC_OUTPUT_DIR/audio.wav"
  fi
  if [[ "$TTS_KIND" == "tts_intelligibility" ]]; then
    TTS_AUDIO_OUT="${stderr_log}.wav"
  fi
  # stderr_log MUST exist by function-exit — the caller cat's it into the
  # combined log even on failure. Touch first, then let wrap_time's redirect
  # append its stderr.
  : > "$stderr_log"
  : > "$rss_log"
  AUDIO_OUT="$stderr_log.wav"
  build_bench_args
  if [[ "$MUSIC_ENABLED" == 1 ]]; then
    jq -n --arg caption "$MUSIC_CAPTION" --arg lyrics "$MUSIC_LYRICS" --arg seed "$MUSIC_SEED" \
      --arg family "$FAMILY" --arg duration "$MUSIC_DURATION" --arg frames "$MUSIC_MAX_FRAMES" \
      --slurpfile provenance "$MUSIC_ARTIFACTS/generation-provenance.json" --arg model_dir "$MODEL_DIR" --args '{provenance:$provenance[0],caption:$caption,lyrics:$lyrics,seed:$seed,family:$family,duration_seconds:(if $family == "minimax" then null else $duration end),max_frames:$frames,model_dir:$model_dir,argv:$ARGS.positional}' \
      -- "$BINARY" "${BENCH_ARGS[@]}" > "$MUSIC_OUTPUT_DIR/generation.json"
  fi
  local start_ns end_ns
  start_ns="$(now_ns)"
  if ! wrap_time "$BINARY" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} > "$stderr_log.stdout" 2> "$rss_log"; then
    cat "$rss_log" >> "$stderr_log"
    preserve_music_generation_logs "$stderr_log"
    tail -20 "$stderr_log" >&2
    return 1
  fi
  end_ns="$(now_ns)"
  cat "$rss_log" >> "$stderr_log"
  preserve_music_generation_logs "$stderr_log"
  if [[ "$MUSIC_ENABLED" == 1 ]]; then
    local music_backend
    music_backend="$(parse_backend_from_logs "$stderr_log" "$stderr_log.stdout")"
    jq --arg backend "${music_backend:-CPU}" '.observed_backend = $backend' "$MUSIC_OUTPUT_DIR/generation.json" > "$MUSIC_OUTPUT_DIR/generation.tmp.json"
    mv "$MUSIC_OUTPUT_DIR/generation.tmp.json" "$MUSIC_OUTPUT_DIR/generation.json"
  fi
  awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.1f", (e - s) / 1000000.0 }'
}

score_audio_correctness() {
  case "$AUDIO_KIND" in audio|sisdr|stoi) ;; *) return 0 ;; esac
  local hypothesis="$1" metrics report
  report="${OUT%.json}.audio-quality.json"
  metrics="$(jq -r --arg f "$FAMILY" '.[$f].correctness | if .kind == "audio" then (.metrics | join(",")) else .kind end' "$FAMILIES_JSON")"
  CORRECTNESS_KIND='"audio"'
  CORRECTNESS_REF="$(jq --arg f "$FAMILY" '.[$f].correctness.reference' "$FAMILIES_JSON")"
  if "$AUDIO_PYTHON" "$(dirname "$0")/compute-audio-quality.py" \
      --reference "$AUDIO_REFERENCE" --hypothesis-file "$hypothesis" \
      --metrics "$metrics" --json-out "$report" > "$tmp_dir/audio-score.stdout"; then
    if jq -e '.metrics | type == "object"' "$report" >/dev/null 2>&1; then
      AUDIO_QUALITY="$(jq -c '.metrics' "$report")"
    fi
  else
    echo "$FAMILY: audio quality scorer failed; performance results retained" >&2
  fi
  local source suffix
  for suffix in reference input hypothesis; do
    case "$suffix" in
      reference) source="$AUDIO_REFERENCE" ;;
      input) source="$AUDIO_INPUT" ;;
      hypothesis) source="$hypothesis" ;;
    esac
    if [[ -f "$source" ]]; then
      cp "$source" "${OUT%.json}.$suffix.wav" || echo "warning: cannot preserve $suffix WAV" >&2
    fi
  done
}

# ---- run --------------------------------------------------------------------
score_tts_intelligibility() {
  [[ "$TTS_KIND" == "tts_intelligibility" ]] || return 0
  local audio="$1" artifact_base="${OUT%.json}"
  if ! cp "$tts_reference" "$artifact_base.tts-reference.txt" ||
     { [[ -f "$audio" ]] && ! cp "$audio" "$artifact_base.tts.wav"; }; then
    TTS_INTELLIGIBILITY='{"status":"error","wer":null,"reason":"could not preserve TTS input/output artifacts"}'
    printf '%s\n' "$TTS_INTELLIGIBILITY" > "$artifact_base.tts-intelligibility.json" || true
    return 0
  fi
  if [[ -z "$tts_asr_model" ]]; then
    TTS_INTELLIGIBILITY="$(printf '%s' "$tts_asr_preparation" | jq '. + {wer:null,kind:"tts_intelligibility"}')"
    printf '%s\n' "$TTS_INTELLIGIBILITY" > "$artifact_base.tts-intelligibility.json"
    return 0
  fi
  python3 "$(dirname "$0")/compute-tts-intelligibility.py" \
    --audio "$artifact_base.tts.wav" --reference "$artifact_base.tts-reference.txt" \
    --asr-binary "$BUILD_DIR/bin/whisper-cli" --asr-model "$tts_asr_model" \
    --json-out "$artifact_base.tts-intelligibility.json" \
    --transcript-out "$artifact_base.tts-transcript.txt" \
    --log-out "$artifact_base.tts-asr.log" > "$tmp_dir/tts-score.stdout" || true
  if [[ -s "$artifact_base.tts-intelligibility.json" ]]; then
    TTS_INTELLIGIBILITY="$(cat "$artifact_base.tts-intelligibility.json")"
  else
    TTS_INTELLIGIBILITY='{"status":"error","wer":null,"reason":"TTS scorer failed without a result"}'
    printf '%s\n' "$TTS_INTELLIGIBILITY" > "$artifact_base.tts-intelligibility.json"
  fi
}

case "$BENCH_KIND" in
  native)
    coreml_compare="$(spec_field coreml_compare_on_darwin)"
    if [[ "$(benchmark_platform)" == "Darwin" && "$coreml_compare" == "true" ]]; then
      coreml_json="$tmp_dir/native-coreml.json"
      coreml_stderr="$tmp_dir/coreml.err"
      JSON_OUT="$coreml_json"
      build_bench_args
      BENCH_ARGS+=("--require-coreml")
      if ! run_native "$coreml_json" "$coreml_stderr" >/dev/null; then
        tail -40 "$coreml_stderr" >&2 || true
        BACKEND="$(parse_backend_from_logs "$coreml_stderr" "$coreml_stderr.stdout")"
        BACKEND="${BACKEND:-unknown}"
        emit_json "run-failed" null null null " (required Core ML benchmark failed; verify the sidecar and runner compatibility)"
        exit 0
      fi

      if [[ "$(jq -r '.encoder_coreml_all_runs // false' "$coreml_json")" != "true" ]] ||
         [[ "$(jq -r '.encoder_backend // ""' "$coreml_json")" != coreml-* ]]; then
        emit_json "run-failed" null null null " (benchmark completed without Core ML on every encoder invocation)"
        exit 0
      fi

      baseline_json="$tmp_dir/native-metal.json"
      baseline_stderr="$tmp_dir/metal.err"
      JSON_OUT="$baseline_json"
      build_bench_args
      if ! PARAKEET_COREML_DISABLE=1 run_native "$baseline_json" "$baseline_stderr" >/dev/null; then
        tail -40 "$baseline_stderr" >&2 || true
        BACKEND="$(jq -r '.backend // "unknown"' "$coreml_json")"
        ENCODER_BACKEND="$(jq -r '.encoder_backend // "unknown"' "$coreml_json")"
        ENCODER_COREML_ALL_RUNS="true"
        emit_json "run-failed" null null null " (forced-ggml baseline benchmark failed)"
        exit 0
      fi

      if [[ "$(jq -r '.encoder_coreml_all_runs // false' "$baseline_json")" == "true" ]]; then
        emit_json "run-failed" null null null " (PARAKEET_COREML_DISABLE was ignored by the baseline run)"
        exit 0
      fi
      require_transcript_match="$(spec_field coreml_require_transcript_match)"
      if [[ "$require_transcript_match" != "false" ]] &&
         [[ "$(jq -r '.transcript // ""' "$coreml_json")" != "$(jq -r '.transcript // ""' "$baseline_json")" ]]; then
        emit_json "run-failed" null null null " (Core ML and forced-ggml transcripts differ)"
        exit 0
      fi

      BACKEND="$(jq -r '.backend // "unknown"' "$coreml_json")"
      ENCODER_BACKEND="$(jq -r '.encoder_backend // "unknown"' "$coreml_json")"
      ENCODER_COREML_ALL_RUNS="true"
      ENCODER_MS_MEDIAN="$(jq -r '.encoder_ms.median // null' "$coreml_json")"
      BASELINE_BACKEND="$(jq -r '.backend // "unknown"' "$baseline_json")"
      BASELINE_ENCODER_BACKEND="$(jq -r '.encoder_backend // "unknown"' "$baseline_json")"
      BASELINE_ENCODER_MS_MEDIAN="$(jq -r '.encoder_ms.median // null' "$baseline_json")"
      BASELINE_INFERENCE_MS_MEDIAN="$(jq -r '.inference_ms.median // null' "$baseline_json")"
      BASELINE_RTF_MEDIAN="$(jq -r '.rtf_median // null' "$baseline_json")"
      coreml_inference="$(jq -r '.inference_ms.median // null' "$coreml_json")"
      RTF_MEDIAN="$(jq -r '.rtf_median // null' "$coreml_json")"

      if [[ "$ENCODER_MS_MEDIAN" != "null" && "$BASELINE_ENCODER_MS_MEDIAN" != "null" ]]; then
        ENCODER_SPEEDUP="$(jq -n --argjson base "$BASELINE_ENCODER_MS_MEDIAN" --argjson active "$ENCODER_MS_MEDIAN" 'if $active > 0 then $base / $active else null end')"
      fi
      if [[ "$coreml_inference" != "null" && "$BASELINE_INFERENCE_MS_MEDIAN" != "null" ]]; then
        INFERENCE_SPEEDUP="$(jq -n --argjson base "$BASELINE_INFERENCE_MS_MEDIAN" --argjson active "$coreml_inference" 'if $active > 0 then $base / $active else null end')"
      fi

      coreml_rss="$(parse_rss_from_time_stderr "$coreml_stderr")"
      baseline_rss="$(parse_rss_from_time_stderr "$baseline_stderr")"
      if [[ -n "$coreml_rss" && -n "$baseline_rss" ]]; then
        PEAK_RSS_MIB="$(awk -v a="$coreml_rss" -v b="$baseline_rss" 'BEGIN { print (a > b ? a : b) }')"
      elif [[ -n "$coreml_rss" ]]; then
        PEAK_RSS_MIB="$coreml_rss"
      elif [[ -n "$baseline_rss" ]]; then
        PEAK_RSS_MIB="$baseline_rss"
      fi

      artifact_dir="$(dirname "$OUT")"
      mkdir -p "$artifact_dir"
      cp "$coreml_json" "$artifact_dir/${FAMILY}-coreml-native.json"
      cp "$baseline_json" "$artifact_dir/${FAMILY}-metal-native.json"

      coreml_corr="$(score_correctness bench-json "$coreml_json" || true)"
      baseline_corr="$(score_correctness bench-json "$baseline_json" || true)"
      if [[ -z "$coreml_corr" || -z "$baseline_corr" ]]; then
        emit_json "run-failed" null null null " (Core ML and baseline correctness must both be scoreable)"
        exit 0
      fi

      IFS='|' read -r coreml_score coreml_kind coreml_ref <<< "$coreml_corr"
      IFS='|' read -r baseline_score baseline_kind baseline_ref <<< "$baseline_corr"
      if [[ "$coreml_kind" != "$baseline_kind" || "$coreml_ref" != "$baseline_ref" ]]; then
        emit_json "run-failed" null null null " (Core ML and baseline correctness metadata differ)"
        exit 0
      fi

      case "$coreml_kind" in
        wer)
          WER_MEDIAN="$coreml_score"
          BASELINE_WER_MEDIAN="$baseline_score"
          max_wer="$(jq -r --arg family "$FAMILY" '.[$family].correctness.max_wer // empty' "$FAMILIES_JSON")"
          if [[ "$require_transcript_match" == "false" && -z "$max_wer" ]]; then
            emit_json "run-failed" null null null " (transcript matching is disabled without correctness.max_wer)"
            exit 0
          fi
          if [[ -n "$max_wer" ]] && ! awk -v active="$coreml_score" -v baseline="$baseline_score" -v limit="$max_wer" \
              'BEGIN { exit ! (active <= limit && baseline <= limit) }'; then
            emit_json "run-failed" null null null " (Core ML or baseline WER exceeds correctness.max_wer=$max_wer)"
            exit 0
          fi
          ;;
        der) DER_MEDIAN="$coreml_score" ;;
        f1)  F1_MEDIAN="$coreml_score" ;;
      esac
      if [[ -n "$coreml_kind" ]]; then
        CORRECTNESS_KIND="$(printf '%s' "$coreml_kind" | jq -R .)"
      fi
      if [[ -n "$coreml_ref" ]]; then
        CORRECTNESS_REF="$(printf '%s' "$coreml_ref" | jq -R .)"
      fi

      coreml_min="$(jq -r '.inference_ms.min // null' "$coreml_json")"
      coreml_max="$(jq -r '.inference_ms.max // null' "$coreml_json")"
      emit_json "ok" "$coreml_inference" "$coreml_min" "$coreml_max"
      exit 0
    fi

    stderr_log="$tmp_dir/stderr.log"
    parsed=""
    if ! parsed="$(run_native "$JSON_OUT" "$stderr_log")"; then
      BACKEND="$(parse_backend_from_logs "$stderr_log" "$stderr_log.stdout")"
      BACKEND="${BACKEND:-unknown}"
      emit_json "run-failed" null null null " (native bench invocation failed)"
      exit 0
    fi
    IFS='|' read -r n_med n_min n_max n_rtf n_backend <<< "$parsed"
    [[ -n "$n_med" ]] || n_med="null"
    [[ -n "$n_min" ]] || n_min="null"
    [[ -n "$n_max" ]] || n_max="null"
    if [[ -n "$n_rtf" ]]; then RTF_MEDIAN="$n_rtf"; fi

    # Prefer the backend the bench JSON declared (parakeet reports the
    # post-fallback active backend); fall back to log scraping, then to CPU —
    # the run finished green with no GPU-engagement evidence.
    if [[ -n "$n_backend" ]]; then
      BACKEND="$n_backend"
    else
      BACKEND="$(parse_backend_from_logs "$stderr_log" "$stderr_log.stdout")"
      BACKEND="${BACKEND:-CPU}"
    fi
    ENCODER_BACKEND="$(jq -r '.encoder_backend // "unknown"' "$JSON_OUT")"
    ENCODER_COREML_ALL_RUNS="$(jq -r '.encoder_coreml_all_runs // null' "$JSON_OUT")"
    ENCODER_MS_MEDIAN="$(jq -r '.encoder_ms.median // null' "$JSON_OUT")"
    rss="$(parse_rss_from_time_stderr "$stderr_log")"
    [[ -n "$rss" ]] && PEAK_RSS_MIB="$rss"

    # Correctness scoring — no-op unless the family declares a correctness
    # block. Runs after perf is captured so a scoring failure never
    # downgrades a green perf run. Score is routed into WER_MEDIAN /
    # DER_MEDIAN / F1_MEDIAN based on the kind the family declared.
    corr_out=""
    corr_out="$(score_correctness bench-json "$JSON_OUT" || true)"
    if [[ -n "$corr_out" ]]; then
      IFS='|' read -r c_score c_kind c_ref <<< "$corr_out"
      case "$c_kind" in
        wer) [[ -n "$c_score" ]] && WER_MEDIAN="$c_score" ;;
        der) [[ -n "$c_score" ]] && DER_MEDIAN="$c_score" ;;
        f1)  [[ -n "$c_score" ]] && F1_MEDIAN="$c_score"  ;;
      esac
      [[ -n "$c_kind" ]] && CORRECTNESS_KIND="$(printf '%s' "$c_kind" | jq -R .)"
      [[ -n "$c_ref"  ]] && CORRECTNESS_REF="$(printf '%s'  "$c_ref"  | jq -R .)"
    fi

    score_tts_intelligibility "$TTS_AUDIO_OUT"
    emit_json "ok" "$n_med" "$n_min" "$n_max"
    ;;

  time-wrapped)
    for ((i = 1; i <= WARMUP; i++)); do
      echo "warmup $i/$WARMUP" >&2
      wu_stderr="$tmp_dir/warmup-$i.err"
      wu_rss="$tmp_dir/warmup-$i.rss"
      if ! run_one_time_wrapped "$i" "$wu_stderr" "$wu_rss" >/dev/null; then
        BACKEND="$(parse_backend_from_logs "$wu_stderr" "$wu_stderr.stdout")"; BACKEND="${BACKEND:-unknown}"
        emit_json "run-failed" null null null " (warmup failed)"; exit 0
      fi
    done

    declare -a wall_ms=()
    max_rss_seen=""
    combined_stderr="$tmp_dir/combined.err"
    combined_stdout="$tmp_dir/combined.out"
    : > "$combined_stderr"
    : > "$combined_stdout"
    for ((i = 1; i <= RUNS; i++)); do
      echo "run $i/$RUNS" >&2
      r_stderr="$tmp_dir/run-$i.err"
      r_rss="$tmp_dir/run-$i.rss"
      if ! ms="$(run_one_time_wrapped "$i" "$r_stderr" "$r_rss")"; then
        cat "$r_stderr" >> "$combined_stderr"
        cat "$r_stderr.stdout" >> "$combined_stdout" 2>/dev/null || true
        BACKEND="$(parse_backend_from_logs "$combined_stderr" "$combined_stdout")"; BACKEND="${BACKEND:-unknown}"
        emit_json "run-failed" null null null " (timed run $i failed)"; exit 0
      fi
      wall_ms+=("$ms")
      cat "$r_stderr" >> "$combined_stderr"
      cat "$r_stderr.stdout" >> "$combined_stdout" 2>/dev/null || true
      rss_i="$(parse_rss_from_time_stderr "$r_stderr")"
      if [[ -n "$rss_i" ]]; then
        if [[ -z "$max_rss_seen" ]] || awk -v a="$rss_i" -v b="$max_rss_seen" 'BEGIN{exit !(a>b)}'; then
          max_rss_seen="$rss_i"
        fi
      fi
    done

    stats="$(printf '%s\n' "${wall_ms[@]}" | jq -s '{
      median: (sort | if length%2==1 then .[length/2|floor] else (.[length/2-1] + .[length/2]) / 2 end),
      min: min,
      max: max
    }')"
    med="$(echo "$stats" | jq '.median')"
    mn="$(echo "$stats" | jq '.min')"
    mx="$(echo "$stats" | jq '.max')"

    BACKEND="$(parse_backend_from_logs "$combined_stderr" "$combined_stdout")"
    BACKEND="${BACKEND:-CPU}"
    if [[ -n "$max_rss_seen" ]]; then PEAK_RSS_MIB="$max_rss_seen"; fi

    # Compute RTF only when families.json declared audio_duration_seconds.
    if [[ "$AUDIO_DURATION_S" != "null" && -n "$AUDIO_DURATION_S" ]]; then
      RTF_MEDIAN="$(awk -v m="$med" -v s="$AUDIO_DURATION_S" 'BEGIN { printf "%.3f", m / (s * 1000.0) }')"
    fi

    # Correctness scoring — no-op unless the family declares a correctness
    # block. The hypothesis is the last successful run's captured stdout
    # ($r_stderr.stdout from the final loop iteration, still on disk in
    # $tmp_dir). Runs after perf is captured so a scoring failure never
    # downgrades a green perf run. Score is routed into WER_MEDIAN /
    # DER_MEDIAN / F1_MEDIAN based on the kind the family declared.
    corr_out=""
    corr_out="$(score_correctness text-file "$r_stderr.stdout" || true)"
    if [[ -n "$corr_out" ]]; then
      IFS='|' read -r c_score c_kind c_ref <<< "$corr_out"
      case "$c_kind" in
        wer) [[ -n "$c_score" ]] && WER_MEDIAN="$c_score" ;;
        der) [[ -n "$c_score" ]] && DER_MEDIAN="$c_score" ;;
        f1)  [[ -n "$c_score" ]] && F1_MEDIAN="$c_score"  ;;
      esac
      [[ -n "$c_kind" ]] && CORRECTNESS_KIND="$(printf '%s' "$c_kind" | jq -R .)"
      [[ -n "$c_ref"  ]] && CORRECTNESS_REF="$(printf '%s'  "$c_ref"  | jq -R .)"

      # Preserve the raw hypothesis alongside result.json so a workflow
      # reviewer can inspect the actual per-segment output that produced
      # a given DER / WER / F1 number, not just the aggregate. $tmp_dir
      # gets trap-cleaned at script exit, so the copy has to happen here
      # (before emit_json returns). Extension mirrors the source shape:
      # .jsonl for parakeet-cli --emit jsonl (DER), .txt for whisper-cli
      # -nt plain text (WER) and whisper-vad-speech-segments --no-prints
      # text (F1). The workflow's upload-artifact step picks up artifacts/*,
      # so the sibling rides along with result.json.
      case "$c_kind" in
        der)     hyp_ext="jsonl" ;;
        wer|f1)  hyp_ext="txt"   ;;
        *)       hyp_ext=""      ;;
      esac
      if [[ -n "$hyp_ext" && -f "$r_stderr.stdout" ]]; then
        cp "$r_stderr.stdout" "${OUT%.json}.hypothesis.$hyp_ext" 2>/dev/null || \
          echo "warning: could not stash hypothesis artifact next to $OUT" >&2
      fi
    fi

    score_tts_intelligibility "${r_stderr}.wav"
    score_audio_correctness "$r_stderr.wav"
    if [[ "$MUSIC_ENABLED" == 1 ]]; then
      music_report="${OUT%.json}.music-alignment.json"
      rm -f -- "$music_report"
      python3 "$(dirname "$0")/music-alignment-postpass.py" \
        --artifacts "$MUSIC_ARTIFACTS" --runs "$RUNS" --python "$MUSIC_PYTHON" --backend "$BACKEND" \
        --model-dir "$MUSIC_CLAP_DIR" --manifest "$MUSIC_MANIFEST" --timeout "${MUSIC_ALIGNMENT_TIMEOUT:-300}" --out "$music_report" || true
      if jq -e 'type == "object" and has("status")' "$music_report" >/dev/null 2>&1; then
        MUSIC_RESULT="$(cat "$music_report")"
      else
        MUSIC_RESULT='{"status":"scorer-error","score":null,"reason":"music postpass failed without a valid report"}'
      fi
    fi
    emit_json "ok" "$med" "$mn" "$mx"
    ;;

  *)
    emit_json "run-failed" null null null " (unknown bench_kind '$BENCH_KIND')"
    exit 1
    ;;
esac
