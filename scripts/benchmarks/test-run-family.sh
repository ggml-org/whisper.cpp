#!/usr/bin/env bash
# Test suite for run-family.sh + families.json.
#
# Two layers:
#   1. Spec pins — assertions against the real families.json that encode the
#      regressions this harness has already hit (sortformer benching through a
#      CLI mode that ignores --bench, cosyvoice missing its GPU flag, HF
#      families without a pinned ref/sha256).
#   2. Driver behavior — a synthetic spec plus stub bench binaries and stub
#      aws/curl on PATH exercise fetch retries, both bench kinds, backend
#      attribution, and the failure-reporting paths without models, engines,
#      or network.
#
# Runs standalone (`scripts/benchmarks/test-run-family.sh`) and as the
# workflow's plan-job self-check. Needs bash, jq, and a `date` with %N —
# the same prerequisites run-family.sh itself enforces.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DRIVER="$HERE/run-family.sh"
REAL_SPEC="$HERE/families.json"
DESKTOP_WORKFLOW="$HERE/../../.github/workflows/speech-benchmark-desktop.yml"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PASS=0
fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { PASS=$((PASS + 1)); echo "ok $PASS - $*"; }

# ---- 1. static checks on the shipped files ----------------------------------

bash -n "$DRIVER" || fail "run-family.sh does not parse"
ok "run-family.sh parses"

jq -e . "$REAL_SPEC" > /dev/null || fail "families.json is not valid JSON"
ok "families.json is valid JSON"

bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not)
  | select((.value.bench_kind // "") == "" or (.value.binary // "") == "" or (.value.cmake_target // "") == "")
  | .key' "$REAL_SPEC")"
[[ -z "$bad" ]] || fail "families missing bench_kind/binary/cmake_target: $bad"
ok "every family declares bench_kind, binary, cmake_target"

jq -e '.sortformer.bench_kind == "time-wrapped"' "$REAL_SPEC" > /dev/null \
  || fail "sortformer must be time-wrapped: parakeet-cli's diarize path returns before the --bench loop and emits no --bench-json"
jq -e '[.sortformer.args[] | select(startswith("--bench"))] | length == 0' "$REAL_SPEC" > /dev/null \
  || fail "sortformer args carry --bench flags the diarize path ignores"
jq -e '.sortformer.models | length == 1' "$REAL_SPEC" > /dev/null \
  || fail "pure-diarization sortformer needs exactly its own GGUF"
# sortformer benches abcba.wav (~160.6 s, 3 speakers) so DER is a real signal;
# jfk.wav (single speaker, 11 s) would score trivially 0.0 and catch nothing.
jq -e '[.sortformer.args[] | select(endswith("/abcba.wav"))] | length == 1' "$REAL_SPEC" > /dev/null \
  || fail "sortformer --wav must be abcba.wav (multi-speaker fixture — jfk.wav gives trivial DER)"
jq -e '.sortformer.audio_duration_seconds == 160.6' "$REAL_SPEC" > /dev/null \
  || fail "sortformer RTF divides by the abcba.wav duration (~160.6 s)"
# parakeet-cli defaults to n_gpu_layers=0, i.e. CPU-only. Whisper's macOS row
# engages Metal automatically; sortformer needs the flag explicitly. Cosyvoice/
# supertonic/parler already pass it — keeping sortformer in sync so the macos
# row reports Metal, not CPU. On hosted linux (no GPU compiled into ggml) the
# flag is a documented no-op — falls back to CPU.
jq -e '.sortformer.args | index("--n-gpu-layers") != null' "$REAL_SPEC" > /dev/null \
  || fail "sortformer must request GPU offload with --n-gpu-layers (else macOS runner idles on CPU when Metal is available)"
ok "sortformer benches the diarize path time-wrapped on abcba.wav with GPU offload requested"

# sortformer DER: correctness block must declare kind='der' with the checked-in
# RTTM that matches the abcba.wav audio, and --emit jsonl must be in argv so
# parakeet-cli's stdout is machine-parseable segment output for compute-der.py.
jq -e '.sortformer.correctness.kind == "der"' "$REAL_SPEC" > /dev/null \
  || fail "sortformer must declare a correctness.kind='der' block (time-wrapped path via captured --emit jsonl stdout)"
jq -e '.sortformer.correctness.reference | test("abcba\\.rttm$")' "$REAL_SPEC" > /dev/null \
  || fail "sortformer correctness.reference must be the abcba.rttm alongside abcba.wav"
sortformer_ref_repo_rel="$(jq -r '.sortformer.correctness.reference' "$REAL_SPEC")"
[[ -f "$HERE/../../$sortformer_ref_repo_rel" ]] \
  || fail "sortformer correctness.reference file missing: $sortformer_ref_repo_rel"
# --emit jsonl gates parakeet-cli into printing one {speaker,start,end} JSON
# per segment on stdout — that's the DER hypothesis source.
emit_pos="$(jq -r '[.sortformer.args[]] | to_entries[] | select(.value == "--emit") | .key' "$REAL_SPEC" | head -1)"
[[ -n "$emit_pos" ]] \
  || fail "sortformer args missing --emit flag"
emit_val="$(jq -r --argjson i "$((emit_pos + 1))" '.sortformer.args[$i]' "$REAL_SPEC")"
[[ "$emit_val" == "jsonl" ]] \
  || fail "sortformer --emit must be 'jsonl' (got '$emit_val') — compute-der.py needs JSONL segments on stdout"
jq -e '(.sortformer.correctness.collar_ms // 250) >= 0' "$REAL_SPEC" > /dev/null \
  || fail "sortformer correctness.collar_ms must be a non-negative integer"
ok "sortformer declares a DER correctness block against abcba.rttm with --emit jsonl"

jq -e '.cosyvoice.args | index("--n-gpu-layers") != null' "$REAL_SPEC" > /dev/null \
  || fail "cosyvoice args lost --n-gpu-layers (Metal offload on the macOS runner)"
ok "cosyvoice requests GPU offload"

jq -e '.parakeet.correctness.kind == "wer"' "$REAL_SPEC" > /dev/null \
  || fail "parakeet must declare a correctness.kind='wer' block — the pilot for the WER-scoring path"
jq -e '.parakeet.correctness.reference | test("parakeet-expected-jfk-output\\.txt$")' "$REAL_SPEC" > /dev/null \
  || fail "parakeet correctness.reference must point at the checked-in JFK expected transcript"
parakeet_ref_repo_rel="$(jq -r '.parakeet.correctness.reference' "$REAL_SPEC")"
[[ -f "$HERE/../../$parakeet_ref_repo_rel" ]] \
  || fail "parakeet correctness.reference file missing: $parakeet_ref_repo_rel"
ok "parakeet declares a WER correctness block pointing at a real reference file"

jq -e '."parakeet-unified".coreml_compare_on_darwin == true and
       ."parakeet-unified".coreml_model_basename == "parakeet-unified-en-0.6b" and
       ."parakeet-unified".coreml_source_gguf == "parakeet-unified-en-0.6b.q8_0.gguf" and
       ."parakeet-unified".coreml_export_mel_frames == 1501 and
       ."parakeet-unified".correctness.kind == "wer"' "$REAL_SPEC" > /dev/null \
  || fail "parakeet-unified must declare its fixed-capacity Core ML and WER benchmark contract"
jq -e '."parakeet-unified".correctness.reference == .parakeet.correctness.reference' \
  "$REAL_SPEC" > /dev/null \
  || fail "parakeet-unified and TDT must use the same JFK WER reference"
unified_cache_key="$(awk '
  /id: parakeet-unified-coreml-cache/ { in_cache = 1 }
  in_cache && /key:/ { print; exit }
' "$DESKTOP_WORKFLOW")"
[[ "$unified_cache_key" == *"scripts/benchmarks/families.json"* ]] \
  || fail "parakeet-unified Core ML cache key must hash families.json so changing coreml_source_gguf invalidates stale sidecars"
ok "parakeet-unified declares the fixed 1501-frame Core ML comparison"

# whisper is the time-wrapped ASR family and must reuse the same JFK reference
# parakeet does — same jfk.wav audio, so a WER delta between the two rows is a
# real model-quality delta, not a normalizer or reference drift.
jq -e '.whisper.correctness.kind == "wer"' "$REAL_SPEC" > /dev/null \
  || fail "whisper must declare a correctness.kind='wer' block (time-wrapped path via captured stdout)"
jq -e '.whisper.correctness.reference == .parakeet.correctness.reference' "$REAL_SPEC" > /dev/null \
  || fail "whisper correctness.reference must match parakeet's — both bench the same jfk.wav"
whisper_ref_repo_rel="$(jq -r '.whisper.correctness.reference' "$REAL_SPEC")"
[[ -f "$HERE/../../$whisper_ref_repo_rel" ]] \
  || fail "whisper correctness.reference file missing: $whisper_ref_repo_rel"
jq -e '.whisper.args | index("-nt") != null' "$REAL_SPEC" > /dev/null \
  || fail "whisper must invoke whisper-cli with -nt so its stdout carries only the decoded transcript (the time-wrapped correctness hypothesis)"
ok "whisper declares a WER correctness block reusing parakeet's JFK reference and keeps -nt on"

jq -e '."parakeet-eou".coreml_compare_on_darwin == true and
       ."parakeet-eou".coreml_model_basename == "parakeet_realtime_eou_120m-v1" and
       ."parakeet-eou".coreml_export_mel_frames == 1101 and
       ."parakeet-eou".correctness.kind == "wer"' "$REAL_SPEC" > /dev/null \
  || fail "parakeet-eou must declare its exact-shape Core ML and WER benchmark contract"
ok "parakeet-eou declares the fixed 1101-frame Core ML comparison"

jq -e '."parakeet-nemotron".coreml_compare_on_darwin == true and
       ."parakeet-nemotron".coreml_require_transcript_match == false and
       ."parakeet-nemotron".coreml_model_basename == "nemotron-3.5-asr-streaming-0.6b" and
       ."parakeet-nemotron".coreml_export_mel_frames == 1101 and
       ."parakeet-nemotron".correctness.kind == "wer" and
       ."parakeet-nemotron".correctness.max_wer == 0.1 and
       (."parakeet-nemotron".args | index("--language") != null)' "$REAL_SPEC" > /dev/null \
  || fail "parakeet-nemotron must declare its exact-shape Core ML, locale, and WER benchmark contract"
ok "parakeet-nemotron declares the fixed 1101-frame Core ML comparison"

# compute-wer.py: shipped self-test must pass. Catches accidental regressions
# in the normalizer / DP without needing a real bench run.
python3 "$HERE/compute-wer.py" --self-test > /dev/null \
  || fail "compute-wer.py --self-test failed"
ok "compute-wer.py self-test passes"

# compute-der.py: same discipline as compute-wer.py's self-test — catches
# regressions in the RTTM parser, framing, collar, and permutation mapping
# without needing sortformer or the abcba fixture to run.
python3 "$HERE/compute-der.py" --self-test > /dev/null \
  || fail "compute-der.py --self-test failed"
ok "compute-der.py self-test passes"

# compute-f1.py: shipped self-test must pass. Catches regressions in the
# frame-based F1 math, the collar handling, and the whisper-vad-text /
# JSON-array parser dual-shape parity without needing silero or jfk.wav
# to run.
python3 "$HERE/compute-f1.py" --self-test > /dev/null \
  || fail "compute-f1.py --self-test failed"
ok "compute-f1.py self-test passes"

jq -e '.vad.source == "huggingface" and (.vad.hf_repo | length > 0) and (.vad.hf_ref | test("^[0-9a-f]{40}$"))' "$REAL_SPEC" > /dev/null \
  || fail "vad family must pin an HF repo at a full commit sha"
jq -e '.vad.models[0].sha256 | test("^[0-9a-f]{64}$")' "$REAL_SPEC" > /dev/null \
  || fail "vad model must pin a sha256"
jq -e '.vad.audio_duration_seconds == 11.0' "$REAL_SPEC" > /dev/null \
  || fail "vad RTF divides by the jfk.wav duration"
ok "vad family pins its HF source and checksum"

# vad F1 correctness: kind='f1' with a JSON-array reference that exists on
# disk, plus --no-prints in argv so whisper-vad-speech-segments' stdout is
# only the "Detected N... / Speech segment N:" lines the compute-f1.py
# parser understands. Mirrors the parakeet/whisper/sortformer spec pins.
jq -e '.vad.correctness.kind == "f1"' "$REAL_SPEC" > /dev/null \
  || fail "vad must declare a correctness.kind='f1' block (frame-based P/R/F1 against a JSON-array reference)"
jq -e '.vad.correctness.reference | test("jfk\\.vad-ref\\.json$")' "$REAL_SPEC" > /dev/null \
  || fail "vad correctness.reference must be jfk.vad-ref.json (the hand-labeled speech-segment reference for jfk.wav)"
vad_ref_repo_rel="$(jq -r '.vad.correctness.reference' "$REAL_SPEC")"
[[ -f "$HERE/../../$vad_ref_repo_rel" ]] \
  || fail "vad correctness.reference file missing: $vad_ref_repo_rel"
jq -e '.vad.args | index("--no-prints") != null' "$REAL_SPEC" > /dev/null \
  || fail "vad must invoke whisper-vad-speech-segments with --no-prints so stdout carries only the segment listing (the F1 hypothesis source)"
jq -e '(.vad.correctness.tolerance_ms // 100) >= 0' "$REAL_SPEC" > /dev/null \
  || fail "vad correctness.tolerance_ms must be a non-negative integer"
ok "vad declares an F1 correctness block against jfk.vad-ref.json with --no-prints"

hf_bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not)
  | select(.value.source == "huggingface")
  | select((.value.hf_repo // "") == "" or (.value.hf_ref // "") == "")
  | .key' "$REAL_SPEC")"
[[ -z "$hf_bad" ]] || fail "huggingface families without hf_repo/hf_ref: $hf_bad"
ok "every huggingface family pins repo and ref"

# ---- new-model families ------------------------------------------------------

jq -e '."parakeet-ctc".bench_kind == "native" and
       ."parakeet-ctc".correctness.kind == "wer" and
       ."parakeet-ctc".correctness.reference == .parakeet.correctness.reference and
       ."parakeet-ctc".coreml_compare_on_darwin == null and
       (."parakeet-ctc".args | index("--bench") != null) and
       (."parakeet-ctc".args | index("--n-gpu-layers") != null)' "$REAL_SPEC" > /dev/null \
  || fail "parakeet-ctc must bench natively on jfk.wav with the TDT WER reference and no Core ML comparison"
ok "parakeet-ctc benches natively against the shared JFK reference"

jq -e '."parakeet-indic".bench_kind == "native" and
       ."parakeet-indic".s3_prefix == "indic_conformer" and
       ."parakeet-indic".correctness.kind == "wer" and
       ."parakeet-indic".correctness.normalizer == "none" and
       ."parakeet-indic".audio_duration_seconds == 7.29 and
       (."parakeet-indic".args | index("--language") != null) and
       ([."parakeet-indic".args[] | select(endswith("/hi-16k.wav"))] | length == 1)' "$REAL_SPEC" > /dev/null \
  || fail "parakeet-indic must bench hi-16k.wav with --language and the 'none' normalizer (the English one strips Devanagari vowel signs)"
indic_ref_repo_rel="$(jq -r '."parakeet-indic".correctness.reference' "$REAL_SPEC")"
[[ -f "$HERE/../../$indic_ref_repo_rel" ]] \
  || fail "parakeet-indic correctness.reference file missing: $indic_ref_repo_rel"
ok "parakeet-indic benches the Hindi fixture with its exact expected transcript"

jq -e '."sortformer-streaming".bench_kind == "time-wrapped" and
       ."sortformer-streaming".correctness.kind == "der" and
       ."sortformer-streaming".correctness.reference == .sortformer.correctness.reference and
       ."sortformer-streaming".audio_duration_seconds == .sortformer.audio_duration_seconds and
       (."sortformer-streaming".args | index("--stream") != null) and
       (."sortformer-streaming".args | index("--n-gpu-layers") != null) and
       ([."sortformer-streaming".args[] | select(startswith("--bench"))] | length == 0)' "$REAL_SPEC" > /dev/null \
  || fail "sortformer-streaming must time-wrap parakeet-cli's --stream diarization on abcba.wav with the batch family's DER reference"
stream_emit_pos="$(jq -r '[."sortformer-streaming".args[]] | to_entries[] | select(.value == "--emit") | .key' "$REAL_SPEC" | head -1)"
[[ -n "$stream_emit_pos" ]] || fail "sortformer-streaming args missing --emit"
[[ "$(jq -r --argjson i "$((stream_emit_pos + 1))" '."sortformer-streaming".args[$i]' "$REAL_SPEC")" == "jsonl" ]] \
  || fail "sortformer-streaming --emit must be jsonl (compute-der.py reads JSONL segments)"
ok "sortformer-streaming streams abcba.wav and scores DER from JSONL segments"

jq -e '."nemotron-diarization".bench_kind == "time-wrapped" and
       ."nemotron-diarization".source == "huggingface" and
       ."nemotron-diarization".hf_repo == "nvidia/Nemotron-3-Diarization" and
       (."nemotron-diarization".hf_ref | test("^[0-9a-f]{40}$")) and
       (."nemotron-diarization".models[0].sha256 | test("^[0-9a-f]{64}$")) and
       ."nemotron-diarization".correctness.kind == "der" and
       ."nemotron-diarization".correctness.reference == .sortformer.correctness.reference and
       ."nemotron-diarization".audio_duration_seconds == .sortformer.audio_duration_seconds and
       (."nemotron-diarization".args | index("--n-gpu-layers") != null)' "$REAL_SPEC" > /dev/null \
  || fail "nemotron-diarization must pin the official HF GGUF (revision + sha256) and score DER on abcba.wav"
# The benchmark must download the same revision the engine's downloader pins,
# or the two could silently diverge on a checkpoint bump.
nemotron_script_rev="$(grep -oE 'MODEL_REVISION = "[0-9a-f]{40}"' "$HERE/../../engines/parakeet/scripts/download_nemotron_diarization.py" | grep -oE '[0-9a-f]{40}')"
[[ "$(jq -r '."nemotron-diarization".hf_ref' "$REAL_SPEC")" == "$nemotron_script_rev" ]] \
  || fail "nemotron-diarization hf_ref must match download_nemotron_diarization.py's MODEL_REVISION ($nemotron_script_rev)"
ok "nemotron-diarization pins the engine downloader's HF revision"

jq -e '."moss-transcribe".bench_kind == "time-wrapped" and
       ."moss-transcribe".binary == "engines/parakeet/moss-transcribe" and
       ."moss-transcribe".s3_prefix == "openmoss" and
       ."moss-transcribe".correctness.kind == "wer" and
       ."moss-transcribe".correctness.hypothesis_format == "moss-segments" and
       ."moss-transcribe".correctness.reference == .parakeet.correctness.reference and
       (."moss-transcribe".args | index("--gpu") != null)' "$REAL_SPEC" > /dev/null \
  || fail "moss-transcribe must time-wrap the moss-transcribe CLI with moss-segments WER scoring against the JFK reference"
ok "moss-transcribe declares moss-segments WER scoring"

jq -e '."vad-v6".source == "huggingface" and
       ."vad-v6".hf_repo == .vad.hf_repo and ."vad-v6".hf_ref == .vad.hf_ref and
       (."vad-v6".models[0].sha256 | test("^[0-9a-f]{64}$")) and
       ."vad-v6".models[0].sha256 != .vad.models[0].sha256 and
       ."vad-v6".correctness.kind == "f1" and
       ."vad-v6".correctness.reference == .vad.correctness.reference and
       (."vad-v6".args | index("--no-prints") != null)' "$REAL_SPEC" > /dev/null \
  || fail "vad-v6 must mirror the vad family (same HF pin, F1 reference, --no-prints) with its own v6.2.0 checksum"
ok "vad-v6 mirrors the silero v5 family with the v6.2.0 checkpoint"

for tts_family in chatterbox-multilingual supertonic1 supertonic2 parler-indic moss-tts; do
  jq -e --arg f "$tts_family" '.[$f].correctness.kind == "tts_intelligibility" and
         .[$f].correctness.asr_model == .supertonic.correctness.asr_model and
         .[$f].correctness.reference == .supertonic.correctness.reference and
         (.[$f].args | index("${TTS_TEXT}") != null) and
         (.[$f].args | index("${TTS_AUDIO_OUT}") != null)' "$REAL_SPEC" > /dev/null \
    || fail "$tts_family must score TTS intelligibility with the shared prompt and pinned reference ASR"
done
jq -e '(."chatterbox-multilingual".args | index("--language") != null) and
       (."chatterbox-multilingual".args | index("--s3gen-gguf") != null)' "$REAL_SPEC" > /dev/null \
  || fail "chatterbox-multilingual must pass --language (required by t3_mtl) and its own s3gen GGUF"
jq -e '(."parler-indic".args | index("--sampled") != null)' "$REAL_SPEC" > /dev/null \
  || fail "parler-indic must use sampled decoding like the parler family (greedy bench output is degenerate)"
ok "new TTS families reuse the intelligibility contract"

for moss_family in moss-tts moss-sfx moss-speech; do
  jq -e --arg f "$moss_family" '.[$f].runners == ["macos"] and .[$f].opt_in == true and
         (.[$f].timeout_minutes | type == "number" and . > 60) and
         (.[$f].args | index("--gpu") != null)' "$REAL_SPEC" > /dev/null \
    || fail "$moss_family must be macOS-only, opt-in, carry a raised timeout and request the GPU"
done
ok "MOSS generation families are macOS-only opt-ins with raised timeouts"

# runners / opt_in / timeout_minutes are read by the workflow's plan job;
# a typo there would silently drop or mis-time cells.
bad_runners="$(jq -r 'to_entries[] | select(.key | startswith("_") | not)
  | select((.value.runners // []) | map(. == "linux" or . == "macos") | all | not) | .key' "$REAL_SPEC")"
[[ -z "$bad_runners" ]] || fail "families with runner labels other than linux/macos: $bad_runners"
bad_opt_in="$(jq -r 'to_entries[] | select(.key | startswith("_") | not)
  | select(has("value") and (.value | has("opt_in")) and (.value.opt_in | type != "boolean")) | .key' "$REAL_SPEC")"
[[ -z "$bad_opt_in" ]] || fail "families with a non-boolean opt_in: $bad_opt_in"
bad_timeouts="$(jq -r 'to_entries[] | select(.key | startswith("_") | not)
  | select((.value | has("timeout_minutes")) and ((.value.timeout_minutes | type != "number") or .value.timeout_minutes < 1)) | .key' "$REAL_SPEC")"
[[ -z "$bad_timeouts" ]] || fail "families with an invalid timeout_minutes: $bad_timeouts"
ok "runners, opt_in and timeout_minutes fields are well-formed"

# The workflow whitelists families against ALL_FAMILIES; every family in the
# spec must be listed there or it can never be dispatched, and a stale entry
# there would expand "all" into a cell no spec backs.
workflow_families="$(grep -oE '^  ALL_FAMILIES: "[^"]+"' "$DESKTOP_WORKFLOW" | sed -E 's/^  ALL_FAMILIES: "([^"]+)"/\1/')"
[[ -n "$workflow_families" ]] || fail "could not read ALL_FAMILIES from $DESKTOP_WORKFLOW"
spec_families="$(jq -r '[keys[] | select(startswith("_") | not)] | sort | join(",")' "$REAL_SPEC")"
[[ "$(tr ',' '\n' <<< "$workflow_families" | sort | paste -sd, -)" == "$spec_families" ]] \
  || fail "workflow ALL_FAMILIES ($workflow_families) must list exactly the families.json keys ($spec_families)"
ok "workflow ALL_FAMILIES matches families.json"

# ---- 2. driver behavior against a synthetic spec -----------------------------

SPEC="$TMP/spec.json"
BUILD="$TMP/build"
MODELS="$TMP/models"
AUDIO="$TMP/audio"
OUT="$TMP/out"
STUBBIN="$TMP/stubbin"
STATE="$TMP/state"
mkdir -p "$BUILD/bin" "$MODELS" "$AUDIO" "$OUT" "$STUBBIN" "$STATE"

if command -v sha256sum > /dev/null; then
  hello_sha="$(printf 'hello' | sha256sum | awk '{print $1}')"
else
  hello_sha="$(printf 'hello' | shasum -a 256 | awk '{print $1}')"
fi

REF_DIR="$TMP/refs"
mkdir -p "$REF_DIR"
printf 'the quick brown fox' > "$REF_DIR/ref-perfect.txt"
printf 'the quick brown fox' > "$REF_DIR/ref-2subs.txt"    # same ref; the stub hyp will differ

# Minimal RTTM for the DER driver tests: 4 s of speaker A followed by 4 s of
# speaker B (matches compute-der.py's self-test shape). Total ref-speech = 8 s,
# so a stub that gets one speaker right and misses the other scores DER 0.5.
cat > "$REF_DIR/ref-der.rttm" <<'RTTM'
SPEAKER stub 1 0.000 4.000 <NA> <NA> A <NA> <NA>
SPEAKER stub 1 4.000 4.000 <NA> <NA> B <NA> <NA>
RTTM

# Minimal JSON-array reference for the F1 (VAD) driver tests: one speech
# segment from 2.0 to 6.0 out of an 8 s window (400 speech frames, 400
# silence frames — matches compute-f1.py's self-test shape). A stub that
# emits the same range scores F1 1.0; one that emits half scores 2/3.
cat > "$REF_DIR/ref-f1.json" <<'JSON'
[{"start": 2.0, "end": 6.0}]
JSON

jq -n --arg sha "$hello_sha" \
      --arg ref_perfect "$REF_DIR/ref-perfect.txt" \
      --arg ref_2subs   "$REF_DIR/ref-2subs.txt" \
      --arg ref_der     "$REF_DIR/ref-der.rttm" \
      --arg ref_f1      "$REF_DIR/ref-f1.json" '{
  "nat-ok": {
    bench_kind: "native", binary: "bin/nat-ok", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: null, notes: "n"
  },
  "nat-nojson": {
    bench_kind: "native", binary: "bin/nat-nojson", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: null, notes: "n"
  },
  "coreml-baseline-empty": {
    bench_kind: "native", binary: "bin/nat-coreml-baseline-empty", cmake_target: "x",
    models: ["stub.f16.gguf"], args: ["${JSON_OUT}"], audio_duration_seconds: 11.0,
    coreml_compare_on_darwin: true,
    coreml_require_transcript_match: false,
    coreml_model_basename: "stub",
    coreml_export_mel_frames: 1101,
    correctness: {kind: "wer", reference: $ref_perfect, normalizer: "english", max_wer: 0.1},
    notes: "forced-ggml baseline emits an empty transcript and must reject the comparison"
  },
  "wer-perfect": {
    bench_kind: "native", binary: "bin/nat-transcript-perfect", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: 11.0,
    correctness: {kind: "wer", reference: $ref_perfect, normalizer: "english"},
    notes: "expects wer_median=0.0"
  },
  "wer-nonzero": {
    bench_kind: "native", binary: "bin/nat-transcript-wrong", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: 11.0,
    correctness: {kind: "wer", reference: $ref_2subs, normalizer: "english"},
    notes: "expects wer_median>0"
  },
  "wer-empty": {
    bench_kind: "native", binary: "bin/nat-transcript-empty", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: 11.0,
    correctness: {kind: "wer", reference: $ref_perfect, normalizer: "english"},
    notes: "expects wer_median=1.0 (present-but-empty transcript = full miss)"
  },
  "wer-notranscript": {
    bench_kind: "native", binary: "bin/nat-ok", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: 11.0,
    correctness: {kind: "wer", reference: $ref_perfect, normalizer: "english"},
    notes: "nat-ok emits no .transcript field; correctness must be gracefully skipped"
  },
  "wer-badref": {
    bench_kind: "native", binary: "bin/nat-transcript-perfect", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: 11.0,
    correctness: {kind: "wer", reference: "/does/not/exist.txt", normalizer: "english"},
    notes: "missing reference file — correctness skipped, perf still ok"
  },
  "wer-tw-perfect": {
    bench_kind: "time-wrapped", binary: "bin/tw-transcript-perfect", cmake_target: "x",
    args: [], audio_duration_seconds: 1.0,
    correctness: {kind: "wer", reference: $ref_perfect, normalizer: "english"},
    notes: "time-wrapped WER path: stdout carries the transcript; exact match => WER 0.0"
  },
  "wer-tw-empty": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    args: [], audio_duration_seconds: 1.0,
    correctness: {kind: "wer", reference: $ref_perfect, normalizer: "english"},
    notes: "time-wrapped catastrophic-miss guard: empty stdout capture must score WER 1.0, not skip"
  },
  "wer-tw-moss-segments": {
    bench_kind: "time-wrapped", binary: "bin/tw-moss-segments", cmake_target: "x",
    args: [], audio_duration_seconds: 1.0,
    correctness: {kind: "wer", reference: $ref_perfect, normalizer: "english", hypothesis_format: "moss-segments"},
    notes: "moss-transcribe shape: [start-end] Sxx: text lines; the markup must be stripped => WER 0.0"
  },
  "wer-tw-moss-plain": {
    bench_kind: "time-wrapped", binary: "bin/tw-moss-segments", cmake_target: "x",
    args: [], audio_duration_seconds: 1.0,
    correctness: {kind: "wer", reference: $ref_perfect, normalizer: "english"},
    notes: "same stub without hypothesis_format: the markup counts as words, so WER > 0 proves the option is threaded"
  },
  "der-tw-perfect": {
    bench_kind: "time-wrapped", binary: "bin/tw-jsonl-perfect", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "der", reference: $ref_der, collar_ms: 0},
    notes: "time-wrapped DER path: stdout JSONL matches ref (speaker IDs may permute) => DER 0.0"
  },
  "der-tw-nonzero": {
    bench_kind: "time-wrapped", binary: "bin/tw-jsonl-confused", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "der", reference: $ref_der, collar_ms: 0},
    notes: "collapsed-to-one-speaker hyp gets ref-A right and ref-B as confusion => DER 0.5"
  },
  "der-tw-empty": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "der", reference: $ref_der, collar_ms: 0},
    notes: "diarizer emitted nothing: full miss => DER 1.0"
  },
  "der-tw-badref": {
    bench_kind: "time-wrapped", binary: "bin/tw-jsonl-perfect", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "der", reference: "/does/not/exist.rttm", collar_ms: 0},
    notes: "missing RTTM: correctness skipped, perf still ok"
  },
  "der-tw-collar": {
    bench_kind: "time-wrapped", binary: "bin/tw-jsonl-slop", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "der", reference: $ref_der, collar_ms: 250},
    notes: "boundary-slop hyp + 250 ms collar; the slop falls inside the collar so DER 0.0. Proves collar_ms>0 is threaded from families.json through to compute-der.py."
  },
  "der-tw-duration-shape": {
    bench_kind: "time-wrapped", binary: "bin/tw-jsonl-duration", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "der", reference: $ref_der, collar_ms: 0},
    notes: "hyp uses {start,duration} instead of {start,end}; must parse identically"
  },
  "der-tw-mixed-stdout": {
    bench_kind: "time-wrapped", binary: "bin/tw-jsonl-mixed", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "der", reference: $ref_der, collar_ms: 0},
    notes: "hyp interleaves non-JSON banner/verbose lines with JSONL segments — parser must skip the noise and score to the same DER 0.0 as the pure-JSONL case"
  },
  "der-nat-warns": {
    bench_kind: "native", binary: "bin/nat-ok", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: 8.0,
    correctness: {kind: "der", reference: $ref_der, collar_ms: 0},
    notes: "native families have no diarize schema in bench JSON; DER must be skipped with a diagnostic"
  },
  "f1-tw-perfect": {
    bench_kind: "time-wrapped", binary: "bin/tw-vadtext-perfect", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "f1", reference: $ref_f1, tolerance_ms: 0},
    notes: "whisper-vad text-format stdout matching the reference exactly => F1 1.0"
  },
  "f1-tw-partial": {
    bench_kind: "time-wrapped", binary: "bin/tw-vadtext-partial", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "f1", reference: $ref_f1, tolerance_ms: 0},
    notes: "hyp covers half the ref (2.0-4.0 vs ref 2.0-6.0); precision=1.0 recall=0.5 F1=2/3"
  },
  "f1-tw-empty": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "f1", reference: $ref_f1, tolerance_ms: 0},
    notes: "VAD emitted nothing: full recall miss => F1 0.0"
  },
  "f1-tw-jsonarr": {
    bench_kind: "time-wrapped", binary: "bin/tw-vadjson-perfect", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "f1", reference: $ref_f1, tolerance_ms: 0},
    notes: "hyp emits JSON-array shape instead of whisper-vad text; parser must accept both and score F1 1.0"
  },
  "f1-tw-badref": {
    bench_kind: "time-wrapped", binary: "bin/tw-vadtext-perfect", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "f1", reference: "/does/not/exist.json", tolerance_ms: 0},
    notes: "missing reference: correctness skipped, perf still ok"
  },
  "f1-tw-collar": {
    bench_kind: "time-wrapped", binary: "bin/tw-vadtext-slop", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "f1", reference: $ref_f1, tolerance_ms: 200},
    notes: "boundary-slop hyp (2.1-6.1 vs ref 2.0-6.0) + 200 ms tolerance; the slop falls inside the collar so F1 1.0. Proves tolerance_ms>0 is threaded from families.json through to compute-f1.py."
  },
  "f1-tw-silero-shape": {
    bench_kind: "time-wrapped", binary: "bin/tw-vadtext-silero-shape", cmake_target: "x",
    args: [], audio_duration_seconds: 8.0,
    correctness: {kind: "f1", reference: $ref_f1, tolerance_ms: 0},
    notes: "multi-segment centiseconds hyp mimicking a realistic silero jfk output — three phrase-level segments with inter-phrase pauses. Exercises the whisper-vad text parser end-to-end at the driver level. Expected F1 = 6/7 ≈ 0.857. A pre-centiseconds-fix parser would score 0.0 here, so this cell would catch the units bug at driver level, not just self-test level."
  },
  "f1-nat-warns": {
    bench_kind: "native", binary: "bin/nat-ok", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: 8.0,
    correctness: {kind: "f1", reference: $ref_f1, tolerance_ms: 0},
    notes: "native families have no VAD schema in bench JSON; F1 must be skipped with a diagnostic"
  },
  "tw-marker": {
    bench_kind: "time-wrapped", binary: "bin/tw-marker", cmake_target: "x",
    args: [], audio_duration_seconds: 1.0, notes: "n"
  },
  "tw-evidence": {
    bench_kind: "time-wrapped", binary: "bin/tw-evidence", cmake_target: "x",
    args: [], audio_duration_seconds: 1.0, notes: "n"
  },
  "tw-silent": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    args: [], audio_duration_seconds: null, notes: "n"
  },
  "s3-retry": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    s3_prefix: "stub", models: ["d/model.bin"],
    args: [], audio_duration_seconds: null, notes: "n"
  },
  "s3-fail": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    s3_prefix: "stub", models: ["d/never.bin"],
    args: [], audio_duration_seconds: null, notes: "n"
  },
  "hf-file": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    source: "huggingface", hf_repo: "stub/repo", hf_ref: "0000000000000000000000000000000000000000",
    models: [{name: "m.bin", sha256: $sha}],
    args: [], audio_duration_seconds: null, notes: "n"
  },
  "whisper": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    source: "huggingface", hf_repo: "stub/repo", hf_ref: "0000000000000000000000000000000000000000",
    models_by_size: {tiny: {name: "ggml-tiny.bin"}},
    args: [], audio_duration_seconds: null, notes: "n"
  }
}' > "$SPEC"

cat > "$BUILD/bin/nat-ok" <<'STUB'
#!/usr/bin/env bash
printf '{"stages":{"tot":{"median_ms":100,"min_ms":90,"max_ms":110}},"rtf":{"median":0.5},"backend":"StubGPU"}' > "$1"
STUB

cat > "$BUILD/bin/nat-nojson" <<'STUB'
#!/usr/bin/env bash
echo "[0.00-2.00] speaker_0"
exit 0
STUB

cat > "$BUILD/bin/nat-coreml-baseline-empty" <<'STUB'
#!/usr/bin/env bash
if [[ "${PARAKEET_COREML_DISABLE:-0}" == "1" ]]; then
  cat > "$1" <<'EOJ'
{"backend":"MTL0","encoder_backend":"MTL0","encoder_coreml_all_runs":false,"encoder_ms":{"median":20},"inference_ms":{"median":40,"min":39,"max":41},"rtf_median":0.004,"transcript":""}
EOJ
else
  cat > "$1" <<'EOJ'
{"backend":"MTL0","encoder_backend":"coreml-all","encoder_coreml_all_runs":true,"encoder_ms":{"median":10},"inference_ms":{"median":30,"min":29,"max":31},"rtf_median":0.003,"transcript":"the quick brown fox"}
EOJ
fi
STUB

cat > "$BUILD/bin/tw-marker" <<'STUB'
#!/usr/bin/env bash
echo "  backend: Metal (kv_attn_type=f16)"
STUB

cat > "$BUILD/bin/tw-evidence" <<'STUB'
#!/usr/bin/env bash
echo "ggml_metal_library_init: using embedded metal library" >&2
STUB

cat > "$BUILD/bin/tw-silent" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB

# Whisper-shape stub for the time-wrapped correctness path: prints the exact
# reference transcript on stdout. whisper-cli -nt behaves the same way — the
# only stdout content is the decoded text.
cat > "$BUILD/bin/tw-transcript-perfect" <<'STUB'
#!/usr/bin/env bash
echo "the quick brown fox"
STUB

# moss-transcribe-shape stub: one "[start-end] Sxx: text" line per segment,
# the two segments together spelling the reference exactly.
cat > "$BUILD/bin/tw-moss-segments" <<'STUB'
#!/usr/bin/env bash
echo "[0.00-0.50] S1: The quick"
echo "[0.50-1.00] S2: brown fox."
STUB

# Sortformer-shape stub for the time-wrapped DER path: prints one JSONL segment
# object per line on stdout. Matches parakeet-cli --emit jsonl output. Speaker
# IDs are numeric (as parakeet-cli emits); the DER mapping search remaps them
# onto ref ids A/B — the "perfect" hyp can therefore label speakers 0/1 while
# the ref uses A/B and still score 0.0.
cat > "$BUILD/bin/tw-jsonl-perfect" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' '{"speaker":0,"start":0.000,"end":4.000}'
printf '%s\n' '{"speaker":1,"start":4.000,"end":8.000}'
STUB

# Collapsed hyp: single speaker across both ref segments. Gets ref-A right
# (correct) and ref-B as speaker confusion. 4 s confusion / 8 s ref-speech
# = DER 0.5.
cat > "$BUILD/bin/tw-jsonl-confused" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' '{"speaker":0,"start":0.000,"end":8.000}'
STUB

# Boundary-slop hyp: speaker change 100 ms before the true 4.0 s boundary.
# Without a collar this scores 10 confusion frames / 800 = DER 0.0125; a
# 250 ms collar around the boundary hides the slop so DER goes to 0.0.
# The der-tw-collar cell exercises the collar_ms>0 branch of the driver.
cat > "$BUILD/bin/tw-jsonl-slop" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' '{"speaker":0,"start":0.000,"end":3.900}'
printf '%s\n' '{"speaker":1,"start":3.900,"end":8.000}'
STUB

# Duration-shape hyp: same content as tw-jsonl-perfect but with `duration`
# instead of `end` — parse_hypothesis_jsonl accepts both, and this pins
# that a future parakeet-cli switch between the two shapes doesn't silently
# break DER.
cat > "$BUILD/bin/tw-jsonl-duration" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' '{"speaker":0,"start":0.000,"duration":4.000}'
printf '%s\n' '{"speaker":1,"start":4.000,"duration":4.000}'
STUB

# Mixed-stdout hyp: real parakeet-cli --verbose leaks non-JSON banner /
# summary lines to stdout alongside the JSONL segments. compute-der.py
# must skip the noise and score the JSONL portion — a single stray print
# from a future --verbose change (or a new backend init log) would
# otherwise null DER on every dispatch. Segments are identical to
# tw-jsonl-perfect, so the expected DER is 0.0.
cat > "$BUILD/bin/tw-jsonl-mixed" <<'STUB'
#!/usr/bin/env bash
echo "parakeet: using Metal backend"
echo "load=42.1ms audio=8.00s samples=128000@16000Hz"
printf '%s\n' '{"speaker":0,"start":0.000,"end":4.000}'
printf '%s\n' '{"speaker":1,"start":4.000,"end":8.000}'
echo "[diarize] total=123.4ms RTF=0.015 segments=2"
STUB

# whisper-vad-speech-segments-shape stubs for the F1 (VAD) driver tests.
# Format matches the real example's output under --no-prints: a blank line,
# "Detected N speech segments:" header, "Speech segment N: start=X, end=Y"
# per detected range, and a trailing blank line. compute-f1.py's parser
# skips the header and blanks and pulls (start,end) from the segment lines.
# UNITS: the example prints values in CENTISECONDS (see the compute-f1.py
# _WHISPER_VAD_SEG_RE comment) — the parser divides by 100. So a stub that
# wants to emit a 2.0-6.0 s segment writes "start = 200.00, end = 600.00".
cat > "$BUILD/bin/tw-vadtext-perfect" <<'STUB'
#!/usr/bin/env bash
printf '\n'
printf 'Detected 1 speech segments:\n'
printf 'Speech segment 0: start = 200.00, end = 600.00\n'
printf '\n'
STUB

cat > "$BUILD/bin/tw-vadtext-partial" <<'STUB'
#!/usr/bin/env bash
printf '\n'
printf 'Detected 1 speech segments:\n'
printf 'Speech segment 0: start = 200.00, end = 400.00\n'
printf '\n'
STUB

# JSON-array-shape stub for the F1 parser dual-shape test. compute-f1.py
# auto-detects on a leading '[' and parses via the JSON path; this stub
# proves the driver plumbs both shapes identically.
cat > "$BUILD/bin/tw-vadjson-perfect" <<'STUB'
#!/usr/bin/env bash
printf '[{"start": 2.0, "end": 6.0}]\n'
STUB

# Boundary-slop hyp for the F1 collar plumbing test: 100 ms late at both
# start and end. Without collar: precision = recall = 390/400 = 0.975
# (10 FP + 10 FN). With a 200 ms collar (±20 frames around each ref
# boundary), both slop regions fall entirely inside the excluded frames,
# so F1 goes back to 1.0. The f1-tw-collar cell asserts the collar case.
cat > "$BUILD/bin/tw-vadtext-slop" <<'STUB'
#!/usr/bin/env bash
printf '\n'
printf 'Detected 1 speech segments:\n'
printf 'Speech segment 0: start = 210.00, end = 610.00\n'
printf '\n'
STUB

# Realistic silero-shape stub: multiple phrase-level segments in
# centiseconds, mimicking what silero actually emits for a continuous
# speech clip like jfk.wav. Values in cs: 3 segments totaling 3.0 s of
# speech across the 2.0-6.0 s ref window with two ~0.5 s pauses. Exercises
# the full pipeline (multi-line whisper-vad parse -> /100 conversion ->
# frame masking -> F1) on a shape closer to production than any of the
# 1-segment cells. Expected F1 = 2*P*R/(P+R) with P=1.0 (all hyp inside
# ref) and R = 300/400 = 0.75, so F1 = 6/7 = 0.8571...
cat > "$BUILD/bin/tw-vadtext-silero-shape" <<'STUB'
#!/usr/bin/env bash
printf '\n'
printf 'Detected 3 speech segments:\n'
printf 'Speech segment 0: start = 200.00, end = 300.00\n'
printf 'Speech segment 1: start = 350.00, end = 450.00\n'
printf 'Speech segment 2: start = 500.00, end = 600.00\n'
printf '\n'
STUB

# Parakeet-shape JSON emitters for correctness tests. The driver expands
# ${JSON_OUT} into args; both binaries take that as their sole arg.
cat > "$BUILD/bin/nat-transcript-perfect" <<'STUB'
#!/usr/bin/env bash
cat > "$1" <<'EOJ'
{"stages":{"tot":{"median_ms":100,"min_ms":90,"max_ms":110}},"rtf":{"median":0.5},"backend":"StubGPU","transcript":"the quick brown fox"}
EOJ
STUB

cat > "$BUILD/bin/nat-transcript-wrong" <<'STUB'
#!/usr/bin/env bash
cat > "$1" <<'EOJ'
{"stages":{"tot":{"median_ms":100,"min_ms":90,"max_ms":110}},"rtf":{"median":0.5},"backend":"StubGPU","transcript":"the SLOW brown RED fox"}
EOJ
STUB

# Catastrophic-collapse case: model shipped `.transcript=""` (present but
# empty). MUST NOT be conflated with the field-absent case — an empty
# transcript is exactly the regression correctness scoring exists to catch,
# so it must score WER 1.0, not skip.
cat > "$BUILD/bin/nat-transcript-empty" <<'STUB'
#!/usr/bin/env bash
cat > "$1" <<'EOJ'
{"stages":{"tot":{"median_ms":100,"min_ms":90,"max_ms":110}},"rtf":{"median":0.5},"backend":"StubGPU","transcript":""}
EOJ
STUB

cat > "$STUBBIN/aws" <<'STUB'
#!/usr/bin/env bash
# Invoked as: aws s3 cp <s3url> <dest> --no-progress
count_file="$STATE_DIR/aws-calls"
n="$(cat "$count_file" 2>/dev/null || echo 0)"
n=$(( n + 1 ))
echo "$n" > "$count_file"
case "$3" in
  */model.bin) if [[ "$n" -ge 3 ]]; then printf 'weights' > "$4"; exit 0; else exit 1; fi ;;
  *) exit 1 ;;
esac
STUB

cat > "$STUBBIN/curl" <<'STUB'
#!/usr/bin/env bash
# Only the -o <dest> pair matters to the test.
dest=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-o" ]]; then dest="$2"; shift 2; else shift; fi
done
[[ -n "$dest" ]] || exit 2
printf 'hello' > "$dest"
STUB

chmod +x "$BUILD"/bin/* "$STUBBIN"/*

run_driver() {
  # run_driver <family> <out.json> <stderr-log> [extra env as VAR=VALUE...]
  local family="$1" out="$2" errlog="$3"; shift 3
  env -i PATH="$STUBBIN:$PATH" HOME="$TMP" "$@" \
    bash "$DRIVER" \
      --family "$family" --runs 2 --warmup 0 \
      --build-dir "$BUILD" --models-root "$MODELS" \
      --audio-dir "$AUDIO" --runner test \
      --out "$out" 2> "$errlog"
}

# native, JSON emitted: stats + rtf + backend come from the bench JSON.
run_driver nat-ok "$OUT/nat-ok.json" "$OUT/nat-ok.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .backend == "StubGPU" and .wall_ms_median == 100 and .rtf_median == 0.5' \
  "$OUT/nat-ok.json" > /dev/null || fail "nat-ok: $(cat "$OUT/nat-ok.json")"
ok "native bench JSON drives stats, rtf, and backend"

# native, exit 0 without JSON (the sortformer failure shape): run-failed and
# the child's output is dumped to the step log instead of vanishing.
run_driver nat-nojson "$OUT/nat-nojson.json" "$OUT/nat-nojson.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "run-failed"' "$OUT/nat-nojson.json" > /dev/null \
  || fail "nat-nojson status: $(cat "$OUT/nat-nojson.json")"
grep -q 'did not emit the --json-out file' "$OUT/nat-nojson.err" \
  || fail "nat-nojson: missing no-JSON diagnosis in driver stderr"
grep -q 'speaker_0' "$OUT/nat-nojson.err" \
  || fail "nat-nojson: child stdout not dumped to the step log"
ok "silent no-JSON native bench reports run-failed with dumped child output"

mkdir -p "$MODELS/coreml-baseline-empty/stub-encoder.mlmodelc"
printf 'stub' > "$MODELS/coreml-baseline-empty/stub.f16.gguf"
run_driver coreml-baseline-empty \
  "$OUT/coreml-baseline-empty.json" "$OUT/coreml-baseline-empty.err" \
  BENCH_FAMILIES_JSON="$SPEC" BENCH_TEST_PLATFORM=Darwin
jq -e '.status == "run-failed" and .wer_median == 0.0 and
       .baseline_wer_median == 1.0' \
  "$OUT/coreml-baseline-empty.json" > /dev/null \
  || fail "coreml-baseline-empty: $(cat "$OUT/coreml-baseline-empty.json")"
grep -q 'baseline WER exceeds correctness.max_wer' \
  "$OUT/coreml-baseline-empty.json" \
  || fail "coreml-baseline-empty: missing baseline WER rejection reason"
ok "Core ML comparison rejects an empty forced-ggml baseline when transcript matching is disabled"

# time-wrapped, explicit `backend:` marker on stdout (supertonic/lavasr shape).
run_driver tw-marker "$OUT/tw-marker.json" "$OUT/tw-marker.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .backend == "Metal" and .rtf_median != null' \
  "$OUT/tw-marker.json" > /dev/null || fail "tw-marker: $(cat "$OUT/tw-marker.json")"
ok "time-wrapped run scrapes 'backend:' markers from stdout"

# time-wrapped, ggml compute-init evidence on stderr (acestep-on-Metal shape).
run_driver tw-evidence "$OUT/tw-evidence.json" "$OUT/tw-evidence.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .backend == "Metal"' "$OUT/tw-evidence.json" > /dev/null \
  || fail "tw-evidence: $(cat "$OUT/tw-evidence.json")"
ok "ggml GPU compute-init evidence attributes the backend"

# time-wrapped, no marker at all: a green run defaults to CPU, not unknown.
run_driver tw-silent "$OUT/tw-silent.json" "$OUT/tw-silent.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .backend == "CPU" and .rtf_median == null' \
  "$OUT/tw-silent.json" > /dev/null || fail "tw-silent: $(cat "$OUT/tw-silent.json")"
ok "green run without GPU evidence is labeled CPU"

# S3 fetch: two transient failures, third attempt lands — one bench, 3 calls.
mkdir -p "$STATE/retry"
run_driver s3-retry "$OUT/s3-retry.json" "$OUT/s3-retry.err" \
  BENCH_FAMILIES_JSON="$SPEC" MODEL_S3_BUCKET=stub-bucket \
  BENCH_RETRY_SLEEP_S=0 STATE_DIR="$STATE/retry"
jq -e '.status == "ok"' "$OUT/s3-retry.json" > /dev/null \
  || fail "s3-retry: $(cat "$OUT/s3-retry.json")"
[[ "$(cat "$STATE/retry/aws-calls")" == "3" ]] \
  || fail "s3-retry: expected 3 aws attempts, got $(cat "$STATE/retry/aws-calls")"
[[ -f "$MODELS/s3-retry/model.bin" ]] || fail "s3-retry: model not staged"
ok "transient S3 resets are retried until the copy lands"

# S3 fetch: permanent failure exhausts the retries and reports fetch-failed
# with no partial file left behind.
mkdir -p "$STATE/fail"
run_driver s3-fail "$OUT/s3-fail.json" "$OUT/s3-fail.err" \
  BENCH_FAMILIES_JSON="$SPEC" MODEL_S3_BUCKET=stub-bucket \
  BENCH_RETRY_SLEEP_S=0 STATE_DIR="$STATE/fail"
jq -e '.status == "fetch-failed"' "$OUT/s3-fail.json" > /dev/null \
  || fail "s3-fail: $(cat "$OUT/s3-fail.json")"
[[ "$(cat "$STATE/fail/aws-calls")" == "3" ]] \
  || fail "s3-fail: expected 3 aws attempts, got $(cat "$STATE/fail/aws-calls")"
[[ ! -f "$MODELS/s3-fail/never.bin" ]] || fail "s3-fail: partial file left behind"
ok "exhausted S3 retries report fetch-failed without a partial file"

# HuggingFace fetch for a non-whisper family: {name, sha256} entries resolve,
# the checksum is enforced, and the model label uses the file name.
run_driver hf-file "$OUT/hf-file.json" "$OUT/hf-file.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .model == "hf-file: m.bin"' "$OUT/hf-file.json" > /dev/null \
  || fail "hf-file: $(cat "$OUT/hf-file.json")"
[[ -f "$MODELS/hf-file/m.bin" ]] || fail "hf-file: model not staged"
ok "non-whisper HF family fetches by name with checksum verification"

# whisper keeps its size-keyed path: a size absent from models_by_size still
# reports not-in-registry (exit-66 contract) rather than failing the cell.
env -i PATH="$STUBBIN:$PATH" HOME="$TMP" BENCH_FAMILIES_JSON="$SPEC" \
  bash "$DRIVER" \
    --family whisper --whisper-size base --runs 2 --warmup 0 \
    --build-dir "$BUILD" --models-root "$MODELS" \
    --audio-dir "$AUDIO" --runner test \
    --out "$OUT/whisper-66.json" 2> "$OUT/whisper-66.err"
jq -e '.status == "not-in-registry" and .model == "whisper-base"' \
  "$OUT/whisper-66.json" > /dev/null || fail "whisper-66: $(cat "$OUT/whisper-66.json")"
ok "whisper size missing from the registry reports not-in-registry"

# ---- correctness scoring (WER path) -----------------------------------------
# The perf portion of every cell here is the nat-transcript-{perfect,wrong}
# stub or nat-ok — all emit a valid perf JSON. The tests below only assert
# the correctness fields the driver merged into result.json.

run_driver wer-perfect "$OUT/wer-perfect.json" "$OUT/wer-perfect.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .wer_median == 0.0 and .correctness_kind == "wer"
       and (.correctness_reference | test("ref-perfect\\.txt$"))' \
  "$OUT/wer-perfect.json" > /dev/null \
  || fail "wer-perfect: $(cat "$OUT/wer-perfect.json")"
ok "correctness: exact-match transcript scores WER 0.0"

run_driver wer-nonzero "$OUT/wer-nonzero.json" "$OUT/wer-nonzero.err" BENCH_FAMILIES_JSON="$SPEC"
# 4 ref words, 1 substitution + 1 insertion = 2 edits / 4 words = 0.5
jq -e '.status == "ok" and .wer_median == 0.5 and .correctness_kind == "wer"' \
  "$OUT/wer-nonzero.json" > /dev/null \
  || fail "wer-nonzero: $(cat "$OUT/wer-nonzero.json")"
ok "correctness: 1 sub + 1 insertion scores WER 0.5"

run_driver wer-notranscript "$OUT/wer-notranscript.json" "$OUT/wer-notranscript.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .wer_median == null and .correctness_kind == null' \
  "$OUT/wer-notranscript.json" > /dev/null \
  || fail "wer-notranscript: $(cat "$OUT/wer-notranscript.json")"
grep -q 'has no .transcript field' "$OUT/wer-notranscript.err" \
  || fail "wer-notranscript: missing transcript diagnosis not surfaced"
ok "correctness: bench JSON without .transcript => wer_median=null + diagnostic"

# Regression guard for the field-absent-vs-field-empty conflation. A model
# collapse that emits `.transcript=""` MUST score 1.0, not null, or the very
# regression this feature exists to catch would be silently hidden.
run_driver wer-empty "$OUT/wer-empty.json" "$OUT/wer-empty.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .wer_median == 1.0 and .correctness_kind == "wer"' \
  "$OUT/wer-empty.json" > /dev/null \
  || fail "wer-empty: $(cat "$OUT/wer-empty.json")"
ok "correctness: present-but-empty transcript => wer_median=1.0 (catastrophic-miss regression guard)"

run_driver wer-badref "$OUT/wer-badref.json" "$OUT/wer-badref.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .wer_median == null and .correctness_kind == null' \
  "$OUT/wer-badref.json" > /dev/null \
  || fail "wer-badref: $(cat "$OUT/wer-badref.json")"
grep -q 'reference file not found' "$OUT/wer-badref.err" \
  || fail "wer-badref: missing-reference diagnosis not surfaced"
ok "correctness: missing reference file => wer_median=null + diagnostic (perf still ok)"

run_driver wer-tw-perfect "$OUT/wer-tw-perfect.json" "$OUT/wer-tw-perfect.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .wer_median == 0.0 and .correctness_kind == "wer"
       and (.correctness_reference | test("ref-perfect\\.txt$"))' \
  "$OUT/wer-tw-perfect.json" > /dev/null \
  || fail "wer-tw-perfect: $(cat "$OUT/wer-tw-perfect.json")"
# Raw-hypothesis artifact: stashed next to result.json as .hypothesis.txt so
# a reviewer of a workflow run can see the actual transcript that scored,
# not just the aggregate WER. .txt for WER (whisper-cli plain text).
[[ -f "$OUT/wer-tw-perfect.hypothesis.txt" ]] \
  || fail "wer-tw-perfect: hypothesis sibling not created next to result.json"
grep -q 'the quick brown fox' "$OUT/wer-tw-perfect.hypothesis.txt" \
  || fail "wer-tw-perfect: hypothesis sibling did not capture the stub's stdout"
ok "correctness (time-wrapped): captured stdout matching the reference scores WER 0.0 + hypothesis stashed as .hypothesis.txt"

# Time-wrapped variant of the catastrophic-miss regression guard from PR #230.
# A binary that exits 0 with an empty stdout capture must score 1.0, not
# silently skip — that's exactly the "model collapsed to silence" case this
# feature exists to catch, whether the transcript came from --json-out or the
# CLI's stdout.
run_driver wer-tw-empty "$OUT/wer-tw-empty.json" "$OUT/wer-tw-empty.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .wer_median == 1.0 and .correctness_kind == "wer"' \
  "$OUT/wer-tw-empty.json" > /dev/null \
  || fail "wer-tw-empty: $(cat "$OUT/wer-tw-empty.json")"
ok "correctness (time-wrapped): empty stdout capture => wer_median=1.0 (catastrophic-miss regression guard)"

# hypothesis_format='moss-segments' strips the timestamp/speaker markup that
# moss-transcribe prints before every segment, so a transcript that matches
# the reference scores 0.0. The plain-format twin runs the same stub and must
# score higher: it proves families.json's hypothesis_format actually reaches
# compute-wer.py instead of being silently ignored.
run_driver wer-tw-moss-segments "$OUT/wer-tw-moss-segments.json" "$OUT/wer-tw-moss-segments.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .wer_median == 0.0 and .correctness_kind == "wer"' \
  "$OUT/wer-tw-moss-segments.json" > /dev/null \
  || fail "wer-tw-moss-segments: $(cat "$OUT/wer-tw-moss-segments.json")"
run_driver wer-tw-moss-plain "$OUT/wer-tw-moss-plain.json" "$OUT/wer-tw-moss-plain.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .wer_median > 0.0 and .correctness_kind == "wer"' \
  "$OUT/wer-tw-moss-plain.json" > /dev/null \
  || fail "wer-tw-moss-plain: $(cat "$OUT/wer-tw-moss-plain.json")"
ok "correctness (time-wrapped): hypothesis_format=moss-segments strips segment markup before WER scoring"

# ---- correctness scoring (DER path, time-wrapped) --------------------------
# All DER cells declare the same 8 s / 2 speaker reference; only the stub's
# JSONL output varies. The kind='der' branch must (a) route the score into
# der_median, not wer_median, (b) remap hyp speaker IDs to ref via the
# permutation search (so `speaker:0/1` labels can match ref `A/B`), and
# (c) preserve the "empty stdout => full miss" invariant from the WER path.

run_driver der-tw-perfect "$OUT/der-tw-perfect.json" "$OUT/der-tw-perfect.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .der_median == 0.0 and .wer_median == null and .correctness_kind == "der"
       and (.correctness_reference | test("ref-der\\.rttm$"))' \
  "$OUT/der-tw-perfect.json" > /dev/null \
  || fail "der-tw-perfect: $(cat "$OUT/der-tw-perfect.json")"
# Raw-hypothesis artifact: stashed next to result.json as .hypothesis.jsonl.
# For DER debugging, the per-segment JSONL is what a reviewer needs to see
# which speaker labels came out wrong or which boundaries slipped past the
# collar — none of that is recoverable from der_median alone.
[[ -f "$OUT/der-tw-perfect.hypothesis.jsonl" ]] \
  || fail "der-tw-perfect: hypothesis sibling not created next to result.json"
jq -e '.speaker == 0 and .start == 0.0 and .end == 4.0' \
  "$OUT/der-tw-perfect.hypothesis.jsonl" > /dev/null 2>&1 || {
  # File may have multiple JSONL lines; parse the first one explicitly.
  first_line="$(head -1 "$OUT/der-tw-perfect.hypothesis.jsonl")"
  echo "$first_line" | jq -e '.speaker == 0 and .start == 0.0 and .end == 4.0' > /dev/null \
    || fail "der-tw-perfect.hypothesis.jsonl first line unexpected: $first_line"
}
ok "correctness (DER, time-wrapped): matching JSONL segments score DER 0.0 + hypothesis stashed as .hypothesis.jsonl"

run_driver der-tw-nonzero "$OUT/der-tw-nonzero.json" "$OUT/der-tw-nonzero.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .der_median == 0.5 and .correctness_kind == "der"' \
  "$OUT/der-tw-nonzero.json" > /dev/null \
  || fail "der-tw-nonzero: $(cat "$OUT/der-tw-nonzero.json")"
ok "correctness (DER, time-wrapped): collapsed-to-one-speaker hyp scores DER 0.5"

# Empty stdout => diarizer emitted nothing => full miss => DER 1.0. Mirrors
# the WER empty-transcript regression guard: correctness scoring must not
# silently skip the catastrophic-collapse case this feature exists to catch.
run_driver der-tw-empty "$OUT/der-tw-empty.json" "$OUT/der-tw-empty.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .der_median == 1.0 and .correctness_kind == "der"' \
  "$OUT/der-tw-empty.json" > /dev/null \
  || fail "der-tw-empty: $(cat "$OUT/der-tw-empty.json")"
ok "correctness (DER, time-wrapped): empty JSONL stdout => der_median=1.0 (catastrophic-miss regression guard)"

run_driver der-tw-badref "$OUT/der-tw-badref.json" "$OUT/der-tw-badref.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .der_median == null and .correctness_kind == null' \
  "$OUT/der-tw-badref.json" > /dev/null \
  || fail "der-tw-badref: $(cat "$OUT/der-tw-badref.json")"
grep -q 'reference file not found' "$OUT/der-tw-badref.err" \
  || fail "der-tw-badref: missing-RTTM diagnosis not surfaced"
# When correctness is skipped no hypothesis sibling should be left behind —
# otherwise a reader sees a stale hypothesis file next to a null der_median
# and wonders why the scorer didn't run.
[[ ! -f "$OUT/der-tw-badref.hypothesis.jsonl" ]] \
  || fail "der-tw-badref: hypothesis sibling should NOT exist when correctness was skipped"
ok "correctness (DER, time-wrapped): missing RTTM => der_median=null + diagnostic + no hypothesis sibling (perf still ok)"

# Non-zero collar plumbing: proves families.json's correctness.collar_ms
# reaches compute-der.py's --collar-ms. The 100 ms boundary slop scores
# non-zero without a collar (compute-der.py self-test pins that at 0.0125);
# a 250 ms collar hides it entirely, so DER=0.0 here. If the driver ever
# stripped collar_ms and compute-der.py's default (250) also gave 0.0 this
# would still pass, so pair this with the compute-der.py self-test's
# "boundary slop, no collar" case which pins the DER-without-collar value.
run_driver der-tw-collar "$OUT/der-tw-collar.json" "$OUT/der-tw-collar.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .der_median == 0.0 and .correctness_kind == "der"' \
  "$OUT/der-tw-collar.json" > /dev/null \
  || fail "der-tw-collar: $(cat "$OUT/der-tw-collar.json")"
ok "correctness (DER, time-wrapped): 250 ms collar hides boundary slop => DER 0.0 (collar_ms plumbed through)"

# Duration-shape hypothesis: parakeet-cli today emits {speaker,start,end}
# but the JSONL contract accepts either end or duration. Pin that a future
# CLI switch between the two shapes doesn't silently regress DER — same
# hyp content in the two shapes must score identically (DER 0.0 here).
run_driver der-tw-duration-shape "$OUT/der-tw-duration-shape.json" "$OUT/der-tw-duration-shape.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .der_median == 0.0 and .correctness_kind == "der"' \
  "$OUT/der-tw-duration-shape.json" > /dev/null \
  || fail "der-tw-duration-shape: $(cat "$OUT/der-tw-duration-shape.json")"
ok "correctness (DER, time-wrapped): {start,duration}-shape JSONL parses identically to {start,end}"

# Mixed-stdout robustness: parakeet-cli --verbose could leak non-JSON banner
# and summary lines to stdout alongside the JSONL segments. compute-der.py
# must skip those and score just the JSONL portion — otherwise a single
# upstream --verbose change (or a new backend init log) would silently null
# DER on every dispatch. This test would fail loudly under the pre-review
# `raise ValueError` behavior, which aborted the whole score on the first
# non-JSON line.
run_driver der-tw-mixed-stdout "$OUT/der-tw-mixed-stdout.json" "$OUT/der-tw-mixed-stdout.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .der_median == 0.0 and .correctness_kind == "der"' \
  "$OUT/der-tw-mixed-stdout.json" > /dev/null \
  || fail "der-tw-mixed-stdout: $(cat "$OUT/der-tw-mixed-stdout.json")"
grep -q 'skipping non-JSON hypothesis line' "$OUT/der-tw-mixed-stdout.err" \
  || fail "der-tw-mixed-stdout: expected 'skipping non-JSON' diagnostic on stderr"
ok "correctness (DER, time-wrapped): non-JSON stdout chatter (--verbose banners) is skipped and JSONL portion still scores correctly"

# Native-mode families have no diarize schema in --json-out today, so a
# native family declaring correctness.kind='der' must be skipped with a
# diagnostic rather than trying to score against garbage. This guards
# against a future WER→DER copy-paste in a native family's spec.
run_driver der-nat-warns "$OUT/der-nat-warns.json" "$OUT/der-nat-warns.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .der_median == null and .correctness_kind == null' \
  "$OUT/der-nat-warns.json" > /dev/null \
  || fail "der-nat-warns: $(cat "$OUT/der-nat-warns.json")"
grep -q "correctness.kind='der' requires text-file mode" "$OUT/der-nat-warns.err" \
  || fail "der-nat-warns: native-DER config diagnostic not surfaced"
ok "correctness (DER): native-mode families with kind='der' are skipped with a diagnostic"

# ---- correctness scoring (F1 path, time-wrapped) ---------------------------
# Every F1 cell uses ref_f1 = [{"start":2.0,"end":6.0}] (400 speech frames out
# of 800 total). Stubs vary only in what they emit on stdout; the driver must
# route the returned f1 into f1_median (not wer_median or der_median), accept
# both whisper-vad text and JSON-array hypothesis shapes, preserve the
# empty-hypothesis => F1 0.0 invariant, and reject a native family declaring
# kind='f1' with a config-error diagnostic.

run_driver f1-tw-perfect "$OUT/f1-tw-perfect.json" "$OUT/f1-tw-perfect.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .f1_median == 1.0 and .wer_median == null and .der_median == null
       and .correctness_kind == "f1"
       and (.correctness_reference | test("ref-f1\\.json$"))' \
  "$OUT/f1-tw-perfect.json" > /dev/null \
  || fail "f1-tw-perfect: $(cat "$OUT/f1-tw-perfect.json")"
[[ -f "$OUT/f1-tw-perfect.hypothesis.txt" ]] \
  || fail "f1-tw-perfect: hypothesis sibling not stashed next to result.json"
grep -q 'Speech segment 0' "$OUT/f1-tw-perfect.hypothesis.txt" \
  || fail "f1-tw-perfect: hypothesis sibling did not capture the stub's stdout"
ok "correctness (F1, time-wrapped): whisper-vad text stdout matching the reference scores F1 1.0 + hypothesis stashed as .hypothesis.txt"

# 2 s of hyp against 4 s of ref: precision = 200/200 = 1.0, recall = 200/400
# = 0.5, F1 = 2*1*0.5/1.5 = 2/3 ≈ 0.6666...
run_driver f1-tw-partial "$OUT/f1-tw-partial.json" "$OUT/f1-tw-partial.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .correctness_kind == "f1"
       and (.f1_median > 0.66 and .f1_median < 0.67)' \
  "$OUT/f1-tw-partial.json" > /dev/null \
  || fail "f1-tw-partial: $(cat "$OUT/f1-tw-partial.json")"
ok "correctness (F1, time-wrapped): hyp covering half the ref scores F1 ~2/3 (precision 1.0, recall 0.5)"

# Empty stdout => VAD emitted nothing => full recall miss => F1 0.0. Mirrors
# the WER/DER catastrophic-collapse regression guard: correctness scoring
# must not silently skip the "model returned nothing" case.
run_driver f1-tw-empty "$OUT/f1-tw-empty.json" "$OUT/f1-tw-empty.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .f1_median == 0.0 and .correctness_kind == "f1"' \
  "$OUT/f1-tw-empty.json" > /dev/null \
  || fail "f1-tw-empty: $(cat "$OUT/f1-tw-empty.json")"
ok "correctness (F1, time-wrapped): empty stdout capture => f1_median=0.0 (catastrophic-miss regression guard)"

# Parser dual-shape parity at the driver level: a stub emitting the same
# content as JSON-array must score identically to the whisper-vad text stub
# (both are F1 1.0). Pins the parse_hypothesis auto-detect path against a
# future regression where the driver could accidentally strip or normalize
# the leading '['.
run_driver f1-tw-jsonarr "$OUT/f1-tw-jsonarr.json" "$OUT/f1-tw-jsonarr.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .f1_median == 1.0 and .correctness_kind == "f1"' \
  "$OUT/f1-tw-jsonarr.json" > /dev/null \
  || fail "f1-tw-jsonarr: $(cat "$OUT/f1-tw-jsonarr.json")"
ok "correctness (F1, time-wrapped): JSON-array hypothesis shape parses identically to whisper-vad text"

# Missing reference => skip + diagnostic + no hypothesis sibling (matches
# the wer/der bad-ref handling).
run_driver f1-tw-badref "$OUT/f1-tw-badref.json" "$OUT/f1-tw-badref.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .f1_median == null and .correctness_kind == null' \
  "$OUT/f1-tw-badref.json" > /dev/null \
  || fail "f1-tw-badref: $(cat "$OUT/f1-tw-badref.json")"
grep -q 'reference file not found' "$OUT/f1-tw-badref.err" \
  || fail "f1-tw-badref: missing-reference diagnosis not surfaced"
[[ ! -f "$OUT/f1-tw-badref.hypothesis.txt" ]] \
  || fail "f1-tw-badref: hypothesis sibling should NOT exist when correctness was skipped"
ok "correctness (F1, time-wrapped): missing reference => f1_median=null + diagnostic + no hypothesis sibling (perf still ok)"

# Non-zero tolerance plumbing: proves families.json's correctness.tolerance_ms
# reaches compute-f1.py's --collar-ms. The 100 ms boundary slop scores
# ~0.975 F1 without a collar (compute-f1.py self-test pins that value);
# a 200 ms collar hides both slop regions, so F1 lands at 1.0 here.
# Mirrors the der-tw-collar plumbing test.
run_driver f1-tw-collar "$OUT/f1-tw-collar.json" "$OUT/f1-tw-collar.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .f1_median == 1.0 and .correctness_kind == "f1"' \
  "$OUT/f1-tw-collar.json" > /dev/null \
  || fail "f1-tw-collar: $(cat "$OUT/f1-tw-collar.json")"
ok "correctness (F1, time-wrapped): 200 ms tolerance hides boundary slop => F1 1.0 (tolerance_ms plumbed through)"

# Realistic multi-segment shape: the compute-f1.py self-test locks the
# centiseconds fix at the module level via the silero-on-jfk replay, but no
# driver cell exercises the same shape end-to-end. This one does: 3 hyp
# segments (in centiseconds) covering 300 out of 400 ref-speech frames with
# ~50 frame gaps. Precision = 1.0 (all hyp inside ref), recall = 300/400 =
# 0.75, F1 = 6/7 ≈ 0.8571. Under the pre-centiseconds-fix parser every
# hypothesis segment would be interpreted as being in seconds (200-300 s
# etc.) and land past the end of the audio window — a pre-fix run of this
# cell would report F1 = 0.0 and fail loudly.
run_driver f1-tw-silero-shape "$OUT/f1-tw-silero-shape.json" "$OUT/f1-tw-silero-shape.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .correctness_kind == "f1"
       and (.f1_median > 0.855 and .f1_median < 0.858)' \
  "$OUT/f1-tw-silero-shape.json" > /dev/null \
  || fail "f1-tw-silero-shape: $(cat "$OUT/f1-tw-silero-shape.json")"
ok "correctness (F1, time-wrapped): realistic multi-segment silero-shape hyp scores F1 6/7 (centiseconds fix pinned end-to-end)"

# Native families have no VAD schema in --json-out, so a native spec that
# declares kind='f1' must be skipped with a diagnostic — guards against a
# WER→F1 copy-paste in a native family spec, same principle as der-nat-warns.
run_driver f1-nat-warns "$OUT/f1-nat-warns.json" "$OUT/f1-nat-warns.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .f1_median == null and .correctness_kind == null' \
  "$OUT/f1-nat-warns.json" > /dev/null \
  || fail "f1-nat-warns: $(cat "$OUT/f1-nat-warns.json")"
grep -q "correctness.kind='f1' requires text-file mode" "$OUT/f1-nat-warns.err" \
  || fail "f1-nat-warns: native-F1 config diagnostic not surfaced"
ok "correctness (F1): native-mode families with kind='f1' are skipped with a diagnostic"

# ---- 3. the workflow's plan-job expansion ------------------------------------
# The `expand` step's shell is extracted from the workflow and run against
# the real families.json, so the family whitelist, the "all" expansion, the
# per-family runner restriction and the per-cell timeout are checked before
# a dispatch spends runner minutes on them.

extract_plan_script() {
  awk '
    /^      - id: expand$/ { in_step = 1; next }
    in_step && /^        run: \|$/ { in_run = 1; next }
    in_run && /^      - / { exit }
    in_run && /^  # ---------- 2\./ { exit }
    in_run { sub(/^          /, ""); print }
  ' "$DESKTOP_WORKFLOW"
}
PLAN_SCRIPT="$TMP/plan.sh"
extract_plan_script > "$PLAN_SCRIPT"
[[ -s "$PLAN_SCRIPT" ]] || fail "could not extract the plan job's expand script from $DESKTOP_WORKFLOW"
bash -n "$PLAN_SCRIPT" || fail "extracted plan script does not parse"

run_plan() {
  # run_plan <out-file> [VAR=VALUE...]; stdout/stderr go to <out-file>.log
  local out="$1"; shift
  : > "$out"
  env -i PATH="$PATH" HOME="$TMP" GITHUB_OUTPUT="$out" \
    ALL_FAMILIES="$workflow_families" FAMILIES_SPEC="$REAL_SPEC" \
    IN_FAMILIES=all IN_RUNNERS=linux,macos IN_WHISPER=tiny IN_RUNS=1 IN_MINIMAX_DIR="" IN_MINIMAX_RUN="" \
    "$@" bash "$PLAN_SCRIPT" > "$out.log" 2>&1
}
plan_matrix() { grep -E '^matrix=' "$1" | sed 's/^matrix=//'; }

cd "$HERE/../.."
run_plan "$TMP/plan-all.out" || fail "plan rejected the default 'all' dispatch: $(cat "$TMP/plan-all.out.log")"
all_families_planned="$(plan_matrix "$TMP/plan-all.out" | jq -r '[.include[].family] | unique | sort | join(",")')"
default_families="$(jq -r '[to_entries[] | select(.key | startswith("_") | not) | select(.value.opt_in != true) | .key] | sort | join(",")' "$REAL_SPEC")"
[[ "$all_families_planned" == "$default_families" ]] \
  || fail "'all' must expand to every non-opt-in family; planned '$all_families_planned', expected '$default_families'"
plan_matrix "$TMP/plan-all.out" | jq -e '[.include[] | select(.family == "parakeet-nemotron")] | length == 2' > /dev/null \
  || fail "'all' must plan parakeet-nemotron on both runners (it was missing from the old hardcoded whitelist)"
plan_matrix "$TMP/plan-all.out" | jq -e --slurpfile spec "$REAL_SPEC" \
  '[.include[] | .timeout_minutes == ($spec[0][.family].timeout_minutes // 60)] | all' > /dev/null \
  || fail "every cell must carry its family's timeout_minutes (60 when the spec sets none)"
[[ "$(grep -E '^expected_count=' "$TMP/plan-all.out")" == "expected_count=$(plan_matrix "$TMP/plan-all.out" | jq '.include | length')" ]] \
  || fail "expected_count must equal the number of planned cells"
ok "plan: 'all' expands to every non-opt-in family on both runners"

run_plan "$TMP/plan-optin.out" IN_FAMILIES="moss-tts,moss-speech,parakeet-ctc" \
  || fail "plan rejected explicitly named opt-in families: $(cat "$TMP/plan-optin.out.log")"
plan_matrix "$TMP/plan-optin.out" | jq -e '
  ([.include[] | select(.family == "moss-tts")] | length == 1) and
  ([.include[] | select(.family == "moss-tts")][0] | .runner_label == "macos" and .timeout_minutes == 120) and
  ([.include[] | select(.family == "moss-speech")] | length == 1) and
  ([.include[] | select(.family == "parakeet-ctc")] | length == 2)' > /dev/null \
  || fail "plan must restrict macOS-only families to the macos cell with their raised timeout: $(plan_matrix "$TMP/plan-optin.out")"
grep -q 'moss-tts is restricted to macos; skipping the linux cell' "$TMP/plan-optin.out.log" \
  || fail "plan must say why a restricted family's cell was dropped"
ok "plan: runners/timeout_minutes restrict and time the opt-in cells"

if run_plan "$TMP/plan-bad.out" IN_FAMILIES="whisper,not-a-family"; then
  fail "plan accepted an unknown family"
fi
grep -q 'rejected family: not-a-family' "$TMP/plan-bad.out.log" \
  || fail "plan must name the rejected family: $(cat "$TMP/plan-bad.out.log")"
ok "plan: unknown families are rejected before any runner is scheduled"

# A macOS-only family dispatched on linux alone leaves no cell. An empty
# include list would pass the bench job's `matrix != ''` guard and fail at
# matrix expansion with no hint, so the plan must refuse it and say why.
if run_plan "$TMP/plan-empty.out" IN_FAMILIES="moss-tts" IN_RUNNERS="linux"; then
  fail "plan accepted a selection that yields no cells"
fi
grep -q "no benchmark cells: families 'moss-tts' accept none of the requested runners (linux)" "$TMP/plan-empty.out.log" \
  || fail "plan must explain the empty selection: $(cat "$TMP/plan-empty.out.log")"
grep -qE '^matrix=' "$TMP/plan-empty.out" && fail "plan must not emit a matrix for an empty selection"
ok "plan: a selection with no cells is rejected with a diagnostic"

# Music driver, cache and pilot aggregation tests use only the standard library.
python3 "$HERE/test-music-alignment-driver.py" || fail "music driver regressions"
ok "music driver output isolation, statuses and provenance"
python3 "$HERE/test_music_preparation.py" || fail "music model preparation regressions"
ok "music model cache integrity"
python3 "$HERE/test_music_pilot.py" || fail "music pilot regressions"
ok "music paired pilot aggregation and coverage"

echo "all $PASS checks passed"
