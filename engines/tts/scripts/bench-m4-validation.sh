#!/usr/bin/env bash
# scripts/bench-m4-validation.sh
#
# Self-contained bench + parity harness for the §3.24–§3.30 Metal portfolio.
# Intended to run on M4 Air / M4 Pro / iPad Pro or any Apple-silicon Mac.
# The §3.27 / §3.28 / §3.30 kernel work is predicted to be proportionally
# larger win on M4 vs the M3 Ultra reference (neutral on M3U due to the
# chip's very low per-dispatch overhead + high core count).  This script
# lets you either confirm or falsify that prediction with one command.
#
# Usage:
#
#   # From a fresh clone of chatterbox.cpp @ multilingual_merged:
#   cd chatterbox.cpp
#   # Grab the voice fixture (any 16 kHz WAV; jfk.wav is the reference):
#   scp <m3u>:/tmp/jfk.wav /tmp/jfk.wav
#   # Make sure you have the model GGUFs (14 GB total):
#   #   models/chatterbox-t3-mtl-q4_0.gguf
#   #   models/chatterbox-s3gen-mtl-q4_0_hift_f16_v2.gguf
#   bash scripts/bench-m4-validation.sh 2>&1 | tee m4-bench.log
#
# Compares current hardware results to the M3 Ultra reference
# captured in PROGRESS §3.30.  Writes JSON to artifacts/bench/ for archiving.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

REFERENCE_MD5="d8a1b22375dbcb2259c686426a7d76c5"
REFERENCE_TEXT="Hola mundo, esta es una prueba multilingue."
REFERENCE_LANG="es"
T3_GGUF="${T3_GGUF:-models/chatterbox-t3-mtl-q4_0.gguf}"
S3GEN_GGUF="${S3GEN_GGUF:-models/chatterbox-s3gen-mtl-q4_0_hift_f16_v2.gguf}"
REF_WAV="${REF_WAV:-/tmp/jfk.wav}"
OUT_DIR="${OUT_DIR:-artifacts/bench}"
RUNS="${RUNS:-5}"

# M3 Ultra reference numbers (post-§3.30, 5 invocations averaged)
M3U_CFM_MS=534.0
M3U_S3GEN_MS=706.6
M3U_T3_MS=432.6
M3U_HIFT_MS=121.1

for f in "$T3_GGUF" "$S3GEN_GGUF" "$REF_WAV"; do
    if [ ! -f "$f" ]; then
        echo "FAIL: required file not found: $f" >&2
        exit 1
    fi
done

HOST_CHIP="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': +' '/Chip:/ {print $2; exit}')"
HOST_MODEL="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': +' '/Model Identifier:/ {print $2; exit}')"

echo "=== Host ==="
echo "Chip:  ${HOST_CHIP:-unknown}"
echo "Model: ${HOST_MODEL:-unknown}"

echo ""
echo "=== Setup ggml (apply Metal + OpenCL patches at pinned commit) ==="
if [ ! -d ggml/.git ]; then
    bash scripts/setup-ggml.sh
else
    echo "ggml/ already present; skipping.  To force a reapply of the patches, remove ggml/ first."
fi

echo ""
echo "=== Build ==="
cmake -S . -B build-metal \
    -DGGML_METAL=ON -DGGML_BLAS=OFF -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build-metal -j --target chatterbox test-metal-ops 2>&1 | tail -3

echo ""
echo "=== test-metal-ops (14 gates: 3 base + 3 conv + 8 fused-mul_mm) ==="
if ./build-metal/test-metal-ops 2>&1 | tee /tmp/m4-metal-ops.log | grep -E "^OK|^FAIL" | tail -20; then
    if grep -q "FAIL" /tmp/m4-metal-ops.log; then
        echo "FAIL: test-metal-ops has failures"
        exit 1
    fi
    echo "test-metal-ops: all gates PASS"
else
    echo "FAIL: test-metal-ops did not produce output"
    exit 1
fi

echo ""
echo "=== Bench: ${RUNS} invocations (Q4_0 + HiFT F16 v2, ES prompt, seed 42) ==="

# Collect per-invocation stats
CFM_MS=()
S3GEN_MS=()
T3_MS=()
HIFT_MS=()
MD5S=()

mkdir -p "$OUT_DIR"

for i in $(seq 1 "$RUNS"); do
    OUT="/tmp/cb_m4_${i}.wav"
    LOG="/tmp/cb_m4_${i}.log"
    ./build-metal/chatterbox \
        --model "$T3_GGUF" \
        --s3gen-gguf "$S3GEN_GGUF" \
        --reference-audio "$REF_WAV" \
        --text "$REFERENCE_TEXT" \
        --language "$REFERENCE_LANG" \
        --seed 42 --temp 0 --top-k 1 \
        --n-gpu-layers 1 \
        --out "$OUT" \
        --verbose 2>&1 | grep -vE "^ggml_metal_" > "$LOG"

    cfm=$(awk '/\[cfm_total\]/ {print $2}' "$LOG")
    hift=$(awk '/\[hift_decode\]/ {print $2}' "$LOG")
    s3gen=$(awk '/S3GEN_INFER_MS/ {gsub("S3GEN_INFER_MS=", "", $2); print $2}' "$LOG" | head -1)
    t3=$(awk '/T3_INFER_MS/ {gsub("T3_INFER_MS=", "", $2); print $2}' "$LOG")

    CFM_MS+=("$cfm")
    S3GEN_MS+=("$s3gen")
    T3_MS+=("$t3")
    HIFT_MS+=("$hift")

    md5=$(md5 -q "$OUT")
    MD5S+=("$md5")
    printf "run %d: cfm=%s  s3gen=%s  t3=%s  hift=%s  md5=%s\n" \
        "$i" "${cfm:-?}" "${s3gen:-?}" "${t3:-?}" "${hift:-?}" "${md5:0:12}"
done

# Compute means (awk — no bash floats)
mean() {
    printf '%s\n' "$@" | awk 'BEGIN{s=0;n=0} {s+=$1; n++} END{if (n>0) printf "%.1f", s/n; else print "?"}'
}

CFM_MEAN=$(mean "${CFM_MS[@]}")
S3GEN_MEAN=$(mean "${S3GEN_MS[@]}")
T3_MEAN=$(mean "${T3_MS[@]}")
HIFT_MEAN=$(mean "${HIFT_MS[@]}")

echo ""
echo "=== Summary: ${HOST_CHIP:-this host} vs M3 Ultra reference ==="
printf "%-20s %15s %15s %15s\n" "stage" "M3 Ultra (ref)" "this host"     "Δ vs M3U"
printf "%-20s %15.1f %15s %15s\n" "[cfm_total] ms"  "$M3U_CFM_MS"   "$CFM_MEAN"   "$(awk -v a=$CFM_MEAN   -v b=$M3U_CFM_MS   'BEGIN{d=a-b; r=(d/b)*100; printf "%+.1f (%+.1f%%)", d, r}')"
printf "%-20s %15.1f %15s %15s\n" "S3GEN_INFER_MS"  "$M3U_S3GEN_MS" "$S3GEN_MEAN" "$(awk -v a=$S3GEN_MEAN -v b=$M3U_S3GEN_MS 'BEGIN{d=a-b; r=(d/b)*100; printf "%+.1f (%+.1f%%)", d, r}')"
printf "%-20s %15.1f %15s %15s\n" "T3_INFER_MS"     "$M3U_T3_MS"    "$T3_MEAN"    "$(awk -v a=$T3_MEAN    -v b=$M3U_T3_MS    'BEGIN{d=a-b; r=(d/b)*100; printf "%+.1f (%+.1f%%)", d, r}')"
printf "%-20s %15.1f %15s %15s\n" "[hift_decode] ms" "$M3U_HIFT_MS"  "$HIFT_MEAN"  "$(awk -v a=$HIFT_MEAN  -v b=$M3U_HIFT_MS  'BEGIN{d=a-b; r=(d/b)*100; printf "%+.1f (%+.1f%%)", d, r}')"

# MD5 comparison: all runs must produce identical output (determinism) and
# the value must match the M3 Ultra reference (byte-exactness across chips).
UNIQUE_MD5=$(printf '%s\n' "${MD5S[@]}" | sort -u | wc -l | tr -d ' ')
FIRST_MD5="${MD5S[0]}"

echo ""
echo "=== Parity ==="
if [ "$UNIQUE_MD5" = "1" ]; then
    echo "determinism: PASS  (md5 $FIRST_MD5 stable across ${RUNS} runs)"
else
    echo "determinism: FAIL  (got $UNIQUE_MD5 distinct md5s across ${RUNS} runs)"
fi

if [ "$FIRST_MD5" = "$REFERENCE_MD5" ]; then
    echo "byte-exact vs M3 Ultra: PASS ($FIRST_MD5)"
else
    echo "byte-exact vs M3 Ultra: DIFFER"
    echo "  M3 Ultra reference: $REFERENCE_MD5"
    echo "  $HOST_CHIP:         $FIRST_MD5"
    echo "  (small divergence expected across chip generations from Q4_0-dequant-order + bias-fusion accumulation;"
    echo "   listen to /tmp/cb_m4_1.wav to verify audio sounds correct)"
fi

# Write JSON summary
JSON="$OUT_DIR/m4-validation.json"
cat > "$JSON" <<EOF
{
  "host_chip": "${HOST_CHIP:-unknown}",
  "host_model": "${HOST_MODEL:-unknown}",
  "runs": ${RUNS},
  "t3_gguf": "$T3_GGUF",
  "s3gen_gguf": "$S3GEN_GGUF",
  "reference_wav": "$REF_WAV",
  "m3u_reference": {
    "cfm_ms": $M3U_CFM_MS,
    "s3gen_ms": $M3U_S3GEN_MS,
    "t3_ms": $M3U_T3_MS,
    "hift_ms": $M3U_HIFT_MS,
    "md5": "$REFERENCE_MD5"
  },
  "this_host": {
    "cfm_ms_mean": $CFM_MEAN,
    "s3gen_ms_mean": $S3GEN_MEAN,
    "t3_ms_mean": $T3_MEAN,
    "hift_ms_mean": $HIFT_MEAN,
    "md5": "$FIRST_MD5",
    "determinism_ok": $([ "$UNIQUE_MD5" = "1" ] && echo true || echo false)
  }
}
EOF

echo ""
echo "wrote $JSON"
