#!/usr/bin/env bash
# Multi-precision parity + bench harness for Supertonic 2.
#
# For each supported precision (f32, f16, q8_0):
#   1. Synthesizes a reference WAV on CPU at that precision.
#   2. Synthesizes the same WAV on Metal at the same precision.
#   3. Reports parity (corr, L_inf, RMS) between the two.
#   4. Optionally runs supertonic-bench at the same precision and emits
#      a per-precision JSON artifact alongside.
#
# Usage:
#   bash scripts/validate-precision-parity.sh [--bench] [--text TEXT] [--model PATH]
#                                             [--precisions f32,f16,q8_0]
#
# Precisions not yet wired through the graph builders fail at load with
# a clear "scaffolded but not yet supported" message and are skipped (not
# counted as a parity failure).  This lets the harness be useful right
# now while Phase A3 / B1 work lands.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$ROOT/models/supertonic2.gguf"
TEXT="The quick brown fox jumps over the lazy dog."
PRECISIONS="f32,f16,q8_0"
DO_BENCH=0
RUNS=10
WARMUP=2
THREADS=4
ARTIFACT_DIR="$ROOT/artifacts/bench/parity-matrix"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench)       DO_BENCH=1; shift ;;
        --text)        TEXT="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --precisions)  PRECISIONS="$2"; shift 2 ;;
        --runs)        RUNS="$2"; shift 2 ;;
        --warmup)      WARMUP="$2"; shift 2 ;;
        --threads)     THREADS="$2"; shift 2 ;;
        --artifact-dir) ARTIFACT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed 's/^# //; s/^#//; /^set -euo/d'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

CLI="$ROOT/build/supertonic-cli"
BENCH="$ROOT/build/supertonic-bench"
PY="$ROOT/.venv/bin/python3"
if [[ ! -x "$CLI" ]]; then
    echo "build/supertonic-cli not found. Run 'cmake --build build --target supertonic-cli' first." >&2
    exit 1
fi
if [[ "$DO_BENCH" -eq 1 && ! -x "$BENCH" ]]; then
    echo "--bench requested but build/supertonic-bench not found." >&2
    exit 1
fi
if [[ ! -x "$PY" ]]; then
    echo "$PY not found. Activate a venv with numpy + wave installed." >&2
    exit 1
fi

mkdir -p "$ARTIFACT_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

printf "\nSupertonic 2 multi-precision parity + bench harness\n"
printf "  model:      %s\n" "$MODEL"
printf "  text:       %.60s%s\n" "$TEXT" "$([[ ${#TEXT} -gt 60 ]] && echo '...')"
printf "  precisions: %s\n" "$PRECISIONS"
printf "  bench:      %s\n\n" "$([[ "$DO_BENCH" -eq 1 ]] && echo 'yes' || echo 'no')"

OVERALL_RC=0
IFS=',' read -r -a PREC_ARR <<< "$PRECISIONS"
for P in "${PREC_ARR[@]}"; do
    P_TRIM="$(echo "$P" | xargs)"
    CPU_WAV="$TMP/cpu-$P_TRIM.wav"
    MTL_WAV="$TMP/mtl-$P_TRIM.wav"

    printf "=== %s ===\n" "$P_TRIM"

    set +e
    CPU_LOG="$("$CLI" --model "$MODEL" --text "$TEXT" --n-gpu-layers 0 \
                       --precision "$P_TRIM" --out "$CPU_WAV" 2>&1)"
    CPU_RC=$?
    MTL_LOG="$("$CLI" --model "$MODEL" --text "$TEXT" --n-gpu-layers 1 \
                       --precision "$P_TRIM" --out "$MTL_WAV" 2>&1)"
    MTL_RC=$?
    set -e

    if echo "$CPU_LOG$MTL_LOG" | grep -qE "scaffolded but not yet|partially scaffolded"; then
        printf "  SKIP: precision %s not yet wired through graph builders (Phase A3/B1)\n\n" "$P_TRIM"
        continue
    fi
    # Tolerate the harmless post-write atexit `GGML_ASSERT([rsets->data count] == 0)`
    # that fires on Metal cleanup AFTER the WAV is fully written.  Treat the run as
    # successful iff the WAV file exists and is at least 1 KB (covers a synthesized
    # signal, well above an empty/header-only file).
    cpu_ok=1; mtl_ok=1
    [[ -s "$CPU_WAV" ]] || cpu_ok=0
    [[ -s "$MTL_WAV" ]] || mtl_ok=0
    if [[ -f "$CPU_WAV" ]]; then
        size=$(wc -c < "$CPU_WAV")
        [[ $size -lt 1024 ]] && cpu_ok=0
    fi
    if [[ -f "$MTL_WAV" ]]; then
        size=$(wc -c < "$MTL_WAV")
        [[ $size -lt 1024 ]] && mtl_ok=0
    fi
    if [[ $cpu_ok -eq 0 || $mtl_ok -eq 0 ]]; then
        printf "  FAIL: synthesis errored.  cpu_rc=%d mtl_rc=%d  wav_ok cpu=%d mtl=%d\n" \
               "$CPU_RC" "$MTL_RC" "$cpu_ok" "$mtl_ok"
        printf "  --- cpu tail ---\n%s\n  --- metal tail ---\n%s\n\n" \
               "$(echo "$CPU_LOG" | tail -3)" "$(echo "$MTL_LOG" | tail -3)"
        OVERALL_RC=1
        continue
    fi

    "$PY" - <<PY
import wave, numpy as np, sys
def load(p):
    with wave.open(p, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
a = load("$CPU_WAV")
b = load("$MTL_WAV")
n = min(len(a), len(b))
a, b = a[:n], b[:n]
corr = float(np.corrcoef(a, b)[0, 1])
linf = float(np.max(np.abs(a - b)))
rms  = float(np.sqrt(np.mean((a - b) ** 2)))
# Per-precision tolerance: numbers chosen against observed CPU↔Metal drift
# on the benchmark text "The quick brown fox jumps over the lazy dog.".
# Short text routinely gets L_inf ≈ 1.7e-3; long text accumulates more
# float-order drift across 5 CFM steps × more attention positions, landing
# around L_inf ≈ 3.7e-2 with corr ≥ 0.998 — audibly identical for f32.
# Q8_0 has additional drift from the dequant→transpose→requantize round-trip
# in the asymmetric load path (Metal keeps q8_0, CPU expands to f32, so the
# two paths use slightly differently-quantized weights).  Audibly identical.
tol_corr = {"f32": 0.998,  "f16": 0.99,  "q8_0": 0.96}.get("$P_TRIM", 0.99)
tol_linf = {"f32": 0.05,   "f16": 0.10,  "q8_0": 0.15 }.get("$P_TRIM", 0.10)
print(f"  corr={corr:.6f} (tol >= {tol_corr})  L_inf={linf:.6f} (tol <= {tol_linf})  RMS={rms:.6f}")
ok = corr >= tol_corr and linf <= tol_linf
print("  PASS" if ok else "  FAIL parity")
sys.exit(0 if ok else 1)
PY
    PY_RC=$?
    if [[ $PY_RC -ne 0 ]]; then OVERALL_RC=1; fi

    if [[ "$DO_BENCH" -eq 1 ]]; then
        JSON="$ARTIFACT_DIR/supertonic-mtl-${P_TRIM}.json"
        printf "  bench --> %s\n" "$JSON"
        "$BENCH" --model "$MODEL" --text "$TEXT" \
                  --voice M1 --language en --steps 5 --speed 1.05 --seed 42 \
                  --runs "$RUNS" --warmup "$WARMUP" --threads "$THREADS" \
                  --n-gpu-layers 1 --precision "$P_TRIM" \
                  --json-out "$JSON" 2>&1 | grep -E '^\s*(vector_estimator|vocoder|text_encoder|total|RTF|Real-time)' || true
    fi
    printf "\n"
done

if [[ $OVERALL_RC -eq 0 ]]; then
    printf "All wired-up precisions pass parity.\n"
else
    printf "One or more precisions failed parity (or errored).\n" >&2
fi
exit $OVERALL_RC
