#!/usr/bin/env bash
# Download the official Supertonic ONNX/assets bundle through Hugging Face and
# convert it into the local GGUF expected by the C++ runtime.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
ARCH="supertonic2"
REPO_ID=""
OUT=""
EXTRA_ARGS=()

usage() {
    cat >&2 <<'EOF'
usage: bash scripts/setup-supertonic2.sh [options] [converter options]

options:
  --arch supertonic2|supertonic  Which upstream bundle to download.
                                default: supertonic2
  --repo-id REPO                Hugging Face repo override.
                                default: Supertone/supertonic-2 or Supertone/supertonic
  --out PATH                    Output GGUF path.
                                default: models/supertonic2.gguf or models/supertonic.gguf
  --python PATH                 Python interpreter. default: $PYTHON or python3
  -h, --help                    Show this help.

Any unknown options are forwarded to scripts/convert-supertonic2-to-gguf.py,
for example: --ftype q8_0 --hf-token TOKEN --download-dir /tmp/supertonic2
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)
            [[ $# -ge 2 ]] || { echo "error: --arch requires a value" >&2; exit 2; }
            ARCH="$2"
            shift 2
            ;;
        --repo-id)
            [[ $# -ge 2 ]] || { echo "error: --repo-id requires a value" >&2; exit 2; }
            REPO_ID="$2"
            shift 2
            ;;
        --out)
            [[ $# -ge 2 ]] || { echo "error: --out requires a value" >&2; exit 2; }
            OUT="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "error: --python requires a value" >&2; exit 2; }
            PYTHON="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$ARCH" in
    supertonic2)
        REPO_ID="${REPO_ID:-Supertone/supertonic-2}"
        OUT="${OUT:-$ROOT/models/supertonic2.gguf}"
        ;;
    supertonic)
        REPO_ID="${REPO_ID:-Supertone/supertonic}"
        OUT="${OUT:-$ROOT/models/supertonic.gguf}"
        ;;
    *)
        echo "error: --arch must be 'supertonic2' or 'supertonic'" >&2
        exit 2
        ;;
esac

exec "$PYTHON" "$ROOT/scripts/convert-supertonic2-to-gguf.py" \
    --arch "$ARCH" \
    --repo-id "$REPO_ID" \
    --out "$OUT" \
    --validate \
    "${EXTRA_ARGS[@]}"
