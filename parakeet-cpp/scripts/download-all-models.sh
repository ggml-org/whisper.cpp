#!/usr/bin/env bash
#
# Download every Parakeet checkpoint we know about into parakeet.cpp/models/
# as `.nemo` archives, ready for `convert-nemo-to-gguf.py`.
#
# Idempotent: skips files that already exist on disk. Re-run any time to top up.
# Total download budget on a clean machine: ~14 GiB at the time of writing
# (TDT v3 + TDT 1.1b + CTC 0.6b + CTC 1.1b + TDT_CTC hybrid + EOU 120M +
# Sortformer v1 + streaming Sortformer v2). Already-cached checkpoints are
# untouched.
#
# Usage:
#     ./scripts/download-all-models.sh             # everything
#     ./scripts/download-all-models.sh tdt         # only the TDT v3 pair

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_DIR="$REPO_ROOT/models"

mkdir -p "$NEMO_DIR"

case "${1:-all}" in
  all) ;;
  tdt) ;;  # filtered below
  *) echo "usage: $0 [all|tdt]" >&2; exit 2 ;;
esac

bytes_human() {
  local b=$1
  if   (( b > 1<<30 )); then printf "%.2f GiB" "$(echo "$b / (1<<30)" | bc -l)"
  elif (( b > 1<<20 )); then printf "%.2f MiB" "$(echo "$b / (1<<20)" | bc -l)"
  else                       printf "%d B"     "$b"
  fi
}

fetch() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    local sz; sz=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest")
    echo "  exists: $dest ($(bytes_human "$sz")) — skipping"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "  fetching: $url"
  echo "          -> $dest"
  curl -L --fail --progress-bar -o "$dest.tmp" "$url"
  mv "$dest.tmp" "$dest"
  local sz; sz=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest")
  echo "  saved: $dest ($(bytes_human "$sz"))"
}

hr() { printf '%.0s=' {1..70}; echo; }

# The TDT block always runs (both `all` and the `tdt` filter need it).
# (When the filter is `tdt`, only the TDT-related blocks fire; the
# CTC / EOU / Sortformer blocks below are gated off.)
hr
echo "== nemo: parakeet-tdt-0.6b-v3 (multilingual, 25 langs, +PnC, ~2.4 GiB)"
fetch "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/parakeet-tdt-0.6b-v3.nemo" \
      "$NEMO_DIR/parakeet-tdt-0.6b-v3.nemo"

hr
echo "== nemo: parakeet-tdt-1.1b (English-only, best WER, 42 layers, ~4.3 GiB)"
fetch "https://huggingface.co/nvidia/parakeet-tdt-1.1b/resolve/main/parakeet-tdt-1.1b.nemo" \
      "$NEMO_DIR/parakeet-tdt-1.1b.nemo"

if [[ "${1:-all}" != "tdt" ]]; then
  hr
  echo "== nemo: parakeet-ctc-0.6b (English, ~2.3 GiB)"
  fetch "https://huggingface.co/nvidia/parakeet-ctc-0.6b/resolve/main/parakeet-ctc-0.6b.nemo" \
        "$NEMO_DIR/parakeet-ctc-0.6b.nemo"

  hr
  echo "== nemo: parakeet-ctc-1.1b (English, ~4 GiB)"
  fetch "https://huggingface.co/nvidia/parakeet-ctc-1.1b/resolve/main/parakeet-ctc-1.1b.nemo" \
        "$NEMO_DIR/parakeet-ctc-1.1b.nemo"

  hr
  echo "== nemo: parakeet-tdt_ctc-110m (small TDT+CTC hybrid, ~440 MiB)"
  echo "         (forward-looking: not yet wired into the C++ Engine; cached"
  echo "          for the planned hybrid TDT+CTC port.)"
  fetch "https://huggingface.co/nvidia/parakeet-tdt_ctc-110m/resolve/main/parakeet-tdt_ctc-110m.nemo" \
        "$NEMO_DIR/parakeet-tdt_ctc-110m.nemo"

  hr
  echo "== nemo: parakeet_realtime_eou_120m-v1 (EOU streaming, FastConformer-RNNT 120M, ~440 MiB)"
  echo "         (cache-aware streaming with att_context_size=[70,1] + <EOU>"
  echo "          end-of-utterance token; English only.)"
  fetch "https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1/resolve/main/parakeet_realtime_eou_120m-v1.nemo" \
        "$NEMO_DIR/parakeet_realtime_eou_120m-v1.nemo"

  hr
  echo "== nemo: diar_sortformer_4spk-v1 (4-speaker diarization, offline, ~490 MiB)"
  fetch "https://huggingface.co/nvidia/diar_sortformer_4spk-v1/resolve/main/diar_sortformer_4spk-v1.nemo" \
        "$NEMO_DIR/diar_sortformer_4spk-v1.nemo"

  hr
  echo "== nemo: diar_streaming_sortformer_4spk-v2 (4-speaker, streaming-trained, ~470 MiB)"
  fetch "https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2/resolve/main/diar_streaming_sortformer_4spk-v2.nemo" \
        "$NEMO_DIR/diar_streaming_sortformer_4spk-v2.nemo"
fi

hr
echo "done. Cached checkpoints:"
echo
ls -lh "$NEMO_DIR" | awk '/\.nemo$/ {print "  " $9, $5}' || true
