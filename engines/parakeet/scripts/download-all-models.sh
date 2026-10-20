#!/usr/bin/env bash
#
# Download every Parakeet checkpoint we know about into parakeet.cpp/models/
# as `.nemo` archives, ready for `convert-nemo-to-gguf.py`.
#
# Idempotent: skips files that already exist on disk. Re-run any time to top up.
# Total download budget on a clean machine: ~19.3 GiB at the time of writing
# (TDT v3 + TDT 1.1b + Unified 0.6b + CTC 0.6b + CTC 1.1b + IndicConformer +
# TDT_CTC hybrid + EOU 120M + Sortformer v1 + streaming Sortformer v2 +
# streaming Sortformer v2.1).
# Already-cached checkpoints are untouched.
#
# TranscribePsy is published as ready-made GGUF and Core ML files rather than
# a .nemo archive, so its selector downloads the runtime artifacts directly.
#
# Usage:
#     ./scripts/download-all-models.sh               # everything
#     ./scripts/download-all-models.sh tdt           # only the TDT pair
#     ./scripts/download-all-models.sh eou           # only EOU 120M
#     ./scripts/download-all-models.sh transcribepsy # TranscribePsy GGUF + Core ML files

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_DIR="$REPO_ROOT/models"

mkdir -p "$NEMO_DIR"

case "${1:-all}" in
  all) ;;
  tdt) ;;  # filtered below
  eou) ;;  # filtered below
  transcribepsy) ;;  # filtered below
  *) echo "usage: $0 [all|tdt|eou|transcribepsy]" >&2; exit 2 ;;
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

TRANSCRIBEPSY_GGUF_REPO="https://huggingface.co/qvac/TranscribePsy-250M-EU-sub-gguf/resolve/main"
TRANSCRIBEPSY_COREML_REPO="https://huggingface.co/qvac/TranscribePsy-250M-EU-sub-coreml/resolve/main"
TRANSCRIBEPSY_COREML_STEM="transcribepsy-250m-eu-sub-coreml"
MLMODELC_FILES=(analytics/coremldata.bin coremldata.bin metadata.json model.mil weights/weight.bin)

# A compiled Core ML bundle is a directory; fetch each of its files so the
# bundle lands next to its GGUF under the name the engine derives.
fetch_mlmodelc() {
  local repo="$1" bundle="$2" file
  for file in "${MLMODELC_FILES[@]}"; do
    fetch "$repo/$bundle/$file" "$NEMO_DIR/$bundle/$file"
  done
}

fetch_transcribepsy() {
  hr
  echo "== gguf: TranscribePsy-250M-EU-sub (FastConformer-TDT 254M, 8 EU languages, q8_0, ~305 MB)"
  fetch "$TRANSCRIBEPSY_GGUF_REPO/transcribepsy-250m-eu-sub.gguf" \
        "$NEMO_DIR/transcribepsy-250m-eu-sub.gguf"

  hr
  echo "== coreml: TranscribePsy-250M-EU-sub (encoder-less GGUF + Neural Engine encoder/decoder, ~212 MB)"
  fetch "$TRANSCRIBEPSY_COREML_REPO/$TRANSCRIBEPSY_COREML_STEM.gguf" \
        "$NEMO_DIR/$TRANSCRIBEPSY_COREML_STEM.gguf"
  fetch_mlmodelc "$TRANSCRIBEPSY_COREML_REPO" "$TRANSCRIBEPSY_COREML_STEM-encoder.mlmodelc"
  fetch_mlmodelc "$TRANSCRIBEPSY_COREML_REPO" "$TRANSCRIBEPSY_COREML_STEM-decoder.mlmodelc"
}

if [[ "${1:-all}" == "transcribepsy" || "${1:-all}" == "all" ]]; then
  fetch_transcribepsy
fi

if [[ "${1:-all}" != "eou" && "${1:-all}" != "transcribepsy" ]]; then
  hr
  echo "== nemo: parakeet-tdt-0.6b-v3 (multilingual, 25 langs, +PnC, ~2.4 GiB)"
  fetch "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/parakeet-tdt-0.6b-v3.nemo" \
        "$NEMO_DIR/parakeet-tdt-0.6b-v3.nemo"

  hr
  echo "== nemo: parakeet-tdt-1.1b (English-only, best WER, 42 layers, ~4.3 GiB)"
  fetch "https://huggingface.co/nvidia/parakeet-tdt-1.1b/resolve/main/parakeet-tdt-1.1b.nemo" \
        "$NEMO_DIR/parakeet-tdt-1.1b.nemo"
fi

if [[ "${1:-all}" == "all" ]]; then
  hr
  echo "== nemo: parakeet-unified-en-0.6b (English RNN-T, ~2.4 GiB)"
  fetch "https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/parakeet-unified-en-0.6b.nemo" \
        "$NEMO_DIR/parakeet-unified-en-0.6b.nemo"

  hr
  echo "== nemo: parakeet-ctc-0.6b (English, ~2.3 GiB)"
  fetch "https://huggingface.co/nvidia/parakeet-ctc-0.6b/resolve/main/parakeet-ctc-0.6b.nemo" \
        "$NEMO_DIR/parakeet-ctc-0.6b.nemo"

  hr
  echo "== nemo: parakeet-ctc-1.1b (English, ~4 GiB)"
  fetch "https://huggingface.co/nvidia/parakeet-ctc-1.1b/resolve/main/parakeet-ctc-1.1b.nemo" \
        "$NEMO_DIR/parakeet-ctc-1.1b.nemo"

  hr
  echo "== nemo: indicconformer_stt_multi_hybrid_rnnt_600m (AI4Bharat, 22 Indic langs, ~2.4 GiB)"
  echo "         (hybrid RNNT+CTC checkpoint; converts CTC-only, --language"
  echo "          selects the per-language token range at run time. The"
  echo "          ai4bharat/indic-conformer-600m-multilingual HF repo ships"
  echo "          only ONNX exports, so this fetches AI4Bharat's .nemo.)"
  fetch "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo" \
        "$NEMO_DIR/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"

  hr
  echo "== nemo: parakeet-tdt_ctc-110m (small TDT+CTC hybrid, ~440 MiB)"
  echo "         (forward-looking: not yet wired into the C++ Engine; cached"
  echo "          for the planned hybrid TDT+CTC port.)"
  fetch "https://huggingface.co/nvidia/parakeet-tdt_ctc-110m/resolve/main/parakeet-tdt_ctc-110m.nemo" \
        "$NEMO_DIR/parakeet-tdt_ctc-110m.nemo"

  hr
  echo "== nemo: diar_sortformer_4spk-v1 (4-speaker diarization, offline, ~490 MiB)"
  fetch "https://huggingface.co/nvidia/diar_sortformer_4spk-v1/resolve/main/diar_sortformer_4spk-v1.nemo" \
        "$NEMO_DIR/diar_sortformer_4spk-v1.nemo"

  hr
  echo "== nemo: diar_streaming_sortformer_4spk-v2 (4-speaker, streaming-trained, ~470 MiB)"
  fetch "https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2/resolve/main/diar_streaming_sortformer_4spk-v2.nemo" \
        "$NEMO_DIR/diar_streaming_sortformer_4spk-v2.nemo"

  hr
  echo "== nemo: diar_streaming_sortformer_4spk-v2.1 (4-speaker, streaming + AOSC fine-tune, ~470 MiB)"
  fetch "https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1/resolve/main/diar_streaming_sortformer_4spk-v2.1.nemo" \
        "$NEMO_DIR/diar_streaming_sortformer_4spk-v2.1.nemo"
fi

if [[ "${1:-all}" != "tdt" && "${1:-all}" != "transcribepsy" ]]; then
  hr
  echo "== nemo: parakeet_realtime_eou_120m-v1 (EOU streaming, FastConformer-RNNT 120M, ~440 MiB)"
  echo "         (cache-aware streaming with att_context_size=[70,1] + <EOU>"
  echo "          end-of-utterance token; English only.)"
  fetch "https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1/resolve/main/parakeet_realtime_eou_120m-v1.nemo" \
        "$NEMO_DIR/parakeet_realtime_eou_120m-v1.nemo"
fi

hr
echo "done. Cached checkpoints:"
echo
ls -lh "$NEMO_DIR" | awk '/\.(nemo|gguf)$/ {print "  " $9, $5}' || true
