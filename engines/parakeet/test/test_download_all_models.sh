#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_SCRIPT="$SCRIPT_DIR/../scripts/download-all-models.sh"
WORK_DIR="$(mktemp -d)"
STUB_DIR="$WORK_DIR/bin"
DOWNLOAD_LOG="$WORK_DIR/downloads.log"
EXPECTED_URL="https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"
EXPECTED_MODEL="$WORK_DIR/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"
EOU_URL="https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1/resolve/main/parakeet_realtime_eou_120m-v1.nemo"
EOU_MODEL="$WORK_DIR/models/parakeet_realtime_eou_120m-v1.nemo"
TRANSCRIBEPSY_URL="https://huggingface.co/qvac/TranscribePsy-250M-EU-sub-gguf/resolve/main/transcribepsy-250m-eu-sub.gguf"
TRANSCRIBEPSY_MODEL="$WORK_DIR/models/transcribepsy-250m-eu-sub.gguf"
TRANSCRIBEPSY_COREML_GGUF="$WORK_DIR/models/transcribepsy-250m-eu-sub-coreml.gguf"
TRANSCRIBEPSY_DECODER_WEIGHTS="$WORK_DIR/models/transcribepsy-250m-eu-sub-coreml-decoder.mlmodelc/weights/weight.bin"
TRANSCRIBEPSY_ENCODER_METADATA="$WORK_DIR/models/transcribepsy-250m-eu-sub-coreml-encoder.mlmodelc/metadata.json"
# The GGUF, the encoder-less GGUF and five files per compiled Core ML bundle.
TRANSCRIBEPSY_DOWNLOADS=12

cleanup() {
  rm -rf "$WORK_DIR"
}

create_curl_stub() {
  mkdir -p "$STUB_DIR"
  cat > "$STUB_DIR/curl" <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

parse_arguments() {
  while (( $# > 0 )); do
    case "$1" in
      -o)
        destination="$2"
        shift 2
        ;;
      http*)
        url="$1"
        shift
        ;;
      *)
        shift
        ;;
    esac
  done
}

destination=
url=
parse_arguments "$@"
printf '%s\n' "$url" >> "$DOWNLOAD_LOG"
: > "$destination"
EOF
  chmod +x "$STUB_DIR/curl"
}

prepare_script() {
  mkdir -p "$WORK_DIR/scripts"
  cp "$SOURCE_SCRIPT" "$WORK_DIR/scripts/download-all-models.sh"
}

assert_download() {
  grep -Fxq "$EXPECTED_URL" "$DOWNLOAD_LOG"
  test -f "$EXPECTED_MODEL"
}

assert_eou_only_download() {
  : > "$DOWNLOAD_LOG"
  rm -rf "$WORK_DIR/models"
  PATH="$STUB_DIR:$PATH" DOWNLOAD_LOG="$DOWNLOAD_LOG" \
    bash "$WORK_DIR/scripts/download-all-models.sh" eou > /dev/null
  grep -Fxq "$EOU_URL" "$DOWNLOAD_LOG"
  test -f "$EOU_MODEL"
  test "$(wc -l < "$DOWNLOAD_LOG" | tr -d ' ')" = "1"
}

assert_transcribepsy_only_download() {
  : > "$DOWNLOAD_LOG"
  rm -rf "$WORK_DIR/models"
  PATH="$STUB_DIR:$PATH" DOWNLOAD_LOG="$DOWNLOAD_LOG" \
    bash "$WORK_DIR/scripts/download-all-models.sh" transcribepsy > /dev/null
  grep -Fxq "$TRANSCRIBEPSY_URL" "$DOWNLOAD_LOG"
  test -f "$TRANSCRIBEPSY_MODEL"
  test -f "$TRANSCRIBEPSY_COREML_GGUF"
  test -f "$TRANSCRIBEPSY_DECODER_WEIGHTS"
  test -f "$TRANSCRIBEPSY_ENCODER_METADATA"
  test "$(wc -l < "$DOWNLOAD_LOG" | tr -d ' ')" = "$TRANSCRIBEPSY_DOWNLOADS"
}

trap cleanup EXIT
create_curl_stub
prepare_script
PATH="$STUB_DIR:$PATH" DOWNLOAD_LOG="$DOWNLOAD_LOG" \
  bash "$WORK_DIR/scripts/download-all-models.sh" > /dev/null
assert_download
grep -Fxq "$TRANSCRIBEPSY_URL" "$DOWNLOAD_LOG"
assert_eou_only_download
assert_transcribepsy_only_download
