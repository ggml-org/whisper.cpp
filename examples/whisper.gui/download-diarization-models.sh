#!/usr/bin/env bash
#
# Download the speaker-diarization models used by diarize.py from the
# sherpa-onnx GitHub releases (no HuggingFace needed). Run once.
#
#   ./download-diarization-models.sh [TARGET_DIR]   (default: models)
#
set -e

DIR="${1:-models}"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"
# note: the embedding release tag really is spelled "recongition" upstream
EMB_URL="$BASE/speaker-recongition-models/nemo_en_titanet_small.onnx"
SEG_URL="$BASE/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"

mkdir -p "$DIR"

echo "Downloading speaker-embedding model (NeMo TitaNet small, English) ..."
curl -L --fail -o "$DIR/nemo_en_titanet_small.onnx" "$EMB_URL"

echo "Downloading pyannote segmentation model ..."
curl -L --fail -o "$DIR/_seg.tar.bz2" "$SEG_URL"
tar xjf "$DIR/_seg.tar.bz2" -C "$DIR"
rm -f "$DIR/_seg.tar.bz2"

echo
echo "Done. Models are in '$DIR/':"
echo "  $DIR/nemo_en_titanet_small.onnx"
echo "  $DIR/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
echo
echo "Now run, e.g.:  python3 diarize.py audio.wav --json audio.json --speakers 3"
