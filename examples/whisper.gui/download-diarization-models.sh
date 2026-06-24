#!/usr/bin/env bash
#
# Download speaker-diarization models for diarize.py from the sherpa-onnx
# GitHub releases (no HuggingFace needed). Run once.
#
#   ./download-diarization-models.sh [TARGET_DIR] [EMBEDDING_MODEL]
#
#   TARGET_DIR       where to put the models (default: models)
#   EMBEDDING_MODEL  one of (default: titanet-small):
#                      titanet-small   ~38 MB  fast, English (default)
#                      titanet-large   ~98 MB  stronger, English
#                      campplus        ~28 MB  3D-Speaker CAM++, English
#                      resnet34        ~26 MB  wespeaker ResNet34, English
#                      resnet152       ~80 MB  wespeaker ResNet152, strongest
#
# For hard audio (similar voices, background noise) try a stronger model and
# pass it to diarize.py with --emb-model.
#
set -e

DIR="${1:-models}"
EMB="${2:-titanet-small}"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEG_URL="$BASE/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"

case "$EMB" in
    titanet-small) EMB_FILE="nemo_en_titanet_small.onnx" ;;
    titanet-large) EMB_FILE="nemo_en_titanet_large.onnx" ;;
    campplus)      EMB_FILE="3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx" ;;
    resnet34)      EMB_FILE="wespeaker_en_voxceleb_resnet34_LM.onnx" ;;
    resnet152)     EMB_FILE="wespeaker_en_voxceleb_resnet152_LM.onnx" ;;
    *) echo "unknown embedding model '$EMB'"; echo \
       "choices: titanet-small titanet-large campplus resnet34 resnet152"; exit 1 ;;
esac
# note: the embedding release tag really is spelled "recongition" upstream
EMB_URL="$BASE/speaker-recongition-models/$EMB_FILE"

mkdir -p "$DIR"

echo "Downloading embedding model ($EMB): $EMB_FILE ..."
curl -L --fail -o "$DIR/$EMB_FILE" "$EMB_URL"

if [ ! -f "$DIR/sherpa-onnx-pyannote-segmentation-3-0/model.onnx" ]; then
    echo "Downloading pyannote segmentation model ..."
    curl -L --fail -o "$DIR/_seg.tar.bz2" "$SEG_URL"
    tar xjf "$DIR/_seg.tar.bz2" -C "$DIR"
    rm -f "$DIR/_seg.tar.bz2"
fi

echo
echo "Done. Models are in '$DIR/':"
echo "  embedding:    $DIR/$EMB_FILE"
echo "  segmentation: $DIR/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
echo
if [ "$EMB" = "titanet-small" ]; then
    echo "Run:  python3 diarize.py audio.wav --json audio.json --speakers N"
else
    echo "Run:  python3 diarize.py audio.wav --json audio.json --speakers N --emb-model $DIR/$EMB_FILE"
fi
