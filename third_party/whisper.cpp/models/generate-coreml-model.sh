#!/bin/sh

set -e

# Usage: ./generate-coreml-model.sh <model-name>
if [ $# -eq 0 ]; then
  echo "No model name supplied"
  echo "Usage for Whisper models: ./generate-coreml-model.sh <model-name>"
  echo "Usage for HuggingFace models: ./generate-coreml-model.sh -h5 <model-name> <model-path>"
  exit 1
elif [ "$1" = "-h5" ] && [ $# != 3 ]; then
  echo "No model name and model path supplied for a HuggingFace model"
  echo "Usage for HuggingFace models: ./generate-coreml-model.sh -h5 <model-name> <model-path>"
  exit 1
fi

mname="$1"

wd=$(dirname "$0")
cd "$wd/../" || exit

if [ "$mname" = "-h5" ]; then
  mname="$2"
  mpath="$3"
  echo "$mpath"
  python3 models/convert-h5-to-coreml.py --model-name "$mname" --model-path "$mpath" --encoder-only True
else
  python3 models/convert-whisper-to-coreml.py --model "$mname" --encoder-only True --optimize-ane True
fi

xcrun coremlc compile models/coreml-encoder-"${mname}".mlpackage models/
rm -rf models/ggml-"${mname}"-encoder.mlmodelc
mv -v models/coreml-encoder-"${mname}".mlmodelc models/ggml-"${mname}"-encoder.mlmodelc

# TODO: decoder (sometime in the future maybe)
#xcrun coremlc compile models/whisper-decoder-${mname}.mlpackage models/
#rm -rf models/ggml-${mname}-decoder.mlmodelc
#mv -v models/coreml_decoder_${mname}.mlmodelc models/ggml-${mname}-decoder.mlmodelc
