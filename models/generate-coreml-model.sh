#!/bin/sh

set -e

# Usage: ./generate-coreml-model.sh [--decoder] <model-name>
with_decoder=false
if [ "$1" = "--decoder" ]; then
    with_decoder=true
    shift
fi

if [ $# -eq 0 ]; then
    echo "No model name supplied"
    echo "Usage for Whisper models: ./generate-coreml-model.sh [--decoder] <model-name>"
    echo "Usage for HuggingFace models: ./generate-coreml-model.sh -h5 <model-name> <model-path>"
    exit 1
elif [ "$1" = "-h5" ] && [ $# != 3 ]; then
    echo "No model name and model path supplied for a HuggingFace model"
    echo "Usage for HuggingFace models: ./generate-coreml-model.sh -h5 <model-name> <model-path>"
    exit 1
elif [ "$1" = "-h5" ] && [ "$with_decoder" = "true" ]; then
    echo "Core ML decoder export is only supported for Whisper models"
    exit 1
fi

mname="$1"

wd=$(dirname "$0")
cd "$wd/../" || exit

if [ "$with_decoder" = "true" ]; then
    rm -rf models/coreml-decoder-"${mname}"-cross-input-no-write-s*.mlpackage
    rm -rf models/ggml-"${mname}"-decoder-cross-input-no-write-s*.mlmodelc
fi

if [ "$mname" = "-h5" ]; then
    mname="$2"
    mpath="$3"
    echo "$mpath"
    python3 models/convert-h5-to-coreml.py --model-name "$mname" --model-path "$mpath" --encoder-only True
elif [ "$with_decoder" = "true" ]; then
    python3 models/convert-whisper-to-coreml.py --model "$mname" --optimize-ane True --decoder-npu
else
    python3 models/convert-whisper-to-coreml.py --model "$mname" --encoder-only True --optimize-ane True
fi

xcrun coremlc compile models/coreml-encoder-"${mname}".mlpackage models/
rm -rf models/ggml-"${mname}"-encoder.mlmodelc
mv -v models/coreml-encoder-"${mname}".mlmodelc models/ggml-"${mname}"-encoder.mlmodelc

if [ "$with_decoder" = "true" ]; then
    found_decoder=false
    for pkg in models/coreml-decoder-"${mname}"-cross-input-no-write-s*.mlpackage; do
        if [ ! -e "$pkg" ]; then
            continue
        fi

        found_decoder=true
        base=$(basename "$pkg" .mlpackage)
        suffix=${base#coreml-decoder-${mname}}
        xcrun coremlc compile "$pkg" models/
        rm -rf "models/ggml-${mname}-decoder${suffix}.mlmodelc"
        mv -v "models/${base}.mlmodelc" "models/ggml-${mname}-decoder${suffix}.mlmodelc"
    done

    if [ "$found_decoder" = "false" ]; then
        echo "No Core ML decoder shard packages were generated"
        exit 1
    fi
fi
