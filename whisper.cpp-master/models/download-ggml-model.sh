#!/bin/sh

# This script downloads Whisper model files that have already been converted to ggml format.
# This way you don't have to convert them yourself.

#src="https://ggml.ggerganov.com"
#pfx="ggml-model-whisper"

src="https://huggingface.co/ggerganov/whisper.cpp"
pfx="resolve/main/ggml"

BOLD="\033[1m"
RESET='\033[0m'

# get the path of this script
get_script_path() {
    if [ -x "$(command -v realpath)" ]; then
        dirname "$(realpath "$0")"
    else
        _ret="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P)"
        echo "$_ret"
    fi
}

script_path="$(get_script_path)"

# Check if the script is inside a /bin/ directory
case "$script_path" in
    */bin) default_download_path="$PWD" ;;  # Use current directory as default download path if in /bin/
    *) default_download_path="$script_path" ;;  # Otherwise, use script directory
esac

models_path="${2:-$default_download_path}"

# Whisper models
models="tiny
tiny.en
tiny-q5_1
tiny.en-q5_1
tiny-q8_0
base
base.en
base-q5_1
base.en-q5_1
base-q8_0
small
small.en
small.en-tdrz
small-q5_1
small.en-q5_1
small-q8_0
medium
medium.en
medium-q5_0
medium.en-q5_0
medium-q8_0
large-v1
large-v2
large-v2-q5_0
large-v2-q8_0
large-v3
large-v3-q5_0
large-v3-turbo
large-v3-turbo-q5_0
large-v3-turbo-q8_0"

# list available models
list_models() {
    printf "\n"
    printf "Available models:"
    model_class=""
    for model in $models; do
        this_model_class="${model%%[.-]*}"
        if [ "$this_model_class" != "$model_class" ]; then
            printf "\n "
            model_class=$this_model_class
        fi
        printf " %s" "$model"
    done
    printf "\n\n"
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    printf "Usage: %s <model> [models_path]\n" "$0"
    list_models
    printf "___________________________________________________________\n"
    printf "${BOLD}.en${RESET} = english-only ${BOLD}-q5_[01]${RESET} = quantized ${BOLD}-tdrz${RESET} = tinydiarize\n"

    exit 1
fi

model=$1

if ! echo "$models" | grep -q -w "$model"; then
    printf "Invalid model: %s\n" "$model"
    list_models

    exit 1
fi

# check if model contains `tdrz` and update the src and pfx accordingly
if echo "$model" | grep -q "tdrz"; then
    src="https://huggingface.co/akashmjn/tinydiarize-whisper.cpp"
    pfx="resolve/main/ggml"
fi

echo "$model" | grep -q '^"tdrz"*$'

# download ggml model

printf "Downloading ggml model %s from '%s' ...\n" "$model" "$src"

cd "$models_path" || exit

if [ -f "ggml-$model.bin" ]; then
    printf "Model %s already exists. Skipping download.\n" "$model"
    exit 0
fi

if [ -x "$(command -v wget2)" ]; then
    wget2 --no-config --progress bar -O ggml-"$model".bin $src/$pfx-"$model".bin
elif [ -x "$(command -v wget)" ]; then
    wget --no-config --quiet --show-progress -O ggml-"$model".bin $src/$pfx-"$model".bin
elif [ -x "$(command -v curl)" ]; then
    curl -L --output ggml-"$model".bin $src/$pfx-"$model".bin
else
    printf "Either wget or curl is required to download models.\n"
    exit 1
fi

if [ $? -ne 0 ]; then
    printf "Failed to download ggml model %s \n" "$model"
    printf "Please try again later or download the original Whisper model files and convert them yourself.\n"
    exit 1
fi

# Check if 'whisper-cli' is available in the system PATH
if command -v whisper-cli >/dev/null 2>&1; then
    # If found, use 'whisper-cli' (relying on PATH resolution)
    whisper_cmd="whisper-cli"
else
    # If not found, use the local build version
    whisper_cmd="./build/bin/whisper-cli"
fi

printf "Done! Model '%s' saved in '%s/ggml-%s.bin'\n" "$model" "$models_path" "$model"
printf "You can now use it like this:\n\n"
printf "  $ %s -m %s/ggml-%s.bin -f samples/jfk.wav\n" "$whisper_cmd" "$models_path" "$model"
printf "\n"
