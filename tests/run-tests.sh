#!/bin/bash

# This scripts run the selected model agains a collection of audio files from the web.
# It downloads, converts and transcribes each file and then compares the result with the expected reference
# transcription. The comparison is performed using git's diff command and shows the differences at the character level.
# It can be used to quickly verify that the model is working as expected across a wide range of audio files.
# I.e. like an integration test. The verification is done by visual inspection of the diff output.
#
# The reference data can be for example generated using the original OpenAI Whisper implementation, or entered manually.
#
# Feel free to suggest extra audio files to add to the list.
# Make sure they are between 1-3 minutes long since we don't want to make the test too slow.
#
# Usage:
#
#   ./tests/run-tests.sh <model_name> [threads]
#

cd `dirname $0`

# Whisper models
models=( "tiny.en" "tiny" "base.en" "base" "small.en" "small" "medium.en" "medium" "large-v1" "large-v2" "large-v3" "large-v3-turbo" )

# list available models
function list_models {
    printf "\n"
    printf "  Available models:"
    for model in "${models[@]}"; do
        printf " $model"
    done
    printf "\n\n"
}

if [ $# -eq 0 ]; then
    printf "Usage: $0 [model] [threads]\n\n"
    printf "No model specified. Aborting\n"
    list_models
    exit 1
fi

model=$1
main="../build/bin/whisper-cli"

threads=""
if [ $# -eq 2 ]; then
    threads="-t $2"
fi

if [ ! -f ../models/ggml-$model.bin ]; then
    printf "Model $model not found. Aborting\n"
    list_models
    exit 1
fi

if [ ! -f $main ]; then
    printf "Executable $main not found. Aborting\n"
    exit 1
fi

# add various audio files for testing purposes here
# the order of the files is important so don't change the existing order
# when adding new files, make sure to add the expected "ref.txt" file with the correct transcript
urls_en=(
    "https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg"
    "https://upload.wikimedia.org/wikipedia/en/d/d4/En.henryfphillips.ogg"
    "https://cdn.openai.com/whisper/draft-20220913a/micro-machines.wav"
)

urls_es=(
    "https://upload.wikimedia.org/wikipedia/commons/c/c1/La_contaminacion_del_agua.ogg"
)

urls_it=(
)

urls_pt=(
)

urls_de=(
)

urls_jp=(
)

urls_ru=(
)

function run_lang() {
    lang=$1
    shift
    urls=("$@")

    i=0
    for url in "${urls[@]}"; do
        echo "- [$lang] Processing '$url' ..."

        ext="${url##*.}"
        fname_src="$lang-${i}.${ext}"
        fname_dst="$lang-${i}-16khz.wav"

        if [ ! -f $fname_src ]; then
            wget --quiet --show-progress -O $fname_src $url
        fi

        if [ ! -f $fname_dst ]; then
            ffmpeg -loglevel -0 -y -i $fname_src -ar 16000 -ac 1 -c:a pcm_s16le $fname_dst
            if [ $? -ne 0 ]; then
                echo "Error: ffmpeg failed to convert $fname_src to $fname_dst"
                exit 1
            fi
        fi

        $main -m ../models/ggml-$model.bin $threads -f $fname_dst -l $lang -otxt 2> /dev/null

        git diff --no-index --word-diff=color --word-diff-regex=. $lang-$i-ref.txt $fname_dst.txt

        i=$(($i+1))
    done
}

run_lang "en" "${urls_en[@]}"

if [[ $model != *.en* ]]; then
    run_lang "es" "${urls_es[@]}"
    run_lang "it" "${urls_it[@]}"
    run_lang "pt" "${urls_pt[@]}"
    run_lang "de" "${urls_de[@]}"
    run_lang "jp" "${urls_jp[@]}"
    run_lang "ru" "${urls_ru[@]}"
fi
