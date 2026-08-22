#!/bin/sh

# This script downloads prebuilt VitisAI encoder cache files for Whisper models.
# The cache file is saved next to the ggml model file and follows the loader
# convention: ggml-<model>-encoder-vitisai.rai

src="https://huggingface.co"
collection_api="https://huggingface.co/api/collections/amd/ryzen-ai-whisper-npu-optimized-onnx-models"
collection_url="https://huggingface.co/collections/amd/ryzen-ai-whisper-npu-optimized-onnx-models"

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

find_python() {
    if command -v python3 >/dev/null 2>&1; then
        printf "%s\n" "python3"
    elif command -v python >/dev/null 2>&1; then
        printf "%s\n" "python"
    else
        return 1
    fi
}

script_path="$(get_script_path)"

# Check if the script is inside a /bin/ directory
case "$script_path" in
    */bin) default_download_path="$PWD" ;;  # Use current directory as default download path if in /bin/
    *) default_download_path="$script_path" ;;  # Otherwise, use script directory
esac

models_path="${2:-$default_download_path}"

discover_models() {
    python_cmd="$(find_python)" || {
        printf "Python is required to query available VitisAI caches from Hugging Face.\n" >&2
        return 1
    }

    "$python_cmd" - "$src" "$collection_api" <<'PY'
import json
import os
import re
import sys
import urllib.parse
import urllib.request

src = sys.argv[1].rstrip("/")
collection_api = sys.argv[2]
headers = {}
token = os.environ.get("HF_TOKEN")
if token:
    headers["Authorization"] = "Bearer " + token

def load_json(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        return json.load(response)

def normalize_model_name(name):
    # HF currently publishes ggml-small-en-encoder-vitisai.rai, while the
    # matching ggml model is ggml-small.en.bin.
    if name.endswith("-en"):
        return name[:-3] + ".en"
    return name

collection = load_json(collection_api)
rows = []
seen = set()

for item in collection.get("items", []):
    if item.get("type") != "model":
        continue

    repo = item.get("id")
    if not repo:
        continue

    model_info = load_json(src + "/api/models/" + repo)
    for sibling in model_info.get("siblings", []):
        filename = sibling.get("rfilename", "")
        match = re.match(r"^ggml-(.+)-encoder-vitisai\.rai$", filename)
        if not match:
            continue

        raw_name = match.group(1)
        model_name = normalize_model_name(raw_name)
        if model_name in seen:
            continue
        seen.add(model_name)

        destination = "ggml-%s-encoder-vitisai.rai" % model_name
        url = "%s/%s/resolve/main/%s" % (src, repo, urllib.parse.quote(filename))
        rows.append((model_name, raw_name, repo, filename, destination, url))

order = {
    "tiny": 10,
    "tiny.en": 11,
    "base": 20,
    "base.en": 21,
    "small": 30,
    "small.en": 31,
    "medium": 40,
    "medium.en": 41,
    "large-v1": 50,
    "large-v2": 60,
    "large-v3": 70,
    "large-v3-turbo": 80,
}

for row in sorted(rows, key=lambda item: (order.get(item[0], 1000), item[0])):
    print("|".join(row))
PY
}

list_models() {
    models="$(discover_models)" || exit 1

    printf "\n"
    printf "Available VitisAI encoder caches from %s:\n" "$collection_url"
    printf "%s\n" "$models" | while IFS='|' read -r model raw repo _source _destination _url; do
        if [ "$model" = "$raw" ]; then
            printf "  %-18s %s\n" "$model" "$repo"
        else
            printf "  %-18s %s (source name: %s)\n" "$model" "$repo" "$raw"
        fi
    done
    printf "\n"
}

usage() {
    printf "Usage: %s --list\n" "$0"
    printf "       %s <model> [models_path]\n" "$0"
    printf "\n"
    printf "Downloads ggml-<model>-encoder-vitisai.rai next to ggml-<model>.bin.\n"
    printf "Use the same model name as %s/download-ggml-model.sh.\n" "$script_path"
    printf "\n"
}

if [ "$#" -eq 1 ] && { [ "$1" = "--list" ] || [ "$1" = "-l" ] || [ "$1" = "list" ]; }; then
    list_models
    exit 0
fi

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
    list_models
    printf "___________________________________________________________\n"
    printf "Example: %s ${BOLD}small${RESET} %s\n" "$0" "$default_download_path"
    exit 1
fi

model=$1
models="$(discover_models)" || exit 1

match="$(printf "%s\n" "$models" | awk -F '|' -v model="$model" '$1 == model || $2 == model { print; exit }')"
if [ -z "$match" ]; then
    printf "Invalid model: %s\n" "$model"
    printf "%s\n" "$models" | while IFS='|' read -r available _raw _repo _source _destination _url; do
        printf "  %s\n" "$available"
    done
    exit 1
fi

IFS='|' read -r model raw_name repo source_file destination_file download_url <<EOF
$match
EOF

printf "Downloading VitisAI encoder cache %s from '%s' ...\n" "$model" "$repo"

mkdir -p "$models_path" || exit
cd "$models_path" || exit

if [ -f "$destination_file" ]; then
    printf "VitisAI encoder cache %s already exists. Skipping download.\n" "$destination_file"
    exit 0
fi

if [ -x "$(command -v wget2)" ]; then
    wget2 --no-config --progress bar -O "$destination_file" ${HF_TOKEN:+--header "Authorization: Bearer $HF_TOKEN"} "$download_url"
elif [ -x "$(command -v curl)" ]; then
    curl -L --fail \
         --retry 5 \
         --retry-delay 5 \
         --retry-all-errors \
         --retry-connrefused \
         ${HF_TOKEN:+--header "Authorization: Bearer $HF_TOKEN"} \
         --output "$destination_file" "$download_url"
elif [ -x "$(command -v wget)" ]; then
    wget --no-config --quiet --show-progress ${HF_TOKEN:+--header "Authorization: Bearer $HF_TOKEN"} -O "$destination_file" "$download_url"
else
    printf "Either wget2, curl, or wget is required to download VitisAI encoder caches.\n"
    exit 1
fi

if [ $? -ne 0 ]; then
    printf "Failed to download VitisAI encoder cache %s from %s\n" "$model" "$download_url"
    rm -f "$destination_file"
    exit 1
fi

# Check if 'whisper-cli' is available in the system PATH
if command -v whisper-cli >/dev/null 2>&1; then
    whisper_cmd="whisper-cli"
else
    whisper_cmd="./build/bin/whisper-cli"
fi

printf "Done! VitisAI encoder cache '%s' saved in '%s/%s'\n" "$model" "$models_path" "$destination_file"
if [ "$raw_name" != "$model" ]; then
    printf "Source cache '%s' was renamed to match ggml model name '%s'.\n" "$source_file" "$model"
fi
printf "Use it with the matching ggml model:\n\n"
printf "  $ %s/download-ggml-model.sh %s %s\n" "$script_path" "$model" "$models_path"
printf "  $ %s -m %s/ggml-%s.bin -f samples/jfk.wav\n" "$whisper_cmd" "$models_path" "$model"
printf "\n"
