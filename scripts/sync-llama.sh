#!/bin/bash
# note: run from the whisper.cpp root directory
set -euo pipefail

# read the current llama.cpp version from the neighboring repo
LLAMA_VERSION=$(sed -nE 's/^set\(LLAMA_VERSION_(MAJOR|MINOR|PATCH) ([0-9]+)\)$/\2/p' ../llama.cpp/CMakeLists.txt | paste -sd. -)

if [ -z "${LLAMA_VERSION}" ]; then
    echo "error: could not read llama.cpp version from ../llama.cpp/CMakeLists.txt" >&2
    exit 1
fi

# update the version number in examples/talk-llama/CMakeLists.txt
sed -i.bak \
    -E -e "s/(find_package[(]llama )[0-9]+\.[0-9]+\.[0-9]+/\1${LLAMA_VERSION}/" \
       -e "s/(set[(]LLAMA_TAG \"v)[0-9]+\.[0-9]+\.[0-9]+/\1${LLAMA_VERSION}/" \
    examples/talk-llama/CMakeLists.txt
rm examples/talk-llama/CMakeLists.txt.bak

echo "synced llama.cpp version: ${LLAMA_VERSION}"
