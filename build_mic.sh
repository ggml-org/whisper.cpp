#!/bin/bash
# Build script for examples/mic/whisper-mic.cpp
set -e

cd "$(dirname "$0")"

mkdir -p build_mic
cd build_mic

cmake ../examples/mic
cmake --build . -j$(nproc)
