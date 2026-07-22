#!/bin/bash

# Simple tool to record audio from the microphone and generate a karaoke video
# Usage:
#
#  cd whisper.cpp
#  make
#
#  ./examples/generate-karaoke.sh [model] [step_ms]
#
# Press Ctrl+C to stop recording
#

executable="./build/bin/whisper-cli"
model="base.en"
model_path="models/ggml-$model.bin"

# require sox and ffmpeg to be installed
if ! command -v sox &> /dev/null
then
    echo "sox could not be found"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg could not be found"
    exit 2
fi

if [ ! -f "$executable" ]; then
    echo "'$executable' does not exist. Please build it first."
    exit 3
fi

if [ ! -f "$model_path" ]; then
    echo "'$model_path' does not exist. Please download it first."
    exit 4
fi

# record some raw audio
sox -d rec.wav

# run Whisper
echo "Processing ..."
${executable} -m models/ggml-base.en.bin rec.wav -owts > /dev/null 2>&1

# generate Karaoke video
echo "Generating video ..."
source rec.wav.wts > /dev/null 2>&1

# play the video
echo "Playing ./rec16.wav.mp4 ..."
ffplay -loglevel 0 -autoexit ./rec.wav.mp4

echo "Done"
exit 0
