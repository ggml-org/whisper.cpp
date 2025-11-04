#!/bin/bash
#
# Transcribe twitch.tv livestream by feeding audio input to whisper.cpp at regular intervals
# Thanks to @keyehzy
# ref: https://github.com/ggml-org/whisper.cpp/issues/209
#
# The script currently depends on the third-party tool "streamlink"
# On Mac OS, you can install it via "brew install streamlink"
#

set -eo pipefail

step=10
model=base.en
threads=4

help()
{
    echo "Example program for captioning a livestream from twitch.tv."
    echo
    echo "Usage: ./twitch.sh -s [step] -m [model] -t [threads] [url]"
    echo "options:"
    echo "-s       Step in seconds (default is $step)."
    echo "-m       Choose model, options are: 'tiny.en' 'tiny' 'base.en' 'base' 'small.en' 'small' 'medium.en' 'medium' 'large-v1' 'large-v2' 'large-v3' 'large-v3-turbo' (default is '$model')."
    echo "-t       Number of threads to use."
    echo "-h       Print this help page."
    echo
}

check_requirements()
{
    if ! command -v ./build/bin/whisper-cli &>/dev/null; then
        echo "whisper.cpp main executable is required (make)"
        exit 1
    fi

    if ! command -v streamlink &>/dev/null; then
        echo "streamlink is required (https://streamlink.github.io)"
        exit 1
    fi

    if ! command -v ffmpeg &>/dev/null; then
        echo "ffmpeg is required (https://ffmpeg.org)"
        exit 1
    fi
}

check_requirements

while getopts ":s:m:t:h" option; do
    case $option in
	s)
            step=$OPTARG;;
	m)
            model=$OPTARG;;
	t)
	    threads=$OPTARG;;
	h)
            help
            exit;;
	\?)
	    help
	    exit;;
    esac
done

url=${@:$OPTIND:1}

if [ -z $url ]; then
    help
    exit
fi

echo "Piping from streamlink url=$url model=$model step=$step threads=$threads"
streamlink $url best -O 2>/dev/null | ffmpeg -loglevel quiet -i - -y -probesize 32 -y -ar 16000 -ac 1 -acodec pcm_s16le /tmp/whisper-live0.wav &

if [ $? -ne 0 ]; then
    printf "error: ffmpeg failed\n"
    exit 1
fi

echo "Buffering stream... (this should take $step seconds)"
sleep $(($step))

set +e

echo "Starting..."

i=0
SECONDS=0
while true
do
    err=1
    while [ $err -ne 0 ]; do
        if [ $i -gt 0 ]; then
            ffmpeg -loglevel quiet -v error -noaccurate_seek -i /tmp/whisper-live0.wav -y -ss $(($i*$step-1)).5 -t $step -c copy /tmp/whisper-live.wav 2> /tmp/whisper-live.err
        else
            ffmpeg -loglevel quiet -v error -noaccurate_seek -i /tmp/whisper-live0.wav -y -ss $(($i*$step)) -t $step -c copy /tmp/whisper-live.wav 2> /tmp/whisper-live.err
        fi
        err=$(cat /tmp/whisper-live.err | wc -l)
    done

    ./build/bin/whisper-cli -t $threads -m ./models/ggml-$model.bin -f /tmp/whisper-live.wav --no-timestamps -otxt 2> /tmp/whispererr | tail -n 1

    while [ $SECONDS -lt $((($i+1)*$step)) ]; do
        sleep 1
    done
    ((i=i+1))
done
