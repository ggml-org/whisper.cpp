#!/bin/bash

# Usage:
#  speak <voice_id> <textfile>

function installed() { command -v $1 >/dev/null 2>&1; }

if installed espeak; then
  espeak -v en-us+m$1 -s 225 -p 50 -a 200 -g 5 -k 5 -f $2

elif installed piper && installed aplay; then
  cat $2 | piper --model ~/en_US-lessac-medium.onnx --output-raw | aplay -q -r 22050 -f S16_LE -t raw -

# for Mac
elif installed say; then
  say -f $2

# Eleven Labs
elif installed python3 && \
  python3 -c 'import importlib.util; exit(not importlib.util.find_spec("elevenlabs"))' && \
  installed ffplay; then
    # It's possible to use the API for free with limited number of characters.
    # To increase this limit register to https://beta.elevenlabs.io to get an api key
    # and paste it after 'ELEVEN_API_KEY='
    # Keep the line commented to use the free version without api key
    #export ELEVEN_API_KEY=your_api_key
    wd=$(dirname $0)
    script=$wd/eleven-labs.py
    python3 $script -q -p -v $1 $2 >/dev/null 2>&1

    # Uncomment to keep the audio file
    #python3 $script -q -s ./audio.mp3 -v $1 $2 >/dev/null 2>&1
    #ffplay -autoexit -nodisp -loglevel quiet -hide_banner -i ./audio.mp3 >/dev/null 2>&1

else
  echo 'Install espeak ("brew install espeak" or "apt-get install espeak"),'
  echo 'piper ("pip install piper-tts" or https://github.com/rhasspy/piper) with aplay,'
  echo 'or elevenlabs ("pip install elevenlabs") with ffplay.'
  echo '(export ELEVEN_API_KEY if you have an api key from https://beta.elevenlabs.io)'
fi
