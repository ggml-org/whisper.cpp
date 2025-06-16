#!/usr/bin/env bash
# shellcheck disable=2086

# MIT License

# Copyright (c) 2022 Daniils Petrovs
# Copyright (c) 2023 Jennifer Capasso

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Small shell script to more easily automatically download and transcribe live stream VODs.
# This uses YT-DLP, ffmpeg and the CPP version of Whisper: https://github.com/ggml-org/whisper.cpp
# Use `./examples/yt-wsp.sh help` to print help info.
#
# Sample usage:
#
#   git clone https://github.com/ggml-org/whisper.cpp
#   cd whisper.cpp
#   make
#   ./examples/yt-wsp.sh https://www.youtube.com/watch?v=1234567890
#

set -Eeuo pipefail

# get script file location
SCRIPT_PATH="$(realpath -e ${BASH_SOURCE[0]})";
SCRIPT_DIR="${SCRIPT_PATH%/*}"

################################################################################
# Documentation on downloading models can be found in the whisper.cpp repo:
# https://github.com/ggml-org/whisper.cpp/#usage
#
# note: unless a multilingual model is specified, WHISPER_LANG will be ignored
# and the video will be transcribed as if the audio were in the English language
################################################################################
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/../models/ggml-base.en.bin}"

################################################################################
# Where to find the whisper.cpp executable.  default to the examples directory
# which holds this script in source control
################################################################################
WHISPER_EXECUTABLE="${WHISPER_EXECUTABLE:-${SCRIPT_DIR}/../build/bin/whisper-cli}";

# Set to desired language to be translated into english
WHISPER_LANG="${WHISPER_LANG:-en}";

# Default to 4 threads (this was most performant on my 2020 M1 MBP)
WHISPER_THREAD_COUNT="${WHISPER_THREAD_COUNT:-4}";

msg() {
    echo >&2 -e "${1-}"
}

cleanup() {
    local -r clean_me="${1}";

    if [ -d "${clean_me}" ]; then
      msg "Cleaning up...";
      rm -rf "${clean_me}";
    else
      msg "'${clean_me}' does not appear to be a directory!";
      exit 1;
    fi;
}

print_help() {
    cat << 'EOF'
Usage:
  MODEL_PATH=<model> \
  WHISPER_EXECUTABLE=<whisper-cli> \
  WHISPER_LANG=en \
  WHISPER_THREAD_COUNT=<int> \
  ./examples/yt-wsp.sh <video_url>

Description:
  This script downloads a YouTube video, generates subtitles using Whisper, 
  and muxes them into an MP4 output file.

Output:
  An MP4 file with embedded subtitles will be produced in the working directory.
  The file will be named using the video title and ID.
  Example: 
    Input:  https://youtu.be/VYJtb2YXae8
    Output: Why_we_all_need_subtitles_now-VYJtb2YXae8-res.mp4

Requirements:
  - ffmpeg
  - yt-dlp
  - whisper.cpp

Environment Variables:
  MODEL_PATH            Path to the Whisper model (e.g., models/ggml-base.en.bin)
  WHISPER_EXECUTABLE    Path to the Whisper CLI executable
  WHISPER_LANG          Language code (e.g., 'en' for English)
  WHISPER_THREAD_COUNT  Number of CPU threads to use

Tip:
  The script has many configurable environment variables.
  Review the source code to explore all options.

EOF
}

check_requirements() {
    if ! command -v ffmpeg &>/dev/null; then
        echo "ffmpeg is required: https://ffmpeg.org";
        exit 1
    fi;

    if ! command -v yt-dlp &>/dev/null; then
        echo "yt-dlp is required: https://github.com/yt-dlp/yt-dlp";
        exit 1;
    fi;

    if ! command -v "${WHISPER_EXECUTABLE}" &>/dev/null; then
        echo "The C++ implementation of Whisper is required: https://github.com/ggml-org/whisper.cpp"
        echo "Sample usage:";
        echo "";
        echo "  git clone https://github.com/ggml-org/whisper.cpp";
        echo "  cd whisper.cpp";
        echo "  make";
        echo "  ./examples/yt-wsp.sh https://www.youtube.com/watch?v=1234567890";
        echo "";
        exit 1;
    fi;

}

if [[ "${#}" -lt 1 ]]; then
    print_help;
    exit 1;
fi

if [[ "${1##-*}" == "help" ]]; then
    print_help;
    exit 0;
fi

check_requirements;

################################################################################
# create a temporary directory to work in
# set the temp_dir and temp_filename variables
################################################################################
temp_dir="$(mktemp -d ${SCRIPT_DIR}/tmp.XXXXXX)";
temp_filename="${temp_dir}/yt-dlp-filename";

################################################################################
# for now we only take one argument
# TODO: a for loop
################################################################################
source_url="${1}"
title_name="";

msg "Downloading VOD...";

################################################################################
# Download the video, put the dynamic output filename into a variable.
# Optionally add --cookies-from-browser BROWSER[+KEYRING][:PROFILE][::CONTAINER]
# for videos only available to logged-in users.
################################################################################
yt-dlp \
    -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
    -o "${temp_dir}/%(title)s-%(id)s.vod.mp4" \
    --print-to-file "%(filename)s" "${temp_filename}" \
    --no-simulate \
    --no-write-auto-subs \
    --restrict-filenames \
    --embed-thumbnail \
    --embed-chapters \
    --xattrs \
    "${source_url}";

title_name="$(xargs basename -s .vod.mp4 < ${temp_filename})";

msg "Extracting audio and resampling...";

ffmpeg -i "${temp_dir}/${title_name}.vod.mp4"  \
    -hide_banner \
    -vn \
    -loglevel error \
    -ar 16000 \
    -ac 1 \
    -c:a pcm_s16le \
    -y \
    "${temp_dir}/${title_name}.vod-resampled.wav";

msg "Transcribing to subtitle file...";
msg "Whisper specified at: '${WHISPER_EXECUTABLE}'";

"${WHISPER_EXECUTABLE}" \
    -m "${MODEL_PATH}" \
    -l "${WHISPER_LANG}" \
    -f "${temp_dir}/${title_name}.vod-resampled.wav" \
    -t "${WHISPER_THREAD_COUNT}" \
    -osrt \
    --translate;

msg "Embedding subtitle track...";

ffmpeg -i "${temp_dir}/${title_name}.vod.mp4" \
    -hide_banner \
    -loglevel error \
    -i "${temp_dir}/${title_name}.vod-resampled.wav.srt" \
    -c copy \
    -c:s mov_text \
    -y "${title_name}-res.mp4";

#cleanup "${temp_dir}";

msg "Done! Your finished file is ready: ${title_name}-res.mp4";
