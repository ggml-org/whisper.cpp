# whisper.cpp/tests/earnings21

[Earnings-21](https://arxiv.org/abs/2104.11348) is a real-world benchmark
dataset that contains 39-hours of long-form English speech, sourced from
public earning calls.

This directory contains a set of scripts to evaluate the performance of
whisper.cpp on Earnings-21 corpus.

## Quick Start

1. (Pre-requirement) Compile `whisper-cli` and prepare the Whisper
   model in `ggml` format.

   ```
   $ # Execute the commands below in the project root dir.
   $ cmake -B build
   $ cmake --build build --config Release
   $ ./models/download-ggml-model.sh tiny
   ```

   Consult [whisper.cpp/README.md](../../README.md) for more details.

2. Download the audio files.

   ```
   $ make get-audio
   ```

3. Set up the environment to compute WER score.

   ```
   $ pip install -r requirements.txt
   ```

   For example, if you use `virtualenv`, you can set up it as follows:

   ```
   $ python3 -m venv venv
   $ . venv/bin/activate
   $ pip install -r requirements.txt
   ```

4. Run the benchmark test.

   ```
   $ make
   ```

## How-to guides

### How to change the inference parameters

Create `eval.conf` and override variables.

```
WHISPER_MODEL = large-v3-turbo
WHISPER_FLAGS = --no-prints --threads 8 --language en --output-txt
```

Check out `eval.mk` for more details.

### How to perform the benchmark test on a 10-hour subset

Earnings-21 provides a small but representative subset (approximately
10-hour audio data) to evaluate ASR systems quickly.

To switch to the subset, create `eval.conf` and add the following line:

```
EARNINGS21_EVAL10 = yes
```

### How to run the benchmark test using VAD

First, you need to download a VAD model:

```
$ # Execute the commands below in the project root dir.
$ ./models/download-vad-model.sh silero-v5.1.2
```

Create `eval.conf` with the following content:

```
WHISPER_FLAGS = --no-prints --language en --output-txt --vad --vad-model ../../models/ggml-silero-v5.1.2.bin
```
