# parakeet.cpp/tests/librispeech

[LibriSpeech](https://www.openslr.org/12) is a standard dataset for
training and evaluating automatic speech recognition systems.

This directory contains a set of tools to evaluate the recognition
performance of parakeet.cpp on LibriSpeech corpus.

## Quick Start

1. (Pre-requirement) Compile `parakeet-cli` and prepare the Parakeet
   model in `ggml` format.

   ```
   $ # Execute the commands below in the project root dir.
   $ cmake -B build
   $ cmake --build build --config Release
   ```

2. Download the audio files from LibriSpeech project.

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
PARAKEET_MODEL = parakeet-tdt-0.6b-v3
PARAKEET_FLAGS = --no-prints --threads 8 --language en --output-txt
```

Check out `eval.mk` for more details.
