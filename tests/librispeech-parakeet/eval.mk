PYTHON = python

PARAKEET_PREFIX = ../../
PARAKEET_MODEL = parakeet-tdt-0.6b-v3

PARAKEET_CLI = $(PARAKEET_PREFIX)build-cuda-89-release/bin/parakeet-cli
PARAKEET_FLAGS = --no-prints --output-txt

# You can create eval.conf to override the PARAKEET_* variables
# defined above.
-include eval.conf

# This follows the file structure of the LibriSpeech project.
AUDIO_SRCS = $(sort $(wildcard LibriSpeech/*/*/*/*.flac))
TRANS_TXTS = $(addsuffix .txt, $(AUDIO_SRCS))

# We output the evaluation result to this file.
DONE = $(PARAKEET_MODEL).txt

all: $(DONE)

$(DONE): $(TRANS_TXTS)
	$(PYTHON) eval.py > $@.tmp
	mv $@.tmp $@

# Note: This task writes to a temporary file first to
# create the target file atomically.
%.flac.txt: %.flac
	$(PARAKEET_CLI) $(PARAKEET_FLAGS) --model $(PARAKEET_PREFIX)models/ggml-$(PARAKEET_MODEL).bin --file $^ --output-file $^.tmp
	mv $^.tmp.txt $^.txt

archive:
	tar -czf $(PARAKEET_MODEL).tar.gz --exclude="*.flac" LibriSpeech $(DONE)

clean:
	@rm -f $(TRANS_TXTS)
	@rm -f $(DONE)

.PHONY: all clean
