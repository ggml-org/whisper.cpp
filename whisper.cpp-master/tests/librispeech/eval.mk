PYTHON = python

WHISPER_PREFIX = ../../
WHISPER_MODEL = tiny

WHISPER_CLI = $(WHISPER_PREFIX)build/bin/whisper-cli
WHISPER_FLAGS = --no-prints --language en --output-txt

# You can create eval.conf to override the WHISPER_* variables
# defined above.
-include eval.conf

# This follows the file structure of the LibriSpeech project.
AUDIO_SRCS = $(sort $(wildcard LibriSpeech/*/*/*/*.flac))
TRANS_TXTS = $(addsuffix .txt, $(AUDIO_SRCS))

# We output the evaluation result to this file.
DONE = $(WHISPER_MODEL).txt

all: $(DONE)

$(DONE): $(TRANS_TXTS)
	$(PYTHON) eval.py > $@.tmp
	mv $@.tmp $@

# Note: This task writes to a temporary file first to
# create the target file atomically.
%.flac.txt: %.flac
	$(WHISPER_CLI) $(WHISPER_FLAGS) --model $(WHISPER_PREFIX)models/ggml-$(WHISPER_MODEL).bin --file $^ --output-file $^.tmp
	mv $^.tmp.txt $^.txt

archive:
	tar -czf $(WHISPER_MODEL).tar.gz --exclude="*.flac" LibriSpeech $(DONE)

clean:
	@rm -f $(TRANS_TXTS)
	@rm -f $(DONE)

.PHONY: all clean
