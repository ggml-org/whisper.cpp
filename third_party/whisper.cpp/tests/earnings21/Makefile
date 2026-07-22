GIT_URL = https://github.com/revdotcom/speech-datasets

all: eval

eval:
	$(MAKE) -f eval.mk

clean:
	$(MAKE) -f eval.mk clean

get-audio:
	git clone --depth 1  --filter=blob:none  --sparse $(GIT_URL)
	git -C speech-datasets sparse-checkout init --cone
	git -C speech-datasets sparse-checkout set earnings21

.PHONY: all eval clean get-audio
