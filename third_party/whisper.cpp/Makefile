#
# Audio samples
#

.PHONY: build
build:
	cmake -B build $(CMAKE_ARGS)
	cmake --build build --config Release

# download a few audio samples into folder "./samples":
.PHONY: samples
samples:
	@echo "Downloading samples..."
	@mkdir -p samples
	@wget --quiet --show-progress -O samples/gb0.ogg https://upload.wikimedia.org/wikipedia/commons/2/22/George_W._Bush%27s_weekly_radio_address_%28November_1%2C_2008%29.oga
	@wget --quiet --show-progress -O samples/gb1.ogg https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg
	@wget --quiet --show-progress -O samples/hp0.ogg https://upload.wikimedia.org/wikipedia/en/d/d4/En.henryfphillips.ogg
	@wget --quiet --show-progress -O samples/mm1.wav https://cdn.openai.com/whisper/draft-20220913a/micro-machines.wav
	@wget --quiet --show-progress -O samples/a13.mp3 https://upload.wikimedia.org/wikipedia/commons/transcoded/6/6f/Apollo13-wehaveaproblem.ogg/Apollo13-wehaveaproblem.ogg.mp3
	@wget --quiet --show-progress -O samples/diffusion2023-07-03.flac https://archive.org/download/diffusion2023-07-03/diffusion2023-07-03.flac

#
# Models
#

# if not already downloaded, the following targets download the specified model and
# runs it on all samples in the folder "./samples":

.PHONY: tiny.en
.PHONY: tiny
.PHONY: base.en
.PHONY: base
.PHONY: small.en
.PHONY: small
.PHONY: medium.en
.PHONY: medium
.PHONY: large-v1
.PHONY: large-v2
.PHONY: large-v3
.PHONY: large-v3-turbo

tiny.en tiny base.en base small.en small medium.en medium large-v1 large-v2 large-v3 large-v3-turbo:
	bash ./models/download-ggml-model.sh $@
	cmake -B build $(CMAKE_ARGS)
	cmake --build build --config Release
	@echo ""
	@echo "==============================================="
	@echo "Running $@ on all samples in ./samples ..."
	@echo "==============================================="
	@echo ""
	@for f in samples/*.{flac,mp3,ogg,wav}; do \
		echo "----------------------------------------------" ; \
		echo "[+] Running $@ on $$f ... (run 'ffplay $$f' to listen)" ; \
		echo "----------------------------------------------" ; \
		echo "" ; \
		./build/bin/whisper-cli -m models/ggml-$@.bin -f $$f ; \
		echo "" ; \
	done
