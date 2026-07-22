#!/bin/sh
#
# This generates:
#   - src/coreml/whisper-encoder-impl.h and src/coreml/whisper-encoder-impl.m
#   - src/coreml/whisper-decoder-impl.h and src/coreml/whisper-decoder-impl.m
#

set -e

wd=$(dirname "$0")
cd "$wd/../" || exit

python3 models/convert-whisper-to-coreml.py --model tiny.en

mv -v models/coreml-encoder-tiny.en.mlpackage models/whisper-encoder-impl.mlpackage
xcrun coremlc generate models/whisper-encoder-impl.mlpackage src/coreml/
mv src/coreml/whisper_encoder_impl.h src/coreml/whisper-encoder-impl.h
mv src/coreml/whisper_encoder_impl.m src/coreml/whisper-encoder-impl.m
sed -i '' 's/whisper_encoder_impl\.h/whisper-encoder-impl.h/g' src/coreml/whisper-encoder-impl.m
sed -i '' 's/whisper_encoder_impl\.m/whisper-encoder-impl.m/g' src/coreml/whisper-encoder-impl.m
sed -i '' 's/whisper_encoder_impl\.h/whisper-encoder-impl.h/g' src/coreml/whisper-encoder-impl.h

mv -v models/coreml-decoder-tiny.en.mlpackage models/whisper-decoder-impl.mlpackage
xcrun coremlc generate models/whisper-decoder-impl.mlpackage src/coreml/
mv src/coreml/whisper_decoder_impl.h src/coreml/whisper-decoder-impl.h
mv src/coreml/whisper_decoder_impl.m src/coreml/whisper-decoder-impl.m
sed -i '' 's/whisper_decoder_impl\.h/whisper-decoder-impl.h/g' src/coreml/whisper-decoder-impl.m
sed -i '' 's/whisper_decoder_impl\.m/whisper-decoder-impl.m/g' src/coreml/whisper-decoder-impl.m
sed -i '' 's/whisper_decoder_impl\.h/whisper-decoder-impl.h/g' src/coreml/whisper-decoder-impl.h

rm -rfv models/whisper-encoder-impl.mlpackage models/whisper-decoder-impl.mlpackage
