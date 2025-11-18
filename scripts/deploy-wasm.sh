#!/bin/bash
#
# This is a helper script to deploy all WebAssembly examples to my node
# Run from the build directory:
#
# cd build-em
# ../scripts/deploy-wasm.sh
#

# check if emcmake is available
if ! command -v emcmake &> /dev/null
then
    echo "Error: emscripten environment is not set up"
    exit
fi

emcmake cmake .. && make -j
if [ $? -ne 0 ]; then
    echo "Error: build failed"
    exit
fi

# copy all wasm files to the node
scp bin/whisper.wasm/* root@linode0:/var/www/html/whisper/         && scp bin/libmain.worker.js    root@linode0:/var/www/html/whisper/
scp bin/stream.wasm/*  root@linode0:/var/www/html/whisper/stream/  && scp bin/libstream.worker.js  root@linode0:/var/www/html/whisper/stream/
scp bin/command.wasm/* root@linode0:/var/www/html/whisper/command/ && scp bin/libcommand.worker.js root@linode0:/var/www/html/whisper/command/
scp bin/talk.wasm/*    root@linode0:/var/www/html/whisper/talk/    && scp bin/libtalk.worker.js    root@linode0:/var/www/html/whisper/talk/
scp bin/bench.wasm/*   root@linode0:/var/www/html/whisper/bench/   && scp bin/libbench.worker.js   root@linode0:/var/www/html/whisper/bench/

echo "Done"
exit
