# whisper.wasm

Inference of [OpenAI's Whisper ASR model](https://github.com/openai/whisper) inside the browser

This example uses a WebAssembly (WASM) port of the [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
implementation of the transformer to run the inference inside a web page. The audio data does not leave your computer -
it is processed locally on your machine. The performance is not great but you should be able to achieve x2 or x3
real-time for the `tiny` and `base` models on a modern CPU and browser (i.e. transcribe a 60 seconds audio in about
~20-30 seconds).

This WASM port utilizes [WASM SIMD 128-bit intrinsics](https://emcc.zcopy.site/docs/porting/simd/) so you have to make
sure that [your browser supports them](https://webassembly.org/roadmap/).

The example is capable of running all models up to size `small` inclusive. Beyond that, the memory requirements and
performance are unsatisfactory. The implementation currently support only the `Greedy` sampling strategy. Both
transcription and translation are supported.

Since the model data is quite big (74MB for the `tiny` model) you need to manually load the model into the web-page.

The example supports both loading audio from a file and recording audio from the microphone. The maximum length of the
audio is limited to 120 seconds.

## Live demo

Link: https://ggml.ai/whisper.cpp/

![image](https://user-images.githubusercontent.com/1991296/197348344-1a7fead8-3dae-4922-8b06-df223a206603.png)

## Build instructions

```bash (v3.1.2)
# build using Emscripten
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
mkdir build-em && cd build-em
emcmake cmake ..
make -j
```
The example can then be started by running a local HTTP server:
```console
python3 examples/server.py
```
And then opening a browser to the following URL:
http://localhost:8000/whisper.wasm

To run the example in a different server, you need to copy the following files
to the server's HTTP path:
```
# copy the produced page to your HTTP path
cp bin/whisper.wasm/*    /path/to/html/
cp bin/libmain.js        /path/to/html/
cp bin/libmain.worker.js /path/to/html/
```

> 📝 **Note:** As of Emscripten 3.1.58 (April 2024), separate worker.js files are no
> longer generated and the worker is embedded in the main JS file. So the worker
> file will not be geneated for versions later than `3.1.58`.
