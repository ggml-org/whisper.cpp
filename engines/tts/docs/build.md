# tts engine: build

Part of the [tts engine documentation](../README.md).

## Build paths

Prerequisites are CMake 3.20 or newer, a C++17 compiler, and either an installed
`ggml` CMake package from the `ggml-speech` port or a `qvac-ext-ggml@speech`
checkout staged by `scripts/setup-ggml.sh` for the bundled path.

| Path | Configure entry | ggml source | Main artifacts |
|---|---|---|---|
| Umbrella speech build | repository root, `-DSPEECH_BUILD_TTS=ON` | system `ggml` | `build/engines/tts/tts-cli` and sibling tools; library under `build/engines/tts/` |
| Direct in-tree package | `engines/tts` | system `ggml` by default (`TTS_CPP_USE_SYSTEM_GGML=ON`) | `engines/tts/build/tts-cli` and `engines/tts/build/libtts-cpp.*` |
| Bundled ggml | `engines/tts` with `TTS_CPP_USE_SYSTEM_GGML=OFF` | `engines/tts/ggml` checkout of `qvac-ext-ggml@speech`, staged by `scripts/setup-ggml.sh` | `engines/tts/build-bundled/tts-cli` and `engines/tts/build-bundled/libtts-cpp.*` |
| vcpkg consumer | `speech-cpp[tts]` feature | `ggml-speech` dependency | `<vcpkg>/installed/<triplet>/lib/` (or `debug/lib/`), headers under `include/tts-cpp/`, config under `share/tts-cpp/` |

The bundled path applies no patch overlay — the speech branch is patched at
the commit level. `scripts/setup-ggml.sh` clones the pinned speech ref into
`engines/tts/ggml/` and is idempotent on re-run; bump `GGML_REF` in the script
to move the pin. The separate
`chatterbox.cpp` repository has its own pinned bundled-ggml setup flow and
produces artifacts under its own `build/`.

### Umbrella build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-speech/install \
  -DSPEECH_BUILD_TTS=ON
cmake --build build --target tts-cli
```

`SPEECH_BUILD_EXECUTABLES` controls all engine CLIs and defaults to `ON`;
`SPEECH_BUILD_TESTS` controls TTS test harnesses and defaults to `OFF`.

When building the `ggml-speech` install yourself, configure the ggml build
with `-DGGML_LLAMAFILE=ON`: standalone ggml defaults it OFF, and without
tinyBLAS every CPU matmul takes the generic vec_dot path (measured 1.8x
slower on the CosyVoice3 DiT). The bundled-ggml path
(`TTS_CPP_USE_SYSTEM_GGML=OFF`) already defaults it ON.

### Direct in-tree build

From the repository root:

```bash
cmake -S engines/tts -B engines/tts/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-speech/install
cmake --build engines/tts/build --target tts-cli
```

The detailed commands below assume `cd engines/tts`, so direct-build
executables are shown as `build/<name>`.

### Bundled-ggml build

Stage `qvac-ext-ggml@speech` as `engines/tts/ggml`, then configure:

```bash
bash engines/tts/scripts/setup-ggml.sh
cmake -S engines/tts -B engines/tts/build-bundled \
  -DCMAKE_BUILD_TYPE=Release -DTTS_CPP_USE_SYSTEM_GGML=OFF
cmake --build engines/tts/build-bundled --target tts-cli
```

### Windows and multi-config generators

Use a Developer PowerShell or Developer Command Prompt and quote paths:

```powershell
cmake -S engines/tts -B engines/tts/build `
  -DCMAKE_TOOLCHAIN_FILE=C:\src\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build engines/tts/build --config Release --target tts-cli
engines\tts\build\Release\tts-cli.exe --help
ctest --test-dir engines/tts/build -C Release -L unit --output-on-failure
```

Visual Studio and other multi-config generators place executables under
`Release\` (or the selected configuration) and require `ctest -C Release`.
Single-config Ninja/Make generators use the artifact paths in the table.

### vcpkg consumer

The port installs the library and public headers, not the development CLIs:

```cmake
find_package(tts-cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE tts-cpp::tts-cpp)
```

### Conversion tooling

- C++17 compiler (clang, gcc, or MSVC)
- CMake 3.20 or newer
- Python 3.10+ with `torch`, `numpy`, `onnx`, `gguf`, `huggingface_hub`,
  `safetensors`, `scipy`, `librosa`, `resampy` — needed **once**, at setup time only, to run the
  weight converters (which bake the precomputed mel filterbanks into the
  GGUFs) and the optional reference-dump scripts. Once the GGUFs exist,
  the C++ binary has zero runtime dependency on Python.

The easiest way to get the Python side is:

```bash
git clone https://github.com/resemble-ai/chatterbox.git chatterbox-ref
cd chatterbox-ref
python -m venv .venv && . .venv/bin/activate
pip install -e .
pip install onnx gguf huggingface_hub safetensors scipy librosa resampy
cd -
```

### Shared ggml and registry features

Production consumers install `speech-cpp[tts]`, whose shared `ggml-speech`
dependency supplies the requested backends. The installed CMake package remains
`tts-cpp`; it does not require a separate legacy engine port. See the
[umbrella package guide](../../../docs/BUILD.md#consumable-packages) for feature
selection and imported targets. Development commands above use an explicit
ggml install prefix; record that source revision when reporting results.

### Useful CMake options

Project-namespaced flags (all default to a sensible standalone build;
override with `-D<flag>=...` at configure time):

| Flag | Default | Meaning |
|------|---------|---------|
| `TTS_CPP_BUILD_LIBRARY` | `ON` | Build the `tts-cpp` library target itself (linkage controlled by `TTS_CPP_BUILD_SHARED`, not `BUILD_SHARED_LIBS` — see below) |
| `TTS_CPP_BUILD_SHARED` | `OFF` | Build `tts-cpp` as `SHARED` instead of `STATIC`. Decoupled from `BUILD_SHARED_LIBS` because ggml's own CMake declares its own `option(BUILD_SHARED_LIBS)` defaulting to `ON` on Windows non-MinGW; using a project-namespaced option keeps the two independent. The supertonic test/bench harnesses link against `tts-cpp` directly and use detail-namespaced symbols outside the `TTS_CPP_API` public surface, so `SHARED` builds hide them and disable those targets — leave OFF for development, flip ON only for downstream packaging where the test harnesses aren't built |
| `TTS_CPP_BUILD_EXECUTABLES` | `ON` standalone / `OFF` subdir | `tts-cli`, `mel2wav`, `cosyvoice-hift`, `cosyvoice-flow`, `cosyvoice-llm`, `cosyvoice-cli`, `cosyvoice-bench`, `supertonic-cli`, `parler-cli`, `parler-bench`, `lavasr-bench`, `audio8-cli`, `pocket-cli`, and `moss-cli` |
| `TTS_CPP_BUILD_TESTS` | `ON` standalone / `OFF` subdir | `test-*` parity / unit harnesses, registered with CTest (label-filterable via `ctest -L unit` / `ctest -L fixture` / `ctest -L gpu`) |
| `TTS_CPP_INSTALL` | `ON` | Generate `install` rules + the `tts-cpp` CMake package config so consumers can `find_package(tts-cpp CONFIG REQUIRED)` |
| `TTS_CPP_USE_SYSTEM_GGML` | `ON` | Use `find_package(ggml CONFIG REQUIRED)` against `ggml-speech`. `OFF` uses `add_subdirectory(ggml)` and requires an `engines/tts/ggml` checkout of `qvac-ext-ggml@speech`, staged by `scripts/setup-ggml.sh`; no patch overlay — the speech branch is patched at the commit level |
| `TTS_CPP_CCACHE` | `ON` | Use ccache as compiler launcher for `tts-cpp`'s own targets when `find_program(ccache)` succeeds. Scoped per target; ggml's independent `GGML_CCACHE` option handles the ggml subdirectory |
| `TTS_CPP_COREML` | `OFF` | Apple-only. Compile the Core ML (Neural Engine) sidecars for the Audio8 codec synthesis stack, the Supertonic vocoder and the Parler-TTS DAC decoder (among others): at load each engine looks for its sidecar next to the GGUF (`audio8-codec-decoder-q8_0.gguf` → `audio8-codec-decoder.mlmodelc`; `supertonic2-q8_0.gguf` → `supertonic2-vocoder.mlmodelc`; `parler-mini-v1-q8_0.gguf` → `parler-mini-v1-dac.mlmodelc`; `chatterbox-s3gen-q8_0.gguf` → `chatterbox-s3gen-{encoder,cfm,hift}.mlmodelc`) and runs the stage on it when present, on ggml otherwise (Supertonic also keeps ggml for vocoders stored below 8 bits, such as `q4_0`). Export with `scripts/export-audio8-codec-coreml.py` / `scripts/export-supertonic-coreml.py` / `scripts/parler/export-dac-coreml.py` / `scripts/export-chatterbox-coreml.py`; see the [Audio8 guide](audio8.md#core-ml-codec-sidecar), the [Supertonic guide](supertonic.md#core-ml-vocoder-sidecar) the [Parler guide](parler.md#core-ml-dac-sidecar-apple-neural-engine) and the [Chatterbox guide](chatterbox.md#core-ml-sidecars). A non-Apple configure with this `ON` is a hard error |
| `TTS_CPP_OPENMP` | `ON` | Link OpenMP when available. On Windows non-MinGW builds it is forced `OFF` to avoid clang-cl/MSVC and msys2 `libgomp` incompatibilities; advanced callers can explicitly set both `TTS_CPP_OPENMP_USER_OVERRIDE=ON` and `TTS_CPP_OPENMP=ON` |

When tests are enabled, the build also exposes three cache-overridable
fixture roots that the registered ctest entries resolve against:

| CACHE PATH | Default | Used by |
|------------|---------|---------|
| `TTS_CPP_TEST_MODEL_DIR` | `${CMAKE_CURRENT_SOURCE_DIR}/models`               | GGUF checkpoints (produced by `scripts/convert-*-to-gguf.py` or `bash scripts/setup-supertonic2.sh`) |
| `TTS_CPP_TEST_AUDIO_DIR` | `${CMAKE_CURRENT_SOURCE_DIR}/test/reference-audio` | Reference WAV fixtures (e.g. `test/reference-audio/jfk.wav`) |
| `TTS_CPP_TEST_REF_DIR`   | `${CMAKE_CURRENT_SOURCE_DIR}/artifacts`            | `.npy` reference dumps from `scripts/dump-{s3gen,supertonic,t3-mtl}-reference.py` |

Tests whose required fixtures don't exist at configure time are
auto-marked `DISABLED` (they still appear in `ctest -N` output but
return `Not Run` instead of failing), so a fresh checkout still gives
a green `ctest` run on the harnesses whose fixtures it does have.

## Repository layout

```
engines/tts/                     multi-engine TTS and enhancement package
  include/tts-cpp/               installed public headers (Engine API)
    tts-cpp.h                    library entry; declares tts_cpp_cli_main()
    chatterbox/engine.h          Engine + EngineOptions (text → wav)
    chatterbox/s3gen_pipeline.h  low-level S3Gen + HiFT pipeline entries
    audio8/engine.h              Audio8 Engine + EngineOptions + VoicePrompt
  src/
    main.cpp                     T3 turbo runtime + shared helpers (libtts-cpp)
    t3_mtl.{h,cpp}               T3 multilingual (Llama-520M) runtime + stage builders
    chatterbox_t3_internal.h     internal T3 declarations shared by main/engine/CLI
    chatterbox_engine.cpp        public Engine API impl (libtts-cpp)
    chatterbox_cli.cpp           CLI entry (`tts-cli` binary)
    cli_main.cpp                 thin int-main forwarder; calls tts_cpp_cli_main()
    chatterbox_tts.cpp           S3Gen + HiFT pipeline        (libtts-cpp)
    mel2wav.cpp                  HiFT-only demo              (mel2wav)
    gpt2_bpe.{h,cpp}             self-contained GPT-2 BPE tokenizer (Turbo)
    mtl_tokenizer.{h,cpp}        multilingual grapheme tokenizer
                                   (HF tokenizers.json + NFKD lowercasing)
    mtl_unicode_tables.inc       embedded NFKD + Korean Jamo lookup tables

    voice_features.{h,cpp}       WAV I/O, sinc resampler, LUFS meter,
                                   24 kHz & 16 kHz log-mel extraction,
                                   Kaldi-style 80-ch fbank
    mel_extract_stft.cpp         STFT-based mel extraction shared by C++ pipelines
    voice_encoder.{h,cpp}        3-layer LSTM → 256-d speaker_emb
                                   (matches Resemble VoiceEncoder)
    campplus.{h,cpp}             FunASR x-vector port (FCM + 3× CAMDense
                                   TDNN) → 192-d embedding
    s3tokenizer.{h,cpp}          6-layer FSMN-attn transformer + FSQ →
                                   25-Hz speech tokens
    dr_wav.h                     vendored single-header WAV reader
    npy.h                        minimal .npy load / save + compare

    audio8_cli.cpp               Audio8 CLI entry (`audio8-cli` binary)
    audio8/engine.cpp            public Engine API impl: prompt → codes → wav
    audio8/{tokenizer,unicode}.{h,cpp}  Qwen2.5 byte-level BPE + ChatML builder
    audio8/unicode_tables.inc    embedded NFC + character-category tables
    audio8/gguf.cpp              hparams, tensors, backend buffers
    audio8/{graph.h,graph.cpp}   shared ggml blocks (RoPE, attention, norms)
    audio8/lm.cpp                slow AR with KV cache + fast AR head
    audio8/codec_ops.{h,cpp}     causal convs, Snake, ConvNeXt, receptive spans
    audio8/codec_{decode,encode}.cpp  chunked synthesis / analysis stacks
    audio8/sampling.{h,cpp}      top-k/top-p, Gumbel draw, repetition-aware resample
    audio8/internal.h            model containers shared by the above

    test_*.cpp                   per-stage numerical-parity harnesses
                                   (S3Gen / HiFT / streaming / MTL T3 /
                                    MTL tokenizer / voice features / Metal ops)
  scripts/
    setup-ggml.sh                clones the pinned qvac-ext-ggml@speech ref
                                   into ggml/ for the bundled-ggml dev build
    synthesize.sh                text → wav wrapper around tts-cli
    convert-t3-turbo-to-gguf.py  Turbo T3 weights + GPT-2 BPE + VE + builtin
                                   voice → T3 GGUF (--quant)
    convert-t3-mtl-to-gguf.py    MTL T3 (Llama-520M) + perceiver + emotion-adv
                                   + tokenizers.json + builtin voice → T3 GGUF (--quant)
    convert-s3gen-to-gguf.py     S3Gen encoder + CFM + HiFT + CAMPPlus +
                                   S3TokenizerV2 + mel filterbanks → S3Gen GGUF
                                   (--variant {turbo,mtl}, --quant)
    requantize-gguf.py           in-place block-quantise of an existing
                                   T3/S3Gen GGUF (canonical deny-list lives here)
    extract-voice.py             one-shot voice-clone prep (silencedetect +
                                   codec-aware EQ + optional `--save-voice` bake)
    gen-nfkd-table.py            generates src/mtl_unicode_tables.inc
    dump-*-reference.py          PyTorch → .npy intermediates for the
                                   per-stage harnesses (S3Gen, CAMPPlus,
                                   S3TokenizerV2, streaming, MTL T3, Audio8)
    audio8_reference.py          shared Audio8 helpers: config + checkpoint
                                   loading, tensor renaming, RoPE tables,
                                   GGUF storage policy
    convert-audio8-lm-to-gguf.py Audio8 slow + fast AR + Qwen2.5 BPE → LM GGUF
    convert-audio8-codec-to-gguf.py Audio8 codec → encoder / decoder GGUF
                                   (weight norm folded, --part)
    verify-audio8-conversion.py  compares Audio8 GGUFs tensor by tensor
                                   against the checkpoint
    gen-audio8-unicode-tables.py generates src/audio8/unicode_tables.inc
    reference-t3-turbo.py        PyTorch T3 bit-exact compare vs C++
    compare-tokenizer.py         10-case BPE tokenizer compare vs HF
  (no patches/ overlay; ggml comes from the system ggml-speech package, or
   scripts/setup-ggml.sh stages an untracked ggml/ checkout for bundled builds)
  voices/                        baked voice profiles (not tracked; populated
                                   by --save-voice)
  models/                        generated GGUFs (not tracked)
  artifacts/                     .npy dumps for validation (not tracked)
  CMakeLists.txt                 top-level build
  README.md                      engine overview
  docs/history/                  archived port and integration reports
```
