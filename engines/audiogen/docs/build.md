# audiogen engine: build

Part of the [audiogen engine documentation](../README.md).

## Build

Standalone from the repository root, against an installed `ggml-speech`:

```sh
cmake -S engines/audiogen -B build/audiogen -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build/audiogen -j
./build/audiogen/music-cli --help
```

The umbrella build uses `SPEECH_BUILD_AUDIOGEN=ON` (the default). In-tree CLIs
and tests link an object library, so they work when `BUILD_SHARED_LIBS=ON`
(whisper's umbrella default) and `audiogen-cpp` is a hidden shared library.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-install -DSPEECH_BUILD_AUDIOGEN=ON
cmake --build build -j
./build/engines/audiogen/music-cli --help
```

Visual Studio, Xcode, and other multi-config generators add a configuration
directory such as `Release/` beneath the executable directory. See the
[shared build guide](../../../docs/BUILD.md) for the ggml build.

| Option | Default | Effect |
|---|---|---|
| `AUDIOGEN_BUILD_LIBRARY` | `ON` | build the library; linkage follows `BUILD_SHARED_LIBS` |
| `AUDIOGEN_BUILD_EXECUTABLES` | `ON` standalone, `OFF` as a subdirectory | CLIs and per-stage smoke harnesses |
| `AUDIOGEN_BUILD_TESTS` | `ON` standalone, `OFF` as a subdirectory | unit, integration, and backend parity tests |
| `AUDIOGEN_BUILD_MINIMAX` | `ON` on desktop, unavailable on Android and iOS | MiniMax-Music3 engine; CPU by default, GPU through `EngineOptions::device` |
| `AUDIOGEN_COREML` | `OFF` | Apple-only Core ML sidecars: [ACE-Step VAE decoder](backends.md#core-ml-vae-decoder-sidecar) and explicit [MiniMax conditioner](backends.md#minimax-core-ml-conditioner) |
| `AUDIOGEN_INSTALL` | `ON` | generate install rules |
| `AUDIOGEN_USE_SYSTEM_GGML` | `ON` | `find_package(ggml)`; required, there is no supported vendored ggml in this tree |
| `AUDIOGEN_CCACHE` | `ON` | use ccache when available |

Consume the installed package with:

```cmake
find_package(audiogen-cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE audiogen-cpp::audiogen-cpp)
```

Minimal C++ usage:

```cpp
#include <audiogen-cpp/acestep/engine.h>

#include <exception>
#include <iostream>

int main() {
    try {
        tts_cpp::acestep::EngineOptions options;
        options.models_dir = "models/acestep";
        options.n_gpu_layers = 99;

        auto engine = tts_cpp::acestep::Engine::create(options);
        tts_cpp::acestep::GenerateParams params;
        params.caption = "Driving synth pop with bright analog leads";
        params.duration = 8.0f;

        bool cancel_requested = false;
        auto result = engine->generate(params, [&cancel_requested](const std::string &, int, int) {
            return !cancel_requested; // return false to cancel cooperatively
        });
        std::cout << result.pcm.size() / 2 << " stereo frames at "
                  << result.sample_rate << " Hz\n";
        // result.pcm is interleaved: pcm[frame * 2 + channel].
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
```
