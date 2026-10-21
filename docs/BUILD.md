# Building the speech stack

Part of the [speech-stack documentation](../README.md).

## Build

Prerequisites: CMake 3.20 or newer, a C++17 compiler, and Git. Model conversion
and optional Core ML export have separate Python dependencies in the engine
guides. Backend toolchains, such as CUDA or the Vulkan SDK, are required only
when building those ggml backends.

From the repository root, build and install a shared speech-branch ggml, then
configure the umbrella against its install prefix. These development commands
use the checked-out branch revision; record `git -C ggml-src rev-parse HEAD`
with your build. To reproduce a registry build, fetch and check out the exact
source `REF` in the [`ggml-speech` portfile](https://github.com/tetherto/qvac-registry-vcpkg/blob/main/ports/ggml-speech/portfile.cmake)
before configuring, rather than using the moving branch tip.

```sh
git clone --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src
cmake -S ggml-src -B ggml-src/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_LLAMAFILE=ON \
  -DCMAKE_INSTALL_PREFIX="$PWD/ggml-install"
cmake --build ggml-src/build --parallel
cmake --install ggml-src/build

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/ggml-install"
cmake --build build --parallel
```

Whisper, Parakeet and AudioGen follow `BUILD_SHARED_LIBS`; Whisper defaults it
to `ON` in the umbrella. TTS independently uses `TTS_CPP_BUILD_SHARED`, default
`OFF`. See [TTS linkage and test restrictions](../engines/tts/docs/build.md#useful-cmake-options)
before enabling it. All umbrella engines use the same installed ggml.

### Windows and multi-config generators

Run from the repository root in Developer PowerShell. This example uses the
Visual Studio generator and a Release configuration:

```powershell
git clone --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src
$speechGgmlInstall = Join-Path (Get-Location) "ggml-install"
cmake -S ggml-src -B ggml-src/build `
  -DBUILD_SHARED_LIBS=ON -DGGML_LLAMAFILE=ON `
  "-DCMAKE_INSTALL_PREFIX=$speechGgmlInstall"
cmake --build ggml-src/build --config Release --parallel
cmake --install ggml-src/build --config Release
cmake -S . -B build "-DCMAKE_PREFIX_PATH=$speechGgmlInstall"
cmake --build build --config Release --parallel
$env:PATH = "$speechGgmlInstall\bin;$speechGgmlInstall\lib;$env:PATH"
build\bin\Release\whisper-cli.exe --help
build\engines\parakeet\Release\parakeet.exe --help
```

Visual Studio and Xcode place executables under the selected configuration
directory. Use `--config Release` for build/install and `-C Release` for CTest.
Ninja and Make use the single-config paths shown in the CLI guides. For a
registry reproduction, select the pinned ggml source before the configure step
here as well. Dynamic backend builds also need their backend modules staged;
see the relevant engine backend guide.

### CMake options

| Option | Default | Effect |
|---|---|---|
| `SPEECH_BUILD_WHISPER` | `ON` | Build `third_party/whisper.cpp` |
| `SPEECH_BUILD_PARAKEET` | `ON` | Build `engines/parakeet` |
| `SPEECH_BUILD_TTS` | `ON` | Build `engines/tts` |
| `SPEECH_BUILD_AUDIOGEN` | `ON` | Build `engines/audiogen` |
| `SPEECH_BUILD_EXECUTABLES` | `ON` | Build development CLIs |
| `SPEECH_BUILD_TESTS` | `OFF` | Build engine test harnesses and enable root CTest |
| `SPEECH_BUILD_WHISPER_TESTS` | `OFF` | Build Whisper tests; use with `SPEECH_BUILD_TESTS=ON` |
| `AUDIOGEN_BUILD_MINIMAX` | Desktop `ON`; mobile forced `OFF` | Build MiniMax-Music3 on desktop |
| `TTS_CPP_BUILD_SHARED` | `OFF` | Select TTS linkage independently of `BUILD_SHARED_LIBS` |

GPU backends come from the ggml build: `GGML_VULKAN`, `GGML_OPENCL`,
`GGML_CUDA`, `GGML_WEBGPU`, or `GGML_HEXAGON`; Metal defaults on for Apple builds.
Core ML options default off and are Apple-only: `WHISPER_COREML`,
`PARAKEET_COREML`, `TTS_CPP_COREML`, `AUDIOGEN_COREML`.
The [sidecar overview](../README.md#apple-core-ml-sidecars) links to each
engine's export, input-routing and fallback contracts.

### WebGPU (Dawn)

`GGML_WEBGPU=ON` builds ggml's WebGPU backend on [Dawn](https://dawn.googlesource.com/dawn),
Google's native WebGPU implementation. The engines drive it through the
native `webgpu.h` API inside the process, so no browser is involved; Dawn
targets Vulkan on Linux, Windows and Android, Metal on Apple platforms, and
D3D12 on Windows. The backend registers as `WebGPU` in the ggml registry
and is selected like any other GPU backend: automatically by each engine's
runtime tiering (it ranks with Vulkan and Metal, below CUDA on the same
NVIDIA card), or explicitly with `--backend webgpu` on the CLIs and
`backend = "webgpu"` in the engine options. Model validation per engine is
documented in the engine backend guides; a backend that compiles does not
imply every model has been checked on it.

The ggml CI lanes use Dawn's prebuilt release binaries; the `ggml-speech`
vcpkg port builds Dawn from source through the official `dawn` port instead
(the same toolchain and C++ runtime as the consumer, which a prebuilt Dawn
cannot promise). To reproduce the CI build, point CMake at a Dawn install
and enable the backend when building ggml:

```sh
curl -fL -o dawn.tar.gz \
  https://github.com/google/dawn/releases/download/v20260317.182325/Dawn-18eb229ef5f707c1464cc581252e7603c73a3ef0-ubuntu-latest-Release.tar.gz
mkdir dawn && tar -xf dawn.tar.gz -C dawn --strip-components=1
cmake -S ggml-src -B ggml-src/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_WEBGPU=ON \
  -DDawn_DIR="$PWD/dawn/lib64/cmake/Dawn" \
  -DCMAKE_INSTALL_PREFIX="$PWD/ggml-install"
cmake --build ggml-src/build --parallel
cmake --install ggml-src/build
```

Pick the `macos-latest` asset on Apple hosts, whose config lives under
`lib/cmake/Dawn`. Shader embedding needs a `python3` on the PATH at configure
time. A static ggml exports Dawn in its link interface, so pass the same
`Dawn_DIR` when configuring the umbrella against a static WebGPU ggml as
well (a shared ggml needs nothing). Dawn hides
`shader-f16` on NVIDIA Vulkan unless the backend enables the
`vulkan_enable_f16_on_nvidia` toggle, which ggml-webgpu does; a host whose
adapter still lacks it, or lacks implicit device synchronization, registers no
WebGPU device and logs why. The backend requests a high-performance adapter
(the discrete GPU on hybrid machines), never a software fallback adapter, and
refuses Dawn's Null adapter, so a host without a usable GPU gets the engines'
ordinary CPU fallback instead of silent empty output. Two limitations to keep
in mind: the device reports Dawn's maximum buffer size as its memory, so the
engines' memory-fit projections are not meaningful on WebGPU, and the WGSL
matmul kernels keep activations in f16, which the fork's `test-backend-ops`
large-activation and `GGML_PREC_F32` cases fail on; the
[validation table](../README.md#webgpu-dawn) records which models that
affects. Desktop Linux is the validated platform; Dawn's Android, Windows and
Apple paths compile but have not been exercised with the speech engines.

### Tests

For the non-GPU engine suite:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/ggml-install" -DSPEECH_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build -LE 'gpu|perf' --output-on-failure
```

Fixture-backed engine tests are disabled when required files are absent at
configure time. Reconfigure after provisioning models/references. If enabling
Whisper tests too, its committed weightless fixtures cover pipeline smokes;
`test-vad-full` requires a downloaded model. The model-download-free CI filter is:

```sh
ctest --test-dir build -LE 'gpu|perf|medium|large' \
  -E '^test-vad-full$' --output-on-failure
```

On multi-config generators add `-C Release`. Disabled or skipped tests are
not executed coverage. Model/backend parity setup is documented in the engine
testing guides.

### Standalone builds

Each engine also configures directly. Defaults and ggml acquisition differ:
[Parakeet](../engines/parakeet/docs/build.md), [TTS](../engines/tts/docs/build.md),
[AudioGen](../engines/audiogen/docs/build.md).

## Consumable packages

The [`speech-cpp`](https://github.com/tetherto/qvac-registry-vcpkg/tree/main/ports/speech-cpp)
vcpkg port builds the umbrella. Select engine features such as
`speech-cpp[whisper,parakeet,vulkan]`; backend features (`metal`, `vulkan`,
`opencl`, `cuda`, `webgpu`) select the corresponding shared `ggml-speech`
dependency features. `webgpu` is opt-in on every platform and pulls the
registry's own `dawn` port (the official port pinned at the version the
speech-branch `ggml-webgpu` compiles against, Linux defaults trimmed to
Vulkan), so a consumer manifest must import `vulkan-headers`,
`vulkan-utility-libraries` and `vcpkg-get-python-packages` from
microsoft/vcpkg. CMake package names remain engine-specific.

| Feature | `find_package(... CONFIG REQUIRED)` | Imported target |
|---|---|---|
| Shared dependency | `ggml` | `ggml::ggml` |
| `speech-cpp[whisper]` | `whisper` | `whisper::whisper` |
| `speech-cpp[parakeet]` | `qvac-parakeet` | `qvac::parakeet` |
| `speech-cpp[tts]` | `tts-cpp` | `tts-cpp::tts-cpp` |
| `speech-cpp[audiogen]` | `audiogen-cpp` | `audiogen-cpp::audiogen-cpp` |

For example, after installing `speech-cpp[tts]` through the registry/toolchain:

```cmake
find_package(tts-cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE tts-cpp::tts-cpp)
```
