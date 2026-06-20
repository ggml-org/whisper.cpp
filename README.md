# whisper.cpp-rocm

<a href="https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/latest" title="Download the latest release">
  <img src="https://img.shields.io/github/v/release/lemonade-sdk/whisper.cpp-rocm?logo=github&logoColor=white" alt="GitHub release (latest by date)" />
</a>
<a href="https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/latest" title="View latest release date">
  <img src="https://img.shields.io/github/release-date/lemonade-sdk/whisper.cpp-rocm?logo=github&logoColor=white" alt="Latest release date" />
</a>
<a href="LICENSE" title="View license">
  <img src="https://img.shields.io/github/license/lemonade-sdk/whisper.cpp-rocm?logo=opensourceinitiative&logoColor=white" alt="License" />
</a>
<a href="https://github.com/ROCm/ROCm" title="Powered by ROCm">
  <img src="https://img.shields.io/badge/ROCm-7.12-blue?logo=amd&logoColor=white" alt="ROCm 7.12" />
</a>
<a href="https://github.com/ggerganov/whisper.cpp" title="Powered by whisper.cpp">
  <img src="https://img.shields.io/badge/🎤 Powered%20by-whisper.cpp-blue" alt="Powered by whisper.cpp" />
</a>
<a href="#-supported-devices" title="Platform support">
  <img src="https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-0078D6?logo=windows&logoColor=white" alt="Platform: Windows | Linux | macOS" />
</a>
<a href="#-supported-devices" title="GPU targets">
  <img src="https://img.shields.io/badge/GPU-gfx110X%20%7C%20gfx1150%20%7C%20gfx1151%20%7C%20gfx120X-00B04F?logo=amd&logoColor=white" alt="GPU Targets" />
</a>
<a href="#-npu--ryzenai" title="NPU support">
  <img src="https://img.shields.io/badge/NPU-Ryzen%20AI%20300-ED1C24?logo=amd&logoColor=white" alt="NPU: Ryzen AI 300" />
</a>

Pre-built releases of **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** with full AMD hardware acceleration — **ROCm™ GPU**, **Vulkan GPU**, **RyzenAI NPU**, and optimised **CPU** builds — for Windows and Linux.

Releases track upstream whisper.cpp exactly: every time upstream publishes a new version, our automated pipeline syncs, builds all backends, and publishes a matching release within 24 hours. No manual steps. No lag.

> [!IMPORTANT]
> **No ROCm installation required.** All ROCm and Vulkan runtime libraries are bundled inside every release archive. Download, extract, and run.

> [!NOTE]
> This project is maintained by the [Lemonade SDK](https://github.com/lemonade-sdk/lemonade) team. Our primary focus is seamless integration with Lemonade and similar AMD-optimised AI applications. We welcome collaborations and contributions that advance AMD whisper.cpp support.

---

## 🎯 Supported Devices

### ROCm GPU

| Architecture | Devices |
|---|---|
| **gfx1151** — RDNA3.5 APU | Ryzen AI MAX+ Pro 395 (Strix Halo) |
| **gfx1150** — RDNA3.5 APU | Ryzen AI 300 series (Strix Point) |
| **gfx120X** — RDNA4 dGPU | Radeon RX 9070 XT / 9070 / 9060 XT / 9060 |
| **gfx110X** — RDNA3 dGPU & iGPU | RX 7900 XTX/XT/GRE, RX 7800 XT, RX 7700 XT, RX 7600 XT/7600; iGPU Radeon 780M / 760M / 740M |

### Vulkan GPU

Any GPU with a Vulkan 1.3-capable driver — AMD, NVIDIA, Intel. Covers iGPUs on all platforms where a Vulkan driver is present.

### NPU — RyzenAI

| Device | OS | Requirement |
|---|---|---|
| Ryzen AI 300 series (Strix Point / Strix Halo) | Windows only | NPU driver ≥ `.280` |

### CPU

Optimised CPU-only builds for x86-64. Windows and Linux. No GPU required.

---

## 📦 Downloads

All builds are self-contained — no separate driver or runtime installation needed (except the NPU driver for the NPU build).

### ROCm — GPU Accelerated

| GPU Target | Linux | Windows |
|---|---|---|
| **gfx1151** (Ryzen AI MAX+ Pro 395) | [![Linux gfx1151](https://img.shields.io/badge/Download-Linux%20gfx1151-blue?logo=linux&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-linux-rocm-gfx1151.tar.gz) | [![Windows gfx1151](https://img.shields.io/badge/Download-Windows%20gfx1151-green?logo=windows&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-windows-rocm-gfx1151.zip) |
| **gfx1150** (Ryzen AI 300) | [![Linux gfx1150](https://img.shields.io/badge/Download-Linux%20gfx1150-blue?logo=linux&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-linux-rocm-gfx1150.tar.gz) | [![Windows gfx1150](https://img.shields.io/badge/Download-Windows%20gfx1150-green?logo=windows&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-windows-rocm-gfx1150.zip) |
| **gfx120X** (RDNA4 dGPU) | [![Linux gfx120X](https://img.shields.io/badge/Download-Linux%20gfx120X-blue?logo=linux&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-linux-rocm-gfx120X.tar.gz) | [![Windows gfx120X](https://img.shields.io/badge/Download-Windows%20gfx120X-green?logo=windows&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-windows-rocm-gfx120X.zip) |
| **gfx110X** (RDNA3 dGPU & iGPU) | [![Linux gfx110X](https://img.shields.io/badge/Download-Linux%20gfx110X-blue?logo=linux&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-linux-rocm-gfx110X.tar.gz) | [![Windows gfx110X](https://img.shields.io/badge/Download-Windows%20gfx110X-green?logo=windows&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-windows-rocm-gfx110X.zip) |

### Vulkan — Cross-Vendor GPU

| Linux | Windows |
|---|---|
| [![Linux Vulkan](https://img.shields.io/badge/Download-Linux%20Vulkan-blue?logo=linux&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-linux-vulkan-x86_64.tar.gz) | [![Windows Vulkan](https://img.shields.io/badge/Download-Windows%20Vulkan-green?logo=windows&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-windows-vulkan-x64.zip) |

### NPU — RyzenAI (Windows only)

| Windows |
|---|
| [![Windows NPU](https://img.shields.io/badge/Download-Windows%20NPU%20(RyzenAI)-red?logo=amd&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-windows-npu-x64.zip) |

> Requires NPU driver ≥ `.280` and a pre-compiled `.rai` encoder model from [AMD's Hugging Face collection](https://huggingface.co/collections/amd/ryzen-ai-16-whisper-npu-optimized-onnx-models). Place the `.rai` file alongside your `ggml-*.bin` model — whisper-cli picks it up automatically.

### macOS — Metal GPU

| macOS (Apple Silicon) |
|---|
| [![macOS Metal](https://img.shields.io/badge/Download-macOS%20Metal%20(arm64)-lightgrey?logo=apple&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-darwin-metal-arm64.tar.gz) |

### CPU — No GPU Required

| Linux | Windows |
|---|---|
| [![Linux CPU](https://img.shields.io/badge/Download-Linux%20CPU-blue?logo=linux&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-linux-cpu-x86_64.tar.gz) | [![Windows CPU](https://img.shields.io/badge/Download-Windows%20CPU-green?logo=windows&logoColor=white)](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/download/v1.8.4/whisper-v1.8.4-release-windows-cpu-x64.zip) |

---

## 🧪 Quick Smoketest

### 1. Get a model

```bash
# Download the tiny.en model (~75 MB) for a fast smoke test
./models/download-ggml-model.sh tiny.en

# Or grab any ggml-*.bin from https://huggingface.co/ggerganov/whisper.cpp
```

### 2. Transcribe the bundled sample

```bash
# Linux
./whisper-cli -m models/ggml-tiny.en.bin -f samples/jfk.wav

# Windows
whisper-cli.exe -m models\ggml-tiny.en.bin -f samples\jfk.wav
```

Expected: a transcription of the JFK "Ask not what your country can do for you" excerpt.

### 3. Verify GPU is active (ROCm)

```bash
# At startup whisper-cli prints the backend in use — look for:
#   ggml_hip: using device ...
./whisper-cli -m models/ggml-tiny.en.bin -f samples/jfk.wav 2>&1 | grep -i "hip\|rocm\|device"
```

### 4. Verify NPU is active (VitisAI)

```
# Place the .rai encoder alongside the .bin model, then run normally.
# Look for this line in stdout:
#   whisper_vitisai_encode: Vitis AI model inference completed.
whisper-cli.exe -m models\ggml-tiny.en.bin -f samples\jfk.wav
```

### 5. Verify portability (Linux ROCm)

```bash
# ROCm runtime libs are bundled — RPATH should point to $ORIGIN (same dir as binary)
readelf -d whisper-cli | grep RPATH    # -> $ORIGIN
ldd whisper-cli | grep "not found"     # -> (empty — all deps resolved locally)
```

---

## 🔄 Release Cadence

Releases are fully automated and mirror upstream whisper.cpp releases with no manual steps:

```
upstream whisper.cpp releases vX.Y.Z
            |
            v  (detected within 24 h by daily sync job)
  sync.yml merges upstream into main, pushes tag vX.Y.Z
            |
            v  (tag push triggers build pipeline)
  build.yml builds all backend/OS combinations in parallel
            |
            v
  GitHub Release: "whisper.cpp vX.Y.Z — AMD Builds"
  with 13 artifacts across all backends and OS targets
```

**Every release ships up to 14 artifacts:**

```
whisper-{version}-linux-rocm-gfx1151.tar.gz
whisper-{version}-linux-rocm-gfx1150.tar.gz
whisper-{version}-linux-rocm-gfx120X.tar.gz
whisper-{version}-linux-rocm-gfx110X.tar.gz
whisper-{version}-windows-rocm-gfx1151.zip
whisper-{version}-windows-rocm-gfx1150.zip
whisper-{version}-windows-rocm-gfx120X.zip
whisper-{version}-windows-rocm-gfx110X.zip
whisper-{version}-linux-vulkan-x86_64.tar.gz
whisper-{version}-windows-vulkan-x64.zip
whisper-{version}-windows-npu-x64.zip         (may be absent if NPU runner offline)
whisper-{version}-linux-cpu-x86_64.tar.gz
whisper-{version}-windows-cpu-x64.zip
whisper-{version}-darwin-metal-arm64.tar.gz
```

> [!TIP]
> **Linux APU out of VRAM despite free memory (gfx1150 / gfx1151)?**
> Add `ttm.pages_limit=12582912` to your kernel command line (e.g. in GRUB), run `update-grub`, and reboot.
> See the [TheRock FAQ](https://github.com/ROCm/TheRock/blob/main/docs/faq.md#gfx1151-strix-halo-specific-questions) for details.

---

## 🖥️ Local Builds (Windows)

Reproduce any CI build locally using the bundled PowerShell script. Produces identical artifacts to what CI publishes.

```powershell
# Prerequisites: CMake, VS Build Tools 2022, 7-Zip, internet access

# CPU only (~2 min, no GPU needed)
.\scripts\local-build.ps1 -Backend cpu

# Vulkan — requires Vulkan SDK from https://vulkan.lunarg.com
.\scripts\local-build.ps1 -Backend vulkan

# ROCm for RDNA3 iGPU — downloads ROCm tarball (~2-4 GB, cached after first run)
.\scripts\local-build.ps1 -Backend rocm -GfxTarget gfx1151

# NPU — requires RyzenAI hardware + NPU driver >= .280
.\scripts\local-build.ps1 -Backend npu

# All backends, version-stamped artifacts placed in .\dist\
.\scripts\local-build.ps1 -Backend all -Version 1.8.4
```

---

## 📦 Dependencies

### Bundled in every release (no installation needed)

| Backend | What is included |
|---|---|
| ROCm | `amdhip64`, `rocblas`, `hipblaslt` + library data, LLVM runtime, all system deps; RPATH=`$ORIGIN` on Linux |
| Vulkan | SPIR-V shaders embedded at build time; links against system Vulkan loader |
| Metal | Uses macOS system Metal framework; no extra bundling needed |
| NPU | FlexML Runtime DLLs (`flexmlrt/bin` + `flexmlrt/lib`) |
| CPU | SDL2.dll included on Windows |

### Build-time only

| Tool | Purpose |
|---|---|
| [whisper.cpp](https://github.com/ggerganov/whisper.cpp) | Upstream source |
| [ROCm / TheRock](https://github.com/ROCm/TheRock) | HIP compiler + GPU runtime (tarball, not installed globally) |
| [FlexML Runtime](https://github.com/lemonade-sdk/whisper.cpp/releases/tag/deps) | VitisAI NPU inference |
| [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) | GLSL to SPIR-V shader compilation |
| [CMake >= 3.21](https://cmake.org/) | Build system |
| [Ninja](https://ninja-build.org/) | Fast build backend (ROCm builds) |
| [VS Build Tools 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) | Windows MSVC toolchain |

---

## 🏗️ Repository Structure

```
whisper.cpp-rocm/
├── .github/
│   └── workflows/
│       ├── build.yml           # All AMD backends — builds + publishes releases
│       └── sync.yml            # Daily upstream sync + auto-tagging
├── ci/
│   ├── resolve-rocm-version.sh    # Resolves AMD tarball URL for a given ROCm version
│   └── map-gpu-target.sh          # Maps gfx110X/gfx120X shorthands to specific arch lists
├── src/
│   └── vitisai/
│       ├── whisper-vitisai-encoder.h    # VitisAI NPU encoder C interface
│       └── whisper-vitisai-encoder.cpp  # FlexML runtime integration
├── scripts/
│   └── local-build.ps1         # Local Windows build script (mirrors CI jobs exactly)
├── ggml/                       # GGML library (all GPU backends live here)
├── src/                        # whisper.cpp source (VitisAI hooks added)
└── CMakeLists.txt              # Adds -DWHISPER_VITISAI option
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

whisper.cpp is copyright Georgi Gerganov and contributors — [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp).
ROCm is copyright Advanced Micro Devices, Inc.
VitisAI encoder copyright 2025 Advanced Micro Devices, Inc.
