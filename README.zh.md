# whisper.cpp

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<div align="center">

![whisper.cpp](https://user-images.githubusercontent.com/1991296/235238348-05d0f6a4-da44-4900-a1de-d0707e75b763.jpeg)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/whisper.cpp?filter=v*)](https://github.com/ggml-org/whisper.cpp/releases)
[![Actions Status](https://github.com/ggml-org/whisper.cpp/workflows/CI/badge.svg)](https://github.com/ggml-org/whisper.cpp/actions)
[![Conan Center](https://shields.io/conan/v/whisper-cpp)](https://conan.io/center/whisper-cpp)
[![npm](https://img.shields.io/npm/v/whisper.cpp.svg)](https://www.npmjs.com/package/whisper.cpp/)

</div>

高性能纯 C/C++ 实现的 [OpenAI Whisper](https://github.com/openai/whisper) 自动语音识别（ASR）模型推理引擎：

- 无外部依赖的原生纯 C/C++ 实现
- Apple Silicon 头等支持 —— 深度集成 ARM NEON、Accelerate 框架、Metal 与 [Core ML](#core-ml-支持)
- 针对 x86 架构的 AVX 指令集优化
- [针对 POWER 架构的 VSX 指令集优化](#power-vsx-指令集)
- 支持混合 F16 / F32 精度
- [支持整型量化（Integer Quantization）](#模型量化)
- 运行时零内存分配（Zero memory allocations at runtime）
- [支持 Vulkan GPU 加速](#vulkan-gpu-支持)
- 支持纯 CPU 推理
- [高效的 NVIDIA GPU 加速支持](#nvidia-gpu-支持)
- [AMD ROCm GPU 加速支持](#amd-rocm-gpu-支持)
- [AMD Ryzen AI NPU 加速支持](#amd-ryzen-ai-npu-支持)
- [Intel OpenVINO 加速支持](#openvino-支持)
- [华为昇腾（Ascend）NPU 支持](#华为昇腾-npu-支持)
- [摩尔线程（Moore Threads）GPU 支持](#摩尔线程-gpu-支持)
- 简洁明了的 [C 风格 API](https://github.com/ggml-org/whisper.cpp/blob/master/include/whisper.h)
- 内置[语音活动检测（VAD）](#语音活动检测vad)

支持的操作系统与平台：

- [x] Mac OS（Intel 与 Apple Silicon Arm）
- [x] [iOS](examples/whisper.objc)
- [x] [Android](examples/whisper.android)
- [x] [Java](bindings/java/README.md)
- [x] Linux / [FreeBSD](https://github.com/ggml-org/whisper.cpp/issues/56#issuecomment-1350920264)
- [x] [WebAssembly](examples/whisper.wasm)
- [x] Windows（[MSVC](https://github.com/ggml-org/whisper.cpp/blob/master/.github/workflows/build.yml#L117-L144) 与 [MinGW](https://github.com/ggml-org/whisper.cpp/issues/168)）
- [x] [树莓派（Raspberry Pi）](https://github.com/ggml-org/whisper.cpp/discussions/166)
- [x] [Docker](https://github.com/ggml-org/whisper.cpp/pkgs/container/whisper.cpp)

模型的全部高级实现逻辑均自包含于 [whisper.h](include/whisper.h) 与 [whisper.cpp](src/whisper.cpp) 中。
底层张量计算与神经网络核心算子归属于 [`ggml`](https://github.com/ggml-org/ggml) 机器学习张量库。

极致轻量级的设计使其能够极其便捷地集成到各类异构平台与端侧应用中。
例如，以下视频演示了在 iPhone 13 设备上完全离线、本地端侧运行 Whisper 模型的效果：[whisper.objc](examples/whisper.objc)

https://user-images.githubusercontent.com/1991296/197385372-962a6dea-bca1-4d50-bf96-1d8c27b98c81.mp4

也可以轻松构建专属的离线语音助手应用：[command](examples/command)

https://user-images.githubusercontent.com/1991296/204038393-2f846eae-c255-4099-a76d-5735c25c49da.mp4

在 Apple Silicon 芯片上，推理过程可通过 Metal 全程在 GPU 上加速运行：

https://github.com/ggml-org/whisper.cpp/assets/1991296/c82e8f86-60dc-49f2-b048-d2fdbd6b5225

## 快速上手

首先克隆代码仓库：

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
```

进入目录：

```
cd whisper.cpp
```

随后下载已转换为 [`ggml` 格式](#ggml-格式)的 Whisper [模型文件](models/README.md)。例如：

```bash
sh ./models/download-ggml-model.sh base.en
```

编译 [whisper-cli](examples/cli) 命令行工具并转录音频文件：

```bash
# 构建项目
cmake -B build
cmake --build build -j --config Release

# 转录音频文件
./build/bin/whisper-cli -f samples/jfk.wav
```

---

如需运行快速演示，只需执行 `make base.en`。

该命令会自动下载转换为 `ggml` 格式的 `base.en` 模型，并对 `samples` 目录下的全部 `.wav` 音频样本执行推理转录。

如需查看详细命令行参数说明，请执行：`./build/bin/whisper-cli -h`。

注意：当前 [whisper-cli](examples/cli) 工具仅支持 **16-bit WAV** 音频格式，运行前请务必完成格式转换。
可以使用 `ffmpeg` 进行标准化转码：

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

## 获取更多测试音频样本

如需获取更多测试音频样本，可执行：

```
make -j samples
```

该命令将从维基百科下载若干音频文件，并通过 `ffmpeg` 自动转码为 16-bit WAV 格式。

可通过如下命令下载并运行其他规格的模型：

```
make -j tiny.en
make -j tiny
make -j base.en
make -j base
make -j small.en
make -j small
make -j medium.en
make -j medium
make -j large-v1
make -j large-v2
make -j large-v3
make -j large-v3-turbo
```

## 显存与内存占用

| 模型规格 | 磁盘占用 | 内存占用 |
| :--- | :--- | :--- |
| tiny | 75 MiB | ~273 MB |
| base | 142 MiB | ~388 MB |
| small | 466 MiB | ~852 MB |
| medium | 1.5 GiB | ~2.1 GB |
| large | 2.9 GiB | ~3.9 GB |

## POWER VSX 指令集

`whisper.cpp` 支持 IBM POWER 架构，并包含深度优化指令，能够在运行 Linux 的 POWER9/10 处理器上大幅提升运行效率，在降频的 Raptor Talos II 设备上亦可实现快于实时的转录吞吐。确保系统中已安装 BLAS 相关软件包，并将标准 CMake 构建参数替换为：

```bash
# 定义 GGML_BLAS 进行编译构建
cmake -B build -DGGML_BLAS=1
cmake --build build -j --config Release
./build/bin/whisper-cli [ .. 其余参数 .. ]
```

## 模型量化

`whisper.cpp` 支持对 Whisper `ggml` 模型进行整型量化（Integer Quantization）。
量化后的模型不仅大幅缩减磁盘与内存占用，且在特定硬件架构上能够实现更高效的算力吞吐。

创建并使用量化模型的步骤如下：

```bash
# 编译构建量化工具
cmake -B build
cmake --build build -j --config Release

# 使用 Q5_0 量化方案对模型进行量化
./build/bin/quantize models/ggml-base.en.bin models/ggml-base.en-q5_0.bin q5_0

# 传入量化模型文件执行常规转录
./build/bin/whisper-cli -m models/ggml-base.en-q5_0.bin ./samples/gb0.wav
```

## Core ML 支持

在 Apple Silicon 芯片设备上，编码器（Encoder）推理可通过 Core ML 调度至 Apple 神经引擎（Apple Neural Engine, ANE）硬件单元执行。与纯 CPU 推理相比，这能带来极为显著的性能跃升（速度提升超 3 倍）。生成 Core ML 模型并在 `whisper.cpp` 中使用的指引如下：

- 安装生成 Core ML 模型所需的 Python 依赖库：

  ```bash
  pip install ane_transformers
  pip install openai-whisper
  pip install coremltools
  ```

  - 为确保 `coremltools` 正常工作，请确认系统已安装 [Xcode](https://developer.apple.com/xcode/) 并执行 `xcode-select --install` 安装命令行开发者工具。
  - 推荐使用 Python 3.11。
  - 推荐使用 macOS Sonoma（版本 14）或更新系统，老版本 macOS 在转录长文本时可能存在偶发幻觉现象。
  - 【可选建议】推荐使用如 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 等虚拟环境管理工具：
    - 创建虚拟环境：`conda create -n py311-whisper python=3.11 -y`
    - 激活虚拟环境：`conda activate py311-whisper`

- 导出生成 Core ML 模型。例如导出 `base.en` 模型：

  ```bash
  ./models/generate-coreml-model.sh base.en
  ```

  该操作将在目录下生成 `models/ggml-base.en-encoder.mlmodelc` 文件夹。

- 开启 Core ML 支持编译 `whisper.cpp`：

  ```bash
  # 使用 CMake 构建
  cmake -B build -DWHISPER_COREML=1
  cmake --build build -j --config Release
  ```

- 运行测试示例：

  ```text
  $ ./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav

  ...

  whisper_init_state: loading Core ML model from 'models/ggml-base.en-encoder.mlmodelc'
  whisper_init_state: first run on a device may take a while ...
  whisper_init_state: Core ML model loaded

  system_info: n_threads = 4 / 10 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 0 | VSX = 0 | COREML = 1 |

  ...
  ```

  首次在设备上运行时耗时较长，因为 ANE 守护服务需将 Core ML 编译为特定芯片架构的机器码缓存。后续运行将显著提速。

关于 Core ML 架构实现的更多技术细节，请参阅 PR [#566](https://github.com/ggml-org/whisper.cpp/pull/566)。

## ANEForge 支持

在 Apple Silicon 芯片上，编码器亦可通过 [ANEForge](https://github.com/sbryngelson/ANEForge) 直接调度 ANE 单元，绕过 Core ML 中间层开销。
在 `tiny` 至 `medium` 规格模型上，其运行速度相比 Core ML 编码器提升约 2 倍（详见[基准测试报告](https://github.com/sbryngelson/ANEForge/tree/main/bench/whisper_encoder_ane)）。
解码器（Decoder）逻辑保持不变，且无需特殊编译构建开关。

将编码器编译打包为 bundle（与当前机器及 OS 构建版本绑定）：

```bash
git clone https://github.com/sbryngelson/ANEForge && cd ANEForge
pip install -e ".[models]"
PYTHONPATH=. python3 bench/whisper_encoder_ane/export_bundle.py \
    --model openai/whisper-base --out /tmp/whisper-base-encoder
```

随后在运行 `whisper.cpp` 时指定环境变量：

```bash
export ANEFORGE_ENCODER=/tmp/whisper-base-encoder
export ANEFORGE_DYLIB=$PWD/aneforge/_lib/libane_e5rt_dispatch.dylib
./build/bin/whisper-cli -m models/ggml-base.bin -f samples/jfk.wav
```

关于 ANEForge 架构支持的更多详情，请参阅 PR [#3905](https://github.com/ggml-org/whisper.cpp/pull/3905)。

## OpenVINO 支持

在支持 [OpenVINO](https://github.com/openvinotoolkit/openvino) 的计算平台上，编码器推理可在 OpenVINO 支持的异构设备（涵盖 x86 CPU 以及 Intel 核显与独显）上执行。

这能大幅提高编码器计算吞吐。生成 OpenVINO 模型并配合 `whisper.cpp` 运行的指引如下：

- 首先配置 Python 虚拟环境并安装依赖（推荐 Python 3.10）：

  Windows 环境：

  ```powershell
  cd models
  python -m venv openvino_conv_env
  openvino_conv_env\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements-openvino.txt
  ```

  Linux 与 macOS 环境：

  ```bash
  cd models
  python3 -m venv openvino_conv_env
  source openvino_conv_env/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements-openvino.txt
  ```

- 导出生成 OpenVINO 编码器模型。例如导出 `base.en` 模型：

  ```
  python convert-whisper-to-openvino.py --model base.en
  ```

  这会生成 `ggml-base.en-encoder-openvino.xml` 与 `.bin` 中间表示（IR）模型文件。建议将其移动至与 `ggml` 模型相同的文件夹中，这是 OpenVINO 扩展模块在运行时默认检索的路径。

- 开启 OpenVINO 支持编译 `whisper.cpp`：

  从[官方发布页](https://github.com/openvinotoolkit/openvino/releases)下载 OpenVINO 工具包。推荐使用的版本为 [2026.3.0](https://github.com/openvinotoolkit/openvino/releases/tag/2026.3.0)。预编译的依赖库可在 [OpenVINO 软件归档中心](https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.3/)获取。

  下载解压至开发环境后，通过执行 `setupvars` 脚本配置环境变量：

  Linux：

  ```bash
  source /path/to/openvino_toolkit_ubuntu/setupvars.sh
  ```

  Windows（cmd）：

  ```powershell
  C:\Path\To\openvino_toolkit_windows\setupvars.bat
  ```

  随后使用 CMake 编译构建：

  ```bash
  cmake -B build -DWHISPER_OPENVINO=1
  cmake --build build -j --config Release
  ```

- 运行测试示例：

  ```text
  $ ./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav

  ...

  whisper_ctx_init_openvino_encoder: loading OpenVINO model from 'models/ggml-base.en-encoder-openvino.xml'
  whisper_ctx_init_openvino_encoder: first run on a device may take a while ...
  whisper_openvino_init: path_model = models/ggml-base.en-encoder-openvino.xml, device = GPU, cache_dir = models/ggml-base.en-encoder-openvino-cache
  whisper_ctx_init_openvino_encoder: OpenVINO model loaded

  system_info: n_threads = 4 / 8 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 | COREML = 0 | OPENVINO = 1 |

  ...
  ```

  在 OpenVINO 硬件设备上首次运行时由于需要编译 IR 模型生成特定设备的硬件二进制（blob），耗时稍长；该二进制文件会自动缓存以加速后续调用。

关于 OpenVINO 支持的技术实现，请参阅 PR [#1037](https://github.com/ggml-org/whisper.cpp/pull/1037)。

## AMD Ryzen™ AI NPU 支持

在配备专用 NPU 的 AMD Ryzen™ AI 300 与 400 系列处理器上，`whisper.cpp` 可通过 VitisAI 将 Whisper 编码器完全卸载至 NPU 硬件运行，相比纯 CPU 推理可带来显著加速。

### 前置环境

支持的操作系统：

- **Windows 11**
- **Linux**（Ubuntu 24.04 LTS，Python 3.12）

安装适合当前平台的 XRT 运行时与 FlexML 运行时：

- **XRT**：提供 NPU 内核驱动与 `xrt-smi` 诊断工具。Windows 下随 NPU 驱动自带；Linux 下请参考 [NPU 驱动安装手册](https://ryzenai.docs.amd.com/en/latest/linux.html#install-npu-drivers)单独安装。
- **FlexML 运行时**（`flexmlrt`）：whisper.cpp 使用的 VitisAI 推理后端引擎。可在 [FlexML 运行时发布页](https://github.com/lemonade-sdk/whisper.cpp-rocm/releases/tag/deps)下载。

安装完成后，在用于编译或运行 whisper.cpp 的每个终端 shell 中配置环境：

```bash
# Linux
source /opt/xilinx/xrt/setup.sh
source /path/to/flexmlrt/setup.sh
```

```cmd
:: Windows
cd /path/to/flexmlrt && call setup.bat
```

可通过如下命令检验 NPU 设备状态：

```bash
xrt-smi examine
```

### 模型下载

下载 ggml 模型及匹配的预编译 VitisAI 编码器缓存：

```bash
# Linux / macOS
sh ./models/download-ggml-model.sh base
sh ./models/download-vitisai-model.sh base
```

```cmd
:: Windows
.\models\download-ggml-model.cmd base
.\models\download-vitisai-model.cmd base
```

两个脚本中请传入相同的模型规格名称。如需查看所有可用的 VitisAI 编码器缓存：

```bash
sh ./models/download-vitisai-model.sh --list
```

```cmd
.\models\download-vitisai-model.cmd --list
```

VitisAI 下载脚本将从 [Hugging Face 上的 AMD Ryzen AI Whisper NPU 模型集](https://huggingface.co/collections/amd/ryzen-ai-whisper-npu-optimized-onnx-models)检索并将 `.rai` 编码器缓存下载至 `models/ggml-<model>-encoder-vitisai.rai`。

> 根据具体的 `.rai` 缓存，VitisAI 可能会单独卸载编码器，或同时卸载编码器与跨注意力投影层。whisper.cpp 会在运行时自动检测，并在模型初始化时输出所选的卸载模式日志。

### 编译构建

```bash
cmake -B build -DWHISPER_VITISAI=1
cmake --build build -j --config Release
```

### 运行

```bash
./build/bin/whisper-cli -m models/ggml-base.bin -f samples/jfk.wav
```

更多详细信息请参阅 [Ryzen AI 官方文档](https://ryzenai.docs.amd.com/en/latest/)。

## NVIDIA GPU 支持

配备 NVIDIA 独立显卡时，模型推理可通过 cuBLAS 及定制 CUDA 内核在 GPU 上高效运行。
首先请确保已安装 `cuda` 工具包：https://developer.nvidia.com/cuda-downloads

开启 CUDA 支持编译 `whisper.cpp`：

```bash
cmake -B build -DGGML_CUDA=1
cmake --build build -j --config Release
```

若针对较新架构的 NVIDIA GPU（如 RTX 5000 / 4000 系列）：

```bash
cmake -B build -DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build -j --config Release
```

## Vulkan GPU 支持

跨硬件厂商的统一 GPU 加速方案。
首先确认显卡驱动程序已支持 Vulkan API。

开启 Vulkan 支持编译 `whisper.cpp`：

```bash
cmake -B build -DGGML_VULKAN=1
cmake --build build -j --config Release
```

## AMD ROCm GPU 支持

配备 AMD 显卡时，可通过 HIP/ROCm 实现运算加速。
首先请确保已安装 [ROCm](https://rocm.docs.amd.com/en/latest/)。

开启 HIP 支持编译 `whisper.cpp`：

```bash
cmake -B build -DGGML_HIP=1 -DAMDGPU_TARGETS="gfx1201"
cmake --build build -j --config Release
```

请将 `gfx1201` 替换为当前显卡的具体架构代号。查询命令如下：

```bash
rocminfo | grep "gfx"
```

常见架构代号：`gfx1100`（RX 7900 XTX）、`gfx1101`（RX 7800 XT）、`gfx1201`（RX 9070 XT）。
若针对多款不同架构的显卡构建，可指定多目标：`-DAMDGPU_TARGETS="gfx1100;gfx1201"`。

## 基于 OpenBLAS 的 CPU BLAS 支持

编码器计算可通过 OpenBLAS 在 CPU 上获得加速。
首先确认系统已安装 `openblas`：https://www.openblas.net/

开启 OpenBLAS 支持编译构建：

```bash
cmake -B build -DGGML_BLAS=1
cmake --build build -j --config Release
```

## 华为昇腾 NPU 支持

华为昇腾（Huawei Ascend）NPU 通过 [`CANN`](https://www.hiascend.com/en/software/cann) 架构与 AI Core 硬件提供推理加速能力。

首先确认昇腾硬件设备支持列表：

**已验证设备列表**
| 华为昇腾 NPU 型号 | 支持状态 |
| :---: | :---: |
| Atlas 300T A2 | 支持 |
| Atlas 300I Duo | 支持 |

确保系统中已安装 [`CANN toolkit`](https://www.hiascend.com/en/software/cann/community)（推荐使用最新版 CANN）。

开启 CANN 支持编译 `whisper.cpp`：

```bash
cmake -B build -DGGML_CANN=1
cmake --build build -j --config Release
```

如常运行转录命令，例如：

```bash
./build/bin/whisper-cli -f samples/jfk.wav -m models/ggml-base.en.bin -t 8
```

*说明：*

- 若在华为昇腾设备上遇到问题，请在 Issue 标题中添加 **[CANN]** 前缀。
- 若昇腾新设备成功运行，欢迎提交 PR 协助更新“已验证设备列表”。

## 摩尔线程 GPU 支持

配备摩尔线程（Moore Threads）GPU 时，模型运算可通过 muBLAS 与定制 MUSA 内核在 GPU 上高效执行。
首先确保已安装 `MUSA SDK rc4.2.0`：https://developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=4.2.0

开启 MUSA 支持编译 `whisper.cpp`：

```bash
cmake -B build -DGGML_MUSA=1
cmake --build build -j --config Release
```

也可以为摩尔线程显卡显式指定架构代号。例如针对 MTT S80 GPU：

```bash
cmake -B build -DGGML_MUSA=1 -DMUSA_ARCHITECTURES="21"
cmake --build build -j --config Release
```

## FFmpeg 支持（仅限示例工程）

默认情况下，项目中的示例工具使用内置的 [miniaudio](https://github.com/mackron/miniaudio) 库进行音频解码。
部分示例工程亦可开启 FFmpeg 以扩展支持更广泛的音频容器格式。通过添加 `WHISPER_COMMON_FFMPEG` 标志即可启用。

首先安装相关依赖库：

```bash
# Debian/Ubuntu
sudo apt install libavcodec-dev libavformat-dev libavutil-dev

# RHEL/Fedora
sudo dnf install libavcodec-free-devel libavformat-free-devel libavutil-free-devel
```

随后执行编译构建：

```bash
cmake -B build -D WHISPER_COMMON_FFMPEG=yes
cmake --build build
```

运行如下命令验证转录功能：

```bash
# 将音频转换为 Opus 格式
ffmpeg -i samples/jfk.wav jfk.opus

# 转录 Opus 音频
./build/bin/whisper-cli --model models/ggml-base.en.bin --file jfk.opus
```

## Docker 容器化

### 前置条件

- 系统中必须安装并启动 Docker 服务。
- 创建用于持久化存储大模型及临时文件的目录（如 `/whisper/models`）。

### 预编译镜像

本项目提供多种官方预构建 Docker 镜像：

1. `ghcr.io/ggml-org/whisper.cpp:main`：包含核心可执行文件及 `curl`、`ffmpeg`（支持平台：`linux/amd64`、`linux/arm64`）。
2. `ghcr.io/ggml-org/whisper.cpp:main-cuda`：在 `main` 基础上集成 CUDA 加速支持（支持平台：`linux/amd64`）。
3. `ghcr.io/ggml-org/whisper.cpp:main-musa`：在 `main` 基础上集成 MUSA 加速支持（支持平台：`linux/amd64`）。
4. `ghcr.io/ggml-org/whisper.cpp:main-vulkan`：在 `main` 基础上集成 Vulkan 加速支持（支持平台：`linux/amd64`）。

### 容器运行示例

```shell
# 可选用 main 镜像标签或：cublas, main-cuda, main-intel, main-musa, main-rocm, main-vulkan。
IMAGE="ghcr.io/ggml-org/whisper.cpp:main"
MODEL_PATH="/tmp/whisper.cpp-models"
AUDIO_PATH="/tmp/whisper.cpp-audio"
AUDIO_URL="https://github.com/ggml-org/whisper.cpp/raw/refs/heads/master/samples/jfk.wav"
mkdir -p "$MODEL_PATH" "$AUDIO_PATH"
wget -O "$AUDIO_PATH/jfk.wav" "$AUDIO_URL"

# 下载模型并持久化保存至本地目录
docker run -it --rm \
  -v $MODEL_PATH:/models \
  $IMAGE \
  download-ggml-model.sh base /models

# 转录指定音频文件
docker run -it --rm \
  -v $MODEL_PATH:/models \
  -v $AUDIO_PATH:/audios \
  $IMAGE \
  whisper-cli -m /models/ggml-base.bin -f /audios/jfk.wav

# 启动 Web 推理服务端
docker run -it --rm \
  -p "8080:8080" \
  -v $MODEL_PATH:/models \
  $IMAGE \
  whisper-server --host 0.0.0.0 -m /models/ggml-base.bin
# 发起 HTTP 请求测试：
curl -v http://127.0.0.1:8080/inference \
  -F "file=@${AUDIO_PATH}/jfk.wav" \
  -F 'response_format=json'

# 下载 small.en 并使用 4 线程运行基准测速
docker run -it --rm \
  -v $MODEL_PATH:/models \
  $IMAGE \
  download-ggml-model.sh small.en /models
docker run -it --rm \
  -v $MODEL_PATH:/models \
  $IMAGE \
  whisper-bench -m /models/ggml-small.en.bin -t 4

# 上述示例默认基于 CPU 运算 —— 如需启用 GPU 加速，可通过设备挂载进行调度（例如基于 Vulkan 调度 AMD 核显）：
docker run --rm \
  --device /dev/dri \
  -v $MODEL_PATH:/models \
  "ghcr.io/ggml-org/whisper.cpp:main-vulkan" \
  whisper-bench -m /models/ggml-small.en.bin -t 4
```

## 通过 Conan 包管理器安装

可通过 [Conan](https://conan.io/) 安装预编译二进制库或直接从源码构建：

```
conan install --requires="whisper-cpp/[*]" --build=missing
```

关于 Conan 的详细使用文档，请参考 [Conan 官方文档](https://docs.conan.io/2/)。

## 局限性说明

- 仅支持前向推理（Inference only），不支持模型训练。

## 实时麦克风音频转录示例

该示例展示了如何对来自麦克风的实时音频流执行端到端连续转录。
[stream](examples/stream) 工具每隔半秒对麦克风音频采样一次并持续执行语音识别。
技术背景可参阅 [Issue #10](https://github.com/ggml-org/whisper.cpp/issues/10)。
运行该工具需确保系统已安装 [SDL2](https://wiki.libsdl.org/SDL2/Installation) 库。

```bash
cmake -B build -DWHISPER_SDL2=ON
cmake --build build -j --config Release
./build/bin/whisper-stream -m ./models/ggml-base.en.bin -t 8 --step 500 --length 5000
```

https://user-images.githubusercontent.com/1991296/194935793-76afede7-cfa8-48d8-a80f-28ba83be7d09.mp4

## 置信度彩色高亮显示

添加 `--print-colors` 参数后，终端将在输出转录文本时根据词级别置信度高低呈现不同颜色：

```bash
./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/gb0.wav --print-colors
```

<img width="965" alt="image" src="https://user-images.githubusercontent.com/1991296/197356445-311c8643-9397-4e5e-b46e-0b4b4daa2530.png">

## 控制生成文本段落长度（实验性特性）

例如，若需将单行文本字符数限制在最多 16 个字符以内，只需传入 `-ml 16` 参数：

```text
$ ./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/jfk.wav -ml 16

whisper_model_load: loading model from './models/ggml-base.en.bin'
...
system_info: n_threads = 4 / 10 | AVX2 = 0 | AVX512 = 0 | NEON = 1 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 |

main: processing './samples/jfk.wav' (176000 samples, 11.0 sec), 4 threads, 1 processors, lang = en, task = transcribe, timestamps = 1 ...

[00:00:00.000 --> 00:00:00.850]   And so my
[00:00:00.850 --> 00:00:01.590]   fellow
[00:00:01.590 --> 00:00:04.140]   Americans, ask
[00:00:04.140 --> 00:00:05.660]   not what your
[00:00:05.660 --> 00:00:06.840]   country can do
[00:00:06.840 --> 00:00:08.430]   for you, ask
[00:00:08.430 --> 00:00:09.440]   what you can do
[00:00:09.440 --> 00:00:10.020]   for your
[00:00:10.020 --> 00:00:11.000]   country.
```

## 词级别时间戳（实验性特性）

通过 `--max-len` 参数可以获取精准的单词级时间戳（Word-level timestamps），只需设置 `-ml 1`：

```text
$ ./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/jfk.wav -ml 1

whisper_model_load: loading model from './models/ggml-base.en.bin'
...
system_info: n_threads = 4 / 10 | AVX2 = 0 | AVX512 = 0 | NEON = 1 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 |

main: processing './samples/jfk.wav' (176000 samples, 11.0 sec), 4 threads, 1 processors, lang = en, task = transcribe, timestamps = 1 ...

[00:00:00.000 --> 00:00:00.320]
[00:00:00.320 --> 00:00:00.370]   And
[00:00:00.370 --> 00:00:00.690]   so
[00:00:00.690 --> 00:00:00.850]   my
[00:00:00.850 --> 00:00:01.590]   fellow
[00:00:01.590 --> 00:00:02.850]   Americans
[00:00:02.850 --> 00:00:03.300]  ,
[00:00:03.300 --> 00:00:04.140]   ask
[00:00:04.140 --> 00:00:04.990]   not
[00:00:04.990 --> 00:00:05.410]   what
[00:00:05.410 --> 00:00:05.660]   your
[00:00:05.660 --> 00:00:06.260]   country
[00:00:06.260 --> 00:00:06.600]   can
[00:00:06.600 --> 00:00:06.840]   do
[00:00:06.840 --> 00:00:07.010]   for
[00:00:07.010 --> 00:00:08.170]   you
[00:00:08.170 --> 00:00:08.190]  ,
[00:00:08.190 --> 00:00:08.430]   ask
[00:00:08.430 --> 00:00:08.910]   what
[00:00:08.910 --> 00:00:09.040]   you
[00:00:09.040 --> 00:00:09.320]   can
[00:00:09.320 --> 00:00:09.440]   do
[00:00:09.440 --> 00:00:09.760]   for
[00:00:09.760 --> 00:00:10.020]   your
[00:00:10.020 --> 00:00:10.510]   country
[00:00:10.510 --> 00:00:11.000]  .
```

## 基于 tinydiarize 的说话人分离（实验性特性）

关于该方案的更多详细设计，请参阅：https://github.com/ggml-org/whisper.cpp/pull/1058

使用示例：

```bash
# 下载兼容 tinydiarize 的专用模型
./models/download-ggml-model.sh small.en-tdrz

# 运行时附加 "-tdrz" 参数
./build/bin/whisper-cli -f ./samples/a13.wav -m ./models/ggml-small.en-tdrz.bin -tdrz
...
main: processing './samples/a13.wav' (480000 samples, 30.0 sec), 4 threads, 1 processors, lang = en, task = transcribe, tdrz = 1, timestamps = 1 ...
...
[00:00:00.000 --> 00:00:03.800]   Okay Houston, we've had a problem here. [SPEAKER_TURN]
[00:00:03.800 --> 00:00:06.200]   This is Houston. Say again please. [SPEAKER_TURN]
[00:00:06.200 --> 00:00:08.260]   Uh Houston we've had a problem.
[00:00:08.260 --> 00:00:11.320]   We've had a main beam up on a volt. [SPEAKER_TURN]
[00:00:11.320 --> 00:00:13.820]   Roger main beam interval. [SPEAKER_TURN]
[00:00:13.820 --> 00:00:15.100]   Uh uh [SPEAKER_TURN]
[00:00:15.100 --> 00:00:18.020]   So okay stand, by thirteen we're looking at it. [SPEAKER_TURN]
[00:00:18.020 --> 00:00:25.740]   Okay uh right now uh Houston the uh voltage is uh is looking good um.
[00:00:27.620 --> 00:00:29.940]   And we had a a pretty large bank or so.
```

## 卡拉 OK 风格字幕视频生成（实验性特性）

[whisper-cli](examples/cli) 支持生成卡拉 OK 风格的高亮字幕视频（随发音逐词高亮）。传入 `-owts` 参数并运行生成的 bash 脚本即可完成渲染（需安装 `ffmpeg`）。

典型运行流程示例：

```bash
./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/jfk.wav -owts
source ./samples/jfk.wav.wts
ffplay ./samples/jfk.wav.mp4
```

https://user-images.githubusercontent.com/1991296/199337465-dbee4b5e-9aeb-48a3-b1c6-323ac4db5b2c.mp4

---

```bash
./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/mm0.wav -owts
source ./samples/mm0.wav.wts
ffplay ./samples/mm0.wav.mp4
```

https://user-images.githubusercontent.com/1991296/199337504-cc8fd233-0cb7-4920-95f9-4227de3570aa.mp4

---

```bash
./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/gb0.wav -owts
source ./samples/gb0.wav.wts
ffplay ./samples/gb0.wav.mp4
```

https://user-images.githubusercontent.com/1991296/199337538-b7b0c7a3-2753-4a88-a0cd-f28a317987ba.mp4

---

## 各规格模型转录效果视频对比

可使用 [scripts/bench-wts.sh](https://github.com/ggml-org/whisper.cpp/blob/master/scripts/bench-wts.sh) 脚本生成全模型效果对比视频：

```bash
./scripts/bench-wts.sh samples/jfk.wav
ffplay ./samples/jfk.wav.all.mp4
```

https://user-images.githubusercontent.com/1991296/223206245-2d36d903-cf8e-4f09-8c3b-eb9f9c39d6fc.mp4

---

## 基准测试（Benchmarks）

为客观评测不同硬件与配置下的推理算力，提供了专用的 [whisper-bench](examples/bench) 性能测试工具。该工具运行编码器（Encoder）并统计计算执行耗时。评测结果汇总于如下 Issue 中：

[基准测试结果汇总（Benchmark results）](https://github.com/ggml-org/whisper.cpp/issues/89)

此外，仓库还提供了使用不同模型与音频文件批量运行评测的 Python 脚本 [bench.py](scripts/bench.py)。

可通过如下命令运行基准测试（默认会对 `models` 目录下的所有标准模型进行测试）：

```bash
python3 scripts/bench.py -f samples/jfk.wav -t 2,4,8 -p 1,2
```

该脚本设计便于针对不同基准测试场景二次修改与扩展，测试完成后将输出包含测试数据的 CSV 报表。

## `ggml` 格式说明

原始模型需转换为专用的二进制格式。这种格式将推理所需的全部核心资产整合打包为单个文件：

- 模型超参数（Model parameters）
- 梅尔倒谱滤波器组（Mel filters）
- 词表数据（Vocabulary）
- 权重张量（Weights）

可直接通过 [models/download-ggml-model.sh](models/download-ggml-model.sh) 脚本下载转换完毕的模型，或通过 Hugging Face 仓库手动下载：

- https://huggingface.co/ggerganov/whisper.cpp

更多转换细节请参阅 [models/convert-pt-to-ggml.py](models/convert-pt-to-ggml.py) 或 [models/README.md](models/README.md)。

## [多语言绑定与生态集成（Bindings）](https://github.com/ggml-org/whisper.cpp/discussions/categories/bindings)

- [x] Rust: [tazz4843/whisper-rs](https://github.com/tazz4843/whisper-rs) | [#310](https://github.com/ggml-org/whisper.cpp/discussions/310)
- [x] JavaScript: [bindings/javascript](bindings/javascript) | [#309](https://github.com/ggml-org/whisper.cpp/discussions/309)
  - React Native（iOS / Android）：[whisper.rn](https://github.com/mybigday/whisper.rn)
- [x] Go: [bindings/go](bindings/go) | [#312](https://github.com/ggml-org/whisper.cpp/discussions/312)
- [x] Java:
  - [GiviMAD/whisper-jni](https://github.com/GiviMAD/whisper-jni)
- [x] Ruby: [bindings/ruby](bindings/ruby) | [#507](https://github.com/ggml-org/whisper.cpp/discussions/507)
- [x] Objective-C / Swift: [ggml-org/whisper.spm](https://github.com/ggml-org/whisper.spm) | [#313](https://github.com/ggml-org/whisper.cpp/discussions/313)
  - [exPHAT/SwiftWhisper](https://github.com/exPHAT/SwiftWhisper)
- [x] .NET: | [#422](https://github.com/ggml-org/whisper.cpp/discussions/422)
  - [sandrohanea/whisper.net](https://github.com/sandrohanea/whisper.net)
  - [NickDarvey/whisper](https://github.com/NickDarvey/whisper)
- [x] Python: | [#9](https://github.com/ggml-org/whisper.cpp/issues/9)
  - [stlukey/whispercpp.py](https://github.com/stlukey/whispercpp.py)（Cython）
  - [AIWintermuteAI/whispercpp](https://github.com/AIWintermuteAI/whispercpp)（更新维护版 fork）
  - [aarnphm/whispercpp](https://github.com/aarnphm/whispercpp)（Pybind11）
  - [abdeladim-s/pywhispercpp](https://github.com/abdeladim-s/pywhispercpp)（Pybind11）
- [x] R: [bnosac/audio.whisper](https://github.com/bnosac/audio.whisper)
- [x] Unity: [macoron/whisper.unity](https://github.com/Macoron/whisper.unity)

## XCFramework

XCFramework 是适用于 iOS、visionOS、tvOS 和 macOS 的预编译二进制框架。它可直接在 Swift 工程中引入使用，而无需从源码手动编译底层 C/C++ 库。以 v1.7.5 版本的 XCFramework 为例，可通过 Swift Package Manager 如下引入：

```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Whisper",
    targets: [
        .executableTarget(
            name: "Whisper",
            dependencies: [
                "WhisperFramework"
            ]),
        .binaryTarget(
            name: "WhisperFramework",
            url: "https://github.com/ggml-org/whisper.cpp/releases/download/v1.7.5/whisper-v1.7.5-xcframework.zip",
            checksum: "c7faeb328620d6012e130f3d705c51a6ea6c995605f2df50f6e1ad68c59c6c4a"
        )
    ]
)
```

## 语音活动检测（VAD）

在 `whisper-cli` 中可通过 `--vad` 参数开启语音活动检测（Voice Activity Detection, VAD）支持。使用该功能需同时传入相应的 VAD 模型文件。

其运行机制如下：输入的原始音频首先传入 VAD 模型检测其中的语音活动片段；识别出语音边界后，系统仅提取含有人声的音频片段传递给 Whisper 进行深度语音转录。该流程大幅降低了 Whisper 实际需要处理的音频总量，能够显著提升整体转录效率并抑制静音阶段的模型幻觉。

目前支持如下 VAD 模型：

### Silero-VAD

[Silero-vad](https://github.com/snakers4/silero-vad) 是一款基于 Python 开发的轻量级、高精度 VAD 算法模型。

在 Linux 或 macOS 上可通过如下命令下载模型：

```console
$ ./models/download-vad-model.sh silero-v6.2.0
Downloading ggml model silero-v6.2.0 from 'https://huggingface.co/ggml-org/whisper-vad' ...
ggml-silero-v6.2.0.bin        100%[==============================================>] 864.35K  --.-KB/s    in 0.04s
Done! Model 'silero-v6.2.0' saved in '/path/models/ggml-silero-v6.2.0.bin'
You can now use it like this:

  $ ./build/bin/whisper-cli -vm /path/models/ggml-silero-v6.2.0.bin --vad -f samples/jfk.wav -m models/ggml-base.en.bin

```

Windows 系统下的下载命令如下：

```console
> .\models\download-vad-model.cmd silero-v6.2.0
Downloading vad model silero-v6.2.0...
Done! Model silero-v6.2.0 saved in C:\Users\danie\work\ai\whisper.cpp\ggml-silero-v6.2.0.bin
You can now use it like this:

C:\path\build\bin\Release\whisper-cli.exe -vm C:\path\ggml-silero-v6.2.0.bin --vad -m models/ggml-base.en.bin -f samples\jfk.wav

```

若不带任何参数运行上述脚本，可列出所有支持下载的 VAD 模型清单。

此外，亦可通过如下脚本将 Silero-VAD 手动转换为 ggml 格式：

```console
$ python3 -m venv venv && source venv/bin/activate
$ (venv) pip install silero-vad
$ (venv) $ python models/convert-silero-vad-to-ggml.py --output models/silero.bin
Saving GGML Silero-VAD model to models/silero-v6.2.0-ggml.bin
```

随后配合 whisper 执行转录：

```console
$ ./build/bin/whisper-cli \
   --file ./samples/jfk.wav \
   --model ./models/ggml-base.en.bin \
   --vad \
   --vad-model ./models/silero-v6.2.0-ggml.bin
```

### VAD 配置参数详解

* `--vad-threshold`：语音检测概率阈值。语音片段/帧的置信度概率高于此阈值时判定为存在有效语音。
* `--vad-min-speech-duration-ms`：最小有效语音时长（毫秒）。短于该阈值的语音片段将被丢弃，以过滤短暂的噪音冲击或误报。
* `--vad-min-silence-duration-ms`：最小静音判停时长（毫秒）。静音持续时间必须至少达到该值才被视作当前语音片段结束；更短的停顿将被忽略并作为语音的一部分连续处理。
* `--vad-max-speech-duration-s`：最大语音切片时长（秒）。超过该长度的语音片段将在静音点（停顿大于 98ms 处）自动拆分为多个子片段，防止生成过长片段。
* `--vad-speech-pad-ms`：语音边界填充缓冲时长（毫秒）。在每个检测出的有效语音片段首尾各扩展填充该时长，避免语音首尾辅音被机械截断。
* `--vad-samples-overlap`：音频切片之间的重叠步长（秒，如 0.10 代表 100ms 重叠）。确保各片段拼接转录时边缘语音平滑过渡。

## 示例应用（Examples）

代码库在 [examples](examples) 目录下提供了丰富的示范应用，部分示例已通过 WebAssembly 移植至浏览器端运行：

| 示例工程 | Web 在线演示 | 功能描述 |
| :--- | :--- | :--- |
| [whisper-cli](examples/cli) | [whisper.wasm](examples/whisper.wasm) | 使用 Whisper 转录与翻译音频的轻量命令行工具 |
| [whisper-bench](examples/bench) | [bench.wasm](examples/bench.wasm) | 评测当前机器运行 Whisper 的性能基准工具 |
| [whisper-stream](examples/stream) | [stream.wasm](examples/stream.wasm) | 麦克风原始音频输入的实时流式转录应用 |
| [whisper-command](examples/command) | [command.wasm](examples/command.wasm) | 基于麦克风输入接收语音指令的轻量语音助手 |
| [whisper-server](examples/server) | | 兼容 OpenAI 风格 API 的 HTTP 音频转录服务端 |
| [whisper-talk-llama](examples/talk-llama) | | 与 LLaMA 大语言模型进行语音对话的端到端范例 |
| [whisper.objc](examples/whisper.objc) | | 基于 whisper.cpp 的原生 iOS 移动端应用程序 |
| [whisper.swiftui](examples/whisper.swiftui) | | 基于 SwiftUI 的现代化 iOS / macOS 语音转录应用 |
| [whisper.android](examples/whisper.android) | | 基于 whisper.cpp 的原生 Android 移动端应用 |
| [whisper.nvim](examples/whisper.nvim) | | 适用于 Neovim 编辑器的语音转文字生产力插件 |
| [generate-karaoke.sh](examples/generate-karaoke.sh) | | 快速[生成卡拉 OK 风格字幕视频](https://youtu.be/uj7hVta4blM)的辅助脚本 |
| [livestream.sh](examples/livestream.sh) | | [直播流音频实时转录](https://github.com/ggml-org/whisper.cpp/issues/185)脚本 |
| [yt-wsp.sh](examples/yt-wsp.sh) | | 自动下载、转录并翻译各类网络音视频点播内容的辅助脚本（[原版参考](https://gist.github.com/DaniruKun/96f763ec1a037cc92fe1a059b643b818)） |
| [wchess](examples/wchess) | [wchess.wasm](examples/wchess) | 基于纯语音指令控制下棋的国际象棋游戏 |

## [技术讨论区（Discussions）](https://github.com/ggml-org/whisper.cpp/discussions)

若对项目有任何反馈或建议，欢迎前往 Discussions 专区发帖交流。
可以在 [Show and tell](https://github.com/ggml-org/whisper.cpp/discussions/categories/show-and-tell) 专栏分享基于 `whisper.cpp` 构建的衍生开源项目。若遇到使用疑问，推荐先查阅[常见问题解答专帖（FAQ #126）](https://github.com/ggml-org/whisper.cpp/discussions/126)。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年11月16日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
