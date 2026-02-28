# Docker Setup (whisper-server)

All commands below assume you are in the project root (the directory containing `whisper.cpp/`).

## Build the Docker image

### CUDA (GPU acceleration)

```bash
cd whisper.cpp
docker build -f .devops/main-cuda.Dockerfile \
  --build-arg CUDA_VERSION=12.8.0 \
  -t whisper-cuda:local .
```

Pass `--build-arg CUDA_VERSION=` matching your host CUDA driver version. Check with `nvidia-smi`. The container CUDA version must not exceed the host driver's supported version.

**Note:** The Dockerfile hardcodes `ENV CUDA_MAIN_VERSION=13.0`, setting `LD_LIBRARY_PATH` to `/usr/local/cuda-13.0/compat`. You must override this at runtime with `-e LD_LIBRARY_PATH=/usr/local/cuda-<YOUR_VERSION>/compat` to match the `--build-arg CUDA_VERSION` you used.

### CPU only (faster build)

```bash
cd whisper.cpp
docker build -f .devops/main.Dockerfile -t whisper-cpu:local .
```

## Download a model

```bash
cd whisper.cpp
sh models/download-ggml-model.sh base.en
```

## Start whisper-server

### CUDA

```bash
cd whisper.cpp
docker run -d --rm --name whisper-server \
  --gpus all \
  -v "$(pwd)/models:/models" \
  -p 8080:8080 \
  -e LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat \
  whisper-cuda:local \
  "whisper-server -m /models/ggml-base.en.bin --host 0.0.0.0 --port 8080"
```

Adjust `LD_LIBRARY_PATH` to match your CUDA version.

### CPU

```bash
cd whisper.cpp
docker run -d --rm --name whisper-server \
  -v "$(pwd)/models:/models" \
  -p 8080:8080 \
  whisper-cpu:local \
  "whisper-server -m /models/ggml-base.en.bin --host 0.0.0.0 --port 8080"
```

## Set environment variable

Add to your shell profile (`~/.zshrc` on macOS, `~/.bashrc` on Linux):

```bash
export WHISPER_CPP_URL=http://localhost:8080
```

This is the single source of truth for the server URL — always use `$WHISPER_CPP_URL`.

## Test

Use an absolute path for the audio file:

```bash
curl -F "file=@$(pwd)/whisper.cpp/samples/jfk.wav" "$WHISPER_CPP_URL/inference"
```

Expected output includes "ask not what your country can do for you".

## Stop

```bash
docker stop whisper-server
```

## Image Variants

| Dockerfile | Backend | Notes |
|------------|---------|-------|
| `main.Dockerfile` | CPU | No GPU needed, fastest build |
| `main-cuda.Dockerfile` | NVIDIA CUDA | Requires `--gpus all`, slow build |
| `main-vulkan.Dockerfile` | Vulkan GPU | AMD/Intel/NVIDIA via Vulkan |
