## Pre-reqs:
Cmake 3.21
```
wget https://github.com/Kitware/CMake/releases/download/v3.21.7/cmake-3.21.7-linux-aarch64.sh
chmod +x cmake-3.21.7-linux-aarch64.sh
sudo mkdir /opt/cmake-3.21
sudo ./cmake-3.21.7-linux-aarch64.sh   --skip-license   --prefix=/opt/cmake-3.21
```

## Build Commands:

```
/opt/cmake-3.21/bin/cmake -S . -B build   -DWHISPER_CUDA=ON   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc   -DCMAKE_CUDA_ARCHITECTURES=53   -DGGML_NATIVE=OFF
/opt/cmake-3.21/bin/cmake --build build -j4
```
