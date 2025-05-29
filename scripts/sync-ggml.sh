#!/bin/bash

cp -rpv ../ggml/CMakeLists.txt       ./ggml/CMakeLists.txt
cp -rpv ../ggml/src/CMakeLists.txt   ./ggml/src/CMakeLists.txt

cp -rpv ../ggml/cmake/*              ./ggml/cmake/
cp -rpv ../ggml/src/ggml-cpu/cmake/* ./ggml/src/ggml-cpu/cmake/

cp -rpv ../ggml/src/ggml*.c        ./ggml/src/
cp -rpv ../ggml/src/ggml*.cpp      ./ggml/src/
cp -rpv ../ggml/src/ggml*.h        ./ggml/src/
cp -rpv ../ggml/src/gguf*.cpp      ./ggml/src/
cp -rpv ../ggml/src/ggml-blas/*    ./ggml/src/ggml-blas/
cp -rpv ../ggml/src/ggml-cann/*    ./ggml/src/ggml-cann/
cp -rpv ../ggml/src/ggml-cpu/*     ./ggml/src/ggml-cpu/
cp -rpv ../ggml/src/ggml-cuda/*    ./ggml/src/ggml-cuda/
cp -rpv ../ggml/src/ggml-hip/*     ./ggml/src/ggml-hip/
cp -rpv ../ggml/src/ggml-kompute/* ./ggml/src/ggml-kompute/
cp -rpv ../ggml/src/ggml-metal/*   ./ggml/src/ggml-metal/
cp -rpv ../ggml/src/ggml-musa/*    ./ggml/src/ggml-musa/
cp -rpv ../ggml/src/ggml-opencl/*  ./ggml/src/ggml-opencl/
cp -rpv ../ggml/src/ggml-rpc/*     ./ggml/src/ggml-rpc/
cp -rpv ../ggml/src/ggml-sycl/*    ./ggml/src/ggml-sycl/
cp -rpv ../ggml/src/ggml-vulkan/*  ./ggml/src/ggml-vulkan/

cp -rpv ../ggml/include/ggml*.h ./ggml/include/
cp -rpv ../ggml/include/gguf*.h ./ggml/include/

cp -rpv ../ggml/examples/common.h        ./examples/common.h
cp -rpv ../ggml/examples/common.cpp      ./examples/common.cpp
cp -rpv ../ggml/examples/common-ggml.h   ./examples/common-ggml.h
cp -rpv ../ggml/examples/common-ggml.cpp ./examples/common-ggml.cpp

cp -rpv ../ggml/LICENSE                ./LICENSE
cp -rpv ../ggml/scripts/gen-authors.sh ./scripts/gen-authors.sh
