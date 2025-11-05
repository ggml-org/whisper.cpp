#!/bin/bash

cp -rpv ../ggml/CMakeLists.txt       ./ggml/CMakeLists.txt
cp -rpv ../ggml/src/CMakeLists.txt   ./ggml/src/CMakeLists.txt

cp -rpv ../ggml/cmake/*              ./ggml/cmake/
cp -rpv ../ggml/src/ggml-cpu/cmake/* ./ggml/src/ggml-cpu/cmake/

cp -rpv ../ggml/src/ggml* ./ggml/src/

cp -rpv ../ggml/include/ggml*.h ./ggml/include/
cp -rpv ../ggml/include/gguf*.h ./ggml/include/

cp -rpv ../ggml/examples/common.h        ./examples/common.h
cp -rpv ../ggml/examples/common.cpp      ./examples/common.cpp
cp -rpv ../ggml/examples/common-ggml.h   ./examples/common-ggml.h
cp -rpv ../ggml/examples/common-ggml.cpp ./examples/common-ggml.cpp

cp -rpv ../ggml/LICENSE                ./LICENSE
cp -rpv ../ggml/scripts/gen-authors.sh ./scripts/gen-authors.sh
