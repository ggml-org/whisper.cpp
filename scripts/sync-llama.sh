#!/bin/bash

cp -rpv ../llama.cpp/include/llama.h ./examples/talk-llama/llama.h

cp -rpv ../llama.cpp/src/llama*.cpp       ./examples/talk-llama/
cp -rpv ../llama.cpp/src/llama*.h         ./examples/talk-llama/
cp -rpv ../llama.cpp/src/unicode.h        ./examples/talk-llama/unicode.h
cp -rpv ../llama.cpp/src/unicode.cpp      ./examples/talk-llama/unicode.cpp
cp -rpv ../llama.cpp/src/unicode-data.h   ./examples/talk-llama/unicode-data.h
cp -rpv ../llama.cpp/src/unicode-data.cpp ./examples/talk-llama/unicode-data.cpp
