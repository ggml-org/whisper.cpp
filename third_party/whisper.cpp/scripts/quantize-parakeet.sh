#!/bin/bash

set -e

build_dir=build
modelname=ggml-parakeet-tdt-0.6b-v3
model=models/${modelname}-f32.bin
cmd=parakeet-quantize

cmake --build ${build_dir} --target $cmd -j 12

${build_dir}/bin/${cmd} $model models/${modelname}-q8_0.bin q8_0
${build_dir}/bin/${cmd} $model models/${modelname}-q4_0.bin q4_0
${build_dir}/bin/${cmd} $model models/${modelname}-q4_k.bin q4_k
${build_dir}/bin/${cmd} $model models/${modelname}-q2_k.bin q2_k
