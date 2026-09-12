#!/bin/bash

set -e

build_dir=whisper-build-install
install_dir=install

rm -rf ${build_dir} ${install_dir}

cmake -S ../../. -B ${build_dir} -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DWHISPER_BUILD_TESTS=OFF \
  -DWHISPER_BUILD_EXAMPLES=OFF \
  -DWHISPER_BUILD_SERVER=OFF \
  -DCMAKE_INSTALL_PREFIX="${PWD}/${install_dir}"

cmake --build ${build_dir} --parallel 12
cmake --install ${build_dir}
