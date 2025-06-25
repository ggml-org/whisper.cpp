ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=12.3.1
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# Target the CUDA runtime image
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build
WORKDIR /app

# Unless otherwise specified, we make a fat build.
ARG CUDA_DOCKER_ARCH=all
# Set nvcc architecture
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}

RUN apt-get update && \
    apt-get install -y build-essential libsdl2-dev wget cmake git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Ref: https://stackoverflow.com/a/53464012
ENV CUDA_MAIN_VERSION=12.3
ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA_MAIN_VERSION}/compat:$LD_LIBRARY_PATH

COPY .. .
# Enable cuBLAS
RUN make base.en CMAKE_ARGS="-DGGML_CUDA=1"

RUN find /app/build -name "*.o" -delete && \
    find /app/build -name "*.a" -delete && \
    rm -rf /app/build/CMakeFiles && \
    rm -rf /app/build/cmake_install.cmake && \
    rm -rf /app/build/_deps

FROM ${BASE_CUDA_RUN_CONTAINER} AS runtime
ENV CUDA_MAIN_VERSION=12.3
ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA_MAIN_VERSION}/compat:$LD_LIBRARY_PATH
WORKDIR /app

RUN apt-get update && \
  apt-get install -y curl ffmpeg wget cmake git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /app /app
RUN du -sh /app/*
RUN find /app -type f -size +100M
ENV PATH=/app/build/bin:$PATH
ENTRYPOINT [ "bash", "-c" ]
