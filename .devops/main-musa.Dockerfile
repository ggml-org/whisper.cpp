ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG MUSA_VERSION=5.2.0
# Target the MUSA build image
ARG BASE_MUSA_DEV_CONTAINER=registry.mthreads.com/mcconline/musa_sdk:${MUSA_VERSION}-devel-ubuntu${UBUNTU_VERSION}-s5000
# Target the MUSA runtime image
ARG BASE_MUSA_RUN_CONTAINER=registry.mthreads.com/mcconline/musa_sdk:${MUSA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}-s5000

FROM ${BASE_MUSA_DEV_CONTAINER} AS build
WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential libsdl2-dev wget cmake git \
    musa-mualg-5-2 musa-muthrust-5-2 libmthreads-compute && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

COPY .. .
# Enable muBLAS
RUN --mount=type=secret,id=HF_TOKEN,required=false,env=HF_TOKEN make base.en CMAKE_ARGS="-DGGML_MUSA=1 -DGGML_NATIVE=OFF"

RUN find /app/build -name "*.o" -delete && \
    find /app/build -name "*.a" -delete && \
    rm -rf /app/build/CMakeFiles && \
    rm -rf /app/build/cmake_install.cmake && \
    rm -rf /app/build/_deps

FROM ${BASE_MUSA_RUN_CONTAINER} AS runtime
WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl ffmpeg wget cmake git libmthreads-compute && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

COPY --from=build /app/build/bin /app/build/bin
COPY --from=build /app/samples /app/samples
COPY --from=build /app/models /app/models

ENV PATH=/app/build/bin:$PATH
ENTRYPOINT [ "bash", "-c" ]
