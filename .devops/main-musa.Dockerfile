ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG MUSA_VERSION=rc4.0.1
# Target the MUSA build image
ARG BASE_MUSA_DEV_CONTAINER=mthreads/musa:${MUSA_VERSION}-mudnn-devel-ubuntu${UBUNTU_VERSION}
# Target the MUSA runtime image
ARG BASE_MUSA_RUN_CONTAINER=mthreads/musa:${MUSA_VERSION}-mudnn-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_MUSA_DEV_CONTAINER} AS build
WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential libsdl2-dev wget cmake git \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY .. .
# Enable muBLAS
RUN make base.en CMAKE_ARGS="-DGGML_MUSA=1"

FROM ${BASE_MUSA_RUN_CONTAINER} AS runtime
WORKDIR /app

RUN apt-get update && \
  apt-get install -y curl ffmpeg wget cmake git \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /app /app
ENV PATH=/app/build/bin:$PATH
ENTRYPOINT [ "bash", "-c" ]
