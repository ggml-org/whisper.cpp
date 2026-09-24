FROM ubuntu:24.04 AS build
WORKDIR /app

RUN apt update && \
  apt install --no-install-recommends -y build-essential ca-certificates cmake git glslc libvulkan-dev spirv-headers wget \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY .. .
RUN --mount=type=secret,id=HF_TOKEN,required=false,env=HF_TOKEN make base.en CMAKE_ARGS="-DGGML_VULKAN=1"
# Copy only CLI executables and shared libraries (not compiled tests, nor source code or other build artifacts)
RUN mkdir -p /runtime/usr/local/bin /runtime/usr/local/lib && \
    for path in build/bin/*; do \
      name="${path##*/}"; \
      case "$name" in \
        *.so|*.so.*) cp -a "$path" /runtime/usr/local/lib/ ;; \
        test-*|main|bench) ;; \
        *) cp -a "$path" /runtime/usr/local/bin/ ;; \
      esac; \
    done


FROM ubuntu:24.04 AS runtime
WORKDIR /app

RUN apt update && \
  apt install --no-install-recommends -y \
    ca-certificates curl ffmpeg libvulkan1 mesa-vulkan-drivers \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /runtime/ /
COPY --from=build /app/models/download-* /usr/local/bin/
RUN ldconfig
