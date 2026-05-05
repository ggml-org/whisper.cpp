FROM ubuntu:24.04 AS build
WORKDIR /app

RUN apt-get update && \
  apt-get install -y build-essential wget cmake git \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY . .

RUN cmake -B build -DGGML_NATIVE=OFF -DWHISPER_BUILD_TESTS=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  && cmake --build build --config Release -j $(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

FROM ubuntu:24.04 AS runtime
WORKDIR /app

RUN apt-get update \
  && apt-get install -y curl ffmpeg wget \
  && apt autoremove -y \
  && apt clean -y \
  && rm -rf /tmp/* /var/tmp/* \
  && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
  && find /var/cache -type f -delete

COPY --from=build /app/build/bin /app
COPY --from=build /app/lib /app
COPY --from=build /app/models /app/models

ENV PATH=/app:$PATH
ENTRYPOINT [ "bash", "-c" ]
