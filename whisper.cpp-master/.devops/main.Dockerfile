FROM ubuntu:22.04 AS build
WORKDIR /app

RUN apt-get update && \
  apt-get install -y build-essential wget cmake git \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY .. .
RUN make base.en

FROM ubuntu:22.04 AS runtime
WORKDIR /app

RUN apt-get update && \
  apt-get install -y curl ffmpeg libsdl2-dev wget cmake git \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /app /app
ENV PATH=/app/build/bin:$PATH
ENTRYPOINT [ "bash", "-c" ]
