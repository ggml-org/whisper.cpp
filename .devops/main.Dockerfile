FROM ubuntu:22.04 AS build
WORKDIR /app

RUN apt-get update && \
  apt-get install -y build-essential \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY .. .
RUN make

FROM ubuntu:22.04 AS runtime
WORKDIR /app

COPY --from=build /app /app
ENTRYPOINT [ "bash", "-c" ]
