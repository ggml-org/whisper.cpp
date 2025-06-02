FROM ubuntu:22.04 AS build
WORKDIR /app

# Set non-interactive frontend and install ca-certificates first
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    sed -i "s|http://archive.ubuntu.com|https://archive.ubuntu.com|g" /etc/apt/sources.list && \
    sed -i "s|http://security.ubuntu.com|https://security.ubuntu.com|g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y build-essential wget cmake git \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY . .
RUN make base.en

FROM ubuntu:22.04 AS runtime
WORKDIR /app

# Set non-interactive frontend and install ca-certificates first
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    sed -i "s|http://archive.ubuntu.com|https://archive.ubuntu.com|g" /etc/apt/sources.list && \
    sed -i "s|http://security.ubuntu.com|https://security.ubuntu.com|g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y curl ffmpeg libsdl2-dev wget cmake git \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /app /app
ENV PATH=/app/build/bin:$PATH
ENTRYPOINT [ "bash", "-c" ]
