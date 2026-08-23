# syntax=docker/dockerfile:1.7

# llmxabe as a container image.
#
# The runtime base is `nvidia/cuda:*-runtime`, not `*-devel`, because nothing
# here wants nvcc: `xabe-cuda` has no build script, and cudarc is built with
# `fallback-dynamic-loading` and `nvrtc`, so it resolves the driver at run
# time and compiles the kernels with NVRTC on the first launch. The three
# libraries that must exist at run time are `libcuda.so.1` — injected by the
# NVIDIA container runtime, never shipped in an image — plus `libnvrtc.so.12`
# and `libcublasLt.so.12`, both of which the runtime base carries.
#
# The builder stage uses that same base so the binary links against the glibc
# it will run on. Swapping it for a Debian `rust:` image is the obvious
# "simplification" and it produces a binary that will not start here.

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG BASE=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# --- build ------------------------------------------------------------------

FROM ${BASE} AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

# `--default-toolchain none` is deliberate: the toolchain is whatever
# rust-toolchain.toml pins. One pin, honoured everywhere.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain none

WORKDIR /src

# Install that pinned toolchain explicitly, from the pin alone and before the
# source arrives. Leaving it to rustup's implicit auto-install on the first
# `cargo` call builds here and fails elsewhere: auto-install is off under
# RUSTUP_AUTO_INSTALL=0, it has varied across rustup versions, and where it
# does not fire the build dies with "rustup could not choose a version of
# cargo to run, because one wasn't specified explicitly, and no default is
# configured" — which reads like a missing toolchain file rather than what it
# is. The no-argument form reads rust-toolchain.toml, so the channel and its
# components stay stated once. It is also a layer no source edit invalidates.
COPY rust-toolchain.toml ./
RUN rustup toolchain install

COPY . .

# The release profile carries `debug = 1` so local profiling has symbols. An
# image never opens a profiler and pays for them in size, so drop them; the
# optimization settings are untouched. CI does the same.
ENV CARGO_PROFILE_RELEASE_DEBUG=0

# The cache mounts make a rebuild after a source edit cheap, but they are not
# part of the layer — so the binary is copied out of the mount inside the same
# RUN that produces it.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    cargo build --release --locked -p xabe-server \
    && install -Dm755 target/release/llmxabe /out/llmxabe

# --- serve ------------------------------------------------------------------

FROM ${BASE}

ARG DEBIAN_FRONTEND=noninteractive
# curl is here for the compose healthcheck and nothing else.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/llmxabe /usr/local/bin/llmxabe

# Bind the container's own interface, not its loopback: the port is published
# by the runtime, and 127.0.0.1 here would be reachable from nothing.
ENV LLMXABE_HOST=0.0.0.0 \
    LLMXABE_PORT=8000

# What the NVIDIA container runtime should inject. `compute` is CUDA itself;
# `utility` brings nvidia-smi, which is what you reach for when a worker fails
# to bind a card.
#
# `all` here only decides what a bare `docker run --runtime=nvidia` sees. An
# explicit device request wins over it — `--gpus '"device=0"'`, or compose's
# `device_ids`, leaves exactly one card visible — so subsetting the fleet
# works as documented rather than being silently widened back to all three.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000

# One worker per visible device, so the image serves whatever the runtime
# gives it. Arguments append to this — see the `command:` in
# docker-compose.yml for the configuration this project measured.
ENTRYPOINT ["llmxabe"]
