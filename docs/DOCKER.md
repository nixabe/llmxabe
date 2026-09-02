# Running in Docker

`docker-compose.yml` and `Dockerfile` at the repository root build and serve
the engine in a container. This is a deployment convenience, not a development
environment: the image contains the `llmxabe` server binary and nothing else —
no benches, no `probe`, no test harness, no toolchain. Development still
happens on the host, as [CONTRIBUTING.md](../CONTRIBUTING.md) describes.

```sh
docker compose up --build      # build, then serve
docker compose logs -f         # preflight, then the serving banner
curl localhost:8000/health     # "ok" once the engine is up
```

## What the host has to provide

- **The NVIDIA container runtime** (`nvidia-container-toolkit`). The image
  ships no driver; `libcuda.so.1` is injected at run time. Without it the
  container still starts, and the preflight stops at the device step reporting
  no CUDA driver on this host before exiting non-zero — which is a correct
  diagnosis of a missing toolkit.
- **A driver new enough for CUDA 12.4**, which is what cudarc's bindings are
  pinned to and what the base image carries.
- **Turing or newer**, per the engine's own `sm_75` gate.
- **Room in VRAM.** The container does not know about an `llmxabe` you left
  running on the host. Both will try to claim the same cards.

## The models mount

The compose file bind-mounts one directory, read-only, at `/models`:

```yaml
volumes:
  - ${LLMXABE_MODELS_DIR:-/home/nixabe/llmxabe/models}:/models:ro
```

Read-only is accurate rather than cautious — the engine mmaps the GGUF and
never writes beside it. Everything under that directory is visible to the
container, so the model, the projector and a drafter can all come from one
mount. `LLMXABE_MODEL` and `LLMXABE_MMPROJ` are set to paths *inside* the
container, and if you point `LLMXABE_MODELS_DIR` at a directory laid out
differently, override `LLMXABE_MODEL_REL` and `LLMXABE_MMPROJ_REL` with the
paths relative to it.

The weights reach the container *only* through this mount; they are never
part of the build. `.dockerignore` excludes `/models/` as a directory, and
that line is load-bearing in a way the file's older `*.gguf` was not:
**`.dockerignore` is not `.gitignore`.** Docker's `*` does not cross a `/`,
so `*.gguf` matched the context root and nothing below it — once the
checkpoints moved in-tree under `models/`, every build would have uploaded
94 GB to the daemon. Excluding the directory also catches the HuggingFace
download caches beside the weights, which are not named `*.gguf` and would
otherwise ship on their own. Any new pattern meant to match at depth needs a
`**/` prefix. The context should measure single-digit MB; `docker build`
prints it as `transferring context`, and that number is the check.

**Images are on by default**, because `LLMXABE_MMPROJ` is set. That loads the
vision tower on every worker; a server started without it allocates nothing
for vision and serves the text path unchanged, and image parts get a 400.
Serving text only means commenting the `LLMXABE_MMPROJ` line out of
`docker-compose.yml`; there is no value that means "off", because an empty
path is still a path and preflight will reject it. Preflight does fail fast
and by name when the projector is not a file, which is what you want from a
typo.

## The default command

```
--spec-type none  -c 405504  -s 3  -tb 4096  -pc 4096
--max-tokens 4096  --temp 1.0  --top-p 0.95  --min-p 0.0
```

Only `-c`, `--max-tokens` and `--top-p` differ from the binary's own defaults.

**`-c 405504`.** `--total-context` and `--slots-per-worker` are *per worker*,
and a worker divides its context across its slots as prompt **plus** output.
405504 / 3 = 135168 per slot: a full 128K prompt (131072) with 4096 of output
room. At the 393216 default a slot gets 131072 *including* its output, so a
literal 128K prompt has nowhere to generate — and admission checks the whole
sequence length rather than the first chunk (AGENTS.md rule 4), so it is
refused rather than truncated. Three cards means three workers, so the process
serves nine such sequences.

**`--max-tokens 4096`** matches the output room `-c` just bought. The binary's
default of 16 exists so a request that sets no limit cannot run away; it
truncates most replies.

**`--spec-type none`** is a measured choice, not an omission. At three
concurrent slots and this context depth every drafter lost when it was
measured, and some do not fit in VRAM at all; the numbers and the mechanism
are in [BENCHMARKS.md](BENCHMARKS.md). Speculation is worth reaching for at
*one* slot and shallow prompts, which is not this configuration.

That measurement predates the verify pass's move onto the flash-decode split,
which took `qwen35`'s N=3 drafter from losing to winning. **The `qwen35moe`
serving arm has not been re-run against it**, so this default stands on the
last measurement rather than a current one — worth re-testing before treating
it as settled.

**`--temp 1.0 --top-p 0.95 --min-p 0.0 --top-k 20`** are per-request defaults for callers
that send none of their own. Any request may override them. Note that
temperature above 0 costs a host round-trip per emitted token per sequence,
where `--temp 0` decides greedily on-device; that is a throughput lever, and
it changes what the model produces.

`-s 3`, `-tb 4096` and `-pc 4096` are the binary's defaults, spelled out so
the shape being served is visible in `docker compose config` rather than
implied.

## Configuration

Set these on the host or in a `.env` file beside `docker-compose.yml`.

| Variable | Default | Effect |
| --- | --- | --- |
| `LLMXABE_MODELS_DIR` | `/home/nixabe/llmxabe/models` | Host directory bind-mounted read-only at `/models`. |
| `LLMXABE_MODEL_REL` | `Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` | Model GGUF, relative to the mount. |
| `LLMXABE_MMPROJ_REL` | `Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf` | Vision projector, relative to the mount. |
| `LLMXABE_SERVED_MODEL_NAME` | `qwen3.6-35b-a3b` | Id reported by `/v1/models` and echoed in responses. |
| `LLMXABE_BIND_ADDR` | `127.0.0.1` | Host interface the port is published on. |
| `LLMXABE_HOST_PORT` | `8000` | Host port. The container always listens on 8000. |
| `LLMXABE_GPU_COUNT` | `all` | How many cards to reserve. |
| `LLMXABE_API_KEY` | unset | Key callers must present. Passed through only when set. |
| `LLMXABE_CACHE_RAM` | unset (full coverage, capped at a quarter of host `MemAvailable`) | Host RAM for pinned prefix snapshots, e.g. `8GiB`, or `full`. Size it generously: below full coverage the arena publishes nothing at all rather than less — see [CLI.md](CLI.md#--cache-ram). Three cards at `-c 405504 -s 3` want ~59.6 GiB for `full`, so `memlock` must be unlimited (it already is in the compose file). |
| `LLMXABE_DFLASH` | unset | Path *inside the container* to a DFlash drafter GGUF. |

`LLMXABE_API_KEY` is passed through by `env_file` rather than named with a
default in `environment:`, and the distinction matters: an empty value is not
"no key", it is a key that happens to be the empty string, and every caller
would then have to present it. Unset must stay unset.

**The port is published on loopback by default.** A server with no key accepts
every caller and its own preflight warns so. Move `LLMXABE_BIND_ADDR` off
127.0.0.1 only together with `LLMXABE_API_KEY`.

To serve on a subset of the cards, replace `count` in the compose file's
device reservation with an explicit list:

```yaml
devices:
  - driver: nvidia
    device_ids: ["0", "1"]
    capabilities: [gpu]
```

The engine creates one worker per *visible* device, so that is all it takes;
`--slots-per-worker` and `--total-context` are unchanged, because they were
already per worker.

## Why the runtime base image, and why the builder shares it

`xabe-cuda` has no build script, and cudarc is built with
`fallback-dynamic-loading` and `nvrtc`: the driver is resolved at run time and
the kernels are compiled with NVRTC at launch. Nothing wants a header or nvcc
at build time — the same property CI relies on to build the whole workspace
without a GPU or a CUDA toolkit. So the image is built on
`nvidia/cuda:*-runtime`, which carries the two libraries that must be present
(`libnvrtc.so.12` and `libcublasLt.so.12`), and not on the several-gigabyte
`*-devel`.

The builder stage uses that same base rather than a Debian `rust:` image.
That looks like an obvious simplification and is not one: the binary would be
linked against a newer glibc than the runtime stage has, and would not start.

`ulimits.memlock: -1` is not boilerplate either. The prefix cache's snapshot
arenas are page-locked host memory, several GiB of it, pre-allocated at
startup because allocating on the hot path is forbidden (AGENTS.md rule 6).
Against a default memlock limit `cuMemHostAlloc` fails.

## First start is slow, and that is expected

The engine loads the weights onto every card and compiles its kernels with
NVRTC before it binds the listener. On this host that is about a minute with
the GGUF already in the page cache; a cold first run has to read 32 GB off
disk and is bounded by that. Until the listener is up the container reports
`starting`, not `unhealthy` — the healthcheck's `start_period` covers it.

`/health` is deliberately a good readiness probe here: it sits outside the
authenticated routes, so it needs no key, and the listener only binds after
preflight, the weights and the vision tower are all up. A 200 means the engine
is serving, not that the process exists.

Watch it come up with `docker compose logs -f`. The preflight prints what it
checked and what it decided — the token budget against the block size, the
VRAM budget against the card's *measured* memory, the worker and block counts,
and the pinned cache size. A failure there names the flag responsible.
