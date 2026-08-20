# Command-line arguments

The `llmxabe` binary (crate `xabe-server`) parses its command line with
[clap](https://docs.rs/clap). Run it with:

```sh
cargo run -p xabe-server -- [OPTIONS]
# or, once built:
./target/release/llmxabe [OPTIONS]
```

`--help` prints the full list with defaults; `--version` prints the crate
version. This document explains what the options mean and why their defaults
are what they are.

## Precedence

For options that also read an environment variable, the order is:

1. the flag on the command line,
2. the environment variable,
3. the built-in default.

`LLMXABE_MODEL` predates the flags and is kept so existing scripts keep
working; new invocations should prefer the flags. The old `LLMXABE_ADDR`
variable (a combined `host:port`) is gone, replaced by `LLMXABE_HOST` and
`LLMXABE_PORT`.

## Options

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `-m, --model <PATH>` | `LLMXABE_MODEL` | the `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` path under `~/llama.cpp/models` | GGUF model file to load. The engine is built for this one model; pointing it elsewhere is only useful for other quantizations of the same model. |
| `--host <HOST>` | `LLMXABE_HOST` | `127.0.0.1` | Host the HTTP server binds. |
| `--port <PORT>` | `LLMXABE_PORT` | `8000` | Port the HTTP server binds. |
| `-tb, --token-budget <N>` | — | `4096` | Per-step token budget for the scheduler. Must exceed `block_size + max_concurrent_decodes`; see below. |
| `-s, --slots-per-worker <N>` | — | `3` | Concurrent request slots per worker. The default matches the llama.cpp baseline's `-np 3` (see [DEVELOPMENT.md](DEVELOPMENT.md)). |
| `-c, --total-context <N>` | — | `393216` | Total context tokens across all slots, used to size the KV pool and the VRAM budget. The default matches the baseline's `-c 393216`. |
| `-pc, --prefill-chunk <N>` | — | `4096` | Tokens per chunked-prefill step. |

The two-letter shorts `-pc` and `-tb` are rewritten to their long forms before
clap parses (clap itself only supports single-character shorts), so they accept
a space-separated or `=`-joined value (`-pc 2048`, `-pc=2048`) but not the
attached form single-character shorts allow (`-c393216` works, `-pc2048` does
not).

## Validation happens at preflight, not at first request

The values feed the same constructors the preflight checks, so an invalid
combination fails at startup with a message naming the rule it broke, and the
server never comes up. In particular:

- `--token-budget` at or below `block_size + max_concurrent_decodes` is
  rejected by `SchedulerConfig` — at budget == block size a single decoding
  request starves prefill admission and execution serializes to batch 1. This
  is design rule 3 in [AGENTS.md](../AGENTS.md).
- A configuration whose VRAM budget (weights + KV for `--total-context` +
  activations) exceeds the card's measured memory fails preflight with the
  computed shortfall.

## `--log-level`

```
--log-level <info|debug|trace>    console verbosity (default: info)
```

This flag is deliberately *not* parsed by clap. `xabe_log::init_from_args`
strips it from the argument list before clap runs, so every binary in the
workspace — the server, `gguf-info`, the bench tools — parses it identically.
It still appears at the bottom of `--help`.

Interaction with `RUST_LOG`: with no `--log-level` given, `RUST_LOG` (if set)
controls the filter; an explicit `--log-level` wins outright and a warning
says `RUST_LOG` was ignored. Levels and their meaning are documented in
[CONTRIBUTING.md](../CONTRIBUTING.md#console-output).

## What is not configurable

The KV cache element size (f16, matching the baseline's `-ctk f16 -ctv f16`)
and the weights size used by the VRAM budget are constants in
`crates/xabe-server/src/main.rs`. The cache geometry — attention block size,
GDN retention interval — comes from `CacheConfig::with_defaults`; exposing
those as flags would invite exactly the misconfigurations design rules 1 and 2
exist to prevent, so they stay out of the CLI until there is a reason.
