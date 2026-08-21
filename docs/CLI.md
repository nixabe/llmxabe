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

`--api-key` is the exception to preferring the flag: an argument is visible in
`ps` output to every user on the host, so prefer `LLMXABE_API_KEY`. Neither
the value nor the variable's contents appear in `--help` or in any log line.

## Options

### Model and network

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `-m, --model <PATH>` | `LLMXABE_MODEL` | the `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` path under `~/llama.cpp/models` | GGUF model file to load. The engine is built for this one model; pointing it elsewhere is only useful for other quantizations of the same model. |
| `--mmproj <PATH>` | `LLMXABE_MMPROJ` | none | Multimodal projector GGUF (the `mmproj-*.gguf` shipped beside the model). Loads the vision tower on every worker and enables image input; without it the server is text-only and image parts get a 400. See [API.md](API.md#image-input). |
| `--image-max-tokens <N>` | — | `1024` | Most prompt tokens one image may occupy; larger images are resized down to fit. Bounded by the model's own `[8, 4096]` budget. Note this is a *ceiling*; the llama.cpp baseline's `--image-min-tokens 1024` is a floor, so at defaults the two spend image tokens differently. |
| `--host <HOST>` | `LLMXABE_HOST` | `127.0.0.1` | Host the HTTP server binds. |
| `--port <PORT>` | `LLMXABE_PORT` | `8000` | Port the HTTP server binds. |
| `--api-key <KEY>` | `LLMXABE_API_KEY` | none | Key callers must present, in `Authorization: Bearer <key>` or `x-api-key: <key>`. With none set the server is open. See [API.md](API.md#authentication). |

### Capacity and scheduling

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `-tb, --token-budget <N>` | — | `4096` | Per-step token budget for the scheduler. Must exceed `block_size + max_concurrent_decodes`; see below. |
| `-s, --slots-per-worker <N>` | — | `3` | Concurrent request slots per worker. The default matches the llama.cpp baseline's `-np 3` (see [DEVELOPMENT.md](DEVELOPMENT.md)). |
| `-c, --total-context <N>` | — | `393216` | Total context tokens across all slots, used to size the KV pool and the VRAM budget. The default matches the baseline's `-c 393216`. |
| `-pc, --prefill-chunk <N>` | — | `4096` | Tokens per chunked-prefill step. |
| `--cache-ram <SIZE>` | `LLMXABE_CACHE_RAM` | 24 snapshots per worker (≈7.2 GiB for three) | Host RAM the prefix cache may pin for retained snapshots, across all workers. See below. |
| `--spec-type <TYPE>` | — | `none` | Speculative decoder: `none`, `ngram`, or `draft-mtp`. See below. |
| `--spec-ngram-n-max <N>` | — | `3` | `ngram`: most tokens proposed from one suffix match. |
| `--spec-ngram-min <N>` | — | `2` | `ngram`: shortest suffix worth matching on. |
| `--spec-ngram-max <N>` | — | `4` | `ngram`: longest suffix matched before giving up. |
| `--spec-draft-n-max <N>` | — | `3` | `draft-mtp`: most tokens the draft head proposes per step. |
| `--watermark <F>` | — | `0.01` | Fraction of the KV pool held back as admission headroom, in `[0, 1)`. Raise it if admission thrashes under load. |

### Serving defaults

Each of these is a default a request may override; they exist so a deployment
can set its own without every client having to.

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `-a, --alias <NAME>` | `LLMXABE_SERVED_MODEL_NAME` | `Qwen3.6-35B-A3B` | The name `/v1/models` reports and responses echo. The engine serves one model per process; this is a label, not a selector. |
| `--max-tokens <N>` | — | `16` | Output limit for a request that sets none. OpenAI's historical 16 truncates most chat replies, so raise it if your clients rely on the default. |
| `--no-reasoning` | — | off | Answer without extended thinking unless a request asks for it. A request that names a mode still wins, either way. See [API.md](API.md#reasoning). |
| `--temperature <T>` (alias `--temp`) | — | `1.0` | Sampling temperature for a request that sets none; `0` makes silent requests greedy, which is what this server always did before it had a sampler. See [API.md](API.md#sampling). |
| `--top-p <P>` | — | `1.0` | Nucleus cutoff for a request that sets none; `1` disables the filter. |
| `--min-p <P>` | — | `0` | Keep only tokens at least this likely relative to the most likely token, for a request that sets none; `0` disables the filter. |

The two-letter shorts `-pc` and `-tb` are rewritten to their long forms before
clap parses (clap itself only supports single-character shorts), so they accept
a space-separated or `=`-joined value (`-pc 2048`, `-pc=2048`) but not the
attached form single-character shorts allow (`-c393216` works, `-pc2048` does
not).

## `--cache-ram`

The prefix cache's retained snapshots live in **pinned host memory**, not
VRAM, and that pool is the one thing sizing it costs. `--cache-ram` sets how
much of it the process may hold across all workers:

```sh
llmxabe --cache-ram 8GiB      # also 8G, 8GB, 8192MiB, or a bare byte count
llmxabe --cache-ram 0         # retain nothing; no prefix sharing
```

Units are powers of 1024 (`B`, `KiB`, `MiB`, `GiB`, `TiB`, and the `K`/`KB`
spellings of each). A number with an unrecognised unit is refused rather than
read as bytes, because a budget silently taken as a thousandth of what was
meant would surface as a puzzling cache miss rate rather than a message.

The budget is divided by the worker count and then by the size of one
snapshot, and preflight prints what it bought:

```
cache ram        7.93 GiB pinned — 79 snapshots per worker at 102.81 MiB each
```

One snapshot is one GDN retention interval of KV plus the full recurrent
state — 102.8125 MiB for this model at the default `R`. Sizing is per worker
because the arenas are per worker; a snapshot pinned on one card's arena is no
use to another.

Two warnings preflight will give you:

- **`0` disables retention and prefix sharing.** Nothing breaks: a sequence
  that cannot check out a slot simply stops snapshotting and serves normally,
  which also makes this the way to measure what the cache is worth.
- **Fewer snapshots than concurrent sequences** turns sharing off just as
  completely, and less obviously. The engine keeps one slot per concurrent
  sequence free before it will publish anything, so below that line it
  publishes nothing — and whichever sequence loses the race for the remaining
  slots stops retaining *for the rest of its life*, because that switch is
  sticky per sequence. Keep at least `--slots-per-worker` snapshots per
  worker, and preferably several times that: a long sequence holds one slot
  per retention interval it has reached, not one in total.

More RAM is not linearly more hit rate. The cache yields slots back to live
sequences before it starves them, so past the point where live work is
comfortable the extra buys only a longer history of *other* requests'
prefixes. [CACHE.md](CACHE.md#why-the-shared-snapshot-cache-yields) has the
policy.

## Speculative decoding

`--spec-type` chooses how a decode step gets more than one token out of one
weight-read pass. It ships as `none`.

| Type | What drafts | Status |
| --- | --- | --- |
| `none` | nothing — one token per step | the default |
| `ngram` | a suffix match against the sequence's own prompt and output | works |
| `draft-mtp` | the model's own multi-token-prediction head | **refused at startup**; see below |

Whatever the type, **the output is the same**. A drafted token is accepted
only if it equals the token the target model itself chose there — its argmax
under greedy decoding, its sample under sampling — and a rejected draft is
replaced by that token rather than dropped, so speculation changes how many
weight reads a token costs and never what the token is (under sampling it just
accepts fewer drafts). That
is a contract, not a hope: `crates/xabe-engine/tests/speculative_identity.rs`
holds the MTP driver to it, and `--spec-type ngram` was checked against
`--spec-type none` on the same prompt and came back byte-identical.

Drafts are not free before they are accepted. Each one consumes a slot of the
per-step token budget and reserves KV that is discarded on rejection, which is
why the scheduler charges `1 + n` tokens per decoding request and why raising
the count too far is refused rather than allowed to starve prefill.

### `ngram`

Suffix lookup, needing no second model: the sequence's own history is indexed
by n-gram (a hash lookup, not a scan — the history is the whole context), the
longest recent suffix that occurred before is found, and what followed it
last time is proposed. It costs almost nothing to run and pays off on
repetitive output — code, tables, quoted text — and very little on prose.

```sh
llmxabe --spec-type ngram --spec-ngram-n-max 4 --spec-ngram-min 2 --spec-ngram-max 6
```

`--spec-ngram-min` and `--spec-ngram-max` bound the suffix length tried;
`--spec-ngram-n-max` bounds how many tokens one match may propose.

Drafted tokens are fed through one **batched verify pass**: every scheduled
sequence's window (`1 + n` positions) runs through the target in a single
weight-read pass, and each sequence keeps its accepted prefix plus the
target's own token at the first mismatch. That is where the weight-read
saving comes from, and it is what
`crates/xabe-engine/tests/serving_speculative_identity.rs` gates: identical
output to `--spec-type none`, fewer steps. The verify machinery costs VRAM —
one snapshot ring set per decode slot (~300 MiB each at `n = 3`) plus the
per-width verify passes — allocated only when `--spec-type ngram` is on.
`LLMXABE_NGRAM_GATED=1` falls back to the older round-gated loop (drafts
gate extra one-token rounds and are never model inputs), kept as the A/B
lever for measuring the verify path against.

### `draft-mtp` is accepted by the parser and refused by preflight

Qwen3.6 ships an MTP head, and this repository has a complete driver for it —
`crates/xabe-engine/src/speculative.rs`, held to the identity contract above
by `tests/speculative_identity.rs`. It is still not reachable from the server,
and that is a decision rather than an omission: milestone 09 measured **65.2%
acceptance for about 7%**, unbatched across sequences, and did not adopt it.
See [MILESTONES.md](MILESTONES.md).

The flag exists so that answer is given once, out loud, instead of being
rediscovered:

```
speculation      FAIL — --spec-type draft-mtp is not wired into the serving
                 path. …Use --spec-type ngram, or none.
```

`SpeculativeSession` owns one sequence's state and verifies a window for it,
while the serving loop verifies a whole batch in one pass. The batched verify
path that adoption was blocked on now exists (`Forward::run_batch_verify`,
serving the `ngram` type); wiring the MTP head's drafts into it and
re-measuring — because ~7% unbatched is not what it would be worth batched —
is what stands between this flag and acceptance. Refusing until then is the
honest answer; quietly running `ngram` under the name that was not asked for
would look like success and measure like a disappointment.

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
- `--watermark` outside `[0, 1)` is rejected by `SchedulerConfig`.
- A draft count high enough that a full house of decodes leaves less than one
  block for prefill is rejected. `SchedulerConfig` charges one token per
  decode, which is only true with drafting off, so this is design rule 3's
  arithmetic re-run against the draft count you actually asked for:

  ```
  scheduler        FAIL — 3 decodes drafting 1500 tokens each consume 4503
                   of a 4096 token budget, leaving less than one 256-token
                   block for prefill
  ```
- `--spec-ngram-min`/`--spec-ngram-max` that do not satisfy `0 < min <= max`,
  or a `max` at least as long as `--total-context`, are rejected by name.
- `--spec-type draft-mtp` is rejected outright; see
  [above](#draft-mtp-is-accepted-by-the-parser-and-refused-by-preflight).
- `--cache-ram` with an unrecognised unit is rejected by clap, before
  anything starts.

`--cache-ram` is the one that cannot fail: every value it accepts is
servable, so an unhelpfully small one is a preflight *warning* rather than an
error. Read those warnings — `0 snapshots per worker` is a working server with
the prefix cache switched off, and nothing later will remind you.

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

That is a real restriction and worth naming: `--cache-ram` sizes *how many*
snapshots are kept, not *how far apart* they are taken. Retention granularity
is `R = 2048` tokens, so a prefix match is always truncated down to a multiple
of it, and no amount of RAM changes that. Rule 2 is why: snapshotting at every
block boundary once consumed ~80% of the pool and collapsed hit rate.

The admission queue depth (`max_waiting_requests`, four times
`--slots-per-worker`) is also derived rather than exposed. It would be a
reasonable flag; nothing needed it yet.
