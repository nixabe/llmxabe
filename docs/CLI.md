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

The container image takes the same arguments — its entrypoint *is* this
binary, and `command:` in `docker-compose.yml` is the argument list. See
[DOCKER.md](DOCKER.md) for the configuration it ships with.

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
| `--cache-ram <SIZE\|full>` | `LLMXABE_CACHE_RAM` | full coverage, capped at a quarter of host `MemAvailable` | Host RAM the prefix cache may pin for retained snapshots, across all workers. See below. |
| `--spec-type <TYPE>` | — | `none` | Speculative decoder: `none`, `ngram`, `ngram-simple`, `ngram-mod`, `ngram-map-k`, `ngram-map-k4v`, `draft-mtp`, or `spec-dflash`. See below. |
| `--spec-ngram-n-max <N>` | — | `3` | `ngram`: most tokens proposed from one suffix match. |
| `--spec-ngram-min <N>` | — | `2` | `ngram`: shortest suffix worth matching on. |
| `--spec-ngram-max <N>` | — | `4` | `ngram`: longest suffix matched before giving up. |
| `--spec-ngram-simple-size-n <N>` | — | `12` | `ngram-simple`: length of the lookup n-gram. |
| `--spec-ngram-simple-size-m <N>` | — | `48` | `ngram-simple`: length of the draft m-gram, and so the per-step draft budget. |
| `--spec-ngram-simple-min-hits <N>` | — | `1` | Accepted for llama.cpp flag compatibility; `ngram-simple` keeps no hit statistics (llama.cpp's ignores it too). |
| `--spec-ngram-mod-n-match <N>` | — | `24` | `ngram-mod`: lookup n-gram length. |
| `--spec-ngram-mod-n-min <N>` | — | `48` | `ngram-mod`: drop a draft whose chain breaks before this many tokens. |
| `--spec-ngram-mod-n-max <N>` | — | `64` | `ngram-mod`: longest chain drafted per step, and so the per-step draft budget. |
| `--spec-ngram-map-k-size-n <N>` | — | `12` | `ngram-map-k`: key n-gram length. |
| `--spec-ngram-map-k-size-m <N>` | — | `48` | `ngram-map-k`: draft m-gram length, and so the per-step draft budget. |
| `--spec-ngram-map-k-min-hits <N>` | — | `1` | Kept for llama.cpp flag parity; the key-only draft path ignores it, as llama.cpp's does. |
| `--spec-ngram-map-k4v-size-n <N>` | — | `12` | `ngram-map-k4v`: key n-gram length. |
| `--spec-ngram-map-k4v-size-m <N>` | — | `48` | `ngram-map-k4v`: draft m-gram length, and so the per-step draft budget. |
| `--spec-ngram-map-k4v-min-hits <N>` | — | `1` | `ngram-map-k4v`: key hits required before a draft is proposed. |
| `--spec-draft-n-max <N>` | — | `3` | `draft-mtp`/`spec-dflash`: most tokens the drafter proposes per step. |
| `--spec-draft-n-min <N>` | — | `0` | Drop any draft that comes out shorter than this; `0` keeps every draft. |
| `--spec-draft-p-min <P>` | — | `0` | `draft-mtp`/`spec-dflash`: stop drafting at the first token whose probability under the drafter's own head falls below this; `0` disables the gate. |
| `--spec-draft-p-split <P>` | — | `0.1` | Accepted for llama.cpp flag compatibility; no current speculative decoder uses a split probability (llama.cpp's ignore it too). |
| `--spec-dflash <PATH>` | `LLMXABE_DFLASH` | — | `spec-dflash`: the trained drafter GGUF. Required by that type. |
| `--watermark <F>` | — | `0.01` | Fraction of the KV pool held back as admission headroom, in `[0, 1)`. Raise it if admission thrashes under load. |

### Serving defaults

Each of these is a default a request may override; they exist so a deployment
can set its own without every client having to.

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `-a, --alias <NAME>` | `LLMXABE_SERVED_MODEL_NAME` | `Qwen3.6-35B-A3B` | The name `/v1/models` reports and responses echo. The engine serves one model per process; this is a label, not a selector. |
| `--max-tokens <N>` | — | `16` | Output limit for a request that sets none. OpenAI's historical 16 truncates most chat replies, so raise it if your clients rely on the default. Raise it with the *reservation* in mind: a request holds `prompt + max output` of KV for its whole life, so a large ceiling here — or from a client that sends one — is taken out of the other slots whether or not it is reached. Anything above one slot's share of the pool is capped to it, and a sequence that reaches the cap stops with `length`. |
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
llmxabe --cache-ram full      # every slot can restore anywhere in its context
llmxabe --cache-ram 0         # retain nothing; no prefix sharing
```

`full` (spelled `max` or `-1` too, the last because llama.cpp's `-cram -1`
means the same thing) sizes the arena from the serving configuration instead
of a byte budget: `--slots-per-worker` slots, each needing one snapshot per
retention interval of its share of `--total-context`. That is what the
default computes as well, capped at a quarter of the host's `MemAvailable`
so the arena cannot displace the page cache the weights are read through.

**Size this generously.** The arena does not degrade gracefully when it is
too small: publishing yields rather than evicts, so a worker that cannot hold
its sessions' retention points publishes *nothing at all* and every turn
re-prefills from zero. Measured with three growing conversations on one card,
the old fixed 24-snapshot default gave **0% prefix reuse on every request**
and 319 s of wall clock, against 73–86% and 132 s once the arena covered the
context. There is no partial-credit region between those: it is not "a bit
less caching", it is none.

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

Preflight also states the number that predicts whether a conversation keeps
hitting — how far into its context a slot can restore from:

```
                 135168 tokens of restore coverage per slot, of 135168 served (100% of the context)
```

Below full coverage it names the shortfall and what `full` would cost, rather
than leaving the miss rate to be inferred from latency.

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
| `ngram` | a variable-length suffix match against the sequence's own prompt and output | works |
| `ngram-simple` | llama.cpp's fixed-length backward scan for the same tail | works; see below |
| `ngram-mod` | llama.cpp's n-gram → next-token table, shared across the worker's sequences | works; see below |
| `ngram-map-k` | llama.cpp's key-n-gram map, drafting from the newest key match | works; see below |
| `ngram-map-k4v` | as `ngram-map-k`, but tracking four continuations per key and drafting only a dominant one | works; see below |
| `draft-mtp` | the model's own multi-token-prediction head | works; see below |
| `spec-dflash` | a trained DFlash drafter (separate GGUF, `--spec-dflash`) | works; see below |

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

### The llama.cpp n-gram family

`ngram-simple`, `ngram-mod`, `ngram-map-k` and `ngram-map-k4v` are ports of
llama.cpp's own self-speculative drafters (upstream PRs #18471 and #19164),
flag names and defaults included, so a head-to-head runs the same policy on
both sides. They share `ngram`'s machinery from the draft outward: the same
batched verify pass, the same exactness contract, the same VRAM cost — only
the choice of what to draft differs.

```sh
llmxabe --spec-type ngram-simple  --spec-ngram-simple-size-n 12 --spec-ngram-simple-size-m 48
llmxabe --spec-type ngram-mod     --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
llmxabe --spec-type ngram-map-k   --spec-ngram-map-k-size-n 12 --spec-ngram-map-k-size-m 48
llmxabe --spec-type ngram-map-k4v --spec-ngram-map-k4v-size-n 12 --spec-ngram-map-k4v-size-m 48 --spec-ngram-map-k4v-min-hits 1
```

Each type's draft cap — `size-m`, or `n-max` for `ngram-mod` — **is** the
scheduler's per-step draft budget, so it is charged against the token budget
like any other draft count. llama.cpp's defaults are large (48 and 64 against
this engine's `ngram` default of 3), and which way that cuts depends entirely
on batch width.

**These have been measured; see
[BENCHMARKS.md](BENCHMARKS.md#speculative-decode-exact-by-construction-priced-by-the-verify-step)
for the numbers and the method.** The short version, and the qualifier
matters more than the numbers: at **N=1 with short prompts they are the best
drafters this engine has**, because the verify pass costs about the same at
49 rows as at 4, so the wider window is nearly free — `ngram-map-k` and
`ngram-map-k4v` +20.9%/+20.8%, `ngram-simple` +16.3%, `ngram-mod` +11.1%,
against `ngram`'s +9.2%. At **N=3 every one of them loses**, from −22.8%
(`ngram-simple`) to −78.8% (`ngram-map-k4v`).

**At long context the picture is worse, and it is the one to plan against.**
At three slots of ~120K, `ngram-map-k4v` is −80.6% on decode and −9.3% on
prefill, and that is at a draft cap of 8 — the wide llama.cpp defaults do
not fit in VRAM at that depth at all. Acceptance there is *perfect* (27 of
27 tokens per step) and loses anyway, because a verify step costs ~1.4 s
against a 30 ms plain decode step whatever it carries. Serving defaults to
`--spec-type none`, and for a full house at long context that default is
simply the fastest thing measured. Reach for a drafter for shallow
single-stream work, and do not extrapolate a shallow win to a deep one.

**Check VRAM before running the defaults.** The verify path holds one GDN
snapshot ring set per decode slot, and it is sized by the draft count: 30 GDN
layers × (32·128·128 + 8192·3) fp32 × `(drafts + 2)` slots, per decoding
sequence. This is arithmetic from the model geometry that lands on the
~300 MiB per slot quoted above for `n = 3` — and the top row of it has since
been confirmed the hard way: `--spec-type ngram-mod` at its default `n-max`
64 fails with `CUDA_ERROR_OUT_OF_MEMORY` before the first step at `-s 3` on
this card.

| Draft cap | Per decode slot | Per worker at `-s 3` | At `-s 3` on this card |
| --- | --- | --- | --- |
| 3 (`ngram` default) | 0.31 GiB | 0.92 GiB | runs |
| 12 | 0.86 GiB | 2.58 GiB | runs |
| 48 (`size-m` default) | 3.07 GiB | 9.20 GiB | runs |
| 64 (`n-max` default) | 4.05 GiB | 12.15 GiB | **out of memory** |

Against a 29.6 GiB model on a 48 GiB card, llama.cpp's defaults leave little
room for the KV pool, so expect to lower `size-m`/`n-max`, `-s`, or `-c`
rather than to run all three at their defaults at once. Lowering the cap is
the cheapest of the three, and for `ngram-simple` it must still stay at or
above `size-n` or that type can never draft.

Two behaviors carried over verbatim, because changing them would change what
is drafted:

- **`ngram-simple` drops any draft shorter than its own `size-n`.** With
  `--spec-ngram-simple-size-m` below `--spec-ngram-simple-size-n` it can
  therefore never draft at all. That is upstream's rule, not a port artifact.
- **`ngram-mod`'s table is shared by every sequence a worker serves**, as
  upstream shares one `common_ngram_mod` across a context's sequences: what
  one request teaches the table, a concurrent request drafts from. It resets
  itself when occupancy passes 25% at a sequence's start, or after five
  consecutive rounds with under a quarter of the draft accepted.

#### `ngram` versus `ngram-simple`

They are different algorithms, not two names for one. Both look for an
earlier occurrence of the sequence's current tail and draft what followed it,
and they differ in four ways that matter:

| | `ngram` | `ngram-simple` |
| --- | --- | --- |
| Tail length | variable: tries `--spec-ngram-max` down to `--spec-ngram-min` (default 4 → 2), longest match wins | fixed at `--spec-ngram-simple-size-n` (default 12) |
| Lookup | hash index per length, one probe, collision-verified against the history before drafting | backward linear scan of the whole history, newest match wins |
| History | fixed-capacity ring; the oldest tokens are evicted | append-only, the whole context |
| Short drafts | proposed — any length from 1 up to the cap | dropped entirely below `size-n` tokens |

The practical split: `ngram` matches short tails, so it fires often and
proposes a few tokens; `ngram-simple` demands a 12-token repeat, so it fires
rarely and proposes up to 48 when it does. `ngram`'s index also makes its
per-step host cost independent of context length, where `ngram-simple`'s scan
grows with it — which is why `ngram` was written that way in the first place
(see the module docs in `crates/xabe-sched/src/ngram.rs`). The port keeps the
scan, because parity with upstream's draft choice is the point of having it.

### `draft-mtp`

Qwen3.6 ships a trained multi-token-prediction head (GGUF block 40), and
`--spec-type draft-mtp` serves it: the head drafts
`--spec-draft-n-max` tokens per step for every scheduled sequence in one
chained batch, and the same batched verify pass the `ngram` type uses
accepts them. Unlike `ngram` it drafts *every* step, on any content — the
trade is what it costs:

```sh
llmxabe --spec-type draft-mtp --spec-draft-n-max 3
```

- **One extra layer resident.** Block 40's expert weights are loaded once
  (they are not part of text serving otherwise).
- **One draft KV cache per resident sequence**, sized to that request's
  `prompt + max_output`, allocated at admission.
- **Catch-up rides prefill.** The draft head's own cache must be filled over
  the whole prompt, pairing each token with the target's hidden state one
  position back; this runs as one extra block-40 pass per prefill chunk.
- The verify machinery is the same as `ngram`'s (ring sets, per-width verify
  passes) and is allocated only when a speculative type is on.

Two classes of request decode plain under this type, by design: sequences
restored from a prefix snapshot (the snapshot carries no draft-head cache,
and drafting over the hole would propose from zeroed state) and image-bearing
sequences (the draft head embeds token ids; image spans have none).

An earlier milestone measured the single-sequence, unbatched driver at 65.2%
acceptance for about +7% and did not adopt it; the serving path above is the
batched re-attempt that measurement asked for.

**Measured:** the batched path is **+1.4% at N=1** on short prompts — the
only trained drafter that is not a loss, and far behind the model-free
`ngram-map-k`'s +20.9% — and **−27.2% at N=3**. As with `spec-dflash`,
raising `--spec-draft-n-max` to 15 makes it much worse (−55.4% at N=1),
because the draft head's own pass scales with the block while the verify
pass does not. At three slots of ~120K it is **−90.8% on decode and −21.8%
on prefill**, the latter because its catch-up pass runs on every prefill
chunk; it accepts every token it drafts there and loses anyway.
See [BENCHMARKS.md](BENCHMARKS.md#speculative-decode-exact-by-construction-priced-by-the-verify-step).

### `spec-dflash`

Block in-fill drafting (arXiv 2602.06036): a small trained drafter — six
dense layers in a separate GGUF — predicts every drafted position in **one**
drafter pass per step, instead of one pass per token. Its picture of the
context is not tokens at all: the target's own residual stream is captured
at eight fixed layers, fused through the drafter's `fc`, and projected
directly into the drafter's KV caches. The query is
`[last_token, MASK × n]`, attended non-causally; the same batched verify as
the other types accepts.

```sh
llmxabe --spec-type spec-dflash --spec-dflash qwen36-35b-a3b-dflash-Q8_0.gguf --spec-draft-n-max 3
```

- `--spec-dflash` names the drafter checkpoint and is required. The
  drafter's trained block bounds `--spec-draft-n-max` (15 for the shipped
  checkpoint); a larger ask is refused at startup.
- Costs: the drafter's weights resident (~420 MB at Q8_0), six small KV
  caches per resident sequence, and one strided feature-tap copy per
  configured target layer on every prefill chunk and verify pass. The
  drafter shares the target's token embeddings and LM head by alias.
- The same two classes decode plain as under `draft-mtp`:
  snapshot-restored sequences and image-bearing sequences.

The two upstream implementations disagree about which residual-stream tap
`target_layers` names (off by one layer) and about causal masking on the
sliding-window layers; this engine follows llama.cpp, whose ecosystem the
GGUF comes from — `xabe_model::dflash`'s module docs carry the details.

**Measured, and it does not currently pay:** −14.8% at N=1 and −39.5% at
N=3 against plain decode, at the default 3-token block, on short prompts.
Raising the block makes it worse, not better — −65.8% at N=1 and −54.8% at
N=3 at the trained maximum of 15 — because one drafter pass per step is
cheap only while the block is short: 3 → 15 adds ~52.6 ms to a step whose
plain form is 9.64 ms. **At long context it does not run at all:** its six
draft-layer KV caches are sized to the full sequence, so they scale with
context rather than with the draft count, and three slots of ~120K is out
of memory even at a 2-token block.
That is the opposite of how the n-gram types behave, where a wider window is
nearly free, and it is the thing to know before reaching for
`--spec-draft-n-max`. See
[BENCHMARKS.md](BENCHMARKS.md#speculative-decode-exact-by-construction-priced-by-the-verify-step)
for the method and the full table.

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
- `--spec-type draft-mtp` with `--spec-draft-n-max 0` is rejected by name
  (drafting nothing is `--spec-type none`).
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
