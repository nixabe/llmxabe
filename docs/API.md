# HTTP API

`llmxabe` serves four generation endpoints in three request dialects, all
backed by one engine and one shared prefix cache:

| Endpoint | Dialect | Streaming |
| --- | --- | --- |
| `POST /v1/completions` | OpenAI text completions | yes |
| `POST /v1/chat/completions` | OpenAI chat completions | yes |
| `POST /v1/responses` | OpenAI Responses | yes |
| `POST /v1/messages` | Anthropic Messages | yes |
| `POST /v1/messages/count_tokens` | Anthropic | — |
| `GET /v1/models` | OpenAI | — |
| `GET /health` | — | — |

`/health` is the only endpoint outside authentication, so a load balancer can
probe the server without holding a key.

## Authentication

Set a key with `--api-key`, or with the `LLMXABE_API_KEY` environment
variable:

```sh
llmxabe --api-key sk-my-key
LLMXABE_API_KEY=sk-my-key llmxabe
```

Callers may present it either way, on any endpoint:

```
Authorization: Bearer sk-my-key
x-api-key: sk-my-key
```

OpenAI clients send the first, Anthropic clients send the second, and a server
that speaks both dialects should not make the caller care which one it
guessed. `x-api-key` wins if both are present.

**With no key configured the server is open** — every caller is accepted. That
is what it did before a key could be set, and what `llama-server` does without
`--api-key`. The startup log says which mode it is in; if it says
`no API key configured — every caller is accepted`, do not put the port on a
network you do not control.

A request with a missing or wrong key gets `401` in the dialect its endpoint
speaks:

```jsonc
// OpenAI endpoints
{"error": {"message": "…", "type": "invalid_request_error",
           "param": null, "code": "invalid_api_key"}}

// /v1/messages
{"type": "error", "error": {"type": "authentication_error", "message": "…"}}
```

Comparison is constant-time in the key's bytes. The key is not logged, and
`--help` does not print the environment variable's value.

## Streaming

Every generation endpoint takes `"stream": true` and answers with
`text/event-stream` in its own dialect's event shape.

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'authorization: Bearer sk-my-key' -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":256,"stream":true}'
```

- **OpenAI completions and chat completions** emit `data:` frames and close
  with `data: [DONE]`. Chat sends the role in the first chunk and the finish
  reason in the last. `{"stream_options": {"include_usage": true}}` adds
  `usage` to that last chunk.
- **Anthropic messages** emit named events: `message_start`,
  `content_block_start`, `content_block_delta`, `content_block_stop`,
  `message_delta`, `message_stop`.
- **Responses** emit named events with a monotonic `sequence_number`:
  `response.created`, `response.in_progress`, `response.output_item.added`,
  `response.output_text.delta` (or `response.reasoning_text.delta`), the
  matching `.done` events, and `response.completed` carrying the whole
  response object.

Failures that happen *before* the response begins — a malformed body, a
refused parameter, a saturated engine — are ordinary HTTP status codes.
Failures after that arrive as a terminal event in the stream, because the
status line has already been sent.

Closing the connection cancels the request and releases its KV capacity.

### Text arrives as fast as it decodes, not as fast as it tokenizes

Deltas are detokenized incrementally, so a multi-byte character split across
two tokens is held back until it is complete rather than emitted as a
replacement character. A chunk can therefore be empty of new text even though
a token was produced.

## Reasoning

Qwen3.6 is a reasoning model, and its generation prompt opens a `<think>`
block. That reasoning is reported separately from the answer rather than
concatenated into it:

| Endpoint | Where the reasoning goes |
| --- | --- |
| `/v1/chat/completions` | `message.reasoning_content`, streamed as `delta.reasoning_content` |
| `/v1/messages` | a `thinking` content block, streamed as `thinking_delta` |
| `/v1/responses` | a `reasoning` output item, streamed as `response.reasoning_text.delta` |
| `/v1/completions` | nowhere — a raw completion is not a chat turn and opens no think block |

Reasoning is on by default, matching the model's own template. A deployment
that never wants it can flip the default with `--no-reasoning`; a request that
names a mode still wins, either way.

Turn it off per request:

```jsonc
// chat completions — vLLM's spelling
{"chat_template_kwargs": {"enable_thinking": false}}
// messages
{"thinking": {"type": "disabled"}}
// responses
{"reasoning": {"effort": "none"}}      // "minimal" also works
```

With reasoning off the prompt hands the model a closed, empty think block, so
it answers directly. That is the model template's own mechanism, not a filter
over its output.

The `thinking` blocks this server emits carry an empty `signature`. The real
Anthropic API signs them so they can be verified when replayed; there is
nothing to verify here, and clients that round-trip a block still need the
field to be present.

## Stop sequences

`stop` (OpenAI) and `stop_sequences` (Anthropic) are honoured in the HTTP
layer: output is cut at the first match, the sequence itself is not returned,
and the engine request is cancelled immediately rather than left to run to
`max_tokens`. Text that could still turn out to be the beginning of a stop
sequence is held back, so a sequence is never missed by falling across a chunk
boundary.

OpenAI reports this as `finish_reason: "stop"`, which is also what an
end-of-turn token reports. Anthropic distinguishes them: `stop_reason:
"stop_sequence"` plus the `stop_sequence` that matched.

## Sampling

Every generation endpoint honours `temperature`, `top_p`, and `min_p` (the
llama.cpp/vLLM extension: keep only tokens at least `min_p` times as likely
as the most likely one); `top_k` is honoured on `/v1/messages` (where it is
Anthropic's own) and on both OpenAI completions shapes (where it is the same
kind of extension); `seed` is honoured on the OpenAI shapes that carry it.

- `temperature: 0` is **greedy argmax**, decided on the device — the path
  every request took before the server had a sampler, at the same cost.
- A request that says nothing gets `--temperature` (alias `--temp`),
  `--top-p`, and `--min-p`, which ship as `1.0`, `1.0`, and `0` — the
  dialect-documented defaults, i.e. plain temperature-1 sampling. An
  operator who wants the old always-greedy behaviour sets the temperature
  default to `0`; one who wants llama-server's flavour sets, say,
  `--temp 0.8 --top-p 0.95 --min-p 0.05`.
- Filters run in the sequence llama.cpp's chain uses: temperature scales
  the logits first, `top_k` keeps the k most likely, `top_p` keeps the
  smallest set of the survivors whose cumulative probability reaches `p`,
  `min_p` drops survivors below `min_p` times the most likely token's
  probability, and the draw renormalizes over what is left. `top_k: 1`,
  `top_p: 0`, and `min_p: 1` all degenerate to greedy and are served as
  such.
- Equal `seed`s with equal parameters replay equal outputs. Without a seed,
  each request draws fresh entropy.
- `temperature` outside `[0, 2]`, and `top_p` or `min_p` outside `[0, 1]`,
  are refused with `400`.

A sampling request pays one logits-row copy to the host (~1 MB) plus an
`O(vocab)` host pass per generated token; greedy requests are untouched.
Speculative decoding stays exact under sampling — a draft is accepted only
when it equals the token the target model drew — it just accepts fewer
drafts as temperature rises.

Still *accepted and ignored*, because refusing them would break standard
clients and they only nudge which plausible answer you get:
`presence_penalty`, `frequency_penalty`, `logit_bias`.

## Tool calling

All three chat dialects take tool definitions and return structured calls:

| Dialect | Definitions | Calls come back as |
| --- | --- | --- |
| `/v1/chat/completions` | `tools: [{type:"function", function:{name, description, parameters}}]` | `message.tool_calls`, finish reason `tool_calls`; streamed as one `delta.tool_calls` entry per call |
| `/v1/messages` | `tools: [{name, description, input_schema}]` | `tool_use` content blocks, stop reason `tool_use`; streamed as a `tool_use` block with one `input_json_delta` |
| `/v1/responses` | `tools: [{type:"function", name, description, parameters}]`, and `namespace` groups of them | `function_call` output items; streamed with `response.function_call_arguments.delta` / `.done` |

A `/v1/responses` tool may also be a **`namespace`** group —
`{type:"namespace", name, description, tools:[…]}` — which harnesses use to
keep a large tool surface from spending its whole prompt budget on schemas.
The group is flattened into its members, whose prompt names become
`group.member` so two namespaces may each hold a `search`. A served call
splits them back apart, because the wire format carries the namespace as its
own field beside the name rather than as a prefix on it:

```json
{ "type": "function_call", "name": "list_orders", "namespace": "crm",
  "call_id": "call_1", "arguments": "{}" }
```

**An unrecognized content part is skipped, not refused.** The parts a
provider invents are overwhelmingly its own artifacts — a redacted or signed
reasoning block, a trace of a search it ran — and failing the request over one
throws away a conversation this server could have served. `refusal` is read as
the assistant turn's text, since that is what it is.

The exception is content belonging to the *caller* rather than the provider:
`input_file`, `document`, `input_audio`, `video` and their spellings are
refused by name. This server cannot read any of them, and skipping one means
answering about something it never saw — a wrong answer, where the 400 is
merely an unsupported one. Video is out of scope for the engine rather than
unimplemented, so it belongs on that list permanently.

**A `reasoning` input item is accepted and replayed.** Continuing a reasoning
conversation means handing the previous `output` back as `input`, reasoning
items included, so refusing them broke every agentic loop that replayed what
this server had just emitted. The text is read from the item's `content`
(`reasoning_text`) or its `summary` (`summary_text`) and folded into the
assistant turn rather than dropped — `Conversation::render` replays the
reasoning that came after the caller's last question, which is exactly a
tool-calling loop's, and is the chain the model needs to keep across the call.

Items recording work a provider did on the model's behalf — `web_search_call`,
`mcp_call`, anything ending `_call` — are skipped with a warning. This server
executed none of it and cannot replay a result it never produced, but the
turns around them are still a conversation. An `item_reference` remains a 400:
it names content held server-side and this server stores nothing, so skipping
it would silently drop conversation rather than a trace.

**Hosted tools are dropped, not refused.** `web_search`, `file_search`,
`code_interpreter`, an `mcp` server, Anthropic's dated `web_search_*` and
`bash_*` — all of these are executed by the provider, and there is no provider
here. Failing the request over one used to take the caller's own function
tools down with it, so a harness offering `web_search` alongside six functions
got nothing served at all. They are now dropped and the rest are served, in
every dialect. The drop is not silent: a `warn!` names the types that went
missing and how many function tools were served instead, so a harness that
genuinely needed one can be diagnosed from the log rather than from a worse
answer. Anything unrecognized is dropped the same way, which keeps a tool type
invented next year from taking a working request down with it. A *malformed*
function tool — no name, an unrenderable name — is still a 400, because that
is the caller's bug rather than a missing capability.

A member's `defer_loading` is accepted and ignored: it exists so a schema can
be fetched later by tool search, which this server does not implement. Every
schema is offered up front instead — it costs prompt tokens and leaves the
tool callable, which is the better failure.

Results go back in the dialect's own shape — the `tool` role with
`tool_call_id`, a `tool_result` content block, a `function_call_output`
item — and render into the model's template as `<tool_response>` blocks.
`POST /v1/messages/count_tokens` accepts tools and prices the same prompt the
request itself would render.

The model emits calls in its template's XML form
(`<tool_call><function=name><parameter=key>…`); the server parses that back
out, typing each argument by its declared schema — a `string` parameter takes
the text verbatim, everything else parses as JSON — the same rules llama.cpp
applies to this format. A block that does not parse is returned as plain
text rather than dropped. Because a call only parses once its closing tag
arrives, streamed calls arrive as one complete delta each, not
token-by-token.

`tool_choice` `"auto"` and `"none"` are honoured (`none` withholds the tools
from the prompt while still rendering tool history). Forcing a call —
`required`, `any`, or a named function — is refused with `400`: it is a
guarantee about the output, and nothing here constrains decoding to keep it.

## Image input

With the server started with `--mmproj` (see [CLI.md](CLI.md#model-and-network)),
user messages may carry images in each dialect's own spelling:

- **OpenAI chat**: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`
- **Anthropic**: `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}`
  (a `"type": "url"` source is accepted only for `data:` URIs)
- **Responses**: `{"type": "input_image", "image_url": "data:image/png;base64,..."}`

Images are inline only — `data:` URIs or base64. Remote `http(s)` URLs are
refused: this server does not fetch content on a caller's behalf. PNG, JPEG,
and WebP decode; the container format is sniffed from the bytes, not from the
declared media type. Images are only meaningful in user messages; in a
system, assistant, or tool message they are refused rather than dropped.

Each image renders into the prompt as the model's own
`<|vision_start|><|image_pad|><|vision_end|>` markup at the position of its
content part, and occupies up to `--image-max-tokens` prompt tokens
(default 1024) once encoded — a 512×512 image costs 256. Those tokens count
toward context and admission like any others, and
`/v1/messages/count_tokens` prices them without running the encoder.

Without `--mmproj`, an image part is refused with a 400 that names the flag.

## What is refused

*Refused with `400`*, because honouring them halfway would answer a different
question than the one asked:

| Parameter | Why |
| --- | --- |
| `n`, `best_of` above 1 | One completion per request; returning one where four were asked for is a wrong answer, not an approximate one. |
| A forcing `tool_choice` | See above. |
| `previous_response_id` | Responses are not stored, so the reference cannot be resolved; the model would answer without context the caller believed it had sent. |
| A system message after the first turn | The model's own template silently discards it. Discarding an instruction the caller wrote is worse than refusing it. |
| A batch of prompts in `/v1/completions` | One prompt per request. |

Video parts are refused as unknown part types, and images are refused
whenever the rules in [Image input](#image-input) are not met — no
`--mmproj`, a remote URL, or an image outside a user message.

## Conversation rendering

Chat requests in all three dialects fold into one conversation and render to
the ChatML markup Qwen3.6 expects:

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{content}<|im_end|>
<|im_start|>assistant
<think>
```

Details that follow the model's own template:

- Leading system messages merge into one, joined with a newline. Anthropic's
  top-level `system` and the Responses API's `instructions` land in the same
  place.
- An assistant turn replays its `<think>` block **only** if it came after the
  last user message. Earlier reasoning is dropped, which is what keeps a long
  conversation's prompt from growing with every past reasoning span.
- The generation prompt ends with an open `<think>` block, or with a closed
  empty one when reasoning is off.
- With tools, the system turn opens with the template's `# Tools` section —
  the definitions inside `<tools>` tags and the call-format instructions —
  and the caller's own system text follows it. Replayed assistant calls
  render as `<tool_call><function=…><parameter=…>` blocks; tool results
  render as `<tool_response>` blocks inside user turns, consecutive results
  sharing one turn. A user turn that is only a tool response does not count
  as "the last user message" for reasoning replay, so reasoning survives
  across an agentic loop's intermediate steps — all exactly as the Jinja
  template does it.

The template that ships in the GGUF is Jinja, and rendering it would mean
carrying a Jinja engine plus shims for the Python string methods it calls.
This markup is written directly instead and pinned by unit tests in
`crates/xabe-server/src/http/chat.rs`.

## Token accounting

`prompt_tokens` / `input_tokens` counts the rendered prompt after templating,
which is larger than the caller's raw text by the ChatML markup — and, for
requests with images, by the expanded `<|image_pad|>` spans.
`completion_tokens` / `output_tokens` counts every token the model produced,
including the reasoning span and the `</think>` that closes it, and excluding
the end-of-turn token.

`POST /v1/messages/count_tokens` returns `{"input_tokens": N}` for a request
body, doing the same templating and tokenization the request itself would do
and nothing else.

## Model name

`GET /v1/models` reports one model, `Qwen3.6-35B-A3B` unless
`--alias` says otherwise. Responses echo back whatever `model` the
request named, or that name if the request named none. The engine serves one
model per process; the field is not a selector.

## Defaults an operator can move

These request defaults are set at startup rather than compiled in, so a
deployment can suit its clients without changing every one of them. All are
overridable per request.

| Default | Flag | Ships as |
| --- | --- | --- |
| Output limit when a request sets none | `--max-tokens` | `16` |
| Extended thinking when a request says nothing | `--no-reasoning` | on |
| Sampling temperature when a request sets none | `--temperature` (alias `--temp`) | `1.0` |
| Nucleus cutoff when a request sets none | `--top-p` | `1.0` |
| Relative-probability floor when a request sets none | `--min-p` | `0` |
| Model name reported and echoed | `--alias` | `Qwen3.6-35B-A3B` |

`16` is OpenAI's historical default and truncates most chat replies; raise it
if your clients lean on it. See [CLI.md](CLI.md#serving-defaults).
