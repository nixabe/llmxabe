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

Every generation endpoint honours `temperature` and `top_p`; `top_k` is
honoured on `/v1/messages` (where it is Anthropic's own) and on both OpenAI
completions shapes (where it is the llama.cpp/vLLM extension clients already
send); `seed` is honoured on the OpenAI shapes that carry it.

- `temperature: 0` is **greedy argmax**, decided on the device — the path
  every request took before the server had a sampler, at the same cost.
- A request that says nothing gets `--temperature`, which ships as
  `1.0` — the default both dialects document. An operator who wants the old
  always-greedy behaviour sets it to `0`.
- Filters chain the way llama.cpp's sampler chain does: temperature scales
  the logits, `top_k` keeps the k most likely, `top_p` keeps the smallest
  set of the survivors whose cumulative probability reaches `p`, and the
  draw renormalizes over what is left. `top_k: 1` and `top_p: 0` both
  degenerate to greedy and are served as such.
- Equal `seed`s with equal parameters replay equal outputs. Without a seed,
  each request draws fresh entropy.
- `temperature` outside `[0, 2]` and `top_p` outside `[0, 1]` are refused
  with `400`.

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
| `/v1/responses` | `tools: [{type:"function", name, description, parameters}]` | `function_call` output items; streamed with `response.function_call_arguments.delta` / `.done` |

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

Image and video content parts are refused as unknown part types: this engine
is text-only and does not load the vision encoder. See the scope note in
[../README.md](../README.md).

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
which is larger than the caller's raw text by the ChatML markup.
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

Four request defaults are set at startup rather than compiled in, so a
deployment can suit its clients without changing every one of them. All are
overridable per request.

| Default | Flag | Ships as |
| --- | --- | --- |
| Output limit when a request sets none | `--max-tokens` | `16` |
| Extended thinking when a request says nothing | `--no-reasoning` | on |
| Sampling temperature when a request sets none | `--temperature` | `1.0` |
| Model name reported and echoed | `--alias` | `Qwen3.6-35B-A3B` |

`16` is OpenAI's historical default and truncates most chat replies; raise it
if your clients lean on it. See [CLI.md](CLI.md#serving-defaults).
