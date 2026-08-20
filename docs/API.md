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

## What is accepted and ignored, and what is refused

Decoding is **greedy argmax**. There is no sampler.

*Accepted and ignored*, because refusing them would break every standard
client and they change only which of several plausible answers you get:
`temperature`, `top_p`, `top_k`, `seed`, `presence_penalty`,
`frequency_penalty`, `logit_bias`.

*Refused with `400`*, because honouring them halfway would answer a different
question than the one asked:

| Parameter | Why |
| --- | --- |
| `n`, `best_of` above 1 | Greedy decoding makes every completion identical; returning one where four were asked for is a wrong answer, not an approximate one. |
| `tools`, and the `tool` role | The model is trained for tool calls and its template has a tool section, but nothing here parses a `<tool_call>` block back out of the output. A caller would get prose describing a call it cannot execute. |
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

`GET /v1/models` reports one model, `Qwen3.6-35B-A3B`. Responses echo back
whatever `model` the request named, or that name if the request named none.
The engine serves one model per process; the field is not a selector.
