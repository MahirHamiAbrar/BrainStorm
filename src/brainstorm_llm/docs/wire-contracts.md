# Wire contracts

This page records the provider contracts the conversion layer implements.

## Command Code Provider API

Base: `https://api.commandcode.ai/provider/v1`

| Endpoint | Format |
|---|---|
| `POST /chat/completions` | OpenAI Chat Completions |
| `POST /messages` | Anthropic Messages |
| `POST /systemone` | TypeSafe System One |
| `GET /models` | OpenAI-style list |

Auth: `Authorization: Bearer <key>`; `/messages` also accepts `x-api-key`.
ZDR: `x-cmd-zdr: 1`.

`/models` item:

```json
{
  "id": "deepseek/deepseek-v4.1-flash",
  "object": "model",
  "created": 1793904707,
  "owned_by": "command-code",
  "name": "DeepSeek V4.1 Flash (latest)",
  "context_length": 1000000,
  "supported_endpoints": ["/chat/completions", "/responses"]
}
```

Routing: `/chat/completions` present → OpenAI wire; only `/messages` → Anthropic.

## OpenAI Chat Completions

Response:

```json
{
  "id": "chatcmpl-...", "object": "chat.completion", "created": 1793,
  "model": "deepseek/deepseek-v4.1-flash",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "text or null",
      "refusal": "string or null",
      "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}}]
    },
    "finish_reason": "stop|length|tool_calls|content_filter"
  }],
  "usage": {
    "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
    "prompt_tokens_details": {"cached_tokens": 30},
    "completion_tokens_details": {"reasoning_tokens": 0}
  }
}
```

Stream chunk: `object: "chat.completion.chunk"`; `choices[].delta` can contain
`role`, `content`, `reasoning_content`, `refusal`, and `tool_calls[]` with
`index` and partial `function.arguments`. With
`stream_options.include_usage=true`, the last chunk carries `usage`; `[DONE]`
ends the stream.

Conversion details:

- `input_tokens = max(0, prompt_tokens - cached_tokens)`.
- `cached_tokens` becomes `cache_read_input_tokens`.
- `finish_reason`: `stop→end_turn`, `length→max_tokens`,
  `tool_calls→tool_use`, `content_filter→refusal`.
- `refusal` becomes a text block and sets `stop_reason="refusal"`.
- Invalid tool arguments become `{"_raw_arguments": "<raw>"}`.

## Anthropic Messages

Canonical response:

```json
{
  "id": "msg_...", "type": "message", "role": "assistant",
  "content": [
    {"type": "text", "text": "..."},
    {"type": "tool_use", "id": "toolu_...", "name": "lookup", "input": {}},
    {"type": "thinking", "thinking": "...", "signature": "..."}
  ],
  "model": "claude-sonnet-4-6",
  "stop_reason": "end_turn|max_tokens|stop_sequence|tool_use|pause_turn|refusal|model_context_window_exceeded",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 100, "output_tokens": 20,
    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 30
  }
}
```

Stream events: `message_start`, `content_block_start`, `content_block_delta`
(`text_delta`, `input_json_delta`, `thinking_delta`, `signature_delta`,
`citations_delta`), `content_block_stop`, `message_delta` (stop reason + usage),
`message_stop`, `ping`, `error`.

Request mapping:

| Anthropic | OpenAI |
|---|---|
| `system` | leading system message |
| `tool_use` block | assistant `tool_calls` |
| `tool_result` block | separate `role:"tool"` message |
| `tools[]` | function tools |
| `tool_choice` auto/any/tool | `auto` / `required` / named function |
| `thinking` | `reasoning_effort` heuristic |
| `stop_sequences` | `stop` |
| `stream` | `stream` + `stream_options.include_usage` |

## TypeSafe System One

Request:

```json
{
  "model": "typesafe/jev-1.14.0",
  "state": "string | object | array",
  "questions": {
    "urgent": {"type": "noul", "instructions": "...", "criteria": {"true": "...", "false": "..."}},
    "team": {"type": "choice", "instructions": "...", "criteria": {"billing": "..."}},
    "frustration": {"type": "score", "instructions": "...", "criteria": ["Calm", "Mad"]}
  }
}
```

Response:

```json
{
  "model": "typesafe/jev",
  "answers": {
    "urgent": {"type": "noul", "noul": 0.96},
    "team": {"type": "choice", "choice": "billing", "confidence": 0.89,
             "probabilities": {"billing": 0.91, "shipping": 0.09}},
    "frustration": {"type": "score", "score": 1.41, "confidence": 0.5,
                    "legend": {"0": "Calm", "1": "Mad"}, "probabilities": {"0": 0.2, "1": 0.8}}
  },
  "usage": {"input_tokens": 84, "output_tokens": 3}
}
```

Limits: 100 questions, 64K total, 32K state + longest question. Output tokens are
currently free.
