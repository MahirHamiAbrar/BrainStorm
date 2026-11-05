# BrainStorm LLM

`brainstorm_llm` is BrainStorm's provider-neutral LLM layer.

Every provider is normalized to the **Anthropic Messages** shape:

- Requests: `MessageParam`, `TextBlock`, `ImageBlock`, `ToolUseBlock`, `ToolResultBlock`.
- Responses: `Message`, `Usage`, `stop_reason`.
- Streaming: Anthropic stream events (`message_start`, `content_block_delta`, `message_delta`, …).

OpenAI-style providers (OpenRouter, Ollama, Gemini, most Command Code models)
are translated both ways. Anthropic-style endpoints pass through unchanged.

## Install

The package ships with the project. Declare it as a direct dependency:

```toml
dependencies = [
    "httpx>=0.28",
    "orjson>=3.13",
    "pydantic>=2.14",
    "selectolax>=1.0",
    "loguru>=0.7",
]
```

```bash
uv sync
```

## Environment

| Variable | Used by |
|---|---|
| `COMMAND_API_KEY` or `CMD_API_KEY` | Command Code LLM + decisions |
| `OPENROUTER_API_KEY` | OpenRouter |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Gemini |
| `OPENCODE_API_KEY` | OpenCode Console |
| `TYPESAFE_API_KEY` | TypeSafe direct |
| `OLLAMA_API_KEY` | Ollama Cloud (optional; local is keyless) |
| `BRAINSTORM_CACHE_DIR` | Pricing cache directory override |

## Quick start

```python
from brainstorm_llm import BrainStormLLM

llm = BrainStormLLM(model="deepseek/deepseek-v4.1-flash")
message = llm.generate(
    [{"role": "user", "content": "Explain event loops in one paragraph."}],
    max_tokens=300,
)
print(message.content[0].text)
print(message.usage.input_tokens, message.usage.output_tokens)
```

## Design

```
BrainStormLLM / BrainStormS1
        │
        ├── provider registry
        │     ├── commandcode   (OpenAI chat + Anthropic messages + systemone)
        │     ├── openrouter    (OpenAI chat)
        │     ├── ollama        (OpenAI chat)
        │     ├── gemini        (OpenAI-compatible endpoint)
        │     ├── opencode      (OpenAI chat + Anthropic messages + systemone)
        │     └── typesafe      (systemone)
        │
        ├── conversion layer    (OpenAI <-> Anthropic, streaming state machine)
        ├── transport           (httpx sync/async, retries, SSE)
        ├── schemas             (Pydantic v2)
        ├── pricing             (live sources + disk cache)
        └── usage               (callbacks, CostTracker)
```

No `anthropic`, `openai`, or `typesafe-sdk` package is required at runtime.

Continue with [quickstart.md](./quickstart.md).
