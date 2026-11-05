# Providers

| Provider | Wire | Base URL | Key |
|---|---|---|---|
| `commandcode` | OpenAI chat or Anthropic messages, per model | `https://api.commandcode.ai/provider/v1` | `COMMAND_API_KEY` / `CMD_API_KEY` |
| `openrouter` | OpenAI chat | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `ollama` | OpenAI chat | `http://localhost:11434/v1` | none (local) / `OLLAMA_API_KEY` |
| `gemini` | OpenAI-compatible | `https://generativelanguage.googleapis.com/v1beta/openai` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| `opencode` | OpenAI chat or Anthropic messages | `https://opencode.ai/inference` | `OPENCODE_API_KEY` |
| `typesafe` | System One | `https://api.typesafe.ai/v1` | `TYPESAFE_API_KEY` |

## Command Code

Command Code serves both wires. The client asks `/models` once, then routes:

- a model supporting `/chat/completions` → OpenAI wire, translated to Anthropic;
- a model that only supports `/messages` (Claude models) → native Anthropic wire.

If the catalog is unreachable it falls back to the model name: `claude*` goes to
`/messages`, everything else to `/chat/completions`.

```python
llm = BrainStormLLM(provider="commandcode", model="deepseek/deepseek-v4.1-flash")
claude = BrainStormLLM(provider="commandcode", model="claude-sonnet-4-6")
```

Both return the same `Message` type.

### Zero data retention

`BrainStormLLM` sends `x-cmd-zdr: 1` by default for Command Code. Disable it with
`zdr=False`. Models without a ZDR upstream fail with `ZDRUnavailableError` (422
`cmd_zdr_no_providers`).

`BrainStormS1` defaults to ZDR **off** because Command Code refuses the header on
`/systemone` (`typesafe/jev` has no ZDR-capable upstream). Passing `zdr=True`
sends it and surfaces the 422 as `ZDRUnavailableError`.

## OpenRouter

```python
llm = BrainStormLLM(
    provider="openrouter",
    model="deepseek/deepseek-v4.1-flash",
    headers={"HTTP-Referer": "https://your.app", "X-Title": "BrainStorm"},
)
```

## Ollama

```python
local = BrainStormLLM(provider="ollama", model="llama3.2")
cloud = BrainStormLLM(
    provider="ollama",
    model="gemma4:31b",
    base_url="https://ollama.com/v1",
)
```

Local inference is free; `model_prices("ollama")` returns zero prices.

## Gemini

```python
llm = BrainStormLLM(provider="gemini", model="gemini-3.8-flash")
```

Uses Gemini's OpenAI-compatible endpoint. Gemini-specific request fields can be
sent through `extra_body`:

```python
llm.generate(
    [{"role": "user", "content": "Hi"}],
    extra_body={"google": {"thinking_config": {"thinking_level": "low"}}},
)
```

## OpenCode

```python
payg = BrainStormLLM(provider="opencode", model="deepseek-v4.1-flash")
go = BrainStormLLM(provider="opencode", model="deepseek-v4.1-flash", plan="go")
```

`plan="go"` targets `https://opencode.ai/inference/go`. Models whose catalog entry
only advertises `messages` use the Anthropic wire; all others use Chat Completions.
The catalog is `https://opencode.ai/inference/v1/models`.

## Custom base URLs

Every provider accepts `base_url=`, `headers=`, `timeout=`, and `max_retries=`.
For a custom OpenAI-compatible endpoint, the quickest path is:

```python
from brainstorm_llm.providers.openai_compat import OpenAIWireMixin
from brainstorm_llm.providers.base import BaseProvider

class MyProvider(OpenAIWireMixin, BaseProvider):
    name = "my-provider"
    default_base_url = "https://llm.example.com/v1"
    api_key_env = ("MY_LLM_KEY",)

    def wire_for(self, model: str) -> str:
        return "openai"

    def model_prices(self, pricing_type="any"):
        from brainstorm_llm import ModelPricing
        raise NotImplementedError

llm = BrainStormLLM(provider=MyProvider(), model="my-model")
```

Only `httpx` is used for transport; no vendor SDK is imported.
