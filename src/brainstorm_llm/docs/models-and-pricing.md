# Models and pricing

The catalog and pricing are deliberately separate.

## `list_models()`

Returns model metadata only — no prices.

```python
from brainstorm_llm import list_models

models = list_models("commandcode")
for model in models:
    print(model.id, model.name, model.context_length, model.supported_endpoints)
```

`ModelInfo` fields include `id`, `name`, `provider`, `kind` (`chat` or `decision`),
`context_length`, `max_output_tokens`, `supported_endpoints`, `input_modalities`,
`supports_tools`, `supports_reasoning`, `reasoning_efforts`, and `description`.

Command Code's catalog appends the decision models `typesafe/jev` and
`typesafe/jev-1.14.0` (`kind="decision"`). Pass `include_decision_models=False` to
omit them.

`BrainStormLLM.list_models()` and `BrainStormS1.list_models(kind="decision")`
delegate to the same provider catalog.

## `model_prices()`

Returns per-model prices in **USD per one million tokens**:

```python
from brainstorm_llm import model_prices

pricing = model_prices("commandcode", "any")
print(pricing.pricing_type)                 # "live" or "cached"
print(pricing.prices["deepseek/deepseek-v4.1-flash"])
# input=0.15 output=0.6 cache_read=0.003 cache_write=None notes=None
```

`pricing_type`:

| Value | Behavior |
|---|---|
| `"cached"` | Disk cache only; raises `PricingUnavailableError` if absent |
| `"live"` | Fetch the live source and update the cache |
| `"any"` | Fresh cache if within TTL, otherwise live; stale cache on failure |

The returned `pricing_type` is the source actually used, which is why `"any"` is
useful: it tells you whether the numbers are live or cached.

`pricing.prices` is keyed by **model ID**, e.g.
`"deepseek/deepseek-v4.1-flash"`, `"typesafe/jev-1.14.0"`.

## Cache

Default path:

- Linux/macOS: `~/.cache/brainstorm/pricing.json`
- Windows: `%LOCALAPPDATA%\brainstorm\pricing.json`

Override with `cache_dir=` or `BRAINSTORM_CACHE_DIR`. The file stores every
provider under `providers`, is written atomically, and tolerates corruption.

```python
from pathlib import Path
from brainstorm_llm import clear_pricing_cache, model_prices

model_prices("commandcode", "live", cache_dir=Path("./.cache"))
clear_pricing_cache("commandcode", cache_dir=Path("./.cache"))
```

## Live sources

| Provider | Source |
|---|---|
| `commandcode` | `https://commandcode.ai/docs/resources/pricing-limits` (HTML rows) + `/models` name join |
| `gemini` | `https://aistudio.google.com/docs/pricing.md` (Markdown tables, Standard tab) |
| `openrouter` | `https://openrouter.ai/api/v1/models` `pricing` fields |
| `opencode` | `https://opencode.ai/inference/v1/models` `cost` fields |
| `ollama` | Model list with zero prices (local inference) |

Refresh everything:

```bash
uv run python scripts/refresh_prices.py
```

## Caveats

- Prices are estimates: discounts, peak/off-peak rates, context-tier overrides,
  and batch tiers are not modeled.
- Gemini shows future price changes in the same cell; the parser takes the first
  (current) amount and records the note.
- Command Code's Jev rows are not in `/models`; they are mapped manually.
- `model_prices("ollama")` requires the Ollama server to be reachable for a live
  refresh; cached prices keep working offline.
