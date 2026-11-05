# Usage and costs

Every client can report token usage and estimated cost after each call.

## `CallUsage`

```python
from decimal import Decimal
from brainstorm_llm import CallUsage

record: CallUsage
record.provider       # "commandcode"
record.model          # "deepseek/deepseek-v4.1-flash"
record.request_id     # upstream request id, when available
record.usage          # Usage(input_tokens, output_tokens, cache_read_input_tokens, ...)
record.cost_usd       # Decimal | None (None when pricing is unavailable)
record.pricing_type   # "live" | "cached" | None
record.latency_ms     # request latency
record.streaming      # True for stream/astream
record.operation      # "generate" | "stream" | "system_one"
```

## Callbacks

Set at construction:

```python
def on_usage(record: CallUsage) -> None:
    print(record.model, record.usage.input_tokens, record.cost_usd)

llm = BrainStormLLM(on_usage=on_usage)
```

Or per call (both constructor and per-call callbacks fire if both are set):

```python
llm.generate([...], on_usage=on_usage)
llm.stream([...], on_usage=on_usage)
```

Async callbacks are awaited by `agenerate`, `astream`, `astream_collect`, and
`asystem_one`:

```python
async def on_usage(record: CallUsage) -> None:
    await sink.write(record.model_dump(mode="json"))

llm = BrainStormLLM(on_usage=on_usage)
await llm.agenerate([...])
```

Callback exceptions are logged and swallowed; they never break the model call.

## `CostTracker`

```python
from brainstorm_llm import BrainStormLLM, CostTracker

tracker = CostTracker()
llm = BrainStormLLM(on_usage=tracker)

llm.generate([{"role": "user", "content": "Hello"}])
llm.generate([{"role": "user", "content": "Again"}])

print(tracker.total_usd)       # Decimal
print(tracker.total_tokens)
print(tracker.by_model())      # per-model cost/tokens/calls
tracker.reset()
```

## Pricing modes

```python
BrainStormLLM(pricing="off")      # no pricing lookup; cost_usd is None
BrainStormLLM(pricing="cached")   # disk cache only
BrainStormLLM(pricing="any")      # cache if fresh, otherwise scrape once (default)
BrainStormLLM(pricing="live")     # force a scrape on the first call
```

The first `"any"`/`"live"` call can do a network request to warm the price cache.
Use `"off"` in latency-sensitive paths or prewarm with `model_prices(..., "live")`.

Cost formula:

```
input_tokens/1e6 * input_price
+ output_tokens/1e6 * output_price
+ cache_read_input_tokens/1e6 * cache_read_price
+ cache_creation_input_tokens/1e6 * cache_write_price
```

Anthropic `input_tokens` excludes cache reads; the OpenAI-wire conversion
subtracts `cached_tokens` from `prompt_tokens` to match that convention.
