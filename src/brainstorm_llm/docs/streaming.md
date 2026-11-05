# Streaming

Streaming always exposes Anthropic events, even when the provider speaks OpenAI.

```python
from brainstorm_llm import BrainStormLLM

llm = BrainStormLLM(model="deepseek/deepseek-v4.1-flash")

for event in llm.stream([{"role": "user", "content": "Count to five."}]):
    if event.type == "content_block_delta" and event.delta.type == "text_delta":
        print(event.delta.text, end="", flush=True)
```

Event sequence:

1. `message_start` — message id/model and initial usage
2. `content_block_start` / `content_block_delta` / `content_block_stop` per block
3. `message_delta` — final `stop_reason` and usage
4. `message_stop`

Tool calls arrive as `tool_use` blocks with `input_json_delta` fragments:

```python
for event in llm.stream(...):
    if event.type == "content_block_start" and event.content_block.type == "tool_use":
        print("tool:", event.content_block.name)
    elif event.type == "content_block_delta" and event.delta.type == "input_json_delta":
        print(event.delta.partial_json, end="")
```

## Text-only stream

```python
for chunk in llm.stream_text([{"role": "user", "content": "Hello"}]):
    print(chunk, end="")
```

## Assemble the final message

```python
message = llm.stream_collect([{"role": "user", "content": "Hello"}])
print(message.text, message.stop_reason, message.usage)
```

`stream_collect` uses `MessageAccumulator`, which parses streamed tool arguments
into `ToolUseBlock.input`.

## Manual accumulation

```python
from brainstorm_llm import MessageAccumulator

accumulator = MessageAccumulator()
for event in llm.stream(...):
    accumulator.feed(event)
message = accumulator.finalize()
```

## Async

```python
async for event in llm.astream([{"role": "user", "content": "Hello"}]):
    ...

async for chunk in llm.astream_text([{"role": "user", "content": "Hello"}]):
    ...

message = await llm.astream_collect([{"role": "user", "content": "Hello"}])
```

## Usage during streaming

The usage callback fires once, after `message_stop`, with merged usage from
`message_start` and `message_delta`:

```python
from brainstorm_llm import CallUsage

def on_usage(record: CallUsage) -> None:
    print(record.streaming, record.usage.input_tokens, record.usage.output_tokens)

for event in llm.stream([...], on_usage=on_usage):
    ...
```

If the consumer stops early, the callback is not fired for that partial stream.
