# Quickstart

## Synchronous generation

```python
from brainstorm_llm import BrainStormLLM, CallUsage

def on_usage(record: CallUsage) -> None:
    print(record.model, record.usage.input_tokens, record.cost_usd)

llm = BrainStormLLM(
    model="deepseek/deepseek-v4.1-flash",   # default Command Code model
    zdr=True,                               # x-cmd-zdr: 1 (default for Command Code)
    on_usage=on_usage,
)

message = llm.generate(
    [{"role": "user", "content": "Write a haiku about race conditions."}],
    system="You are concise.",
    max_tokens=256,
    temperature=0.7,
)

print(message.text)
print(message.stop_reason, message.usage)
```

## Streaming

```python
for event in llm.stream(
    [{"role": "user", "content": "Count to five."}],
    max_tokens=64,
):
    if event.type == "content_block_delta" and event.delta.type == "text_delta":
        print(event.delta.text, end="", flush=True)
```

Text-only convenience:

```python
text = "".join(llm.stream_text([{"role": "user", "content": "Hello"}]))
```

Collect the final assembled message:

```python
message = llm.stream_collect([{"role": "user", "content": "Hello"}])
print(message.text)
```

## Async

```python
async with BrainStormLLM(model="deepseek/deepseek-v4.1-flash") as llm:
    message = await llm.agenerate([{"role": "user", "content": "Hello"}])

    async for event in llm.astream([{"role": "user", "content": "Count to five."}]):
        ...

    text = "".join([part async for part in llm.astream_text([{"role": "user", "content": "Hi"}])])
    final = await llm.astream_collect([{"role": "user", "content": "Hi"}])
```

## Decisions

```python
from brainstorm_llm import BrainStormS1, Noul, Choice, Score

s1 = BrainStormS1()   # typesafe/jev-1.14.0 on Command Code

result = s1.system_one(
    "My payments have failed for three days and I am losing sales.",
    {
        "urgent": Noul(instructions="Does this need urgent attention?"),
        "team": Choice(
            instructions="Which team should handle it?",
            criteria={"billing": "payments", "shipping": "delivery"},
        ),
        "frustration": Score(
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated", "Very angry"],
        ),
    },
)

print(result.nouls["urgent"].noul)        # probability of yes
print(result.choices["team"].choice)     # selected label
print(result.scores["frustration"].score)
```

## Choosing a provider

```python
BrainStormLLM(provider="commandcode", model="deepseek/deepseek-v4.1-flash")
BrainStormLLM(provider="openrouter", model="deepseek/deepseek-v4.1-flash")
BrainStormLLM(provider="gemini", model="gemini-3.8-flash")
BrainStormLLM(provider="opencode", model="deepseek-v4.1-flash")
BrainStormLLM(provider="ollama", model="llama3.2")
```

See [providers.md](./providers.md) for keys, base URLs, and routing.
