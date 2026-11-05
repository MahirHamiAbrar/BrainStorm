# Tools

Tools use the Anthropic definition shape. OpenAI-wire providers receive
`function` tools automatically.

```python
from brainstorm_llm import Tool, ToolChoiceAuto, ToolChoiceTool, ToolUseBlock

tools = [
    Tool(
        name="get_weather",
        description="Get the current weather for a city.",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"],
        },
    )
]

message = llm.generate(
    [{"role": "user", "content": "What is the weather in Paris?"}],
    tools=tools,
    tool_choice=ToolChoiceAuto(),
    max_tokens=512,
)
```

## Tool choice

```python
ToolChoiceAuto()                              # model decides
ToolChoiceAny()                               # must call at least one tool
ToolChoiceTool(name="get_weather")            # force a specific tool
ToolChoiceNone()                              # disable tools for this call
```

`disable_parallel_tool_use=True` maps to OpenAI `parallel_tool_calls=false`.

## Reading tool calls

```python
for block in message.content:
    if isinstance(block, ToolUseBlock):
        print(block.id, block.name, block.input)
```

`stop_reason` is `"tool_use"` when the model wants a tool result.

## Returning tool results

```python
from brainstorm_llm import ToolResultBlock

messages = [
    {"role": "user", "content": "What is the weather in Paris?"},
    message,  # assistant message with the ToolUseBlock
    {
        "role": "user",
        "content": [
            ToolResultBlock(
                tool_use_id=message.tool_uses[0].id,
                content='{"temperature_c": 22, "conditions": "sunny"}',
            )
        ],
    },
]

final = llm.generate(messages, tools=tools, max_tokens=512)
```

## Streaming tool calls

Streamed arguments are incremental. `stream_collect()` parses them into a dict;
manual consumers should feed events into `MessageAccumulator`.

```python
message = llm.stream_collect(messages, tools=tools)
for block in message.content:
    if isinstance(block, ToolUseBlock):
        print(block.name, block.input)
```

If an upstream returns invalid JSON arguments, the block input is
`{"_raw_arguments": "<raw string>"}` and a warning is logged.
