# Messages and content blocks

`generate()` accepts Anthropic-style messages. A plain string is shorthand for a
single `TextBlock`.

```python
message = llm.generate(
    [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "Summarize this file."},
    ],
    system="You are a careful assistant.",
    max_tokens=1024,
)
```

## Content blocks

```python
from brainstorm_llm import TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock

messages = [
    {
        "role": "user",
        "content": [
            TextBlock(text="What is in this screenshot?"),
            ImageBlock(
                source={
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "<base64 bytes>",
                }
            ),
        ],
    }
]
```

Images can also use a remote URL:

```python
ImageBlock(source={"type": "url", "url": "https://example.com/diagram.png"})
```

On OpenAI-wire providers, base64 images become `data:` URLs and remote images are
passed through. Documents are not sent over the OpenAI wire in v1.

## System prompts

```python
llm.generate([...], system="You are concise.")
llm.generate([...], system=[TextBlock(text="You are concise.")])
```

OpenAI-wire providers receive the system prompt as a leading `system` message.

## Multi-turn and tool results

`ToolResultBlock` is only sent in a user message and must reference a preceding
`ToolUseBlock.id`:

```python
messages = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "content": [
            ToolUseBlock(id="toolu_1", name="weather", input={"city": "Paris"})
        ],
    },
    {
        "role": "user",
        "content": [
            ToolResultBlock(tool_use_id="toolu_1", content="Sunny, 22 C"),
            TextBlock(text="Thanks — should I bring a jacket?"),
        ],
    },
]
```

OpenAI-wire providers receive one `role: "tool"` message per tool result, followed
by the remaining user content.

## Response

```python
message.text                 # concatenated text blocks
message.content              # list of typed blocks
message.tool_uses            # tool_use blocks
message.stop_reason          # end_turn | max_tokens | tool_use | refusal | ...
message.usage                # input/output/cache token counts
message.request_id           # upstream request id when available
```

## Defaults

`max_tokens` is required by the Anthropic wire and defaults to `4096` in the
client. Pass a smaller value for cheap calls.
