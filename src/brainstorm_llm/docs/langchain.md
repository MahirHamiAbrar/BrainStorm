# LangChain and LangGraph

The integration requires `langchain-core` (installed automatically with
`langchain`). It does not add an `openai` or `anthropic` dependency.

```python
from brainstorm_llm.integrations.langchain import ChatBrainStorm
```

## Chat model

`ChatBrainStorm` is a standard `BaseChatModel`:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from brainstorm_llm.integrations.langchain import ChatBrainStorm

model = ChatBrainStorm(
    model="deepseek/deepseek-v4.1-flash",   # default
    provider="commandcode",
    max_tokens=1024,
    temperature=0.2,
)

response = model.invoke([
    SystemMessage("You are concise."),
    HumanMessage("Explain event loops in one paragraph."),
])
print(response.text)
print(response.usage_metadata)
```

Provider connection fields mirror `BrainStormLLM`: `api_key`, `base_url`,
`timeout`, `max_retries`, `zdr`, `pricing`, `headers`, and `extra_body`.
You can also inject a preconfigured client:

```python
from brainstorm_llm import BrainStormLLM

llm = BrainStormLLM(model="deepseek/deepseek-v4.1-flash")
model = ChatBrainStorm(client=llm)
# The adapter uses llm.model and llm.provider_name.
```

## Embeddings

`BrainStormEmbeddings` is a standard LangChain `Embeddings` implementation. It
works with the providers that expose an embeddings endpoint (`ollama` and
`gemini`), and uses the client's pooled sync and async transports.

```python
from brainstorm_llm.integrations.langchain import BrainStormEmbeddings

embeddings = BrainStormEmbeddings(
    model="nomic-embed-text",
    provider="ollama",
)

vectors = embeddings.embed_documents(["first document", "second document"])
query = embeddings.embed_query("first document")

print(len(vectors), len(query))
```

Constructor fields:

| Field | Default | Description |
|---|---|---|
| `model` | `None` | Embedding model id; required unless `client=` is supplied |
| `provider` | `"commandcode"` | Provider id used when no `client=` is given |
| `api_key` | `None` | Overrides the provider's environment variable |
| `base_url` | `None` | Overrides the provider's default endpoint |
| `timeout` | `60.0` | Request timeout in seconds |
| `max_retries` | `2` | Retries for transient provider failures |
| `dimensions` | `None` | Requested vector size, when the model supports truncation |
| `encoding_format` | `None` | `"float"` or `"base64"`; `None` leaves the provider default |
| `headers` | `None` | Extra headers for every request |
| `extra_body` | `None` | Provider-specific request fields merged into the body |
| `client` | `None` | Preconfigured `BrainStormLLM`; its provider is always used |

The default `provider` is `commandcode`, which has no embeddings endpoint, so
pass `provider="ollama"` / `provider="gemini"` or inject a client.

Methods:

| Method | Returns |
|---|---|
| `embed_documents(texts)` | `list[list[float]]`, one vector per document; an empty list returns `[]` without a request |
| `embed_query(text)` | `list[float]` for one query text |
| `aembed_documents(texts)` | Native async version of `embed_documents` |
| `aembed_query(text)` | Native async version of `embed_query` |
| `close()` / `aclose()` | Close the client, but only when this adapter owns it |

To reuse a configured client, inject it. The injected client's provider always
wins, and `model=` still overrides its model:

```python
from brainstorm_llm import BrainStormLLM
from brainstorm_llm.integrations.langchain import BrainStormEmbeddings

llm = BrainStormLLM(provider="ollama", model="nomic-embed-text")

embeddings = BrainStormEmbeddings(client=llm)                # provider + model from the client
override = BrainStormEmbeddings("all-minilm", client=llm)    # model= wins, provider stays ollama
```

An injected client is never closed by the adapter, so `close()` and `aclose()`
are no-ops in that case.

Any LangChain component that accepts an `Embeddings` instance takes
`BrainStormEmbeddings` as-is. A minimal vector-store-style flow: embed the
documents once, embed the query, and rank by cosine similarity.

```python
import math

documents = [
    "Event loops schedule callbacks.",
    "Embeddings map text to coordinates.",
    "Retrievers rank documents against a query.",
]

index = embeddings.embed_documents(documents)
query = embeddings.embed_query("How do event loops work?")

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0

for score, text in sorted(
    ((cosine(query, vector), text) for vector, text in zip(index, documents)),
    reverse=True,
)[:2]:
    print(f"{score:.3f} {text}")
```

Async stores call `await embeddings.aembed_documents(...)` and
`await embeddings.aembed_query(...)` instead. Failures are mapped through the
same `raise_langchain_error` path as the chat model, so retry middleware sees
`ModelAuthenticationError`, `ModelRateLimitError`, `ModelAPIError`, and the
other standard exceptions. Provider support and request options are covered in
[embeddings.md](./embeddings.md).

## Tools

`bind_tools` accepts LangChain tools, callables, Pydantic classes, OpenAI tool
dicts, and Anthropic tool dicts:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"

model_with_tools = model.bind_tools([get_weather], tool_choice="auto")
response = model_with_tools.invoke("What is the weather in Paris?")

for call in response.tool_calls:
    print(call["name"], call["args"], call["id"])
```

Tool results flow back through normal `ToolMessage` objects, so LangGraph's
`ToolNode` and agent loops work unchanged.

## Structured output

`with_structured_output` uses the standard tool-calling method:

```python
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    year: int
    rating: float

structured = model.with_structured_output(Movie)
movie = structured.invoke("Describe Inception.")
print(movie.title, movie.year)
```

## Streaming

```python
for chunk in model.stream("Count to five."):
    print(chunk.text, end="", flush=True)
```

`AIMessageChunk` values carry text, reasoning blocks, tool-call chunks, usage
metadata, and `chunk_position="last"` on the final chunk. Aggregate chunks with
`+` as usual:

```python
chunks = list(model.stream("Hello"))
full = chunks[0]
for chunk in chunks[1:]:
    full += chunk
print(full.text)
```

Async works through the standard APIs:

```python
response = await model.ainvoke("Hello")
async for chunk in model.astream("Hello"):
    ...
```

## LangGraph

Any LangChain graph accepts the model directly:

```python
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

model_with_tools = model.bind_tools([get_weather])

def call_model(state: MessagesState) -> dict:
    return {"messages": [model_with_tools.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode([get_weather]))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()

result = app.invoke({"messages": [{"role": "user", "content": "Weather in Paris?"}]})
print(result["messages"][-1].text)
```

LangChain's newer `create_agent` / `create_react_agent` helpers work the same
way because the adapter implements the normal chat-model contract.

## Decision models

LangChain supports decision models (for example `OpenAIDecisions`). BrainStorm
mirrors that API with `BrainStormDecisions`:

```python
from brainstorm_llm.integrations.langchain import (
    BrainStormDecisions,
    Choice,
    Level,
    Predicate,
    Score,
)

decisions = BrainStormDecisions()   # typesafe/jev-1.14.0 on Command Code

result = decisions.invoke({
    "input": "Production is down. Please help immediately.",
    "questions": {
        "urgent": Predicate(instructions="Is this request urgent?"),
        "team": Choice(
            instructions="Which team should handle it?",
            choices={"billing": "Payments", "technical": "Product bugs"},
        ),
        "severity": Score(
            instructions="How severe is the issue?",
            levels=[Level(label="Cosmetic"), "Workaround", "Blocked"],
        ),
    },
})

print(result.predicates["urgent"].probability)
print(result.choices["team"].choice)
print(result.scores["severity"].score)
```

`BrainStormDecisions` is a normal `Runnable`: it composes in chains and can be
used as a LangGraph node. Inputs are `{"input": <state>, "questions": {...}}`;
the state can be text, a message, a JSON object, or a list. Base64 images in
message content are extracted into System One assets automatically.

### Decision tool

For agents, bind fixed questions as a tool with a `state` input:

```python
tool = decisions.make_tool(
    {
        "urgent": Predicate(instructions="Is this request urgent?"),
        "team": Choice(
            instructions="Which team?",
            choices={"billing": "Payments", "technical": "Product bugs"},
        ),
    },
    name="triage",
)

tool.invoke("Production is down.")
# -> JSON string with model, answers, usage, request_id
```

Async invocation is supported: `await decisions.ainvoke(...)` and
`await tool.ainvoke(...)`.

## Errors, usage, and ZDR

BrainStorm transport failures are re-raised as LangChain's standard model
exceptions (`ModelAuthenticationError`, `ModelRateLimitError`,
`ModelTimeoutError`, `ModelConnectionError`, `ModelAPIError`,
`ModelInvalidRequestError`, `ContextOverflowError`), so retry middleware and
agent error handling work normally.

Usage is exposed through `AIMessage.usage_metadata` and
`DecisionResponse.usage`; response metadata also carries provider, request id,
and estimated cost when pricing is enabled.

Command Code ZDR behavior is unchanged: `ChatBrainStorm` uses the client default
(`x-cmd-zdr: 1`), while `BrainStormDecisions` defaults it off because
`/systemone` refuses the header.

## Notes

- Only client-side function tools are supported; provider-hosted tools
  (`web_search`, code interpreter, etc.) pass through the core library's
  limitations.
- Use `ChatBrainStorm` directly. It is not in LangChain's built-in
  `init_chat_model` partner registry; any chain, graph, or agent accepts the
  instance because it implements `BaseChatModel`.
- `reasoning_effort` is mapped to an Anthropic thinking budget on the Anthropic
  wire and to the provider's `reasoning_effort` on OpenAI-compatible wires.
- Streaming usage is emitted on the final `AIMessageChunk`; aggregate chunks
  with `+` to recover the complete message and usage.
