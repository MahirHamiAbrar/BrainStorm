# Embeddings

`embed()` and `aembed()` turn text into vectors. Requests go to the provider's
OpenAI-compatible `/embeddings` endpoint and come back as a normalized
`EmbeddingResponse`.

## Providers

| Provider | Embeddings | Notes |
|---|---|---|
| `ollama` | yes | Local or Ollama Cloud; chat and embedding models share the `/v1` base URL |
| `gemini` | yes | OpenAI-compatible endpoint, e.g. `text-embedding-004` |
| `commandcode` | no | Chat, Anthropic messages, and System One only |
| `openrouter` | no | Chat completions only |
| `opencode` | no | Chat and Anthropic messages only |
| `typesafe` | no | Decision models only |

Only `ollama` and `gemini` implement the embeddings wire. Every other provider
raises `ConfigurationError("Provider '<name>' does not support embeddings")`
before a request is made. See [providers.md](./providers.md) for keys and base
URLs.

## Synchronous example

```python
from brainstorm_llm import BrainStormLLM

llm = BrainStormLLM(provider="ollama", model="nomic-embed-text")

response = llm.embed([
    "The cat sat on the mat.",
    "Event loops schedule callbacks.",
])

print(len(response.vectors))              # 2
print(response.dimensions)                # 768
print(response.usage.prompt_tokens)
```

`input` also accepts a single string, which produces one vector.

## Async example

```python
import asyncio

from brainstorm_llm import BrainStormLLM


async def main() -> None:
    async with BrainStormLLM(
        provider="gemini",
        model="text-embedding-004",
    ) as llm:
        response = await llm.aembed("Explain event loops in one paragraph.")
        print(response.dimensions, response.request_id)


asyncio.run(main())
```

`aembed()` uses the client's pooled async transport; it does not wrap the sync
call in a thread.

## Response

| Attribute | Type | Description |
|---|---|---|
| `vectors` | `list[list[float]]` | One vector per input text, sorted by response `index`. |
| `dimensions` | `int` \| `None` | Width of the returned vectors; `None` when the response is empty. |
| `usage` | `EmbeddingUsage` | `prompt_tokens` and `total_tokens`; embeddings have no output tokens. |
| `request_id` | `str` \| `None` | Upstream request id when the provider sends one. |
| `data` | `list[Embedding]` | Raw entries with `object="embedding"`, `index`, and `embedding`. |
| `model` | `str` | Model that produced the vectors. |
| `provider` | `str` \| `None` | Provider id that served the request. |
| `object` | `str` | Always `"list"`, matching the OpenAI response shape. |

```python
response = llm.embed("hello")

response.vectors[0][:4]          # first four dimensions of the first vector
response.dimensions              # 768
response.usage.total_tokens
response.request_id              # x-request-id when present
```

Base64 responses are decoded transparently: `Embedding.embedding` accepts an
OpenAI base64 string and unpacks it as little-endian `float32`, so
`response.vectors` always contains floats regardless of `encoding_format`.

`Embedding`, `EmbeddingUsage`, `EmbeddingResponse`, `EmbeddingCreateParams`, and
`EmbeddingInput` are exported from both `brainstorm_llm` and
`brainstorm_llm.schemas`.

## Model selection

`model=` overrides the client's configured model for one call. When it is
omitted, `embed()` sends `BrainStormLLM.model`:

```python
llm = BrainStormLLM(provider="ollama", model="llama3.2")   # chat model
response = llm.embed("hello", model="nomic-embed-text")     # embedding model
```

The client does not consult the catalog before sending, so a client configured
for chat can embed with an explicit `model=`. Use `list_embedding_models()` to
find valid ids.

## Options

| Option | Type | Description |
|---|---|---|
| `dimensions` | `int` \| `None` | Requested vector size, for models that support truncation; must be greater than zero. |
| `encoding_format` | `"float"` \| `"base64"` \| `None` | Wire format to request; `None` leaves the provider default (float). Decoding is identical either way. |
| `user` | `str` \| `None` | End-user identifier forwarded to the provider. |
| `extra_body` | `dict` \| `None` | Provider-specific fields merged into the request body last. |
| `headers` | `Mapping[str, str]` \| `None` | Extra headers for this call only. |

The options work on both `embed()` and `aembed()`; Gemini-specific fields such
as `task_type` go through `extra_body`:

```python
response = await llm.aembed(
    "hello",
    model="text-embedding-004",
    dimensions=256,
    encoding_format="float",
    user="reader-42",
    extra_body={"task_type": "RETRIEVAL_QUERY"},
    headers={"x-trace": "abc123"},
)
```

`extra_body` is merged after the standard fields, so it can override `model`,
`input`, or any option above.

## Discovering embedding models

`list_embedding_models()` returns only catalog entries with `kind="embedding"`.
Pass `force_refresh=True` to bypass the one-hour catalog cache:

```python
from brainstorm_llm import BrainStormLLM

llm = BrainStormLLM(provider="ollama", model="llama3.2")

for model in llm.list_embedding_models():
    print(model.id, model.kind)          # nomic-embed-text:latest embedding

fresh = llm.list_embedding_models(force_refresh=True)
```

Catalogs classify ids containing `embed` as `kind="embedding"` — for example
`nomic-embed-text` and `text-embedding-004` — along with common embedding
families such as `all-minilm`, `bge-*`, `e5-*`, and `gte-*`. The full
`list_models()` catalog still returns every entry, with `ModelKind` now
`"chat"`, `"decision"`, or `"embedding"`. See
[models-and-pricing.md](./models-and-pricing.md).

Discovery only reads the provider catalog; it never calls `/embeddings`.
Providers whose catalog has no embedding entries return an empty list.

## Errors

| Situation | Error |
|---|---|
| Provider has no embeddings endpoint | `ConfigurationError("Provider '<name>' does not support embeddings")` |
| Empty string, empty sequence, or a blank entry in the list | `ConfigurationError` |
| `dimensions` not greater than zero | `ConfigurationError` |

```python
from brainstorm_llm import BrainStormLLM

llm = BrainStormLLM(provider="commandcode")
llm.embed("hello")
# ConfigurationError: Provider 'commandcode' does not support embeddings

llm = BrainStormLLM(provider="ollama", model="nomic-embed-text")
llm.embed([])
# ConfigurationError: embed() requires at least one input text
```

No HTTP request is made for these failures. Provider-side status errors keep the
normal mapping — `AuthenticationError`, `RateLimitError`, `BadRequestError`, and
so on — as described in [errors-and-retries.md](./errors-and-retries.md).

## LangChain

`BrainStormEmbeddings` implements LangChain's `Embeddings` interface on top of
the same client:

```python
from brainstorm_llm.integrations.langchain import BrainStormEmbeddings

embeddings = BrainStormEmbeddings(
    model="nomic-embed-text",
    provider="ollama",
)

documents = embeddings.embed_documents(["first document", "second document"])
query = embeddings.embed_query("first document")

print(len(documents), len(query))
embeddings.close()
```

Both methods have native async counterparts: `await embeddings.aembed_documents(...)`
and `await embeddings.aembed_query(...)`. An injected `client=` reuses its
provider and, unless `model=` is passed, its model. See
[langchain.md](./langchain.md#embeddings) for constructor fields and a
vector-store-style example.
