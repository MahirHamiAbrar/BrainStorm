from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import ModelAuthenticationError
from langchain_core.vectorstores import InMemoryVectorStore

from brainstorm_llm import BrainStormLLM
from brainstorm_llm.errors import ConfigurationError
from brainstorm_llm.integrations.langchain import BrainStormEmbeddings
from brainstorm_llm.providers.gemini import GeminiProvider
from brainstorm_llm.providers.ollama import OllamaProvider
from tests.helpers import openai_embeddings


def make_embeddings(
    handler,
    *,
    async_handler=None,
    model: str | None = None,
    **client_kwargs,
) -> BrainStormEmbeddings:
    provider = OllamaProvider(
        transport=httpx.MockTransport(handler),
        async_transport=httpx.MockTransport(async_handler or handler),
        **client_kwargs,
    )
    client = BrainStormLLM(
        model=model or "nomic-embed-text",
        provider=provider,
        pricing="off",
    )
    return BrainStormEmbeddings(client=client)


def test_embed_documents_and_query() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=openai_embeddings([0.1, 0.2], [0.3, 0.4]))

    embeddings = make_embeddings(handler)

    assert isinstance(embeddings, Embeddings)
    assert embeddings.embed_documents(["first", "second"]) == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    assert seen["body"]["input"] == ["first", "second"]

    assert embeddings.embed_query("first") == [0.1, 0.2]
    assert seen["body"]["input"] == ["first"]
    assert embeddings.model == "nomic-embed-text"
    assert embeddings.provider == "ollama"
    assert "BrainStormEmbeddings" in repr(embeddings)


def test_embed_documents_empty_returns_empty_without_request() -> None:
    embeddings = make_embeddings(lambda request: httpx.Response(500))
    assert embeddings.embed_documents([]) == []


async def test_async_embeddings_use_async_transport() -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=openai_embeddings([1.0, 2.0]))

    embeddings = make_embeddings(
        lambda request: httpx.Response(500),
        async_handler=handler,
    )

    assert await embeddings.aembed_documents(["first"]) == [[1.0, 2.0]]
    assert seen["body"]["input"] == ["first"]
    assert await embeddings.aembed_query("query") == [1.0, 2.0]
    assert seen["body"]["input"] == ["query"]


def test_model_is_required_without_client() -> None:
    with pytest.raises(ConfigurationError, match="requires model="):
        BrainStormEmbeddings()


def test_options_and_model_override_are_forwarded() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json=openai_embeddings([0.5, 0.6]))

    provider = GeminiProvider(
        api_key="secret",
        transport=httpx.MockTransport(handler),
    )
    client = BrainStormLLM(model="gemini-3.8-flash", provider=provider, pricing="off")
    embeddings = BrainStormEmbeddings(
        "text-embedding-004",
        client=client,
        dimensions=2,
        encoding_format="float",
        headers={"x-test": "yes"},
        extra_body={"task_type": "RETRIEVAL_DOCUMENT"},
    )

    assert embeddings.model == "text-embedding-004"
    assert embeddings.provider == "gemini"
    assert embeddings.embed_documents(["document"]) == [[0.5, 0.6]]
    assert seen["body"] == {
        "model": "text-embedding-004",
        "input": ["document"],
        "dimensions": 2,
        "encoding_format": "float",
        "task_type": "RETRIEVAL_DOCUMENT",
    }
    assert seen["headers"]["x-test"] == "yes"
    assert seen["headers"]["authorization"] == "Bearer secret"


def test_error_mapping_to_langchain_exceptions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "bad key", "type": "authentication_error"}},
        )

    embeddings = make_embeddings(handler)
    with pytest.raises(ModelAuthenticationError):
        embeddings.embed_query("hello")


def test_in_memory_vector_store_uses_embeddings() -> None:
    pytest.importorskip("numpy")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        texts = [payload["input"]] if isinstance(payload["input"], str) else payload["input"]
        vectors = [
            [float(len(text)), float(sum(ord(char) for char in text) % 10)]
            for text in texts
        ]
        return httpx.Response(200, json=openai_embeddings(*vectors))

    embeddings = make_embeddings(handler)
    store = InMemoryVectorStore(embeddings)
    store.add_texts(["cat", "dog"])

    results = store.similarity_search("cat", k=1)
    assert results[0].page_content == "cat"
