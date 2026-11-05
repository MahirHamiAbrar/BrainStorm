from __future__ import annotations

import base64
import json
import struct

import httpx
import pytest

from brainstorm_llm import BrainStormLLM
from brainstorm_llm.errors import ConfigurationError
from brainstorm_llm.providers.commandcode import CommandCodeProvider
from brainstorm_llm.providers.gemini import GeminiProvider
from brainstorm_llm.providers.ollama import OllamaProvider
from brainstorm_llm.schemas import EmbeddingResponse
from tests.helpers import model_list, openai_embeddings


def test_ollama_embed_sync_request_and_response() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        seen["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json=openai_embeddings([0.1, 0.2], [0.3, 0.4]),
            headers={"x-request-id": "emb-1"},
        )

    provider = OllamaProvider(transport=httpx.MockTransport(handler))
    llm = BrainStormLLM(model="nomic-embed-text", provider=provider, pricing="off")
    response = llm.embed(["first", "second"])

    assert seen["path"] == "/v1/embeddings"
    assert seen["body"] == {
        "model": "nomic-embed-text",
        "input": ["first", "second"],
    }
    assert "authorization" not in seen["headers"]
    assert response.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert response.dimensions == 2
    assert response.model == "embed-model"
    assert response.provider == "ollama"
    assert response.usage.prompt_tokens == 4
    assert response.usage.total_tokens == 4
    assert response.request_id == "emb-1"


async def test_gemini_async_embed_forwards_options() -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        seen["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json=openai_embeddings([1.0, 2.0], model="text-embedding-004"),
        )

    provider = GeminiProvider(
        api_key="secret",
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        async_transport=httpx.MockTransport(handler),
    )
    llm = BrainStormLLM(model="gemini-3.8-flash", provider=provider, pricing="off")

    async with llm:
        response = await llm.aembed(
            "hello",
            model="text-embedding-004",
            dimensions=2,
            encoding_format="float",
            user="reader",
            extra_body={"task_type": "RETRIEVAL_QUERY"},
            headers={"x-test": "1"},
        )

    assert str(seen["path"]).endswith("/embeddings")
    assert seen["body"] == {
        "model": "text-embedding-004",
        "input": "hello",
        "dimensions": 2,
        "encoding_format": "float",
        "user": "reader",
        "task_type": "RETRIEVAL_QUERY",
    }
    assert seen["headers"]["authorization"] == "Bearer secret"
    assert seen["headers"]["x-test"] == "1"
    assert response.vectors == [[1.0, 2.0]]


def test_embed_uses_client_model_by_default() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=openai_embeddings([0.5]))

    provider = OllamaProvider(transport=httpx.MockTransport(handler))
    llm = BrainStormLLM(model="embed-default", provider=provider, pricing="off")
    llm.embed("hello")

    assert seen["body"]["model"] == "embed-default"


def test_provider_without_embeddings_raises() -> None:
    provider = CommandCodeProvider(
        api_key="test-key",
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError("no request expected"))
        ),
    )
    llm = BrainStormLLM(model="chat-model", provider=provider, pricing="off")

    with pytest.raises(ConfigurationError, match="does not support embeddings"):
        llm.embed("hello")


def test_empty_input_raises_configuration_error() -> None:
    provider = OllamaProvider(
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError("no request expected"))
        )
    )
    llm = BrainStormLLM(model="nomic-embed-text", provider=provider, pricing="off")

    with pytest.raises(ConfigurationError):
        llm.embed("")
    with pytest.raises(ConfigurationError):
        llm.embed([])
    with pytest.raises(ConfigurationError):
        llm.embed(["valid", " "])
    with pytest.raises(ConfigurationError):
        llm.embed("hello", dimensions=0)


def test_embedding_models_are_classified_in_catalog() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=model_list(
                {"id": "nomic-embed-text:latest"},
                {"id": "llama3.2"},
                {"id": "all-minilm"},
            ),
        )

    provider = OllamaProvider(transport=httpx.MockTransport(handler))
    llm = BrainStormLLM(model="llama3.2", provider=provider, pricing="off")

    assert [model.id for model in provider.embedding_models()] == [
        "nomic-embed-text:latest",
        "all-minilm",
    ]
    assert [model.id for model in llm.list_embedding_models()] == [
        "nomic-embed-text:latest",
        "all-minilm",
    ]
    assert all(model.kind == "embedding" for model in llm.list_embedding_models())


def test_embedding_response_sorts_vectors_and_decodes_base64() -> None:
    raw = base64.b64encode(struct.pack("<2f", 1.5, -2.0)).decode()
    response = EmbeddingResponse.model_validate(
        {
            "model": "m",
            "data": [
                {"index": 1, "embedding": [3.0]},
                {"index": 0, "embedding": raw},
            ],
        }
    )

    assert response.vectors == [[1.5, -2.0], [3.0]]
    assert response.dimensions == 2


def test_gemini_catalog_marks_embedding_models() -> None:
    payload = model_list(
        {"id": "models/text-embedding-004"},
        {"id": "models/gemini-3.8-flash"},
    )
    provider = GeminiProvider(
        api_key="x",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    models = provider.list_models()

    assert models[0].id == "text-embedding-004"
    assert models[0].kind == "embedding"
    assert models[1].kind == "chat"
