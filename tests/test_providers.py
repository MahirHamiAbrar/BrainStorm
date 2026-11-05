from __future__ import annotations

import json

import httpx
import pytest

from brainstorm_llm import BrainStormLLM
from brainstorm_llm.errors import UnsupportedModelError
from brainstorm_llm.providers import get_provider
from brainstorm_llm.providers.commandcode import CommandCodeProvider
from brainstorm_llm.providers.gemini import GeminiProvider
from brainstorm_llm.providers.ollama import OllamaProvider
from brainstorm_llm.providers.opencode import OpenCodeProvider
from brainstorm_llm.providers.openrouter import OpenRouterProvider
from brainstorm_llm.schemas import MessageAccumulator
from tests.helpers import anthropic_stream_bytes, model_list


def test_registry_and_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(Exception):
        get_provider("openrouter")
    assert get_provider("ollama").name == "ollama"


def test_commandcode_catalog_includes_decisions() -> None:
    def handler(request):
        return httpx.Response(
            200,
            json=model_list(
                {
                    "id": "deepseek/deepseek-v4.1-flash",
                    "name": "DeepSeek V4.1 Flash (latest)",
                    "supported_endpoints": ["/chat/completions"],
                }
            ),
        )

    provider = CommandCodeProvider(api_key="x", transport=httpx.MockTransport(handler))
    models = provider.list_models()
    assert [model.id for model in models][:2] == [
        "deepseek/deepseek-v4.1-flash",
        "typesafe/jev",
    ]
    assert models[1].kind == "decision"
    assert provider.list_models(include_decision_models=False)[0].id == "deepseek/deepseek-v4.1-flash"


def test_commandcode_wire_fallback_from_model_name() -> None:
    def handler(request):
        raise AssertionError("catalog must not be called")

    provider = CommandCodeProvider(api_key="x", transport=httpx.MockTransport(handler), max_retries=0)
    # find_model swallows the transport error and the name heuristic decides the wire.
    assert provider.wire_for("claude-sonnet-4-6") == "anthropic"
    assert provider.wire_for("deepseek/deepseek-v4.1-flash") == "openai"


def test_openrouter_catalog_mapping() -> None:
    payload = {
        "data": [
            {
                "id": "deepseek/deepseek-v4.1-flash",
                "name": "DeepSeek: DeepSeek V4.1 Flash",
                "context_length": 1000000,
                "architecture": {"input_modalities": ["text", "image"]},
                "top_provider": {"context_length": 1000000, "max_completion_tokens": 32000},
                "supported_parameters": ["tools", "reasoning"],
                "reasoning": {"supported_efforts": ["low", "high"]},
                "endpoints": ["/api/v1/chat/completions"],
                "created": 1,
            }
        ]
    }

    provider = OpenRouterProvider(
        api_key="x",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    model = provider.list_models()[0]
    assert model.input_modalities == ["text", "image"]
    assert model.supports_tools is True
    assert model.supports_reasoning is True
    assert model.reasoning_efforts == ["low", "high"]
    assert model.max_output_tokens == 32000


def test_gemini_catalog_strips_models_prefix() -> None:
    payload = {
        "object": "list",
        "data": [
            {"id": "models/gemini-3.8-flash", "object": "model", "created": 1},
            {"id": "gemini-4-argon", "object": "model", "created": 2},
        ],
    }
    provider = GeminiProvider(
        api_key="x",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    assert [model.id for model in provider.list_models()] == [
        "gemini-3.8-flash",
        "gemini-4-argon",
    ]


def test_opencode_catalog_and_wire_routing() -> None:
    payload = {
        "data": [
            {
                "id": "deepseek-v4.1-flash",
                "name": "DeepSeek V4.1 Flash",
                "limit": {"context": 1000000, "output": 384000},
                "modalities": {"input": ["text", "image/png"]},
                "capabilities": {
                    "tools": {"supported": True},
                    "reasoning": {"supported": True, "effort": ["low", "high"]},
                },
                "api": {"chat-completions": {"base_url": "https://opencode.ai/inference/openai/v1"}},
                "created": 1,
            },
            {
                "id": "claude-sonnet-5",
                "name": "Claude Sonnet 5",
                "api": {"messages": {"base_url": "https://opencode.ai/inference/anthropic/v1"}},
            },
        ]
    }
    provider = OpenCodeProvider(
        api_key="x",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    models = provider.list_models()
    assert models[0].id == "deepseek-v4.1-flash"
    assert models[0].context_length == 1000000
    assert models[0].supports_reasoning is True
    assert provider.wire_for("deepseek-v4.1-flash") == "openai"
    assert provider.wire_for("claude-sonnet-5") == "anthropic"


def test_ollama_is_keyless() -> None:
    provider = OllamaProvider(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=model_list({"id": "llama3"}))
        )
    )
    assert provider.api_key is None
    assert [model.id for model in provider.list_models()] == ["llama3"]


def test_ollama_host_gets_v1_suffix() -> None:
    provider = OllamaProvider(host="http://localhost:11434", api_key=None)
    assert provider.base_url == "http://localhost:11434/v1"
    provider.close()


def test_anthropic_stream_keeps_input_tokens() -> None:
    body = anthropic_stream_bytes(text="Hello")

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json=model_list(
                    {
                        "id": "claude-test",
                        "name": "Claude Test",
                        "supported_endpoints": ["/messages"],
                    }
                ),
            )
        return httpx.Response(
            200,
            content=iter([body]),
            headers={"content-type": "text/event-stream"},
        )

    provider = CommandCodeProvider(api_key="x", transport=httpx.MockTransport(handler))
    llm = BrainStormLLM(model="claude-test", provider=provider, pricing="off")
    accumulator = MessageAccumulator()
    for event in llm.stream([{"role": "user", "content": "hi"}]):
        accumulator.feed(event)
    message = accumulator.finalize()
    assert message.text == "Hello"
    assert message.usage.input_tokens == 10
    assert message.usage.output_tokens == 2


def test_commandcode_rejects_unsupported_endpoint() -> None:
    def handler(request):
        return httpx.Response(
            200,
            json=model_list(
                {
                    "id": "weird-model",
                    "name": "Weird",
                    "supported_endpoints": ["/responses"],
                }
            ),
        )

    provider = CommandCodeProvider(api_key="x", transport=httpx.MockTransport(handler))
    assert provider.wire_for("weird-model") == "unsupported"
    with pytest.raises(UnsupportedModelError):
        provider.ensure_model_routable("weird-model")
