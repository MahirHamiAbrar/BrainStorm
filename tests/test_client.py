from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from brainstorm_llm import BrainStormLLM, CallUsage, CostTracker, ModelPricing, TokenPrice
from brainstorm_llm.errors import BadRequestError, RateLimitError, ZDRUnavailableError
from brainstorm_llm.pricing.cache import PricingCache
from brainstorm_llm.providers.commandcode import CommandCodeProvider
from brainstorm_llm.schemas import TextBlock, Usage
from brainstorm_llm.schemas.streaming import ContentBlockDeltaEvent, MessageStopEvent
from tests.helpers import (
    AsyncStream,
    SyncStream,
    anthropic_message,
    anthropic_stream_bytes,
    model_list,
    openai_completion,
    sse_bytes,
)


def commandcode_provider(handler, async_handler=None, **kwargs):
    return CommandCodeProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        async_transport=httpx.MockTransport(async_handler or handler),
        **kwargs,
    )


def chat_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/models"):
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
    return httpx.Response(
        200,
        json=openai_completion(content="hello", prompt_tokens=10, completion_tokens=3),
        headers={"x-request-id": "req-1"},
    )


def test_generate_returns_anthropic_message() -> None:
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        if request.method == "POST":
            seen["body"] = json.loads(request.content)
            seen["headers"] = dict(request.headers)
        return chat_handler(request)

    llm = BrainStormLLM(
        model="deepseek/deepseek-v4.1-flash",
        provider=commandcode_provider(handler, zdr=True),
        pricing="off",
    )
    message = llm.generate(
        [{"role": "user", "content": "hi"}],
        system="be concise",
        max_tokens=64,
        temperature=0.2,
    )
    assert isinstance(message.content[0], TextBlock)
    assert message.content[0].text == "hello"
    assert message.request_id == "req-1"
    assert seen["path"].endswith("/chat/completions")
    assert seen["body"]["messages"][0] == {"role": "system", "content": "be concise"}
    assert seen["body"]["max_tokens"] == 64
    assert seen["headers"]["x-cmd-zdr"] == "1"
    assert seen["headers"]["authorization"] == "Bearer test-key"


def test_zdr_can_be_disabled() -> None:
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        return chat_handler(request)

    llm = BrainStormLLM(
        model="deepseek/deepseek-v4.1-flash",
        provider=commandcode_provider(handler, zdr=False),
        pricing="off",
    )
    llm.generate([{"role": "user", "content": "hi"}])
    assert "x-cmd-zdr" not in seen["headers"]


def test_claude_routes_to_messages() -> None:
    paths = []

    def handler(request):
        paths.append(request.url.path)
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
        return httpx.Response(200, json=anthropic_message(text="claude says hi"))

    llm = BrainStormLLM(
        model="claude-test",
        provider=commandcode_provider(handler),
        pricing="off",
    )
    message = llm.generate([{"role": "user", "content": "hi"}])
    assert message.content[0].text == "claude says hi"
    assert any(path.endswith("/messages") for path in paths)


def test_stream_events_and_usage_callback() -> None:
    body = sse_bytes(
        [
            {
                "id": "chatcmpl-stream",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "delta": {"content": "Hi"}}],
            },
            {
                "id": "chatcmpl-stream",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        ]
    )

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json=model_list(
                    {
                        "id": "deepseek/deepseek-v4.1-flash",
                        "supported_endpoints": ["/chat/completions"],
                    }
                ),
            )
        return httpx.Response(
            200,
            content=SyncStream([body]),
            headers={"content-type": "text/event-stream", "x-request-id": "req-stream"},
        )

    records: list[CallUsage] = []
    llm = BrainStormLLM(
        model="deepseek/deepseek-v4.1-flash",
        provider=commandcode_provider(handler),
        pricing="off",
    )
    events = list(llm.stream([{"role": "user", "content": "hi"}], on_usage=records.append))
    assert isinstance(events[-1], MessageStopEvent)
    assert any(isinstance(event, ContentBlockDeltaEvent) for event in events)
    assert len(records) == 1
    assert records[0].streaming is True
    assert records[0].usage.input_tokens == 10
    assert records[0].usage.output_tokens == 2
    assert records[0].request_id == "req-stream"


def test_stream_collect_returns_final_message() -> None:
    body = sse_bytes(
        [
            {
                "id": "chatcmpl-stream",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "delta": {"content": "Hello"}}],
            },
            {
                "id": "chatcmpl-stream",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        ]
    )

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json=model_list(
                    {
                        "id": "deepseek/deepseek-v4.1-flash",
                        "supported_endpoints": ["/chat/completions"],
                    }
                ),
            )
        return httpx.Response(
            200,
            content=SyncStream([body]),
            headers={"content-type": "text/event-stream"},
        )

    llm = BrainStormLLM(
        model="deepseek/deepseek-v4.1-flash",
        provider=commandcode_provider(handler),
        pricing="off",
    )
    message = llm.stream_collect([{"role": "user", "content": "hi"}])
    assert message.text == "Hello"
    assert message.usage.output_tokens == 2


def test_stream_text() -> None:
    body = sse_bytes(
        [
            {
                "id": "c",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "delta": {"content": "a"}}],
            },
            {
                "id": "c",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "delta": {"content": "b"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        ]
    )

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json=model_list(
                    {
                        "id": "deepseek/deepseek-v4.1-flash",
                        "supported_endpoints": ["/chat/completions"],
                    }
                ),
            )
        return httpx.Response(200, content=SyncStream([body]))

    llm = BrainStormLLM(
        model="deepseek/deepseek-v4.1-flash",
        provider=commandcode_provider(handler),
        pricing="off",
    )
    assert "".join(llm.stream_text([{"role": "user", "content": "hi"}])) == "ab"


async def test_async_generate_and_stream() -> None:
    async def async_handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json=model_list(
                    {
                        "id": "deepseek/deepseek-v4.1-flash",
                        "supported_endpoints": ["/chat/completions"],
                    }
                ),
            )
        return httpx.Response(
            200,
            json=openai_completion(content="async"),
            headers={"x-request-id": "req-async"},
        )

    body = sse_bytes(
        [
            {
                "id": "c",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "delta": {"content": "stream"}}],
            },
            {
                "id": "c",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        ]
    )

    async def async_stream_handler(request):
        if request.url.path.endswith("/models"):
            return await async_handler(request)
        payload = json.loads(request.content) if request.content else {}
        if payload.get("stream"):
            return httpx.Response(
                200,
                content=AsyncStream([body]),
                headers={"content-type": "text/event-stream"},
            )
        return await async_handler(request)

    provider = commandcode_provider(chat_handler, async_handler=async_stream_handler)
    llm = BrainStormLLM(
        model="deepseek/deepseek-v4.1-flash",
        provider=provider,
        pricing="off",
    )
    async with llm:
        message = await llm.agenerate([{"role": "user", "content": "hi"}])
        assert message.content[0].text == "async"
        assert message.request_id == "req-async"

        collected = await llm.astream_collect([{"role": "user", "content": "hi"}])
        assert collected.text == "stream"


def test_usage_callback_and_cost_tracker(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BRAINSTORM_CACHE_DIR", str(tmp_path))
    PricingCache().put(
        "commandcode",
        {
            "deepseek/deepseek-v4.1-flash": TokenPrice(
                input=1.0, output=2.0, cache_read=0.5
            )
        },
        source="test",
    )

    tracker = CostTracker()
    llm = BrainStormLLM(
        model="deepseek/deepseek-v4.1-flash",
        provider=commandcode_provider(chat_handler),
        pricing="cached",
        on_usage=tracker,
    )
    llm.generate([{"role": "user", "content": "hi"}])
    assert len(tracker.records) == 1
    assert tracker.records[0].usage.input_tokens == 10
    assert tracker.total_usd == Decimal("0.000016")
    assert tracker.by_model()["deepseek/deepseek-v4.1-flash"]["calls"] == 1


def test_usage_callback_errors_are_swallowed() -> None:
    def bad_callback(record):
        raise RuntimeError("nope")

    llm = BrainStormLLM(
        model="deepseek/deepseek-v4.1-flash",
        provider=commandcode_provider(chat_handler),
        pricing="off",
        on_usage=bad_callback,
    )
    assert llm.generate([{"role": "user", "content": "hi"}]).text == "hello"


def test_bad_request_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json=model_list(
                    {
                        "id": "deepseek/deepseek-v4.1-flash",
                        "supported_endpoints": ["/chat/completions"],
                    }
                ),
            )
        return httpx.Response(
            400,
            json={"error": {"message": "bad", "type": "invalid_request_error"}},
        )

    llm = BrainStormLLM(
        model="deepseek/deepseek-v4.1-flash",
        provider=commandcode_provider(handler),
        pricing="off",
    )
    with pytest.raises(BadRequestError):
        llm.generate([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2  # one models call + one chat call


def test_zdr_error_mapping() -> None:
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json=model_list(
                    {
                        "id": "deepseek/deepseek-v4.1-flash",
                        "supported_endpoints": ["/chat/completions"],
                    }
                ),
            )
        return httpx.Response(
            422,
            json={
                "error": {
                    "message": "no zdr providers",
                    "type": "invalid_request_error",
                    "code": "cmd_zdr_no_providers",
                }
            },
        )

    llm = BrainStormLLM(
        model="deepseek/deepseek-v4.1-flash",
        provider=commandcode_provider(handler),
        pricing="off",
    )
    with pytest.raises(ZDRUnavailableError):
        llm.generate([{"role": "user", "content": "hi"}])
