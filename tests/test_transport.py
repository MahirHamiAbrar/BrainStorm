from __future__ import annotations

import httpx
import pytest

from brainstorm_llm.errors import APITimeoutError, BadRequestError, RateLimitError
from brainstorm_llm.transport import (
    HttpTransport,
    RetryPolicy,
    TransportConfig,
    aiter_sse_events,
    iter_sse_events,
)


def test_sse_parser_handles_events_comments_and_crlf() -> None:
    lines = [
        "event: message_start\r",
        "data: {\"a\": 1}\r",
        "\r",
        ": comment",
        "data: one",
        "data: two",
        "",
        "data: [DONE]",
        "",
    ]
    events = list(iter_sse_events(lines))
    assert events[0].event == "message_start"
    assert events[0].data == '{"a": 1}'
    assert events[1].event is None
    assert events[1].data == "one\ntwo"
    assert events[2].data == "[DONE]"


async def test_async_sse_parser() -> None:
    async def lines():
        yield "event: x"
        yield "data: 1"
        yield ""
        yield "data: 2"
        yield ""

    events = [event async for event in aiter_sse_events(lines())]
    assert events[0].event == "x"
    assert events[1].data == "2"


def make_transport(handler, *, retries=1) -> HttpTransport:
    return HttpTransport(
        TransportConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            timeout=1,
            retry=RetryPolicy(max_retries=retries, backoff_base=0, jitter=False),
        ),
        transport=httpx.MockTransport(handler),
    )


def test_retries_retryable_status() -> None:
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, json={"error": {"message": "boom", "type": "server_error"}})
        return httpx.Response(200, json={"ok": True})

    transport = make_transport(handler)
    response = transport.request("GET", "/models")
    assert response.json() == {"ok": True}
    assert calls["n"] == 2
    transport.close()


def test_rate_limit_retry_after() -> None:
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                json={"error": {"message": "slow", "type": "rate_limit_error"}},
                headers={"retry-after": "0"},
            )
        return httpx.Response(200, json={"ok": True})

    transport = make_transport(handler)
    assert transport.request("GET", "/models").json() == {"ok": True}
    assert calls["n"] == 2
    transport.close()


def test_no_retry_on_bad_request() -> None:
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(
            400,
            json={"error": {"message": "bad", "type": "invalid_request_error"}},
        )

    transport = make_transport(handler)
    with pytest.raises(BadRequestError):
        transport.request("GET", "/models")
    assert calls["n"] == 1
    transport.close()


def test_rate_limit_error_carries_retry_after() -> None:
    def handler(request):
        return httpx.Response(
            429,
            json={"error": {"message": "slow", "type": "rate_limit_error"}},
            headers={"retry-after": "3"},
        )

    transport = make_transport(handler, retries=0)
    with pytest.raises(RateLimitError) as exc:
        transport.request("GET", "/models")
    assert exc.value.retry_after == 3
    transport.close()


def test_timeout_is_wrapped() -> None:
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    transport = make_transport(handler)
    with pytest.raises(APITimeoutError):
        transport.request("GET", "/models")
    transport.close()
