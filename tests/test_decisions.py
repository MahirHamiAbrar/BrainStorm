from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from brainstorm_llm import BrainStormS1, CallUsage, CostTracker, Noul, TokenPrice
from brainstorm_llm.errors import ZDRUnavailableError
from brainstorm_llm.pricing.cache import PricingCache
from brainstorm_llm.providers.commandcode import CommandCodeProvider
from brainstorm_llm.schemas.systemone import NoulAnswer, SystemOneResponse

ANSWER = {
    "model": "typesafe/jev-1.14.0",
    "answers": {
        "urgent": {"type": "noul", "noul": 0.93},
        "team": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.9,
            "probabilities": {"billing": 0.9, "shipping": 0.1},
        },
        "frustration": {
            "type": "score",
            "score": 1.4,
            "confidence": 0.6,
            "legend": {"0": "Calm", "1": "Mad", "2": "Furious"},
            "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
        },
    },
    "usage": {"input_tokens": 84, "output_tokens": 3},
}


def provider(handler, async_handler=None, **kwargs):
    return CommandCodeProvider(
        api_key="test",
        transport=httpx.MockTransport(handler),
        async_transport=httpx.MockTransport(async_handler or handler),
        **kwargs,
    )


def test_system_one_request_and_response() -> None:
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ANSWER, headers={"x-typesafe-request-id": "ts-1"})

    s1 = BrainStormS1(provider=provider(handler), pricing="off")
    response = s1.system_one(
        "payments failed",
        {
            "urgent": Noul(instructions="urgent?"),
        },
    )
    assert seen["path"].endswith("/systemone")
    assert seen["body"]["model"] == "typesafe/jev-1.14.0"
    assert seen["body"]["questions"]["urgent"] == {"type": "noul", "instructions": "urgent?"}
    assert response.nouls["urgent"].noul == 0.93
    assert response.request_id == "ts-1"


async def test_async_system_one() -> None:
    async def handler(request):
        return httpx.Response(200, json=ANSWER, headers={"x-typesafe-request-id": "ts-async"})

    s1 = BrainStormS1(provider=provider(lambda request: httpx.Response(200), async_handler=handler), pricing="off")
    async with s1:
        response = await s1.asystem_one("state", {"urgent": Noul(instructions="urgent?")})
    assert response.request_id == "ts-async"


def test_typed_response_model() -> None:
    class BillingResponse(SystemOneResponse):
        urgent: NoulAnswer

    s1 = BrainStormS1(
        provider=provider(lambda request: httpx.Response(200, json=ANSWER)),
        pricing="off",
    )
    response = s1.system_one(
        "state",
        {"urgent": Noul(instructions="urgent?")},
        response_model=BillingResponse,
    )
    assert isinstance(response, BillingResponse)
    assert response.urgent.noul == 0.93
    assert response.request_id is None


def test_usage_callback_and_cost(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BRAINSTORM_CACHE_DIR", str(tmp_path))
    PricingCache().put(
        "commandcode",
        {"typesafe/jev-1.14.0": TokenPrice(input=0.042, output=0.0)},
        source="test",
    )
    tracker = CostTracker()
    s1 = BrainStormS1(
        provider=provider(lambda request: httpx.Response(200, json=ANSWER)),
        pricing="cached",
        on_usage=tracker,
    )
    s1.system_one("state", {"urgent": Noul(instructions="urgent?")})
    assert len(tracker.records) == 1
    record = tracker.records[0]
    assert record.operation == "system_one"
    assert record.usage.input_tokens == 84
    assert record.cost_usd == Decimal("0.000003528")
    assert record.provider == "commandcode"


def test_zdr_refusal_is_mapped() -> None:
    def handler(request):
        return httpx.Response(
            422,
            json={
                "error": {
                    "message": "no zdr upstream for systemone",
                    "type": "invalid_request_error",
                    "code": "cmd_zdr_no_providers",
                }
            },
        )

    s1 = BrainStormS1(provider=provider(handler), pricing="off", zdr=True)
    with pytest.raises(ZDRUnavailableError):
        s1.system_one("state", {"urgent": Noul(instructions="urgent?")})


def test_system_one_request_bounds() -> None:
    s1 = BrainStormS1(provider=provider(lambda request: httpx.Response(200, json=ANSWER)), pricing="off")
    with pytest.raises(ValueError):
        s1.system_one("state", {})
