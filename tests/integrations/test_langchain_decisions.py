from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
from langchain_core.exceptions import ModelAuthenticationError
from langchain_core.messages import HumanMessage

from brainstorm_llm import BrainStormS1
from brainstorm_llm.integrations.langchain import (
    BrainStormDecisions,
    Choice,
    Level,
    Predicate,
    Score,
)
from brainstorm_llm.providers.commandcode import CommandCodeProvider

ANSWER = {
    "model": "typesafe/jev-1.14.0",
    "answers": {
        "urgent": {"type": "noul", "noul": 0.94},
        "team": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.9,
            "probabilities": {"billing": 0.9, "shipping": 0.1},
        },
        "tone": {
            "type": "score",
            "score": 1.3,
            "confidence": 0.7,
            "legend": {"0": "Calm", "1": "Mad", "2": "Furious"},
            "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4},
        },
    },
    "usage": {"input_tokens": 84, "output_tokens": 3},
}


def make_decisions(
    handler,
    *,
    async_handler=None,
    **client_kwargs,
) -> BrainStormDecisions:
    provider = CommandCodeProvider(
        api_key="test",
        transport=httpx.MockTransport(handler),
        async_transport=httpx.MockTransport(async_handler or handler),
        **client_kwargs,
    )
    client = BrainStormS1(provider=provider, pricing="off")
    return BrainStormDecisions(client=client)


def test_decisions_invoke_and_request_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ANSWER, headers={"x-typesafe-request-id": "ts-1"})

    decisions = make_decisions(handler)
    result = decisions.invoke(
        {
            "input": "Production is down.",
            "questions": {
                "urgent": Predicate(instructions="Is it urgent?"),
                "team": Choice(
                    instructions="Which team?",
                    choices={"billing": "payments", "shipping": "delivery"},
                ),
                "tone": Score(
                    instructions="Tone?",
                    levels=[Level(label="Calm"), "Mad", Level(label="Furious", description="very")],
                ),
            },
        }
    )

    body = seen["body"]
    assert body["state"] == "Production is down."
    assert body["questions"]["urgent"] == {
        "type": "noul",
        "instructions": "Is it urgent?",
    }
    assert body["questions"]["team"]["criteria"] == {
        "billing": "payments",
        "shipping": "delivery",
    }
    assert body["questions"]["tone"]["criteria"] == ["Calm", "Mad", "Furious: very"]

    assert result.predicates["urgent"].probability == 0.94
    assert result.choices["team"].choice == "billing"
    assert result.scores["tone"].score == 1.3
    assert result.usage.input_tokens == 84
    assert result.request_id == "ts-1"


async def test_decisions_async_invoke() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ANSWER, headers={"x-typesafe-request-id": "ts-async"})

    decisions = make_decisions(lambda request: httpx.Response(500), async_handler=handler)
    result = await decisions.ainvoke(
        {
            "input": "state",
            "questions": {"urgent": Predicate(instructions="Urgent?")},
        }
    )
    assert result.predicates["urgent"].probability == 0.94
    assert result.request_id == "ts-async"


def test_decisions_extracts_base64_images_into_assets() -> None:
    seen: dict[str, Any] = {}
    image_bytes = b"\x89PNG\r\n\x1a\nfake"

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ANSWER)

    decisions = make_decisions(handler)
    decisions.invoke(
        {
            "input": [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Inspect this image."},
                        {
                            "type": "image",
                            "base64": base64.b64encode(image_bytes).decode(),
                            "mime_type": "image/png",
                        },
                    ]
                )
            ],
            "questions": {"urgent": Predicate(instructions="Is it urgent?")},
        }
    )

    body = seen["body"]
    assert "assets" in body
    assert len(body["assets"]) == 1
    placeholder, asset = next(iter(body["assets"].items()))
    assert placeholder.startswith("<brainstorm-image-")
    assert asset["type"] == "image"
    assert asset["format"] == "png"
    assert asset["data_base64"] == base64.b64encode(image_bytes).decode()
    assert placeholder in json.dumps(body["state"])


def test_decisions_tool_invocation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ANSWER)

    decisions = make_decisions(handler)
    tool = decisions.make_tool(
        {"urgent": Predicate(instructions="Is it urgent?")},
        name="triage",
    )
    assert tool.name == "triage"
    payload = json.loads(tool.invoke("Production is down."))
    assert payload["answers"]["urgent"]["probability"] == 0.94


async def test_decisions_tool_async_invocation() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ANSWER)

    decisions = make_decisions(lambda request: httpx.Response(500), async_handler=handler)
    tool = decisions.make_tool({"urgent": Predicate(instructions="Is it urgent?")})
    payload = json.loads(await tool.ainvoke("Production is down."))
    assert payload["answers"]["urgent"]["type"] == "predicate"


def test_decisions_error_mapping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "bad key", "type": "authentication_error"}},
        )

    decisions = make_decisions(handler)
    with pytest.raises(ModelAuthenticationError):
        decisions.invoke(
            {
                "input": "state",
                "questions": {"urgent": Predicate(instructions="Urgent?")},
            }
        )
