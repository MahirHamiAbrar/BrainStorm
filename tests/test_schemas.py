from __future__ import annotations

import pytest

from brainstorm_llm.schemas import (
    Message,
    MessageAccumulator,
    Noul,
    Score,
    SystemOneRequest,
    SystemOneResponse,
    TextBlock,
    TokenPrice,
    ToolResultBlock,
    Usage,
)
from brainstorm_llm.schemas.streaming import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    TextDelta,
)


def test_message_accepts_string_content() -> None:
    message = Message(id="m", model="x", content="hello")
    assert isinstance(message.content[0], TextBlock)
    assert message.text == "hello"


def test_usage_total_includes_cache_tokens() -> None:
    usage = Usage(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=3,
        cache_creation_input_tokens=2,
    )
    assert usage.total_tokens == 20


def test_token_price_aliases() -> None:
    price = TokenPrice(**{"in": 0.15, "out": 0.6, "cache_read": 0.003})
    assert price.input == 0.15
    assert price.model_dump(by_alias=True, exclude_none=True) == {
        "in": 0.15,
        "out": 0.6,
        "cache_read": 0.003,
    }


def test_tool_result_accepts_text_and_blocks() -> None:
    block = ToolResultBlock(tool_use_id="t1", content="done")
    assert block.content == "done"
    block = ToolResultBlock(
        tool_use_id="t1",
        content=[TextBlock(text="one"), TextBlock(text="two")],
    )
    assert len(block.content) == 2


def test_systemone_typed_answer_properties() -> None:
    response = SystemOneResponse.model_validate(
        {
            "model": "typesafe/jev",
            "answers": {
                "urgent": {"type": "noul", "noul": 0.9},
                "tone": {
                    "type": "score",
                    "score": 1.5,
                    "confidence": 0.5,
                    "legend": {"0": "calm", "1": "mad"},
                    "probabilities": {"0": 0.2, "1": 0.8},
                },
            },
            "usage": {"input_tokens": 4, "output_tokens": 1},
        }
    )
    assert response.nouls["urgent"].noul == 0.9
    assert response.scores["tone"].legend == {0: "calm", 1: "mad"}
    assert response.choices == {}


def test_systemone_request_validation() -> None:
    request = SystemOneRequest(
        model="typesafe/jev-1.14.0",
        state={"document": "text"},
        questions={
            "urgent": Noul(instructions="urgent?"),
            "tone": Score(instructions="tone?", criteria=["calm", "mad"]),
        },
    )
    payload = request.to_wire()
    assert payload["questions"]["urgent"] == {"type": "noul", "instructions": "urgent?"}
    assert payload["questions"]["tone"]["criteria"] == ["calm", "mad"]


def test_systemone_choice_keeps_none_criteria_keys() -> None:
    from brainstorm_llm.schemas import Choice

    request = SystemOneRequest(
        model="typesafe/jev-1.14.0",
        state="state",
        questions={
            "team": Choice(
                instructions="Which team?",
                criteria={"billing": None, "shipping": "delivery"},
            )
        },
    )
    criteria = request.to_wire()["questions"]["team"]["criteria"]
    assert criteria == {"billing": None, "shipping": "delivery"}


def test_message_accumulator_builds_final_message() -> None:
    accumulator = MessageAccumulator()
    accumulator.feed(
        MessageStartEvent(
            message=Message(
                id="msg_1",
                model="m",
                content=[],
                usage=Usage(input_tokens=10, output_tokens=0),
            )
        )
    )
    accumulator.feed(
        ContentBlockStartEvent(index=0, content_block=TextBlock(text=""))
    )
    accumulator.feed(
        ContentBlockDeltaEvent(index=0, delta=TextDelta(text="Hello"))
    )
    accumulator.feed(ContentBlockStopEvent(index=0))
    accumulator.feed(
        MessageDeltaEvent(
            delta=MessageDelta(stop_reason="end_turn"),
            usage=Usage(output_tokens=2),
        )
    )
    accumulator.feed(MessageStopEvent())
    message = accumulator.finalize()
    assert message.text == "Hello"
    assert message.usage.input_tokens == 10
    assert message.usage.output_tokens == 2
    assert message.stop_reason == "end_turn"


def test_invalid_score_criteria_rejected() -> None:
    with pytest.raises(ValueError):
        Score(instructions="x", criteria=["only one"])
