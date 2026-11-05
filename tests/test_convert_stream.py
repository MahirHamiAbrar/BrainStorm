from __future__ import annotations

from brainstorm_llm.convert.stream import OpenAIStreamTranslator
from brainstorm_llm.schemas import MessageAccumulator, TextBlock, ToolUseBlock
from brainstorm_llm.schemas.streaming import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJSONDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    TextDelta,
)


def chunk(delta=None, finish_reason=None, usage=None, chunk_id="chatcmpl-1"):
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def collect(chunks):
    translator = OpenAIStreamTranslator(model="test-model")
    events = []
    for item in chunks:
        events.extend(translator.process_chunk(item))
    events.extend(translator.finalize())
    return events


def test_text_stream_and_usage() -> None:
    events = collect(
        [
            chunk({"role": "assistant", "content": "Hel"}),
            chunk({"content": "lo"}),
            chunk({}, finish_reason="stop"),
            chunk(
                {},
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            ),
        ]
    )
    assert isinstance(events[0], MessageStartEvent)
    assert isinstance(events[1], ContentBlockStartEvent)
    assert isinstance(events[1].content_block, TextBlock)
    deltas = [e for e in events if isinstance(e, ContentBlockDeltaEvent)]
    assert [d.delta.text for d in deltas if isinstance(d.delta, TextDelta)] == ["Hel", "lo"]
    assert isinstance(events[-2], MessageDeltaEvent)
    assert events[-2].usage.input_tokens == 6
    assert events[-2].usage.cache_read_input_tokens == 4
    assert isinstance(events[-1], MessageStopEvent)

    accumulator = MessageAccumulator()
    for event in events:
        accumulator.feed(event)
    message = accumulator.finalize()
    assert message.text == "Hello"
    assert message.stop_reason == "end_turn"
    assert message.usage.output_tokens == 2


def test_tool_call_arguments_split_across_chunks() -> None:
    events = collect(
        [
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q":'},
                        }
                    ]
                }
            ),
            chunk(
                {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '"x"}'}}
                    ]
                }
            ),
            chunk({}, finish_reason="tool_calls"),
        ]
    )
    starts = [e for e in events if isinstance(e, ContentBlockStartEvent)]
    assert isinstance(starts[0].content_block, ToolUseBlock)
    deltas = [e for e in events if isinstance(e, ContentBlockDeltaEvent)]
    assert all(isinstance(d.delta, InputJSONDelta) for d in deltas)
    stops = [e for e in events if isinstance(e, ContentBlockStopEvent)]
    assert len(stops) == 1

    accumulator = MessageAccumulator()
    for event in events:
        accumulator.feed(event)
    message = accumulator.finalize()
    assert message.content[0].input == {"q": "x"}
    assert message.stop_reason == "tool_use"


def test_text_then_tool_closes_text_block() -> None:
    events = collect(
        [
            chunk({"content": "thinking"}),
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ]
                }
            ),
            chunk({}, finish_reason="tool_calls"),
        ]
    )
    types = [(e.__class__.__name__, getattr(e, "index", None)) for e in events]
    assert types[1] == ("ContentBlockStartEvent", 0)
    assert types[3] == ("ContentBlockStopEvent", 0)
    assert types[4] == ("ContentBlockStartEvent", 1)


def test_finalize_is_idempotent() -> None:
    translator = OpenAIStreamTranslator(model="m")
    translator.process_chunk(chunk({"content": "hi"}))
    first = translator.finalize()
    second = translator.finalize()
    assert first and second == []


def test_empty_stream_still_finishes() -> None:
    events = collect([])
    assert isinstance(events[0], MessageStartEvent)
    assert isinstance(events[-2], MessageDeltaEvent)
    assert events[-2].delta.stop_reason == "end_turn"
    assert isinstance(events[-1], MessageStopEvent)
