from __future__ import annotations

from brainstorm_llm.convert.response import (
    map_finish_reason,
    openai_to_anthropic_message,
    usage_from_openai,
)
from brainstorm_llm.schemas import TextBlock, ToolUseBlock, Usage
from tests.helpers import openai_completion


def test_text_response_and_usage_normalization() -> None:
    payload = openai_completion(
        content="hi", prompt_tokens=100, completion_tokens=20, cached_tokens=30
    )
    message = openai_to_anthropic_message(payload)
    assert isinstance(message.content[0], TextBlock)
    assert message.content[0].text == "hi"
    assert message.usage.input_tokens == 70
    assert message.usage.output_tokens == 20
    assert message.usage.cache_read_input_tokens == 30
    assert message.usage.total_tokens == 120
    assert message.stop_reason == "end_turn"
    assert message.id == "chatcmpl-test"


def test_tool_calls_parse_arguments() -> None:
    payload = openai_completion(
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q": "x"}'},
            }
        ],
        finish_reason="tool_calls",
    )
    message = openai_to_anthropic_message(payload)
    block = message.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.input == {"q": "x"}
    assert message.stop_reason == "tool_use"


def test_invalid_tool_arguments_preserved() -> None:
    payload = openai_completion(
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{not json"},
            }
        ],
        finish_reason="tool_calls",
    )
    message = openai_to_anthropic_message(payload)
    assert message.content[0].input == {"_raw_arguments": "{not json"}


def test_refusal_maps_to_refusal_stop_reason() -> None:
    payload = openai_completion(content=None)
    payload["choices"][0]["message"]["refusal"] = "I cannot help with that"
    message = openai_to_anthropic_message(payload)
    assert message.content[0].text == "I cannot help with that"
    assert message.stop_reason == "refusal"


def test_finish_reason_map() -> None:
    assert map_finish_reason("stop") == "end_turn"
    assert map_finish_reason("length") == "max_tokens"
    assert map_finish_reason("tool_calls") == "tool_use"
    assert map_finish_reason("content_filter") == "refusal"
    assert map_finish_reason(None) is None


def test_usage_missing_returns_zeroes() -> None:
    assert usage_from_openai(None) == Usage()


def test_reasoning_content_becomes_thinking_block() -> None:
    payload = openai_completion(content="answer")
    payload["choices"][0]["message"]["reasoning_content"] = "thinking"
    message = openai_to_anthropic_message(payload)
    assert message.content[0].type == "thinking"
    assert message.content[1].text == "answer"
