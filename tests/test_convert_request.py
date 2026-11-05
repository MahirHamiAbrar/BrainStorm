from __future__ import annotations

from brainstorm_llm.convert.request import anthropic_to_openai_payload
from brainstorm_llm.schemas import (
    ImageBlock,
    MessageCreateParams,
    TextBlock,
    ThinkingConfig,
    Tool,
    ToolChoiceAny,
    ToolChoiceTool,
    ToolResultBlock,
    ToolUseBlock,
)


def make_params(**overrides):
    base = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 128,
    }
    base.update(overrides)
    return MessageCreateParams.model_validate(base)


def test_system_string_and_blocks() -> None:
    payload = anthropic_to_openai_payload(make_params(system="be nice"))
    assert payload["messages"][0] == {"role": "system", "content": "be nice"}

    payload = anthropic_to_openai_payload(
        make_params(system=[TextBlock(text="one"), TextBlock(text="two")])
    )
    assert payload["messages"][0] == {"role": "system", "content": "one\ntwo"}


def test_user_content_with_image() -> None:
    params = make_params(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc",
                        },
                    },
                ],
            }
        ]
    )
    payload = anthropic_to_openai_payload(params)
    parts = payload["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this?"}
    assert parts[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc"},
    }


def test_assistant_tool_use_and_user_tool_result_ordering() -> None:
    params = make_params(
        messages=[
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [
                    TextBlock(text="checking"),
                    ToolUseBlock(id="toolu_1", name="weather", input={"city": "Paris"}),
                ],
            },
            {
                "role": "user",
                "content": [
                    ToolResultBlock(tool_use_id="toolu_1", content="sunny"),
                    TextBlock(text="thanks"),
                ],
            },
        ]
    )
    payload = anthropic_to_openai_payload(params)
    assert payload["messages"][1]["content"] == "checking"
    assert payload["messages"][1]["tool_calls"][0]["id"] == "toolu_1"
    assert payload["messages"][2] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "sunny",
    }
    assert payload["messages"][3] == {"role": "user", "content": "thanks"}


def test_tools_and_tool_choice() -> None:
    tool = Tool(
        name="lookup",
        description="look things up",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    payload = anthropic_to_openai_payload(
        make_params(tools=[tool], tool_choice=ToolChoiceTool(name="lookup", disable_parallel_tool_use=True))
    )
    assert payload["tools"][0]["function"]["name"] == "lookup"
    assert payload["tools"][0]["function"]["parameters"]["properties"]["q"]["type"] == "string"
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert payload["parallel_tool_calls"] is False

    payload = anthropic_to_openai_payload(make_params(tool_choice=ToolChoiceAny()))
    assert payload["tool_choice"] == "required"


def test_thinking_maps_to_reasoning_effort() -> None:
    payload = anthropic_to_openai_payload(
        make_params(thinking=ThinkingConfig(type="enabled", budget_tokens=8000))
    )
    assert payload["reasoning_effort"] == "medium"


def test_extra_body_wins() -> None:
    payload = anthropic_to_openai_payload(
        make_params(temperature=0.5),
        extra_body={"temperature": 0.1, "provider": {"order": ["x"]}},
    )
    assert payload["temperature"] == 0.1
    assert payload["provider"] == {"order": ["x"]}


def test_streaming_includes_usage() -> None:
    payload = anthropic_to_openai_payload(make_params(), stream=True)
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_output_config_json_schema() -> None:
    payload = anthropic_to_openai_payload(
        make_params(
            output_config={
                "format": {"type": "json_schema", "schema": {"type": "object"}}
            }
        )
    )
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == {"type": "object"}
