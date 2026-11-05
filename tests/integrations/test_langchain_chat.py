from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
from langchain_core.exceptions import ModelAuthenticationError, ModelRateLimitError
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from brainstorm_llm.integrations.langchain import ChatBrainStorm
from tests.helpers import AsyncStream, SyncStream, openai_completion, sse_bytes
from tests.integrations.helpers import make_chat_model


class Lookup(BaseModel):
    """Look up a value."""

    query: str


def test_invoke_returns_ai_message_with_usage_and_metadata() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=openai_completion(
                content="hello",
                prompt_tokens=10,
                completion_tokens=3,
                cached_tokens=4,
            ),
            headers={"x-request-id": "req-1"},
        )

    model = make_chat_model(handler)
    response = model.invoke("hi")

    assert isinstance(response, AIMessage)
    assert response.text == "hello"
    assert response.usage_metadata is not None
    assert response.usage_metadata.get("input_tokens") == 6
    assert response.usage_metadata.get("output_tokens") == 3
    details = response.usage_metadata.get("input_token_details") or {}
    assert details.get("cache_read") == 4
    assert response.response_metadata["provider"] == "commandcode"
    assert response.response_metadata["request_id"] == "req-1"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_system_and_multimodal_message_conversion() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=openai_completion(content="seen"))

    model = make_chat_model(handler)
    model.invoke(
        [
            SystemMessage("You are helpful."),
            HumanMessage(
                content=[
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "base64": "aGVsbG8=",
                        "mime_type": "image/png",
                    },
                ]
            ),
        ]
    )

    messages = seen["body"]["messages"]
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][1]["image_url"]["url"] == (
        "data:image/png;base64,aGVsbG8="
    )


def test_bind_tools_and_parallel_flag() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=openai_completion(
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Lookup",
                            "arguments": json.dumps({"query": "x"}),
                        },
                    }
                ],
                finish_reason="tool_calls",
            ),
        )

    model = make_chat_model(handler)
    response = cast(
        AIMessage,
        model.bind_tools(
            [Lookup],
            tool_choice="Lookup",
            parallel_tool_calls=False,
        ).invoke("find x"),
    )

    assert response.tool_calls == [
        {"name": "Lookup", "args": {"query": "x"}, "id": "call_1", "type": "tool_call"}
    ]
    body = seen["body"]
    assert body["tools"][0]["function"]["name"] == "Lookup"
    assert body["tool_choice"] == {"type": "function", "function": {"name": "Lookup"}}
    assert body["parallel_tool_calls"] is False


def test_reasoning_effort_maps_to_provider_request() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=openai_completion(content="ok"))

    model = make_chat_model(handler)
    model.invoke("think", reasoning_effort="high")
    assert seen["body"]["reasoning_effort"] == "high"


def test_with_structured_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=openai_completion(
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Lookup",
                            "arguments": json.dumps({"query": "answer"}),
                        },
                    }
                ],
                finish_reason="tool_calls",
            ),
        )

    model = make_chat_model(handler)
    parsed = model.with_structured_output(Lookup).invoke("find it")
    assert parsed == Lookup(query="answer")


def test_streaming_text_and_final_usage() -> None:
    body = sse_bytes(
        [
            {
                "id": "c1",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "delta": {"content": "Hel"}}],
            },
            {
                "id": "c1",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [
                    {"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=SyncStream([body]),
            headers={"content-type": "text/event-stream"},
        )

    model = make_chat_model(handler)
    chunks = list(model.stream("hi"))
    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert [chunk.text for chunk in chunks] == ["Hel", "lo", ""]
    final = chunks[-1]
    assert final.chunk_position == "last"
    assert final.usage_metadata is not None
    assert final.usage_metadata.get("input_tokens") == 5
    assert final.usage_metadata.get("output_tokens") == 2

    accumulated = chunks[0]
    for chunk in chunks[1:]:
        accumulated = accumulated + chunk
    assert accumulated.text == "Hello"


def test_streaming_tool_call_chunks() -> None:
    body = sse_bytes(
        [
            {
                "id": "c1",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "Lookup", "arguments": '{"query":'},
                                }
                            ]
                        },
                    }
                ],
            },
            {
                "id": "c1",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"x"}'}}
                            ]
                        },
                    }
                ],
            },
            {
                "id": "c1",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=SyncStream([body]),
            headers={"content-type": "text/event-stream"},
        )

    model = make_chat_model(handler)
    chunks = cast(
        list[AIMessageChunk],
        list(model.bind_tools([Lookup], tool_choice="Lookup").stream("hi")),
    )
    accumulated = chunks[0]
    for chunk in chunks[1:]:
        accumulated = accumulated + chunk
    assert accumulated.tool_calls == [
        {"name": "Lookup", "args": {"query": "x"}, "id": "call_1", "type": "tool_call"}
    ]


async def test_async_invoke_and_stream() -> None:
    stream_body = sse_bytes(
        [
            {
                "id": "c1",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "delta": {"content": "async"}}],
            },
            {
                "id": "c1",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.method == "POST" else {}
        if body.get("stream"):
            return httpx.Response(
                200,
                content=AsyncStream([stream_body]),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=openai_completion(content="async-invoke"))

    model = make_chat_model(lambda request: httpx.Response(500), async_handler=handler)
    response = await model.ainvoke("hi")
    assert response.text == "async-invoke"

    chunks = [chunk async for chunk in model.astream("hi")]
    accumulated = chunks[0]
    for chunk in chunks[1:]:
        accumulated = accumulated + chunk
    assert accumulated.text == "async"
    assert accumulated.usage_metadata is not None
    assert accumulated.usage_metadata.get("output_tokens") == 1


def test_error_mapping_to_langchain_exceptions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "bad key", "type": "authentication_error"}},
        )

    model = make_chat_model(handler)
    with pytest.raises(ModelAuthenticationError):
        model.invoke("hi")

    def limit_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "slow", "type": "rate_limit_error"}},
            headers={"retry-after": "0"},
        )

    with pytest.raises(ModelRateLimitError):
        make_chat_model(limit_handler).invoke("hi")


def test_langgraph_agent_tool_loop() -> None:
    calls = {"n": 0}

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        if any(message.get("role") == "tool" for message in body.get("messages", [])):
            completion = openai_completion(content="It is sunny in Paris.")
        else:
            completion = openai_completion(
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    }
                ],
                finish_reason="tool_calls",
            )
        completion["id"] = f"chatcmpl-{calls['n']}"
        return httpx.Response(200, json=completion)

    model = make_chat_model(handler)
    model_with_tools = model.bind_tools([get_weather])

    def call_model(state: MessagesState) -> dict:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode([get_weather]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    app = graph.compile()

    result = app.invoke({"messages": [HumanMessage("What is the weather in Paris?")]})
    assert calls["n"] == 2
    assert result["messages"][-1].text == "It is sunny in Paris."
    assert any(message.type == "tool" for message in result["messages"])