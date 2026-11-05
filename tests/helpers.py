from __future__ import annotations

import json
from typing import Any, Iterable

import httpx


def sse_bytes(items: Iterable[dict[str, Any] | str]) -> bytes:
    parts: list[bytes] = []
    for item in items:
        if isinstance(item, str):
            parts.append(f"data: {item}\n\n".encode())
        else:
            parts.append(f"data: {json.dumps(item)}\n\n".encode())
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


class SyncStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks


class AsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def model_list(*models: dict[str, Any]) -> dict[str, Any]:
    return {"object": "list", "data": list(models)}


def openai_completion(
    *,
    content: str | None = "hello",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 3,
    cached_tokens: int = 0,
    model: str = "test-model",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 123,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


def anthropic_message(
    *,
    text: str = "hello",
    input_tokens: int = 10,
    output_tokens: int = 3,
    model: str = "claude-test",
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def anthropic_stream_bytes(*, text: str = "Hello") -> bytes:
    events = [
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-test",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 2}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    parts = []
    for name, payload in events:
        parts.append(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())
    return b"".join(parts)
