from __future__ import annotations

import uuid
from typing import Any

import orjson

from ..log import library_logger
from ..schemas.content import TextBlock, ThinkingBlock, ToolUseBlock
from ..schemas.messages import Message, StopReason, Usage

FINISH_REASON_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def map_finish_reason(reason: str | None) -> StopReason | None:
    """Map an OpenAI ``finish_reason`` to an Anthropic ``stop_reason``."""

    if reason is None:
        return None
    return FINISH_REASON_MAP.get(reason, "end_turn")


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        parsed = orjson.loads(raw) if raw else {}
    except orjson.JSONDecodeError:
        library_logger().warning("tool arguments were not valid JSON; preserving raw value")
        return {"_raw_arguments": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def usage_from_openai(payload: dict[str, Any] | None) -> Usage:
    """Normalize OpenAI usage, subtracting cached tokens from the input count."""

    if not payload:
        return Usage()

    prompt_tokens = payload.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = payload.get("input_tokens", 0)
    completion_tokens = payload.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = payload.get("output_tokens", 0)

    details = payload.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)

    completion_details = payload.get("completion_tokens_details") or {}
    reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)

    return Usage(
        input_tokens=max(0, prompt_tokens - cached),
        output_tokens=completion_tokens,
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=0,
        output_tokens_details={"thinking_tokens": reasoning_tokens} if reasoning_tokens else None,
        service_tier=payload.get("service_tier"),
    )


def _content_from_message(message: dict[str, Any]) -> list[Any]:
    blocks: list[Any] = []

    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        blocks.append(ThinkingBlock(thinking=reasoning))

    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append(TextBlock(text=content))
    elif isinstance(content, list):
        text = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        )
        if text:
            blocks.append(TextBlock(text=text))

    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal:
        blocks.append(TextBlock(text=refusal))

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        blocks.append(
            ToolUseBlock(
                id=str(tool_call.get("id") or f"toolu_{uuid.uuid4().hex}"),
                name=str(function.get("name") or ""),
                input=_parse_arguments(str(function.get("arguments") or "")),
            )
        )

    return blocks


def openai_to_anthropic_message(
    payload: dict[str, Any],
    *,
    request_model: str | None = None,
) -> Message:
    """Convert one OpenAI Chat Completion response into an Anthropic ``Message``.

    Handles text, refusals, reasoning traces, and tool calls. Invalid tool
    argument JSON is preserved under ``_raw_arguments``.
    """

    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("OpenAI response contained no choices")
    if len(choices) > 1:
        library_logger().debug("multiple choices in response; using index 0")

    choice = choices[0]
    message = choice.get("message") or {}
    content = _content_from_message(message)

    stop_reason = map_finish_reason(choice.get("finish_reason"))
    if message.get("refusal"):
        stop_reason = "refusal"

    data: dict[str, Any] = {
        "id": str(payload.get("id") or f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": str(payload.get("model") or request_model or ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_from_openai(payload.get("usage")).model_dump(),
    }
    if payload.get("created") is not None:
        data["created"] = payload["created"]
    if payload.get("service_tier") is not None:
        data["service_tier"] = payload["service_tier"]

    return Message.model_validate(data)


def anthropic_message_from_payload(payload: dict[str, Any]) -> Message:
    """Validate a native Anthropic Messages response payload."""

    return Message.model_validate(payload)
