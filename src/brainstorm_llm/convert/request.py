from __future__ import annotations

from typing import Any

import orjson

from ..log import library_logger
from ..schemas.content import (
    DocumentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from ..schemas.messages import MessageCreateParams
from ..schemas.tools import ToolChoiceAny, ToolChoiceAuto, ToolChoiceNone, ToolChoiceTool


def system_text(system: str | list[TextBlock] | None) -> str | None:
    """Flatten a string or text-block system prompt into one string."""

    if system is None:
        return None
    if isinstance(system, str):
        return system
    return "\n".join(block.text for block in system if block.type == "text")


def _image_url(block: ImageBlock) -> str | None:
    source = block.source
    if source.type == "base64":
        return f"data:{source.media_type};base64,{source.data}"
    if source.type == "url":
        return source.url
    return None


def _tool_result_content(block: ToolResultBlock) -> str:
    if block.content is None:
        return ""
    if isinstance(block.content, str):
        return block.content
    parts: list[str] = []
    for part in block.content:
        if isinstance(part, TextBlock):
            parts.append(part.text)
        else:
            library_logger().debug("dropping non-text tool_result content for OpenAI wire")
    return "\n".join(parts)


def _convert_user_message(message) -> list[dict[str, Any]]:
    blocks = message.to_blocks()
    tool_messages: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []

    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            url = _image_url(block)
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                library_logger().debug("dropping unsupported file image for OpenAI wire")
        elif isinstance(block, ToolResultBlock):
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.tool_use_id,
                    "content": _tool_result_content(block),
                }
            )
        elif isinstance(block, DocumentBlock):
            library_logger().debug("dropping document block for OpenAI wire")
        else:
            library_logger().debug("dropping unsupported block {} for OpenAI wire", getattr(block, "type", "?"))

    result = list(tool_messages)
    if parts:
        if len(parts) == 1 and parts[0]["type"] == "text":
            result.append({"role": "user", "content": parts[0]["text"]})
        else:
            result.append({"role": "user", "content": parts})
    return result


def _convert_assistant_message(message) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in message.to_blocks():
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": orjson.dumps(block.input).decode("utf-8"),
                    },
                }
            )
        elif isinstance(block, ThinkingBlock):
            library_logger().debug("dropping thinking block for OpenAI wire")
        else:
            library_logger().debug("dropping unsupported block {} for OpenAI wire", getattr(block, "type", "?"))

    payload: dict[str, Any] = {"role": "assistant"}
    content = "\n".join(text_parts) if text_parts else None
    if content is not None:
        payload["content"] = content
    if tool_calls:
        payload["tool_calls"] = tool_calls
    if content is None and not tool_calls:
        payload["content"] = ""
    return payload


def _convert_messages(messages) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "assistant":
            converted.append(_convert_assistant_message(message))
        else:
            converted.extend(_convert_user_message(message))
    return converted


def _convert_tools(tools) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        function: dict[str, Any] = {
            "name": tool.name,
            "parameters": tool.input_schema,
        }
        if tool.description is not None:
            function["description"] = tool.description
        if tool.strict is not None:
            function["strict"] = tool.strict
        result.append({"type": "function", "function": function})
    return result


def _convert_tool_choice(choice) -> tuple[Any, bool | None]:
    if isinstance(choice, ToolChoiceAuto):
        return "auto", choice.disable_parallel_tool_use
    if isinstance(choice, ToolChoiceAny):
        return "required", choice.disable_parallel_tool_use
    if isinstance(choice, ToolChoiceTool):
        return {"type": "function", "function": {"name": choice.name}}, choice.disable_parallel_tool_use
    if isinstance(choice, ToolChoiceNone):
        return "none", None
    return None, None


def _reasoning_effort(budget_tokens: int | None) -> str | None:
    if budget_tokens is None:
        return None
    if budget_tokens < 4_000:
        return "low"
    if budget_tokens < 16_000:
        return "medium"
    return "high"


def anthropic_to_openai_payload(
    params: MessageCreateParams,
    *,
    stream: bool = False,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert Anthropic-style request parameters into an OpenAI Chat payload.

    Args:
        params: Validated Anthropic-style request arguments.
        stream: Request streaming and include the final usage chunk.
        extra_body: Provider-specific fields merged last, so they can override
            any generated field.
    """

    payload: dict[str, Any] = {
        "model": params.model,
        "messages": _convert_messages(params.messages),
        "max_tokens": params.max_tokens,
    }

    system = system_text(params.system)
    if system is not None:
        payload["messages"].insert(0, {"role": "system", "content": system})

    if params.temperature is not None:
        payload["temperature"] = params.temperature
    if params.top_p is not None:
        payload["top_p"] = params.top_p
    if params.top_k is not None:
        payload["top_k"] = params.top_k
    if params.stop_sequences:
        payload["stop"] = params.stop_sequences
    if params.metadata is not None:
        payload["metadata"] = params.metadata
    if params.service_tier is not None:
        payload["service_tier"] = params.service_tier

    if params.tools:
        payload["tools"] = _convert_tools(params.tools)
    if params.tool_choice is not None:
        choice, disable_parallel = _convert_tool_choice(params.tool_choice)
        if choice is not None:
            payload["tool_choice"] = choice
        if disable_parallel is not None:
            payload["parallel_tool_calls"] = not disable_parallel

    if params.thinking is not None and params.thinking.type == "enabled":
        effort = _reasoning_effort(params.thinking.budget_tokens)
        if effort is not None:
            payload["reasoning_effort"] = effort

    if params.output_config:
        output_format = params.output_config.get("format")
        if isinstance(output_format, dict) and output_format.get("type") == "json_schema":
            schema = output_format.get("schema", {"type": "object"})
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema},
            }

    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

    if extra_body:
        payload.update(extra_body)

    return payload
