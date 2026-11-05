"""Convert LangChain messages and content blocks to BrainStorm's canonical schema."""

from __future__ import annotations

import base64
import re
import uuid
from collections.abc import Sequence
from typing import Any, Literal, cast

import orjson
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)

from ...schemas import (
    Base64ImageSource,
    Base64PDFSource,
    DocumentBlock,
    FileDocumentSource,
    FileImageSource,
    ImageBlock,
    Message,
    MessageParam,
    PlainTextSource,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    URLImageSource,
    URLPDFSource,
    Usage,
)
from ...usage import CallUsage

ImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
_IMAGE_MEDIA_TYPES: set[ImageMediaType] = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}
_DATA_URL = re.compile(r"^data:([^;,]+);base64,(.*)$", re.DOTALL)
_SUPPORTED_TOOL_RESULT_BLOCKS = (TextBlock, ImageBlock, DocumentBlock)


def _normalize_media_type(value: Any) -> ImageMediaType:
    media_type = str(value or "image/png").lower()
    if media_type == "image/jpg":
        media_type = "image/jpeg"
    if media_type not in _IMAGE_MEDIA_TYPES:
        raise ValueError(
            f"Unsupported image MIME type {media_type!r}; "
            f"expected one of {sorted(_IMAGE_MEDIA_TYPES)}"
        )
    return cast("ImageMediaType", media_type)


def _parse_data_url(url: str) -> tuple[ImageMediaType, str] | None:
    match = _DATA_URL.match(url)
    if match is None:
        return None
    return _normalize_media_type(match.group(1)), match.group(2)


def _image_block(block: dict[str, Any]) -> ImageBlock | None:
    block_type = block.get("type")
    if block_type == "image":
        source = block.get("source")
        if isinstance(source, dict):
            source_type = source.get("type")
            if source_type == "base64":
                return ImageBlock(
                    source=Base64ImageSource(
                        media_type=_normalize_media_type(source.get("media_type")),
                        data=str(source.get("data") or ""),
                    )
                )
            if source_type == "url":
                return ImageBlock(source=URLImageSource(url=str(source.get("url") or "")))
            if source_type == "file":
                return ImageBlock(source=FileImageSource(file_id=str(source.get("file_id") or "")))

        base64_data = block.get("base64")
        if base64_data:
            return ImageBlock(
                source=Base64ImageSource(
                    media_type=_normalize_media_type(block.get("mime_type")),
                    data=str(base64_data),
                )
            )
        if block.get("url"):
            return ImageBlock(source=URLImageSource(url=str(block["url"])))
        if block.get("file_id"):
            return ImageBlock(source=FileImageSource(file_id=str(block["file_id"])))
        return None

    if block_type in {"image_url", "input_image"}:
        raw = block.get("image_url")
        if isinstance(raw, dict):
            url = raw.get("url")
        elif isinstance(raw, str):
            url = raw
        else:
            url = None
        if not isinstance(url, str):
            return None
        parsed = _parse_data_url(url)
        if parsed is not None:
            media_type, data = parsed
            return ImageBlock(source=Base64ImageSource(media_type=media_type, data=data))
        return ImageBlock(source=URLImageSource(url=url))

    return None


def _document_block(block: dict[str, Any]) -> DocumentBlock | None:
    block_type = block.get("type")
    if block_type == "document":
        source = block.get("source")
        if isinstance(source, dict):
            source_type = source.get("type")
            if source_type == "base64":
                media_type = str(source.get("media_type") or "")
                if media_type == "application/pdf":
                    return DocumentBlock(
                        source=Base64PDFSource(data=str(source.get("data") or ""))
                    )
                if media_type.startswith("text/"):
                    raw = str(source.get("data") or "")
                    return DocumentBlock(
                        source=PlainTextSource(
                            data=base64.b64decode(raw).decode("utf-8", errors="replace")
                        )
                    )
            if source_type == "url":
                return DocumentBlock(source=URLPDFSource(url=str(source.get("url") or "")))
            if source_type == "file":
                return DocumentBlock(
                    source=FileDocumentSource(file_id=str(source.get("file_id") or ""))
                )
        return None

    if block_type == "text-plain":
        return DocumentBlock(source=PlainTextSource(data=str(block.get("text") or "")))

    if block_type == "file":
        base64_data = block.get("base64")
        media_type = str(block.get("mime_type") or "")
        if base64_data and media_type == "application/pdf":
            return DocumentBlock(source=Base64PDFSource(data=str(base64_data)))
        if base64_data and media_type.startswith("text/"):
            return DocumentBlock(
                source=PlainTextSource(
                    data=base64.b64decode(str(base64_data)).decode("utf-8", errors="replace")
                )
            )
        if block.get("url"):
            return DocumentBlock(source=URLPDFSource(url=str(block["url"])))
        if block.get("file_id"):
            return DocumentBlock(source=FileDocumentSource(file_id=str(block["file_id"])))
    return None


def _tool_use_from_dict(block: dict[str, Any]) -> ToolUseBlock:
    raw_args = block.get("input", block.get("args", {}))
    if isinstance(raw_args, str):
        try:
            args = orjson.loads(raw_args)
        except orjson.JSONDecodeError:
            args = {"_raw_arguments": raw_args}
    else:
        args = raw_args
    if not isinstance(args, dict):
        args = {"value": args}
    return ToolUseBlock(
        id=str(block.get("id") or f"toolu_{uuid.uuid4().hex}"),
        name=str(block.get("name") or "tool"),
        input=args,
    )


def _tool_result_from_dict(block: dict[str, Any]) -> ToolResultBlock:
    content = block.get("content")
    if isinstance(content, str):
        result: Any = content
    elif isinstance(content, list):
        result = [
            converted
            for item in content
            if (converted := _block_from_dict(item)) is not None
            and isinstance(converted, _SUPPORTED_TOOL_RESULT_BLOCKS)
        ]
    else:
        result = "" if content is None else str(content)
    return ToolResultBlock(
        tool_use_id=str(block.get("tool_use_id") or block.get("tool_call_id") or ""),
        content=result,
        is_error=bool(block.get("is_error")),
    )


def _block_from_dict(block: dict[str, Any]) -> Any:
    block_type = block.get("type")
    if block_type in {"text", "text-plain", "input_text"}:
        return TextBlock(text=str(block.get("text") or ""))
    if block_type in {"image", "image_url", "input_image"}:
        return _image_block(block)
    if block_type in {"file", "document"}:
        return _document_block(block)
    if block_type in {"tool_use", "tool_call"}:
        return _tool_use_from_dict(block)
    if block_type == "tool_result":
        return _tool_result_from_dict(block)
    if block_type in {"reasoning", "thinking"}:
        summary = block.get("reasoning")
        if summary is None and isinstance(block.get("summary"), list):
            summary = "".join(
                str(part.get("text", ""))
                for part in block["summary"]
                if isinstance(part, dict)
            )
        extras: dict[str, Any] = (
            block["extras"] if isinstance(block.get("extras"), dict) else {}
        )
        return ThinkingBlock(
            thinking=str(summary or block.get("thinking") or ""),
            signature=extras.get("signature") or block.get("signature"),
        )
    if block_type == "refusal":
        return TextBlock(text=str(block.get("refusal") or ""))
    return None


def _blocks_from_content(content: Any) -> list[Any]:
    if isinstance(content, str):
        return [TextBlock(text=content)] if content else []
    if not isinstance(content, list):
        return []
    blocks: list[Any] = []
    for item in content:
        if isinstance(item, str):
            if item:
                blocks.append(TextBlock(text=item))
        elif isinstance(item, dict):
            block = _block_from_dict(item)
            if block is not None:
                blocks.append(block)
    return blocks


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "text-plain", "input_text"}
        )
    if isinstance(content, dict):
        return str(content.get("text") or "")
    return str(content or "")


def _human_param(message: HumanMessage) -> MessageParam | None:
    content = message.content
    if isinstance(content, str):
        return MessageParam(role="user", content=content) if content else None
    blocks = _blocks_from_content(content)
    return MessageParam(role="user", content=blocks) if blocks else None


def _ai_param(message: AIMessage) -> MessageParam | None:
    blocks = _blocks_from_content(message.content)
    reasoning = message.additional_kwargs.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning and not any(
        isinstance(block, ThinkingBlock) for block in blocks
    ):
        blocks.insert(0, ThinkingBlock(thinking=reasoning))

    seen_ids = {block.id for block in blocks if isinstance(block, ToolUseBlock)}
    for tool_call in message.tool_calls:
        tool_id = str(tool_call.get("id") or f"toolu_{uuid.uuid4().hex}")
        if tool_id in seen_ids:
            continue
        raw_args = tool_call.get("args", {})
        if isinstance(raw_args, str):
            try:
                args = orjson.loads(raw_args)
            except orjson.JSONDecodeError:
                args = {"_raw_arguments": raw_args}
        else:
            args = raw_args
        if not isinstance(args, dict):
            args = {"value": args}
        blocks.append(
            ToolUseBlock(
                id=tool_id,
                name=str(tool_call.get("name") or "tool"),
                input=args,
            )
        )
        seen_ids.add(tool_id)

    if not blocks:
        return None
    return MessageParam(role="assistant", content=blocks)


def _tool_result(message: ToolMessage) -> ToolResultBlock:
    if not message.tool_call_id:
        raise ValueError("ToolMessage is missing tool_call_id")
    content = message.content
    if isinstance(content, str):
        result: Any = content
    else:
        converted = _blocks_from_content(content)
        result = [
            block for block in converted if isinstance(block, _SUPPORTED_TOOL_RESULT_BLOCKS)
        ]
    return ToolResultBlock(
        tool_use_id=message.tool_call_id,
        content=result,
        is_error=message.status == "error",
    )


def convert_messages(
    messages: Sequence[BaseMessage | Any],
) -> tuple[str | None, list[MessageParam]]:
    """Convert LangChain messages into a system prompt and Anthropic messages.

    Consecutive tool messages are coalesced into one user message so parallel
    tool results arrive together, as the Anthropic wire expects.
    """

    normalized = convert_to_messages(list(messages))
    system_parts: list[str] = []
    converted: list[MessageParam] = []
    pending_tool_results: list[ToolResultBlock] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            converted.append(
                MessageParam(role="user", content=list(pending_tool_results))
            )
            pending_tool_results.clear()

    for message in normalized:
        if isinstance(message, RemoveMessage):
            continue

        if isinstance(message, SystemMessage):
            flush_tool_results()
            text = _content_text(message.content)
            if text:
                system_parts.append(text)
            continue

        if isinstance(message, ToolMessage):
            pending_tool_results.append(_tool_result(message))
            continue

        flush_tool_results()

        if isinstance(message, HumanMessage):
            param = _human_param(message)
            if param is not None:
                converted.append(param)
        elif isinstance(message, AIMessage):
            param = _ai_param(message)
            if param is not None:
                converted.append(param)
        elif isinstance(message, ChatMessage):
            if message.role == "system":
                text = _content_text(message.content)
                if text:
                    system_parts.append(text)
            elif message.role == "assistant":
                param = _ai_param(AIMessage(content=message.content))
                if param is not None:
                    converted.append(param)
            else:
                param = _human_param(HumanMessage(content=message.content))
                if param is not None:
                    converted.append(param)
        elif isinstance(message, FunctionMessage):
            converted.append(
                MessageParam(
                    role="user",
                    content=f"{message.name}: {_content_text(message.content)}",
                )
            )
        else:
            text = _content_text(getattr(message, "content", ""))
            if text:
                converted.append(MessageParam(role="user", content=text))

    flush_tool_results()
    return ("\n\n".join(system_parts) if system_parts else None), converted


def usage_metadata(usage: Usage) -> dict[str, Any]:
    """Convert BrainStorm usage into LangChain's ``UsageMetadata`` shape."""

    details = usage.output_tokens_details or {}
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "input_token_details": {
            "cache_read": usage.cache_read_input_tokens or 0,
            "cache_creation": usage.cache_creation_input_tokens or 0,
        },
        "output_token_details": {
            "reasoning": int(details.get("thinking_tokens", 0) or 0),
        },
    }


def response_metadata(
    message: Message,
    record: CallUsage | None = None,
) -> dict[str, Any]:
    """Build provider metadata for a LangChain ``AIMessage``."""

    metadata: dict[str, Any] = {
        "model_name": message.model,
        "provider": record.provider if record is not None else None,
        "stop_reason": message.stop_reason,
        "stop_sequence": message.stop_sequence,
        "request_id": message.request_id
        or (record.request_id if record is not None else None),
    }
    if record is not None:
        metadata["cost_usd"] = str(record.cost_usd) if record.cost_usd is not None else None
        metadata["pricing_type"] = record.pricing_type
        metadata["latency_ms"] = record.latency_ms
    return metadata


def to_ai_message(message: Message, record: CallUsage | None = None) -> AIMessage:
    """Convert a BrainStorm ``Message`` into a LangChain ``AIMessage``."""

    content_blocks: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    invalid_tool_calls: list[dict[str, Any]] = []
    refusal_text: str | None = None

    for block in message.content:
        if isinstance(block, TextBlock):
            if message.stop_reason == "refusal":
                refusal_text = (refusal_text or "") + block.text
            else:
                content_blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingBlock):
            block_data: dict[str, Any] = {
                "type": "reasoning",
                "reasoning": block.thinking,
            }
            if block.signature:
                block_data["extras"] = {"signature": block.signature}
            content_blocks.append(block_data)
        elif isinstance(block, ToolUseBlock):
            if "_raw_arguments" in block.input:
                invalid_tool_calls.append(
                    {
                        "name": block.name,
                        "args": block.input["_raw_arguments"],
                        "id": block.id,
                        "error": "Tool arguments were not valid JSON.",
                    }
                )
            else:
                tool_calls.append(
                    {
                        "name": block.name,
                        "args": block.input,
                        "id": block.id,
                        "type": "tool_call",
                    }
                )
        elif getattr(block, "type", None) in {"redacted_thinking", "server_tool_use"}:
            content_blocks.append(
                {
                    "type": "non_standard",
                    "value": block.model_dump(mode="json", exclude_none=True),
                }
            )

    if message.stop_reason == "refusal":
        content: str | list[dict[str, Any]] = [
            {"type": "refusal", "refusal": refusal_text or ""}
        ]
    elif content_blocks and all(block["type"] == "text" for block in content_blocks):
        content = "".join(block["text"] for block in content_blocks)
    elif content_blocks:
        content = content_blocks
    else:
        content = ""

    return AIMessage(
        id=message.id,
        content=cast("Any", content),
        tool_calls=cast("Any", tool_calls),
        invalid_tool_calls=cast("Any", invalid_tool_calls),
        usage_metadata=usage_metadata(
            record.usage if record is not None else message.usage
        ),
        response_metadata=response_metadata(message, record),
    )
