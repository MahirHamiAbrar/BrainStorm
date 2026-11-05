from __future__ import annotations

import copy
from typing import Annotated, Any, Literal, Union

import orjson
from pydantic import Field

from .base import BrainStormModel
from .content import ContentBlock, TextBlock, ThinkingBlock, ToolUseBlock
from .messages import Message, StopReason, Usage


class TextDelta(BrainStormModel):
    """A fragment of text appended to the current text block."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class InputJSONDelta(BrainStormModel):
    """A fragment of a tool call's JSON arguments."""

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ThinkingDelta(BrainStormModel):
    """A fragment of reasoning text."""

    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class SignatureDelta(BrainStormModel):
    """A fragment of a thinking block signature."""

    type: Literal["signature_delta"] = "signature_delta"
    signature: str


class CitationsDelta(BrainStormModel):
    """A citation attached to the current text block."""

    type: Literal["citations_delta"] = "citations_delta"
    citation: dict[str, Any]


ContentBlockDelta = Annotated[
    Union[TextDelta, InputJSONDelta, ThinkingDelta, SignatureDelta, CitationsDelta],
    Field(discriminator="type"),
]


class MessageStartEvent(BrainStormModel):
    """First stream event: message identity, model, and initial usage."""

    type: Literal["message_start"] = "message_start"
    message: Message


class ContentBlockStartEvent(BrainStormModel):
    """Start of a text, thinking, or tool_use block at ``index``."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlock


class ContentBlockDeltaEvent(BrainStormModel):
    """Incremental content for the block at ``index``."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: ContentBlockDelta


class ContentBlockStopEvent(BrainStormModel):
    """End of the content block at ``index``."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDelta(BrainStormModel):
    """Final stop reason information for a streamed message."""

    stop_reason: StopReason | None = None
    stop_sequence: str | None = None


class MessageDeltaEvent(BrainStormModel):
    """Terminal metadata: stop reason and final usage for the message."""

    type: Literal["message_delta"] = "message_delta"
    delta: MessageDelta
    usage: Usage = Field(default_factory=Usage)


class MessageStopEvent(BrainStormModel):
    """The stream is complete."""

    type: Literal["message_stop"] = "message_stop"


class PingEvent(BrainStormModel):
    """Keepalive event; safe to ignore."""

    type: Literal["ping"] = "ping"


class StreamErrorDetail(BrainStormModel):
    """Error type and message carried by an ``error`` stream event."""

    type: str
    message: str


class StreamErrorEvent(BrainStormModel):
    """An error emitted by the provider during a stream."""

    type: Literal["error"] = "error"
    error: StreamErrorDetail


class UnknownStreamEvent(BrainStormModel):
    """Fallback for event types added after this release."""

    type: str


RawMessageStreamEvent = Annotated[
    Union[
        MessageStartEvent,
        ContentBlockStartEvent,
        ContentBlockDeltaEvent,
        ContentBlockStopEvent,
        MessageDeltaEvent,
        MessageStopEvent,
        PingEvent,
        StreamErrorEvent,
        UnknownStreamEvent,
    ],
    Field(union_mode="left_to_right"),
]


class MessageAccumulator:
    """Rebuild a complete :class:`Message` from stream events.

    Feed every event to :meth:`feed`, then call :meth:`finalize`. Tool call
    argument fragments are parsed into ``ToolUseBlock.input`` at block end.
    """

    def __init__(self) -> None:
        self._message: Message | None = None
        self._positions: dict[int, int] = {}
        self._tool_buffers: dict[int, str] = {}

    def feed(self, event: RawMessageStreamEvent) -> None:
        """Apply one stream event to the in-progress message."""

        if isinstance(event, MessageStartEvent):
            self._message = event.message.model_copy(deep=True)
            self._message.content = list(self._message.content)
            return

        if event.type == "ping" or event.type == "error":
            return

        message = self._message
        if message is None:
            return

        if isinstance(event, ContentBlockStartEvent):
            self._positions[event.index] = len(message.content)
            message.content.append(copy.deepcopy(event.content_block))
            if isinstance(event.content_block, ToolUseBlock):
                self._tool_buffers[event.index] = ""
            return

        if isinstance(event, ContentBlockDeltaEvent):
            position = self._positions.get(event.index)
            if position is None or position >= len(message.content):
                return
            block = message.content[position]
            delta = event.delta
            if isinstance(delta, TextDelta) and isinstance(block, TextBlock):
                block.text += delta.text
            elif isinstance(delta, ThinkingDelta) and isinstance(block, ThinkingBlock):
                block.thinking += delta.thinking
            elif isinstance(delta, SignatureDelta) and isinstance(block, ThinkingBlock):
                block.signature = (block.signature or "") + delta.signature
            elif isinstance(delta, InputJSONDelta):
                self._tool_buffers[event.index] = (
                    self._tool_buffers.get(event.index, "") + delta.partial_json
                )
            elif isinstance(delta, CitationsDelta) and isinstance(block, TextBlock):
                existing = block.citations or []
                existing.append(delta.citation)
                block.citations = existing
            return

        if isinstance(event, ContentBlockStopEvent):
            position = self._positions.get(event.index)
            if position is None or position >= len(message.content):
                return
            block = message.content[position]
            if isinstance(block, ToolUseBlock):
                raw = self._tool_buffers.pop(event.index, "")
                if raw.strip():
                    try:
                        parsed = orjson.loads(raw)
                        block.input = parsed if isinstance(parsed, dict) else {"value": parsed}
                    except orjson.JSONDecodeError:
                        block.input = {"_raw_arguments": raw}
            return

        if isinstance(event, MessageDeltaEvent):
            message.stop_reason = event.delta.stop_reason
            message.stop_sequence = event.delta.stop_sequence
            merged = message.usage.model_dump()
            merged.update(event.usage.model_dump(exclude_none=True, exclude_unset=True))
            message.usage = Usage.model_validate(merged)
            return

    @property
    def message(self) -> Message | None:
        """The in-progress message, or ``None`` before ``message_start``."""

        return self._message

    def finalize(self) -> Message:
        """Return the assembled message.

        Raises:
            ValueError: if no ``message_start`` event was fed.
        """

        if self._message is None:
            raise ValueError("stream ended before a message_start event")
        return self._message
