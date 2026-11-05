from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from ..log import library_logger
from ..schemas.content import TextBlock, ThinkingBlock, ToolUseBlock
from ..schemas.messages import Message, StopReason, Usage
from ..schemas.streaming import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJSONDelta,
    MessageDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    SignatureDelta,
    TextDelta,
    ThinkingDelta,
)
from .response import map_finish_reason, usage_from_openai


@dataclass
class _ToolState:
    content_index: int
    tool_id: str
    name: str
    arguments: str = ""


@dataclass
class OpenAIStreamTranslator:
    model: str
    message_id: str | None = None
    created: int | None = None
    request_id: str | None = None
    started: bool = False
    finished: bool = False
    stop_reason: StopReason | None = None
    usage: Usage = field(default_factory=Usage)
    next_index: int = 0
    text_index: int | None = None
    thinking_index: int | None = None
    open_blocks: dict[int, str] = field(default_factory=dict)
    tool_states: dict[int, _ToolState] = field(default_factory=dict)

    def _new_block(self, kind: str) -> int:
        index = self.next_index
        self.next_index += 1
        self.open_blocks[index] = kind
        return index

    def _close(self, index: int) -> ContentBlockStopEvent | None:
        if index in self.open_blocks:
            self.open_blocks.pop(index)
            return ContentBlockStopEvent(index=index)
        return None

    def _close_text_and_thinking(self) -> list[Any]:
        events: list[Any] = []
        if self.text_index is not None:
            event = self._close(self.text_index)
            if event:
                events.append(event)
            self.text_index = None
        if self.thinking_index is not None:
            event = self._close(self.thinking_index)
            if event:
                events.append(event)
            self.thinking_index = None
        return events

    def _start(self, chunk: dict[str, Any]) -> list[Any]:
        if self.started:
            return []
        self.started = True
        self.message_id = str(chunk.get("id") or self.message_id or f"msg_{uuid.uuid4().hex}")
        model = str(chunk.get("model") or self.model)
        if chunk.get("created") is not None:
            self.created = chunk["created"]
        message = Message(
            id=self.message_id,
            content=[],
            model=model,
            stop_reason=None,
            usage=Usage(),
            request_id=self.request_id,
        )
        return [MessageStartEvent(message=message)]

    def _ensure_text(self) -> list[Any]:
        if self.text_index is not None:
            return []
        index = self._new_block("text")
        self.text_index = index
        return [
            ContentBlockStartEvent(index=index, content_block=TextBlock(text="")),
        ]

    def _ensure_thinking(self) -> list[Any]:
        if self.thinking_index is not None:
            return []
        index = self._new_block("thinking")
        self.thinking_index = index
        return [
            ContentBlockStartEvent(index=index, content_block=ThinkingBlock(thinking="")),
        ]

    def _ensure_tool(self, tool_delta: dict[str, Any]) -> list[Any]:
        key = tool_delta.get("index")
        if not isinstance(key, int):
            key = len(self.tool_states)
        if key in self.tool_states:
            return []

        events = self._close_text_and_thinking()
        function = tool_delta.get("function") or {}
        index = self._new_block("tool_use")
        state = _ToolState(
            content_index=index,
            tool_id=str(tool_delta.get("id") or f"toolu_{uuid.uuid4().hex}"),
            name=str(function.get("name") or ""),
        )
        self.tool_states[key] = state
        events.append(
            ContentBlockStartEvent(
                index=index,
                content_block=ToolUseBlock(id=state.tool_id, name=state.name, input={}),
            )
        )
        return events

    def process_chunk(self, chunk: dict[str, Any]) -> list[Any]:
        if chunk.get("error"):
            return []

        events: list[Any] = []
        events.extend(self._start(chunk))

        if chunk.get("usage"):
            self.usage = usage_from_openai(chunk.get("usage"))

        choices = chunk.get("choices") or []
        for choice in choices:
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}

            content = delta.get("content")
            if isinstance(content, str) and content:
                events.extend(self._ensure_text())
                events.append(
                    ContentBlockDeltaEvent(
                        index=self.text_index or 0,
                        delta=TextDelta(text=content),
                    )
                )

            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                events.extend(self._ensure_thinking())
                events.append(
                    ContentBlockDeltaEvent(
                        index=self.thinking_index or 0,
                        delta=ThinkingDelta(thinking=reasoning),
                    )
                )

            for tool_delta in delta.get("tool_calls") or []:
                events.extend(self._ensure_tool(tool_delta))
                key = tool_delta.get("index")
                if not isinstance(key, int):
                    key = len(self.tool_states) - 1
                state = self.tool_states.get(key)
                if state is None:
                    continue
                function = tool_delta.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    state.arguments += arguments
                    events.append(
                        ContentBlockDeltaEvent(
                            index=state.content_index,
                            delta=InputJSONDelta(partial_json=arguments),
                        )
                    )

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                self.stop_reason = map_finish_reason(finish_reason)
                if any(self.tool_states.values()) and self.stop_reason == "end_turn":
                    self.stop_reason = "tool_use"

        return events

    def finalize(self) -> list[Any]:
        if self.finished:
            return []
        self.finished = True

        events: list[Any] = []
        if not self.started:
            events.extend(self._start({}))

        for index in sorted(self.open_blocks):
            event = self._close(index)
            if event:
                events.append(event)

        stop_reason = self.stop_reason
        if stop_reason is None:
            stop_reason = "tool_use" if self.tool_states else "end_turn"

        events.append(
            MessageDeltaEvent(
                delta=MessageDelta(stop_reason=stop_reason, stop_sequence=None),
                usage=self.usage,
            )
        )
        events.append(MessageStopEvent())
        return events
