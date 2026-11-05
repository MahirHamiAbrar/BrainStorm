from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from .base import BrainStormModel, StrictModel
from .content import ContentBlock, TextBlock
from .tools import Tool, ToolChoice

StopReason = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "refusal",
    "model_context_window_exceeded",
]


class Usage(BrainStormModel):
    """Token counts reported for one request.

    ``input_tokens`` excludes cache reads. ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens`` are reported separately so costs can be
    computed with the provider's cache rates.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = 0
    cache_read_input_tokens: int | None = 0
    output_tokens_details: dict[str, int] | None = None
    server_tool_use: dict[str, int] | None = None
    service_tier: str | None = None

    @property
    def total_tokens(self) -> int:
        """Sum of input, output, cache-read, and cache-write tokens."""

        return (
            self.input_tokens
            + self.output_tokens
            + (self.cache_creation_input_tokens or 0)
            + (self.cache_read_input_tokens or 0)
        )


def _coerce_content_blocks(value: Any) -> Any:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    return value


class MessageParam(BrainStormModel):
    """One request message.

    ``content`` may be a plain string or a list of typed content blocks.
    Use :meth:`to_blocks` when you always want a block list.
    """

    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, value: Any) -> Any:
        return _coerce_content_blocks(value) if not isinstance(value, str) else value

    def to_blocks(self) -> list[ContentBlock]:
        """Return the content as a block list, wrapping a string in a TextBlock."""

        if isinstance(self.content, str):
            return [TextBlock(text=self.content)]
        return self.content


class Message(BrainStormModel):
    """A provider response normalized to the Anthropic Messages shape.

    ``content`` holds typed blocks (text, thinking, tool_use, ...). Use
    :attr:`text` for concatenated text and :attr:`tool_uses` for tool calls.
    """

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    model: str
    stop_reason: StopReason | None = None
    stop_sequence: str | None = None
    usage: Usage = Field(default_factory=Usage)
    request_id: str | None = Field(default=None, exclude=True, repr=False)

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, value: Any) -> Any:
        return _coerce_content_blocks(value)

    @property
    def text(self) -> str:
        """Concatenated text from every ``TextBlock`` in the response."""

        return "".join(
            getattr(block, "text", "")
            for block in self.content
            if getattr(block, "type", None) == "text"
        )

    @property
    def tool_uses(self) -> list[Any]:
        """Every ``ToolUseBlock`` in the response."""

        return [block for block in self.content if getattr(block, "type", None) == "tool_use"]


class ThinkingConfig(BrainStormModel):
    """Reasoning controls passed through the Anthropic-style request surface."""

    type: Literal["enabled", "disabled"] = "enabled"
    budget_tokens: int | None = Field(default=None, gt=0)


class MessageCreateParams(StrictModel):
    """Validated arguments for one message request.

    Built internally by :class:`brainstorm_llm.BrainStormLLM`; use the client
    methods directly unless you are implementing a custom provider.
    """

    model: str
    messages: list[MessageParam]
    max_tokens: int = Field(gt=0)
    system: str | list[TextBlock] | None = None
    temperature: float | None = Field(default=None, ge=0, le=1)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    thinking: ThinkingConfig | None = None
    metadata: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    service_tier: str | None = None
    extra_body: dict[str, Any] | None = Field(default=None, exclude=True)

    def to_wire(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict, omitting unset fields."""

        return self.model_dump(mode="json", exclude_none=True)
