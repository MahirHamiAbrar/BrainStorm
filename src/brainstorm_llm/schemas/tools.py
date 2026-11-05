from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import Field

from .base import BrainStormModel
from .content import CacheControlEphemeral


class Tool(BrainStormModel):
    """A function tool the model may call.

    ``input_schema`` is a JSON Schema object describing the function arguments.
    """

    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    cache_control: CacheControlEphemeral | None = None
    strict: bool | None = None


class ToolChoiceAuto(BrainStormModel):
    """Let the model decide whether and which tool to call."""

    type: Literal["auto"] = "auto"
    disable_parallel_tool_use: bool | None = None


class ToolChoiceAny(BrainStormModel):
    """Require the model to call at least one tool."""

    type: Literal["any"] = "any"
    disable_parallel_tool_use: bool | None = None


class ToolChoiceTool(BrainStormModel):
    """Force a specific named tool."""

    type: Literal["tool"] = "tool"
    name: str
    disable_parallel_tool_use: bool | None = None


class ToolChoiceNone(BrainStormModel):
    """Disable tool use for this request."""

    type: Literal["none"] = "none"


ToolChoice = Annotated[
    Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool, ToolChoiceNone],
    Field(discriminator="type"),
]
