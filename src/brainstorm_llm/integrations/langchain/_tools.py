"""Convert LangChain tool definitions and tool choices to BrainStorm types."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from ...schemas import (
    Tool,
    ToolChoice,
    ToolChoiceAny,
    ToolChoiceAuto,
    ToolChoiceNone,
    ToolChoiceTool,
)

_DEFAULT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def convert_tool(tool: Any) -> Tool:
    """Convert a LangChain tool, callable, Pydantic model, or dict to ``Tool``."""

    if isinstance(tool, Tool):
        return tool

    if isinstance(tool, dict):
        if "input_schema" in tool and "name" in tool:
            return Tool.model_validate(tool)
        if "function" in tool and isinstance(tool["function"], dict):
            function = tool["function"]
        elif "name" in tool and "parameters" in tool:
            function = tool
        elif tool.get("type") == "function":
            function = tool.get("function") or {}
        else:
            raise ValueError(
                "Only function tools are supported by BrainStorm; "
                f"got tool type {tool.get('type')!r}."
            )
    else:
        try:
            openai_tool = convert_to_openai_tool(tool)
        except Exception as exc:  # noqa: BLE001 - normalize framework conversion errors
            raise ValueError(f"Could not convert tool {tool!r} to a function tool") from exc
        function = openai_tool.get("function") if isinstance(openai_tool, dict) else None
        if not isinstance(function, dict):
            raise ValueError(f"Could not convert tool {tool!r} to a function tool")

    name = str(function.get("name") or "")
    if not name:
        raise ValueError("Tool definitions must have a non-empty name")
    parameters = function.get("parameters")
    return Tool(
        name=name,
        description=function.get("description"),
        input_schema=parameters if isinstance(parameters, dict) else dict(_DEFAULT_SCHEMA),
        strict=function.get("strict"),
    )


def convert_tools(tools: Sequence[Any]) -> list[Tool]:
    """Convert a sequence of LangChain tools into BrainStorm tools."""

    return [convert_tool(tool) for tool in tools]


def _with_parallel_flag(choice: ToolChoice, parallel_tool_calls: bool | None) -> ToolChoice:
    if parallel_tool_calls is None:
        return choice
    disable_parallel = not parallel_tool_calls
    if isinstance(choice, (ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool)):
        return choice.model_copy(update={"disable_parallel_tool_use": disable_parallel})
    return choice


def convert_tool_choice(
    choice: Any,
    parallel_tool_calls: bool | None = None,
) -> ToolChoice | None:
    """Convert a LangChain ``tool_choice`` value to a BrainStorm tool choice."""

    if choice is None:
        return None
    if isinstance(choice, (ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool, ToolChoiceNone)):
        return _with_parallel_flag(choice, parallel_tool_calls)
    if isinstance(choice, str):
        if choice == "auto":
            return _with_parallel_flag(ToolChoiceAuto(), parallel_tool_calls)
        if choice in {"any", "required"}:
            return _with_parallel_flag(ToolChoiceAny(), parallel_tool_calls)
        if choice == "none":
            return ToolChoiceNone()
        return _with_parallel_flag(ToolChoiceTool(name=choice), parallel_tool_calls)
    if isinstance(choice, dict):
        choice_type = choice.get("type")
        if choice_type == "function":
            function = choice.get("function") or {}
            name = function.get("name") if isinstance(function, dict) else None
        elif choice_type == "tool":
            name = choice.get("name")
        else:
            name = choice.get("name")
        if name:
            return _with_parallel_flag(ToolChoiceTool(name=str(name)), parallel_tool_calls)
        if choice_type in {"auto", "any", "required", "none"}:
            return convert_tool_choice(choice_type, parallel_tool_calls)
    raise ValueError(f"Unsupported tool_choice value: {choice!r}")


__all__ = ["BaseTool", "convert_tool", "convert_tool_choice", "convert_tools"]
