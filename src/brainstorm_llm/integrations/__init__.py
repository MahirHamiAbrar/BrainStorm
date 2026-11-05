"""Optional integrations for agent frameworks.

Import the concrete module directly, for example::

    from brainstorm_llm.integrations.langchain import ChatBrainStorm

The submodules import their framework dependencies lazily, so the core package
does not require LangChain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .langchain import BrainStormDecisions, ChatBrainStorm

__all__ = ["ChatBrainStorm", "BrainStormDecisions"]


def __getattr__(name: str) -> Any:
    if name in {"ChatBrainStorm", "BrainStormDecisions"}:
        from .langchain import BrainStormDecisions, ChatBrainStorm

        return {"ChatBrainStorm": ChatBrainStorm, "BrainStormDecisions": BrainStormDecisions}[name]
    raise AttributeError(name)
