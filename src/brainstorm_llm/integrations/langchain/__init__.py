"""LangChain and LangGraph integration.

Install LangChain (or just ``langchain-core``) to use this module::

    from brainstorm_llm.integrations.langchain import ChatBrainStorm

``ChatBrainStorm`` is a standard ``BaseChatModel``: tools, structured output,
streaming, async calls, usage metadata, and graph composition work through the
normal LangChain APIs. ``BrainStormEmbeddings`` is a standard ``Embeddings``
implementation for providers with an embeddings endpoint. ``BrainStormDecisions``
is a Runnable mirroring LangChain's decisions integrations for System One models.
"""

from __future__ import annotations

from .chat import ChatBrainStorm
from .decisions import (
    Answer,
    BrainStormDecisions,
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    Level,
    Predicate,
    PredicateAnswer,
    Question,
    RefusalAnswer,
    Score,
    ScoreAnswer,
    State,
    Usage,
    as_tool,
)
from .embeddings import BrainStormEmbeddings

__all__ = [
    "Answer",
    "BrainStormDecisions",
    "BrainStormEmbeddings",
    "ChatBrainStorm",
    "Choice",
    "ChoiceAnswer",
    "DecisionRequest",
    "DecisionResponse",
    "Level",
    "Predicate",
    "PredicateAnswer",
    "Question",
    "RefusalAnswer",
    "Score",
    "ScoreAnswer",
    "State",
    "Usage",
    "as_tool",
]
