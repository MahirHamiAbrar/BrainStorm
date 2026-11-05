from __future__ import annotations

from typing import Any, Literal

from .base import BrainStormModel

ModelKind = Literal["chat", "decision"]


class ModelInfo(BrainStormModel):
    """Catalog metadata for one model.

    Returned by :func:`brainstorm_llm.list_models`; contains no pricing data.
    ``kind`` distinguishes chat models from System One decision models.
    """

    id: str
    name: str
    provider: str
    kind: ModelKind = "chat"
    context_length: int | None = None
    max_output_tokens: int | None = None
    supported_endpoints: list[str] = []
    input_modalities: list[str] = []
    supports_tools: bool | None = None
    supports_reasoning: bool | None = None
    reasoning_efforts: list[str] = []
    description: str | None = None
    created: int | None = None
    release_date: str | None = None
    api: dict[str, Any] = {}
