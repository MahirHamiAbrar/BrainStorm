from __future__ import annotations

from typing import Any, Literal

from .base import BrainStormModel


class ErrorDetail(BrainStormModel):
    """Provider-agnostic error fields from an OpenAI or Anthropic envelope."""

    type: str
    message: str
    code: str | None = None
    param: str | None = None


class AnthropicErrorEnvelope(BrainStormModel):
    """The ``{"type": "error", "error": {...}}`` shape used by Anthropic routes."""

    type: Literal["error"] = "error"
    error: ErrorDetail


class OpenAIErrorEnvelope(BrainStormModel):
    """The ``{"error": {...}}`` shape used by OpenAI-compatible routes."""

    error: ErrorDetail


class OpenAIErrorBody(BrainStormModel):
    """Fields inside an OpenAI ``error`` object."""

    message: str
    type: str | None = None
    code: str | None = None
    param: str | None = None


class WireError(BrainStormModel):
    """Normalized HTTP failure details captured before raising a client error."""

    status_code: int
    error_type: str
    message: str
    code: str | None = None
    param: str | None = None
    request_id: str | None = None
    body: dict[str, Any] | None = None
