from __future__ import annotations

from typing import Any, Literal

from .base import BrainStormModel


class ErrorDetail(BrainStormModel):
    type: str
    message: str
    code: str | None = None
    param: str | None = None


class AnthropicErrorEnvelope(BrainStormModel):
    type: Literal["error"] = "error"
    error: ErrorDetail


class OpenAIErrorEnvelope(BrainStormModel):
    error: ErrorDetail


class OpenAIErrorBody(BrainStormModel):
    message: str
    type: str | None = None
    code: str | None = None
    param: str | None = None


class WireError(BrainStormModel):
    status_code: int
    error_type: str
    message: str
    code: str | None = None
    param: str | None = None
    request_id: str | None = None
    body: dict[str, Any] | None = None
