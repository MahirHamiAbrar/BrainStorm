from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BrainStormModel(BaseModel):
    """Tolerant base model for wire payloads."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=False,
    )


class StrictModel(BaseModel):
    """Request-side base model that rejects unknown fields."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=False,
    )
