from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AliasChoices, ConfigDict, Field

from .base import BrainStormModel

PricingMode = Literal["live", "cached", "any"]
PricingSource = Literal["live", "cached"]


class TokenPrice(BrainStormModel):
    """Prices for one model in USD per one million tokens.

    The JSON form uses ``in`` and ``out`` keys; Python accepts either
    ``input=``/``output=`` or ``in=``/``out=``.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        frozen=True,
    )

    input: float = Field(
        validation_alias=AliasChoices("in", "input"),
        serialization_alias="in",
    )
    output: float = Field(
        validation_alias=AliasChoices("out", "output"),
        serialization_alias="out",
    )
    cache_read: float | None = None
    cache_write: float | None = None
    notes: str | None = None


class ModelPricing(BrainStormModel):
    """A provider's price table, keyed by model ID.

    ``pricing_type`` reports the source actually used, so a call made with
    ``model_prices(..., "any")`` can tell whether the values were live or cached.
    """

    provider: str
    prices: dict[str, TokenPrice]
    pricing_type: PricingSource
    fetched_at: datetime
    source: str
    currency: Literal["USD"] = "USD"
