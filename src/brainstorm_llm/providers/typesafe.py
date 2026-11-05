from __future__ import annotations

from typing import Any, Mapping

from ..errors import ConfigurationError, PricingUnavailableError
from ..schemas.models import ModelInfo
from ..schemas.pricing import ModelPricing, PricingMode
from .base import BaseProvider
from .systemone import SystemOneWireMixin


class TypeSafeProvider(SystemOneWireMixin, BaseProvider):
    """Direct TypeSafe System One API for decision models."""

    name = "typesafe"
    default_base_url = "https://api.typesafe.ai/v1"
    api_key_env = ("TYPESAFE_API_KEY",)
    requires_api_key = True

    def wire_for(self, model: str) -> str:
        """TypeSafe only serves System One decision requests."""

        return "systemone"

    def _model_info_from_item(self, item: Mapping[str, Any]) -> ModelInfo:
        model_id = str(item.get("name") or item.get("id") or "")
        return ModelInfo(
            id=model_id,
            name=model_id,
            provider=self.name,
            kind="decision",
            description=item.get("description"),
            release_date=item.get("release_date"),
        )

    def model_prices(self, pricing_type: PricingMode = "any") -> ModelPricing:
        """TypeSafe does not publish machine-readable prices."""

        raise PricingUnavailableError(
            "TypeSafe does not publish machine-readable pricing; use the Command Code provider "
            "for Jev pricing or consult https://docs.typesafe.ai/."
        )

    def generate(self, params, **kwargs):
        """Not supported; TypeSafe only serves System One decisions."""

        raise ConfigurationError("The TypeSafe provider only serves System One decision requests")

    async def agenerate(self, params, **kwargs):
        """Not supported; TypeSafe only serves System One decisions."""

        raise ConfigurationError("The TypeSafe provider only serves System One decision requests")

    def stream(self, params, **kwargs):
        """Not supported; System One decisions never stream."""

        raise ConfigurationError("System One decisions never stream")

    def astream(self, params, **kwargs):
        """Not supported; System One decisions never stream."""

        raise ConfigurationError("System One decisions never stream")
