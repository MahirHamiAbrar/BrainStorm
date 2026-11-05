from __future__ import annotations

from typing import Any, Mapping

from ..schemas.models import ModelInfo
from ..schemas.pricing import ModelPricing, PricingMode
from .base import BaseProvider
from .openai_compat import OpenAIWireMixin


class OpenRouterProvider(OpenAIWireMixin, BaseProvider):
    """OpenRouter gateway over its OpenAI-compatible Chat Completions API."""

    name = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    api_key_env = ("OPENROUTER_API_KEY",)
    requires_api_key = True

    def wire_for(self, model: str) -> str:
        """OpenRouter serves every model through Chat Completions."""

        return "openai"

    def _model_info_from_item(self, item: Mapping[str, Any]) -> ModelInfo:
        model_id = str(item.get("id") or "")
        architecture = item.get("architecture") or {}
        top_provider = item.get("top_provider") or {}
        supported_parameters = set(item.get("supported_parameters") or [])
        reasoning = item.get("reasoning") or {}
        return ModelInfo(
            id=model_id,
            name=str(item.get("name") or model_id),
            provider=self.name,
            context_length=item.get("context_length") or top_provider.get("context_length"),
            max_output_tokens=top_provider.get("max_completion_tokens"),
            supported_endpoints=list(item.get("endpoints") or []),
            input_modalities=list(architecture.get("input_modalities") or []),
            supports_tools="tools" in supported_parameters,
            supports_reasoning=bool(reasoning) or "reasoning" in supported_parameters,
            reasoning_efforts=list(reasoning.get("supported_efforts") or []),
            description=item.get("description"),
            created=item.get("created"),
        )

    def model_prices(self, pricing_type: PricingMode = "any") -> ModelPricing:
        """Return OpenRouter's catalog prices, converted to per-million rates."""

        from ..pricing import model_prices

        return model_prices(self.name, pricing_type, base_url=self.base_url)
