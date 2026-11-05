from __future__ import annotations

from typing import Any, Mapping

from ..schemas.models import ModelInfo, model_kind_from_id
from ..schemas.pricing import ModelPricing, PricingMode
from .base import BaseProvider
from .openai_compat import OpenAIWireMixin
from .openai_embeddings import OpenAIEmbeddingWireMixin


class GeminiProvider(OpenAIEmbeddingWireMixin, OpenAIWireMixin, BaseProvider):
    """Google Gemini through its OpenAI-compatible endpoint."""

    name = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    api_key_env = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    requires_api_key = True

    def wire_for(self, model: str) -> str:
        """Gemini is called through its OpenAI-compatible endpoint."""

        return "openai"

    def _model_info_from_item(self, item: Mapping[str, Any]) -> ModelInfo:
        model_id = str(item.get("id") or item.get("name") or "").removeprefix("models/")
        return ModelInfo(
            id=model_id,
            name=str(item.get("display_name") or item.get("name") or model_id).removeprefix("models/"),
            provider=self.name,
            kind=model_kind_from_id(model_id),
            context_length=item.get("input_token_limit") or item.get("context_length"),
            max_output_tokens=item.get("output_token_limit"),
            created=item.get("created"),
        )

    def model_prices(self, pricing_type: PricingMode = "any") -> ModelPricing:
        """Return Gemini's scraped or cached price table."""

        from ..pricing import model_prices

        return model_prices(self.name, pricing_type, base_url=self.base_url)
