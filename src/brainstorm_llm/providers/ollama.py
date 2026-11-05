from __future__ import annotations

from ..schemas.pricing import ModelPricing, PricingMode
from .base import BaseProvider
from .openai_compat import OpenAIWireMixin


class OllamaProvider(OpenAIWireMixin, BaseProvider):
    name = "ollama"
    default_base_url = "http://localhost:11434/v1"
    api_key_env = ("OLLAMA_API_KEY",)
    requires_api_key = False

    def __init__(self, *, host: str | None = None, base_url: str | None = None, **kwargs) -> None:
        target = base_url or host
        if target and not target.rstrip("/").endswith("/v1"):
            target = f"{target.rstrip('/')}/v1"
        super().__init__(base_url=target, **kwargs)

    def wire_for(self, model: str) -> str:
        return "openai"

    def model_prices(self, pricing_type: PricingMode = "any") -> ModelPricing:
        from ..pricing import model_prices

        return model_prices(self.name, pricing_type, base_url=self.base_url)
