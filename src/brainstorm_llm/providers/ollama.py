from __future__ import annotations

from ..schemas.pricing import ModelPricing, PricingMode
from .base import BaseProvider
from .openai_compat import OpenAIWireMixin
from .openai_embeddings import OpenAIEmbeddingWireMixin


class OllamaProvider(OpenAIEmbeddingWireMixin, OpenAIWireMixin, BaseProvider):
    """Ollama's OpenAI-compatible endpoint, local or Ollama Cloud.

    Chat and embedding models share the same ``/v1`` base URL.

    Args:
        host: Base host such as ``http://localhost:11434``. ``/v1`` is appended
            automatically when missing.
        base_url: Explicit API root, taking precedence over ``host``.
    """

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
        """Ollama serves every model through its OpenAI-compatible endpoint."""

        return "openai"

    def model_prices(self, pricing_type: PricingMode = "any") -> ModelPricing:
        """Return zero prices for local inference, or catalog-zero cloud prices."""

        from ..pricing import model_prices

        return model_prices(self.name, pricing_type, base_url=self.base_url)
