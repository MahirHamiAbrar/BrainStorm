from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from ..log import library_logger
from ..schemas.models import ModelInfo
from ..schemas.pricing import ModelPricing, PricingMode
from ..schemas.streaming import RawMessageStreamEvent
from .anthropic_compat import AnthropicWireMixin
from .base import BaseProvider
from .openai_compat import OpenAIWireMixin
from .systemone import SystemOneWireMixin

DECISION_MODELS = (
    ModelInfo(
        id="typesafe/jev",
        name="Jev",
        provider="commandcode",
        kind="decision",
        context_length=32768,
        supported_endpoints=["/systemone"],
        description="Decision model: typed questions with probabilities, never text.",
    ),
    ModelInfo(
        id="typesafe/jev-1.14.0",
        name="Jev 1.14.0",
        provider="commandcode",
        kind="decision",
        context_length=32768,
        supported_endpoints=["/systemone"],
        description="Pinned 1.14.0 release of Jev.",
    ),
)


class CommandCodeProvider(OpenAIWireMixin, AnthropicWireMixin, SystemOneWireMixin, BaseProvider):
    """Command Code Provider API.

    Routes each model to the wire its catalog entry supports: Claude and other
    ``/messages``-only models use the Anthropic wire, everything else uses
    Chat Completions and is converted to the canonical shape. ZDR defaults are
    applied by the clients: on for ``BrainStormLLM``, off for ``BrainStormS1``.
    """

    name = "commandcode"
    default_base_url = "https://api.commandcode.ai/provider/v1"
    default_model = "deepseek/deepseek-v4.1-flash"
    api_key_env = ("COMMAND_API_KEY", "CMD_API_KEY")
    requires_api_key = True
    zdr_supported = True

    def wire_for(self, model: str) -> str:
        """Resolve the wire from ``supported_endpoints``, with a name-based fallback."""

        info: ModelInfo | None = None
        try:
            info = self.find_model(model)
        except Exception as exc:  # noqa: BLE001 - routing falls back to model name
            library_logger(self.name, model).debug("model catalog lookup failed: {}", exc)

        if info is not None and info.supported_endpoints:
            if "/chat/completions" in info.supported_endpoints:
                return "openai"
            if "/messages" in info.supported_endpoints:
                return "anthropic"
            return "unsupported"
        return "anthropic" if model.startswith("claude") else "openai"

    def _resolve_wire(self, model: str) -> str:
        wire = self.wire_for(model)
        if wire == "unsupported":
            self.ensure_model_routable(model)
        return wire

    def generate(self, params, **kwargs):
        """Generate on the wire selected for the requested model."""

        if self._resolve_wire(params.model) == "anthropic":
            return AnthropicWireMixin.generate(self, params, **kwargs)
        return OpenAIWireMixin.generate(self, params, **kwargs)

    async def agenerate(self, params, **kwargs):
        """Async version of :meth:`generate`."""

        if self._resolve_wire(params.model) == "anthropic":
            return await AnthropicWireMixin.agenerate(self, params, **kwargs)
        return await OpenAIWireMixin.agenerate(self, params, **kwargs)

    def stream(self, params, **kwargs) -> Iterator[RawMessageStreamEvent]:
        """Stream on the wire selected for the requested model."""

        if self._resolve_wire(params.model) == "anthropic":
            yield from AnthropicWireMixin.stream(self, params, **kwargs)
        else:
            yield from OpenAIWireMixin.stream(self, params, **kwargs)

    async def astream(self, params, **kwargs) -> AsyncIterator[RawMessageStreamEvent]:
        """Async version of :meth:`stream`."""

        if self._resolve_wire(params.model) == "anthropic":
            async for event in AnthropicWireMixin.astream(self, params, **kwargs):
                yield event
        else:
            async for event in OpenAIWireMixin.astream(self, params, **kwargs):
                yield event

    def list_models(
        self,
        *,
        force_refresh: bool = False,
        include_decision_models: bool = True,
    ) -> list[ModelInfo]:
        """Return the live catalog, optionally including Jev decision models."""

        models = super().list_models(force_refresh=force_refresh)
        if include_decision_models:
            known = {model.id for model in models}
            models.extend(model for model in DECISION_MODELS if model.id not in known)
        return models

    async def alist_models(
        self,
        *,
        force_refresh: bool = False,
        include_decision_models: bool = True,
    ) -> list[ModelInfo]:
        """Async wrapper around :meth:`list_models`."""

        import asyncio

        return await asyncio.to_thread(
            self.list_models,
            force_refresh=force_refresh,
            include_decision_models=include_decision_models,
        )

    def model_prices(self, pricing_type: PricingMode = "any") -> ModelPricing:
        """Return Command Code's scraped or cached price table."""

        from ..pricing import model_prices

        return model_prices(self.name, pricing_type, base_url=self.base_url)
