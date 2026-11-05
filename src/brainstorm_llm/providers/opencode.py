from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, Mapping

from ..log import library_logger
from ..schemas.models import ModelInfo
from ..schemas.pricing import ModelPricing, PricingMode
from ..schemas.streaming import RawMessageStreamEvent
from .anthropic_compat import AnthropicWireMixin
from .base import BaseProvider
from .openai_compat import OpenAIWireMixin
from .systemone import SystemOneWireMixin


class OpenCodeProvider(OpenAIWireMixin, AnthropicWireMixin, SystemOneWireMixin, BaseProvider):
    name = "opencode"
    default_base_url = "https://opencode.ai/inference"
    api_key_env = ("OPENCODE_API_KEY",)
    requires_api_key = True
    models_path = "/v1/models"

    def __init__(self, *, plan: str = "payg", base_url: str | None = None, **kwargs) -> None:
        super().__init__(base_url=base_url, **kwargs)
        prefix = "/go" if plan == "go" else ""
        self.plan = plan
        self.chat_completions_path = f"{prefix}/openai/v1/chat/completions"
        self.messages_path = f"{prefix}/anthropic/v1/messages"
        self.systemone_path = "/systemone/v1/systemone"

    def wire_for(self, model: str) -> str:
        info: ModelInfo | None = None
        try:
            info = self.find_model(model)
        except Exception as exc:  # noqa: BLE001 - catalog is advisory
            library_logger(self.name, model).debug("model catalog lookup failed: {}", exc)

        if info is not None:
            api = getattr(info, "api", None) or {}
            if isinstance(api, dict) and api:
                if "messages" in api and "chat-completions" not in api and "responses" not in api:
                    return "anthropic"
                return "openai"

        if model.startswith(("claude", "qwen3.5-plus", "qwen3.6-plus", "qwen3.8-flash")):
            return "anthropic"
        return "openai"

    def _model_info_from_item(self, item: Mapping[str, Any]) -> ModelInfo:
        model_id = str(item.get("id") or "")
        limits = item.get("limit") or {}
        modalities = item.get("modalities") or {}
        capabilities = item.get("capabilities") or {}
        tools = capabilities.get("tools") or {}
        reasoning = capabilities.get("reasoning") or {}
        return ModelInfo(
            id=model_id,
            name=str(item.get("name") or model_id),
            provider=self.name,
            context_length=limits.get("context"),
            max_output_tokens=limits.get("output"),
            supported_endpoints=list((item.get("api") or {}).keys()),
            input_modalities=list(modalities.get("input") or []),
            supports_tools=bool(tools.get("supported")),
            supports_reasoning=bool(reasoning.get("supported")),
            reasoning_efforts=list(reasoning.get("effort") or []),
            description=item.get("description"),
            created=item.get("created"),
            api=item.get("api") or {},
        )

    def generate(self, params, **kwargs):
        if self.wire_for(params.model) == "anthropic":
            return AnthropicWireMixin.generate(self, params, **kwargs)
        return OpenAIWireMixin.generate(self, params, **kwargs)

    async def agenerate(self, params, **kwargs):
        if self.wire_for(params.model) == "anthropic":
            return await AnthropicWireMixin.agenerate(self, params, **kwargs)
        return await OpenAIWireMixin.agenerate(self, params, **kwargs)

    def stream(self, params, **kwargs) -> Iterator[RawMessageStreamEvent]:
        if self.wire_for(params.model) == "anthropic":
            yield from AnthropicWireMixin.stream(self, params, **kwargs)
        else:
            yield from OpenAIWireMixin.stream(self, params, **kwargs)

    async def astream(self, params, **kwargs) -> AsyncIterator[RawMessageStreamEvent]:
        if self.wire_for(params.model) == "anthropic":
            async for event in AnthropicWireMixin.astream(self, params, **kwargs):
                yield event
        else:
            async for event in OpenAIWireMixin.astream(self, params, **kwargs):
                yield event

    def model_prices(self, pricing_type: PricingMode = "any") -> ModelPricing:
        from ..pricing import model_prices

        return model_prices(self.name, pricing_type, base_url=self.base_url)
