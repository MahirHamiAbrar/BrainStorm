from __future__ import annotations

import os
import time
from typing import Any, Iterator, Mapping

import httpx
import orjson

from ..errors import ConfigurationError, MissingAPIKeyError, PricingUnavailableError, UnsupportedModelError
from ..schemas.models import ModelInfo
from ..schemas.pricing import ModelPricing, PricingMode
from ..transport.http import AsyncHttpTransport, HttpTransport, RetryPolicy, TransportConfig

MODEL_CACHE_TTL = 3600.0


def resolve_api_key(
    provider: str,
    explicit: str | None,
    env_vars: tuple[str, ...],
    *,
    required: bool,
) -> str | None:
    """Resolve an explicit key or the first non-empty environment variable."""

    if explicit:
        return explicit
    for name in env_vars:
        value = os.getenv(name)
        if value:
            return value
    if required:
        raise MissingAPIKeyError(provider, list(env_vars))
    return None


class BaseProvider:
    """Shared HTTP, catalog, retry, and pricing behavior for providers.

    Subclasses combine this with a wire mixin (OpenAI, Anthropic, or System One)
    and implement :meth:`wire_for`. A provider owns one sync and one async
    ``httpx`` client; close them with :meth:`close` or :meth:`aclose`.

    Args:
        api_key: Explicit key; falls back to the provider's environment variables.
        base_url: API root override.
        headers: Extra headers for every request.
        timeout: Request timeout in seconds.
        max_retries: Retry count for transient failures.
        zdr: Send ``x-cmd-zdr: 1`` when the provider supports it.
        retry_policy: Optional fully customized retry policy.
        transport: Optional ``httpx`` transport, primarily for tests.
        async_transport: Optional async ``httpx`` transport, primarily for tests.
    """

    name = "base"
    default_base_url = ""
    default_model: str | None = None
    api_key_env: tuple[str, ...] = ()
    requires_api_key = True
    zdr_supported = False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        zdr: bool = False,
        retry_policy: RetryPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = resolve_api_key(self.name, api_key, self.api_key_env, required=self.requires_api_key)
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        if not self.base_url:
            raise ConfigurationError(f"Provider '{self.name}' has no base URL")
        self.zdr = bool(zdr and self.zdr_supported)
        self.user_headers = dict(headers or {})
        self.timeout = timeout
        self.max_retries = max_retries
        self._retry = retry_policy or RetryPolicy(max_retries=max_retries)
        config = TransportConfig(
            base_url=self.base_url,
            api_key=self.api_key,
            headers=self.user_headers,
            timeout=timeout,
            retry=self._retry,
        )
        self._sync = HttpTransport(config, transport=transport)
        self._async = AsyncHttpTransport(config, transport=async_transport)
        self._models_cache: list[ModelInfo] | None = None
        self._models_cached_at: float = 0.0

    # -- lifecycle -----------------------------------------------------
    def close(self) -> None:
        """Close the pooled synchronous HTTP client."""

        self._sync.close()

    async def aclose(self) -> None:
        """Close the pooled asynchronous HTTP client."""

        await self._async.aclose()

    def __enter__(self) -> "BaseProvider":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    async def __aenter__(self) -> "BaseProvider":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # -- request helpers ----------------------------------------------
    def request_headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """Build per-request headers, including the ZDR flag when enabled."""

        headers: dict[str, str] = {}
        if self.zdr:
            headers["x-cmd-zdr"] = "1"
        if extra:
            headers.update(extra)
        return headers

    def post(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        """POST a JSON body through the retrying synchronous transport."""

        return self._sync.request(
            "POST",
            path,
            json_body=payload,
            headers=self.request_headers(headers),
            stream=stream,
        )

    async def apost(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        """POST a JSON body through the retrying asynchronous transport."""

        return await self._async.request(
            "POST",
            path,
            json_body=payload,
            headers=self.request_headers(headers),
            stream=stream,
        )

    def get(
        self,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """GET a path through the retrying synchronous transport."""

        return self._sync.request("GET", path, headers=self.request_headers(headers))

    async def aget(
        self,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """GET a path through the retrying asynchronous transport."""

        return await self._async.request("GET", path, headers=self.request_headers(headers))

    @staticmethod
    def json_body(response: httpx.Response) -> dict[str, Any]:
        """Decode a response body with orjson and require a JSON object."""

        payload = orjson.loads(response.content)
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object response")
        return payload

    # -- model catalog -------------------------------------------------
    models_path = "/models"

    def _model_info_from_item(self, item: Mapping[str, Any]) -> ModelInfo:
        model_id = str(item.get("id") or item.get("name") or "")
        return ModelInfo(
            id=model_id,
            name=str(item.get("name") or model_id),
            provider=self.name,
            context_length=item.get("context_length") or item.get("context_window"),
            max_output_tokens=item.get("max_output_tokens") or item.get("max_completion_tokens"),
            supported_endpoints=list(item.get("supported_endpoints") or []),
            input_modalities=list(item.get("input_modalities") or []),
            supports_tools=item.get("supports_tools"),
            supports_reasoning=item.get("supports_reasoning"),
            description=item.get("description"),
            created=item.get("created"),
        )

    def _parse_model_list(self, payload: Mapping[str, Any]) -> list[ModelInfo]:
        items = payload.get("data") or payload.get("models") or []
        models: list[ModelInfo] = []
        for item in items:
            if isinstance(item, dict):
                info = self._model_info_from_item(item)
                if info.id:
                    models.append(info)
        return models

    def list_models(self, *, force_refresh: bool = False) -> list[ModelInfo]:
        """Fetch the provider's model catalog, cached in memory for one hour.

        Args:
            force_refresh: Ignore the in-process cache and request the catalog.
        """

        now = time.monotonic()
        if (
            not force_refresh
            and self._models_cache is not None
            and now - self._models_cached_at < MODEL_CACHE_TTL
        ):
            return list(self._models_cache)

        response = self.get(self.models_path)
        models = self._parse_model_list(self.json_body(response))
        self._models_cache = models
        self._models_cached_at = now
        return list(models)

    async def alist_models(self, *, force_refresh: bool = False) -> list[ModelInfo]:
        """Async wrapper around :meth:`list_models`."""

        import asyncio

        return await asyncio.to_thread(self.list_models, force_refresh=force_refresh)

    def find_model(self, model_id: str, *, force_refresh: bool = False) -> ModelInfo | None:
        """Return catalog metadata for ``model_id``, or ``None`` if unknown."""

        for model in self.list_models(force_refresh=force_refresh):
            if model.id == model_id:
                return model
        return None

    # -- pricing -------------------------------------------------------
    def model_prices(self, pricing_type: PricingMode = "any") -> ModelPricing:
        """Return per-model prices; providers without pricing raise."""

        raise PricingUnavailableError(f"Provider '{self.name}' does not expose pricing data")

    async def amodel_prices(self, pricing_type: PricingMode = "any") -> ModelPricing:
        """Async wrapper around :meth:`model_prices`."""

        import asyncio

        return await asyncio.to_thread(self.model_prices, pricing_type)

    # -- generation (implemented by wire mixins) -----------------------
    def generate(self, params, **kwargs):  # pragma: no cover - abstract
        """Run one non-streaming request; implemented by a wire mixin."""

        raise NotImplementedError

    async def agenerate(self, params, **kwargs):  # pragma: no cover - abstract
        """Async version of :meth:`generate`."""

        raise NotImplementedError

    def stream(self, params, **kwargs) -> Iterator[Any]:  # pragma: no cover - abstract
        """Stream one request; implemented by a wire mixin."""

        raise NotImplementedError

    def astream(self, params, **kwargs):  # pragma: no cover - abstract
        """Async version of :meth:`stream`."""

        raise NotImplementedError

    # -- model routing -------------------------------------------------
    def wire_for(self, model: str) -> str:
        """Return the wire to use for ``model`` (``openai``, ``anthropic``, ...).

        Implemented by each provider; used by the client's per-model routing.
        """

        raise NotImplementedError

    def ensure_model_routable(self, model: str) -> None:
        """Raise :class:`UnsupportedModelError` when no implemented wire serves the model."""

        info = self.find_model(model)
        if info is not None and info.supported_endpoints and self.wire_for(model) == "unsupported":
            raise UnsupportedModelError(
                f"Model '{model}' does not support a wire implemented by this client",
                status_code=400,
                error_type="unsupported_model",
                code="unsupported_model",
            )
