from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Literal, Mapping, TypeVar

from pydantic import BaseModel

from .errors import BrainStormError, ConfigurationError
from .log import library_logger
from .providers import BaseProvider, get_provider
from .schemas.messages import Usage
from .schemas.models import ModelInfo
from .schemas.pricing import ModelPricing, PricingMode
from .schemas.systemone import (
    Choice,
    Image,
    Noul,
    Score,
    SystemOneRequest,
    SystemOneResponse,
)
from .usage import CallUsage, UsageCallback, aemit_usage, emit_usage

SystemOneResponseT = TypeVar("SystemOneResponseT", bound=BaseModel)
PricingPreference = Literal["off", "cached", "any", "live"]

DEFAULT_S1_MODEL = "typesafe/jev-1.14.0"


class BrainStormS1:
    """System One decision client with TypeSafe-SDK-compatible types.

    Sends a shared ``state`` plus named typed questions and returns probabilities,
    not text.

    Args:
        model: Decision model id. Defaults to ``typesafe/jev-1.14.0`` for the
            Command Code and TypeSafe providers.
        provider: ``"commandcode"``, ``"typesafe"``, ``"opencode"``, or a
            provider instance.
        api_key: Overrides the provider's environment variable.
        base_url: Overrides the provider's default endpoint.
        headers: Extra HTTP headers for every request.
        timeout: Request timeout in seconds.
        max_retries: Retries for transient statuses and connection errors.
        zdr: Zero-data-retention header. Defaults to ``False`` because Command
            Code's ``/systemone`` refuses it.
        pricing: ``"off"``, ``"cached"``, ``"any"``, or ``"live"`` cost lookup mode.
        on_usage: Default callback invoked with a :class:`CallUsage` after each call.

    Example:
        >>> s1 = BrainStormS1()
        >>> result = s1.system_one("ticket text", {"urgent": Noul(instructions="Urgent?")})
        >>> result.nouls["urgent"].noul
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: str | BaseProvider = "commandcode",
        api_key: str | None = None,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        zdr: bool | None = None,
        pricing: PricingPreference = "any",
        on_usage: UsageCallback | None = None,
    ) -> None:
        if isinstance(provider, BaseProvider):
            self._provider = provider
            if zdr is not None:
                self._provider.zdr = bool(zdr and self._provider.zdr_supported)
        else:
            # System One on Command Code has no ZDR-capable upstream and refuses the header.
            default_zdr = False
            self._provider = get_provider(
                provider,
                api_key=api_key,
                base_url=base_url,
                headers=headers,
                timeout=timeout,
                max_retries=max_retries,
                zdr=default_zdr if zdr is None else zdr,
            )

        self.provider_name = self._provider.name
        if model is None and self.provider_name in {"commandcode", "typesafe"}:
            model = DEFAULT_S1_MODEL
        if not model:
            raise ConfigurationError(
                f"Provider '{self.provider_name}' has no default decision model; pass model=."
            )
        self.model: str = model
        self.pricing_mode = pricing
        self.on_usage = on_usage
        self.models = _ModelsResource(self)
        self._pricing: ModelPricing | None = None
        self._pricing_loaded = False
        self._pricing_lock = threading.Lock()
        self._warned_models: set[str] = set()

    # -- lifecycle -----------------------------------------------------
    def close(self) -> None:
        """Close the pooled synchronous HTTP client."""

        self._provider.close()

    async def aclose(self) -> None:
        """Close the pooled asynchronous HTTP client."""

        await self._provider.aclose()

    def __enter__(self) -> "BrainStormS1":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    async def __aenter__(self) -> "BrainStormS1":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # -- pricing -------------------------------------------------------
    def _load_pricing(self) -> ModelPricing | None:
        if self._pricing_loaded:
            return self._pricing
        with self._pricing_lock:
            if self._pricing_loaded:
                return self._pricing
            self._pricing_loaded = True
            if self.pricing_mode == "off":
                return None
            if self.pricing_mode == "live":
                mode: PricingMode = "live"
            elif self.pricing_mode == "cached":
                mode = "cached"
            else:
                mode = "any"
            try:
                self._pricing = self._provider.model_prices(mode)
            except BrainStormError as exc:
                library_logger(self.provider_name, self.model).debug(
                    "pricing unavailable: {}", exc
                )
            except Exception as exc:  # noqa: BLE001 - pricing must not break calls
                library_logger(self.provider_name, self.model).warning(
                    "pricing lookup failed: {}", exc
                )
            return self._pricing

    def _make_record(
        self,
        usage: Usage,
        *,
        request_id: str | None,
        latency_ms: float,
    ) -> CallUsage:
        cost = None
        pricing_type = None
        pricing = self._load_pricing()
        if pricing is not None:
            price = pricing.prices.get(self.model or "")
            if price is None:
                if self.model not in self._warned_models:
                    self._warned_models.add(self.model or "")
                    library_logger(self.provider_name, self.model).warning(
                        "no pricing entry for {}; cost will be reported as None", self.model
                    )
            else:
                from .pricing import compute_cost

                cost = compute_cost(price, usage)
                pricing_type = pricing.pricing_type

        return CallUsage(
            provider=self.provider_name,
            model=self.model or "",
            request_id=request_id,
            usage=usage,
            cost_usd=cost,
            pricing_type=pricing_type,
            latency_ms=latency_ms,
            streaming=False,
            operation="system_one",
        )

    # -- decisions -----------------------------------------------------
    def _request(
        self,
        state: Any,
        questions: Mapping[str, Noul | Choice | Score | Mapping[str, Any]],
        model: str | None,
    ) -> SystemOneRequest:
        return SystemOneRequest.model_validate(
            {
                "model": model or self.model,
                "state": state,
                "questions": dict(questions),
            }
        )

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Noul | Choice | Score | Mapping[str, Any]],
        *,
        model: str | None = None,
        assets: Mapping[str, Image] | None = None,
        response_model: type[SystemOneResponseT] | None = None,
        headers: Mapping[str, str] | None = None,
        on_usage: UsageCallback | None = None,
    ) -> SystemOneResponse | SystemOneResponseT:
        """Evaluate typed questions against one state.

        Args:
            state: Shared context as a string, object, or array.
            questions: Named ``Noul``, ``Choice``, or ``Score`` questions.
            model: Per-call model override.
            assets: Named images referenced by placeholder in ``state`` or questions.
            response_model: Optional ``SystemOneResponse`` subclass or Pydantic
                model used to validate and expose the answers.
            headers: Extra headers for this call.
            on_usage: Per-call usage callback in addition to the client callback.

        Returns:
            A :class:`SystemOneResponse` with answers keyed by question name.

        Raises:
            pydantic.ValidationError: if the request violates System One limits.
            APIStatusError: if the provider rejects the request.
        """

        request = self._request(state, questions, model)
        method = getattr(self._provider, "system_one", None)
        if method is None:
            raise ConfigurationError(
                f"Provider '{self.provider_name}' does not support System One decisions"
            )

        log = library_logger(self.provider_name, self.model)
        log.debug("system_one: questions={}", len(request.questions))
        start = time.perf_counter()
        response = method(
            request,
            assets=assets,
            headers=dict(headers) if headers else None,
        )
        elapsed = (time.perf_counter() - start) * 1000
        log.debug("system_one finished in {:.0f} ms", elapsed)
        usage = Usage(
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
        )
        record = self._make_record(usage, request_id=response.request_id, latency_ms=elapsed)
        emit_usage(self.on_usage, record)
        if on_usage is not None and on_usage is not self.on_usage:
            emit_usage(on_usage, record)
        return _apply_response_model(response, response_model)

    async def asystem_one(
        self,
        state: Any,
        questions: Mapping[str, Noul | Choice | Score | Mapping[str, Any]],
        *,
        model: str | None = None,
        assets: Mapping[str, Image] | None = None,
        response_model: type[SystemOneResponseT] | None = None,
        headers: Mapping[str, str] | None = None,
        on_usage: UsageCallback | None = None,
    ) -> SystemOneResponse | SystemOneResponseT:
        """Async version of :meth:`system_one`."""

        request = self._request(state, questions, model)
        method = getattr(self._provider, "asystem_one", None)
        if method is None:
            raise ConfigurationError(
                f"Provider '{self.provider_name}' does not support System One decisions"
            )

        log = library_logger(self.provider_name, self.model)
        log.debug("asystem_one: questions={}", len(request.questions))
        start = time.perf_counter()
        response = await method(
            request,
            assets=assets,
            headers=dict(headers) if headers else None,
        )
        elapsed = (time.perf_counter() - start) * 1000
        log.debug("asystem_one finished in {:.0f} ms", elapsed)
        usage = Usage(
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
        )
        record = await asyncio.to_thread(
            self._make_record,
            usage,
            request_id=response.request_id,
            latency_ms=elapsed,
        )
        await aemit_usage(self.on_usage, record)
        if on_usage is not None and on_usage is not self.on_usage:
            await aemit_usage(on_usage, record)
        return _apply_response_model(response, response_model)

    # -- catalog / pricing ---------------------------------------------
    def list_models(self, *, kind: str | None = None, **kwargs: Any) -> list[ModelInfo]:
        """List available models, optionally filtered by ``kind``.

        Use ``kind="decision"`` to show only System One models.
        """

        models = self._provider.list_models(**kwargs)
        if kind is not None:
            models = [model for model in models if model.kind == kind]
        return models

    def model_prices(self, pricing_type: PricingMode = "any") -> ModelPricing:
        """Return the provider's price table for decision models."""

        return self._provider.model_prices(pricing_type)


class _ModelsResource:
    """TypeSafe-SDK-style ``models`` resource exposed as ``s1.models``."""

    def __init__(self, client: "BrainStormS1") -> None:
        self._client = client

    def list(self, *, kind: str | None = None, **kwargs: Any) -> list[ModelInfo]:
        """List models, optionally filtered to ``"decision"``."""

        return self._client.list_models(kind=kind, **kwargs)


def _apply_response_model(
    response: SystemOneResponse,
    response_model: type[SystemOneResponseT] | None,
) -> SystemOneResponse | SystemOneResponseT:
    if response_model is None:
        return response

    data = response.model_dump(mode="python")
    if issubclass(response_model, SystemOneResponse):
        data.update(response.answers)
        result = response_model.model_validate(data)
        result.request_id = response.request_id
        result.raw_http_response = response.raw_http_response
        return result
    return response_model.model_validate(data)
