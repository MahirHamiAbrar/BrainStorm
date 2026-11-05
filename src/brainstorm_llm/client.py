from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, AsyncIterator, Iterator, Literal, Mapping, Sequence

from pydantic import ValidationError

from .errors import APIStreamError, BrainStormError, ConfigurationError
from .log import library_logger
from .providers import BaseProvider, get_provider
from .schemas.embeddings import EmbeddingCreateParams, EmbeddingResponse
from .schemas.messages import (
    Message,
    MessageCreateParams,
    MessageParam,
    ThinkingConfig,
    Usage,
)
from .schemas.models import ModelInfo
from .schemas.pricing import ModelPricing, PricingMode
from .schemas.streaming import (
    ContentBlockDeltaEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageAccumulator,
    RawMessageStreamEvent,
    TextDelta,
)
from .schemas.tools import Tool, ToolChoice
from .usage import CallUsage, UsageCallback, aemit_usage, emit_usage

DEFAULT_MAX_TOKENS = 4096
PricingPreference = Literal["off", "cached", "any", "live"]


class BrainStormLLM:
    """Anthropic-native LLM client for one provider.

    OpenAI-style providers are translated to and from the Anthropic Messages
    schema, so every call returns the same :class:`Message` type.

    Args:
        model: Model id. Defaults to ``deepseek/deepseek-v4.1-flash`` on Command
            Code; other providers require an explicit model.
        provider: Provider id or a preconfigured provider instance.
        api_key: Overrides the provider's environment variable.
        base_url: Overrides the provider's default endpoint.
        headers: Extra HTTP headers for every request.
        timeout: Request timeout in seconds.
        max_retries: Retries for transient statuses, timeouts, and connection errors.
        zdr: Command Code zero-data-retention header. Defaults to ``True`` for the
            Command Code provider and is ignored by other providers.
        pricing: ``"off"``, ``"cached"``, ``"any"``, or ``"live"`` cost lookup mode.
        on_usage: Default callback invoked with a :class:`CallUsage` after each call.

    Example:
        >>> llm = BrainStormLLM(model="deepseek/deepseek-v4.1-flash")
        >>> message = llm.generate([{"role": "user", "content": "Hello"}])
        >>> print(message.text)
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: str | BaseProvider = "commandcode",
        api_key: str | None = None,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
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
            default_zdr = provider == "commandcode"
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
        resolved_model = model or self._provider.default_model
        if not resolved_model:
            raise ConfigurationError(
                f"Provider '{self.provider_name}' has no default model; pass model=."
            )
        self.model: str = resolved_model
        self.pricing_mode = pricing
        self.on_usage = on_usage
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

    def __enter__(self) -> "BrainStormLLM":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    async def __aenter__(self) -> "BrainStormLLM":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # -- parameters ----------------------------------------------------
    def _build_params(
        self,
        messages: Sequence[MessageParam | Mapping[str, Any]],
        *,
        system: str | Sequence[Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        tools: Sequence[Tool | Mapping[str, Any]] | None = None,
        tool_choice: ToolChoice | Mapping[str, Any] | None = None,
        thinking: ThinkingConfig | Mapping[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        service_tier: str | None = None,
        extra_body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> MessageCreateParams:
        return MessageCreateParams.model_validate(
            {
                "model": self.model,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "system": system,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "stop_sequences": stop_sequences,
                "stream": stream,
                "tools": tools,
                "tool_choice": tool_choice,
                "thinking": thinking,
                "metadata": metadata,
                "output_config": output_config,
                "service_tier": service_tier,
                "extra_body": extra_body,
            }
        )

    # -- pricing / cost ------------------------------------------------
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

    def _cost_for(self, usage: Usage):
        pricing = self._load_pricing()
        if pricing is None:
            return None, None
        price = pricing.prices.get(self.model)
        if price is None:
            if self.model not in self._warned_models:
                self._warned_models.add(self.model)
                library_logger(self.provider_name, self.model).warning(
                    "no pricing entry for {}; cost will be reported as None", self.model
                )
            return None, pricing.pricing_type
        from .pricing import compute_cost

        return compute_cost(price, usage), pricing.pricing_type

    def _make_record(
        self,
        usage: Usage,
        *,
        request_id: str | None,
        latency_ms: float,
        streaming: bool,
        operation: Literal["generate", "stream"] = "generate",
    ) -> CallUsage:
        cost, pricing_type = self._cost_for(usage)
        return CallUsage(
            provider=self.provider_name,
            model=self.model,
            request_id=request_id,
            usage=usage,
            cost_usd=cost,
            pricing_type=pricing_type,
            latency_ms=latency_ms,
            streaming=streaming,
            operation=operation,
        )

    def _emit(self, record: CallUsage, per_call: UsageCallback | None) -> None:
        if self.on_usage is not None:
            emit_usage(self.on_usage, record)
        if per_call is not None and per_call is not self.on_usage:
            emit_usage(per_call, record)

    async def _aemit(self, record: CallUsage, per_call: UsageCallback | None) -> None:
        if self.on_usage is not None:
            await aemit_usage(self.on_usage, record)
        if per_call is not None and per_call is not self.on_usage:
            await aemit_usage(per_call, record)

    # -- generation ----------------------------------------------------
    def generate(
        self,
        messages: Sequence[MessageParam | Mapping[str, Any]],
        *,
        system: str | Sequence[Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        tools: Sequence[Tool | Mapping[str, Any]] | None = None,
        tool_choice: ToolChoice | Mapping[str, Any] | None = None,
        thinking: ThinkingConfig | Mapping[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        service_tier: str | None = None,
        extra_body: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        on_usage: UsageCallback | None = None,
    ) -> Message:
        """Run one non-streaming generation and return an Anthropic-style message.

        Args:
            messages: Conversation history; strings or typed content blocks.
            system: System prompt as a string or text-block list.
            max_tokens: Output token cap. Defaults to ``4096``.
            temperature: Sampling temperature, 0-1.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling cutoff.
            stop_sequences: Strings that stop generation.
            tools: Tool definitions the model may call.
            tool_choice: ``ToolChoiceAuto``, ``ToolChoiceAny``,
                ``ToolChoiceTool``, or ``ToolChoiceNone``.
            thinking: Optional reasoning budget.
            metadata: Provider metadata attached to the request.
            output_config: Structured-output configuration.
            service_tier: Provider service tier, when supported.
            extra_body: Provider-specific fields merged into the request body.
            headers: Extra headers for this call.
            on_usage: Per-call usage callback in addition to the client callback.

        Returns:
            A validated :class:`Message` with typed content blocks and usage.

        Raises:
            APIStatusError: for non-success provider responses.
            APIConnectionError: when the provider cannot be reached.
        """

        params = self._build_params(
            messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            thinking=thinking,
            metadata=metadata,
            output_config=output_config,
            service_tier=service_tier,
            extra_body=extra_body,
        )
        log = library_logger(self.provider_name, self.model)
        log.debug("generate: messages={} max_tokens={}", len(messages), max_tokens)
        start = time.perf_counter()
        message = self._provider.generate(
            params,
            extra_body=params.extra_body,
            headers=dict(headers) if headers else None,
        )
        elapsed = (time.perf_counter() - start) * 1000
        log.debug("generate finished in {:.0f} ms", elapsed)
        record = self._make_record(
            message.usage,
            request_id=message.request_id,
            latency_ms=elapsed,
            streaming=False,
            operation="generate",
        )
        self._emit(record, on_usage)
        return message

    async def agenerate(
        self,
        messages: Sequence[MessageParam | Mapping[str, Any]],
        *,
        system: str | Sequence[Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        tools: Sequence[Tool | Mapping[str, Any]] | None = None,
        tool_choice: ToolChoice | Mapping[str, Any] | None = None,
        thinking: ThinkingConfig | Mapping[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        service_tier: str | None = None,
        extra_body: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        on_usage: UsageCallback | None = None,
    ) -> Message:
        """Async version of :meth:`generate` using the pooled async client."""

        params = self._build_params(
            messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            thinking=thinking,
            metadata=metadata,
            output_config=output_config,
            service_tier=service_tier,
            extra_body=extra_body,
        )
        log = library_logger(self.provider_name, self.model)
        log.debug("agenerate: messages={} max_tokens={}", len(messages), max_tokens)
        start = time.perf_counter()
        message = await self._provider.agenerate(
            params,
            extra_body=params.extra_body,
            headers=dict(headers) if headers else None,
        )
        elapsed = (time.perf_counter() - start) * 1000
        log.debug("agenerate finished in {:.0f} ms", elapsed)
        record = await asyncio.to_thread(
            self._make_record,
            message.usage,
            request_id=message.request_id,
            latency_ms=elapsed,
            streaming=False,
            operation="generate",
        )
        await self._aemit(record, on_usage)
        return message

    # -- streaming -----------------------------------------------------
    def _stream_events(
        self,
        messages: Sequence[MessageParam | Mapping[str, Any]],
        *,
        on_usage: UsageCallback | None,
        **kwargs: Any,
    ) -> Iterator[RawMessageStreamEvent]:
        extra_body = kwargs.pop("extra_body", None)
        headers = kwargs.pop("headers", None)
        params = self._build_params(messages, stream=True, extra_body=extra_body, **kwargs)
        log = library_logger(self.provider_name, self.model)
        log.debug("stream: messages={} max_tokens={}", len(messages), params.max_tokens)
        start = time.perf_counter()
        usage = Usage()
        request_id: str | None = None
        completed = False
        try:
            for event in self._provider.stream(
                params,
                extra_body=params.extra_body,
                headers=dict(headers) if headers else None,
            ):
                if isinstance(event, MessageStartEvent):
                    usage = event.message.usage
                    request_id = event.message.request_id
                elif isinstance(event, MessageDeltaEvent):
                    merged = usage.model_dump()
                    merged.update(event.usage.model_dump(exclude_none=True, exclude_unset=True))
                    usage = Usage.model_validate(merged)
                yield event
            completed = True
        finally:
            if completed:
                elapsed = (time.perf_counter() - start) * 1000
                log.debug("stream finished in {:.0f} ms", elapsed)
                record = self._make_record(
                    usage,
                    request_id=request_id,
                    latency_ms=elapsed,
                    streaming=True,
                    operation="stream",
                )
                self._emit(record, on_usage)

    async def _astream_events(
        self,
        messages: Sequence[MessageParam | Mapping[str, Any]],
        *,
        on_usage: UsageCallback | None,
        **kwargs: Any,
    ) -> AsyncIterator[RawMessageStreamEvent]:
        extra_body = kwargs.pop("extra_body", None)
        headers = kwargs.pop("headers", None)
        params = self._build_params(messages, stream=True, extra_body=extra_body, **kwargs)
        log = library_logger(self.provider_name, self.model)
        log.debug("astream: messages={} max_tokens={}", len(messages), params.max_tokens)
        start = time.perf_counter()
        usage = Usage()
        request_id: str | None = None
        completed = False
        try:
            async for event in self._provider.astream(
                params,
                extra_body=params.extra_body,
                headers=dict(headers) if headers else None,
            ):
                if isinstance(event, MessageStartEvent):
                    usage = event.message.usage
                    request_id = event.message.request_id
                elif isinstance(event, MessageDeltaEvent):
                    merged = usage.model_dump()
                    merged.update(event.usage.model_dump(exclude_none=True, exclude_unset=True))
                    usage = Usage.model_validate(merged)
                yield event
            completed = True
        finally:
            if completed:
                elapsed = (time.perf_counter() - start) * 1000
                log.debug("astream finished in {:.0f} ms", elapsed)
                record = await asyncio.to_thread(
                    self._make_record,
                    usage,
                    request_id=request_id,
                    latency_ms=elapsed,
                    streaming=True,
                    operation="stream",
                )
                await self._aemit(record, on_usage)

    def stream(
        self,
        messages: Sequence[MessageParam | Mapping[str, Any]],
        *,
        on_usage: UsageCallback | None = None,
        **kwargs: Any,
    ) -> Iterator[RawMessageStreamEvent]:
        """Stream one generation as Anthropic stream events.

        Accepts the same keyword arguments as :meth:`generate`. The iterator
        ends with ``message_stop``; usage is reported after the final event.
        """

        return self._stream_events(messages, on_usage=on_usage, **kwargs)

    def astream(
        self,
        messages: Sequence[MessageParam | Mapping[str, Any]],
        *,
        on_usage: UsageCallback | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RawMessageStreamEvent]:
        """Async version of :meth:`stream`."""

        return self._astream_events(messages, on_usage=on_usage, **kwargs)

    def stream_text(
        self,
        messages: Sequence[MessageParam | Mapping[str, Any]],
        *,
        on_usage: UsageCallback | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield only text deltas from a streamed generation."""

        for event in self.stream(messages, on_usage=on_usage, **kwargs):
            if isinstance(event, ContentBlockDeltaEvent) and isinstance(event.delta, TextDelta):
                yield event.delta.text

    async def astream_text(
        self,
        messages: Sequence[MessageParam | Mapping[str, Any]],
        *,
        on_usage: UsageCallback | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async version of :meth:`stream_text`."""

        async for event in self.astream(messages, on_usage=on_usage, **kwargs):
            if isinstance(event, ContentBlockDeltaEvent) and isinstance(event.delta, TextDelta):
                yield event.delta.text

    def stream_collect(
        self,
        messages: Sequence[MessageParam | Mapping[str, Any]],
        *,
        on_usage: UsageCallback | None = None,
        **kwargs: Any,
    ) -> Message:
        """Consume a stream and return the assembled final message.

        Tool call argument fragments are parsed into ``ToolUseBlock.input``.
        """

        accumulator = MessageAccumulator()
        for event in self.stream(messages, on_usage=on_usage, **kwargs):
            accumulator.feed(event)
        if accumulator.message is None:
            raise APIStreamError("stream ended without a message_start event")
        return accumulator.finalize()

    async def astream_collect(
        self,
        messages: Sequence[MessageParam | Mapping[str, Any]],
        *,
        on_usage: UsageCallback | None = None,
        **kwargs: Any,
    ) -> Message:
        """Async version of :meth:`stream_collect`."""

        accumulator = MessageAccumulator()
        async for event in self.astream(messages, on_usage=on_usage, **kwargs):
            accumulator.feed(event)
        if accumulator.message is None:
            raise APIStreamError("stream ended without a message_start event")
        return accumulator.finalize()

    # -- embeddings ----------------------------------------------------
    @staticmethod
    def _input_count(params: EmbeddingCreateParams) -> int:
        return 1 if isinstance(params.input, str) else len(params.input)

    def _embedding_params(
        self,
        texts: str | Sequence[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: Literal["float", "base64"] | None = None,
        user: str | None = None,
    ) -> EmbeddingCreateParams:
        if isinstance(texts, str):
            if not texts.strip():
                raise ConfigurationError("embed() requires non-empty input text")
            normalized: str | list[str] = texts
        else:
            normalized = list(texts)
            if not normalized:
                raise ConfigurationError("embed() requires at least one input text")
            if any(not text.strip() for text in normalized):
                raise ConfigurationError("embed() input texts must not be empty")
        try:
            return EmbeddingCreateParams(
                model=model or self.model,
                input=normalized,
                dimensions=dimensions,
                encoding_format=encoding_format,
                user=user,
            )
        except ValidationError as exc:
            raise ConfigurationError(f"invalid embedding request: {exc}") from exc

    def embed(
        self,
        texts: str | Sequence[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: Literal["float", "base64"] | None = None,
        user: str | None = None,
        extra_body: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> EmbeddingResponse:
        """Create embeddings for one or more texts.

        Args:
            texts: A single string or a sequence of non-empty strings.
            model: Embedding model id. Defaults to the client's ``model``.
            dimensions: Optional output vector size, when the model supports it.
            encoding_format: ``"float"`` or ``"base64"``; providers default to float.
            user: Optional end-user identifier forwarded to the provider.
            extra_body: Provider-specific fields merged into the request body.
            headers: Extra headers for this call.

        Returns:
            An :class:`EmbeddingResponse` whose :attr:`~EmbeddingResponse.vectors`
            are ordered by input index.

        Raises:
            ConfigurationError: if the provider has no embeddings endpoint or the
                input is empty.
            APIStatusError: for non-success provider responses.
            APIConnectionError: when the provider cannot be reached.

        Example:
            >>> llm = BrainStormLLM(provider="ollama", model="nomic-embed-text")
            >>> response = llm.embed(["first document", "second document"])
            >>> len(response.vectors)
            2
        """

        params = self._embedding_params(
            texts,
            model=model,
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user,
        )
        log = library_logger(self.provider_name, params.model)
        log.debug("embed: texts={} dimensions={}", self._input_count(params), dimensions)
        return self._provider.embed(
            params,
            extra_body=extra_body,
            headers=dict(headers) if headers else None,
        )

    async def aembed(
        self,
        texts: str | Sequence[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: Literal["float", "base64"] | None = None,
        user: str | None = None,
        extra_body: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> EmbeddingResponse:
        """Async version of :meth:`embed` using the pooled async client."""

        params = self._embedding_params(
            texts,
            model=model,
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user,
        )
        log = library_logger(self.provider_name, params.model)
        log.debug("aembed: texts={} dimensions={}", self._input_count(params), dimensions)
        return await self._provider.aembed(
            params,
            extra_body=extra_body,
            headers=dict(headers) if headers else None,
        )

    # -- catalog / pricing ---------------------------------------------
    def list_models(self, **kwargs: Any) -> list[ModelInfo]:
        """Return this provider's catalog; accepts ``force_refresh=True``."""

        return self._provider.list_models(**kwargs)

    def list_embedding_models(self, **kwargs: Any) -> list[ModelInfo]:
        """Return only catalog entries of kind ``"embedding"``.

        Accepts ``force_refresh=True``. Providers without an embeddings endpoint
        return an empty list.
        """

        return self._provider.embedding_models(**kwargs)

    def model_prices(self, pricing_type: PricingMode = "any") -> ModelPricing:
        """Return this provider's price table.

        ``pricing_type`` is ``"live"``, ``"cached"``, or ``"any"``.
        """

        return self._provider.model_prices(pricing_type)
