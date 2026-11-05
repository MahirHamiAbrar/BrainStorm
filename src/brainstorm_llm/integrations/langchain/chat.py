"""LangChain chat model backed by :class:`brainstorm_llm.BrainStormLLM`."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr

from ...client import BrainStormLLM
from ...errors import BrainStormError
from ...schemas import (
    MessageParam,
    TextBlock,
    ThinkingConfig,
    Tool,
    ToolChoice,
    ToolUseBlock,
    Usage,
)
from ...schemas.streaming import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    InputJSONDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    RawMessageStreamEvent,
    TextDelta,
    ThinkingDelta,
)
from ...usage import CallUsage
from ._errors import raise_langchain_error
from ._messages import convert_messages, response_metadata, to_ai_message, usage_metadata
from ._tools import convert_tool, convert_tool_choice

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
_REASONING_BUDGETS = {
    "minimal": 1024,
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
    "max": 32768,
}


def _thinking_from_effort(effort: str | None) -> ThinkingConfig | None:
    if effort is None:
        return None
    normalized = effort.lower()
    if normalized in {"none", "disabled"}:
        return ThinkingConfig(type="disabled")
    return ThinkingConfig(type="enabled", budget_tokens=_REASONING_BUDGETS.get(normalized, 8192))


@dataclass
class _PreparedRequest:
    system: str | list[TextBlock] | None
    messages: list[MessageParam]
    max_tokens: int
    temperature: float | None
    top_p: float | None
    top_k: int | None
    stop_sequences: list[str] | None
    tools: list[Tool] | None
    tool_choice: ToolChoice | None
    thinking: ThinkingConfig | None
    extra_body: dict[str, Any] | None
    headers: dict[str, str] | None


class ChatBrainStorm(BaseChatModel):
    """A LangChain ``BaseChatModel`` over BrainStorm's Anthropic-native client.

    This adapter supports chat messages, tools, structured output, sync/async
    calls, streaming, token usage metadata, and LangGraph composition. It works
    with every provider exposed by :class:`BrainStormLLM`.

    Args:
        model: Model id. Defaults to ``deepseek/deepseek-v4.1-flash`` on Command
            Code.
        provider: Provider id or a preconfigured ``BrainStormLLM`` via ``client``.
        api_key: Overrides the provider's environment variable.
        base_url: Overrides the provider's default endpoint.
        timeout: Request timeout in seconds.
        max_retries: Retries for transient provider failures.
        max_tokens: Default output token cap.
        temperature: Default sampling temperature.
        top_p: Default nucleus sampling probability.
        top_k: Default top-k cutoff.
        stop_sequences: Default stop strings.
        reasoning_effort: Standard LangChain reasoning level; mapped to an
            Anthropic thinking budget or an OpenAI-wire ``reasoning_effort``.
        pricing: Cost lookup mode (``"off"``, ``"cached"``, ``"any"``, ``"live"``).
        zdr: Command Code zero-data-retention header. Defaults to the client's
            provider default.
        extra_body: Provider-specific request fields.
        headers: Extra HTTP headers for every request.
        client: Optional preconfigured ``BrainStormLLM``. Its model and provider
            take precedence.
        on_usage: Optional usage callback forwarded to the underlying client.

    Example:
        >>> model = ChatBrainStorm(model="deepseek/deepseek-v4.1-flash")
        >>> model.invoke("Say hello").text
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
    )

    model: str = DEFAULT_MODEL
    provider: str = "commandcode"
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout: float = 60.0
    max_retries: int = 2
    max_tokens: int = 4096
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    reasoning_effort: str | None = None
    thinking: ThinkingConfig | None = None
    pricing: Literal["off", "cached", "any", "live"] = "any"
    zdr: bool | None = None
    extra_body: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    client: Any = Field(default=None, exclude=True, repr=False)
    on_usage: Any = Field(default=None, exclude=True, repr=False)

    _llm: BrainStormLLM = PrivateAttr()
    _owns_client: bool = PrivateAttr(default=True)

    def model_post_init(self, __context: Any) -> None:
        if self.client is not None:
            if not isinstance(self.client, BrainStormLLM):
                raise TypeError("client must be a BrainStormLLM instance")
            self._llm = self.client
            self._owns_client = False
            self.model = self._llm.model
            self.provider = self._llm.provider_name
            if self.on_usage is not None and self._llm.on_usage is None:
                self._llm.on_usage = self.on_usage
        else:
            self._llm = BrainStormLLM(
                model=self.model,
                provider=self.provider,
                api_key=self.api_key.get_secret_value() if self.api_key else None,
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
                max_retries=self.max_retries,
                zdr=self.zdr,
                pricing=self.pricing,
                on_usage=self.on_usage,
            )
        if self.profile is None:
            self.profile = {
                "text_inputs": True,
                "text_outputs": True,
                "image_inputs": True,
                "image_url_inputs": True,
                "pdf_inputs": True,
                "audio_inputs": False,
                "video_inputs": False,
                "tool_calling": True,
                "tool_choice": True,
                "tool_call_streaming": True,
                "structured_output": True,
                "reasoning_output": True,
                "temperature": True,
            }

    @property
    def _llm_type(self) -> str:
        return "brainstorm-chat"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = self.provider
        params["ls_model_name"] = kwargs.get("model", self.model)
        return params

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind function tools and return a runnable chat model."""

        converted = [convert_tool(tool) for tool in tools]
        return cast(
            "Runnable[LanguageModelInput, AIMessage]",
            self.bind(tools=converted, tool_choice=tool_choice, **kwargs),
        )

    def close(self) -> None:
        """Close the underlying BrainStorm client when this instance owns it."""

        if self._owns_client:
            self._llm.close()

    async def aclose(self) -> None:
        """Close the underlying BrainStorm async client when owned."""

        if self._owns_client:
            await self._llm.aclose()

    def _prepare(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None,
        kwargs: dict[str, Any],
    ) -> _PreparedRequest:
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        parallel_tool_calls = kwargs.pop("parallel_tool_calls", None)
        kwargs.pop("ls_structured_output_format", None)
        kwargs.pop("structured_output_format", None)
        reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)
        thinking = kwargs.pop("thinking", self.thinking)
        if thinking is not None and isinstance(thinking, dict):
            thinking = ThinkingConfig.model_validate(thinking)
        if thinking is None:
            thinking = _thinking_from_effort(reasoning_effort)

        max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
        temperature = kwargs.pop("temperature", self.temperature)
        top_p = kwargs.pop("top_p", self.top_p)
        top_k = kwargs.pop("top_k", self.top_k)
        extra_body = {**(self.extra_body or {})}
        extra_body.update(kwargs.pop("extra_body", None) or {})
        headers = {**(self.headers or {})}
        headers.update(kwargs.pop("headers", None) or {})
        kwargs.pop("metadata", None)
        kwargs.pop("model", None)
        bound_stop = kwargs.pop("stop", None)
        if kwargs:
            logger.debug("Ignoring unsupported ChatBrainStorm kwargs: %s", sorted(kwargs))

        system, converted = convert_messages(messages)
        stops: list[str] = []
        for value in [
            *(self.stop_sequences or []),
            *(stop or []),
            *(bound_stop or []),
        ]:
            if value not in stops:
                stops.append(value)
        stop_sequences = stops or None
        converted_tools = (
            [convert_tool(tool) for tool in tools] if tools is not None else None
        )
        return _PreparedRequest(
            system=system,
            messages=converted,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            tools=converted_tools,
            tool_choice=convert_tool_choice(tool_choice, parallel_tool_calls),
            thinking=thinking,
            extra_body=extra_body or None,
            headers=headers or None,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prepared = self._prepare(messages, stop, kwargs)
        records: list[CallUsage] = []
        try:
            message = self._llm.generate(
                prepared.messages,
                system=prepared.system,
                max_tokens=prepared.max_tokens,
                temperature=prepared.temperature,
                top_p=prepared.top_p,
                top_k=prepared.top_k,
                stop_sequences=prepared.stop_sequences,
                tools=prepared.tools,
                tool_choice=prepared.tool_choice,
                thinking=prepared.thinking,
                extra_body=prepared.extra_body,
                headers=prepared.headers,
                on_usage=records.append,
            )
        except BrainStormError as exc:
            raise_langchain_error(exc)
        record = records[0] if records else None
        ai_message = to_ai_message(message, record)
        generation = ChatGeneration(
            message=ai_message,
            generation_info={
                "finish_reason": message.stop_reason,
                "request_id": message.request_id,
            },
        )
        return ChatResult(
            generations=[generation],
            llm_output={
                "model_name": self.model,
                "provider": self.provider,
                "request_id": message.request_id,
                "token_usage": {
                    "prompt_tokens": message.usage.input_tokens,
                    "completion_tokens": message.usage.output_tokens,
                    "total_tokens": message.usage.total_tokens,
                },
            },
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prepared = self._prepare(messages, stop, kwargs)
        records: list[CallUsage] = []
        try:
            message = await self._llm.agenerate(
                prepared.messages,
                system=prepared.system,
                max_tokens=prepared.max_tokens,
                temperature=prepared.temperature,
                top_p=prepared.top_p,
                top_k=prepared.top_k,
                stop_sequences=prepared.stop_sequences,
                tools=prepared.tools,
                tool_choice=prepared.tool_choice,
                thinking=prepared.thinking,
                extra_body=prepared.extra_body,
                headers=prepared.headers,
                on_usage=records.append,
            )
        except BrainStormError as exc:
            raise_langchain_error(exc)
        record = records[0] if records else None
        ai_message = to_ai_message(message, record)
        generation = ChatGeneration(
            message=ai_message,
            generation_info={
                "finish_reason": message.stop_reason,
                "request_id": message.request_id,
            },
        )
        return ChatResult(
            generations=[generation],
            llm_output={
                "model_name": self.model,
                "provider": self.provider,
                "request_id": message.request_id,
                "token_usage": {
                    "prompt_tokens": message.usage.input_tokens,
                    "completion_tokens": message.usage.output_tokens,
                    "total_tokens": message.usage.total_tokens,
                },
            },
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prepared = self._prepare(messages, stop, kwargs)
        records: list[CallUsage] = []
        translator = _ChatStreamTranslator(self.model, self.provider)
        try:
            for event in self._llm.stream(
                prepared.messages,
                system=prepared.system,
                max_tokens=prepared.max_tokens,
                temperature=prepared.temperature,
                top_p=prepared.top_p,
                top_k=prepared.top_k,
                stop_sequences=prepared.stop_sequences,
                tools=prepared.tools,
                tool_choice=prepared.tool_choice,
                thinking=prepared.thinking,
                extra_body=prepared.extra_body,
                headers=prepared.headers,
                on_usage=records.append,
            ):
                for chunk, token in translator.process(event):
                    if run_manager is not None:
                        run_manager.on_llm_new_token(token, chunk=chunk)
                    yield chunk
            final = translator.finalize(records[0] if records else None)
            if run_manager is not None:
                run_manager.on_llm_new_token("", chunk=final)
            yield final
        except BrainStormError as exc:
            raise_langchain_error(exc)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        prepared = self._prepare(messages, stop, kwargs)
        records: list[CallUsage] = []
        translator = _ChatStreamTranslator(self.model, self.provider)
        try:
            async for event in self._llm.astream(
                prepared.messages,
                system=prepared.system,
                max_tokens=prepared.max_tokens,
                temperature=prepared.temperature,
                top_p=prepared.top_p,
                top_k=prepared.top_k,
                stop_sequences=prepared.stop_sequences,
                tools=prepared.tools,
                tool_choice=prepared.tool_choice,
                thinking=prepared.thinking,
                extra_body=prepared.extra_body,
                headers=prepared.headers,
                on_usage=records.append,
            ):
                for chunk, token in translator.process(event):
                    if run_manager is not None:
                        await run_manager.on_llm_new_token(token, chunk=chunk)
                    yield chunk
            final = translator.finalize(records[0] if records else None)
            if run_manager is not None:
                await run_manager.on_llm_new_token("", chunk=final)
            yield final
        except BrainStormError as exc:
            raise_langchain_error(exc)


class _ChatStreamTranslator:
    """Translate BrainStorm stream events into LangChain chat chunks."""

    def __init__(self, model: str, provider: str) -> None:
        self.model = model
        self.provider = provider
        self.message_id: str | None = None
        self.request_id: str | None = None
        self.stop_reason: str | None = None
        self.usage = Usage()
        self._tool_indices: dict[int, int] = {}
        self._next_tool_index = 0

    def _chunk(
        self,
        *,
        content: Any = "",
        tool_call_chunks: list[dict[str, Any]] | None = None,
        usage: Usage | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_position: Literal["last"] | None = None,
    ) -> ChatGenerationChunk:
        message = AIMessageChunk(
            id=self.message_id,
            content=cast("Any", content),
            tool_call_chunks=cast("Any", tool_call_chunks or []),
            usage_metadata=cast("Any", usage_metadata(usage) if usage is not None else None),
            response_metadata=metadata or {},
            chunk_position=chunk_position,
        )
        return ChatGenerationChunk(message=message)

    def process(self, event: RawMessageStreamEvent) -> list[tuple[ChatGenerationChunk, str]]:
        if isinstance(event, MessageStartEvent):
            self.message_id = event.message.id
            self.request_id = event.message.request_id
            self.model = event.message.model or self.model
            return []

        if isinstance(event, ContentBlockStartEvent):
            block = event.content_block
            if isinstance(block, ToolUseBlock):
                tool_index = self._next_tool_index
                self._next_tool_index += 1
                self._tool_indices[event.index] = tool_index
                chunk = self._chunk(
                    tool_call_chunks=[
                        {
                            "name": block.name,
                            "args": "",
                            "id": block.id,
                            "index": tool_index,
                            "type": "tool_call_chunk",
                        }
                    ]
                )
                return [(chunk, "")]
            return []

        if isinstance(event, ContentBlockDeltaEvent):
            delta = event.delta
            if isinstance(delta, TextDelta):
                return [(self._chunk(content=delta.text), delta.text)]
            if isinstance(delta, ThinkingDelta):
                return [
                    (
                        self._chunk(
                            content=[{"type": "reasoning", "reasoning": delta.thinking}]
                        ),
                        "",
                    )
                ]
            if isinstance(delta, InputJSONDelta):
                tool_index = self._tool_indices.get(event.index)
                if tool_index is None:
                    return []
                chunk = self._chunk(
                    tool_call_chunks=[
                        {
                            "name": None,
                            "args": delta.partial_json,
                            "id": None,
                            "index": tool_index,
                            "type": "tool_call_chunk",
                        }
                    ]
                )
                return [(chunk, "")]
            return []

        if isinstance(event, MessageDeltaEvent):
            self.stop_reason = event.delta.stop_reason
            merged = self.usage.model_dump()
            merged.update(event.usage.model_dump(exclude_none=True, exclude_unset=True))
            self.usage = Usage.model_validate(merged)
            return []

        if isinstance(event, MessageStopEvent):
            return []

        return []

    def finalize(self, record: CallUsage | None) -> ChatGenerationChunk:
        usage = record.usage if record is not None else self.usage
        metadata: dict[str, Any] = {
            "model_name": self.model,
            "provider": self.provider,
            "stop_reason": self.stop_reason,
            "request_id": self.request_id,
        }
        if record is not None:
            metadata["cost_usd"] = (
                str(record.cost_usd) if record.cost_usd is not None else None
            )
            metadata["pricing_type"] = record.pricing_type
            metadata["latency_ms"] = record.latency_ms
        return self._chunk(
            usage=usage,
            metadata=metadata,
            chunk_position="last",
        )


__all__ = ["ChatBrainStorm"]
