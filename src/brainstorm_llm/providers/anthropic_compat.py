from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Mapping

import httpx
import orjson
from pydantic import TypeAdapter

from ..errors import APIStreamError
from ..schemas.messages import Message, MessageCreateParams
from ..schemas.streaming import MessageStartEvent, RawMessageStreamEvent, StreamErrorEvent
from ..transport.sse import aiter_sse_events, iter_sse_events

_STREAM_EVENT_ADAPTER = TypeAdapter(RawMessageStreamEvent)


def _request_id(response) -> str | None:
    return (
        response.headers.get("x-request-id")
        or response.headers.get("request-id")
        or response.headers.get("x-trace-id")
        or response.headers.get("x-opencode-request-id")
    )


class AnthropicWireMixin:
    messages_path = "/messages"

    if TYPE_CHECKING:
        def post(
            self,
            path: str,
            payload: Mapping[str, Any],
            *,
            headers: Mapping[str, str] | None = None,
            stream: bool = False,
        ) -> httpx.Response: ...

        async def apost(
            self,
            path: str,
            payload: Mapping[str, Any],
            *,
            headers: Mapping[str, str] | None = None,
            stream: bool = False,
        ) -> httpx.Response: ...

        @staticmethod
        def json_body(response: httpx.Response) -> dict[str, Any]: ...

    def generate(
        self,
        params: MessageCreateParams,
        *,
        extra_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Message:
        payload = params.to_wire()
        payload.pop("stream", None)
        if extra_body:
            payload.update(extra_body)
        response = self.post(self.messages_path, payload, headers=headers)
        message = Message.model_validate(self.json_body(response))
        message.request_id = _request_id(response)
        return message

    async def agenerate(
        self,
        params: MessageCreateParams,
        *,
        extra_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Message:
        payload = params.to_wire()
        payload.pop("stream", None)
        if extra_body:
            payload.update(extra_body)
        response = await self.apost(self.messages_path, payload, headers=headers)
        message = Message.model_validate(self.json_body(response))
        message.request_id = _request_id(response)
        return message

    def stream(
        self,
        params: MessageCreateParams,
        *,
        extra_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Iterator[RawMessageStreamEvent]:
        payload = params.to_wire()
        payload["stream"] = True
        if extra_body:
            payload.update(extra_body)
        response = self.post(self.messages_path, payload, headers=headers, stream=True)
        try:
            for sse in iter_sse_events(response.iter_lines()):
                data = sse.data.strip()
                if not data or data == "[DONE]":
                    continue
                event = _STREAM_EVENT_ADAPTER.validate_python(orjson.loads(data))
                if isinstance(event, StreamErrorEvent):
                    yield event
                    raise APIStreamError(event.error.message, error_type=event.error.type)
                if isinstance(event, MessageStartEvent):
                    event.message.request_id = _request_id(response)
                yield event
        finally:
            response.close()

    async def astream(
        self,
        params: MessageCreateParams,
        *,
        extra_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RawMessageStreamEvent]:
        payload = params.to_wire()
        payload["stream"] = True
        if extra_body:
            payload.update(extra_body)
        response = await self.apost(self.messages_path, payload, headers=headers, stream=True)
        try:
            async for sse in aiter_sse_events(response.aiter_lines()):
                data = sse.data.strip()
                if not data or data == "[DONE]":
                    continue
                event = _STREAM_EVENT_ADAPTER.validate_python(orjson.loads(data))
                if isinstance(event, StreamErrorEvent):
                    yield event
                    raise APIStreamError(event.error.message, error_type=event.error.type)
                if isinstance(event, MessageStartEvent):
                    event.message.request_id = _request_id(response)
                yield event
        finally:
            await response.aclose()
