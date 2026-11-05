from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Mapping

import httpx
import orjson

from ..convert.request import anthropic_to_openai_payload
from ..convert.response import openai_to_anthropic_message
from ..convert.stream import OpenAIStreamTranslator
from ..errors import APIStreamError
from ..log import library_logger
from ..schemas.messages import Message, MessageCreateParams
from ..schemas.streaming import RawMessageStreamEvent
from ..transport.sse import aiter_sse_events, iter_sse_events


def _request_id(response) -> str | None:
    return (
        response.headers.get("x-request-id")
        or response.headers.get("request-id")
        or response.headers.get("x-trace-id")
        or response.headers.get("x-opencode-request-id")
    )


class OpenAIWireMixin:
    chat_completions_path = "/chat/completions"

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
        payload = anthropic_to_openai_payload(params, stream=False, extra_body=extra_body)
        response = self.post(self.chat_completions_path, payload, headers=headers)
        message = openai_to_anthropic_message(self.json_body(response), request_model=params.model)
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
        payload = anthropic_to_openai_payload(params, stream=False, extra_body=extra_body)
        response = await self.apost(self.chat_completions_path, payload, headers=headers)
        message = openai_to_anthropic_message(self.json_body(response), request_model=params.model)
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
        payload = anthropic_to_openai_payload(params, stream=True, extra_body=extra_body)
        response = self.post(self.chat_completions_path, payload, headers=headers, stream=True)
        translator = OpenAIStreamTranslator(model=params.model, request_id=_request_id(response))
        try:
            for sse in iter_sse_events(response.iter_lines()):
                data = sse.data.strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                chunk = orjson.loads(data)
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    error = chunk["error"]
                    message = error.get("message") if isinstance(error, dict) else str(error)
                    raise APIStreamError(message or "upstream stream error")
                yield from translator.process_chunk(chunk)
            yield from translator.finalize()
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
        payload = anthropic_to_openai_payload(params, stream=True, extra_body=extra_body)
        response = await self.apost(self.chat_completions_path, payload, headers=headers, stream=True)
        translator = OpenAIStreamTranslator(model=params.model, request_id=_request_id(response))
        try:
            async for sse in aiter_sse_events(response.aiter_lines()):
                data = sse.data.strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                chunk = orjson.loads(data)
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    error = chunk["error"]
                    message = error.get("message") if isinstance(error, dict) else str(error)
                    raise APIStreamError(message or "upstream stream error")
                for event in translator.process_chunk(chunk):
                    yield event
            for event in translator.finalize():
                yield event
        finally:
            await response.aclose()
