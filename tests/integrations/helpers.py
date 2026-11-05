from __future__ import annotations

from typing import Any, Callable

import httpx

from brainstorm_llm import BrainStormLLM
from brainstorm_llm.integrations.langchain import ChatBrainStorm
from brainstorm_llm.providers.commandcode import CommandCodeProvider
from tests.helpers import model_list

MODEL = "deepseek/deepseek-v4.1-flash"


def with_model_catalog(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[httpx.Request], httpx.Response]:
    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json=model_list(
                    {
                        "id": MODEL,
                        "name": "DeepSeek V4.1 Flash (latest)",
                        "supported_endpoints": ["/chat/completions"],
                    }
                ),
            )
        return handler(request)

    return wrapped


def make_chat_model(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    async_handler: Callable[[httpx.Request], Any] | None = None,
    **client_kwargs: Any,
) -> ChatBrainStorm:
    wrapped = with_model_catalog(handler)
    if async_handler is not None:
        async def wrapped_async(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(
                    200,
                    json=model_list(
                        {
                            "id": MODEL,
                            "name": "DeepSeek V4.1 Flash (latest)",
                            "supported_endpoints": ["/chat/completions"],
                        }
                    ),
                )
            return await async_handler(request)
    else:
        async def wrapped_async(request: httpx.Request) -> httpx.Response:
            return wrapped(request)

    provider = CommandCodeProvider(
        api_key="test-key",
        transport=httpx.MockTransport(wrapped),
        async_transport=httpx.MockTransport(wrapped_async),
        **client_kwargs,
    )
    llm = BrainStormLLM(model=MODEL, provider=provider, pricing="off")
    return ChatBrainStorm(client=llm)
