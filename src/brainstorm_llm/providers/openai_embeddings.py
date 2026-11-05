from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import httpx

from ..convert.embedding import embedding_payload, openai_to_embedding_response
from ..schemas.embeddings import EmbeddingCreateParams, EmbeddingResponse


def _request_id(response: httpx.Response) -> str | None:
    return (
        response.headers.get("x-request-id")
        or response.headers.get("request-id")
        or response.headers.get("x-trace-id")
    )


class OpenAIEmbeddingWireMixin:
    """Embed through an OpenAI-compatible ``/embeddings`` endpoint.

    Mix into a provider alongside :class:`BaseProvider` and set
    ``embeddings_path`` if the provider mounts the endpoint elsewhere.
    """

    embeddings_path = "/embeddings"
    supports_embeddings = True

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

    def embed(
        self,
        params: EmbeddingCreateParams,
        *,
        extra_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Request embeddings and normalize the response."""

        payload = embedding_payload(params, extra_body=extra_body)
        response = self.post(self.embeddings_path, payload, headers=headers)
        return openai_to_embedding_response(
            self.json_body(response),
            request_model=params.model,
            provider=getattr(self, "name", None),
            request_id=_request_id(response),
        )

    async def aembed(
        self,
        params: EmbeddingCreateParams,
        *,
        extra_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Async version of :meth:`embed`."""

        payload = embedding_payload(params, extra_body=extra_body)
        response = await self.apost(self.embeddings_path, payload, headers=headers)
        return openai_to_embedding_response(
            self.json_body(response),
            request_model=params.model,
            provider=getattr(self, "name", None),
            request_id=_request_id(response),
        )


__all__ = ["OpenAIEmbeddingWireMixin"]
