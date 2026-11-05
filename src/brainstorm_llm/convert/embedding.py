from __future__ import annotations

from typing import Any

from ..schemas.embeddings import EmbeddingCreateParams, EmbeddingResponse


def embedding_payload(
    params: EmbeddingCreateParams,
    *,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert :class:`EmbeddingCreateParams` to an OpenAI embeddings payload.

    ``extra_body`` is merged last so provider-specific fields can override the
    defaults.
    """

    payload: dict[str, Any] = {"model": params.model, "input": params.input}
    if params.dimensions is not None:
        payload["dimensions"] = params.dimensions
    if params.encoding_format is not None:
        payload["encoding_format"] = params.encoding_format
    if params.user is not None:
        payload["user"] = params.user
    if extra_body:
        payload.update(extra_body)
    return payload


def openai_to_embedding_response(
    payload: dict[str, Any],
    *,
    request_model: str,
    provider: str | None = None,
    request_id: str | None = None,
) -> EmbeddingResponse:
    """Normalize an OpenAI-style embeddings response.

    Args:
        payload: Decoded JSON response body.
        request_model: Model id sent in the request, used when the response omits one.
        provider: Provider id to attach to the normalized response.
        request_id: Upstream request id to attach.
    """

    return EmbeddingResponse.model_validate(
        {
            "object": payload.get("object") or "list",
            "data": payload.get("data") or [],
            "model": payload.get("model") or request_model,
            "usage": payload.get("usage") or {},
            "provider": provider,
            "request_id": request_id,
        }
    )


__all__ = ["embedding_payload", "openai_to_embedding_response"]
