from __future__ import annotations

import base64
import binascii
import struct
from typing import Any, Literal, TypeAlias

from pydantic import Field, field_validator

from .base import BrainStormModel, StrictModel

EmbeddingInput: TypeAlias = str | list[str]


class EmbeddingUsage(BrainStormModel):
    """Token usage reported for one embeddings request.

    Args:
        prompt_tokens: Tokens consumed by the input texts.
        total_tokens: Total billable tokens; embeddings have no output tokens.
    """

    prompt_tokens: int = 0
    total_tokens: int = 0


class Embedding(BrainStormModel):
    """One vector in an embeddings response.

    ``embedding`` accepts a float list or an OpenAI-style base64 string, which
    is decoded as little-endian ``float32`` values.
    """

    object: Literal["embedding"] = "embedding"
    index: int = 0
    embedding: list[float]

    @field_validator("embedding", mode="before")
    @classmethod
    def _decode_base64(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("embedding is not valid base64") from exc
        if len(raw) % 4:
            raise ValueError("base64 embedding length must be a multiple of 4 bytes")
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))


class EmbeddingResponse(BrainStormModel):
    """An embeddings response normalized across providers.

    Args:
        object: Always ``"list"``, matching the OpenAI response shape.
        data: One :class:`Embedding` per input text.
        model: Model that produced the vectors.
        usage: Prompt and total token counts.
        provider: Provider id that served the request.
        request_id: Upstream request id, when supplied.
    """

    object: Literal["list"] = "list"
    data: list[Embedding] = Field(default_factory=list)
    model: str = ""
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)
    provider: str | None = None
    request_id: str | None = Field(default=None, exclude=True, repr=False)

    @property
    def vectors(self) -> list[list[float]]:
        """Vectors ordered by their response ``index``."""

        return [item.embedding for item in sorted(self.data, key=lambda item: item.index)]

    @property
    def dimensions(self) -> int | None:
        """Vector width, or ``None`` when the response is empty."""

        vectors = self.vectors
        return len(vectors[0]) if vectors else None


class EmbeddingCreateParams(StrictModel):
    """Provider-neutral request passed to a provider's embeddings endpoint.

    Args:
        model: Embedding model id.
        input: One string or a list of strings to embed.
        dimensions: Optional output vector size, when the model supports truncation.
        encoding_format: ``"float"`` or ``"base64"``; providers default to float.
        user: Optional end-user identifier forwarded to the provider.
    """

    model: str
    input: EmbeddingInput
    dimensions: int | None = Field(default=None, gt=0)
    encoding_format: Literal["float", "base64"] | None = None
    user: str | None = None

    @field_validator("input")
    @classmethod
    def _validate_input(cls, value: EmbeddingInput) -> EmbeddingInput:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("input must not be empty")
            return value
        if not value:
            raise ValueError("input must contain at least one text")
        if any(not item.strip() for item in value):
            raise ValueError("input texts must not be empty")
        return value


__all__ = [
    "Embedding",
    "EmbeddingCreateParams",
    "EmbeddingInput",
    "EmbeddingResponse",
    "EmbeddingUsage",
]
