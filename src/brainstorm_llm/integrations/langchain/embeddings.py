"""LangChain embeddings backed by :class:`brainstorm_llm.BrainStormLLM`."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from langchain_core.embeddings import Embeddings

from ...client import BrainStormLLM
from ...errors import BrainStormError, ConfigurationError
from ._errors import raise_langchain_error


class BrainStormEmbeddings(Embeddings):
    """A LangChain ``Embeddings`` implementation over BrainStorm's client.

    Works with providers that expose an embeddings endpoint (Ollama and Gemini).
    Synchronous calls use the client's pooled sync transport; the async methods
    use the pooled async transport rather than a thread pool.

    Args:
        model: Embedding model id. Required unless ``client`` is provided.
        provider: Provider id used when no ``client`` is provided.
        api_key: Overrides the provider's environment variable.
        base_url: Overrides the provider's default endpoint.
        timeout: Request timeout in seconds.
        max_retries: Retries for transient provider failures.
        dimensions: Optional output vector size, when the model supports it.
        encoding_format: ``"float"`` or ``"base64"``; providers default to float.
        headers: Extra HTTP headers for every request.
        extra_body: Provider-specific request fields merged into the body.
        client: Optional preconfigured ``BrainStormLLM``. Its provider is always
            used, and its model is used when ``model`` is omitted. The adapter
            does not close an injected client.

    Example:
        >>> embeddings = BrainStormEmbeddings(
        ...     model="nomic-embed-text", provider="ollama"
        ... )
        >>> vectors = embeddings.embed_documents(["first", "second"])
        >>> embeddings.embed_query("first")
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: str = "commandcode",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        dimensions: int | None = None,
        encoding_format: Literal["float", "base64"] | None = None,
        headers: Mapping[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        client: BrainStormLLM | None = None,
    ) -> None:
        if client is not None:
            if not isinstance(client, BrainStormLLM):
                raise TypeError("client must be a BrainStormLLM instance")
            self._llm = client
            self._owns_client = False
        else:
            if not model:
                raise ConfigurationError(
                    "BrainStormEmbeddings requires model= such as "
                    "'nomic-embed-text' with provider='ollama'"
                )
            self._llm = BrainStormLLM(
                model=model,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                headers=headers,
            )
            self._owns_client = True

        self.model = model or self._llm.model
        self.provider = self._llm.provider_name
        self._dimensions = dimensions
        self._encoding_format: Literal["float", "base64"] | None = encoding_format
        self._headers = dict(headers) if headers else None
        self._extra_body = dict(extra_body) if extra_body else None

    def __repr__(self) -> str:
        return (
            f"BrainStormEmbeddings(model={self.model!r}, "
            f"provider={self.provider!r})"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents and return one vector per input text.

        Args:
            texts: Documents to embed. An empty list returns an empty list.

        Raises:
            langchain_core.exceptions.ModelError: for configuration failures.
            langchain_core.exceptions.ModelAPIError: for provider API failures.
        """

        if not texts:
            return []
        try:
            response = self._llm.embed(
                texts,
                model=self.model,
                dimensions=self._dimensions,
                encoding_format=self._encoding_format,
                headers=self._headers,
                extra_body=self._extra_body,
            )
        except BrainStormError as exc:
            raise_langchain_error(exc)
        return response.vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one query text and return its vector."""

        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async version of :meth:`embed_documents` using the async transport."""

        if not texts:
            return []
        try:
            response = await self._llm.aembed(
                texts,
                model=self.model,
                dimensions=self._dimensions,
                encoding_format=self._encoding_format,
                headers=self._headers,
                extra_body=self._extra_body,
            )
        except BrainStormError as exc:
            raise_langchain_error(exc)
        return response.vectors

    async def aembed_query(self, text: str) -> list[float]:
        """Async version of :meth:`embed_query`."""

        vectors = await self.aembed_documents([text])
        return vectors[0]

    def close(self) -> None:
        """Close the underlying client when this instance owns it."""

        if self._owns_client:
            self._llm.close()

    async def aclose(self) -> None:
        """Close the underlying async client when this instance owns it."""

        if self._owns_client:
            await self._llm.aclose()


__all__ = ["BrainStormEmbeddings"]
