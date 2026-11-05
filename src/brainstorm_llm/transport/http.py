from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx
import orjson

from .._version import __version__
from ..errors import APIConnectionError, APITimeoutError, error_from_response


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    backoff_base: float = 0.5
    backoff_max: float = 8.0
    jitter: bool = True
    retry_statuses: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 409, 429, 500, 502, 503, 504, 529})
    )

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(retry_after, self.backoff_max)
        delay = min(self.backoff_base * (2**attempt), self.backoff_max)
        if self.jitter:
            delay += random.uniform(0, delay * 0.25)
        return delay


@dataclass(frozen=True)
class TransportConfig:
    base_url: str
    api_key: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout: float = 60.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    user_agent: str = f"brainstorm-llm/{__version__}"

    def build_headers(self, extra: Mapping[str, str] | None = None, *, stream: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update({key: value for key, value in self.headers.items()})
        if extra:
            headers.update(extra)
        return headers

    def url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


class HttpTransport:
    def __init__(
        self,
        config: TransportConfig,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self._client = httpx.Client(
            timeout=httpx.Timeout(config.timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            follow_redirects=True,
            transport=transport,
        )

    @property
    def client(self) -> httpx.Client:
        return self._client

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        content = None
        request_headers = self.config.build_headers(headers, stream=stream)
        if json_body is not None:
            content = orjson.dumps(json_body)
            request_headers["Content-Type"] = "application/json"

        last_error: Exception | None = None
        for attempt in range(self.config.retry.max_retries + 1):
            request = self._client.build_request(
                method,
                self.config.url(path),
                content=content,
                headers=request_headers,
            )
            try:
                response = self._client.send(request, stream=stream)
            except httpx.TimeoutException as exc:
                last_error = APITimeoutError(f"Request timed out: {exc}")
                if attempt < self.config.retry.max_retries:
                    time.sleep(self.config.retry.delay(attempt))
                    continue
                raise last_error from exc
            except httpx.TransportError as exc:
                last_error = APIConnectionError(f"Connection error: {exc}")
                if attempt < self.config.retry.max_retries:
                    time.sleep(self.config.retry.delay(attempt))
                    continue
                raise last_error from exc

            if response.status_code in self.config.retry.retry_statuses and attempt < self.config.retry.max_retries:
                retry_after = _retry_after(response)
                response.close()
                time.sleep(self.config.retry.delay(attempt, retry_after))
                continue

            if response.status_code >= 400:
                body = response.read() if stream else response.content
                response.close()
                raise error_from_response(response.status_code, response.headers, body)

            return response

        raise last_error or APIConnectionError("Request failed after retries")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpTransport":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AsyncHttpTransport:
    def __init__(
        self,
        config: TransportConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            follow_redirects=True,
            transport=transport,
        )

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        import asyncio

        content = None
        request_headers = self.config.build_headers(headers, stream=stream)
        if json_body is not None:
            content = orjson.dumps(json_body)
            request_headers["Content-Type"] = "application/json"

        last_error: Exception | None = None
        for attempt in range(self.config.retry.max_retries + 1):
            request = self._client.build_request(
                method,
                self.config.url(path),
                content=content,
                headers=request_headers,
            )
            try:
                response = await self._client.send(request, stream=stream)
            except httpx.TimeoutException as exc:
                last_error = APITimeoutError(f"Request timed out: {exc}")
                if attempt < self.config.retry.max_retries:
                    await asyncio.sleep(self.config.retry.delay(attempt))
                    continue
                raise last_error from exc
            except httpx.TransportError as exc:
                last_error = APIConnectionError(f"Connection error: {exc}")
                if attempt < self.config.retry.max_retries:
                    await asyncio.sleep(self.config.retry.delay(attempt))
                    continue
                raise last_error from exc

            if response.status_code in self.config.retry.retry_statuses and attempt < self.config.retry.max_retries:
                retry_after = _retry_after(response)
                await response.aclose()
                await asyncio.sleep(self.config.retry.delay(attempt, retry_after))
                continue

            if response.status_code >= 400:
                body = await response.aread() if stream else response.content
                await response.aclose()
                raise error_from_response(response.status_code, response.headers, body)

            return response

        raise last_error or APIConnectionError("Request failed after retries")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncHttpTransport":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None
