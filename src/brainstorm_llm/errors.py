from __future__ import annotations

from typing import Any, Mapping

import orjson


class BrainStormError(Exception):
    """Base error for the BrainStorm LLM package."""


class ConfigurationError(BrainStormError):
    """The client or provider was configured incorrectly."""


class MissingAPIKeyError(ConfigurationError):
    """No API key was supplied or found in the provider's environment variables."""

    def __init__(self, provider: str, env_vars: list[str]) -> None:
        names = " or ".join(env_vars)
        super().__init__(f"No API key for provider '{provider}'. Set {names} or pass api_key=.")
        self.provider = provider
        self.env_vars = env_vars


class APIError(BrainStormError):
    """Base class for provider transport and API failures."""


class APIConnectionError(APIError):
    """The provider could not be reached."""


class APITimeoutError(APIConnectionError):
    """The provider did not respond within the configured timeout."""


class APIStatusError(APIError):
    """A non-success HTTP response, normalized across OpenAI and Anthropic envelopes.

    Attributes:
        status_code: HTTP status from the provider.
        error_type: Provider error type, e.g. ``invalid_request_error``.
        code: Optional provider error code.
        param: Optional parameter associated with the error.
        request_id: Upstream request or trace id, when supplied.
        body: Parsed error body, when available.
        retry_after: Parsed ``Retry-After`` value in seconds.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str = "api_error",
        code: str | None = None,
        param: str | None = None,
        request_id: str | None = None,
        body: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.param = param
        self.request_id = request_id
        self.body = body
        self.retry_after = retry_after

    def __str__(self) -> str:
        suffix = f" (request_id={self.request_id})" if self.request_id else ""
        return f"[{self.status_code} {self.error_type}] {self.message}{suffix}"


class AuthenticationError(APIStatusError):
    """401: the API key is missing, invalid, or expired."""


class PermissionDeniedError(APIStatusError):
    """403: the key is valid but the plan or region cannot use the resource."""


class BadRequestError(APIStatusError):
    """400: the request body, model, or endpoint combination was rejected."""


class UnsupportedModelError(BadRequestError):
    """The model does not exist or does not support the chosen wire endpoint."""


class RequestTooLargeError(APIStatusError):
    """413: the request body exceeded the provider's size limit."""


class ZDRUnavailableError(APIStatusError):
    """422: zero data retention was requested but no ZDR-capable upstream exists."""


class RateLimitError(APIStatusError):
    """429: the provider asked the client to retry later."""


class ServerError(APIStatusError):
    """5xx/529: the provider or its upstream failed."""


class APIStreamError(APIError):
    """A stream failed or emitted an error event."""

    def __init__(self, message: str, *, error_type: str = "api_error") -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type


class ResponseValidationError(APIError):
    """A provider response did not match the expected schema."""


class PricingUnavailableError(BrainStormError):
    """No cached price exists and the live pricing source could not be used."""


class PricingParseError(BrainStormError):
    """A pricing page or API response was reachable but could not be parsed."""


def _header(headers: Mapping[str, str], *names: str) -> str | None:
    lowered = {key.lower(): value for key, value in headers.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value:
            return value
    return None


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Parse a ``Retry-After`` header in seconds; return ``None`` when absent."""

    raw = _header(headers, "retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _extract_error(body: Any) -> tuple[str, str, str | None, str | None]:
    if not isinstance(body, dict):
        return "api_error", str(body), None, None

    error = body.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("detail") or "Unknown error")
        error_type = str(error.get("type") or "api_error")
        code = error.get("code")
        param = error.get("param")
        return error_type, message, str(code) if code is not None else None, (
            str(param) if param is not None else None
        )

    if isinstance(error, str):
        return "api_error", error, None, None

    detail = body.get("detail")
    if isinstance(detail, str):
        return "api_error", detail, None, None

    message = body.get("message")
    if isinstance(message, str):
        return "api_error", message, None, None

    return "api_error", orjson.dumps(body).decode("utf-8"), None, None


def error_from_response(
    status_code: int,
    headers: Mapping[str, str],
    content: bytes | str | None,
) -> APIStatusError:
    """Build the most specific :class:`APIStatusError` for an HTTP response.

    Handles both OpenAI and Anthropic error envelopes and attaches the upstream
    request id and ``Retry-After`` when present.
    """

    body: dict[str, Any] | None = None
    if content:
        try:
            parsed = orjson.loads(content)
            body = parsed if isinstance(parsed, dict) else {"response": parsed}
        except orjson.JSONDecodeError:
            body = {"response": content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content}

    error_type, message, code, param = _extract_error(body)
    request_id = _header(
        headers,
        "x-request-id",
        "request-id",
        "x-trace-id",
        "x-typesafe-request-id",
    )
    retry_after = parse_retry_after(headers)

    common: dict[str, Any] = {
        "status_code": status_code,
        "error_type": error_type,
        "code": code,
        "param": param,
        "request_id": request_id,
        "body": body,
        "retry_after": retry_after,
    }

    lowered = f"{error_type} {code} {message}".lower()
    if status_code == 401:
        return AuthenticationError(message, **common)
    if status_code == 403:
        return PermissionDeniedError(message, **common)
    if status_code == 413:
        return RequestTooLargeError(message, **common)
    if status_code == 422 and ("zdr" in lowered or "zero data" in lowered):
        return ZDRUnavailableError(message, **common)
    if status_code == 429:
        return RateLimitError(message, **common)
    if status_code >= 500:
        return ServerError(message, **common)
    if status_code == 400:
        if code == "unsupported_model" or "must be called via" in lowered:
            return UnsupportedModelError(message, **common)
        return BadRequestError(message, **common)
    return APIStatusError(message, **common)
