"""Translate BrainStorm transport errors into LangChain's standard model errors."""

from __future__ import annotations

from typing import NoReturn

from langchain_core import exceptions as lc

from ...errors import (
    APIConnectionError,
    APIStatusError,
    APIStreamError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    BrainStormError,
    ConfigurationError,
    MissingAPIKeyError,
    PermissionDeniedError,
    RateLimitError,
    RequestTooLargeError,
    ResponseValidationError,
    ServerError,
    UnsupportedModelError,
    ZDRUnavailableError,
)


def map_error(exc: BaseException) -> BaseException:
    """Return the LangChain exception that best represents ``exc``."""

    if isinstance(exc, MissingAPIKeyError):
        return lc.ModelAuthenticationError(str(exc))
    if isinstance(exc, AuthenticationError):
        return lc.ModelAuthenticationError(str(exc))
    if isinstance(exc, PermissionDeniedError):
        return lc.ModelPermissionDeniedError(str(exc))
    if isinstance(exc, APITimeoutError):
        return lc.ModelTimeoutError(str(exc))
    if isinstance(exc, APIConnectionError):
        return lc.ModelConnectionError(str(exc))
    if isinstance(exc, RateLimitError):
        return lc.ModelRateLimitError(str(exc))
    if isinstance(exc, RequestTooLargeError):
        return lc.ContextOverflowError(str(exc))
    if isinstance(exc, ZDRUnavailableError):
        return lc.ModelInvalidRequestError(str(exc))
    if isinstance(exc, UnsupportedModelError):
        if getattr(exc, "code", None) == "unsupported_model":
            return lc.ModelNotFoundError(str(exc))
        return lc.ModelInvalidRequestError(str(exc))
    if isinstance(exc, BadRequestError):
        return lc.ModelInvalidRequestError(str(exc))
    if isinstance(exc, ServerError):
        return lc.ModelAPIError(str(exc))
    if isinstance(exc, APIStreamError):
        return lc.ModelAPIError(str(exc))
    if isinstance(exc, ResponseValidationError):
        return lc.ModelAPIError(str(exc))
    if isinstance(exc, APIStatusError):
        return lc.ModelAPIError(str(exc))
    if isinstance(exc, ConfigurationError):
        return lc.ModelError(str(exc))
    if isinstance(exc, BrainStormError):
        return lc.ModelError(str(exc))
    return exc


def raise_langchain_error(exc: BaseException) -> NoReturn:
    """Raise the mapped LangChain exception, preserving the original cause."""

    mapped = map_error(exc)
    if mapped is exc:
        raise exc
    raise mapped from exc
