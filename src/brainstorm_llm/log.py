from __future__ import annotations

import sys
from typing import Any, Mapping

from loguru import logger

_DEBUG_ENABLED = False
_SECRET_HEADERS = {"authorization", "x-api-key", "api-key"}

# Library logs stay quiet until the host application opts in.
logger.disable("brainstorm_llm")


def get_logger(**context: Any):
    """Return a loguru logger bound to the supplied structured context."""

    return logger.bind(**context) if context else logger


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``headers`` with credential values replaced by ``***``."""

    return {
        key: ("***" if key.lower() in _SECRET_HEADERS else value)
        for key, value in headers.items()
    }


def enable_debug() -> None:
    """Enable structured debug logs for the whole package.

    Adds one stderr sink and enables the ``brainstorm_llm`` loguru namespace.
    Safe to call more than once.
    """
    global _DEBUG_ENABLED
    if _DEBUG_ENABLED:
        return
    logger.enable("brainstorm_llm")
    logger.add(
        sys.stderr,
        level="DEBUG",
        filter=lambda record: record["extra"].get("logger_name") == "brainstorm_llm",
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
            "{extra[provider]} | {extra[model]} | <level>{message}</level>"
        ),
    )
    _DEBUG_ENABLED = True


def library_logger(provider: str = "-", model: str = "-"):
    """Return the package logger bound to provider and model fields."""

    return logger.bind(logger_name="brainstorm_llm", provider=provider, model=model)
