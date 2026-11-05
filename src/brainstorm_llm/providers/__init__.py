from __future__ import annotations

from typing import Any

from ..errors import ConfigurationError
from .base import BaseProvider
from .commandcode import CommandCodeProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .opencode import OpenCodeProvider
from .openrouter import OpenRouterProvider
from .typesafe import TypeSafeProvider

PROVIDERS: dict[str, type[BaseProvider]] = {
    "commandcode": CommandCodeProvider,
    "openrouter": OpenRouterProvider,
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "opencode": OpenCodeProvider,
    "typesafe": TypeSafeProvider,
}


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    try:
        provider_class = PROVIDERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROVIDERS))
        raise ConfigurationError(f"Unknown provider '{name}'. Available providers: {available}") from exc
    return provider_class(**kwargs)


__all__ = [
    "PROVIDERS",
    "BaseProvider",
    "CommandCodeProvider",
    "GeminiProvider",
    "OllamaProvider",
    "OpenCodeProvider",
    "OpenRouterProvider",
    "TypeSafeProvider",
    "get_provider",
]
