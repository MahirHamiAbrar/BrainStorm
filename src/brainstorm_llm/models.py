from __future__ import annotations

import os

from .providers import get_provider
from .schemas.models import ModelInfo

PUBLIC_CATALOG_PROVIDERS = {"commandcode", "openrouter", "opencode"}
PROVIDER_KEY_ENVS = {
    "commandcode": ("COMMAND_API_KEY", "CMD_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "opencode": ("OPENCODE_API_KEY",),
    "ollama": ("OLLAMA_API_KEY",),
}


def list_models(
    provider: str = "commandcode",
    *,
    include_decision_models: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    force_refresh: bool = False,
) -> list[ModelInfo]:
    """List the models a provider exposes, without pricing data.

    Args:
        provider: Provider id, e.g. ``"commandcode"`` or ``"openrouter"``.
        include_decision_models: Append System One models to the Command Code
            catalog.
        api_key: Optional key; public catalogs can be browsed without one.
        base_url: Optional provider base URL override.
        timeout: HTTP timeout for the catalog request.
        force_refresh: Bypass the in-process one-hour catalog cache.

    Returns:
        Catalog metadata for each model. Use :func:`model_prices` for prices.
    """

    kwargs: dict[str, object] = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout or 60.0,
    }
    has_env_key = any(
        os.getenv(name) for name in PROVIDER_KEY_ENVS.get(provider, ())
    )
    if provider in PUBLIC_CATALOG_PROVIDERS and not api_key and not has_env_key:
        # These catalog endpoints are public; avoid requiring a key just to browse.
        kwargs["api_key"] = "catalog-only"

    client = get_provider(provider, **kwargs)
    try:
        if provider == "commandcode":
            from typing import Any, cast

            return cast(Any, client).list_models(
                force_refresh=force_refresh,
                include_decision_models=include_decision_models,
            )
        return client.list_models(force_refresh=force_refresh)
    finally:
        client.close()
