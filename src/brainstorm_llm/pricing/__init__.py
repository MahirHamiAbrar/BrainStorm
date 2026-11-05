from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable

from loguru import logger

from ..errors import BrainStormError, PricingUnavailableError
from ..schemas.messages import Usage
from ..schemas.pricing import ModelPricing, PricingMode, PricingSource, TokenPrice
from .cache import PricingCache, default_cache_dir, pricing_file
from .commandcode import fetch_commandcode_prices
from .gemini import fetch_gemini_prices
from .ollama import fetch_ollama_prices
from .opencode import fetch_opencode_prices
from .openrouter import fetch_openrouter_prices

PricingSourceFn = Callable[..., tuple[dict[str, TokenPrice], str]]

SOURCES: dict[str, PricingSourceFn] = {
    "commandcode": fetch_commandcode_prices,
    "gemini": fetch_gemini_prices,
    "openrouter": fetch_openrouter_prices,
    "opencode": fetch_opencode_prices,
    "ollama": fetch_ollama_prices,
}

DEFAULT_CACHE_TTL = timedelta(hours=24)


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = datetime.now(timezone.utc)
    else:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _entry_to_model(provider: str, entry: dict, pricing_type: PricingSource) -> ModelPricing:
    raw_prices = entry.get("prices") or {}
    prices: dict[str, TokenPrice] = {}
    for model_id, value in raw_prices.items():
        try:
            prices[str(model_id)] = TokenPrice.model_validate(value)
        except Exception as exc:  # noqa: BLE001 - one bad row must not break the cache
            logger.warning("ignoring invalid cached price for {}: {}", model_id, exc)

    if not prices:
        raise PricingUnavailableError(f"cached pricing for '{provider}' is empty")

    return ModelPricing(
        provider=provider,
        prices=prices,
        pricing_type=pricing_type,
        fetched_at=_parse_datetime(entry.get("fetched_at")),
        source=str(entry.get("source") or ""),
    )


def _is_fresh(entry: dict, ttl: timedelta | None) -> bool:
    if ttl is None:
        return False
    fetched_at = _parse_datetime(entry.get("fetched_at"))
    return datetime.now(timezone.utc) - fetched_at <= ttl


def _live_prices(
    provider: str,
    *,
    cache: PricingCache,
    timeout: float | None,
    base_url: str | None,
) -> ModelPricing:
    source_fn = SOURCES.get(provider)
    if source_fn is None:
        raise PricingUnavailableError(f"Provider '{provider}' does not expose pricing data")

    try:
        prices, source = source_fn(timeout=timeout, base_url=base_url)
    except BrainStormError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize network/parse failures
        raise PricingUnavailableError(f"could not fetch live pricing for '{provider}': {exc}") from exc

    fetched_at = datetime.now(timezone.utc)
    cache.put(provider, prices, source=source, fetched_at=fetched_at)
    return ModelPricing(
        provider=provider,
        prices=prices,
        pricing_type="live",
        fetched_at=fetched_at,
        source=source,
    )


def model_prices(
    provider: str = "commandcode",
    pricing_type: PricingMode = "any",
    *,
    cache_dir: Path | None = None,
    cache_ttl: timedelta | None = DEFAULT_CACHE_TTL,
    timeout: float | None = None,
    base_url: str | None = None,
) -> ModelPricing:
    """Return per-model prices in USD per one million tokens.

    Args:
        provider: Provider id, e.g. ``"commandcode"`` or ``"openrouter"``.
        pricing_type: ``"live"`` fetches and caches, ``"cached"`` reads only,
            and ``"any"`` prefers a fresh cache but refreshes it when stale.
        cache_dir: Override the default ``~/.cache/brainstorm`` directory.
        cache_ttl: Maximum age accepted by ``"any"`` before refreshing.
        timeout: HTTP timeout for live sources.
        base_url: Optional catalog base URL for provider-specific lookups.

    Returns:
        A :class:`ModelPricing` whose ``pricing_type`` reports the source that
        actually served the values.

    Raises:
        PricingUnavailableError: if no cache exists and the live source fails.
    """

    if pricing_type not in {"live", "cached", "any"}:
        raise ValueError("pricing_type must be 'live', 'cached', or 'any'")

    cache = PricingCache(cache_dir)
    entry = cache.get(provider)

    if pricing_type == "cached":
        if entry is None:
            raise PricingUnavailableError(
                f"no cached pricing for '{provider}' at {cache.path}; run with pricing_type='live' first"
            )
        return _entry_to_model(provider, entry, "cached")

    if pricing_type == "live":
        return _live_prices(provider, cache=cache, timeout=timeout, base_url=base_url)

    if entry is not None and _is_fresh(entry, cache_ttl):
        return _entry_to_model(provider, entry, "cached")

    try:
        return _live_prices(provider, cache=cache, timeout=timeout, base_url=base_url)
    except BrainStormError as exc:
        if entry is None:
            raise
        logger.warning(
            "live pricing for '{}' failed ({}); using cached prices from {}",
            provider,
            exc,
            entry.get("fetched_at"),
        )
        return _entry_to_model(provider, entry, "cached")


def clear_pricing_cache(provider: str | None = None, *, cache_dir: Path | None = None) -> None:
    """Delete cached prices for one provider, or all providers when ``None``."""

    PricingCache(cache_dir).clear(provider)


def compute_cost(price: TokenPrice, usage: Usage) -> Decimal:
    """Estimate a request's cost in USD.

    Uses the per-million input, output, cache-read, and cache-write prices.
    Discounts, peak pricing, and context-tier overrides are not modeled.
    """

    million = Decimal(1_000_000)

    def value(raw: float | None) -> Decimal:
        """Convert an optional price to Decimal, treating ``None`` as zero."""

        return Decimal(str(raw or 0))

    return (
        Decimal(usage.input_tokens) * value(price.input) / million
        + Decimal(usage.output_tokens) * value(price.output) / million
        + Decimal(usage.cache_read_input_tokens or 0) * value(price.cache_read) / million
        + Decimal(usage.cache_creation_input_tokens or 0) * value(price.cache_write) / million
    )


__all__ = [
    "DEFAULT_CACHE_TTL",
    "SOURCES",
    "clear_pricing_cache",
    "compute_cost",
    "default_cache_dir",
    "model_prices",
    "pricing_file",
]
