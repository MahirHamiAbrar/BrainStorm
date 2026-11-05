from __future__ import annotations

import httpx
import orjson

from ..errors import PricingParseError
from ..schemas.pricing import TokenPrice

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
USER_AGENT = "brainstorm-llm/0.1"


def _per_million(value: object) -> float | None:
    if value is None:
        return None
    try:
        price = float(str(value)) * 1_000_000
    except (TypeError, ValueError):
        return None
    return price if price >= 0 else None


def parse_models_payload(payload: dict) -> dict[str, TokenPrice]:
    """Convert OpenRouter's per-token prices to per-million USD prices."""

    prices: dict[str, TokenPrice] = {}
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        pricing = item.get("pricing") or {}
        if not model_id or not isinstance(pricing, dict):
            continue
        input_price = _per_million(pricing.get("prompt"))
        output_price = _per_million(pricing.get("completion"))
        if input_price is None or output_price is None:
            continue
        notes = "context-tier overrides exist" if pricing.get("overrides") else None
        prices[str(model_id)] = TokenPrice(
            input=input_price,
            output=output_price,
            cache_read=_per_million(pricing.get("input_cache_read")),
            cache_write=_per_million(pricing.get("input_cache_write")),
            notes=notes,
        )
    return prices


def fetch_openrouter_prices(
    *,
    timeout: float | None = None,
    base_url: str | None = None,
) -> tuple[dict[str, TokenPrice], str]:
    """Fetch OpenRouter's model catalog and extract its embedded prices."""

    url = f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/models"
    response = httpx.get(
        url,
        timeout=timeout or 60.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    prices = parse_models_payload(orjson.loads(response.content))
    if not prices:
        raise PricingParseError("OpenRouter model list contained no prices")
    return prices, url
