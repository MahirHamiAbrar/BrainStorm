from __future__ import annotations

import httpx
import orjson

from ..errors import PricingParseError
from ..schemas.pricing import TokenPrice

MODELS_URL = "https://opencode.ai/inference/v1/models"
USER_AGENT = "brainstorm-llm/0.1"


def _price(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def parse_models_payload(payload: dict) -> dict[str, TokenPrice]:
    prices: dict[str, TokenPrice] = {}
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        cost = ((item.get("cost") or {}).get("text") or {})
        if not model_id or not cost:
            continue
        input_cost = cost.get("input") or {}
        output_cost = cost.get("output") or {}
        input_price = _price(input_cost.get("price"))
        output_price = _price(output_cost.get("price"))
        if input_price is None or output_price is None:
            continue
        cache = input_cost.get("cache") or {}
        notes = "context-tier overrides exist" if input_cost.get("overrides") else None
        prices[str(model_id)] = TokenPrice(
            input=input_price,
            output=output_price,
            cache_read=_price(cache.get("read")),
            cache_write=_price(cache.get("write")),
            notes=notes,
        )
    return prices


def fetch_opencode_prices(
    *,
    timeout: float | None = None,
    base_url: str | None = None,
) -> tuple[dict[str, TokenPrice], str]:
    response = httpx.get(
        MODELS_URL,
        timeout=timeout or 60.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    prices = parse_models_payload(orjson.loads(response.content))
    if not prices:
        raise PricingParseError("OpenCode model list contained no prices")
    return prices, MODELS_URL
