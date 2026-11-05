from __future__ import annotations

import httpx
import orjson

from ..schemas.pricing import TokenPrice

DEFAULT_BASE_URL = "http://localhost:11434/v1"
USER_AGENT = "brainstorm-llm/0.1"


def parse_models_payload(payload: dict, *, cloud: bool = False) -> dict[str, TokenPrice]:
    note = "cloud pricing is not published via the models API" if cloud else "local inference"
    prices: dict[str, TokenPrice] = {}
    for item in payload.get("data") or []:
        if isinstance(item, dict) and item.get("id"):
            prices[str(item["id"])] = TokenPrice(
                input=0.0,
                output=0.0,
                cache_read=0.0,
                cache_write=0.0,
                notes=note,
            )
    return prices


def fetch_ollama_prices(
    *,
    timeout: float | None = None,
    base_url: str | None = None,
) -> tuple[dict[str, TokenPrice], str]:
    url = f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/models"
    response = httpx.get(
        url,
        timeout=timeout or 60.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    cloud = "ollama.com" in url
    return parse_models_payload(orjson.loads(response.content), cloud=cloud), url
