from __future__ import annotations

import re
import httpx
import orjson

from typing import Any
from loguru import logger

from dataclasses import dataclass
from selectolax.lexbor import LexborHTMLParser

from ..schemas.pricing import TokenPrice

PRICING_URL = "https://commandcode.ai/docs/resources/pricing-limits"
DEFAULT_BASE_URL = "https://api.commandcode.ai/provider/v1"
USER_AGENT = "brainstorm-llm/0.1"

MANUAL_IDS = {
    "Jev": "typesafe/jev",
    "Jev 1.14.0": "typesafe/jev-1.14.0",
    "Claude Opus 4.6": "claude-opus-4-6",
    "Claude Sonnet 4.5": "claude-sonnet-4-5",
}


@dataclass(frozen=True)
class ParsedPriceRow:
    """One model row parsed from the Command Code pricing page."""

    name: str
    context: str | None
    input: float | None
    output: float | None
    cache_read: float | None
    cache_write: float | None
    capabilities: tuple[str, ...] = ()


def _children(node) -> list[Any]:
    result: list[Any] = []
    child = node.child
    while child is not None:
        result.append(child)
        child = child.next
    return result


def _parse_price(text: str | None) -> float | None:
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if "free" in lowered:
        return 0.0
    match = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", cleaned)
    if match:
        return float(match.group(1))
    return None


def _model_name(cell) -> str:
    link = cell.css_first("a")
    if link is not None and link.text(strip=True):
        return link.text(strip=True)
    span = cell.css_first("span")
    if span is not None and span.text(strip=True):
        return span.text(strip=True)
    return cell.text(strip=True)


def parse_pricing_html(html: str) -> list[ParsedPriceRow]:
    """Parse the pricing page's ``role="row"`` grid into price rows."""

    tree = LexborHTMLParser(html)
    rows = tree.css('div[role="row"]')
    parsed: list[ParsedPriceRow] = []

    for row in rows:
        cells = _children(row)
        if len(cells) < 7:
            continue
        name = _model_name(cells[0])
        if not name or name == "Model":
            continue
        capabilities: tuple[str, ...] = ()
        button = cells[6].css_first("button")
        if button is not None:
            label = button.attributes.get("aria-label") or ""
            if ":" in label:
                capabilities = tuple(
                    part.strip() for part in label.split(":", 1)[1].split(",") if part.strip()
                )
        parsed.append(
            ParsedPriceRow(
                name=name,
                context=cells[1].text(strip=True) or None,
                input=_parse_price(cells[2].text(strip=True)),
                output=_parse_price(cells[3].text(strip=True)),
                cache_read=_parse_price(cells[4].text(strip=True)),
                cache_write=_parse_price(cells[5].text(strip=True)),
                capabilities=capabilities,
            )
        )
    return parsed


def _fetch_model_names(base_url: str, timeout: float) -> dict[str, str]:
    url = f"{base_url.rstrip('/')}/models"
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        payload = orjson.loads(response.content)
    except Exception as exc:  # noqa: BLE001 - pricing can still use manual ids
        logger.warning("could not fetch Command Code model catalog for pricing: {}", exc)
        return {}

    mapping: dict[str, str] = {}
    for item in payload.get("data") or []:
        if isinstance(item, dict) and item.get("id") and item.get("name"):
            mapping[str(item["name"])] = str(item["id"])
    return mapping


def _is_free(row: ParsedPriceRow) -> bool:
    return all((value or 0) == 0 for value in (row.input, row.output, row.cache_read, row.cache_write))


def fetch_commandcode_prices(
    *,
    timeout: float | None = None,
    base_url: str | None = None,
) -> tuple[dict[str, TokenPrice], str]:
    """Fetch live Command Code prices and map display names to model IDs."""

    timeout = timeout or 60.0
    headers = {"User-Agent": USER_AGENT}
    response = httpx.get(PRICING_URL, timeout=timeout, follow_redirects=True, headers=headers)
    response.raise_for_status()

    rows = parse_pricing_html(response.text)
    if not rows:
        from ..errors import PricingParseError

        raise PricingParseError("Command Code pricing page contained no parseable rows")

    mapping = _fetch_model_names(base_url or DEFAULT_BASE_URL, timeout)
    mapping.update(MANUAL_IDS)

    selected: dict[str, ParsedPriceRow] = {}
    for row in rows:
        model_id = mapping.get(row.name)
        if model_id is None:
            continue
        existing = selected.get(model_id)
        if existing is None or (_is_free(existing) and not _is_free(row)):
            selected[model_id] = row

    prices = {
        model_id: TokenPrice(
            input=row.input or 0.0,
            output=row.output or 0.0,
            cache_read=row.cache_read,
            cache_write=row.cache_write,
        )
        for model_id, row in selected.items()
    }
    if not prices:
        from ..errors import PricingParseError

        raise PricingParseError("no Command Code prices could be mapped to model ids")
    return prices, PRICING_URL
