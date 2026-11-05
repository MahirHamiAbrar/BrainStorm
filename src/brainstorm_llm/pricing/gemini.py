from __future__ import annotations

import re
import httpx

from loguru import logger

from ..errors import PricingParseError
from ..schemas.pricing import TokenPrice

PRICING_URL = "https://aistudio.google.com/docs/pricing.md"
USER_AGENT = "brainstorm-llm/0.1"

_HEADING = re.compile(r"^##\s+(.+?)\s*\{%\s*id=\"([^\"]+)\"\s*%\}", re.MULTILINE)
_STANDARD_TAB = re.compile(r"\{%\s*tab title=\"Standard\"\s*%\}(.*?)\{%\s*/tab\s*%\}", re.DOTALL)
_PRICE = re.compile(r"\$([0-9]+(?:\.[0-9]+)?)")


def _row_cells(body: str, label_prefix: str) -> list[str]:
    lines = body.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("*"):
            continue
        label = stripped[1:].strip().lower()
        if not label.startswith(label_prefix.lower()):
            continue

        cells: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            cell_line = lines[cursor].strip()
            if cell_line.startswith("* "):
                cells.append(cell_line[2:].strip())
            elif cell_line.startswith("---"):
                break
            elif cell_line and cells:
                cells[-1] = f"{cells[-1]} {cell_line}".strip()
            cursor += 1
        return cells[:2]
    return []


def _paid_value(cells: list[str]) -> tuple[float | None, str | None]:
    if len(cells) < 2:
        return None, None
    paid = cells[1]
    lowered = paid.lower()
    if "not available" in lowered and "$" not in paid:
        return None, None
    if "free" in lowered and "$" not in paid:
        return 0.0, None
    match = _PRICE.search(paid)
    if match is None:
        return None, None
    notes = paid if paid.count("$") > 1 or "starting" in lowered else None
    return float(match.group(1)), notes


def parse_pricing_markdown(markdown: str) -> dict[str, TokenPrice]:
    headings = list(_HEADING.finditer(markdown))
    prices: dict[str, TokenPrice] = {}

    for index, heading in enumerate(headings):
        model_id = heading.group(2)
        section_end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        section = markdown[heading.end() : section_end]

        tab = _STANDARD_TAB.search(section)
        body = tab.group(1) if tab else section

        input_cells = _row_cells(body, "Input price")
        output_cells = _row_cells(body, "Output price")
        cache_cells = _row_cells(body, "Context caching price")

        input_price, input_note = _paid_value(input_cells)
        output_price, output_note = _paid_value(output_cells)
        cache_price, cache_note = _paid_value(cache_cells)

        if input_price is None or output_price is None:
            continue

        notes = "; ".join(note for note in (input_note, output_note, cache_note) if note) or None
        prices[model_id] = TokenPrice(
            input=input_price,
            output=output_price,
            cache_read=cache_price,
            notes=notes,
        )

    return prices


def fetch_gemini_prices(
    *,
    timeout: float | None = None,
    base_url: str | None = None,
) -> tuple[dict[str, TokenPrice], str]:
    response = httpx.get(
        PRICING_URL,
        timeout=timeout or 60.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    prices = parse_pricing_markdown(response.text)
    if not prices:
        logger.warning("Gemini pricing page produced no parseable prices")
        raise PricingParseError("no Gemini prices could be parsed")
    return prices, PRICING_URL
