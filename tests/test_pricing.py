from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import orjson
import pytest

from brainstorm_llm import TokenPrice
from brainstorm_llm.errors import PricingParseError, PricingUnavailableError
from brainstorm_llm.pricing import (
    SOURCES,
    clear_pricing_cache,
    compute_cost,
    default_cache_dir,
    model_prices,
    pricing_file,
)
from brainstorm_llm.pricing.cache import PricingCache
from brainstorm_llm.pricing.commandcode import fetch_commandcode_prices, parse_pricing_html
from brainstorm_llm.pricing.gemini import parse_pricing_markdown
from brainstorm_llm.pricing.opencode import parse_models_payload as parse_opencode_payload
from brainstorm_llm.pricing.openrouter import parse_models_payload as parse_openrouter_payload
from brainstorm_llm.schemas import Usage

FIXTURES = Path(__file__).parent / "fixtures"


def test_commandcode_html_parser() -> None:
    rows = parse_pricing_html((FIXTURES / "commandcode_pricing.html").read_text())
    by_name = {row.name: row for row in rows if row.name != "Model"}
    assert by_name["DeepSeek V4.1 Flash (latest)"].input == 0.15
    assert by_name["DeepSeek V4.1 Flash (latest)"].output == 0.60
    assert by_name["DeepSeek V4.1 Flash (latest)"].cache_read == 0.003
    assert by_name["Glyph Cluster"].input == 0.0
    assert by_name["Jev"].input == 0.042
    assert "Vision" in by_name["DeepSeek V4.1 Flash (latest)"].capabilities


def test_commandcode_live_join_and_duplicate_selection(monkeypatch) -> None:
    html = (FIXTURES / "commandcode_pricing.html").read_text()
    models = {
        "object": "list",
        "data": [
            {"id": "deepseek/deepseek-v4.1-flash", "name": "DeepSeek V4.1 Flash (latest)"},
            {"id": "minimax/minimax-m3.1-flash", "name": "MiniMax M3.1 Flash"},
        ],
    }

    def fake_get(url, **kwargs):
        if url.endswith("/models"):
            return httpx_response(models)
        return httpx_response(text=html)

    monkeypatch.setattr("brainstorm_llm.pricing.commandcode.httpx.get", fake_get)
    prices, source = fetch_commandcode_prices(timeout=1)
    assert source.endswith("/pricing-limits")
    assert prices["deepseek/deepseek-v4.1-flash"].input == 0.15
    # duplicate free/paid rows prefer the paid row
    assert prices["minimax/minimax-m3.1-flash"].input == 0.15
    assert prices["typesafe/jev"].input == 0.042
    # Glyph Cluster has no live model id and is skipped
    assert not any("glyph" in key for key in prices)


def httpx_response(payload=None, *, text=None):
    import httpx

    request = httpx.Request("GET", "https://example.test")
    if payload is not None:
        return httpx.Response(200, json=payload, request=request)
    return httpx.Response(200, text=text, request=request)


def test_gemini_markdown_parser() -> None:
    prices = parse_pricing_markdown((FIXTURES / "gemini_pricing.md").read_text())
    assert prices["gemini-4-argon"].input == 2.0
    assert prices["gemini-4-argon"].output == 10.0
    assert prices["gemini-4-argon"].cache_read == 0.1
    assert "starting" in (prices["gemini-4-argon"].notes or "")
    assert prices["gemini-3.8-flash"].input == 0.75
    assert prices["gemini-3.8-flash"].output == 3.75
    assert prices["gemini-omni-1.2-flash"].input == 1.5
    assert prices["gemini-omni-1.2-flash"].output == 9.0


def test_openrouter_price_conversion() -> None:
    payload = {
        "data": [
            {
                "id": "deepseek/deepseek-v4.1-flash",
                "pricing": {
                    "prompt": "0.0000015",
                    "completion": "0.000006",
                    "input_cache_read": "0.00000003",
                },
            }
        ]
    }
    price = parse_openrouter_payload(payload)["deepseek/deepseek-v4.1-flash"]
    assert price.input == 1.5
    assert price.output == 6.0
    assert price.cache_read == 0.03


def test_opencode_price_mapping() -> None:
    payload = {
        "data": [
            {
                "id": "deepseek-v4.1-flash",
                "cost": {
                    "text": {
                        "input": {"price": 0.3, "cache": {"read": 0.006}},
                        "output": {"price": 1.2},
                    }
                },
            }
        ]
    }
    price = parse_opencode_payload(payload)["deepseek-v4.1-flash"]
    assert price.input == 0.3
    assert price.output == 1.2
    assert price.cache_read == 0.006


def test_pricing_modes_and_cache_writes(monkeypatch, tmp_path) -> None:
    calls = {"n": 0}

    def fake_source(*, timeout=None, base_url=None):
        calls["n"] += 1
        return {"model": TokenPrice(input=1.0, output=2.0)}, "fake-source"

    monkeypatch.setitem(SOURCES, "fake", fake_source)

    live = model_prices("fake", "live", cache_dir=tmp_path)
    assert live.pricing_type == "live"
    assert calls["n"] == 1

    cached = model_prices("fake", "cached", cache_dir=tmp_path)
    assert cached.pricing_type == "cached"
    assert cached.prices["model"].input == 1.0
    assert calls["n"] == 1

    fresh = model_prices("fake", "any", cache_dir=tmp_path)
    assert fresh.pricing_type == "cached"
    assert calls["n"] == 1


def test_any_refreshes_stale_cache(monkeypatch, tmp_path) -> None:
    calls = {"n": 0}

    def fake_source(*, timeout=None, base_url=None):
        calls["n"] += 1
        return {"model": TokenPrice(input=float(calls["n"]), output=0.0)}, "fake"

    monkeypatch.setitem(SOURCES, "fake", fake_source)
    model_prices("fake", "live", cache_dir=tmp_path)

    cache_file = pricing_file(tmp_path)
    data = orjson.loads(cache_file.read_bytes())
    data["providers"]["fake"]["fetched_at"] = "2020-01-01T00:00:00+00:00"
    cache_file.write_bytes(orjson.dumps(data))

    refreshed = model_prices("fake", "any", cache_dir=tmp_path, cache_ttl=timedelta(hours=1))
    assert refreshed.pricing_type == "live"
    assert refreshed.prices["model"].input == 2.0
    assert calls["n"] == 2


def test_any_falls_back_to_stale_cache(monkeypatch, tmp_path) -> None:
    def ok_source(*, timeout=None, base_url=None):
        return {"model": TokenPrice(input=1.0, output=0.0)}, "fake"

    monkeypatch.setitem(SOURCES, "fake", ok_source)
    model_prices("fake", "live", cache_dir=tmp_path)

    cache_file = pricing_file(tmp_path)
    data = orjson.loads(cache_file.read_bytes())
    data["providers"]["fake"]["fetched_at"] = "2020-01-01T00:00:00+00:00"
    cache_file.write_bytes(orjson.dumps(data))

    def broken_source(*, timeout=None, base_url=None):
        raise RuntimeError("network down")

    monkeypatch.setitem(SOURCES, "fake", broken_source)
    cached = model_prices("fake", "any", cache_dir=tmp_path)
    assert cached.pricing_type == "cached"
    assert cached.prices["model"].input == 1.0


def test_any_falls_back_to_cache_on_parse_error(monkeypatch, tmp_path) -> None:
    def ok_source(*, timeout=None, base_url=None):
        return {"model": TokenPrice(input=1.0, output=0.0)}, "fake"

    monkeypatch.setitem(SOURCES, "fake", ok_source)
    model_prices("fake", "live", cache_dir=tmp_path)

    cache_file = pricing_file(tmp_path)
    data = orjson.loads(cache_file.read_bytes())
    data["providers"]["fake"]["fetched_at"] = "2020-01-01T00:00:00+00:00"
    cache_file.write_bytes(orjson.dumps(data))

    def broken_source(*, timeout=None, base_url=None):
        raise PricingParseError("layout changed")

    monkeypatch.setitem(SOURCES, "fake", broken_source)
    cached = model_prices("fake", "any", cache_dir=tmp_path)
    assert cached.pricing_type == "cached"
    assert cached.prices["model"].input == 1.0


def test_cached_mode_without_cache_raises(tmp_path) -> None:
    with pytest.raises(PricingUnavailableError):
        model_prices("unknown-provider", "cached", cache_dir=tmp_path)


def test_corrupt_cache_is_ignored(tmp_path) -> None:
    cache_file = pricing_file(tmp_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("{not json")
    assert PricingCache(tmp_path).get("fake") is None
    with pytest.raises(PricingUnavailableError):
        model_prices("fake", "cached", cache_dir=tmp_path)


def test_clear_pricing_cache(tmp_path) -> None:
    cache = PricingCache(tmp_path)
    cache.put("fake", {"m": TokenPrice(input=1, output=1)}, source="x")
    assert cache.get("fake") is not None
    clear_pricing_cache("fake", cache_dir=tmp_path)
    assert cache.get("fake") is None


def test_cache_dir_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BRAINSTORM_CACHE_DIR", str(tmp_path / "custom"))
    assert default_cache_dir() == tmp_path / "custom"
    assert pricing_file() == tmp_path / "custom" / "pricing.json"


def test_default_cache_path_uses_xdg(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("BRAINSTORM_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_cache_dir() == tmp_path / "brainstorm"


def test_compute_cost() -> None:
    price = TokenPrice(input=1.0, output=2.0, cache_read=0.5, cache_write=3.0)
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    assert compute_cost(price, usage) == Decimal("6.5")
