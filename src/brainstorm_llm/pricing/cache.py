from __future__ import annotations

import os
import orjson
import tempfile
import threading

from pathlib import Path
from datetime import datetime, timezone

from typing import Any
from loguru import logger

from ..schemas.pricing import TokenPrice

CACHE_VERSION = 1


def default_cache_dir() -> Path:
    """Return the platform cache directory for BrainStorm data.

    Uses ``BRAINSTORM_CACHE_DIR`` when set, ``%LOCALAPPDATA%`` on Windows, and
    ``$XDG_CACHE_HOME`` or ``~/.cache`` elsewhere, always under ``brainstorm/``.
    """

    override = os.getenv("BRAINSTORM_CACHE_DIR")
    if override:
        return Path(override).expanduser()

    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    else:
        base = os.getenv("XDG_CACHE_HOME")
        root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "brainstorm"


def pricing_file(cache_dir: Path | None = None) -> Path:
    """Return the path of the JSON pricing cache file."""

    return (cache_dir or default_cache_dir()) / "pricing.json"


class PricingCache:
    """Read and atomically write the per-provider pricing cache.

    The cache lives in one JSON file with a ``providers`` mapping. Corrupt or
    missing files are treated as empty so pricing failures never break calls.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.path = pricing_file(cache_dir)
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return {"version": CACHE_VERSION, "providers": {}}
        except OSError as exc:
            logger.warning("could not read pricing cache {}: {}", self.path, exc)
            return {"version": CACHE_VERSION, "providers": {}}

        try:
            payload = orjson.loads(raw)
        except orjson.JSONDecodeError as exc:
            logger.warning("ignoring corrupted pricing cache {}: {}", self.path, exc)
            return {"version": CACHE_VERSION, "providers": {}}

        if not isinstance(payload, dict):
            return {"version": CACHE_VERSION, "providers": {}}
        if not isinstance(payload.get("providers"), dict):
            payload["providers"] = {}
        return payload

    def get(self, provider: str) -> dict[str, Any] | None:
        """Return the cached entry for ``provider``, or ``None`` if absent."""

        with self._lock:
            entry = self._read().get("providers", {}).get(provider)
        return entry if isinstance(entry, dict) else None

    def put(
        self,
        provider: str,
        prices: dict[str, TokenPrice],
        *,
        source: str,
        fetched_at: datetime | None = None,
    ) -> None:
        """Write one provider's prices, replacing any previous entry."""

        fetched_at = fetched_at or datetime.now(timezone.utc)
        entry = {
            "fetched_at": fetched_at.astimezone(timezone.utc).isoformat(),
            "source": source,
            "prices": {
                key: value.model_dump(mode="json", by_alias=True, exclude_none=True)
                for key, value in prices.items()
            },
        }

        with self._lock:
            payload = self._read()
            providers = payload.setdefault("providers", {})
            providers[provider] = entry
            payload["version"] = CACHE_VERSION
            self._write(payload)

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = orjson.dumps(payload, option=orjson.OPT_INDENT_2)
        fd, tmp_name = tempfile.mkstemp(prefix=".pricing-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_name, self.path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def clear(self, provider: str | None = None) -> None:
        """Remove one provider's prices, or every provider when ``None``."""

        with self._lock:
            payload = self._read()
            if provider is None:
                payload["providers"] = {}
            else:
                payload.get("providers", {}).pop(provider, None)
            self._write(payload)
