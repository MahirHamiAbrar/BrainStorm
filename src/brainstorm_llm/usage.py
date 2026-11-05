from __future__ import annotations

import inspect
from decimal import Decimal
from typing import Any, Callable, Literal

from pydantic import Field

from .log import library_logger
from .schemas.base import BrainStormModel
from .schemas.messages import Usage

UsageCallback = Callable[["CallUsage"], Any]


class CallUsage(BrainStormModel):
    """Usage and cost information emitted after one model call.

    ``cost_usd`` is ``None`` when pricing is disabled or the model has no price
    entry. ``pricing_type`` reports whether the price came from a live fetch or
    the disk cache.
    """

    provider: str
    model: str
    request_id: str | None = None
    usage: Usage = Field(default_factory=Usage)
    cost_usd: Decimal | None = None
    pricing_type: Literal["live", "cached"] | None = None
    latency_ms: float = 0.0
    streaming: bool = False
    operation: Literal["generate", "stream", "system_one"] = "generate"


def emit_usage(callback: UsageCallback | None, record: CallUsage) -> None:
    """Invoke a usage callback, swallowing callback failures.

    Async callbacks are not awaited here; use the async client methods for those.
    """

    if callback is None:
        return
    try:
        result = callback(record)
        if inspect.isawaitable(result):
            result.close()  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001 - callbacks must not break requests
        library_logger(record.provider, record.model).warning(
            "usage callback failed: {}", exc
        )


async def aemit_usage(callback: UsageCallback | None, record: CallUsage) -> None:
    """Invoke a usage callback, awaiting it when it returns an awaitable."""

    if callback is None:
        return
    try:
        result = callback(record)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001 - callbacks must not break requests
        library_logger(record.provider, record.model).warning(
            "usage callback failed: {}", exc
        )


class CostTracker:
    """Callable accumulator for usage records from one or more clients.

    Pass an instance as ``on_usage`` to a client, then inspect :attr:`records`,
    :attr:`total_usd`, or :meth:`by_model`.
    """

    def __init__(self) -> None:
        self.records: list[CallUsage] = []

    def __call__(self, record: CallUsage) -> None:
        """Record one completed model call."""

        self.records.append(record)

    @property
    def total_usd(self) -> Decimal:
        """Sum of all known costs; unknown costs count as zero."""

        return sum((record.cost_usd or Decimal("0") for record in self.records), Decimal("0"))

    @property
    def total_tokens(self) -> int:
        """Total tokens across every recorded call."""

        return sum(record.usage.total_tokens for record in self.records)

    def by_model(self) -> dict[str, dict[str, Any]]:
        """Aggregate cost, token counts, and call counts per model."""

        summary: dict[str, dict[str, Any]] = {}
        for record in self.records:
            entry = summary.setdefault(
                record.model,
                {"cost_usd": Decimal("0"), "input_tokens": 0, "output_tokens": 0, "calls": 0},
            )
            entry["cost_usd"] += record.cost_usd or Decimal("0")
            entry["input_tokens"] += record.usage.input_tokens
            entry["output_tokens"] += record.usage.output_tokens
            entry["calls"] += 1
        return summary

    def reset(self) -> None:
        """Discard all recorded calls and totals."""

        self.records.clear()
