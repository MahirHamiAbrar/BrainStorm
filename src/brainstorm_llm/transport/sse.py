from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterable, AsyncIterator, Iterable, Iterator


@dataclass(frozen=True)
class SSEEvent:
    data: str
    event: str | None = None


def _dispatch(event: str | None, data_lines: list[str]) -> SSEEvent | None:
    if not data_lines:
        return None
    return SSEEvent(data="\n".join(data_lines), event=event)


def iter_sse_events(lines: Iterable[str]) -> Iterator[SSEEvent]:
    event: str | None = None
    data_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            parsed = _dispatch(event, data_lines)
            event = None
            data_lines = []
            if parsed is not None:
                yield parsed
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)

    parsed = _dispatch(event, data_lines)
    if parsed is not None:
        yield parsed


async def aiter_sse_events(lines: AsyncIterable[str]) -> AsyncIterator[SSEEvent]:
    event: str | None = None
    data_lines: list[str] = []

    async for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            parsed = _dispatch(event, data_lines)
            event = None
            data_lines = []
            if parsed is not None:
                yield parsed
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)

    parsed = _dispatch(event, data_lines)
    if parsed is not None:
        yield parsed
