from __future__ import annotations

from collections.abc import Iterable, Iterator


_STRONG_BOUNDARIES = frozenset("。！？!?；;\n")
_SOFT_BOUNDARIES = frozenset("，,、：: ")
_STRONG_BOUNDARY_LOOKAHEAD = 1


def split_reply_text(text: str, max_chars: int) -> list[str]:
    """Split reply text into natural, bounded TTS input chunks."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1.")

    remaining = text.strip()
    if not remaining:
        return []

    chunks: list[str] = []
    minimum_boundary = min(6, max_chars)

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        split_at = _last_boundary(
            window,
            _STRONG_BOUNDARIES,
            minimum_boundary,
        )
        if split_at is None:
            split_at = _last_boundary(
                window,
                _SOFT_BOUNDARIES,
                minimum_boundary,
            )
        if split_at is None:
            split_at = max_chars

        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def stream_reply_text(
    parts: Iterable[str],
    *,
    first_chunk_chars: int,
    max_chars: int,
) -> Iterator[str]:
    """Yield an early first TTS segment while later text is arriving."""
    if first_chunk_chars < 1:
        raise ValueError("first_chunk_chars must be at least 1.")
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1.")
    if first_chunk_chars > max_chars:
        raise ValueError("first_chunk_chars cannot exceed max_chars.")

    buffer = ""
    limit = first_chunk_chars
    for part in parts:
        buffer += part
        while buffer:
            boundary_window = buffer[
                : limit + _STRONG_BOUNDARY_LOOKAHEAD
            ]
            strong_boundary = _first_boundary(
                boundary_window,
                _STRONG_BOUNDARIES,
            )
            if strong_boundary is not None:
                split_at = strong_boundary
            elif len(buffer) > limit:
                window = buffer[:limit]
                split_at = _last_boundary(
                    window,
                    _SOFT_BOUNDARIES,
                    min(3, limit),
                )
                if split_at is None:
                    split_at = limit
            else:
                break

            chunk = buffer[:split_at].strip()
            buffer = buffer[split_at:].lstrip()
            if chunk:
                yield chunk
                limit = max_chars

    remaining = buffer.strip()
    if remaining:
        yield remaining


def _last_boundary(
    text: str,
    boundaries: frozenset[str],
    minimum_boundary: int,
) -> int | None:
    for index in range(len(text) - 1, minimum_boundary - 2, -1):
        if text[index] in boundaries:
            return index + 1
    return None


def _first_boundary(
    text: str,
    boundaries: frozenset[str],
) -> int | None:
    for index, character in enumerate(text):
        if character in boundaries:
            return index + 1
    return None
