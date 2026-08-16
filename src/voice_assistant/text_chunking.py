from __future__ import annotations


_STRONG_BOUNDARIES = frozenset("。！？!?；;\n")
_SOFT_BOUNDARIES = frozenset("，,、：: ")


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


def _last_boundary(
    text: str,
    boundaries: frozenset[str],
    minimum_boundary: int,
) -> int | None:
    for index in range(len(text) - 1, minimum_boundary - 2, -1):
        if text[index] in boundaries:
            return index + 1
    return None
