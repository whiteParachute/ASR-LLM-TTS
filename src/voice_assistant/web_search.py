from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Callable
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


URLopener = Callable[..., Any]
_TIME_RANGES = frozenset({"day", "month", "year"})
_WHITESPACE = re.compile(r"\s+")


class WebSearchError(RuntimeError):
    """Raised when a configured search provider cannot return safe results."""


class WebSearchProvider(Protocol):
    def search(
        self,
        query: str,
        *,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        ...


class SearXNGSearchProvider:
    """Query one operator-configured SearXNG instance through its JSON API."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 6.0,
        max_results: int = 5,
        max_response_bytes: int = 1_000_000,
        language: str = "zh-CN",
        safesearch: int = 1,
        opener: URLopener = urlopen,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Web search timeout must be positive")
        if not 1 <= max_results <= 10:
            raise ValueError("Web search max_results must be between 1 and 10")
        if not 1024 <= max_response_bytes <= 5_000_000:
            raise ValueError(
                "Web search response limit must be between 1024 and 5000000"
            )
        if safesearch not in {0, 1, 2}:
            raise ValueError("SearXNG safesearch must be 0, 1, or 2")
        if not language.strip() or len(language) > 32:
            raise ValueError("SearXNG language must contain 1 to 32 characters")

        self._search_endpoint = _normalize_search_endpoint(endpoint)
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results
        self._max_response_bytes = max_response_bytes
        self._language = language.strip()
        self._safesearch = safesearch
        self._opener = opener

    def search(
        self,
        query: str,
        *,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        cleaned_query = _clean_text(query, max_chars=300)
        if not cleaned_query:
            raise ValueError("Web search query cannot be empty")
        if time_range is not None and time_range not in _TIME_RANGES:
            raise ValueError("Web search time_range must be day, month, or year")

        parameters: dict[str, str | int] = {
            "q": cleaned_query,
            "format": "json",
            "language": self._language,
            "safesearch": self._safesearch,
        }
        if time_range is not None:
            parameters["time_range"] = time_range
        request_url = f"{self._search_endpoint}?{urlencode(parameters)}"
        request = Request(
            request_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "local-voice-assistant/0.1",
            },
            method="GET",
        )

        try:
            with self._opener(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                if "application/json" not in content_type.lower():
                    raise WebSearchError(
                        "SearXNG returned a non-JSON response; enable JSON "
                        "format in the instance settings"
                    )
                payload = response.read(self._max_response_bytes + 1)
        except WebSearchError:
            raise
        except HTTPError as exc:
            if exc.code == 403:
                raise WebSearchError(
                    "SearXNG rejected JSON output; enable the json format"
                ) from exc
            raise WebSearchError(
                f"SearXNG request failed with HTTP {exc.code}"
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise WebSearchError("SearXNG request failed") from exc

        if len(payload) > self._max_response_bytes:
            raise WebSearchError("SearXNG response exceeded the size limit")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebSearchError("SearXNG returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise WebSearchError("SearXNG response must be a JSON object")

        raw_results = decoded.get("results", [])
        if not isinstance(raw_results, list):
            raise WebSearchError("SearXNG results must be a list")

        results: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for raw_result in raw_results:
            normalized = _normalize_result(raw_result)
            if normalized is None or normalized["url"] in seen_urls:
                continue
            seen_urls.add(normalized["url"])
            results.append(normalized)
            if len(results) >= self._max_results:
                break

        return {
            "query": cleaned_query,
            "result_count": len(results),
            "results": results,
        }


def _normalize_search_endpoint(endpoint: str) -> str:
    cleaned = endpoint.strip()
    try:
        parsed = urlsplit(cleaned)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("SearXNG endpoint is invalid") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("SearXNG endpoint must be an HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("SearXNG endpoint cannot contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("SearXNG endpoint cannot contain query or fragment")

    path = parsed.path.rstrip("/")
    if not path.endswith("/search"):
        path = f"{path}/search"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _normalize_result(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    title = _clean_text(value.get("title", ""), max_chars=120)
    result_url = _clean_text(value.get("url", ""), max_chars=1000)
    if not title or not _is_safe_result_url(result_url):
        return None

    result = {
        "title": title,
        "url": result_url,
        "snippet": _clean_text(value.get("content", ""), max_chars=240),
    }
    published_at = _clean_text(
        value.get("publishedDate", value.get("published_date", "")),
        max_chars=64,
    )
    if published_at:
        result["published_at"] = published_at

    engines = value.get("engines", value.get("engine", ""))
    if isinstance(engines, list):
        source = ",".join(
            _clean_text(engine, max_chars=32)
            for engine in engines[:3]
            if _clean_text(engine, max_chars=32)
        )
    else:
        source = _clean_text(engines, max_chars=64)
    if source:
        result["source"] = source
    return result


def _is_safe_result_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    hostname = parsed.hostname
    if not (
        parsed.scheme in {"http", "https"}
        and hostname
        and parsed.username is None
        and parsed.password is None
    ):
        return False
    lowered_hostname = hostname.lower().rstrip(".")
    if lowered_hostname == "localhost" or lowered_hostname.endswith(".local"):
        return False
    try:
        address = ipaddress.ip_address(lowered_hostname)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _clean_text(value: Any, *, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", value).strip()[:max_chars]
