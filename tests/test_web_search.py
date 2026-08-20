import json
import unittest
from urllib.parse import parse_qs, urlsplit

from voice_assistant.web_search import (
    SearXNGSearchProvider,
    WebSearchError,
)


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        content_type: str = "application/json; charset=utf-8",
    ) -> None:
        self.payload = payload
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self.payload[:limit]


class RecordingOpener:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.request = None
        self.timeout = None

    def __call__(self, request, *, timeout):
        self.request = request
        self.timeout = timeout
        return self.response


class SearXNGSearchProviderTest(unittest.TestCase):
    def test_queries_json_api_and_returns_bounded_unique_results(self) -> None:
        payload = json.dumps(
            {
                "results": [
                    {
                        "title": "  第一条  新闻 ",
                        "url": "https://example.com/one",
                        "content": "第一条摘要。",
                        "publishedDate": "2026-08-21",
                        "engines": ["bing", "brave"],
                    },
                    {
                        "title": "重复结果",
                        "url": "https://example.com/one",
                        "content": "不应重复。",
                    },
                    {
                        "title": "脚本结果",
                        "url": "javascript:alert(1)",
                    },
                    {
                        "title": "本机结果",
                        "url": "http://127.0.0.1/private",
                    },
                    {
                        "title": "第二条",
                        "url": "https://example.org/two",
                        "content": "第二条摘要。",
                    },
                ]
            },
            ensure_ascii=False,
        ).encode("utf-8")
        opener = RecordingOpener(FakeResponse(payload))
        provider = SearXNGSearchProvider(
            "https://search.example/searxng",
            timeout_seconds=4,
            max_results=2,
            opener=opener,
        )

        result = provider.search(" 最新 AI 新闻 ", time_range="day")

        self.assertEqual(result["query"], "最新 AI 新闻")
        self.assertEqual(result["result_count"], 2)
        self.assertEqual(
            [item["url"] for item in result["results"]],
            ["https://example.com/one", "https://example.org/two"],
        )
        self.assertEqual(result["results"][0]["title"], "第一条 新闻")
        self.assertEqual(result["results"][0]["source"], "bing,brave")
        self.assertEqual(opener.timeout, 4)
        request_url = urlsplit(opener.request.full_url)
        self.assertEqual(request_url.path, "/searxng/search")
        parameters = parse_qs(request_url.query)
        self.assertEqual(parameters["q"], ["最新 AI 新闻"])
        self.assertEqual(parameters["format"], ["json"])
        self.assertEqual(parameters["time_range"], ["day"])
        self.assertEqual(parameters["language"], ["zh-CN"])

    def test_rejects_unsafe_endpoint_and_invalid_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTP or HTTPS"):
            SearXNGSearchProvider("file:///tmp/search")
        with self.assertRaisesRegex(ValueError, "credentials"):
            SearXNGSearchProvider("https://user:secret@example.com")
        with self.assertRaisesRegex(ValueError, "max_results"):
            SearXNGSearchProvider("https://example.com", max_results=11)

        provider = SearXNGSearchProvider(
            "https://example.com",
            opener=RecordingOpener(FakeResponse(b'{"results":[]}')),
        )
        with self.assertRaisesRegex(ValueError, "time_range"):
            provider.search("news", time_range="week")

    def test_rejects_non_json_and_oversized_responses(self) -> None:
        html_provider = SearXNGSearchProvider(
            "https://example.com",
            opener=RecordingOpener(
                FakeResponse(b"<html></html>", content_type="text/html")
            ),
        )
        with self.assertRaisesRegex(WebSearchError, "non-JSON"):
            html_provider.search("news")

        oversized_provider = SearXNGSearchProvider(
            "https://example.com",
            max_response_bytes=1024,
            opener=RecordingOpener(FakeResponse(b"x" * 1025)),
        )
        with self.assertRaisesRegex(WebSearchError, "size limit"):
            oversized_provider.search("news")

    def test_rejects_malformed_result_shape(self) -> None:
        provider = SearXNGSearchProvider(
            "https://example.com",
            opener=RecordingOpener(
                FakeResponse(b'{"results":{"title":"wrong"}}')
            ),
        )

        with self.assertRaisesRegex(WebSearchError, "must be a list"):
            provider.search("news")


if __name__ == "__main__":
    unittest.main()
