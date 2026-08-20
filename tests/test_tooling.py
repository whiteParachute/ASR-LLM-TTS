import json
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path

from voice_assistant.contracts import (
    Message,
    ToolAwareResponse,
    ToolCall,
    ToolDefinition,
)
from voice_assistant.tooling import (
    BoundedToolLoop,
    ToolLoopError,
    ToolRegistry,
    build_builtin_tool_registry,
)
from voice_assistant.observability import PerformanceLogger


class FakeToolLLM:
    def __init__(
        self,
        responses: list[ToolAwareResponse],
        *,
        plain_reply: str | None = None,
    ) -> None:
        self.responses = responses
        self.plain_reply = plain_reply
        self.plain_conversations: list[list[Message]] = []
        self.conversations: list[list[Message]] = []
        self.tools: list[tuple[ToolDefinition, ...]] = []

    def generate(self, messages):
        if self.plain_reply is None:
            raise AssertionError("plain generation should not be used")
        self.plain_conversations.append(list(messages))
        return self.plain_reply

    def generate_with_tools(self, messages, tools):
        self.conversations.append(list(messages))
        self.tools.append(tuple(tools))
        return self.responses.pop(0)


class FakeWebSearch:
    def __init__(self) -> None:
        self.query = ""
        self.time_range: str | None = None

    def search(self, query: str, *, time_range: str | None = None):
        self.query = query
        self.time_range = time_range
        return {
            "query": query,
            "result_count": 1,
            "results": [
                {
                    "title": "AI 新闻",
                    "url": "https://example.com/news",
                    "snippet": "一条最新消息。",
                }
            ],
        }


class FailingWebSearch:
    def search(self, query: str, *, time_range: str | None = None):
        raise RuntimeError("network unavailable")


class ToolingTest(unittest.TestCase):
    def test_executes_calculator_without_eval(self) -> None:
        registry = build_builtin_tool_registry(
            timeout_seconds=1,
            max_result_chars=500,
        )

        result = registry.execute(
            ToolCall(
                name="calculate",
                arguments={"expression": "(12 + 3) * 4 / 2"},
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(
            json.loads(result.content)["result"]["value"],
            30,
        )
        self.assertEqual(
            json.loads(result.content)["result"]["answer"],
            "30",
        )

        unsafe = registry.execute(
            ToolCall(
                name="calculate",
                arguments={"expression": "__import__('os').getcwd()"},
            )
        )
        self.assertFalse(unsafe.ok)
        self.assertEqual(unsafe.error_type, "execution_error")

    def test_returns_fixed_current_time_in_requested_timezone(self) -> None:
        def fixed_now(timezone):
            return datetime(2026, 8, 20, 22, 30, tzinfo=timezone)

        registry = build_builtin_tool_registry(
            timeout_seconds=1,
            max_result_chars=500,
            now_factory=fixed_now,
        )

        result = registry.execute(
            ToolCall(
                name="get_current_time",
                arguments={"timezone": "Asia/Shanghai"},
            )
        )

        payload = json.loads(result.content)
        self.assertTrue(result.ok)
        self.assertEqual(payload["result"]["timezone"], "Asia/Shanghai")
        self.assertEqual(
            payload["result"]["local_time"],
            "2026-08-20 22:30:00",
        )
        self.assertEqual(
            result.direct_reply,
            "现在是2026年8月20日，星期四，22点30分。",
        )

        clock_only = registry.execute(
            ToolCall(
                name="get_current_time",
                arguments={"timezone": "Asia/Shanghai"},
            ),
            reply_context="现在几点了",
        )
        self.assertEqual(clock_only.direct_reply, "现在是22点30分。")

    def test_routes_only_tools_matching_explicit_intent(self) -> None:
        registry = build_builtin_tool_registry(
            timeout_seconds=1,
            max_result_chars=500,
        )

        self.assertEqual(registry.definitions_for_text("你好"), ())
        self.assertEqual(
            [
                definition.name
                for definition in registry.definitions_for_text(
                    "一二三加四等于多少"
                )
            ],
            ["calculate"],
        )
        self.assertEqual(
            [
                definition.name
                for definition in registry.definitions_for_text(
                    "现在几点了"
                )
            ],
            ["get_current_time"],
        )

    def test_rejects_unknown_tool_and_invalid_arguments(self) -> None:
        registry = build_builtin_tool_registry(
            timeout_seconds=1,
            max_result_chars=500,
        )

        unknown = registry.execute(ToolCall(name="missing", arguments={}))
        invalid = registry.execute(
            ToolCall(name="calculate", arguments={"extra": "1+1"})
        )

        self.assertEqual(unknown.error_type, "unknown_tool")
        self.assertEqual(invalid.error_type, "invalid_arguments")

    def test_times_out_and_bounds_tool_results(self) -> None:
        registry = ToolRegistry(timeout_seconds=0.01, max_result_chars=128)
        release = threading.Event()
        definition = ToolDefinition(
            name="slow_tool",
            description="Test tool.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
        registry.register(definition, lambda _: release.wait(timeout=1))

        timeout = registry.execute(ToolCall(name="slow_tool", arguments={}))
        release.set()

        self.assertFalse(timeout.ok)
        self.assertEqual(timeout.error_type, "timeout")

        bounded = ToolRegistry(timeout_seconds=1, max_result_chars=128)
        bounded.register(definition, lambda _: "x" * 1000)
        result = bounded.execute(ToolCall(name="slow_tool", arguments={}))
        payload = json.loads(result.content)
        self.assertLessEqual(len(result.content), 128)
        self.assertTrue(payload["truncated"])

    def test_supports_per_tool_timeout_and_failure_reply(self) -> None:
        registry = ToolRegistry(timeout_seconds=1, max_result_chars=500)
        release = threading.Event()
        registry.register(
            ToolDefinition(
                name="network_tool",
                description="Test network timeout.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            lambda _: release.wait(timeout=1),
            timeout_seconds=0.01,
            failure_reply="网络暂时不可用。",
        )

        result = registry.execute(
            ToolCall(name="network_tool", arguments={})
        )
        release.set()

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "timeout")
        self.assertEqual(result.direct_reply, "网络暂时不可用。")

    def test_returns_builtin_calculator_result_without_second_llm(self) -> None:
        registry = build_builtin_tool_registry(
            timeout_seconds=1,
            max_result_chars=500,
        )
        llm = FakeToolLLM(
            [
                ToolAwareResponse(
                    tool_calls=(
                        ToolCall(
                            name="calculate",
                            arguments={"expression": "6 * 7"},
                        ),
                    )
                ),
            ]
        )
        loop = BoundedToolLoop(llm, registry, max_rounds=2)

        reply = loop.generate([Message(role="user", content="6乘7是多少？")])

        self.assertEqual(reply, "答案是42。")
        self.assertEqual(len(llm.conversations), 1)
        self.assertEqual([tool.name for tool in llm.tools[0]], ["calculate"])

    def test_plain_chat_bypasses_tool_prompt(self) -> None:
        registry = build_builtin_tool_registry(
            timeout_seconds=1,
            max_result_chars=500,
        )
        llm = FakeToolLLM([], plain_reply="你好呀。")
        loop = BoundedToolLoop(llm, registry, max_rounds=2)

        reply = loop.generate([Message(role="user", content="你好")])

        self.assertEqual(reply, "你好呀。")
        self.assertEqual(len(llm.plain_conversations), 1)
        self.assertEqual(llm.conversations, [])

    def test_keeps_native_second_round_for_unformatted_tools(self) -> None:
        registry = ToolRegistry(timeout_seconds=1, max_result_chars=500)
        registry.register(
            ToolDefinition(
                name="lookup",
                description="Look up a value.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            lambda _: {"value": "found"},
        )
        llm = FakeToolLLM(
            [
                ToolAwareResponse(
                    tool_calls=(ToolCall(name="lookup", arguments={}),)
                ),
                ToolAwareResponse(content="找到了。"),
            ]
        )
        loop = BoundedToolLoop(llm, registry, max_rounds=2)

        reply = loop.generate([Message(role="user", content="查一下")])

        self.assertEqual(reply, "找到了。")
        followup = llm.conversations[1]
        self.assertEqual(followup[-2].role, "assistant")
        self.assertEqual(followup[-1].role, "tool")

    def test_routes_web_search_through_native_summary_round(self) -> None:
        web_search = FakeWebSearch()
        registry = build_builtin_tool_registry(
            timeout_seconds=1,
            max_result_chars=2000,
            web_search=web_search,
            web_search_timeout_seconds=2,
        )
        llm = FakeToolLLM(
            [
                ToolAwareResponse(
                    tool_calls=(
                        ToolCall(
                            name="web_search",
                            arguments={
                                "query": "最新 AI 新闻",
                                "time_range": "day",
                            },
                        ),
                    )
                ),
                ToolAwareResponse(content="今天有一条新的AI消息。"),
            ]
        )
        loop = BoundedToolLoop(llm, registry, max_rounds=2)

        reply = loop.generate(
            [Message(role="user", content="帮我查一下最新AI新闻")]
        )

        self.assertEqual(reply, "今天有一条新的AI消息。")
        self.assertEqual(web_search.query, "最新 AI 新闻")
        self.assertEqual(web_search.time_range, "day")
        self.assertEqual(
            [definition.name for definition in llm.tools[0]],
            ["web_search"],
        )
        tool_result = json.loads(llm.conversations[1][-1].content)
        self.assertEqual(
            tool_result["result"]["results"][0]["url"],
            "https://example.com/news",
        )

    def test_returns_one_fallback_when_web_search_fails(self) -> None:
        registry = build_builtin_tool_registry(
            timeout_seconds=1,
            max_result_chars=2000,
            web_search=FailingWebSearch(),
            web_search_timeout_seconds=2,
        )
        llm = FakeToolLLM(
            [
                ToolAwareResponse(
                    tool_calls=(
                        ToolCall(
                            name="web_search",
                            arguments={"query": "最新 AI 新闻"},
                        ),
                    )
                )
            ]
        )
        loop = BoundedToolLoop(llm, registry, max_rounds=3)

        reply = loop.generate(
            [Message(role="user", content="帮我查一下最新AI新闻")]
        )

        self.assertEqual(reply, "暂时无法联网查询，请稍后再试。")
        self.assertEqual(len(llm.conversations), 1)

    def test_rejects_invalid_web_search_time_range_before_execution(
        self,
    ) -> None:
        web_search = FakeWebSearch()
        registry = build_builtin_tool_registry(
            timeout_seconds=1,
            max_result_chars=2000,
            web_search=web_search,
        )

        result = registry.execute(
            ToolCall(
                name="web_search",
                arguments={"query": "AI 新闻", "time_range": "week"},
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "invalid_arguments")
        self.assertEqual(web_search.query, "")

    def test_stops_when_model_exceeds_tool_round_limit(self) -> None:
        registry = ToolRegistry(timeout_seconds=1, max_result_chars=500)
        registry.register(
            ToolDefinition(
                name="repeat_tool",
                description="Always requests another round.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            lambda _: {"ok": True},
        )
        repeated_call = ToolAwareResponse(
            tool_calls=(
                ToolCall(
                    name="repeat_tool",
                    arguments={},
                ),
            )
        )
        llm = FakeToolLLM([repeated_call, repeated_call])
        loop = BoundedToolLoop(llm, registry, max_rounds=1)

        with self.assertRaisesRegex(ToolLoopError, "round limit"):
            loop.generate([Message(role="user", content="继续")])

    def test_logs_tool_metadata_without_arguments_or_results(self) -> None:
        registry = build_builtin_tool_registry(
            timeout_seconds=1,
            max_result_chars=500,
        )
        llm = FakeToolLLM(
            [
                ToolAwareResponse(
                    tool_calls=(
                        ToolCall(
                            name="calculate",
                            arguments={"expression": "12345 * 6789"},
                        ),
                    )
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            performance = PerformanceLogger(
                enabled=True,
                console=False,
                jsonl=True,
                log_dir=Path(temp_dir),
                session_id="tool-session",
            )
            loop = BoundedToolLoop(
                llm,
                registry,
                max_rounds=2,
                performance=performance,
            )
            with performance.turn("turn_0001"):
                loop.generate(
                    [Message(role="user", content="计算12345乘6789")]
                )
            performance.close()
            log_text = (Path(temp_dir) / "performance.jsonl").read_text(
                encoding="utf-8"
            )

        events = [json.loads(line) for line in log_text.splitlines()]
        tool_event = next(
            event for event in events if event["stage"] == "tool_execute"
        )
        self.assertEqual(tool_event["tool_name"], "calculate")
        self.assertTrue(tool_event["tool_ok"])
        self.assertNotIn("12345", log_text)
        self.assertNotIn("result", log_text)


if __name__ == "__main__":
    unittest.main()
