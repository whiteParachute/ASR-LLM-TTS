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
    def __init__(self, responses: list[ToolAwareResponse]) -> None:
        self.responses = responses
        self.conversations: list[list[Message]] = []
        self.tools: list[tuple[ToolDefinition, ...]] = []

    def generate(self, messages):
        raise AssertionError("plain generation should not be used")

    def generate_with_tools(self, messages, tools):
        self.conversations.append(list(messages))
        self.tools.append(tuple(tools))
        return self.responses.pop(0)


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

    def test_runs_bounded_native_tool_conversation(self) -> None:
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
                ToolAwareResponse(content="答案是42。"),
            ]
        )
        loop = BoundedToolLoop(llm, registry, max_rounds=2)

        reply = loop.generate([Message(role="user", content="6乘7是多少？")])

        self.assertEqual(reply, "答案是42。")
        followup = llm.conversations[1]
        self.assertEqual(followup[-2].role, "assistant")
        self.assertEqual(followup[-1].role, "tool")
        self.assertEqual(
            json.loads(followup[-1].content)["result"]["value"],
            42,
        )

    def test_stops_when_model_exceeds_tool_round_limit(self) -> None:
        registry = build_builtin_tool_registry(
            timeout_seconds=1,
            max_result_chars=500,
        )
        repeated_call = ToolAwareResponse(
            tool_calls=(
                ToolCall(
                    name="calculate",
                    arguments={"expression": "1+1"},
                ),
            )
        )
        llm = FakeToolLLM([repeated_call, repeated_call])
        loop = BoundedToolLoop(llm, registry, max_rounds=1)

        with self.assertRaisesRegex(ToolLoopError, "round limit"):
            loop.generate([Message(role="user", content="计算")])

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
                ToolAwareResponse(content="已计算。"),
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
                loop.generate([Message(role="user", content="计算")])
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
