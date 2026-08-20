from __future__ import annotations

import ast
import json
import math
import operator
import queue
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from threading import Thread
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from voice_assistant.contracts import (
    Message,
    ToolAwareResponse,
    ToolCall,
    ToolCallingLLMProvider,
    ToolDefinition,
)
from voice_assistant.observability import PerformanceLogger, measure_stage


ToolHandler = Callable[[dict[str, Any]], Any]
NowFactory = Callable[[ZoneInfo], datetime]
_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class ToolLoopError(RuntimeError):
    """Raised when a bounded tool conversation cannot produce a reply."""


@dataclass(frozen=True, slots=True)
class ToolExecution:
    content: str
    ok: bool
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class _RegisteredTool:
    definition: ToolDefinition
    handler: ToolHandler


class ToolRegistry:
    """Register trusted tools and execute them behind strict boundaries."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 2.0,
        max_result_chars: int = 2000,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Tool timeout must be positive")
        if max_result_chars < 128:
            raise ValueError("Tool result limit must be at least 128 chars")
        self._timeout_seconds = timeout_seconds
        self._max_result_chars = max_result_chars
        self._tools: dict[str, _RegisteredTool] = {}

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools.values())

    def register(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
    ) -> None:
        if not _TOOL_NAME.fullmatch(definition.name):
            raise ValueError(f"Invalid tool name: {definition.name}")
        if definition.name in self._tools:
            raise ValueError(f"Tool is already registered: {definition.name}")
        if definition.parameters.get("type") != "object":
            raise ValueError("Tool parameters schema must describe an object")
        self._tools[definition.name] = _RegisteredTool(
            definition=definition,
            handler=handler,
        )

    def execute(self, call: ToolCall) -> ToolExecution:
        registered = self._tools.get(call.name)
        if registered is None:
            return self._error("unknown_tool", f"Unknown tool: {call.name}")

        try:
            arguments = _validate_arguments(
                call.arguments,
                registered.definition.parameters,
            )
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))

        results: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                results.put((True, registered.handler(arguments)))
            except BaseException as exc:
                results.put((False, exc))

        worker = Thread(
            target=invoke,
            name=f"tool-{call.name}",
            daemon=True,
        )
        worker.start()
        worker.join(self._timeout_seconds)
        if worker.is_alive():
            return self._error(
                "timeout",
                f"Tool exceeded {self._timeout_seconds:g}s timeout",
            )

        succeeded, value = results.get_nowait()
        if not succeeded:
            return self._error("execution_error", "Tool execution failed")
        return ToolExecution(
            content=_bounded_json(
                {"ok": True, "result": value},
                self._max_result_chars,
            ),
            ok=True,
        )

    def _error(self, error_type: str, message: str) -> ToolExecution:
        return ToolExecution(
            content=_bounded_json(
                {
                    "ok": False,
                    "error": {
                        "type": error_type,
                        "message": message,
                    },
                },
                self._max_result_chars,
            ),
            ok=False,
            error_type=error_type,
        )


class BoundedToolLoop:
    """Run native model tool calls with a hard maximum number of rounds."""

    def __init__(
        self,
        llm: ToolCallingLLMProvider,
        registry: ToolRegistry,
        *,
        max_rounds: int = 3,
        performance: PerformanceLogger | None = None,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("Tool loop max_rounds must be at least 1")
        if not callable(getattr(llm, "generate_with_tools", None)):
            raise ValueError("Configured LLM does not support native tools")
        if not registry.definitions:
            raise ValueError("Tool loop requires at least one registered tool")
        self._llm = llm
        self._registry = registry
        self._max_rounds = max_rounds
        self._performance = performance

    def generate(self, messages: Sequence[Message]) -> str:
        conversation = list(messages)
        tool_rounds = 0
        tool_calls = 0

        with measure_stage(
            self._performance,
            "tool_loop",
            max_rounds=self._max_rounds,
        ) as loop_span:
            while True:
                response = self._llm.generate_with_tools(
                    conversation,
                    self._registry.definitions,
                )
                if not response.tool_calls:
                    reply = response.content.strip()
                    if not reply:
                        raise ToolLoopError(
                            "LLM returned neither a reply nor a tool call"
                        )
                    loop_span.add_fields(
                        tool_rounds=tool_rounds,
                        tool_calls=tool_calls,
                        output_chars=len(reply),
                    )
                    return reply

                if tool_rounds >= self._max_rounds:
                    raise ToolLoopError(
                        "Tool loop exceeded the configured round limit"
                    )

                conversation.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                tool_rounds += 1
                for call_index, call in enumerate(
                    response.tool_calls,
                    start=1,
                ):
                    with measure_stage(
                        self._performance,
                        "tool_execute",
                        tool_name=call.name,
                        tool_round=tool_rounds,
                        call_index=call_index,
                    ) as tool_span:
                        execution = self._registry.execute(call)
                        tool_span.add_fields(
                            tool_ok=execution.ok,
                            tool_error_type=execution.error_type,
                        )
                    tool_calls += 1
                    conversation.append(
                        Message(role="tool", content=execution.content)
                    )


def build_builtin_tool_registry(
    *,
    timeout_seconds: float,
    max_result_chars: int,
    now_factory: NowFactory | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(
        timeout_seconds=timeout_seconds,
        max_result_chars=max_result_chars,
    )
    registry.register(
        ToolDefinition(
            name="get_current_time",
            description=(
                "Get the current date and time in an IANA timezone. "
                "Use this instead of guessing the current time."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "IANA timezone such as Asia/Shanghai."
                        ),
                    }
                },
                "additionalProperties": False,
            },
        ),
        _build_time_handler(now_factory or datetime.now),
    )
    registry.register(
        ToolDefinition(
            name="calculate",
            description=(
                "Evaluate a basic arithmetic expression accurately. "
                "Supports +, -, *, /, //, %, **, and parentheses."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        ),
        _calculate,
    )
    return registry


def _build_time_handler(now_factory: NowFactory) -> ToolHandler:
    def get_current_time(arguments: dict[str, Any]) -> dict[str, str]:
        timezone_name = str(arguments.get("timezone", "Asia/Shanghai"))
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown timezone: {timezone_name}") from exc
        current = now_factory(timezone)
        return {
            "timezone": timezone_name,
            "iso8601": current.isoformat(),
            "local_time": current.strftime("%Y-%m-%d %H:%M:%S"),
        }

    return get_current_time


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _calculate(arguments: dict[str, Any]) -> dict[str, Any]:
    expression = str(arguments["expression"]).strip()
    if not expression or len(expression) > 200:
        raise ValueError("Expression must contain 1 to 200 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Expression is not valid arithmetic") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression is too complex")
    value = _evaluate_arithmetic(tree.body)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return {"expression": expression, "value": value}


def _evaluate_arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value,
            (int, float),
        ):
            raise ValueError("Only numeric constants are allowed")
        value: int | float = node.value
    elif isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Arithmetic operator is not allowed")
        left = _evaluate_arithmetic(node.left)
        right = _evaluate_arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent is too large")
        try:
            value = operation(left, right)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError("Arithmetic operation is invalid") from exc
    elif isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unary operator is not allowed")
        value = operation(_evaluate_arithmetic(node.operand))
    else:
        raise ValueError("Expression contains unsupported syntax")

    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Arithmetic result is not finite")
    if abs(value) > 1e15:
        raise ValueError("Arithmetic result is too large")
    return value


def _validate_arguments(
    arguments: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Tool schema properties must be an object")

    required = schema.get("required", [])
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError("Missing required argument: " + ", ".join(missing))

    if schema.get("additionalProperties", True) is False:
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            raise ValueError(
                "Unexpected argument: " + ", ".join(unexpected)
            )

    validated: dict[str, Any] = {}
    for name, value in arguments.items():
        property_schema = properties.get(name, {})
        expected_type = property_schema.get("type")
        validated[name] = _coerce_argument(value, expected_type, name)
    return validated


def _coerce_argument(value: Any, expected_type: Any, name: str) -> Any:
    if expected_type in (None, "string"):
        if not isinstance(value, str):
            raise ValueError(f"Argument {name} must be a string")
        return value
    if expected_type == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise ValueError(f"Argument {name} must be an integer")
    if expected_type == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise ValueError(f"Argument {name} must be a number")
    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError(f"Argument {name} must be a boolean")
    if expected_type in {"object", "array"} and isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Argument {name} contains invalid JSON") from exc
    expected_class = dict if expected_type == "object" else list
    if expected_type in {"object", "array"} and isinstance(
        value,
        expected_class,
    ):
        return value
    raise ValueError(f"Unsupported schema type for argument {name}")


def _bounded_json(payload: dict[str, Any], max_chars: int) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= max_chars:
        return encoded

    source = encoded
    low = 0
    high = len(source)
    best = ""
    while low <= high:
        midpoint = (low + high) // 2
        candidate = json.dumps(
            {
                "ok": payload.get("ok", False),
                "truncated": True,
                "text": source[:midpoint],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(candidate) <= max_chars:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best
