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
    SourceReference,
    ToolAwareResponse,
    ToolCall,
    ToolCallingLLMProvider,
    ToolDefinition,
)
from voice_assistant.observability import PerformanceLogger, measure_stage
from voice_assistant.web_search import WebSearchProvider


ToolHandler = Callable[[dict[str, Any]], Any]
ToolIntentMatcher = Callable[[str], bool]
ToolReplyFormatter = Callable[[Any, str], str]
ToolSourceExtractor = Callable[[Any], Sequence[SourceReference]]
NowFactory = Callable[[ZoneInfo], datetime]
_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class ToolLoopError(RuntimeError):
    """Raised when a bounded tool conversation cannot produce a reply."""


@dataclass(frozen=True, slots=True)
class ToolExecution:
    content: str
    ok: bool
    error_type: str | None = None
    direct_reply: str | None = None
    source_references: tuple[SourceReference, ...] = ()


@dataclass(frozen=True, slots=True)
class _RegisteredTool:
    definition: ToolDefinition
    handler: ToolHandler
    intent_matcher: ToolIntentMatcher | None = None
    reply_formatter: ToolReplyFormatter | None = None
    timeout_seconds: float | None = None
    failure_reply: str | None = None
    source_extractor: ToolSourceExtractor | None = None


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

    def definitions_for_text(self, text: str) -> tuple[ToolDefinition, ...]:
        """Return only tools whose deterministic intent gate matches."""
        return tuple(
            tool.definition
            for tool in self._tools.values()
            if tool.intent_matcher is None or tool.intent_matcher(text)
        )

    def register(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
        *,
        intent_matcher: ToolIntentMatcher | None = None,
        reply_formatter: ToolReplyFormatter | None = None,
        timeout_seconds: float | None = None,
        failure_reply: str | None = None,
        source_extractor: ToolSourceExtractor | None = None,
    ) -> None:
        if not _TOOL_NAME.fullmatch(definition.name):
            raise ValueError(f"Invalid tool name: {definition.name}")
        if definition.name in self._tools:
            raise ValueError(f"Tool is already registered: {definition.name}")
        if definition.parameters.get("type") != "object":
            raise ValueError("Tool parameters schema must describe an object")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("Tool timeout override must be positive")
        if failure_reply is not None and not failure_reply.strip():
            raise ValueError("Tool failure reply cannot be empty")
        self._tools[definition.name] = _RegisteredTool(
            definition=definition,
            handler=handler,
            intent_matcher=intent_matcher,
            reply_formatter=reply_formatter,
            timeout_seconds=timeout_seconds,
            failure_reply=(failure_reply.strip() if failure_reply else None),
            source_extractor=source_extractor,
        )

    def execute(
        self,
        call: ToolCall,
        *,
        reply_context: str = "",
    ) -> ToolExecution:
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
        timeout_seconds = (
            registered.timeout_seconds or self._timeout_seconds
        )
        worker.start()
        worker.join(timeout_seconds)
        if worker.is_alive():
            return self._error(
                "timeout",
                f"Tool exceeded {timeout_seconds:g}s timeout",
                direct_reply=registered.failure_reply,
            )

        succeeded, value = results.get_nowait()
        if not succeeded:
            return self._error(
                "execution_error",
                "Tool execution failed",
                direct_reply=registered.failure_reply,
            )
        direct_reply = None
        if registered.reply_formatter is not None:
            direct_reply = (
                registered.reply_formatter(value, reply_context).strip()
                or None
            )
        source_references = (
            tuple(registered.source_extractor(value))
            if registered.source_extractor is not None
            else ()
        )
        return ToolExecution(
            content=_bounded_json(
                {"ok": True, "result": value},
                self._max_result_chars,
            ),
            ok=True,
            direct_reply=direct_reply,
            source_references=source_references,
        )

    def _error(
        self,
        error_type: str,
        message: str,
        *,
        direct_reply: str | None = None,
    ) -> ToolExecution:
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
            direct_reply=direct_reply,
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
        self._source_references: tuple[SourceReference, ...] = ()

    @property
    def source_references(self) -> tuple[SourceReference, ...]:
        return self._source_references

    def generate(self, messages: Sequence[Message]) -> str:
        conversation = list(messages)
        tool_rounds = 0
        tool_calls = 0
        user_text = _last_user_text(messages)
        definitions = self._registry.definitions_for_text(user_text)
        collected_sources: list[SourceReference] = []
        seen_source_urls: set[str] = set()
        self._source_references = ()

        with measure_stage(
            self._performance,
            "tool_loop",
            max_rounds=self._max_rounds,
        ) as loop_span:
            if not definitions:
                reply = self._llm.generate(conversation).strip()
                if not reply:
                    raise ToolLoopError("LLM returned an empty reply")
                loop_span.add_fields(
                    route="plain",
                    tool_rounds=0,
                    tool_calls=0,
                    output_chars=len(reply),
                )
                return reply

            while True:
                response = self._llm.generate_with_tools(
                    conversation,
                    definitions,
                )
                if not response.tool_calls:
                    reply = response.content.strip()
                    if not reply:
                        raise ToolLoopError(
                            "LLM returned neither a reply nor a tool call"
                        )
                    loop_span.add_fields(
                        route="native_tool",
                        tool_rounds=tool_rounds,
                        tool_calls=tool_calls,
                        direct_reply=False,
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
                direct_replies: list[tuple[str, bool]] = []
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
                        execution = self._registry.execute(
                            call,
                            reply_context=user_text,
                        )
                        tool_span.add_fields(
                            tool_ok=execution.ok,
                            tool_error_type=execution.error_type,
                        )
                    tool_calls += 1
                    conversation.append(
                        Message(role="tool", content=execution.content)
                    )
                    for source in execution.source_references:
                        if source.url in seen_source_urls:
                            continue
                        seen_source_urls.add(source.url)
                        collected_sources.append(source)
                    self._source_references = tuple(collected_sources)
                    if execution.direct_reply is not None:
                        direct_replies.append(
                            (execution.direct_reply, execution.ok)
                        )

                if (
                    len(response.tool_calls) == 1
                    and len(direct_replies) == 1
                ):
                    reply, direct_reply_ok = direct_replies[0]
                    loop_span.add_fields(
                        route=(
                            "builtin_fast_path"
                            if direct_reply_ok
                            else "tool_fallback"
                        ),
                        tool_rounds=tool_rounds,
                        tool_calls=tool_calls,
                        direct_reply=True,
                        output_chars=len(reply),
                    )
                    return reply


def build_builtin_tool_registry(
    *,
    timeout_seconds: float,
    max_result_chars: int,
    now_factory: NowFactory | None = None,
    web_search: WebSearchProvider | None = None,
    web_search_timeout_seconds: float | None = None,
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
        intent_matcher=_is_time_request,
        reply_formatter=_format_time_reply,
    )
    registry.register(
        ToolDefinition(
            name="calculate",
            description=(
                "Evaluate a basic arithmetic expression accurately. "
                "Supports +, -, *, /, //, %, **, and parentheses. "
                "The returned result.answer is authoritative: copy its "
                "Arabic digits exactly into the final answer without "
                "recalculating or converting them to Chinese numerals."
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
        intent_matcher=_is_calculation_request,
        reply_formatter=_format_calculation_reply,
    )
    if web_search is not None:
        registry.register(
            ToolDefinition(
                name="web_search",
                description=(
                    "Search the web for current or explicitly requested "
                    "information. Use the returned titles, snippets, dates, "
                    "and URLs as sources; do not invent facts or URLs."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A concise web search query.",
                        },
                        "time_range": {
                            "type": "string",
                            "enum": ["day", "month", "year"],
                            "description": (
                                "Optional freshness filter for the search."
                            ),
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            _build_web_search_handler(web_search),
            intent_matcher=_is_web_search_request,
            timeout_seconds=web_search_timeout_seconds,
            failure_reply="暂时无法联网查询，请稍后再试。",
            source_extractor=_extract_web_search_sources,
        )
    return registry


_TIME_REQUEST = re.compile(
    r"(?:现在|当前|此刻|今天).{0,8}(?:几点|时间|日期|几号|星期|礼拜)"
    r"|(?:几点了|现在几点|今天几号|今天星期几|今天礼拜几)"
)
_CALCULATION_NUMBER = re.compile(
    r"[0-9０-９零〇一二三四五六七八九十百千万亿两]"
)
_CALCULATION_OPERATION = re.compile(
    r"(?:加|减|乘|除|计算|等于|次方|平方|求和|百分之|[+*/%×÷])"
)
_WEB_SEARCH_REQUEST = re.compile(
    r"(?:联网|上网|网上).{0,8}(?:搜|查|搜索|查询)"
    r"|(?:搜一下|搜索|查一下|查一查|查询)"
    r"|(?:最新|最近|目前|当前).{0,24}"
    r"(?:新闻|消息|动态|进展|情况|价格|行情|天气|比赛|政策|版本|发布)"
    r"|(?:今天|昨日|昨天|本周).{0,24}"
    r"(?:新闻|消息|天气|发生|赛事|比赛)"
    r"|(?:天气|气温|降雨|空气质量)(?:怎么样|如何|多少|预报)?"
    r"|(?:比赛结果|最新比分|实时比分)"
    r"|(?:现任|目前|当前).{0,16}"
    r"(?:总统|总理|主席|CEO|首席执行官|负责人)"
)


def _last_user_text(messages: Sequence[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content.partition("\n\n")[0].strip()
    return ""


def _is_time_request(text: str) -> bool:
    return bool(_TIME_REQUEST.search(text))


def _is_calculation_request(text: str) -> bool:
    return bool(
        _CALCULATION_NUMBER.search(text)
        and _CALCULATION_OPERATION.search(text)
    )


def _is_web_search_request(text: str) -> bool:
    return bool(_WEB_SEARCH_REQUEST.search(text))


def _format_time_reply(result: Any, question: str) -> str:
    if not isinstance(result, dict):
        raise ValueError("Time tool returned an invalid result")
    try:
        current = datetime.fromisoformat(str(result["iso8601"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("Time tool returned an invalid timestamp") from exc
    asks_date = bool(re.search(r"日期|几号", question))
    asks_weekday = bool(re.search(r"星期|礼拜", question))
    asks_time = bool(re.search(r"几点|时间", question))
    if asks_weekday and not (asks_date or asks_time):
        weekdays = "一二三四五六日"
        return f"今天是星期{weekdays[current.weekday()]}。"
    if asks_date and not (asks_weekday or asks_time):
        return f"今天是{current.year}年{current.month}月{current.day}日。"
    if asks_time and not (asks_date or asks_weekday):
        if current.minute == 0:
            return f"现在是{current.hour}点整。"
        return f"现在是{current.hour}点{current.minute}分。"
    weekdays = "一二三四五六日"
    minute = "整" if current.minute == 0 else f"{current.minute}分"
    return (
        f"现在是{current.year}年{current.month}月{current.day}日，"
        f"星期{weekdays[current.weekday()]}，"
        f"{current.hour}点{minute}。"
    )


def _format_calculation_reply(result: Any, _question: str) -> str:
    if not isinstance(result, dict) or "answer" not in result:
        raise ValueError("Calculator returned an invalid result")
    return f"答案是{result['answer']}。"


def _build_web_search_handler(
    provider: WebSearchProvider,
) -> ToolHandler:
    def search_web(arguments: dict[str, Any]) -> dict[str, Any]:
        return provider.search(
            str(arguments["query"]),
            time_range=(
                str(arguments["time_range"])
                if "time_range" in arguments
                else None
            ),
        )

    return search_web


def _extract_web_search_sources(value: Any) -> tuple[SourceReference, ...]:
    if not isinstance(value, dict):
        return ()
    results = value.get("results")
    if not isinstance(results, list):
        return ()

    references: list[SourceReference] = []
    seen_urls: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        title = result.get("title")
        url = result.get("url")
        if not isinstance(title, str) or not isinstance(url, str):
            continue
        cleaned_title = title.strip()
        cleaned_url = url.strip()
        if not cleaned_title or not cleaned_url or cleaned_url in seen_urls:
            continue
        seen_urls.add(cleaned_url)
        references.append(
            SourceReference(title=cleaned_title, url=cleaned_url)
        )
        if len(references) >= 10:
            break
    return tuple(references)


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
    return {
        "expression": expression,
        "value": value,
        "answer": str(value),
    }


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
        coerced = _coerce_argument(value, expected_type, name)
        allowed_values = property_schema.get("enum")
        if isinstance(allowed_values, list) and coerced not in allowed_values:
            raise ValueError(
                f"Argument {name} must be one of: "
                + ", ".join(str(item) for item in allowed_values)
            )
        validated[name] = coerced
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
