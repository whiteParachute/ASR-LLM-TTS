from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPORTED_STAGES = (
    "asr",
    "llm",
    "response_prepare",
    "tts_first_chunk",
    "time_to_first_audio",
    "tts_stream",
    "playback_stream",
    "turn_total",
)


@dataclass(frozen=True, slots=True)
class StageSummary:
    count: int
    median_ms: float
    p95_ms: float


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    session_id: str
    successful_turns: int
    model_load_ms: float | None
    stages: dict[str, StageSummary]


def load_events(log_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Unable to read performance log: {log_path}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on performance log line {line_number}"
            ) from exc
        if isinstance(event, dict):
            events.append(event)
    if not events:
        raise ValueError(f"Performance log is empty: {log_path}")
    return events


def build_report(
    events: Sequence[dict[str, Any]],
    *,
    session_id: str | None = None,
    warmed_only: bool = False,
) -> PerformanceReport:
    selected_session = session_id or _latest_session_id(events)
    session_events = [
        event
        for event in events
        if event.get("session_id") == selected_session
    ]
    if not session_events:
        raise ValueError(f"Session not found: {selected_session}")

    successful_turn_ids = _successful_turn_ids(session_events)
    if warmed_only and successful_turn_ids:
        successful_turn_ids = successful_turn_ids[1:]
    if not successful_turn_ids:
        qualifier = " warmed" if warmed_only else ""
        raise ValueError(
            f"Session {selected_session} has no successful{qualifier} turns"
        )

    selected_turns = set(successful_turn_ids)
    durations: dict[str, list[float]] = {
        stage: [] for stage in REPORTED_STAGES
    }
    per_turn: dict[str, dict[str, float]] = {}
    for event in session_events:
        turn_id = event.get("turn_id")
        stage = event.get("stage")
        duration = event.get("duration_ms")
        if (
            turn_id not in selected_turns
            or event.get("status") != "ok"
            or not isinstance(stage, str)
            or not isinstance(duration, (int, float))
        ):
            continue
        per_turn.setdefault(str(turn_id), {})[stage] = float(duration)
        if stage in durations and stage != "tts_first_chunk":
            durations[stage].append(float(duration))

    for turn_id in successful_turn_ids:
        turn_stages = per_turn.get(turn_id, {})
        first_audio = turn_stages.get("time_to_first_audio")
        preparation = turn_stages.get("response_prepare")
        if first_audio is not None and preparation is not None:
            durations["tts_first_chunk"].append(
                max(first_audio - preparation, 0.0)
            )

    model_load_ms = next(
        (
            float(event["duration_ms"])
            for event in reversed(session_events)
            if event.get("stage") == "model_load"
            and event.get("status") == "ok"
            and isinstance(event.get("duration_ms"), (int, float))
        ),
        None,
    )
    return PerformanceReport(
        session_id=selected_session,
        successful_turns=len(successful_turn_ids),
        model_load_ms=model_load_ms,
        stages={
            stage: _summarize(values)
            for stage, values in durations.items()
            if values
        },
    )


def format_report(report: PerformanceReport, *, warmed_only: bool) -> str:
    scope = "warmed turns" if warmed_only else "successful turns"
    lines = [
        f"session: {report.session_id}",
        f"scope: {scope}",
        f"turns: {report.successful_turns}",
    ]
    if report.model_load_ms is not None:
        lines.append(f"model_load_ms: {report.model_load_ms:.1f}")
    lines.extend(
        [
            "",
            f"{'stage':<24} {'count':>5} {'median_ms':>11} {'p95_ms':>11}",
        ]
    )
    for stage in REPORTED_STAGES:
        summary = report.stages.get(stage)
        if summary is None:
            continue
        lines.append(
            f"{stage:<24} {summary.count:>5} "
            f"{summary.median_ms:>11.1f} {summary.p95_ms:>11.1f}"
        )
    return "\n".join(lines)


def _latest_session_id(events: Sequence[dict[str, Any]]) -> str:
    for event in reversed(events):
        session_id = event.get("session_id")
        if (
            event.get("stage") == "turn_total"
            and event.get("status") == "ok"
            and isinstance(session_id, str)
            and session_id
        ):
            return session_id
    raise ValueError("Performance log contains no successful sessions")


def _successful_turn_ids(events: Sequence[dict[str, Any]]) -> list[str]:
    turn_ids: list[str] = []
    for event in events:
        turn_id = event.get("turn_id")
        if (
            event.get("stage") == "turn_total"
            and event.get("status") == "ok"
            and isinstance(turn_id, str)
            and turn_id not in turn_ids
        ):
            turn_ids.append(turn_id)
    return turn_ids


def _summarize(values: Sequence[float]) -> StageSummary:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return StageSummary(
        count=len(ordered),
        median_ms=round(statistics.median(ordered), 3),
        p95_ms=round(ordered[p95_index], 3),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize one voice-assistant performance session.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/wsl/performance.jsonl"),
        help="Path to the performance JSONL file.",
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--warmed-only",
        action="store_true",
        help="Exclude the first successful turn from the report.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = build_report(
            load_events(args.log),
            session_id=args.session_id,
            warmed_only=args.warmed_only,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(format_report(report, warmed_only=args.warmed_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
