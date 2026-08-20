from __future__ import annotations

import json
import logging
import wave
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO
from uuid import uuid4

from voice_assistant.config import ObservabilityConfig


Clock = Callable[[], float]
TimestampFactory = Callable[[], datetime]


@dataclass(slots=True)
class StageSpan:
    """Mutable metadata collected while a measured stage is running."""

    fields: dict[str, Any] = field(default_factory=dict)

    def add_fields(self, **fields: Any) -> None:
        self.fields.update(fields)


class _JSONEventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "performance_event")
        return json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
        )


class PerformanceLogger:
    """Write privacy-safe stage timings to the console and JSONL."""

    def __init__(
        self,
        *,
        enabled: bool,
        console: bool,
        jsonl: bool,
        log_dir: Path,
        clock: Clock = perf_counter,
        timestamp_factory: TimestampFactory | None = None,
        session_id: str | None = None,
        console_stream: TextIO | None = None,
    ) -> None:
        self.enabled = enabled
        self.session_id = session_id or uuid4().hex[:12]
        self.log_path: Path | None = None
        self._clock = clock
        self._timestamp_factory = timestamp_factory or _utc_now
        self._turn_id: ContextVar[str | None] = ContextVar(
            f"voice_assistant_turn_{self.session_id}",
            default=None,
        )
        self._logger = logging.Logger(
            f"voice_assistant.performance.{self.session_id}",
            level=logging.INFO,
        )
        self._logger.propagate = False

        if not enabled:
            return

        if console:
            console_handler = logging.StreamHandler(console_stream)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(console_handler)

        if jsonl:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = log_dir / "performance.jsonl"
            file_handler = logging.FileHandler(
                self.log_path,
                encoding="utf-8",
            )
            file_handler.setFormatter(_JSONEventFormatter())
            self._logger.addHandler(file_handler)

    @contextmanager
    def turn(self, turn_id: str) -> Iterator[None]:
        token = self._turn_id.set(turn_id)
        try:
            yield
        finally:
            self._turn_id.reset(token)

    @contextmanager
    def measure(
        self,
        stage: str,
        **fields: Any,
    ) -> Iterator[StageSpan]:
        span = StageSpan(dict(fields))
        if not self.enabled:
            yield span
            return

        started_at = self._clock()
        try:
            yield span
        except BaseException as exc:
            self._emit(
                stage=stage,
                duration_ms=(self._clock() - started_at) * 1000,
                status="error",
                fields={
                    **span.fields,
                    "error_type": type(exc).__name__,
                },
            )
            raise
        else:
            self._emit(
                stage=stage,
                duration_ms=(self._clock() - started_at) * 1000,
                status="ok",
                fields=span.fields,
            )

    def close(self) -> None:
        for handler in list(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)

    def _emit(
        self,
        *,
        stage: str,
        duration_ms: float,
        status: str,
        fields: dict[str, Any],
    ) -> None:
        rounded_duration = round(max(duration_ms, 0.0), 3)
        event: dict[str, Any] = {
            "schema_version": 1,
            "event": "stage_completed",
            "timestamp": self._timestamp_factory().isoformat(),
            "session_id": self.session_id,
            "turn_id": self._turn_id.get(),
            "stage": stage,
            "status": status,
            "duration_ms": rounded_duration,
        }
        event.update(_normalise_fields(fields, reserved=set(event)))

        audio_duration = event.get("audio_duration_ms")
        if (
            stage in {"asr", "tts", "tts_stream"}
            and isinstance(audio_duration, (int, float))
            and audio_duration > 0
        ):
            event["rtf"] = round(rounded_duration / audio_duration, 4)

        self._logger.info(
            _format_console_event(event),
            extra={"performance_event": event},
        )


def build_performance_logger(
    config: ObservabilityConfig,
) -> PerformanceLogger:
    return PerformanceLogger(
        enabled=config.enabled,
        console=config.console,
        jsonl=config.jsonl,
        log_dir=config.log_dir,
    )


@contextmanager
def measure_stage(
    performance: PerformanceLogger | None,
    stage: str,
    **fields: Any,
) -> Iterator[StageSpan]:
    if performance is None:
        yield StageSpan(dict(fields))
        return

    with performance.measure(stage, **fields) as span:
        yield span


def wav_duration_ms(audio_path: Path) -> float | None:
    """Return PCM WAV duration without introducing an audio dependency."""
    try:
        with wave.open(str(audio_path), "rb") as audio_file:
            frame_rate = audio_file.getframerate()
            if frame_rate <= 0:
                return None
            return round(
                audio_file.getnframes() / frame_rate * 1000,
                3,
            )
    except (OSError, EOFError, wave.Error):
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalise_fields(
    fields: dict[str, Any],
    *,
    reserved: set[str],
) -> dict[str, Any]:
    normalised: dict[str, Any] = {}
    for key, value in fields.items():
        if key in reserved or value is None:
            continue
        if isinstance(value, Path):
            normalised[key] = str(value)
        elif isinstance(value, (str, int, float, bool)):
            normalised[key] = value
        else:
            normalised[key] = str(value)
    return normalised


def _format_console_event(event: dict[str, Any]) -> str:
    turn_id = event.get("turn_id") or "-"
    parts = [
        "[性能]",
        f"turn={turn_id}",
        f"stage={event['stage']}",
        f"duration_ms={event['duration_ms']}",
        f"status={event['status']}",
    ]
    for key in (
        "audio_duration_ms",
        "rtf",
        "input_chars",
        "output_chars",
        "text_chars",
        "reply_chunks",
        "first_chunk_chars",
        "first_chunk_duration_ms",
        "chunk_index",
        "chunk_count",
        "sample_rate",
        "streaming_audio",
        "streaming_text",
        "target_chars",
        "max_rounds",
        "tool_name",
        "tool_round",
        "tool_rounds",
        "tool_calls",
        "tool_ok",
        "tool_error_type",
        "call_index",
        "route",
        "direct_reply",
        "max_chunk_gap_ms",
        "min_buffer_ahead_ms",
        "estimated_underflow_ms",
        "error_type",
    ):
        if key in event:
            parts.append(f"{key}={event[key]}")
    return " ".join(parts)
