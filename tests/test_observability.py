import io
import json
import tempfile
import unittest
import wave
from datetime import datetime, timezone
from pathlib import Path

from voice_assistant.observability import (
    PerformanceLogger,
    wav_duration_ms,
)


class PerformanceLoggerTest(unittest.TestCase):
    def test_writes_structured_success_event_and_console_line(self) -> None:
        ticks = iter([10.0, 10.125])
        console = io.StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = PerformanceLogger(
                enabled=True,
                console=True,
                jsonl=True,
                log_dir=Path(temp_dir),
                clock=lambda: next(ticks),
                timestamp_factory=lambda: datetime(
                    2026,
                    8,
                    17,
                    tzinfo=timezone.utc,
                ),
                session_id="session-test",
                console_stream=console,
            )

            with logger.turn("turn_0001"):
                with logger.measure(
                    "asr",
                    audio_duration_ms=500.0,
                ) as span:
                    span.add_fields(output_chars=4)
            logger.close()

            lines = (Path(temp_dir) / "performance.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()

        self.assertEqual(len(lines), 1)
        event = json.loads(lines[0])
        self.assertEqual(event["schema_version"], 1)
        self.assertEqual(event["session_id"], "session-test")
        self.assertEqual(event["turn_id"], "turn_0001")
        self.assertEqual(event["stage"], "asr")
        self.assertEqual(event["status"], "ok")
        self.assertEqual(event["duration_ms"], 125.0)
        self.assertEqual(event["audio_duration_ms"], 500.0)
        self.assertEqual(event["rtf"], 0.25)
        self.assertEqual(event["output_chars"], 4)
        self.assertNotIn("transcript", event)
        self.assertIn("stage=asr", console.getvalue())

    def test_records_error_type_and_reraises_exception(self) -> None:
        ticks = iter([1.0, 1.01])

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = PerformanceLogger(
                enabled=True,
                console=False,
                jsonl=True,
                log_dir=Path(temp_dir),
                clock=lambda: next(ticks),
                session_id="session-error",
            )

            with self.assertRaisesRegex(RuntimeError, "failed"):
                with logger.turn("turn_0002"):
                    with logger.measure("llm"):
                        raise RuntimeError("failed")
            logger.close()

            event = json.loads(
                (Path(temp_dir) / "performance.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )

        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error_type"], "RuntimeError")
        self.assertNotIn("failed", json.dumps(event))

    def test_disabled_logger_creates_no_log_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "disabled"
            logger = PerformanceLogger(
                enabled=False,
                console=True,
                jsonl=True,
                log_dir=log_dir,
            )

            with logger.measure("asr"):
                pass
            logger.close()

            self.assertFalse(log_dir.exists())

    def test_reads_pcm_wav_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "one_second.wav"
            with wave.open(str(audio_path), "wb") as audio_file:
                audio_file.setnchannels(1)
                audio_file.setsampwidth(2)
                audio_file.setframerate(16000)
                audio_file.writeframes(b"\x00\x00" * 16000)

            self.assertEqual(wav_duration_ms(audio_path), 1000.0)


if __name__ == "__main__":
    unittest.main()
