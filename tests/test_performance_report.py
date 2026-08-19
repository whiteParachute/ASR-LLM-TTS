import tempfile
import unittest
from pathlib import Path

from voice_assistant.performance_report import (
    build_report,
    format_report,
    load_events,
)


def event(
    session_id: str,
    turn_id: str | None,
    stage: str,
    duration_ms: float,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "stage": stage,
        "duration_ms": duration_ms,
        "status": status,
    }


class PerformanceReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events = [
            event("old", None, "model_load", 20000),
            event("old", "turn_0001", "turn_total", 5000),
            event("latest", None, "model_load", 18000),
            event("latest", "turn_0001", "asr", 100),
            event("latest", "turn_0001", "llm", 200),
            event("latest", "turn_0001", "response_prepare", 300),
            event("latest", "turn_0001", "time_to_first_audio", 700),
            event("latest", "turn_0001", "turn_total", 1000),
            event("latest", "turn_0002", "asr", 200),
            event("latest", "turn_0002", "llm", 200),
            event("latest", "turn_0002", "response_prepare", 400),
            event("latest", "turn_0002", "time_to_first_audio", 900),
            event("latest", "turn_0002", "turn_total", 1400),
            event("latest", "turn_0003", "asr", 900, status="error"),
            event("latest", "turn_0003", "turn_total", 900, status="error"),
            event("newer-failed", "turn_0001", "record", 100, "error"),
        ]

    def test_reports_latest_successful_session(self) -> None:
        report = build_report(self.events)

        self.assertEqual(report.session_id, "latest")
        self.assertEqual(report.successful_turns, 2)
        self.assertEqual(report.model_load_ms, 18000)
        self.assertEqual(report.stages["asr"].median_ms, 150)
        self.assertEqual(report.stages["asr"].p95_ms, 200)
        self.assertEqual(
            report.stages["tts_first_chunk"].median_ms,
            450,
        )

    def test_can_exclude_first_successful_turn(self) -> None:
        report = build_report(self.events, warmed_only=True)

        self.assertEqual(report.successful_turns, 1)
        self.assertEqual(report.stages["asr"].median_ms, 200)
        self.assertEqual(
            report.stages["tts_first_chunk"].median_ms,
            500,
        )

    def test_rejects_session_without_successful_turns(self) -> None:
        with self.assertRaisesRegex(ValueError, "no successful"):
            build_report(
                [event("failed", "turn_0001", "turn_total", 1, "error")]
            )

    def test_loads_and_formats_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "performance.jsonl"
            log_path.write_text(
                "\n".join(
                    [
                        '{"session_id":"one","turn_id":"turn_0001",'
                        '"stage":"turn_total","duration_ms":1000,'
                        '"status":"ok"}',
                    ]
                ),
                encoding="utf-8",
            )
            report = build_report(load_events(log_path))

        output = format_report(report, warmed_only=False)
        self.assertIn("session: one", output)
        self.assertIn("turn_total", output)


if __name__ == "__main__":
    unittest.main()
