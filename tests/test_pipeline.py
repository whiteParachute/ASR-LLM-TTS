import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from voice_assistant.contracts import Message
from voice_assistant.pipeline import VoicePipeline
from voice_assistant.observability import PerformanceLogger


class FakeASR:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.received_path: Path | None = None

    def transcribe(self, audio_path: Path) -> str:
        self.received_path = audio_path
        return self.transcript


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.received_messages: list[Message] = []

    def generate(self, messages: Sequence[Message]) -> str:
        self.received_messages = list(messages)
        return self.reply


class FakeTTS:
    def __init__(self) -> None:
        self.received_text = ""
        self.received_path: Path | None = None

    def synthesize(self, text: str, output_path: Path) -> Path:
        self.received_text = text
        self.received_path = output_path
        return output_path


class VoicePipelineTest(unittest.TestCase):
    def test_runs_asr_llm_tts_in_order(self) -> None:
        # Arrange
        expected_transcript = "今天天气怎么样？"
        expected_reply = "今天天气不错。"
        expected_path = Path("answer.mp3")

        asr = FakeASR(transcript=expected_transcript)
        llm = FakeLLM(reply=expected_reply)
        tts = FakeTTS()
        system_prompt = "你是一个中文语音助手。"
        reply_instructions = "回答保持在50字以内。"

        pipeline = VoicePipeline(
            asr=asr,
            llm=llm,
            tts=tts,
            system_prompt=system_prompt,
            reply_instructions=reply_instructions,
        )

        # Act
        result = pipeline.run(audio_path=Path("question.wav"), output_path=expected_path)

        # Assert
        self.assertEqual(result.transcript, expected_transcript)
        self.assertEqual(result.reply, expected_reply)
        self.assertEqual(result.audio_path, expected_path)

        self.assertEqual(asr.received_path, Path("question.wav"))
        self.assertEqual(tts.received_text, expected_reply)
        self.assertEqual(tts.received_path, expected_path)

        self.assertEqual(
            llm.received_messages,
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content="今天天气怎么样？\n\n回答保持在50字以内。"),
            ],
        )

    def test_stops_when_asr_returns_empty_text(self) -> None:
        # Arrange
        pipeline = VoicePipeline(
            asr=FakeASR(transcript=""),
            llm=FakeLLM(reply="不应该生成"),
            tts=FakeTTS(),
            system_prompt="测试助手",
        )

        with self.assertRaises(ValueError):
            pipeline.run(audio_path=Path("empty.wav"), output_path=Path("answer.mp3"))

    def test_records_asr_llm_and_tts_stage_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            performance = PerformanceLogger(
                enabled=True,
                console=False,
                jsonl=True,
                log_dir=Path(temp_dir),
                session_id="pipeline-session",
            )
            pipeline = VoicePipeline(
                asr=FakeASR(transcript="你好"),
                llm=FakeLLM(reply="你好呀"),
                tts=FakeTTS(),
                system_prompt="测试助手",
                performance=performance,
            )

            with performance.turn("turn_0003"):
                pipeline.run(
                    audio_path=Path("question.wav"),
                    output_path=Path("answer.wav"),
                )
            performance.close()

            events = [
                json.loads(line)
                for line in (Path(temp_dir) / "performance.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(
            [event["stage"] for event in events],
            ["asr", "llm", "tts"],
        )
        self.assertTrue(
            all(event["turn_id"] == "turn_0003" for event in events)
        )
        self.assertTrue(
            all("transcript" not in event for event in events)
        )


if __name__ == "__main__":
    unittest.main()
