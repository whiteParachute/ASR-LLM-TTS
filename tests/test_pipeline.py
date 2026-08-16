import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from voice_assistant.contracts import AudioChunk, Message, PipelineResult
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


class FakeStreamingTTS(FakeTTS):
    def stream_synthesize(self, text: str):
        self.received_text = text
        yield AudioChunk(b"\x01\x00\x02\x00", sample_rate=24000)
        yield AudioChunk(b"\x03\x00\x04\x00", sample_rate=24000)


class VoicePipelineTest(unittest.TestCase):
    def test_pipeline_result_defaults_to_primary_audio_path(self) -> None:
        audio_path = Path("answer.wav")

        result = PipelineResult(
            transcript="问题",
            reply="回答",
            audio_path=audio_path,
        )

        self.assertEqual(result.audio_paths, (audio_path,))

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
        self.assertEqual(result.audio_paths, (expected_path,))

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

    def test_can_prepare_text_before_synthesizing_audio(self) -> None:
        tts = FakeTTS()
        pipeline = VoicePipeline(
            asr=FakeASR(transcript="问题"),
            llm=FakeLLM(reply="先得到文字，再生成语音。"),
            tts=tts,
            system_prompt="测试助手",
        )

        prepared = pipeline.prepare(Path("question.wav"))

        self.assertEqual(prepared.transcript, "问题")
        self.assertEqual(prepared.reply, "先得到文字，再生成语音。")
        self.assertEqual(tts.received_text, "")

    def test_streams_tts_chunks_when_provider_supports_it(self) -> None:
        tts = FakeStreamingTTS()
        pipeline = VoicePipeline(
            asr=FakeASR(transcript="问题"),
            llm=FakeLLM(reply="流式回答"),
            tts=tts,
            system_prompt="测试助手",
        )

        chunks = list(pipeline.stream_synthesize("流式回答"))

        self.assertTrue(pipeline.supports_streaming_tts)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(tts.received_text, "流式回答")

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
