import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from voice_assistant.contracts import PipelineResult
from voice_assistant.realtime import RealtimeVoiceAssistant


class FakeRecorder:
    def __init__(self) -> None:
        self.output_path: Path | None = None

    def record(self, output_path: Path) -> Path:
        self.output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake input")
        return output_path


class FakePipeline:
    def __init__(self) -> None:
        self.audio_path: Path | None = None
        self.output_path: Path | None = None

    def run(self, audio_path: Path, output_path: Path) -> PipelineResult:
        self.audio_path = audio_path
        self.output_path = output_path
        output_path.write_bytes(b"fake reply")
        return PipelineResult(
            transcript="你好",
            reply="你好，有什么可以帮你？",
            audio_path=output_path,
        )


class FakePlayer:
    def __init__(self) -> None:
        self.audio_path: Path | None = None

    def play(self, audio_path: Path) -> None:
        self.audio_path = audio_path


class RealtimeVoiceAssistantTest(unittest.TestCase):
    def test_continues_after_recoverable_turn_error(self) -> None:
        assistant = RealtimeVoiceAssistant(
            pipeline=FakePipeline(),
            recorder=FakeRecorder(),
            player=FakePlayer(),
            output_dir=Path("turns"),
        )
        assistant.run_turn = Mock(
            side_effect=[RuntimeError("empty transcript"), KeyboardInterrupt],
        )

        with self.assertRaises(KeyboardInterrupt):
            assistant.run_forever()

        self.assertEqual(assistant.run_turn.call_count, 2)

    def test_runs_record_inference_and_playback_in_order(self) -> None:
        recorder = FakeRecorder()
        pipeline = FakePipeline()
        player = FakePlayer()
        observed: list[PipelineResult] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "turns"
            assistant = RealtimeVoiceAssistant(
                pipeline=pipeline,
                recorder=recorder,
                player=player,
                output_dir=output_dir,
                output_format="wav",
                result_observer=observed.append,
            )

            result = assistant.run_turn()

            expected_input = output_dir / "turn_0001_input.wav"
            expected_reply = output_dir / "turn_0001_reply.wav"
            self.assertEqual(recorder.output_path, expected_input)
            self.assertEqual(pipeline.audio_path, expected_input)
            self.assertEqual(pipeline.output_path, expected_reply)
            self.assertEqual(player.audio_path, expected_reply)

        self.assertEqual(result.transcript, "你好")
        self.assertEqual(observed, [result])


if __name__ == "__main__":
    unittest.main()
