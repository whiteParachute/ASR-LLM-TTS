import tempfile
import unittest
from pathlib import Path
from typing import Any

from voice_assistant.providers.sensevoice_asr import SenseVoiceASR

class FakeSenseVoiceModel:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.received_arguments: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> Any:
        self.received_arguments = kwargs
        return self.result


class SenseVoiceASRTest(unittest.TestCase):
    def test_transcribes_audio_and_removes_tags(self) -> None:
        # Arrange
        fake_model = FakeSenseVoiceModel(
            [
                {
                    "text": (
                        "<|zh|><|NEUTRAL|>"
                        "<|Speech|><|woitn|>"
                        "今天天气怎么样？"
                    )
                }
            ]
        )

        provider = SenseVoiceASR(
            model_name="iic/SenseVoiceSmall",
            language="auto",
            use_itn=False,
            model=fake_model,
        )

        # Act
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "question.wav"
            audio_path.touch()  # Create an empty file for testing
            transcript = provider.transcribe(audio_path)

        self.assertEqual(transcript, "今天天气怎么样？")
        self.assertEqual(fake_model.received_arguments["input"], str(audio_path))
        self.assertEqual(fake_model.received_arguments["language"], "auto")
        self.assertFalse(fake_model.received_arguments["use_itn"])
        self.assertEqual(fake_model.received_arguments["cache"], {})

    def test_rejects_empty_model_result(self) -> None:
        # Arrange
        provider = SenseVoiceASR(
            model_name="iic/SenseVoiceSmall",
            model=FakeSenseVoiceModel(result=[]),  # Simulate empty result
        )

        # Act & Assert
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "question.wav"
            audio_path.touch()  # Create an empty file for testing
            with self.assertRaises(RuntimeError):
                provider.transcribe(audio_path)


if __name__ == "__main__":
    unittest.main()