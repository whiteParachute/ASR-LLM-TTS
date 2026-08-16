import tempfile
import unittest
from pathlib import Path

from voice_assistant.providers.edge_tts_provider import EdgeTTSProvider


class FakeCommunicator:
    def __init__(self, text: str, voice: str) -> None:
        self.text = text
        self.voice = voice
        self.saved_path: str | None = None

    async def save(self, output_file: str) -> None:
        self.saved_path = output_file
        Path(output_file).write_bytes(b"fake audio")


class FakeCommunicatorFactory:
    def __init__(self) -> None:
        self.communicator: FakeCommunicator | None = None

    def __call__(self, text: str, voice: str) -> FakeCommunicator:
        self.communicator = FakeCommunicator(text=text, voice=voice)
        return self.communicator


class EdgeTTSProviderTest(unittest.TestCase):
    def test_synthesizes_audio_file(self) -> None:
        factory = FakeCommunicatorFactory()
        provider = EdgeTTSProvider(default_voice="zh-CN-XiaoyiNeural", communicator_factory=factory)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = (
                Path(temp_dir)
                / "nested"
                / "answer.mp3"
            )

            result = provider.synthesize(
                text="你好",
                output_path=output_path
            )

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.is_file())

            self.assertIsNotNone(factory.communicator)
            self.assertEqual(factory.communicator.text, "你好")
            self.assertEqual(factory.communicator.voice, "zh-CN-XiaoyiNeural")
            self.assertEqual(factory.communicator.saved_path, str(output_path))

    def test_rejects_empty_text(self) -> None:
        provider = EdgeTTSProvider(default_voice="zh-CN-XiaoyiNeural", communicator_factory=FakeCommunicatorFactory())

        with self.assertRaises(ValueError):
            provider.synthesize(
                text="  ",
                output_path=Path("answer.mp3")
            )


if __name__ == "__main__":
    unittest.main()