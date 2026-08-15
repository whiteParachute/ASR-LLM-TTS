import tempfile
import unittest
from pathlib import Path
from typing import Any

from voice_assistant.providers.kokoro_tts_provider import KokoroTTSProvider


class FakeKokoroPipeline:
    def __init__(self, audio_chunks: list[Any] | None = None) -> None:
        self.audio_chunks = audio_chunks or []
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        split_pattern: str,
    ):
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "split_pattern": split_pattern,
            }
        )
        return iter(
            ("graphemes", "phonemes", audio)
            for audio in self.audio_chunks
        )


class FakeAudioWriter:
    def __init__(self) -> None:
        self.path: str | None = None
        self.audio: Any = None
        self.sample_rate: int | None = None

    def __call__(
        self,
        path: str,
        audio: Any,
        sample_rate: int,
    ) -> None:
        self.path = path
        self.audio = audio
        self.sample_rate = sample_rate
        Path(path).write_bytes(b"fake wav")


class KokoroTTSProviderTest(unittest.TestCase):
    def test_uses_voice_file_from_local_model_directory(self) -> None:
        pipeline = FakeKokoroPipeline(audio_chunks=[[0.1]])
        writer = FakeAudioWriter()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            voice_path = model_path / "voices" / "zf_xiaoxiao.pt"
            voice_path.parent.mkdir()
            voice_path.write_bytes(b"voice")
            provider = KokoroTTSProvider(
                model_name=str(model_path),
                language_code="z",
                default_voice="zf_xiaoxiao",
                pipeline=pipeline,
                audio_writer=writer,
                audio_concatenator=lambda chunks: chunks[0],
            )

            provider.synthesize("你好", model_path / "answer.wav")

            self.assertEqual(
                pipeline.calls[0]["voice"],
                str(voice_path),
            )

    def test_synthesizes_all_audio_chunks_into_one_file(self) -> None:
        pipeline = FakeKokoroPipeline(
            audio_chunks=[
                [0.1, 0.2],
                [0.3],
            ]
        )
        writer = FakeAudioWriter()
        provider = KokoroTTSProvider(
            model_name="hexgrad/Kokoro-82M",
            language_code="z",
            default_voice="zf_xiaoxiao",
            speed=1.1,
            pipeline=pipeline,
            audio_writer=writer,
            audio_concatenator=lambda chunks: [
                sample
                for chunk in chunks
                for sample in chunk
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "answer.wav"
            result = provider.synthesize(" 你好 ", output_path)

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.is_file())
            self.assertEqual(writer.path, str(output_path))

        self.assertEqual(writer.sample_rate, 24000)
        self.assertEqual(writer.audio, [0.1, 0.2, 0.3])
        self.assertEqual(
            pipeline.calls,
            [
                {
                    "text": "你好",
                    "voice": "zf_xiaoxiao",
                    "speed": 1.1,
                    "split_pattern": r"\n+",
                }
            ],
        )

    def test_rejects_empty_text(self) -> None:
        pipeline = FakeKokoroPipeline()
        provider = KokoroTTSProvider(
            model_name="hexgrad/Kokoro-82M",
            language_code="z",
            default_voice="zf_xiaoxiao",
            pipeline=pipeline,
            audio_writer=FakeAudioWriter(),
            audio_concatenator=lambda chunks: chunks,
        )

        with self.assertRaises(ValueError):
            provider.synthesize("   ", Path("answer.wav"))

        self.assertEqual(pipeline.calls, [])

    def test_rejects_empty_model_output(self) -> None:
        provider = KokoroTTSProvider(
            model_name="hexgrad/Kokoro-82M",
            language_code="z",
            default_voice="zf_xiaoxiao",
            pipeline=FakeKokoroPipeline(),
            audio_writer=FakeAudioWriter(),
            audio_concatenator=lambda chunks: chunks,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                RuntimeError,
                "Kokoro returned no audio",
            ):
                provider.synthesize(
                    "你好",
                    Path(temp_dir) / "answer.wav",
                )


if __name__ == "__main__":
    unittest.main()
