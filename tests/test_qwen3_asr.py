from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from voice_assistant.providers.qwen3_asr import Qwen3ASR


class FakeInputIds:
    shape = (1, 3)


class FakeBatch(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__(input_ids=FakeInputIds())
        self.moved_to: tuple[Any, Any] | None = None

    def to(self, device: Any, dtype: Any) -> "FakeBatch":
        self.moved_to = (device, dtype)
        return self


class FakeGeneratedIds:
    def __init__(self) -> None:
        self.received_slice: Any = None

    def __getitem__(self, key: Any) -> str:
        self.received_slice = key
        return "generated-token-slice"


class FakeModel:
    device = "cuda:0"
    dtype = "bfloat16"

    def __init__(self) -> None:
        self.generated = FakeGeneratedIds()
        self.received_arguments: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> FakeGeneratedIds:
        self.received_arguments = kwargs
        return self.generated


class FakeProcessor:
    def __init__(self, decoded: Any = None) -> None:
        self.batch = FakeBatch()
        self.decoded = [" 今天天气怎么样？ "] if decoded is None else decoded
        self.request_arguments: dict[str, Any] = {}
        self.decode_arguments: dict[str, Any] = {}

    def apply_transcription_request(self, **kwargs: Any) -> FakeBatch:
        self.request_arguments = kwargs
        return self.batch

    def decode(self, token_ids: Any, **kwargs: Any) -> Any:
        self.decode_arguments = {
            "token_ids": token_ids,
            **kwargs,
        }
        return self.decoded


class Qwen3ASRTest(unittest.TestCase):
    def test_transcribes_with_auto_language(self) -> None:
        model = FakeModel()
        processor = FakeProcessor()
        provider = Qwen3ASR(
            model_name="Qwen/Qwen3-ASR-0.6B-hf",
            language="auto",
            max_new_tokens=256,
            model=model,
            processor=processor,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "question.wav"
            audio_path.touch()
            transcript = provider.transcribe(audio_path)

        self.assertEqual(transcript, "今天天气怎么样？")
        self.assertEqual(processor.request_arguments["audio"], str(audio_path))
        self.assertIsNone(processor.request_arguments["language"])
        self.assertNotIn("prompt", processor.request_arguments)
        self.assertEqual(
            processor.batch.moved_to,
            (model.device, model.dtype),
        )
        self.assertEqual(model.received_arguments["max_new_tokens"], 256)
        self.assertEqual(
            processor.decode_arguments["token_ids"],
            "generated-token-slice",
        )
        self.assertEqual(
            processor.decode_arguments["return_format"],
            "transcription_only",
        )

    def test_passes_language_and_context_prompt(self) -> None:
        processor = FakeProcessor()
        provider = Qwen3ASR(
            model_name="test-model",
            language="Chinese",
            prompt="Vocabulary: Qwen3-ASR, Qwen3.5.",
            model=FakeModel(),
            processor=processor,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "question.wav"
            audio_path.touch()
            provider.transcribe(audio_path)

        self.assertEqual(processor.request_arguments["language"], "Chinese")
        self.assertEqual(
            processor.request_arguments["prompt"],
            "Vocabulary: Qwen3-ASR, Qwen3.5.",
        )

    def test_rejects_empty_transcript(self) -> None:
        provider = Qwen3ASR(
            model_name="test-model",
            model=FakeModel(),
            processor=FakeProcessor(decoded=["  "]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "question.wav"
            audio_path.touch()
            with self.assertRaisesRegex(RuntimeError, "empty transcript"):
                provider.transcribe(audio_path)

    def test_rejects_missing_audio(self) -> None:
        provider = Qwen3ASR(
            model_name="test-model",
            model=FakeModel(),
            processor=FakeProcessor(),
        )

        with self.assertRaises(FileNotFoundError):
            provider.transcribe(Path("missing.wav"))

    def test_rejects_invalid_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "compute_dtype"):
            Qwen3ASR(
                model_name="test-model",
                compute_dtype="int4",
                model=FakeModel(),
                processor=FakeProcessor(),
            )
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            Qwen3ASR(
                model_name="test-model",
                max_new_tokens=0,
                model=FakeModel(),
                processor=FakeProcessor(),
            )


if __name__ == "__main__":
    unittest.main()
