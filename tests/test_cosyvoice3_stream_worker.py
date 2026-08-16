import base64
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock

from voice_assistant.providers.cosyvoice3_stream_worker import (
    CosyVoice3StreamingWorkerProvider,
)


class FakeWorkerProcess:
    def __init__(self, messages: list[dict]) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(
            "".join(
                json.dumps(message, ensure_ascii=False) + "\n"
                for message in messages
            )
        )
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def audio_message(request_id: int, pcm: bytes) -> dict:
    return {
        "id": request_id,
        "event": "audio_chunk",
        "sample_rate": 24000,
        "channels": 1,
        "pcm_s16le_base64": base64.b64encode(pcm).decode("ascii"),
    }


class CosyVoice3StreamingWorkerProviderTest(unittest.TestCase):
    def test_streams_chunks_and_writes_complete_wav(self) -> None:
        messages = [
            {"event": "ready", "sample_rate": 24000},
            audio_message(1, b"\x01\x00\x02\x00"),
            audio_message(1, b"\x03\x00\x04\x00"),
            {"id": 1, "event": "complete"},
        ]
        process = FakeWorkerProcess(messages)
        process_factory = Mock(return_value=process)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            runtime_entrypoint = (
                temp_path / "runtime/cosyvoice/cli/cosyvoice.py"
            )
            runtime_entrypoint.parent.mkdir(parents=True)
            runtime_entrypoint.write_text("# fake", encoding="utf-8")
            reference_audio = temp_path / "reference.wav"
            reference_audio.write_bytes(b"fake")
            output_path = temp_path / "reply.wav"
            provider = CosyVoice3StreamingWorkerProvider(
                model_name="models/cosyvoice3",
                runtime_dir=temp_path / "runtime",
                reference_audio=reference_audio,
                reference_text="参考文本",
                process_factory=process_factory,
            )

            result = provider.synthesize("你好", output_path)

            with wave.open(str(result), "rb") as audio_file:
                self.assertEqual(audio_file.getframerate(), 24000)
                self.assertEqual(audio_file.getnchannels(), 1)
                self.assertEqual(audio_file.readframes(2), b"\x01\x00\x02\x00")
                self.assertEqual(audio_file.readframes(2), b"\x03\x00\x04\x00")
            provider.close()

        command = process_factory.call_args.args[0]
        self.assertIn("--runtime-dir", command)
        self.assertIn("--fp16", command)
        requests = [
            json.loads(line)
            for line in process.stdin.getvalue().splitlines()
        ]
        self.assertEqual(requests[0]["op"], "stream_synthesize")
        self.assertEqual(requests[0]["text"], "你好")
        self.assertEqual(requests[-1]["op"], "shutdown")

    def test_rejects_empty_synthesis_text(self) -> None:
        messages = [{"event": "ready", "sample_rate": 24000}]
        process = FakeWorkerProcess(messages)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            runtime_entrypoint = (
                temp_path / "runtime/cosyvoice/cli/cosyvoice.py"
            )
            runtime_entrypoint.parent.mkdir(parents=True)
            runtime_entrypoint.write_text("# fake", encoding="utf-8")
            reference_audio = temp_path / "reference.wav"
            reference_audio.write_bytes(b"fake")
            provider = CosyVoice3StreamingWorkerProvider(
                model_name="model",
                runtime_dir=temp_path / "runtime",
                reference_audio=reference_audio,
                reference_text="参考文本",
                process_factory=Mock(return_value=process),
            )

            with self.assertRaises(ValueError):
                list(provider.stream_synthesize("  "))
            provider.close()


if __name__ == "__main__":
    unittest.main()
