from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from typing import Any

from voice_assistant.providers.qwen3_tts_worker import (
    Qwen3TTSWorkerProvider,
)


class FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self.lines = deque(lines)

    def readline(self) -> str:
        return self.lines.popleft() if self.lines else ""


class FakeStdin(io.StringIO):
    def __init__(self, process: "FakeProcess", fail: bool = False) -> None:
        super().__init__()
        self.process = process
        self.fail = fail

    def write(self, value: str) -> int:
        request = json.loads(value)
        if request.get("op") == "synthesize":
            output_path = Path(request["output_path"])
            if self.fail:
                response = {
                    "id": request["id"],
                    "ok": False,
                    "error": "generation failed",
                }
            else:
                output_path.write_bytes(b"RIFFfake-wave")
                response = {
                    "id": request["id"],
                    "ok": True,
                    "output_path": str(output_path),
                }
            self.process.stdout.lines.append(json.dumps(response) + "\n")
        elif request.get("op") == "shutdown":
            self.process.returncode = 0
        return super().write(value)


class FakeProcess:
    def __init__(self, fail: bool = False) -> None:
        self.stdout = FakeStdout(['{"event": "ready"}\n'])
        self.stdin = FakeStdin(self, fail=fail)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class FakeProcessFactory:
    def __init__(self, fail: bool = False) -> None:
        self.process = FakeProcess(fail=fail)
        self.command: list[str] | None = None
        self.options: dict[str, Any] | None = None

    def __call__(
        self,
        command: list[str],
        **options: Any,
    ) -> FakeProcess:
        self.command = command
        self.options = options
        return self.process


class Qwen3TTSWorkerProviderTest(unittest.TestCase):
    def test_starts_worker_and_synthesizes_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.wav"
            reference.write_bytes(b"RIFFreference")
            factory = FakeProcessFactory()

            provider = Qwen3TTSWorkerProvider(
                model_name="models/qwen-tts",
                reference_audio=reference,
                reference_text="参考文本",
                process_factory=factory,
            )
            result = provider.synthesize(" 你好 ", root / "answer.wav")
            provider.close()

            self.assertTrue(result.is_file())
            self.assertIn("--reference-audio", factory.command or [])
            self.assertEqual(factory.process.returncode, 0)

    def test_surfaces_worker_generation_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.wav"
            reference.write_bytes(b"RIFFreference")
            provider = Qwen3TTSWorkerProvider(
                model_name="models/qwen-tts",
                reference_audio=reference,
                reference_text="参考文本",
                process_factory=FakeProcessFactory(fail=True),
            )

            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                provider.synthesize("你好", root / "answer.wav")
            provider.close()

    def test_requires_reference_text_for_full_clone(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "reference.wav"
            reference.write_bytes(b"RIFFreference")

            with self.assertRaisesRegex(ValueError, "reference_text"):
                Qwen3TTSWorkerProvider(
                    model_name="models/qwen-tts",
                    reference_audio=reference,
                    reference_text="",
                    process_factory=FakeProcessFactory(),
                )

    def test_rejects_empty_synthesis_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "reference.wav"
            reference.write_bytes(b"RIFFreference")
            provider = Qwen3TTSWorkerProvider(
                model_name="models/qwen-tts",
                reference_audio=reference,
                reference_text="参考文本",
                process_factory=FakeProcessFactory(),
            )

            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                provider.synthesize("  ", Path(temp_dir) / "answer.wav")
            provider.close()


if __name__ == "__main__":
    unittest.main()
