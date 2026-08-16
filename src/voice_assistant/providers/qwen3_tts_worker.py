from __future__ import annotations

import atexit
import json
import queue
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Protocol


class WorkerProcess(Protocol):
    stdin: IO[str] | None
    stdout: IO[str] | None
    returncode: int | None

    def poll(self) -> int | None:
        ...

    def terminate(self) -> None:
        ...

    def wait(self, timeout: float | None = None) -> int:
        ...

    def kill(self) -> None:
        ...


ProcessFactory = Callable[..., WorkerProcess]


class Qwen3TTSWorkerProvider:
    """Use Qwen3-TTS through a persistent, dependency-isolated worker."""

    def __init__(
        self,
        model_name: str,
        reference_audio: Path | None,
        reference_text: str,
        language: str = "Chinese",
        device: str | None = None,
        worker_python: str = ".venv-tts/bin/python",
        worker_script: str = "scripts/qwen3_tts_worker.py",
        x_vector_only_mode: bool = False,
        dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        startup_timeout_seconds: float = 180.0,
        process_factory: ProcessFactory = subprocess.Popen,
    ) -> None:
        if reference_audio is None:
            raise ValueError("Qwen3-TTS reference_audio is required")
        if not x_vector_only_mode and not reference_text.strip():
            raise ValueError(
                "Qwen3-TTS reference_text is required when "
                "x_vector_only_mode is false"
            )
        if startup_timeout_seconds <= 0:
            raise ValueError("Qwen3-TTS startup timeout must be positive")

        reference_path = reference_audio.expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Qwen3-TTS reference audio does not exist: {reference_path}"
            )

        command = [
            str(Path(worker_python).expanduser()),
            str(Path(worker_script).expanduser()),
            "--model",
            model_name,
            "--reference-audio",
            str(reference_path),
            "--reference-text",
            reference_text,
            "--language",
            language,
            "--dtype",
            dtype,
            "--attention-implementation",
            attention_implementation,
        ]
        if device:
            command.extend(("--device", device))
        if x_vector_only_mode:
            command.append("--x-vector-only-mode")

        self._lock = threading.Lock()
        self._request_id = 0
        self._closed = False
        self._process = process_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise RuntimeError("Qwen3-TTS worker pipes are unavailable")

        try:
            ready = self._read_message(startup_timeout_seconds)
            if ready.get("event") != "ready":
                raise RuntimeError(self._format_worker_error(ready))
        except BaseException:
            self.close()
            raise

        atexit.register(self.close)

    def synthesize(self, text: str, output_path: Path) -> Path:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("TTS text cannot be empty")

        resolved_output = output_path.expanduser().resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if self._closed or self._process.poll() is not None:
                raise RuntimeError("Qwen3-TTS worker is not running")

            self._request_id += 1
            request_id = self._request_id
            self._write_message(
                {
                    "id": request_id,
                    "op": "synthesize",
                    "text": cleaned_text,
                    "output_path": str(resolved_output),
                }
            )
            response = self._read_message(timeout=None)

        if response.get("id") != request_id:
            raise RuntimeError(
                "Qwen3-TTS worker returned a mismatched response id"
            )
        if not response.get("ok"):
            raise RuntimeError(self._format_worker_error(response))
        if not resolved_output.is_file():
            raise RuntimeError(
                f"Qwen3-TTS did not create output file: {resolved_output}"
            )
        return resolved_output

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        process = self._process

        if process.poll() is None:
            try:
                self._write_message({"op": "shutdown"})
                process.wait(timeout=5.0)
            except (
                BrokenPipeError,
                OSError,
                ValueError,
                subprocess.TimeoutExpired,
            ):
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)

    def _write_message(self, message: dict[str, Any]) -> None:
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("Qwen3-TTS worker stdin is unavailable")
        stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        stdin.flush()

    def _read_message(self, timeout: float | None) -> dict[str, Any]:
        stdout = self._process.stdout
        if stdout is None:
            raise RuntimeError("Qwen3-TTS worker stdout is unavailable")

        result: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

        def read_line() -> None:
            try:
                result.put(stdout.readline())
            except BaseException as exc:
                result.put(exc)

        reader = threading.Thread(target=read_line, daemon=True)
        reader.start()
        try:
            value = result.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(
                "Timed out while waiting for the Qwen3-TTS worker"
            ) from exc

        if isinstance(value, BaseException):
            raise RuntimeError("Unable to read from Qwen3-TTS worker") from value
        if not value:
            raise RuntimeError(
                "Qwen3-TTS worker exited before returning a response "
                f"(exit_code={self._process.poll()})"
            )
        try:
            message = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid response from Qwen3-TTS worker: {value.rstrip()}"
            ) from exc
        if not isinstance(message, dict):
            raise RuntimeError("Qwen3-TTS worker response must be an object")
        return message

    @staticmethod
    def _format_worker_error(message: dict[str, Any]) -> str:
        detail = message.get("error", "unknown worker error")
        return f"Qwen3-TTS worker failed: {detail}"
