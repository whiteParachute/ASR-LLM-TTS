from __future__ import annotations

import atexit
import base64
import json
import queue
import subprocess
import threading
import wave
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, Any, Protocol

from voice_assistant.contracts import AudioChunk


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


class CosyVoice3StreamingWorkerProvider:
    """Run CosyVoice3 in an isolated worker and yield PCM audio chunks."""

    def __init__(
        self,
        model_name: str,
        runtime_dir: Path,
        reference_audio: Path | None,
        reference_text: str,
        worker_python: str = ".venv-cosyvoice/bin/python",
        worker_script: str = "scripts/cosyvoice3_stream_worker.py",
        fp16: bool = True,
        warmup_text: str = "你好，很高兴和你对话。",
        startup_timeout_seconds: float = 300.0,
        inference_mode: str = "zero_shot",
        speaker: str = "",
        load_jit: bool = False,
        process_factory: ProcessFactory = subprocess.Popen,
    ) -> None:
        if inference_mode not in {"zero_shot", "sft"}:
            raise ValueError(
                f"Unsupported CosyVoice inference mode: {inference_mode}"
            )
        if inference_mode == "zero_shot":
            if reference_audio is None:
                raise ValueError("CosyVoice reference_audio is required")
            if not reference_text.strip():
                raise ValueError("CosyVoice reference_text is required")
        elif not speaker.strip():
            raise ValueError("CosyVoice SFT speaker is required")
        if startup_timeout_seconds <= 0:
            raise ValueError("CosyVoice startup timeout must be positive")

        runtime_path = runtime_dir.expanduser().resolve()
        runtime_entrypoint = runtime_path / "cosyvoice/cli/cosyvoice.py"
        if not runtime_entrypoint.is_file():
            raise FileNotFoundError(
                "CosyVoice3 runtime is missing; run "
                "scripts/install_wsl_cosyvoice_runtime.sh first: "
                f"{runtime_path}"
            )

        reference_path: Path | None = None
        if reference_audio is not None:
            reference_path = reference_audio.expanduser().resolve()
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"CosyVoice reference audio does not exist: "
                    f"{reference_path}"
                )

        model_argument = model_name
        model_path = Path(model_name).expanduser()
        if model_path.exists():
            model_argument = str(model_path.resolve())

        command = [
            str(Path(worker_python).expanduser()),
            str(Path(worker_script).expanduser()),
            "--runtime-dir",
            str(runtime_path),
            "--model",
            model_argument,
            "--inference-mode",
            inference_mode,
            "--warmup-text",
            warmup_text,
        ]
        if inference_mode == "zero_shot":
            if reference_path is None:
                raise RuntimeError("CosyVoice reference path is unavailable")
            command.extend(
                [
                    "--reference-audio",
                    str(reference_path),
                    "--reference-text",
                    reference_text,
                ]
            )
        else:
            command.extend(["--speaker", speaker.strip()])
        if fp16:
            command.append("--fp16")
        if load_jit:
            command.append("--load-jit")

        self._lock = threading.Lock()
        self._request_id = 0
        self._closed = False
        self._messages: queue.Queue[str | BaseException] = queue.Queue()
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
            raise RuntimeError("CosyVoice3 worker pipes are unavailable")

        self._reader = threading.Thread(
            target=self._read_stdout,
            name="cosyvoice3-worker-reader",
            daemon=True,
        )
        self._reader.start()

        try:
            ready = self._read_message(startup_timeout_seconds)
            if ready.get("event") != "ready":
                raise RuntimeError(self._format_worker_error(ready))
            self.sample_rate = int(ready["sample_rate"])
            if self.sample_rate <= 0:
                raise RuntimeError("CosyVoice3 returned an invalid sample rate")
        except BaseException:
            self.close()
            raise

        atexit.register(self.close)

    def stream_synthesize(self, text: str) -> Iterator[AudioChunk]:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("TTS text cannot be empty")

        with self._lock:
            if self._closed or self._process.poll() is not None:
                raise RuntimeError("CosyVoice3 worker is not running")

            self._request_id += 1
            request_id = self._request_id
            self._write_message(
                {
                    "id": request_id,
                    "op": "stream_synthesize",
                    "text": cleaned_text,
                }
            )

            received_chunks = 0
            while True:
                response = self._read_message(timeout=None)
                if response.get("id") != request_id:
                    raise RuntimeError(
                        "CosyVoice3 worker returned a mismatched response id"
                    )

                event = response.get("event")
                if event == "audio_chunk":
                    encoded_audio = response.get("pcm_s16le_base64")
                    if not isinstance(encoded_audio, str):
                        raise RuntimeError(
                            "CosyVoice3 worker returned invalid audio data"
                        )
                    try:
                        pcm = base64.b64decode(encoded_audio, validate=True)
                    except ValueError as exc:
                        raise RuntimeError(
                            "CosyVoice3 worker returned malformed base64 audio"
                        ) from exc
                    chunk = AudioChunk(
                        pcm_s16le=pcm,
                        sample_rate=int(response["sample_rate"]),
                        channels=int(response.get("channels", 1)),
                    )
                    if chunk.sample_rate != self.sample_rate:
                        raise RuntimeError(
                            "CosyVoice3 changed sample rate during streaming"
                        )
                    received_chunks += 1
                    yield chunk
                    continue

                if event == "complete":
                    if received_chunks == 0:
                        raise RuntimeError("CosyVoice3 returned no audio chunks")
                    return
                if event == "error":
                    raise RuntimeError(self._format_worker_error(response))
                raise RuntimeError(
                    f"CosyVoice3 worker returned unknown event: {event}"
                )

    def synthesize(self, text: str, output_path: Path) -> Path:
        resolved_output = output_path.expanduser().resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(resolved_output), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(self.sample_rate)
            for chunk in self.stream_synthesize(text):
                if chunk.channels != 1:
                    raise RuntimeError(
                        "CosyVoice3 WAV output currently requires mono audio"
                    )
                audio_file.writeframesraw(chunk.pcm_s16le)

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

    def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            self._messages.put(
                RuntimeError("CosyVoice3 worker stdout is unavailable")
            )
            return
        try:
            while True:
                line = stdout.readline()
                self._messages.put(line)
                if not line:
                    return
        except BaseException as exc:
            self._messages.put(exc)

    def _write_message(self, message: dict[str, Any]) -> None:
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("CosyVoice3 worker stdin is unavailable")
        stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        stdin.flush()

    def _read_message(self, timeout: float | None) -> dict[str, Any]:
        try:
            value = self._messages.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(
                "Timed out while waiting for the CosyVoice3 worker"
            ) from exc

        if isinstance(value, BaseException):
            raise RuntimeError("Unable to read from CosyVoice3 worker") from value
        if not value:
            raise RuntimeError(
                "CosyVoice3 worker exited before returning a response "
                f"(exit_code={self._process.poll()})"
            )
        try:
            message = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Invalid response from CosyVoice3 worker: "
                f"{value.rstrip()}"
            ) from exc
        if not isinstance(message, dict):
            raise RuntimeError("CosyVoice3 worker response must be an object")
        return message

    @staticmethod
    def _format_worker_error(message: dict[str, Any]) -> str:
        detail = message.get("error", "unknown worker error")
        return f"CosyVoice3 worker failed: {detail}"
