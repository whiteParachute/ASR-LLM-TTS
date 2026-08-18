from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Any, Protocol

from voice_assistant.contracts import AudioChunk


CommandRunner = Callable[..., Any]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]


class StreamProcess(Protocol):
    stdin: IO[bytes] | None
    returncode: int | None

    def poll(self) -> int | None:
        ...

    def terminate(self) -> None:
        ...

    def wait(self, timeout: float | None = None) -> int:
        ...


ProcessFactory = Callable[..., StreamProcess]


class PaplayAudioPlayer:
    """Play audio synchronously through the WSLg PulseAudio server."""

    def __init__(
        self,
        pulse_server: str | None = None,
        command_runner: CommandRunner | None = None,
        process_factory: ProcessFactory | None = None,
        stream_tail_guard_ms: int = 0,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        if stream_tail_guard_ms < 0:
            raise ValueError("Stream tail guard cannot be negative")
        self._pulse_server = pulse_server
        self._command_runner = command_runner or subprocess.run
        self._process_factory = process_factory or subprocess.Popen
        self._stream_tail_guard_seconds = stream_tail_guard_ms / 1000
        self._clock = clock
        self._sleeper = sleeper

    def play(self, audio_path: Path) -> None:
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file does not exist: {audio_path}"
            )

        environment = self._build_environment()

        self._command_runner(
            ["paplay", str(audio_path)],
            check=True,
            env=environment,
        )

    def play_stream(self, chunks: Iterable[AudioChunk]) -> None:
        iterator = iter(chunks)
        try:
            first_chunk = next(iterator)
        except StopIteration as exc:
            raise ValueError("Audio stream cannot be empty") from exc

        command = [
            "paplay",
            "--raw",
            "--format=s16le",
            f"--rate={first_chunk.sample_rate}",
            f"--channels={first_chunk.channels}",
        ]
        process = self._process_factory(
            command,
            stdin=subprocess.PIPE,
            env=self._build_environment(),
        )
        if process.stdin is None:
            process.terminate()
            process.wait(timeout=5.0)
            raise RuntimeError("paplay streaming stdin is unavailable")

        playback_deadline = self._clock()
        try:
            self._write_chunk(process.stdin, first_chunk, first_chunk)
            playback_deadline = self._extend_playback_deadline(
                playback_deadline,
                first_chunk,
            )
            for chunk in iterator:
                self._write_chunk(process.stdin, chunk, first_chunk)
                playback_deadline = self._extend_playback_deadline(
                    playback_deadline,
                    chunk,
                )
            process.stdin.close()
            return_code = process.wait()
        except BaseException:
            try:
                process.stdin.close()
            except OSError:
                pass
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5.0)
            raise

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        remaining_seconds = (
            playback_deadline
            + self._stream_tail_guard_seconds
            - self._clock()
        )
        if remaining_seconds > 0:
            self._sleeper(remaining_seconds)

    def _extend_playback_deadline(
        self,
        current_deadline: float,
        chunk: AudioChunk,
    ) -> float:
        return max(current_deadline, self._clock()) + (
            chunk.duration_ms / 1000
        )

    @staticmethod
    def _write_chunk(
        stdin: IO[bytes],
        chunk: AudioChunk,
        expected: AudioChunk,
    ) -> None:
        if (
            chunk.sample_rate != expected.sample_rate
            or chunk.channels != expected.channels
        ):
            raise ValueError("Audio stream format changed between chunks")
        stdin.write(chunk.pcm_s16le)
        stdin.flush()

    def _build_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        pulse_server = self._pulse_server
        if pulse_server is None:
            wslg_socket = Path("/mnt/wslg/PulseServer")
            if wslg_socket.exists():
                pulse_server = "unix:/mnt/wslg/PulseServer"
        if pulse_server is not None:
            environment["PULSE_SERVER"] = pulse_server
        return environment
