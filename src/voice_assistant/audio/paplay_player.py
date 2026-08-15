from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


CommandRunner = Callable[..., Any]


class PaplayAudioPlayer:
    """Play audio synchronously through the WSLg PulseAudio server."""

    def __init__(
        self,
        pulse_server: str | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self._pulse_server = pulse_server
        self._command_runner = command_runner or subprocess.run

    def play(self, audio_path: Path) -> None:
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file does not exist: {audio_path}"
            )

        environment = os.environ.copy()
        pulse_server = self._pulse_server
        if pulse_server is None:
            wslg_socket = Path("/mnt/wslg/PulseServer")
            if wslg_socket.exists():
                pulse_server = "unix:/mnt/wslg/PulseServer"
        if pulse_server is not None:
            environment["PULSE_SERVER"] = pulse_server

        self._command_runner(
            ["paplay", str(audio_path)],
            check=True,
            env=environment,
        )
