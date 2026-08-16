from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


AudioReader = Callable[..., tuple[Any, int]]
PlayFunction = Callable[..., Any]
WaitFunction = Callable[[], Any]


class SoundDeviceAudioPlayer:
    """Play an audio file synchronously through the selected output device."""

    def __init__(
        self,
        output_device: int | str | None = None,
        audio_reader: AudioReader | None = None,
        play_function: PlayFunction | None = None,
        wait_function: WaitFunction | None = None,
    ) -> None:
        self._output_device = output_device

        if audio_reader is None:
            import soundfile as sf

            self._audio_reader = sf.read
        else:
            self._audio_reader = audio_reader

        if play_function is None or wait_function is None:
            import sounddevice as sd

            self._play_function = play_function or sd.play
            self._wait_function = wait_function or sd.wait
        else:
            self._play_function = play_function
            self._wait_function = wait_function

    def play(self, audio_path: Path) -> None:
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file does not exist: {audio_path}"
            )

        audio, sample_rate = self._audio_reader(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )
        self._play_function(
            audio,
            sample_rate,
            device=self._output_device,
        )
        self._wait_function()
