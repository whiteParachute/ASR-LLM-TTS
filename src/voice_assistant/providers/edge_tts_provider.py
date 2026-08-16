from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class EdgeCommunicator(Protocol):
    async def save(self, output_file: str) -> None:
        ...

CommunicatorFactory = Callable[
    [str, str],
    EdgeCommunicator,
]


class EdgeTTSProvider:
    def __init__(self, default_voice: str, communicator_factory: CommunicatorFactory | None = None) -> None:
        self._default_voice = default_voice

        if communicator_factory is None:
            from edge_tts import Communicate

            self._communicator_factory = Communicate
        else:
            self._communicator_factory = communicator_factory

    def synthesize(self, text: str, output_path: Path) -> Path:
        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("TTS text cannot be empty")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        communicator = self._communicator_factory(cleaned_text, self._default_voice)
        asyncio.run(communicator.save(str(output_path)))

        if not output_path.is_file():
            raise RuntimeError(f"TTS did not create output file: {output_path}")

        return output_path