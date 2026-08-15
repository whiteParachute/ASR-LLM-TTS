from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

Role = Literal["user", "system", "assistant"]

@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class PipelineResult:
    transcript: str
    reply: str
    audio_path: Path


class ASRProvider(Protocol):
    def transcribe(self, audio_path: Path) -> str:
        ...


class LLMProvider(Protocol):
    def generate(self, messages: Sequence[Message]) -> str:
        ...


class TTSProvider(Protocol):
    def synthesize(self, text: str, output_path: Path) -> Path:
        ...
