from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Protocol, Sequence

Role = Literal["user", "system", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]

    def as_chat_template_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolAwareResponse:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceReference:
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class PipelineResult:
    transcript: str
    reply: str
    audio_path: Path
    audio_paths: tuple[Path, ...] = ()
    sources: tuple[SourceReference, ...] = ()

    def __post_init__(self) -> None:
        if not self.audio_paths:
            object.__setattr__(self, "audio_paths", (self.audio_path,))


@dataclass(frozen=True, slots=True)
class PreparedResponse:
    transcript: str
    reply: str
    sources: tuple[SourceReference, ...] = ()


@dataclass(frozen=True, slots=True)
class AudioChunk:
    pcm_s16le: bytes
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        if not self.pcm_s16le:
            raise ValueError("Audio chunk cannot be empty.")
        if self.sample_rate < 1:
            raise ValueError("Audio chunk sample rate must be positive.")
        if self.channels < 1:
            raise ValueError("Audio chunk channels must be positive.")
        if len(self.pcm_s16le) % (2 * self.channels) != 0:
            raise ValueError("Audio chunk contains incomplete PCM frames.")

    @property
    def duration_ms(self) -> float:
        frame_count = len(self.pcm_s16le) / (2 * self.channels)
        return frame_count / self.sample_rate * 1000


class ASRProvider(Protocol):
    def transcribe(self, audio_path: Path) -> str:
        ...


class LLMProvider(Protocol):
    def generate(self, messages: Sequence[Message]) -> str:
        ...


class ToolCallingLLMProvider(LLMProvider, Protocol):
    def generate_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ToolAwareResponse:
        ...


class TTSProvider(Protocol):
    def synthesize(self, text: str, output_path: Path) -> Path:
        ...


class StreamingTTSProvider(TTSProvider, Protocol):
    def stream_synthesize(self, text: str) -> Iterator[AudioChunk]:
        ...


class UtteranceRecorder(Protocol):
    def record(self, output_path: Path) -> Path:
        ...


class AudioPlayer(Protocol):
    def play(self, audio_path: Path) -> None:
        ...


class StreamingAudioPlayer(AudioPlayer, Protocol):
    def play_stream(self, chunks: Iterable[AudioChunk]) -> None:
        ...
