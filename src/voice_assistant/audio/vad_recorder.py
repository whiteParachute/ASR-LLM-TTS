from __future__ import annotations

import math
import wave
from collections import deque
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol


class VoiceActivityDetector(Protocol):
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        ...


class RawInputStream(Protocol):
    def read(self, frames: int) -> tuple[Any, bool]:
        ...


StreamFactory = Callable[..., AbstractContextManager[RawInputStream]]


class SoundDeviceVADRecorder:
    """Record one utterance and stop after VAD detects trailing silence."""

    SUPPORTED_SAMPLE_RATES = {8000, 16000, 32000, 48000}
    SUPPORTED_FRAME_DURATIONS = {10, 20, 30}

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        vad_mode: int = 2,
        start_trigger_ms: int = 60,
        end_silence_ms: int = 800,
        pre_roll_ms: int = 200,
        speech_timeout_seconds: float = 30.0,
        max_utterance_seconds: float = 30.0,
        input_device: int | str | None = None,
        vad: VoiceActivityDetector | None = None,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        self._validate_settings(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
            vad_mode=vad_mode,
            speech_timeout_seconds=speech_timeout_seconds,
            max_utterance_seconds=max_utterance_seconds,
        )

        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        self._frame_samples = sample_rate * frame_duration_ms // 1000
        self._expected_frame_bytes = self._frame_samples * 2
        self._start_trigger_frames = self._duration_to_frames(
            start_trigger_ms
        )
        self._end_silence_frames = self._duration_to_frames(
            end_silence_ms
        )
        pre_roll_frames = self._duration_to_frames(pre_roll_ms)
        self._pre_roll_frames = pre_roll_frames + self._start_trigger_frames
        self._speech_timeout_frames = math.ceil(
            speech_timeout_seconds * 1000 / frame_duration_ms
        )
        self._max_utterance_frames = math.ceil(
            max_utterance_seconds * 1000 / frame_duration_ms
        )
        self._input_device = input_device

        if vad is None:
            import webrtcvad

            self._vad = webrtcvad.Vad(vad_mode)
        else:
            self._vad = vad

        if stream_factory is None:
            import sounddevice as sd

            self._stream_factory = sd.RawInputStream
        else:
            self._stream_factory = stream_factory

    def record(self, output_path: Path) -> Path:
        pre_roll: deque[bytes] = deque(maxlen=self._pre_roll_frames)
        recorded_frames: list[bytes] = []
        consecutive_speech = 0
        consecutive_silence = 0
        waited_frames = 0
        triggered = False

        with self._stream_factory(
            samplerate=self._sample_rate,
            blocksize=self._frame_samples,
            device=self._input_device,
            channels=1,
            dtype="int16",
        ) as stream:
            while True:
                data, overflowed = stream.read(self._frame_samples)
                if overflowed:
                    raise RuntimeError("Microphone input overflowed")

                frame = bytes(data)
                if len(frame) != self._expected_frame_bytes:
                    raise RuntimeError(
                        "Microphone returned an invalid PCM frame: "
                        f"expected {self._expected_frame_bytes} bytes, "
                        f"got {len(frame)}"
                    )

                is_speech = self._vad.is_speech(
                    frame,
                    self._sample_rate,
                )

                if not triggered:
                    waited_frames += 1
                    pre_roll.append(frame)
                    consecutive_speech = (
                        consecutive_speech + 1 if is_speech else 0
                    )

                    if consecutive_speech >= self._start_trigger_frames:
                        triggered = True
                        recorded_frames.extend(pre_roll)
                        pre_roll.clear()
                        continue

                    if waited_frames >= self._speech_timeout_frames:
                        raise TimeoutError(
                            "No speech detected before timeout"
                        )
                    continue

                recorded_frames.append(frame)
                consecutive_silence = (
                    0 if is_speech else consecutive_silence + 1
                )

                if consecutive_silence >= self._end_silence_frames:
                    break
                if len(recorded_frames) >= self._max_utterance_frames:
                    break

        return self._write_wav(output_path, recorded_frames)

    def _duration_to_frames(self, duration_ms: int) -> int:
        return max(1, math.ceil(duration_ms / self._frame_duration_ms))

    def _write_wav(
        self,
        output_path: Path,
        frames: list[bytes],
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(b"".join(frames))
        return output_path

    @classmethod
    def _validate_settings(
        cls,
        *,
        sample_rate: int,
        frame_duration_ms: int,
        vad_mode: int,
        speech_timeout_seconds: float,
        max_utterance_seconds: float,
    ) -> None:
        if sample_rate not in cls.SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported WebRTC VAD sample rate: {sample_rate}"
            )
        if frame_duration_ms not in cls.SUPPORTED_FRAME_DURATIONS:
            raise ValueError(
                "WebRTC VAD frame duration must be 10, 20, or 30 ms"
            )
        if vad_mode not in {0, 1, 2, 3}:
            raise ValueError("WebRTC VAD mode must be between 0 and 3")
        if speech_timeout_seconds <= 0:
            raise ValueError("Speech timeout must be greater than zero")
        if max_utterance_seconds <= 0:
            raise ValueError(
                "Maximum utterance duration must be greater than zero"
            )
