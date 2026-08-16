import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any

from voice_assistant.audio.vad_recorder import SoundDeviceVADRecorder


SILENCE_FRAME = b"\x00\x00" * 320
SPEECH_FRAME = b"\x01\x00" * 320


class FakeVAD:
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self.asserted_sample_rate = sample_rate
        return frame == SPEECH_FRAME


class FakeInputStream:
    def __init__(self, frames: list[bytes]) -> None:
        self._frames = iter(frames)

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self, frames: int) -> tuple[bytes, bool]:
        return next(self._frames), False


class FakeStreamFactory:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames
        self.kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> FakeInputStream:
        self.kwargs = kwargs
        return FakeInputStream(self.frames)


class SoundDeviceVADRecorderTest(unittest.TestCase):
    def test_records_until_trailing_silence(self) -> None:
        frames = [
            SILENCE_FRAME,
            SPEECH_FRAME,
            SPEECH_FRAME,
            SILENCE_FRAME,
            SILENCE_FRAME,
        ]
        stream_factory = FakeStreamFactory(frames)
        recorder = SoundDeviceVADRecorder(
            sample_rate=16000,
            frame_duration_ms=20,
            start_trigger_ms=40,
            end_silence_ms=40,
            pre_roll_ms=20,
            speech_timeout_seconds=1.0,
            max_utterance_seconds=2.0,
            vad=FakeVAD(),
            stream_factory=stream_factory,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "input.wav"
            result = recorder.record(output_path)

            self.assertEqual(result, output_path)
            with wave.open(str(output_path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 16000)
                recorded = wav_file.readframes(wav_file.getnframes())

        self.assertEqual(recorded, b"".join(frames))
        self.assertEqual(stream_factory.kwargs["blocksize"], 320)
        self.assertEqual(stream_factory.kwargs["dtype"], "int16")

    def test_times_out_without_speech(self) -> None:
        recorder = SoundDeviceVADRecorder(
            sample_rate=16000,
            frame_duration_ms=20,
            speech_timeout_seconds=0.06,
            vad=FakeVAD(),
            stream_factory=FakeStreamFactory(
                [SILENCE_FRAME, SILENCE_FRAME, SILENCE_FRAME]
            ),
        )

        with self.assertRaisesRegex(
            TimeoutError,
            "No speech detected before timeout",
        ):
            recorder.record(Path("unused.wav"))


if __name__ == "__main__":
    unittest.main()
