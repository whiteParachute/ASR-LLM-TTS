import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from voice_assistant.audio.paplay_player import PaplayAudioPlayer
from voice_assistant.contracts import AudioChunk


class RecordingBinaryInput(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.close_called = False

    def close(self) -> None:
        self.close_called = True


class FakeStreamProcess:
    def __init__(self) -> None:
        self.stdin = RecordingBinaryInput()
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class PaplayAudioPlayerTest(unittest.TestCase):
    def test_plays_audio_through_selected_pulse_server(self) -> None:
        runner = Mock()
        player = PaplayAudioPlayer(
            pulse_server="unix:/test/PulseServer",
            command_runner=runner,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "answer.wav"
            audio_path.write_bytes(b"fake audio")
            player.play(audio_path)

        args, kwargs = runner.call_args
        self.assertEqual(args[0], ["paplay", str(audio_path)])
        self.assertTrue(kwargs["check"])
        self.assertEqual(
            kwargs["env"]["PULSE_SERVER"],
            "unix:/test/PulseServer",
        )

    def test_streams_pcm_chunks_through_one_paplay_process(self) -> None:
        process = FakeStreamProcess()
        process_factory = Mock(return_value=process)
        clock = FakeClock()
        player = PaplayAudioPlayer(
            pulse_server="unix:/test/PulseServer",
            process_factory=process_factory,
            stream_tail_guard_ms=300,
            clock=clock.monotonic,
            sleeper=clock.sleep,
        )
        chunks = [
            AudioChunk(b"\x01\x00\x02\x00", sample_rate=24000),
            AudioChunk(b"\x03\x00\x04\x00", sample_rate=24000),
        ]

        player.play_stream(chunks)

        args, kwargs = process_factory.call_args
        self.assertEqual(
            args[0],
            [
                "paplay",
                "--raw",
                "--format=s16le",
                "--rate=24000",
                "--channels=1",
            ],
        )
        self.assertEqual(
            kwargs["env"]["PULSE_SERVER"],
            "unix:/test/PulseServer",
        )
        self.assertEqual(
            process.stdin.getvalue(),
            b"\x01\x00\x02\x00\x03\x00\x04\x00",
        )
        self.assertTrue(process.stdin.close_called)
        expected_audio_seconds = 4 / 24000
        self.assertEqual(len(clock.sleeps), 1)
        self.assertAlmostEqual(
            clock.sleeps[0],
            expected_audio_seconds + 0.3,
        )

    def test_rejects_negative_stream_tail_guard(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            PaplayAudioPlayer(stream_tail_guard_ms=-1)

    def test_rejects_stream_format_changes(self) -> None:
        process = FakeStreamProcess()
        player = PaplayAudioPlayer(
            process_factory=Mock(return_value=process),
        )

        with self.assertRaisesRegex(ValueError, "format changed"):
            player.play_stream(
                [
                    AudioChunk(b"\x00\x00", sample_rate=24000),
                    AudioChunk(b"\x00\x00", sample_rate=16000),
                ]
            )

        self.assertTrue(process.terminated)


if __name__ == "__main__":
    unittest.main()
