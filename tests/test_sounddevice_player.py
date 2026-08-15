import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from voice_assistant.audio.sounddevice_player import SoundDeviceAudioPlayer


class SoundDeviceAudioPlayerTest(unittest.TestCase):
    def test_reads_plays_and_waits_for_audio(self) -> None:
        reader = Mock(return_value=("audio-data", 24000))
        play = Mock()
        wait = Mock()
        player = SoundDeviceAudioPlayer(
            output_device=3,
            audio_reader=reader,
            play_function=play,
            wait_function=wait,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "answer.wav"
            audio_path.write_bytes(b"fake audio")
            player.play(audio_path)

        reader.assert_called_once_with(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )
        play.assert_called_once_with(
            "audio-data",
            24000,
            device=3,
        )
        wait.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
