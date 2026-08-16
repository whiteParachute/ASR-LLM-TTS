import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from voice_assistant.audio.paplay_player import PaplayAudioPlayer


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


if __name__ == "__main__":
    unittest.main()
