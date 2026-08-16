import unittest
from pathlib import Path

from voice_assistant.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_loads_baseline_config(self) -> None:
        config = load_config(
            PROJECT_ROOT / "configs" / "baseline.yaml"
        )

        self.assertEqual(config.asr.provider, "sensevoice")
        self.assertEqual(config.asr.model, "iic/SenseVoiceSmall")
        self.assertEqual(
            config.llm.model,
            "Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.assertEqual(config.llm.max_new_tokens, 128)
        self.assertEqual(config.tts.provider, "kokoro")
        self.assertEqual(config.tts.model, "hexgrad/Kokoro-82M")
        self.assertEqual(config.tts.language_code, "z")
        self.assertEqual(config.tts.default_voice, "zf_xiaoxiao")
        self.assertEqual(config.tts.sample_rate, 24000)
        self.assertEqual(config.tts.output_format, "wav")
        self.assertEqual(config.audio.sample_rate, 16000)
        self.assertEqual(
            config.audio.playback_backend,
            "sounddevice",
        )
        self.assertEqual(config.audio.frame_duration_ms, 20)
        self.assertEqual(config.audio.vad_mode, 2)
        self.assertEqual(config.audio.end_silence_ms, 800)
        self.assertEqual(config.runtime.output_dir, Path("output"))
        self.assertTrue(config.observability.enabled)
        self.assertTrue(config.observability.console)
        self.assertTrue(config.observability.jsonl)
        self.assertEqual(config.observability.log_dir, Path("logs"))

if  __name__ == "__main__":
    unittest.main()
