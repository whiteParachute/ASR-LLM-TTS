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
        self.assertEqual(config.tts.provider, "edge_tts")
        self.assertEqual(config.runtime.output_dir, Path("output"))

if  __name__ == "__main__":
    unittest.main()