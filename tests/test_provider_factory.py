import unittest
from pathlib import Path
from unittest.mock import patch

from voice_assistant.config import ASRConfig, LLMConfig, TTSConfig
from voice_assistant.providers.factory import build_asr, ProviderConfigError, build_llm, build_tts


class ProviderFactoryTest(unittest.TestCase):
    @patch("voice_assistant.providers.factory.SenseVoiceASR")

    def test_builds_sensevoice_provider(self, mock_sensevoice_asr) -> None:
        config = ASRConfig(
            provider="sensevoice",
            model="iic/SenseVoiceSmall",
            language="auto",
            use_itn=False,
        )

        provider = build_asr(config)

        mock_sensevoice_asr.assert_called_once_with(
            model_name="iic/SenseVoiceSmall",
            language="auto",
            use_itn=False,
            device=None,
        )
        self.assertIs(provider, mock_sensevoice_asr.return_value)

    def test_rejects_unknown_asr_provider(self) -> None:
        config = ASRConfig(
            provider="unknown_provider",
            model="some_model",
        )

        with self.assertRaisesRegex(ProviderConfigError, "Unsupported ASR provider: unknown"):
            build_asr(config)

    @patch("voice_assistant.providers.factory.Qwen25LLM")
    def test_builds_qwen25_llm(self, mock_qwen25_llm) -> None:
        config = LLMConfig(
            provider="qwen25_transformers",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            max_new_tokens=128,
            system_prompt="你是一个语音助手。",
            reply_instruction="回答保持在50字以内。"
        )

        provider = build_llm(config)

        mock_qwen25_llm.assert_called_once_with(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_new_tokens=128,
        )

        self.assertIs(provider, mock_qwen25_llm.return_value)

    @patch("voice_assistant.providers.factory.Qwen35LLM")
    def test_builds_qwen35_llm(self, mock_qwen35_llm) -> None:
        config = LLMConfig(
            provider="qwen35_transformers",
            model="Qwen/Qwen3.5-4B",
            max_new_tokens=128,
            system_prompt="你是一个语音助手。",
            load_in_4bit=True,
            compute_dtype="float16",
            enable_thinking=False,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

        provider = build_llm(config)

        mock_qwen35_llm.assert_called_once_with(
            model_name="Qwen/Qwen3.5-4B",
            max_new_tokens=128,
            load_in_4bit=True,
            compute_dtype="float16",
            enable_thinking=False,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
        self.assertIs(provider, mock_qwen35_llm.return_value)

    def test_reject_unknown_llm_provider(self) -> None:
        config = LLMConfig(
            provider="unknown_provider",
            model="unknown-model",
            max_new_tokens=128,
            system_prompt="测试",
        )

        with self.assertRaisesRegex(ProviderConfigError, "Unsupported LLM provider: unknown_provider"):
            build_llm(config)

    @patch("voice_assistant.providers.factory.EdgeTTSProvider")
    def test_build_edge_tts(self, mock_edge_tts) -> None:
        config = TTSConfig(provider="edge_tts", default_voice="zh-CN-XiaoyiNeural")
        provider = build_tts(config)

        mock_edge_tts.assert_called_once_with(default_voice="zh-CN-XiaoyiNeural")
        self.assertIs(provider, mock_edge_tts.return_value)

    @patch("voice_assistant.providers.factory.KokoroTTSProvider")
    def test_builds_kokoro_tts(self, mock_kokoro_tts) -> None:
        config = TTSConfig(
            provider="kokoro",
            model="hexgrad/Kokoro-82M",
            language_code="z",
            default_voice="zf_xiaoxiao",
            speed=1.0,
            sample_rate=24000,
            output_format="wav",
        )

        provider = build_tts(config)

        mock_kokoro_tts.assert_called_once_with(
            model_name="hexgrad/Kokoro-82M",
            language_code="z",
            default_voice="zf_xiaoxiao",
            speed=1.0,
            sample_rate=24000,
            device=None,
        )
        self.assertIs(provider, mock_kokoro_tts.return_value)

    @patch("voice_assistant.providers.factory.Qwen3TTSWorkerProvider")
    def test_builds_qwen3_tts_worker(self, mock_qwen3_tts) -> None:
        config = TTSConfig(
            provider="qwen3_tts_worker",
            model="models/Qwen3-TTS-12Hz-0.6B-Base",
            language_code="Chinese",
            default_voice="reference",
            device="cuda:0",
            reference_audio=Path("voices/reference.wav"),
            reference_text="参考音频文本。",
        )

        provider = build_tts(config)

        mock_qwen3_tts.assert_called_once_with(
            model_name="models/Qwen3-TTS-12Hz-0.6B-Base",
            reference_audio=Path("voices/reference.wav"),
            reference_text="参考音频文本。",
            language="Chinese",
            device="cuda:0",
            worker_python=".venv-tts/bin/python",
            worker_script="scripts/qwen3_tts_worker.py",
            x_vector_only_mode=False,
            dtype="bfloat16",
            attention_implementation="sdpa",
            max_new_tokens=256,
            startup_timeout_seconds=180.0,
        )
        self.assertIs(provider, mock_qwen3_tts.return_value)

    def test_rejects_unknown_tts_provider(self) -> None:
        config = TTSConfig(provider="unknown_provider", default_voice="test-voice")
        with self.assertRaisesRegex(ProviderConfigError, "Unsupported TTS provider: unknown_provider"):
            build_tts(config)


if __name__ == "__main__":
    unittest.main()
