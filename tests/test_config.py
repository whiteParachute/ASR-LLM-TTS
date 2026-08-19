import unittest
from pathlib import Path

from voice_assistant.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_loads_baseline_config(self) -> None:
        config = load_config(
            PROJECT_ROOT / "configs" / "baseline.yaml"
        )

        self.assertEqual(config.asr.provider, "qwen3_asr_transformers")
        self.assertEqual(config.asr.model, "Qwen/Qwen3-ASR-0.6B-hf")
        self.assertEqual(config.asr.language, "auto")
        self.assertEqual(config.asr.compute_dtype, "bfloat16")
        self.assertEqual(config.asr.attention_implementation, "sdpa")
        self.assertEqual(config.asr.max_new_tokens, 256)
        self.assertEqual(config.asr.prompt, "")
        self.assertEqual(
            config.llm.model,
            "Qwen/Qwen3.5-4B",
        )
        self.assertEqual(config.llm.max_new_tokens, 128)
        self.assertTrue(config.llm.load_in_4bit)
        self.assertEqual(config.llm.compute_dtype, "float16")
        self.assertFalse(config.llm.enable_thinking)
        self.assertTrue(config.llm.do_sample)
        self.assertEqual(config.llm.temperature, 0.7)
        self.assertEqual(config.llm.top_p, 0.8)
        self.assertEqual(config.llm.top_k, 20)
        self.assertEqual(config.tts.provider, "qwen3_tts_worker")
        self.assertEqual(
            config.tts.model,
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        )
        self.assertEqual(config.tts.language_code, "Chinese")
        self.assertEqual(config.tts.default_voice, "reference")
        self.assertEqual(config.tts.sample_rate, 24000)
        self.assertEqual(config.tts.output_format, "wav")
        self.assertEqual(
            config.tts.worker_python,
            ".venv-tts/bin/python",
        )
        self.assertEqual(
            config.tts.reference_audio,
            Path("voices/reference.wav"),
        )
        self.assertIn("I resent you", config.tts.reference_text)
        self.assertFalse(config.tts.x_vector_only_mode)
        self.assertEqual(config.tts.dtype, "bfloat16")
        self.assertEqual(config.tts.attention_implementation, "sdpa")
        self.assertEqual(config.tts.max_new_tokens, 256)
        self.assertEqual(config.audio.sample_rate, 16000)
        self.assertEqual(
            config.audio.playback_backend,
            "sounddevice",
        )
        self.assertIsNone(config.audio.playback_latency_ms)
        self.assertIsNone(config.audio.playback_process_time_ms)
        self.assertEqual(config.audio.playback_tail_guard_ms, 0)
        self.assertEqual(config.audio.frame_duration_ms, 20)
        self.assertEqual(config.audio.vad_mode, 2)
        self.assertEqual(config.audio.end_silence_ms, 800)
        self.assertEqual(config.runtime.output_dir, Path("output"))
        self.assertEqual(config.runtime.reply_chunk_max_chars, 18)
        self.assertFalse(config.runtime.stream_llm_to_tts)
        self.assertEqual(config.runtime.first_reply_chunk_chars, 6)
        self.assertTrue(config.observability.enabled)
        self.assertTrue(config.observability.console)
        self.assertTrue(config.observability.jsonl)
        self.assertEqual(config.observability.log_dir, Path("logs"))

    def test_loads_cosyvoice_sft_default_wsl_config(self) -> None:
        config = load_config(
            PROJECT_ROOT / "configs" / "wsl_cuda.yaml"
        )

        self.assertEqual(
            config.tts.provider,
            "cosyvoice3_stream_worker",
        )
        self.assertEqual(
            config.tts.model,
            "models/CosyVoice-300M-SFT",
        )
        self.assertEqual(
            config.tts.worker_python,
            ".venv-cosyvoice/bin/python",
        )
        self.assertEqual(
            config.tts.runtime_dir,
            Path(".runtime/CosyVoice"),
        )
        self.assertEqual(config.tts.sample_rate, 22050)
        self.assertIsNone(config.tts.reference_audio)
        self.assertEqual(config.tts.reference_text, "")
        self.assertEqual(config.tts.dtype, "float16")
        self.assertEqual(config.tts.startup_timeout_seconds, 300)
        self.assertEqual(config.tts.inference_mode, "sft")
        self.assertEqual(config.tts.speaker, "中文女")
        self.assertFalse(config.tts.load_jit)
        self.assertEqual(config.audio.playback_latency_ms, 40)
        self.assertEqual(config.audio.playback_process_time_ms, 20)
        self.assertEqual(config.audio.playback_tail_guard_ms, 300)
        self.assertEqual(config.audio.vad_mode, 3)
        self.assertEqual(config.audio.start_trigger_ms, 200)
        self.assertEqual(config.audio.end_silence_ms, 500)
        self.assertIn("不超过15个汉字", config.llm.reply_instruction)
        self.assertTrue(config.runtime.stream_llm_to_tts)
        self.assertEqual(config.runtime.first_reply_chunk_chars, 6)

    def test_loads_cosyvoice3_experimental_config(self) -> None:
        config = load_config(
            PROJECT_ROOT / "configs" / "wsl_cuda_cosyvoice3.yaml"
        )

        self.assertEqual(
            config.tts.provider,
            "cosyvoice3_stream_worker",
        )
        self.assertEqual(
            config.tts.model,
            "models/Fun-CosyVoice3-0.5B-2512",
        )
        self.assertEqual(config.tts.sample_rate, 24000)
        self.assertEqual(config.tts.inference_mode, "zero_shot")
        self.assertEqual(config.tts.speaker, "")
        self.assertEqual(
            config.tts.reference_audio,
            Path(".runtime/CosyVoice/asset/zero_shot_prompt.wav"),
        )
        self.assertEqual(
            config.tts.reference_text,
            "希望你以后能够做的比我还好呦。",
        )
        self.assertFalse(config.tts.load_jit)
        self.assertEqual(config.audio.playback_latency_ms, 40)
        self.assertEqual(config.audio.playback_process_time_ms, 20)
        self.assertEqual(config.audio.playback_tail_guard_ms, 300)
        self.assertEqual(
            config.runtime.output_dir,
            Path("output/wsl-cosyvoice3"),
        )
        self.assertEqual(
            config.observability.log_dir,
            Path("logs/wsl-cosyvoice3"),
        )
        self.assertTrue(config.runtime.stream_llm_to_tts)
        self.assertEqual(config.runtime.first_reply_chunk_chars, 6)

if  __name__ == "__main__":
    unittest.main()
