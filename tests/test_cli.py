import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voice_assistant.cli import execute
from voice_assistant.config import (
    AppConfig,
    ASRConfig,
    LLMConfig,
    RuntimeConfig,
    TTSConfig,
)
from voice_assistant.contracts import PipelineResult


def make_config(output_dir: Path) -> AppConfig:
    return AppConfig(
        asr=ASRConfig(
            provider="sensevoice",
            model="test-asr",
        ),
        llm=LLMConfig(
            provider="qwen25_transformers",
            model="test-llm",
            max_new_tokens=128,
            system_prompt="你是一个语音助手。",
            reply_instruction="回答简短一些。",
        ),
        tts=TTSConfig(
            provider="edge_tts",
            default_voice="zh-CN-XiaoyiNeural",
        ),
        runtime=RuntimeConfig(
            output_dir=output_dir,
        ),
    )


class CLITest(unittest.TestCase):
    def test_executes_pipeline_with_default_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "question.wav"
            audio_path.write_bytes(b"fake audio")

            config_path = temp_path / "config.yaml"
            output_dir = temp_path / "generated"
            expected_output_path = output_dir / "question_reply.wav"
            config = make_config(output_dir)
            expected_result = PipelineResult(
                transcript="你好",
                reply="你好，有什么可以帮你？",
                audio_path=expected_output_path,
            )

            with (
                patch(
                    "voice_assistant.cli.load_config",
                    return_value=config,
                ) as load_config_mock,
                patch(
                    "voice_assistant.cli.build_pipeline",
                ) as build_pipeline_mock,
            ):
                pipeline = build_pipeline_mock.return_value
                pipeline.run.return_value = expected_result

                result = execute(
                    config_path=config_path,
                    audio_path=audio_path,
                )

            load_config_mock.assert_called_once_with(config_path)
            build_pipeline_mock.assert_called_once_with(config)
            pipeline.run.assert_called_once_with(
                audio_path=audio_path,
                output_path=expected_output_path,
            )
            self.assertIs(result, expected_result)
            self.assertTrue(output_dir.is_dir())

    def test_rejects_missing_audio_before_loading_models(self) -> None:
        missing_audio = Path("missing.wav")

        with (
            patch("voice_assistant.cli.load_config") as load_config_mock,
            patch(
                "voice_assistant.cli.build_pipeline"
            ) as build_pipeline_mock,
        ):
            with self.assertRaises(FileNotFoundError):
                execute(
                    config_path=Path("configs/baseline.yaml"),
                    audio_path=missing_audio,
                )

        load_config_mock.assert_not_called()
        build_pipeline_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
