import unittest
from pathlib import Path
from unittest.mock import patch

from voice_assistant.bootstrap import build_pipeline
from voice_assistant.config import (
    AppConfig,
    ASRConfig,
    LLMConfig,
    RuntimeConfig,
    TTSConfig,
    ToolUseConfig,
)


class BootstrapTest(unittest.TestCase):
    def test_builds_pipeline_from_config(self) -> None:
        config = AppConfig(
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
                output_dir=Path("output"),
            ),
        )

        asr = object()
        llm = object()
        tts = object()
        pipeline = object()

        with (
            patch(
                "voice_assistant.bootstrap.build_asr",
                return_value=asr,
            ) as build_asr_mock,
            patch(
                "voice_assistant.bootstrap.build_llm",
                return_value=llm,
            ) as build_llm_mock,
            patch(
                "voice_assistant.bootstrap.build_tts",
                return_value=tts,
            ) as build_tts_mock,
            patch(
                "voice_assistant.bootstrap.VoicePipeline",
                return_value=pipeline,
            ) as pipeline_mock,
        ):
            result = build_pipeline(config)

        build_asr_mock.assert_called_once_with(config.asr)
        build_llm_mock.assert_called_once_with(config.llm)
        build_tts_mock.assert_called_once_with(config.tts)
        pipeline_mock.assert_called_once_with(
            asr=asr,
            llm=llm,
            tts=tts,
            system_prompt=config.llm.system_prompt,
            reply_instructions=config.llm.reply_instruction,
            performance=None,
            tool_loop=None,
        )
        self.assertIs(result, pipeline)

    def test_builds_enabled_builtin_tool_loop(self) -> None:
        config = AppConfig(
            asr=ASRConfig(provider="sensevoice", model="test-asr"),
            llm=LLMConfig(
                provider="qwen35_transformers",
                model="test-llm",
                max_new_tokens=128,
                system_prompt="测试助手",
            ),
            tts=TTSConfig(provider="edge_tts", default_voice="test"),
            runtime=RuntimeConfig(output_dir=Path("output")),
            tools=ToolUseConfig(
                enabled=True,
                max_rounds=2,
                timeout_seconds=1.5,
                max_result_chars=800,
            ),
        )
        registry = object()
        loop = object()

        with (
            patch(
                "voice_assistant.bootstrap.build_asr",
                return_value=object(),
            ),
            patch(
                "voice_assistant.bootstrap.build_llm",
                return_value=object(),
            ) as llm_mock,
            patch(
                "voice_assistant.bootstrap.build_tts",
                return_value=object(),
            ),
            patch(
                "voice_assistant.bootstrap.build_builtin_tool_registry",
                return_value=registry,
            ) as registry_mock,
            patch(
                "voice_assistant.bootstrap.BoundedToolLoop",
                return_value=loop,
            ) as loop_mock,
            patch(
                "voice_assistant.bootstrap.VoicePipeline"
            ) as pipeline_mock,
        ):
            build_pipeline(config)

        registry_mock.assert_called_once_with(
            timeout_seconds=1.5,
            max_result_chars=800,
        )
        loop_mock.assert_called_once_with(
            llm_mock.return_value,
            registry,
            max_rounds=2,
            performance=None,
        )
        self.assertIs(pipeline_mock.call_args.kwargs["tool_loop"], loop)


if __name__ == "__main__":
    unittest.main()
