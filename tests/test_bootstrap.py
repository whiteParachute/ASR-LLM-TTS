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
    WebSearchConfig,
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
            web_search=None,
            web_search_timeout_seconds=None,
        )
        loop_mock.assert_called_once_with(
            llm_mock.return_value,
            registry,
            max_rounds=2,
            performance=None,
        )
        self.assertIs(pipeline_mock.call_args.kwargs["tool_loop"], loop)

    def test_builds_enabled_searxng_tool(self) -> None:
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
                max_result_chars=6000,
                web_search=WebSearchConfig(
                    enabled=True,
                    endpoint="https://search.example/searxng",
                    timeout_seconds=5,
                    max_results=4,
                    max_response_bytes=500_000,
                    language="zh-CN",
                    safesearch=2,
                ),
            ),
        )
        web_provider = object()

        with (
            patch(
                "voice_assistant.bootstrap.build_asr",
                return_value=object(),
            ),
            patch(
                "voice_assistant.bootstrap.build_llm",
                return_value=object(),
            ),
            patch(
                "voice_assistant.bootstrap.build_tts",
                return_value=object(),
            ),
            patch(
                "voice_assistant.bootstrap.SearXNGSearchProvider",
                return_value=web_provider,
            ) as web_provider_mock,
            patch(
                "voice_assistant.bootstrap.build_builtin_tool_registry",
                return_value=object(),
            ) as registry_mock,
            patch("voice_assistant.bootstrap.BoundedToolLoop"),
            patch("voice_assistant.bootstrap.VoicePipeline"),
        ):
            build_pipeline(config)

        web_provider_mock.assert_called_once_with(
            endpoint="https://search.example/searxng",
            timeout_seconds=5,
            max_results=4,
            max_response_bytes=500_000,
            language="zh-CN",
            safesearch=2,
        )
        registry_mock.assert_called_once_with(
            timeout_seconds=2.0,
            max_result_chars=6000,
            web_search=web_provider,
            web_search_timeout_seconds=6.0,
        )


if __name__ == "__main__":
    unittest.main()
