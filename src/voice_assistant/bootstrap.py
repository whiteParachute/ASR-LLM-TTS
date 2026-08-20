from __future__ import annotations

from voice_assistant.config import AppConfig
from voice_assistant.observability import PerformanceLogger
from voice_assistant.pipeline import VoicePipeline
from voice_assistant.providers.factory import build_asr, build_llm, build_tts
from voice_assistant.tooling import BoundedToolLoop, build_builtin_tool_registry
from voice_assistant.web_search import SearXNGSearchProvider


def build_pipeline(
    config: AppConfig,
    performance: PerformanceLogger | None = None,
) -> VoicePipeline:
    """Build a complete voice pipeline from application configuration."""
    web_search = None
    web_search_timeout_seconds = None
    if config.tools.enabled:
        web_config = config.tools.web_search
        if web_config.enabled:
            if web_config.provider != "searxng":
                raise ValueError(
                    "Unsupported web search provider: "
                    f"{web_config.provider}"
                )
            web_search = SearXNGSearchProvider(
                endpoint=web_config.endpoint,
                timeout_seconds=web_config.timeout_seconds,
                max_results=web_config.max_results,
                max_response_bytes=web_config.max_response_bytes,
                language=web_config.language,
                safesearch=web_config.safesearch,
            )
            web_search_timeout_seconds = web_config.timeout_seconds + 1.0

    asr = build_asr(config.asr)
    llm = build_llm(config.llm)
    tts = build_tts(config.tts)
    tool_loop: BoundedToolLoop | None = None
    if config.tools.enabled:
        registry = build_builtin_tool_registry(
            timeout_seconds=config.tools.timeout_seconds,
            max_result_chars=config.tools.max_result_chars,
            web_search=web_search,
            web_search_timeout_seconds=web_search_timeout_seconds,
        )
        tool_loop = BoundedToolLoop(
            llm,
            registry,
            max_rounds=config.tools.max_rounds,
            performance=performance,
        )

    return VoicePipeline(
        asr=asr,
        llm=llm,
        tts=tts,
        system_prompt=config.llm.system_prompt,
        reply_instructions=config.llm.reply_instruction,
        performance=performance,
        tool_loop=tool_loop,
    )
