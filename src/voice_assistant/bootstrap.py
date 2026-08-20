from __future__ import annotations

from voice_assistant.config import AppConfig
from voice_assistant.observability import PerformanceLogger
from voice_assistant.pipeline import VoicePipeline
from voice_assistant.providers.factory import build_asr, build_llm, build_tts
from voice_assistant.tooling import BoundedToolLoop, build_builtin_tool_registry


def build_pipeline(
    config: AppConfig,
    performance: PerformanceLogger | None = None,
) -> VoicePipeline:
    """Build a complete voice pipeline from application configuration."""
    asr = build_asr(config.asr)
    llm = build_llm(config.llm)
    tts = build_tts(config.tts)
    tool_loop: BoundedToolLoop | None = None
    if config.tools.enabled:
        registry = build_builtin_tool_registry(
            timeout_seconds=config.tools.timeout_seconds,
            max_result_chars=config.tools.max_result_chars,
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
