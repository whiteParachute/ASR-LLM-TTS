from __future__ import annotations

from voice_assistant.config import AppConfig
from voice_assistant.pipeline import VoicePipeline
from voice_assistant.providers.factory import build_asr, build_llm, build_tts


def build_pipeline(config: AppConfig) -> VoicePipeline:
    """Build a complete voice pipeline from application configuration."""
    asr = build_asr(config.asr)
    llm = build_llm(config.llm)
    tts = build_tts(config.tts)

    return VoicePipeline(
        asr=asr,
        llm=llm,
        tts=tts,
        system_prompt=config.llm.system_prompt,
        reply_instructions=config.llm.reply_instruction,
    )
