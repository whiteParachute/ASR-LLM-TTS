from __future__ import annotations

from voice_assistant.config import ASRConfig, LLMConfig, TTSConfig
from voice_assistant.contracts import ASRProvider, LLMProvider, TTSProvider
from voice_assistant.providers.edge_tts_provider import EdgeTTSProvider
from voice_assistant.providers.kokoro_tts_provider import KokoroTTSProvider
from voice_assistant.providers.qwen3_tts_worker import Qwen3TTSWorkerProvider
from voice_assistant.providers.qwen25_llm import Qwen25LLM
from voice_assistant.providers.qwen35_llm import Qwen35LLM
from voice_assistant.providers.sensevoice_asr import SenseVoiceASR


class ProviderConfigError(ValueError):
    """Raised when the provider configuration is invalid."""

def build_asr(config: ASRConfig) -> ASRProvider:
    """Build an ASR provider based on the configuration."""
    if config.provider == "sensevoice":
        return SenseVoiceASR(
            model_name=config.model,
            language=config.language,
            use_itn=config.use_itn,
            device=config.device,
        )
    raise ProviderConfigError(f"Unsupported ASR provider: {config.provider}")

def build_llm(config: LLMConfig) -> LLMProvider:
    if config.provider == "qwen25_transformers":
        return Qwen25LLM(model_name=config.model, max_new_tokens=config.max_new_tokens)
    if config.provider == "qwen35_transformers":
        return Qwen35LLM(
            model_name=config.model,
            max_new_tokens=config.max_new_tokens,
            load_in_4bit=config.load_in_4bit,
            compute_dtype=config.compute_dtype,
            enable_thinking=config.enable_thinking,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
    raise ProviderConfigError(f"Unsupported LLM provider: {config.provider}")

def build_tts(config: TTSConfig) -> TTSProvider:
    if config.provider == "edge_tts":
        return EdgeTTSProvider(default_voice=config.default_voice)
    if config.provider == "kokoro":
        return KokoroTTSProvider(
            model_name=config.model,
            language_code=config.language_code,
            default_voice=config.default_voice,
            speed=config.speed,
            sample_rate=config.sample_rate,
            device=config.device,
        )
    if config.provider == "qwen3_tts_worker":
        return Qwen3TTSWorkerProvider(
            model_name=config.model,
            reference_audio=config.reference_audio,
            reference_text=config.reference_text,
            language=config.language_code,
            device=config.device,
            worker_python=config.worker_python,
            worker_script=config.worker_script,
            x_vector_only_mode=config.x_vector_only_mode,
            dtype=config.dtype,
            attention_implementation=config.attention_implementation,
            startup_timeout_seconds=config.startup_timeout_seconds,
        )
    raise ProviderConfigError(f"Unsupported TTS provider: {config.provider}")
