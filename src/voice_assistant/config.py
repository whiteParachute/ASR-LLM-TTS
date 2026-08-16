from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

class ConfigError(ValueError):
    """Raised when there is an error in the configuration."""
    pass


@dataclass(frozen=True, slots=True)
class ASRConfig:
    provider: str
    model: str
    language: str = "auto"
    use_itn: bool = False
    device: str | None = None


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str
    model: str
    max_new_tokens: int
    system_prompt: str
    reply_instruction: str = ""


@dataclass(frozen=True, slots=True)
class TTSConfig:
    provider: str
    default_voice: str
    model: str = "hexgrad/Kokoro-82M"
    language_code: str = "z"
    speed: float = 1.0
    sample_rate: int = 24000
    output_format: str = "wav"
    device: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    output_dir: Path


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    enabled: bool = False
    console: bool = True
    jsonl: bool = True
    log_dir: Path = Path("logs")


@dataclass(frozen=True, slots=True)
class AudioConfig:
    playback_backend: str = "sounddevice"
    sample_rate: int = 16000
    frame_duration_ms: int = 20
    vad_mode: int = 2
    start_trigger_ms: int = 60
    end_silence_ms: int = 800
    pre_roll_ms: int = 200
    speech_timeout_seconds: float = 30.0
    max_utterance_seconds: float = 30.0
    input_device: int | str | None = None
    output_device: int | str | None = None


@dataclass(frozen=True, slots=True)
class AppConfig:
    asr: ASRConfig
    llm: LLMConfig
    tts: TTSConfig
    runtime: RuntimeConfig
    audio: AudioConfig = field(default_factory=AudioConfig)
    observability: ObservabilityConfig = field(
        default_factory=ObservabilityConfig,
    )


def _get_section(
    raw_config: dict[str, Any], section_name: str
) -> dict[str, Any]:
    section = raw_config.get(section_name)
    if not isinstance(section, dict):
        raise ConfigError(f"Missing or Invalid config session: {section_name}")
    return section

def load_config(config_path: Path) -> AppConfig:
    """Load the configuration from a YAML file."""
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except OSError as exc:
        raise ConfigError(f"Unable to read config file: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML file: {config_path}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigError(f"Config root must be a mapping")

    asr = _get_section(raw_config, "asr")
    llm = _get_section(raw_config, "llm")
    tts = _get_section(raw_config, "tts")
    runtime = _get_section(raw_config, "runtime")
    audio = raw_config.get("audio", {})
    if not isinstance(audio, dict):
        raise ConfigError("Missing or Invalid config session: audio")
    observability = raw_config.get("observability", {})
    if not isinstance(observability, dict):
        raise ConfigError(
            "Missing or Invalid config session: observability"
        )

    try:
        return AppConfig(
            asr=ASRConfig(
                provider=asr["provider"],
                model=asr["model"],
                language=asr.get("language", "auto"),
                use_itn=asr.get("use_itn", False),
                device=asr.get("device"),
            ),
            llm=LLMConfig(
                provider=llm["provider"],
                model=llm["model"],
                max_new_tokens=llm["max_new_tokens"],
                system_prompt=llm["system_prompt"],
                reply_instruction=llm.get("reply_instruction", ""),
            ),
            tts=TTSConfig(
                provider=tts["provider"],
                default_voice=tts["default_voice"],
                model=tts.get("model", "hexgrad/Kokoro-82M"),
                language_code=tts.get("language_code", "z"),
                speed=tts.get("speed", 1.0),
                sample_rate=tts.get("sample_rate", 24000),
                output_format=tts.get("output_format", "wav"),
                device=tts.get("device"),
            ),
            runtime=RuntimeConfig(
                output_dir=Path(runtime["output_dir"]),
            ),
            audio=AudioConfig(
                playback_backend=audio.get(
                    "playback_backend",
                    "sounddevice",
                ),
                sample_rate=audio.get("sample_rate", 16000),
                frame_duration_ms=audio.get("frame_duration_ms", 20),
                vad_mode=audio.get("vad_mode", 2),
                start_trigger_ms=audio.get("start_trigger_ms", 60),
                end_silence_ms=audio.get("end_silence_ms", 800),
                pre_roll_ms=audio.get("pre_roll_ms", 200),
                speech_timeout_seconds=audio.get(
                    "speech_timeout_seconds",
                    30.0,
                ),
                max_utterance_seconds=audio.get(
                    "max_utterance_seconds",
                    30.0,
                ),
                input_device=audio.get("input_device"),
                output_device=audio.get("output_device"),
            ),
            observability=ObservabilityConfig(
                enabled=observability.get("enabled", False),
                console=observability.get("console", True),
                jsonl=observability.get("jsonl", True),
                log_dir=Path(observability.get("log_dir", "logs")),
            ),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required config key: {exc.args[0]}") from exc
