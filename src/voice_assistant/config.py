from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    output_dir: Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    asr: ASRConfig
    llm: LLMConfig
    tts: TTSConfig
    runtime: RuntimeConfig


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

    try:
        return AppConfig(
            asr=ASRConfig(
                provider=asr["provider"],
                model=asr["model"],
                language=asr.get("language", "auto"),
                use_itn=asr.get("use_itn", False),
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
            ),
            runtime=RuntimeConfig(
                output_dir=Path(runtime["output_dir"]),
            ),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required config key: {exc.args[0]}") from exc