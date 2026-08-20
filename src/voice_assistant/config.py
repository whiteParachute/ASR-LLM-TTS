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
    compute_dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    max_new_tokens: int = 256
    prompt: str = ""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str
    model: str
    max_new_tokens: int
    system_prompt: str
    reply_instruction: str = ""
    load_in_4bit: bool = False
    compute_dtype: str = "float16"
    enable_thinking: bool = False
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20


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
    worker_python: str = ".venv-tts/bin/python"
    worker_script: str = "scripts/qwen3_tts_worker.py"
    runtime_dir: Path = Path(".runtime/CosyVoice")
    reference_audio: Path | None = None
    reference_text: str = ""
    x_vector_only_mode: bool = False
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    max_new_tokens: int = 256
    startup_timeout_seconds: float = 180.0
    warmup_text: str = "你好，很高兴和你对话。"
    inference_mode: str = "zero_shot"
    speaker: str = ""
    load_jit: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    output_dir: Path
    reply_chunk_max_chars: int = 18
    stream_llm_to_tts: bool = False
    first_reply_chunk_chars: int = 6


@dataclass(frozen=True, slots=True)
class ToolUseConfig:
    enabled: bool = False
    max_rounds: int = 3
    timeout_seconds: float = 2.0
    max_result_chars: int = 2000


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    enabled: bool = False
    console: bool = True
    jsonl: bool = True
    log_dir: Path = Path("logs")


@dataclass(frozen=True, slots=True)
class AudioConfig:
    playback_backend: str = "sounddevice"
    playback_latency_ms: int | None = None
    playback_process_time_ms: int | None = None
    playback_tail_guard_ms: int = 0
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
    tools: ToolUseConfig = field(default_factory=ToolUseConfig)
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
    tools = raw_config.get("tools", {})
    if not isinstance(tools, dict):
        raise ConfigError("Missing or Invalid config session: tools")
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
                compute_dtype=asr.get("compute_dtype", "bfloat16"),
                attention_implementation=asr.get(
                    "attention_implementation",
                    "sdpa",
                ),
                max_new_tokens=asr.get("max_new_tokens", 256),
                prompt=asr.get("prompt", ""),
            ),
            llm=LLMConfig(
                provider=llm["provider"],
                model=llm["model"],
                max_new_tokens=llm["max_new_tokens"],
                system_prompt=llm["system_prompt"],
                reply_instruction=llm.get("reply_instruction", ""),
                load_in_4bit=llm.get("load_in_4bit", False),
                compute_dtype=llm.get("compute_dtype", "float16"),
                enable_thinking=llm.get("enable_thinking", False),
                do_sample=llm.get("do_sample", True),
                temperature=llm.get("temperature", 0.7),
                top_p=llm.get("top_p", 0.8),
                top_k=llm.get("top_k", 20),
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
                worker_python=tts.get(
                    "worker_python",
                    ".venv-tts/bin/python",
                ),
                worker_script=tts.get(
                    "worker_script",
                    "scripts/qwen3_tts_worker.py",
                ),
                runtime_dir=Path(
                    tts.get("runtime_dir", ".runtime/CosyVoice")
                ),
                reference_audio=(
                    Path(tts["reference_audio"])
                    if tts.get("reference_audio")
                    else None
                ),
                reference_text=tts.get("reference_text", ""),
                x_vector_only_mode=tts.get(
                    "x_vector_only_mode",
                    False,
                ),
                dtype=tts.get("dtype", "bfloat16"),
                attention_implementation=tts.get(
                    "attention_implementation",
                    "sdpa",
                ),
                max_new_tokens=tts.get("max_new_tokens", 256),
                startup_timeout_seconds=tts.get(
                    "startup_timeout_seconds",
                    180.0,
                ),
                warmup_text=tts.get(
                    "warmup_text",
                    "你好，很高兴和你对话。",
                ),
                inference_mode=tts.get("inference_mode", "zero_shot"),
                speaker=tts.get("speaker", ""),
                load_jit=tts.get("load_jit", False),
            ),
            runtime=RuntimeConfig(
                output_dir=Path(runtime["output_dir"]),
                reply_chunk_max_chars=runtime.get(
                    "reply_chunk_max_chars",
                    18,
                ),
                stream_llm_to_tts=runtime.get(
                    "stream_llm_to_tts",
                    False,
                ),
                first_reply_chunk_chars=runtime.get(
                    "first_reply_chunk_chars",
                    6,
                ),
            ),
            tools=ToolUseConfig(
                enabled=tools.get("enabled", False),
                max_rounds=tools.get("max_rounds", 3),
                timeout_seconds=tools.get("timeout_seconds", 2.0),
                max_result_chars=tools.get("max_result_chars", 2000),
            ),
            audio=AudioConfig(
                playback_backend=audio.get(
                    "playback_backend",
                    "sounddevice",
                ),
                playback_latency_ms=audio.get("playback_latency_ms"),
                playback_process_time_ms=audio.get(
                    "playback_process_time_ms"
                ),
                playback_tail_guard_ms=audio.get(
                    "playback_tail_guard_ms",
                    0,
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
