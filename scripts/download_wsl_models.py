from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from modelscope import snapshot_download


@dataclass(frozen=True)
class ModelDownload:
    repo_id: str
    local_dir: Path
    files: tuple[str, ...]


DOWNLOADS = {
    "kokoro": ModelDownload(
        repo_id="hexgrad/Kokoro-82M",
        local_dir=Path("models/Kokoro-82M"),
        files=(
            "config.json",
            "kokoro-v1_0.pth",
            "voices/zf_xiaoxiao.pt",
        ),
    ),
    "sensevoice": ModelDownload(
        repo_id="iic/SenseVoiceSmall",
        local_dir=Path("models/SenseVoiceSmall"),
        files=(
            "am.mvn",
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
            "config.yaml",
            "configuration.json",
            "model.pt",
            "tokens.json",
        ),
    ),
    "qwen": ModelDownload(
        repo_id="Qwen/Qwen3.5-4B",
        local_dir=Path("models/Qwen3.5-4B"),
        files=(
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors-00001-of-00002.safetensors",
            "model.safetensors-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ),
    ),
    "qwen_tts": ModelDownload(
        repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        local_dir=Path("models/Qwen3-TTS-12Hz-0.6B-Base"),
        files=(
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "preprocessor_config.json",
            "speech_tokenizer/config.json",
            "speech_tokenizer/configuration.json",
            "speech_tokenizer/model.safetensors",
            "speech_tokenizer/preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
    ),
}

DEFAULT_DOWNLOADS = ("sensevoice", "qwen", "qwen_tts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the local WSL model set from ModelScope.",
    )
    parser.add_argument(
        "models",
        nargs="*",
        choices=tuple(DOWNLOADS),
        default=list(DEFAULT_DOWNLOADS),
        help="Models to download; defaults to the complete local stack.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for name in args.models:
        download = DOWNLOADS[name]
        print(f"Downloading {name}: {download.repo_id}")
        model_path = snapshot_download(
            download.repo_id,
            local_dir=str(download.local_dir),
            allow_file_pattern=list(download.files),
        )
        print(f"Ready: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
