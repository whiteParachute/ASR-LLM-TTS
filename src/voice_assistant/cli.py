from __future__ import annotations

import argparse
from pathlib import Path

from voice_assistant.bootstrap import build_pipeline
from voice_assistant.config import load_config
from voice_assistant.contracts import PipelineResult


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the ASR-LLM-TTS voice assistant.",
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to the input audio file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the generated speech file.",
    )
    return parser


def execute(
    config_path: Path,
    audio_path: Path,
    output_path: Path | None = None,
) -> PipelineResult:
    """Run one ASR-LLM-TTS turn from an existing audio file."""
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Input audio file does not exist: {audio_path}"
        )

    config = load_config(config_path)

    if output_path is None:
        output_path = (
            config.runtime.output_dir
            / f"{audio_path.stem}_reply.{config.tts.output_format}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline(config)

    return pipeline.run(
        audio_path=audio_path,
        output_path=output_path,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = execute(
            config_path=args.config,
            audio_path=args.audio,
            output_path=args.output,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(
            status=1,
            message=f"error: {exc}\n",
        )

    print(f"识别文本：{result.transcript}")
    print(f"模型回复：{result.reply}")
    print(f"语音文件：{result.audio_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
