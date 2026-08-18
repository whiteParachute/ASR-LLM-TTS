from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

from voice_assistant.config import load_config
from voice_assistant.contracts import AudioChunk, StreamingTTSProvider
from voice_assistant.providers.factory import build_tts


DEFAULT_TEXT = "今天天气不错，很适合出去走走。"


@dataclass(frozen=True, slots=True)
class TTSBenchmarkRun:
    run: int
    time_to_first_audio_seconds: float
    total_seconds: float
    audio_seconds: float
    first_chunk_audio_seconds: float
    real_time_factor: float
    chunk_count: int


Clock = Callable[[], float]


def benchmark_streaming_tts(
    provider: StreamingTTSProvider,
    text: str,
    runs: int,
    clock: Clock = perf_counter,
) -> list[TTSBenchmarkRun]:
    """Measure warmed streaming synthesis without playback or WAV I/O."""
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Benchmark text cannot be empty")
    if runs < 1:
        raise ValueError("Benchmark runs must be positive")

    results: list[TTSBenchmarkRun] = []
    for run_index in range(1, runs + 1):
        started_at = clock()
        chunks = provider.stream_synthesize(cleaned_text)
        try:
            first_chunk = next(chunks)
        except StopIteration as exc:
            raise RuntimeError("TTS provider returned no audio chunks") from exc
        first_audio_at = clock()

        audio_seconds = first_chunk.duration_ms / 1000
        chunk_count = 1
        for chunk in chunks:
            audio_seconds += chunk.duration_ms / 1000
            chunk_count += 1
        completed_at = clock()

        total_seconds = completed_at - started_at
        if audio_seconds <= 0:
            raise RuntimeError("TTS provider returned invalid audio duration")
        results.append(
            TTSBenchmarkRun(
                run=run_index,
                time_to_first_audio_seconds=first_audio_at - started_at,
                total_seconds=total_seconds,
                audio_seconds=audio_seconds,
                first_chunk_audio_seconds=first_chunk.duration_ms / 1000,
                real_time_factor=total_seconds / audio_seconds,
                chunk_count=chunk_count,
            )
        )

    return results


def summarize_runs(runs: list[TTSBenchmarkRun]) -> dict[str, float]:
    if not runs:
        raise ValueError("Cannot summarize an empty benchmark")
    return {
        "median_time_to_first_audio_seconds": median(
            run.time_to_first_audio_seconds for run in runs
        ),
        "median_total_seconds": median(run.total_seconds for run in runs),
        "median_audio_seconds": median(run.audio_seconds for run in runs),
        "median_first_chunk_audio_seconds": median(
            run.first_chunk_audio_seconds for run in runs
        ),
        "median_real_time_factor": median(
            run.real_time_factor for run in runs
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the configured streaming TTS provider.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/wsl_cuda.yaml"),
        help="Application YAML whose TTS section should be benchmarked.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text synthesized by every measured run.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of warmed synthesis runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path. Defaults to the configured log directory.",
    )
    return parser


def execute(
    config_path: Path,
    text: str,
    runs: int,
    output_path: Path | None,
) -> tuple[dict[str, object], Path]:
    config = load_config(config_path)

    load_started_at = perf_counter()
    provider = build_tts(config.tts)
    load_seconds = perf_counter() - load_started_at
    if not hasattr(provider, "stream_synthesize"):
        close = getattr(provider, "close", None)
        if callable(close):
            close()
        raise ValueError(
            f"TTS provider is not streaming: {config.tts.provider}"
        )

    try:
        measured_runs = benchmark_streaming_tts(
            provider,
            text=text,
            runs=runs,
        )
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    created_at = datetime.now(UTC)
    if output_path is None:
        timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
        output_path = (
            config.observability.log_dir / f"tts-benchmark-{timestamp}.json"
        )
    resolved_output = output_path.expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": created_at.isoformat(),
        "provider": config.tts.provider,
        "model": config.tts.model,
        "text_characters": len(text.strip()),
        "runs_requested": runs,
        "model_load_seconds": load_seconds,
        "runs": [asdict(run) for run in measured_runs],
        "summary": summarize_runs(measured_runs),
    }
    resolved_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report, resolved_output


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report, output_path = execute(
            config_path=args.config,
            text=args.text,
            runs=args.runs,
            output_path=args.output,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    print(f"TTS provider: {report['provider']}")
    print(f"TTS model: {report['model']}")
    print(f"Model load: {report['model_load_seconds']:.3f}s")
    for run in report["runs"]:
        print(
            "run {run}: first={time_to_first_audio_seconds:.3f}s "
            "total={total_seconds:.3f}s audio={audio_seconds:.3f}s "
            "first_chunk={first_chunk_audio_seconds:.3f}s "
            "rtf={real_time_factor:.3f} chunks={chunk_count}".format(
                **run
            )
        )
    summary = report["summary"]
    print(
        "median: first={median_time_to_first_audio_seconds:.3f}s "
        "total={median_total_seconds:.3f}s "
        "rtf={median_real_time_factor:.3f}".format(**summary)
    )
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
