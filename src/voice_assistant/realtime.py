from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Protocol

from voice_assistant.audio import (
    PaplayAudioPlayer,
    SoundDeviceAudioPlayer,
    SoundDeviceVADRecorder,
)
from voice_assistant.bootstrap import build_pipeline
from voice_assistant.config import AppConfig, load_config
from voice_assistant.contracts import (
    AudioPlayer,
    PipelineResult,
    UtteranceRecorder,
)
from voice_assistant.observability import (
    PerformanceLogger,
    build_performance_logger,
    measure_stage,
    wav_duration_ms,
)


class TurnPipeline(Protocol):
    def run(self, audio_path: Path, output_path: Path) -> PipelineResult:
        ...


ResultObserver = Callable[[PipelineResult], None]


class RealtimeVoiceAssistant:
    """Run automatic microphone-to-reply turns in a synchronous loop."""

    def __init__(
        self,
        pipeline: TurnPipeline,
        recorder: UtteranceRecorder,
        player: AudioPlayer,
        output_dir: Path,
        output_format: str = "wav",
        result_observer: ResultObserver | None = None,
        performance: PerformanceLogger | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._recorder = recorder
        self._player = player
        self._output_dir = output_dir
        self._output_format = output_format.lstrip(".")
        self._result_observer = result_observer
        self._performance = performance
        self._turn_number = 0

    @property
    def performance_log_path(self) -> Path | None:
        if self._performance is None:
            return None
        return self._performance.log_path

    def run_turn(self) -> PipelineResult:
        self._turn_number += 1
        turn_name = f"turn_{self._turn_number:04d}"
        input_path = self._output_dir / f"{turn_name}_input.wav"
        reply_path = (
            self._output_dir
            / f"{turn_name}_reply.{self._output_format}"
        )

        turn_context = (
            self._performance.turn(turn_name)
            if self._performance is not None
            else nullcontext()
        )
        with turn_context:
            with measure_stage(
                self._performance,
                "record",
            ) as record_span:
                recorded_path = self._recorder.record(input_path)
                input_audio_duration = wav_duration_ms(recorded_path)
                record_span.add_fields(
                    audio_duration_ms=input_audio_duration,
                )

            with measure_stage(
                self._performance,
                "turn_total",
                audio_duration_ms=input_audio_duration,
            ):
                with measure_stage(
                    self._performance,
                    "response_prepare",
                ):
                    result = self._pipeline.run(
                        audio_path=recorded_path,
                        output_path=reply_path,
                    )

                if self._result_observer is not None:
                    self._result_observer(result)
                with measure_stage(
                    self._performance,
                    "playback",
                    audio_duration_ms=wav_duration_ms(result.audio_path),
                ):
                    self._player.play(result.audio_path)
                return result

    def run_forever(self) -> None:
        while True:
            try:
                self.run_turn()
            except TimeoutError:
                continue
            except (ValueError, RuntimeError) as exc:
                print(f"本轮已跳过：{exc}")
                continue

    def close(self) -> None:
        if self._performance is not None:
            self._performance.close()


def build_realtime_assistant(config: AppConfig) -> RealtimeVoiceAssistant:
    performance = build_performance_logger(config.observability)
    with measure_stage(
        performance,
        "model_load",
        asr_model=config.asr.model,
        llm_model=config.llm.model,
        tts_model=config.tts.model,
    ):
        pipeline = build_pipeline(config, performance=performance)
    recorder = SoundDeviceVADRecorder(
        sample_rate=config.audio.sample_rate,
        frame_duration_ms=config.audio.frame_duration_ms,
        vad_mode=config.audio.vad_mode,
        start_trigger_ms=config.audio.start_trigger_ms,
        end_silence_ms=config.audio.end_silence_ms,
        pre_roll_ms=config.audio.pre_roll_ms,
        speech_timeout_seconds=config.audio.speech_timeout_seconds,
        max_utterance_seconds=config.audio.max_utterance_seconds,
        input_device=config.audio.input_device,
    )
    if config.audio.playback_backend == "paplay":
        player: AudioPlayer = PaplayAudioPlayer()
    elif config.audio.playback_backend == "sounddevice":
        player = SoundDeviceAudioPlayer(
            output_device=config.audio.output_device,
        )
    else:
        raise ValueError(
            "Unsupported audio playback backend: "
            f"{config.audio.playback_backend}"
        )

    return RealtimeVoiceAssistant(
        pipeline=pipeline,
        recorder=recorder,
        player=player,
        output_dir=config.runtime.output_dir / "turns",
        output_format=config.tts.output_format,
        result_observer=_print_result,
        performance=performance,
    )


def _print_result(result: PipelineResult) -> None:
    print(f"识别文本：{result.transcript}")
    print(f"模型回复：{result.reply}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run continuous local voice conversations.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)

    print("正在加载本地模型……")
    assistant = build_realtime_assistant(config)
    print("模型加载完成，请对着麦克风说话；按 Ctrl+C 退出。")
    if assistant.performance_log_path is not None:
        print(f"性能日志：{assistant.performance_log_path}")

    try:
        assistant.run_forever()
    except KeyboardInterrupt:
        print("\n语音助手已停止。")
    finally:
        assistant.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
