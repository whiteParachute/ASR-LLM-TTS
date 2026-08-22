from __future__ import annotations

import argparse
import queue
import time
import wave
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from threading import Thread
from typing import Iterator, Protocol

from voice_assistant.audio import (
    MicrophoneStreamError,
    PaplayAudioPlayer,
    SoundDeviceAudioPlayer,
    SoundDeviceVADRecorder,
)
from voice_assistant.bootstrap import build_pipeline
from voice_assistant.config import AppConfig, load_config
from voice_assistant.contracts import (
    AudioPlayer,
    AudioChunk,
    PipelineResult,
    PreparedResponse,
    UtteranceRecorder,
)
from voice_assistant.observability import (
    PerformanceLogger,
    build_performance_logger,
    measure_stage,
    wav_duration_ms,
)
from voice_assistant.text_chunking import (
    split_reply_text,
    stream_reply_text,
)


class TurnPipeline(Protocol):
    def prepare(self, audio_path: Path) -> PreparedResponse:
        ...

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        chunk_index: int = 1,
        chunk_count: int = 1,
    ) -> Path:
        ...

    @property
    def supports_streaming_tts(self) -> bool:
        ...

    def stream_synthesize(self, text: str) -> Iterator[AudioChunk]:
        ...

    @property
    def supports_streaming_llm(self) -> bool:
        ...

    def transcribe(self, audio_path: Path) -> str:
        ...

    def stream_reply(self, transcript: str) -> Iterator[str]:
        ...


ResultObserver = Callable[[PreparedResponse], None]


@dataclass(slots=True)
class _AudioStreamStats:
    chunk_count: int = 0
    duration_ms: float = 0.0
    max_chunk_gap_ms: float = 0.0
    min_buffer_ahead_ms: float | None = None
    estimated_underflow_ms: float = 0.0


_TEXT_STREAM_END = object()


@dataclass(slots=True)
class _TextStreamState:
    queue: queue.Queue[object]
    raw_parts: list[str]
    segments: list[str]
    thread: Thread


class RealtimeVoiceAssistant:
    """Run automatic microphone-to-reply turns in a synchronous loop."""

    def __init__(
        self,
        pipeline: TurnPipeline,
        recorder: UtteranceRecorder,
        player: AudioPlayer,
        output_dir: Path,
        output_format: str = "wav",
        reply_chunk_max_chars: int = 18,
        stream_llm_to_tts: bool = False,
        first_reply_chunk_chars: int = 6,
        result_observer: ResultObserver | None = None,
        performance: PerformanceLogger | None = None,
    ) -> None:
        if reply_chunk_max_chars < 1:
            raise ValueError("reply_chunk_max_chars must be at least 1.")
        if first_reply_chunk_chars < 1:
            raise ValueError("first_reply_chunk_chars must be at least 1.")
        if first_reply_chunk_chars > reply_chunk_max_chars:
            raise ValueError(
                "first_reply_chunk_chars cannot exceed "
                "reply_chunk_max_chars."
            )
        self._pipeline = pipeline
        self._recorder = recorder
        self._player = player
        self._output_dir = output_dir
        self._output_format = output_format.lstrip(".")
        self._reply_chunk_max_chars = reply_chunk_max_chars
        self._stream_llm_to_tts = stream_llm_to_tts
        self._first_reply_chunk_chars = first_reply_chunk_chars
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
            ) as turn_span:
                if self._can_stream_text_and_audio():
                    result = self._run_streaming_text_audio_turn(
                        recorded_path=recorded_path,
                        reply_path=reply_path,
                    )
                    turn_span.add_fields(
                        reply_chunks=len(result.audio_paths),
                        streaming_audio=True,
                        streaming_text=True,
                    )
                    return result

                if self._can_stream_audio():
                    result = self._run_streaming_turn(
                        recorded_path=recorded_path,
                        reply_path=reply_path,
                    )
                    turn_span.add_fields(
                        reply_chunks=len(result.audio_paths),
                        streaming_audio=True,
                    )
                    return result

                result = self._run_chunked_turn(
                    recorded_path=recorded_path,
                    reply_path=reply_path,
                )
                turn_span.add_fields(
                    reply_chunks=len(result.audio_paths),
                    streaming_audio=False,
                )
                return result

    def _run_streaming_text_audio_turn(
        self,
        *,
        recorded_path: Path,
        reply_path: Path,
    ) -> PipelineResult:
        state: _TextStreamState | None = None
        try:
            with measure_stage(
                self._performance,
                "time_to_first_audio",
                streaming_audio=True,
                streaming_text=True,
            ) as first_audio_span:
                transcript = self._pipeline.transcribe(recorded_path)
                state = self._start_text_stream(transcript)
                audio_chunks = self._stream_text_audio(state)
                try:
                    first_chunk = next(audio_chunks)
                except StopIteration as exc:
                    raise RuntimeError(
                        "Streaming LLM/TTS returned no audio chunks"
                    ) from exc
                first_audio_span.add_fields(
                    first_chunk_duration_ms=round(
                        first_chunk.duration_ms,
                        3,
                    ),
                    first_text_chars=len(state.segments[0]),
                    sample_rate=first_chunk.sample_rate,
                )

            stats = _AudioStreamStats()
            recorded_chunks = _record_audio_stream(
                chunks=chain((first_chunk,), audio_chunks),
                output_path=reply_path,
                stats=stats,
            )
            play_stream = getattr(self._player, "play_stream")
            with measure_stage(
                self._performance,
                "playback_stream",
                streaming_audio=True,
                streaming_text=True,
            ) as playback_span:
                play_stream(recorded_chunks)
                playback_span.add_fields(
                    chunk_count=stats.chunk_count,
                    audio_duration_ms=round(stats.duration_ms, 3),
                    max_chunk_gap_ms=round(stats.max_chunk_gap_ms, 3),
                    min_buffer_ahead_ms=(
                        round(stats.min_buffer_ahead_ms, 3)
                        if stats.min_buffer_ahead_ms is not None
                        else None
                    ),
                    estimated_underflow_ms=round(
                        stats.estimated_underflow_ms,
                        3,
                    ),
                )
        finally:
            if state is not None:
                state.thread.join()

        if state is None:
            raise RuntimeError("Streaming text state was not initialized")
        reply = "".join(state.raw_parts).strip()
        if not reply:
            raise RuntimeError("Streaming LLM returned an empty reply")
        prepared = PreparedResponse(transcript=transcript, reply=reply)
        if self._result_observer is not None:
            self._result_observer(prepared)

        return PipelineResult(
            transcript=transcript,
            reply=reply,
            audio_path=reply_path,
            audio_paths=(reply_path,),
        )

    def _start_text_stream(self, transcript: str) -> _TextStreamState:
        text_queue: queue.Queue[object] = queue.Queue()
        raw_parts: list[str] = []
        segments: list[str] = []

        def capture_parts() -> Iterator[str]:
            for part in self._pipeline.stream_reply(transcript):
                raw_parts.append(part)
                yield part

        def produce() -> None:
            try:
                segment_iterator = iter(
                    stream_reply_text(
                        capture_parts(),
                        first_chunk_chars=self._first_reply_chunk_chars,
                        max_chars=self._reply_chunk_max_chars,
                    )
                )
                with measure_stage(
                    self._performance,
                    "llm_first_segment",
                    target_chars=self._first_reply_chunk_chars,
                ) as first_segment_span:
                    try:
                        first_segment = next(segment_iterator)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "Streaming LLM returned an empty reply"
                        ) from exc
                    first_segment_span.add_fields(
                        output_chars=len(first_segment),
                    )
                segments.append(first_segment)
                text_queue.put(first_segment)
                for segment in segment_iterator:
                    segments.append(segment)
                    text_queue.put(segment)
            except BaseException as exc:
                text_queue.put(exc)
            finally:
                text_queue.put(_TEXT_STREAM_END)

        context = copy_context()
        producer = Thread(
            target=context.run,
            args=(produce,),
            name="llm-text-segments",
            daemon=True,
        )
        state = _TextStreamState(
            queue=text_queue,
            raw_parts=raw_parts,
            segments=segments,
            thread=producer,
        )
        producer.start()
        return state

    def _stream_text_audio(
        self,
        state: _TextStreamState,
    ) -> Iterator[AudioChunk]:
        while True:
            item = state.queue.get()
            if item is _TEXT_STREAM_END:
                return
            if isinstance(item, BaseException):
                raise item
            if not isinstance(item, str):
                raise RuntimeError("Streaming LLM returned invalid text")
            yield from self._pipeline.stream_synthesize(item)

    def _run_chunked_turn(
        self,
        *,
        recorded_path: Path,
        reply_path: Path,
    ) -> PipelineResult:
        with measure_stage(
            self._performance,
            "time_to_first_audio",
        ) as first_audio_span:
            with measure_stage(
                self._performance,
                "response_prepare",
            ):
                prepared = self._pipeline.prepare(recorded_path)

            chunks = split_reply_text(
                prepared.reply,
                max_chars=self._reply_chunk_max_chars,
            )
            if not chunks:
                raise ValueError("LLM returned an empty reply.")
            chunk_paths = _build_chunk_paths(reply_path, len(chunks))

            if self._result_observer is not None:
                self._result_observer(prepared)

            first_path = self._pipeline.synthesize(
                chunks[0],
                chunk_paths[0],
                chunk_index=1,
                chunk_count=len(chunks),
            )
            first_audio_span.add_fields(
                reply_chunks=len(chunks),
                first_chunk_chars=len(chunks[0]),
                audio_duration_ms=wav_duration_ms(first_path),
            )

        generated_paths = self._play_with_prefetch(
            chunks=chunks,
            chunk_paths=chunk_paths,
            first_path=first_path,
        )
        return PipelineResult(
            transcript=prepared.transcript,
            reply=prepared.reply,
            audio_path=generated_paths[0],
            audio_paths=tuple(generated_paths),
            sources=prepared.sources,
        )

    def _run_streaming_turn(
        self,
        *,
        recorded_path: Path,
        reply_path: Path,
    ) -> PipelineResult:
        with measure_stage(
            self._performance,
            "time_to_first_audio",
            streaming_audio=True,
        ) as first_audio_span:
            with measure_stage(
                self._performance,
                "response_prepare",
            ):
                prepared = self._pipeline.prepare(recorded_path)

            if self._result_observer is not None:
                self._result_observer(prepared)

            audio_chunks = self._pipeline.stream_synthesize(prepared.reply)
            try:
                first_chunk = next(audio_chunks)
            except StopIteration as exc:
                raise RuntimeError(
                    "Streaming TTS returned no audio chunks"
                ) from exc
            first_audio_span.add_fields(
                first_chunk_duration_ms=round(
                    first_chunk.duration_ms,
                    3,
                ),
                sample_rate=first_chunk.sample_rate,
            )

        stats = _AudioStreamStats()
        recorded_chunks = _record_audio_stream(
            chunks=chain((first_chunk,), audio_chunks),
            output_path=reply_path,
            stats=stats,
        )
        play_stream = getattr(self._player, "play_stream")
        with measure_stage(
            self._performance,
            "playback_stream",
            streaming_audio=True,
        ) as playback_span:
            play_stream(recorded_chunks)
            playback_span.add_fields(
                chunk_count=stats.chunk_count,
                audio_duration_ms=round(stats.duration_ms, 3),
                max_chunk_gap_ms=round(stats.max_chunk_gap_ms, 3),
                min_buffer_ahead_ms=(
                    round(stats.min_buffer_ahead_ms, 3)
                    if stats.min_buffer_ahead_ms is not None
                    else None
                ),
                estimated_underflow_ms=round(
                    stats.estimated_underflow_ms,
                    3,
                ),
            )

        return PipelineResult(
            transcript=prepared.transcript,
            reply=prepared.reply,
            audio_path=reply_path,
            audio_paths=(reply_path,),
            sources=prepared.sources,
        )

    def _can_stream_audio(self) -> bool:
        return bool(
            getattr(self._pipeline, "supports_streaming_tts", False)
        ) and callable(getattr(self._player, "play_stream", None))

    def _can_stream_text_and_audio(self) -> bool:
        return (
            self._stream_llm_to_tts
            and bool(getattr(self._pipeline, "supports_streaming_llm", False))
            and self._can_stream_audio()
        )

    def _play_with_prefetch(
        self,
        *,
        chunks: list[str],
        chunk_paths: list[Path],
        first_path: Path,
    ) -> list[Path]:
        generated_paths = [first_path]
        current_path = first_path

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tts-prefetch",
        ) as executor:
            for index in range(len(chunks)):
                next_audio: Future[Path] | None = None
                if index + 1 < len(chunks):
                    next_audio = _submit_synthesis(
                        executor=executor,
                        pipeline=self._pipeline,
                        text=chunks[index + 1],
                        output_path=chunk_paths[index + 1],
                        chunk_index=index + 2,
                        chunk_count=len(chunks),
                    )

                with measure_stage(
                    self._performance,
                    "playback",
                    audio_duration_ms=wav_duration_ms(current_path),
                    chunk_index=index + 1,
                    chunk_count=len(chunks),
                ):
                    self._player.play(current_path)

                if next_audio is not None:
                    current_path = next_audio.result()
                    generated_paths.append(current_path)

        return generated_paths

    def run_forever(self) -> None:
        while True:
            try:
                self.run_turn()
            except TimeoutError:
                continue
            except MicrophoneStreamError as exc:
                print(f"麦克风连接已中断，正在重连：{exc}")
                time.sleep(0.5)
                continue
            except (ValueError, RuntimeError) as exc:
                print(f"本轮已跳过：{exc}")
                continue

    def close(self) -> None:
        try:
            close = getattr(self._pipeline, "close", None)
            if callable(close):
                close()
        finally:
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
        player: AudioPlayer = PaplayAudioPlayer(
            stream_latency_ms=config.audio.playback_latency_ms,
            stream_process_time_ms=(
                config.audio.playback_process_time_ms
            ),
            stream_tail_guard_ms=config.audio.playback_tail_guard_ms,
        )
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
        reply_chunk_max_chars=config.runtime.reply_chunk_max_chars,
        stream_llm_to_tts=config.runtime.stream_llm_to_tts,
        first_reply_chunk_chars=config.runtime.first_reply_chunk_chars,
        result_observer=_print_result,
        performance=performance,
    )


def _print_result(result: PreparedResponse) -> None:
    print(f"识别文本：{result.transcript}")
    print(f"模型回复：{result.reply}")
    if result.sources:
        print("参考来源：")
        for index, source in enumerate(result.sources, start=1):
            print(f"  {index}. {source.title} - {source.url}")


def _build_chunk_paths(reply_path: Path, chunk_count: int) -> list[Path]:
    if chunk_count == 1:
        return [reply_path]
    return [
        reply_path.with_name(
            f"{reply_path.stem}_{index:03d}{reply_path.suffix}"
        )
        for index in range(1, chunk_count + 1)
    ]


def _submit_synthesis(
    *,
    executor: ThreadPoolExecutor,
    pipeline: TurnPipeline,
    text: str,
    output_path: Path,
    chunk_index: int,
    chunk_count: int,
) -> Future[Path]:
    context = copy_context()
    return executor.submit(
        context.run,
        lambda: pipeline.synthesize(
            text,
            output_path,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
        ),
    )


def _record_audio_stream(
    *,
    chunks: Iterator[AudioChunk],
    output_path: Path,
    stats: _AudioStreamStats,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[AudioChunk]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_sample_rate: int | None = None
    expected_channels: int | None = None
    stream_started_at: float | None = None
    previous_chunk_at: float | None = None

    with wave.open(str(output_path), "wb") as audio_file:
        for chunk in chunks:
            chunk_arrived_at = clock()
            if stream_started_at is None:
                stream_started_at = chunk_arrived_at
            else:
                elapsed_ms = (
                    chunk_arrived_at - stream_started_at
                ) * 1000
                buffer_ahead_ms = stats.duration_ms - elapsed_ms
                if stats.min_buffer_ahead_ms is None:
                    stats.min_buffer_ahead_ms = buffer_ahead_ms
                else:
                    stats.min_buffer_ahead_ms = min(
                        stats.min_buffer_ahead_ms,
                        buffer_ahead_ms,
                    )
                stats.estimated_underflow_ms = max(
                    stats.estimated_underflow_ms,
                    -buffer_ahead_ms,
                )
            if previous_chunk_at is not None:
                stats.max_chunk_gap_ms = max(
                    stats.max_chunk_gap_ms,
                    (chunk_arrived_at - previous_chunk_at) * 1000,
                )
            previous_chunk_at = chunk_arrived_at

            if expected_sample_rate is None:
                expected_sample_rate = chunk.sample_rate
                expected_channels = chunk.channels
                audio_file.setnchannels(chunk.channels)
                audio_file.setsampwidth(2)
                audio_file.setframerate(chunk.sample_rate)
            elif (
                chunk.sample_rate != expected_sample_rate
                or chunk.channels != expected_channels
            ):
                raise RuntimeError(
                    "Streaming TTS changed audio format between chunks"
                )

            audio_file.writeframesraw(chunk.pcm_s16le)
            stats.chunk_count += 1
            stats.duration_ms += chunk.duration_ms
            yield chunk


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
