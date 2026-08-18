import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from voice_assistant.audio import MicrophoneStreamError
from voice_assistant.contracts import AudioChunk, PreparedResponse
from voice_assistant.realtime import RealtimeVoiceAssistant
from voice_assistant.observability import PerformanceLogger, measure_stage


class FakeRecorder:
    def __init__(self) -> None:
        self.output_path: Path | None = None

    def record(self, output_path: Path) -> Path:
        self.output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake input")
        return output_path


class FakePipeline:
    def __init__(
        self,
        reply: str = "你好，有什么可以帮你？",
    ) -> None:
        self.reply = reply
        self.audio_path: Path | None = None
        self.synthesized_texts: list[str] = []
        self.output_paths: list[Path] = []

    def prepare(self, audio_path: Path) -> PreparedResponse:
        self.audio_path = audio_path
        return PreparedResponse(
            transcript="你好",
            reply=self.reply,
        )

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        chunk_index: int = 1,
        chunk_count: int = 1,
    ) -> Path:
        self.synthesized_texts.append(text)
        self.output_paths.append(output_path)
        output_path.write_bytes(b"fake reply")
        return output_path


class FakePlayer:
    def __init__(self) -> None:
        self.audio_paths: list[Path] = []

    def play(self, audio_path: Path) -> None:
        self.audio_paths.append(audio_path)


class BlockingSecondChunkPipeline(FakePipeline):
    def __init__(self) -> None:
        super().__init__(reply="1234567890abcdefghij")
        self.second_started = threading.Event()
        self.second_may_finish = threading.Event()

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        chunk_index: int = 1,
        chunk_count: int = 1,
    ) -> Path:
        if chunk_index == 2:
            self.second_started.set()
            if not self.second_may_finish.wait(timeout=1):
                raise TimeoutError("second chunk did not overlap playback")
        return super().synthesize(
            text,
            output_path,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
        )


class CoordinatedPlayer(FakePlayer):
    def __init__(self, pipeline: BlockingSecondChunkPipeline) -> None:
        super().__init__()
        self._pipeline = pipeline
        self.prefetch_started_during_playback = False

    def play(self, audio_path: Path) -> None:
        if not self.audio_paths:
            self.prefetch_started_during_playback = (
                self._pipeline.second_started.wait(timeout=1)
            )
            self._pipeline.second_may_finish.set()
        super().play(audio_path)


class InstrumentedFakePipeline(FakePipeline):
    def __init__(self, performance: PerformanceLogger) -> None:
        super().__init__(reply="1234567890abcdefghij")
        self._performance = performance

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        chunk_index: int = 1,
        chunk_count: int = 1,
    ) -> Path:
        with measure_stage(
            self._performance,
            "fake_tts",
            chunk_index=chunk_index,
            chunk_count=chunk_count,
        ):
            return super().synthesize(
                text,
                output_path,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
            )


class FakeStreamingPipeline(FakePipeline):
    @property
    def supports_streaming_tts(self) -> bool:
        return True

    def stream_synthesize(self, text: str):
        yield AudioChunk(b"\x01\x00\x02\x00", sample_rate=24000)
        yield AudioChunk(b"\x03\x00\x04\x00", sample_rate=24000)


class FakeStreamingPlayer(FakePlayer):
    def __init__(self) -> None:
        super().__init__()
        self.streamed_chunks: list[AudioChunk] = []

    def play_stream(self, chunks) -> None:
        self.streamed_chunks.extend(chunks)


class RealtimeVoiceAssistantTest(unittest.TestCase):
    def test_continues_after_recoverable_turn_error(self) -> None:
        assistant = RealtimeVoiceAssistant(
            pipeline=FakePipeline(),
            recorder=FakeRecorder(),
            player=FakePlayer(),
            output_dir=Path("turns"),
        )
        assistant.run_turn = Mock(
            side_effect=[RuntimeError("empty transcript"), KeyboardInterrupt],
        )

        with self.assertRaises(KeyboardInterrupt):
            assistant.run_forever()

        self.assertEqual(assistant.run_turn.call_count, 2)

    def test_reopens_microphone_after_stream_error(self) -> None:
        assistant = RealtimeVoiceAssistant(
            pipeline=FakePipeline(),
            recorder=FakeRecorder(),
            player=FakePlayer(),
            output_dir=Path("turns"),
        )
        assistant.run_turn = Mock(
            side_effect=[
                MicrophoneStreamError("stream stopped"),
                KeyboardInterrupt,
            ],
        )

        with (
            patch("voice_assistant.realtime.print") as print_mock,
            patch("voice_assistant.realtime.time.sleep") as sleep_mock,
            self.assertRaises(KeyboardInterrupt),
        ):
            assistant.run_forever()

        self.assertEqual(assistant.run_turn.call_count, 2)
        sleep_mock.assert_called_once_with(0.5)
        print_mock.assert_called_once_with(
            "麦克风连接已中断，正在重连：stream stopped"
        )

    def test_runs_record_inference_and_playback_in_order(self) -> None:
        recorder = FakeRecorder()
        pipeline = FakePipeline()
        player = FakePlayer()
        observed: list[PreparedResponse] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "turns"
            assistant = RealtimeVoiceAssistant(
                pipeline=pipeline,
                recorder=recorder,
                player=player,
                output_dir=output_dir,
                output_format="wav",
                result_observer=observed.append,
            )

            result = assistant.run_turn()

            expected_input = output_dir / "turn_0001_input.wav"
            expected_reply = output_dir / "turn_0001_reply.wav"
            self.assertEqual(recorder.output_path, expected_input)
            self.assertEqual(pipeline.audio_path, expected_input)
            self.assertEqual(pipeline.output_paths, [expected_reply])
            self.assertEqual(player.audio_paths, [expected_reply])

        self.assertEqual(result.transcript, "你好")
        self.assertEqual(
            observed,
            [PreparedResponse(transcript="你好", reply=result.reply)],
        )

    def test_prefetches_next_tts_chunk_while_current_chunk_plays(self) -> None:
        pipeline = BlockingSecondChunkPipeline()
        player = CoordinatedPlayer(pipeline)

        with tempfile.TemporaryDirectory() as temp_dir:
            assistant = RealtimeVoiceAssistant(
                pipeline=pipeline,
                recorder=FakeRecorder(),
                player=player,
                output_dir=Path(temp_dir) / "turns",
                reply_chunk_max_chars=10,
            )

            result = assistant.run_turn()

        self.assertTrue(player.prefetch_started_during_playback)
        self.assertEqual(pipeline.synthesized_texts, [
            "1234567890",
            "abcdefghij",
        ])
        self.assertEqual(player.audio_paths, list(result.audio_paths))
        self.assertEqual(len(result.audio_paths), 2)

    def test_streams_audio_and_saves_one_complete_wav(self) -> None:
        player = FakeStreamingPlayer()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "turns"
            assistant = RealtimeVoiceAssistant(
                pipeline=FakeStreamingPipeline(),
                recorder=FakeRecorder(),
                player=player,
                output_dir=output_dir,
            )

            result = assistant.run_turn()

            self.assertEqual(len(player.streamed_chunks), 2)
            self.assertEqual(
                result.audio_path,
                output_dir / "turn_0001_reply.wav",
            )
            self.assertTrue(result.audio_path.is_file())
            self.assertEqual(result.audio_paths, (result.audio_path,))

    def test_records_realtime_stage_sequence_with_one_turn_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            performance = PerformanceLogger(
                enabled=True,
                console=False,
                jsonl=True,
                log_dir=temp_path / "logs",
                session_id="realtime-session",
            )
            assistant = RealtimeVoiceAssistant(
                pipeline=FakePipeline(),
                recorder=FakeRecorder(),
                player=FakePlayer(),
                output_dir=temp_path / "turns",
                performance=performance,
            )

            assistant.run_turn()
            assistant.close()

            events = [
                json.loads(line)
                for line in (temp_path / "logs" / "performance.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(
            [event["stage"] for event in events],
            [
                "record",
                "response_prepare",
                "time_to_first_audio",
                "playback",
                "turn_total",
            ],
        )
        self.assertTrue(
            all(event["turn_id"] == "turn_0001" for event in events)
        )

    def test_preserves_turn_id_in_background_tts_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            performance = PerformanceLogger(
                enabled=True,
                console=False,
                jsonl=True,
                log_dir=temp_path / "logs",
                session_id="thread-context-session",
            )
            assistant = RealtimeVoiceAssistant(
                pipeline=InstrumentedFakePipeline(performance),
                recorder=FakeRecorder(),
                player=FakePlayer(),
                output_dir=temp_path / "turns",
                reply_chunk_max_chars=10,
                performance=performance,
            )

            assistant.run_turn()
            assistant.close()

            events = [
                json.loads(line)
                for line in (temp_path / "logs" / "performance.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        tts_events = [
            event for event in events if event["stage"] == "fake_tts"
        ]
        self.assertEqual(len(tts_events), 2)
        self.assertTrue(
            all(event["turn_id"] == "turn_0001" for event in tts_events)
        )

    def test_records_streaming_realtime_stage_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            performance = PerformanceLogger(
                enabled=True,
                console=False,
                jsonl=True,
                log_dir=temp_path / "logs",
                session_id="streaming-session",
            )
            assistant = RealtimeVoiceAssistant(
                pipeline=FakeStreamingPipeline(),
                recorder=FakeRecorder(),
                player=FakeStreamingPlayer(),
                output_dir=temp_path / "turns",
                performance=performance,
            )

            assistant.run_turn()
            assistant.close()

            events = [
                json.loads(line)
                for line in (temp_path / "logs" / "performance.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(
            [event["stage"] for event in events],
            [
                "record",
                "response_prepare",
                "time_to_first_audio",
                "playback_stream",
                "turn_total",
            ],
        )
        first_audio_event = events[2]
        self.assertTrue(first_audio_event["streaming_audio"])
        self.assertEqual(first_audio_event["sample_rate"], 24000)


if __name__ == "__main__":
    unittest.main()
