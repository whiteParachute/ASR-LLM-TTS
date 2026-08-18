import unittest
from pathlib import Path

from voice_assistant.config import load_config
from voice_assistant.contracts import AudioChunk
from voice_assistant.tts_benchmark import (
    benchmark_streaming_tts,
    override_reference_voice,
    summarize_runs,
)


def chunk(milliseconds: int) -> AudioChunk:
    sample_rate = 1000
    samples = milliseconds
    return AudioChunk(
        pcm_s16le=b"\x00\x00" * samples,
        sample_rate=sample_rate,
    )


class FakeStreamingTTS:
    def stream_synthesize(self, text: str):
        self.text = text
        yield chunk(400)
        yield chunk(600)

    def synthesize(self, text, output_path):
        raise NotImplementedError


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class TTSBenchmarkTest(unittest.TestCase):
    def test_overrides_reference_audio_and_text_together(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "configs/wsl_cuda.yaml")

        overridden = override_reference_voice(
            config,
            reference_audio=Path("voices/chinese.wav"),
            reference_text=" 这是中文参考。 ",
        )

        self.assertEqual(
            overridden.tts.reference_audio,
            Path("voices/chinese.wav"),
        )
        self.assertEqual(overridden.tts.reference_text, "这是中文参考。")
        self.assertNotEqual(
            overridden.tts.reference_audio,
            config.tts.reference_audio,
        )

    def test_requires_paired_reference_override(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "configs/wsl_cuda.yaml")

        with self.assertRaisesRegex(ValueError, "together"):
            override_reference_voice(
                config,
                reference_audio=Path("voices/chinese.wav"),
                reference_text=None,
            )

    def test_measures_first_audio_total_audio_and_rtf(self) -> None:
        provider = FakeStreamingTTS()
        clock = SequenceClock([10.0, 10.25, 10.5, 20.0, 20.4, 20.8])

        runs = benchmark_streaming_tts(
            provider,
            text=" 你好 ",
            runs=2,
            clock=clock,
        )

        self.assertEqual(provider.text, "你好")
        self.assertEqual(len(runs), 2)
        self.assertAlmostEqual(runs[0].time_to_first_audio_seconds, 0.25)
        self.assertAlmostEqual(runs[0].total_seconds, 0.5)
        self.assertAlmostEqual(runs[0].audio_seconds, 1.0)
        self.assertAlmostEqual(runs[0].first_chunk_audio_seconds, 0.4)
        self.assertAlmostEqual(runs[0].real_time_factor, 0.5)
        self.assertEqual(runs[0].chunk_count, 2)

        summary = summarize_runs(runs)
        self.assertAlmostEqual(
            summary["median_time_to_first_audio_seconds"],
            0.325,
        )
        self.assertAlmostEqual(summary["median_total_seconds"], 0.65)
        self.assertAlmostEqual(summary["median_real_time_factor"], 0.65)

    def test_rejects_empty_text_or_non_positive_runs(self) -> None:
        provider = FakeStreamingTTS()
        with self.assertRaisesRegex(ValueError, "text"):
            benchmark_streaming_tts(provider, " ", runs=1)
        with self.assertRaisesRegex(ValueError, "runs"):
            benchmark_streaming_tts(provider, "你好", runs=0)

    def test_rejects_provider_that_yields_no_audio(self) -> None:
        class EmptyProvider(FakeStreamingTTS):
            def stream_synthesize(self, text: str):
                return iter(())

        with self.assertRaisesRegex(RuntimeError, "no audio"):
            benchmark_streaming_tts(
                EmptyProvider(),
                "你好",
                runs=1,
                clock=SequenceClock([1.0]),
            )


if __name__ == "__main__":
    unittest.main()
