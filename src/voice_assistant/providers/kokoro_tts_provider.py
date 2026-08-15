from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol


class KokoroPipeline(Protocol):
    def __call__(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        split_pattern: str,
    ) -> Iterable[tuple[Any, Any, Any]]:
        ...


AudioWriter = Callable[[str, Any, int], None]
AudioConcatenator = Callable[[list[Any]], Any]


class KokoroTTSProvider:
    """Synthesize speech locally with Kokoro-82M."""

    def __init__(
        self,
        model_name: str,
        language_code: str,
        default_voice: str,
        speed: float = 1.0,
        sample_rate: int = 24000,
        device: str | None = None,
        pipeline: KokoroPipeline | None = None,
        audio_writer: AudioWriter | None = None,
        audio_concatenator: AudioConcatenator | None = None,
    ) -> None:
        if speed <= 0:
            raise ValueError("Kokoro speed must be greater than zero")
        if sample_rate <= 0:
            raise ValueError("Kokoro sample rate must be greater than zero")

        model_path = Path(model_name).expanduser()
        self._default_voice = self._resolve_voice(
            model_path,
            default_voice,
        )
        self._speed = speed
        self._sample_rate = sample_rate

        if pipeline is None:
            self._pipeline = self._build_pipeline(
                model_name=model_name,
                model_path=model_path,
                language_code=language_code,
                device=device,
            )
        else:
            self._pipeline = pipeline

        if audio_writer is None:
            import soundfile as sf

            self._audio_writer = sf.write
        else:
            self._audio_writer = audio_writer

        self._audio_concatenator = (
            audio_concatenator or self._concatenate_audio
        )

    @staticmethod
    def _resolve_voice(model_path: Path, default_voice: str) -> str:
        if not model_path.is_dir():
            return default_voice

        voice_path = Path(default_voice).expanduser()
        if voice_path.suffix != ".pt":
            voice_path = model_path / "voices" / f"{default_voice}.pt"
        elif not voice_path.is_absolute():
            voice_path = model_path / voice_path

        if not voice_path.is_file():
            raise FileNotFoundError(
                f"Kokoro voice file does not exist: {voice_path}"
            )
        return str(voice_path)

    @staticmethod
    def _build_pipeline(
        model_name: str,
        model_path: Path,
        language_code: str,
        device: str | None,
    ) -> KokoroPipeline:
        from kokoro import KModel, KPipeline

        if not model_path.is_dir():
            return KPipeline(
                lang_code=language_code,
                repo_id=model_name,
                device=device,
            )

        config_path = model_path / "config.json"
        model_files = sorted(model_path.glob("kokoro-*.pth"))
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Kokoro config file does not exist: {config_path}"
            )
        if not model_files:
            raise FileNotFoundError(
                f"Kokoro model file does not exist under: {model_path}"
            )

        import torch

        target_device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        local_model = KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(config_path),
            model=str(model_files[-1]),
        ).to(target_device).eval()

        return KPipeline(
            lang_code=language_code,
            repo_id="hexgrad/Kokoro-82M",
            model=local_model,
            device=target_device,
        )

    def synthesize(self, text: str, output_path: Path) -> Path:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("TTS text cannot be empty")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        generated = self._pipeline(
            cleaned_text,
            voice=self._default_voice,
            speed=self._speed,
            split_pattern=r"\n+",
        )

        audio_chunks: list[Any] = []
        for _, _, audio in generated:
            if audio is None:
                continue
            audio_chunks.append(audio)

        if not audio_chunks:
            raise RuntimeError("Kokoro returned no audio")

        audio = self._audio_concatenator(audio_chunks)
        self._audio_writer(str(output_path), audio, self._sample_rate)

        if not output_path.is_file():
            raise RuntimeError(
                f"Kokoro did not create output file: {output_path}"
            )

        return output_path

    @staticmethod
    def _concatenate_audio(audio_chunks: list[Any]) -> Any:
        import numpy as np

        converted: list[Any] = []
        for audio in audio_chunks:
            value = audio
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()

            chunk = np.asarray(value, dtype=np.float32).reshape(-1)
            if chunk.size:
                converted.append(chunk)

        if not converted:
            raise RuntimeError("Kokoro returned empty audio chunks")

        return converted[0] if len(converted) == 1 else np.concatenate(converted)
