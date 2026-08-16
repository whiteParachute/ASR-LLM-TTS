from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Qwen3ASRModel(Protocol):
    device: Any
    dtype: Any

    def generate(self, **kwargs: Any) -> Any:
        ...


class Qwen3ASRProcessor(Protocol):
    def apply_transcription_request(self, **kwargs: Any) -> Any:
        ...

    def decode(
        self,
        token_ids: Any,
        *,
        return_format: str,
    ) -> Any:
        ...


class Qwen3ASR:
    """Transcribe audio with native Transformers Qwen3-ASR."""

    _SUPPORTED_DTYPES = frozenset({"float16", "bfloat16", "float32"})

    def __init__(
        self,
        model_name: str,
        language: str = "auto",
        device: str | None = None,
        compute_dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        max_new_tokens: int = 256,
        prompt: str = "",
        model: Qwen3ASRModel | None = None,
        processor: Qwen3ASRProcessor | None = None,
    ) -> None:
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be provided together")
        if compute_dtype not in self._SUPPORTED_DTYPES:
            raise ValueError(
                "compute_dtype must be one of: "
                + ", ".join(sorted(self._SUPPORTED_DTYPES))
            )
        if max_new_tokens <= 0:
            raise ValueError("Qwen3-ASR max_new_tokens must be positive")

        self._language = self._normalise_language(language)
        self._max_new_tokens = max_new_tokens
        self._prompt = prompt.strip()

        if model is None:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            dtype = getattr(torch, compute_dtype)
            model_options: dict[str, Any] = {
                "device_map": device or "auto",
                "dtype": dtype,
                "low_cpu_mem_usage": True,
            }
            if attention_implementation:
                model_options["attn_implementation"] = (
                    attention_implementation
                )

            self._processor = AutoProcessor.from_pretrained(model_name)
            self._model = AutoModelForMultimodalLM.from_pretrained(
                model_name,
                **model_options,
            )
        else:
            self._model = model
            self._processor = processor

    def transcribe(self, audio_path: Path) -> str:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        request_options: dict[str, Any] = {
            "audio": str(audio_path),
            "language": self._language,
        }
        if self._prompt:
            request_options["prompt"] = self._prompt

        model_inputs = self._processor.apply_transcription_request(
            **request_options,
        )
        model_inputs = model_inputs.to(
            self._model.device,
            self._model.dtype,
        )
        output_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self._max_new_tokens,
        )
        input_length = model_inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]
        decoded = self._processor.decode(
            generated_ids,
            return_format="transcription_only",
        )

        transcript = decoded if isinstance(decoded, str) else None
        if isinstance(decoded, (list, tuple)) and decoded:
            transcript = decoded[0]
        if not isinstance(transcript, str):
            raise RuntimeError("Qwen3-ASR returned an invalid result")

        transcript = transcript.strip()
        if not transcript:
            raise RuntimeError("Qwen3-ASR returned an empty transcript")
        return transcript

    @staticmethod
    def _normalise_language(language: str | None) -> str | None:
        if language is None:
            return None
        cleaned = language.strip()
        if cleaned.lower() in {"", "auto", "none"}:
            return None
        return cleaned
