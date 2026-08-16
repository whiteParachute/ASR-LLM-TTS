from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

class SenseVoiceModel(Protocol):
    def generate(self, **kwargs: Any) -> Any:
        ...


class SenseVoiceASR:
    def __init__(
        self,
        model_name: str,
        language: str = "auto",
        use_itn: bool = False,
        device: str | None = None,
        model: SenseVoiceModel | None = None,
    ) -> None:
        self._language = language
        self._use_itn = use_itn

        if model is None:
            from funasr import AutoModel

            model_path = Path(model_name).expanduser()
            model_options: dict[str, Any] = {
                "model": model_name,
                "trust_remote_code": not model_path.is_dir(),
                "disable_update": True,
            }
            if device is not None:
                model_options["device"] = device

            self._model = AutoModel(**model_options)
        else:
            self._model = model
    
    def transcribe(self, audio_path: Path) -> str:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        
        result = self._model.generate(
            input=str(audio_path),
            cache={},
            language=self._language,
            use_itn=self._use_itn,
        )

        return self._extract_transcript(result)

    @staticmethod
    def _extract_transcript(result: Any) -> str:
        if not isinstance(result, list) or not result:
            raise RuntimeError("SenseVoice returned no result")

        first_result = result[0]

        if not isinstance(first_result, dict):
            raise RuntimeError("SenseVoice returned an invalid result")

        raw_text = first_result.get("text")

        if not isinstance(raw_text, str):
            raise RuntimeError("SenseVoice result does not contain text")

        transcript = raw_text.rsplit(">", 1)[-1].strip()

        if not transcript:
            raise RuntimeError("SenseVoice returned an empty transcript")

        return transcript
