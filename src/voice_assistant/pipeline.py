from pathlib import Path

from .contracts import (
    ASRProvider,
    LLMProvider,
    Message,
    PipelineResult,
    PreparedResponse,
    TTSProvider,
)
from .observability import (
    PerformanceLogger,
    measure_stage,
    wav_duration_ms,
)

class VoicePipeline:
    def __init__(
        self,
        asr: ASRProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        system_prompt: str,
        reply_instructions: str = "",
        performance: PerformanceLogger | None = None,
    ) -> None:
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._system_prompt = system_prompt
        self._reply_instructions = reply_instructions
        self._performance = performance

    def prepare(self, audio_path: Path) -> PreparedResponse:
        input_audio_duration = wav_duration_ms(audio_path)
        with measure_stage(
            self._performance,
            "asr",
            audio_duration_ms=input_audio_duration,
        ) as asr_span:
            transcript = self._asr.transcribe(audio_path).strip()

            if not transcript:
                raise ValueError("ASR returned an empty transcript.")
            asr_span.add_fields(output_chars=len(transcript))

        user_content = transcript
        if self._reply_instructions:
            user_content = f"{transcript}\n\n{self._reply_instructions}"

        messages = [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=user_content),
        ]
        with measure_stage(
            self._performance,
            "llm",
            input_chars=sum(len(message.content) for message in messages),
        ) as llm_span:
            reply = self._llm.generate(messages).strip()

            if not reply:
                raise ValueError("LLM returned an empty reply.")
            llm_span.add_fields(output_chars=len(reply))

        return PreparedResponse(
            transcript=transcript,
            reply=reply,
        )

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        chunk_index: int = 1,
        chunk_count: int = 1,
    ) -> Path:
        if not text.strip():
            raise ValueError("TTS received empty text.")

        with measure_stage(
            self._performance,
            "tts",
            text_chars=len(text),
            chunk_index=chunk_index,
            chunk_count=chunk_count,
        ) as tts_span:
            synthesized_path = self._tts.synthesize(
                text=text,
                output_path=output_path,
            )
            tts_span.add_fields(
                audio_duration_ms=wav_duration_ms(synthesized_path),
            )

        return synthesized_path

    def run(self, audio_path: Path, output_path: Path) -> PipelineResult:
        prepared = self.prepare(audio_path)
        synthesized_path = self.synthesize(
            text=prepared.reply,
            output_path=output_path,
        )

        return PipelineResult(
            transcript=prepared.transcript,
            reply=prepared.reply,
            audio_path=synthesized_path,
            audio_paths=(synthesized_path,),
        )

    def close(self) -> None:
        close = getattr(self._tts, "close", None)
        if callable(close):
            close()
