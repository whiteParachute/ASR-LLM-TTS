from itertools import chain
from pathlib import Path
from typing import Iterator

from .contracts import (
    ASRProvider,
    AudioChunk,
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
from .tooling import BoundedToolLoop

class VoicePipeline:
    def __init__(
        self,
        asr: ASRProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        system_prompt: str,
        reply_instructions: str = "",
        performance: PerformanceLogger | None = None,
        tool_loop: BoundedToolLoop | None = None,
    ) -> None:
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._system_prompt = system_prompt
        self._reply_instructions = reply_instructions
        self._performance = performance
        self._tool_loop = tool_loop

    def prepare(self, audio_path: Path) -> PreparedResponse:
        transcript = self.transcribe(audio_path)
        reply = self.generate_reply(transcript)

        return PreparedResponse(
            transcript=transcript,
            reply=reply,
            sources=(
                self._tool_loop.source_references
                if self._tool_loop is not None
                else ()
            ),
        )

    def transcribe(self, audio_path: Path) -> str:
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

        return transcript

    def generate_reply(self, transcript: str) -> str:
        messages = self._build_messages(transcript)
        with measure_stage(
            self._performance,
            "llm",
            input_chars=sum(len(message.content) for message in messages),
        ) as llm_span:
            if self._tool_loop is None:
                reply = self._llm.generate(messages).strip()
            else:
                reply = self._tool_loop.generate(messages).strip()

            if not reply:
                raise ValueError("LLM returned an empty reply.")
            llm_span.add_fields(output_chars=len(reply))

        return reply

    @property
    def supports_streaming_llm(self) -> bool:
        return self._tool_loop is None and callable(
            getattr(self._llm, "stream_generate", None)
        )

    def stream_reply(self, transcript: str) -> Iterator[str]:
        if self._tool_loop is not None:
            raise RuntimeError(
                "Streaming LLM replies are disabled while tools are enabled"
            )
        stream_generate = getattr(self._llm, "stream_generate", None)
        if not callable(stream_generate):
            raise RuntimeError("Configured LLM provider does not stream text")

        messages = self._build_messages(transcript)
        with measure_stage(
            self._performance,
            "llm_stream",
            input_chars=sum(len(message.content) for message in messages),
        ) as llm_span:
            iterator = iter(stream_generate(messages))
            try:
                with measure_stage(self._performance, "llm_first_text"):
                    first_part = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("LLM returned an empty reply.") from exc

            output_chars = 0
            for part in chain((first_part,), iterator):
                output_chars += len(part)
                yield part
            llm_span.add_fields(output_chars=output_chars)

    def _build_messages(self, transcript: str) -> list[Message]:
        cleaned_transcript = transcript.strip()
        if not cleaned_transcript:
            raise ValueError("Transcript cannot be empty.")

        user_content = cleaned_transcript
        if self._reply_instructions:
            user_content = f"{transcript}\n\n{self._reply_instructions}"

        return [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=user_content),
        ]

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

    @property
    def supports_streaming_tts(self) -> bool:
        return callable(getattr(self._tts, "stream_synthesize", None))

    def stream_synthesize(self, text: str) -> Iterator[AudioChunk]:
        stream_synthesize = getattr(self._tts, "stream_synthesize", None)
        if not callable(stream_synthesize):
            raise RuntimeError("Configured TTS provider does not stream audio")
        if not text.strip():
            raise ValueError("TTS received empty text.")

        with measure_stage(
            self._performance,
            "tts_stream",
            text_chars=len(text),
        ) as tts_span:
            chunk_count = 0
            audio_duration_ms = 0.0
            for chunk in stream_synthesize(text):
                chunk_count += 1
                audio_duration_ms += chunk.duration_ms
                yield chunk
            if chunk_count == 0:
                raise RuntimeError("Streaming TTS returned no audio chunks")
            tts_span.add_fields(
                chunk_count=chunk_count,
                audio_duration_ms=round(audio_duration_ms, 3),
            )

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
            sources=prepared.sources,
        )

    def close(self) -> None:
        close = getattr(self._tts, "close", None)
        if callable(close):
            close()
