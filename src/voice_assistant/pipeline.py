from pathlib import Path

from .contracts import (
    ASRProvider,
    LLMProvider,
    Message,
    PipelineResult,
    TTSProvider,
)

class VoicePipeline:
    def __init__(
        self, 
        asr: ASRProvider, 
        llm: LLMProvider, 
        tts: TTSProvider, 
        system_prompt: str, 
        reply_instructions: str = "",
    ) -> None:
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._system_prompt = system_prompt
        self._reply_instructions = reply_instructions

    def run(self, audio_path: Path, output_path: Path) -> PipelineResult:
        # Step 1: Transcribe the audio
        transcript = self._asr.transcribe(audio_path).strip()

        if not transcript:
            raise ValueError("ASR returned an empty transcript.")

        user_content = transcript
        if self._reply_instructions:
            user_content = f"{transcript}\n\n{self._reply_instructions}"

        # Step 2: Generate a reply using the LLM
        messages = [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=user_content),
        ]
        reply = self._llm.generate(messages).strip()

        if not reply:
            raise ValueError("LLM returned an empty reply.")

        # Step 3: Synthesize the reply into audio
        synthesized_path = self._tts.synthesize(text=reply, output_path=output_path)

        return PipelineResult(
            transcript=transcript,
            reply=reply,
            audio_path=synthesized_path,
        )
