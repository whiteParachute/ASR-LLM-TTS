#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent streaming worker for CosyVoice3.",
    )
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--warmup-text", default="你好，很高兴和你对话。")
    parser.add_argument("--fp16", action="store_true")
    return parser


def build_prompt_text(reference_text: str) -> str:
    cleaned_text = reference_text.strip()
    if "<|endofprompt|>" in cleaned_text:
        return cleaned_text
    return f"You are a helpful assistant.<|endofprompt|>{cleaned_text}"


def main() -> int:
    args = build_parser().parse_args()
    runtime_dir = args.runtime_dir.expanduser().resolve()
    matcha_dir = runtime_dir / "third_party/Matcha-TTS"
    sys.path.insert(0, str(matcha_dir))
    sys.path.insert(0, str(runtime_dir))

    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            from cosyvoice.cli.cosyvoice import AutoModel

            model = AutoModel(
                model_dir=args.model,
                load_trt=False,
                load_vllm=False,
                fp16=args.fp16,
            )
            voice_id = "assistant_reference"
            prompt_text = build_prompt_text(args.reference_text)
            added = model.add_zero_shot_spk(
                prompt_text,
                str(args.reference_audio.expanduser().resolve()),
                voice_id,
            )
            if added is not True:
                raise RuntimeError("Unable to cache the reference voice")

            if args.warmup_text.strip():
                for _ in model.inference_zero_shot(
                    args.warmup_text.strip(),
                    "",
                    "",
                    zero_shot_spk_id=voice_id,
                    stream=True,
                ):
                    pass
    except BaseException as exc:
        send(
            {
                "event": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    send(
        {
            "event": "ready",
            "sample_rate": int(model.sample_rate),
        }
    )

    try:
        for line in sys.stdin:
            request_id: Any = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Request must be a JSON object")
                request_id = request.get("id")
                operation = request.get("op")
                if operation == "shutdown":
                    return 0
                if operation != "stream_synthesize":
                    raise ValueError(f"Unsupported operation: {operation}")

                text = str(request.get("text", "")).strip()
                if not text:
                    raise ValueError("TTS text cannot be empty")

                chunk_count = 0
                sample_count = 0
                with contextlib.redirect_stdout(sys.stderr):
                    outputs = model.inference_zero_shot(
                        text,
                        "",
                        "",
                        zero_shot_spk_id=voice_id,
                        stream=True,
                    )
                    for output in outputs:
                        speech = output["tts_speech"]
                        speech = speech.detach().float().cpu().flatten()
                        speech = speech.clamp(-1.0, 1.0)
                        pcm = (
                            (speech * 32767.0)
                            .round()
                            .to(torch.int16)
                            .numpy()
                            .tobytes()
                        )
                        if not pcm:
                            continue
                        chunk_count += 1
                        sample_count += len(pcm) // 2
                        send(
                            {
                                "id": request_id,
                                "event": "audio_chunk",
                                "chunk_index": chunk_count,
                                "sample_rate": int(model.sample_rate),
                                "channels": 1,
                                "pcm_s16le_base64": base64.b64encode(
                                    pcm
                                ).decode("ascii"),
                            }
                        )

                if chunk_count == 0:
                    raise RuntimeError("CosyVoice3 returned no audio")
                send(
                    {
                        "id": request_id,
                        "event": "complete",
                        "chunk_count": chunk_count,
                        "sample_count": sample_count,
                    }
                )
            except Exception as exc:
                send(
                    {
                        "id": request_id,
                        "event": "error",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    except KeyboardInterrupt:
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
