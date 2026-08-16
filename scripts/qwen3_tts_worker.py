#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        description="Persistent JSON-lines worker for Qwen3-TTS Base.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", default="")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--x-vector-only-mode", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            import soundfile as sf
            import torch
            from qwen_tts import Qwen3TTSModel

            dtype = getattr(torch, args.dtype, None)
            if dtype is None or not isinstance(dtype, torch.dtype):
                raise ValueError(f"Unsupported torch dtype: {args.dtype}")

            load_options: dict[str, Any] = {
                "device_map": args.device,
                "dtype": dtype,
            }
            if args.attention_implementation:
                load_options["attn_implementation"] = (
                    args.attention_implementation
                )

            model = Qwen3TTSModel.from_pretrained(
                args.model,
                **load_options,
            )
            voice_prompt = model.create_voice_clone_prompt(
                ref_audio=str(args.reference_audio),
                ref_text=args.reference_text or None,
                x_vector_only_mode=args.x_vector_only_mode,
            )
    except BaseException as exc:
        send(
            {
                "event": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    send({"event": "ready"})

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
            if operation != "synthesize":
                raise ValueError(f"Unsupported operation: {operation}")

            text = str(request.get("text", "")).strip()
            if not text:
                raise ValueError("TTS text cannot be empty")
            output_path = Path(str(request["output_path"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with contextlib.redirect_stdout(sys.stderr):
                wavs, sample_rate = model.generate_voice_clone(
                    text=text,
                    language=args.language,
                    voice_clone_prompt=voice_prompt,
                    non_streaming_mode=True,
                    max_new_tokens=args.max_new_tokens,
                )
                if not wavs:
                    raise RuntimeError("Qwen3-TTS returned no audio")
                sf.write(str(output_path), wavs[0], sample_rate)

            send(
                {
                    "id": request_id,
                    "ok": True,
                    "output_path": str(output_path),
                    "sample_rate": sample_rate,
                }
            )
        except BaseException as exc:
            send(
                {
                    "id": request_id,
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
