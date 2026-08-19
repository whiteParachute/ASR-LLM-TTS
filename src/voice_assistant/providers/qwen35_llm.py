from __future__ import annotations

import queue
from collections.abc import Iterator
from threading import Thread
from typing import Any, Callable, Protocol, Sequence

from voice_assistant.contracts import Message


StreamerFactory = Callable[..., Any]


class Qwen35Model(Protocol):
    device: Any

    def generate(self, **kwargs: Any) -> Any:
        ...


class Qwen35Processor(Protocol):
    tokenizer: Any

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        return_tensors: str,
        enable_thinking: bool,
    ) -> Any:
        ...

    def decode(self, token_ids: Any, *, skip_special_tokens: bool) -> str:
        ...


class Qwen35LLM:
    """Qwen3.5 multimodal model used in text-only voice-assistant mode."""

    _SUPPORTED_DTYPES = frozenset({"float16", "bfloat16", "float32"})

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 128,
        *,
        load_in_4bit: bool = True,
        compute_dtype: str = "float16",
        enable_thinking: bool = False,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        model: Qwen35Model | None = None,
        processor: Qwen35Processor | None = None,
        streamer_factory: StreamerFactory | None = None,
    ) -> None:
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be provided together")
        if compute_dtype not in self._SUPPORTED_DTYPES:
            raise ValueError(
                "compute_dtype must be one of: "
                + ", ".join(sorted(self._SUPPORTED_DTYPES))
            )

        self._max_new_tokens = max_new_tokens
        self._enable_thinking = enable_thinking
        self._do_sample = do_sample
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._streamer_factory = streamer_factory

        if model is None:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            dtype = getattr(torch, compute_dtype)
            model_kwargs: dict[str, Any] = {
                "device_map": "auto",
                "dtype": dtype,
                "low_cpu_mem_usage": True,
            }

            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

            self._processor = AutoProcessor.from_pretrained(model_name)
            self._model = AutoModelForMultimodalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
        else:
            self._model = model
            self._processor = processor

    def generate(self, messages: Sequence[Message]) -> str:
        model_inputs, generation_kwargs = self._prepare_generation(messages)

        generated_ids = self._model.generate(
            **model_inputs,
            **generation_kwargs,
        )
        input_length = len(model_inputs["input_ids"][0])
        answer_ids = generated_ids[0][input_length:]
        reply = self._processor.decode(
            answer_ids,
            skip_special_tokens=True,
        ).strip()

        if not reply:
            raise RuntimeError("Qwen3.5 returned an empty response")

        return reply

    def stream_generate(self, messages: Sequence[Message]) -> Iterator[str]:
        streamer_factory = self._streamer_factory
        if streamer_factory is None:
            from transformers import TextIteratorStreamer

            streamer_factory = TextIteratorStreamer

        model_inputs, generation_kwargs = self._prepare_generation(messages)
        streamer = streamer_factory(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,
        )
        errors: queue.SimpleQueue[BaseException] = queue.SimpleQueue()

        def run_generation() -> None:
            try:
                self._model.generate(
                    **model_inputs,
                    **generation_kwargs,
                    streamer=streamer,
                )
            except BaseException as exc:
                errors.put(exc)
                streamer.on_finalized_text("", stream_end=True)

        generation_thread = Thread(
            target=run_generation,
            name="qwen35-stream-generate",
            daemon=True,
        )
        generation_thread.start()

        combined_reply = ""
        for text in streamer:
            if text:
                combined_reply += text
                yield text
        generation_thread.join()

        if not errors.empty():
            error = errors.get()
            raise RuntimeError(
                "Qwen3.5 streaming generation failed"
            ) from error
        if not combined_reply.strip():
            raise RuntimeError("Qwen3.5 returned an empty response")

    def _prepare_generation(
        self,
        messages: Sequence[Message],
    ) -> tuple[Any, dict[str, Any]]:
        if not messages:
            raise ValueError("Messages cannot be empty")

        raw_messages = [
            {
                "role": message.role,
                "content": [
                    {
                        "type": "text",
                        "text": message.content,
                    }
                ],
            }
            for message in messages
        ]

        model_inputs = self._processor.apply_chat_template(
            raw_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self._enable_thinking,
        )
        model_inputs = model_inputs.to(self._model.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._do_sample,
        }
        if self._do_sample:
            generation_kwargs.update(
                temperature=self._temperature,
                top_p=self._top_p,
                top_k=self._top_k,
            )
        return model_inputs, generation_kwargs
