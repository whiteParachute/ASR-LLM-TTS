from __future__ import annotations

from typing import Any, Protocol, Sequence

from voice_assistant.contracts import Message


class QwenModel(Protocol):
    device: Any

    def generate(self, **kwargs: Any) -> Any:
        ...


class QwenTokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        ...

    def __call__(self, texts: list[str], *, return_tensors: str) -> Any:
        ...

    def batch_decode(self, token_ids: Any, *, skip_special_tokens: bool) -> list[str]:
        ...


class Qwen25LLM:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 128,
        model: QwenModel | None = None,
        tokenizer: QwenTokenizer | None = None,
    ) -> None:
        if (model is None) != (tokenizer is None):
            raise ValueError("model and tokenizer must be provided together")

        self._max_new_tokens = max_new_tokens

        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self._model = model
            self._tokenizer = tokenizer

    def generate(self, messages: Sequence[Message]) -> str:
        if not messages:
            raise ValueError("Messages cannot be empty")

        raw_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        prompt = self._tokenizer.apply_chat_template(
            raw_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self._tokenizer(
            [prompt],
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(self._model.device)

        # Generate output from the model
        generated_ids = self._model.generate(**model_inputs, max_new_tokens=self._max_new_tokens)
        prompt_ids = model_inputs["input_ids"]

        answer_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(
                prompt_ids,
                generated_ids,
            )
        ]

        decoded = self._tokenizer.batch_decode(
            answer_ids,
            skip_special_tokens=True,
        )

        if not decoded:
            raise RuntimeError("Qwen returned no response")

        reply = decoded[0].strip()

        if not reply:
            raise RuntimeError("Qwen returned an empty response")

        return reply