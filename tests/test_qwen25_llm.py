import unittest
from typing import Any

from voice_assistant.contracts import Message
from voice_assistant.providers.qwen25_llm import Qwen25LLM


class FakeBatch(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__(input_ids=[[1, 2, 3]])
        self.moved_to: Any = None

    def to(self, device: Any) -> "FakeBatch":
        self.moved_to = device
        return self


class FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.received_arguments: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> Any:
        self.received_arguments = kwargs
        return [[1, 2, 3, 9, 10]]


class FakeTokenizer:
    def __init__(
        self,
        decoded: list[str] | None = None,
    ) -> None:
        self.batch = FakeBatch()
        self.decoded = decoded or [" 模型回答 "]
        self.received_messages: Any = None
        self.received_decode_ids: Any = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.received_messages = messages
        return "formatted prompt"

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
    ) -> FakeBatch:
        return self.batch

    def batch_decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool,
    ) -> list[str]:
        self.received_decode_ids = token_ids
        return self.decoded


class Qwen25LLMTest(unittest.TestCase):
    def test_generates_reply(self) -> None:
        model = FakeModel()
        tokenizer = FakeTokenizer()

        provider = Qwen25LLM(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            max_new_tokens=128,
            model=model,
            tokenizer=tokenizer,
        )

        reply = provider.generate(
            [
                Message(
                    role="system",
                    content="你是语音助手。",
                ),
                Message(
                    role="user",
                    content="你好。",
                ),
            ],
        )
        self.assertEqual(
            tokenizer.batch.moved_to,
            "cpu"
        )
        self.assertEqual(model.received_arguments["max_new_tokens"], 128)
        self.assertEqual(tokenizer.received_decode_ids, [[9, 10]])
        self.assertEqual(reply, "模型回答")

    def test_rejects_empty_response(self) -> None:
        provider = Qwen25LLM(
            model_name="test-model",
            model=FakeModel(),
            tokenizer=FakeTokenizer(decoded=["  "]),
        )

        with self.assertRaises(RuntimeError):
            provider.generate(
                [
                    Message(role="user", content="你好")
                ]
            )


if __name__ == '__main__':
    unittest.main()