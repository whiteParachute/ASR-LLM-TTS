import unittest
from typing import Any

from voice_assistant.contracts import Message
from voice_assistant.providers.qwen35_llm import Qwen35LLM


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


class FakeProcessor:
    def __init__(self, decoded: str = " 模型回答 ") -> None:
        self.batch = FakeBatch()
        self.decoded = decoded
        self.template_arguments: dict[str, Any] = {}
        self.received_decode_ids: Any = None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> FakeBatch:
        self.template_arguments = {
            "messages": messages,
            **kwargs,
        }
        return self.batch

    def decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool,
    ) -> str:
        self.received_decode_ids = token_ids
        return self.decoded


class Qwen35LLMTest(unittest.TestCase):
    def test_generates_non_thinking_4bit_reply(self) -> None:
        model = FakeModel()
        processor = FakeProcessor()
        provider = Qwen35LLM(
            model_name="Qwen/Qwen3.5-4B",
            max_new_tokens=128,
            load_in_4bit=True,
            compute_dtype="float16",
            enable_thinking=False,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            model=model,
            processor=processor,
        )

        reply = provider.generate(
            [
                Message(role="system", content="你是语音助手。"),
                Message(role="user", content="你好。"),
            ]
        )

        self.assertEqual(processor.batch.moved_to, "cpu")
        self.assertFalse(processor.template_arguments["enable_thinking"])
        self.assertEqual(
            processor.template_arguments["messages"][1]["content"],
            [{"type": "text", "text": "你好。"}],
        )
        self.assertEqual(model.received_arguments["max_new_tokens"], 128)
        self.assertTrue(model.received_arguments["do_sample"])
        self.assertEqual(model.received_arguments["temperature"], 0.7)
        self.assertEqual(model.received_arguments["top_p"], 0.8)
        self.assertEqual(model.received_arguments["top_k"], 20)
        self.assertEqual(processor.received_decode_ids, [9, 10])
        self.assertEqual(reply, "模型回答")

    def test_rejects_empty_messages(self) -> None:
        provider = Qwen35LLM(
            model_name="test-model",
            model=FakeModel(),
            processor=FakeProcessor(),
        )

        with self.assertRaisesRegex(ValueError, "Messages cannot be empty"):
            provider.generate([])

    def test_rejects_empty_response(self) -> None:
        provider = Qwen35LLM(
            model_name="test-model",
            model=FakeModel(),
            processor=FakeProcessor(decoded="  "),
        )

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            provider.generate([Message(role="user", content="你好")])

    def test_rejects_unknown_compute_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "compute_dtype"):
            Qwen35LLM(
                model_name="test-model",
                compute_dtype="int4",
                model=FakeModel(),
                processor=FakeProcessor(),
            )


if __name__ == "__main__":
    unittest.main()
