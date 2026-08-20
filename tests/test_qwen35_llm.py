import unittest
from typing import Any

from voice_assistant.contracts import Message, ToolCall, ToolDefinition
from voice_assistant.providers.qwen35_llm import (
    Qwen35LLM,
    parse_qwen_tool_response,
)


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


class FailingModel(FakeModel):
    def generate(self, **kwargs: Any) -> Any:
        self.received_arguments = kwargs
        raise ValueError("generation failed")


class FakeProcessor:
    def __init__(self, decoded: str = " 模型回答 ") -> None:
        self.tokenizer = FakeTokenizer()
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


class FakeTokenizer:
    eos_token_id = 1001
    pad_token_id = 1002


class FakeTextIteratorStreamer:
    parts = ["模型", "回答"]

    def __init__(self, tokenizer: Any, **kwargs: Any) -> None:
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.finalized = False

    def __iter__(self):
        return iter(self.parts)

    def on_finalized_text(
        self,
        text: str,
        *,
        stream_end: bool,
    ) -> None:
        self.finalized = stream_end


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
        self.assertEqual(model.received_arguments["eos_token_id"], 1001)
        self.assertEqual(model.received_arguments["pad_token_id"], 1002)
        self.assertEqual(processor.received_decode_ids, [9, 10])
        self.assertEqual(reply, "模型回答")

    def test_generates_native_tool_call(self) -> None:
        processor = FakeProcessor(
            decoded=(
                "<tool_call>\n"
                "<function=calculate>\n"
                "<parameter=expression>\n2 + 3\n</parameter>\n"
                "</function>\n"
                "</tool_call>"
            )
        )
        provider = Qwen35LLM(
            model_name="test-model",
            model=FakeModel(),
            processor=processor,
        )
        tool = ToolDefinition(
            name="calculate",
            description="Calculate arithmetic.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )

        response = provider.generate_with_tools(
            [Message(role="user", content="2+3是多少？")],
            [tool],
        )

        self.assertEqual(
            response.tool_calls,
            (ToolCall(name="calculate", arguments={"expression": "2 + 3"}),),
        )
        self.assertEqual(
            processor.template_arguments["tools"],
            [tool.as_chat_template_dict()],
        )
        self.assertFalse(provider._model.received_arguments["do_sample"])

    def test_renders_tool_call_and_result_in_followup_messages(self) -> None:
        processor = FakeProcessor(decoded="结果是5。")
        provider = Qwen35LLM(
            model_name="test-model",
            model=FakeModel(),
            processor=processor,
        )
        call = ToolCall(name="calculate", arguments={"expression": "2+3"})
        tool = ToolDefinition(
            name="calculate",
            description="Calculate arithmetic.",
            parameters={"type": "object", "properties": {}},
        )

        response = provider.generate_with_tools(
            [
                Message(role="user", content="2+3是多少？"),
                Message(role="assistant", content="", tool_calls=(call,)),
                Message(role="tool", content='{"ok":true,"result":5}'),
            ],
            [tool],
        )

        rendered_messages = processor.template_arguments["messages"]
        self.assertEqual(response.content, "结果是5。")
        self.assertEqual(
            rendered_messages[1]["tool_calls"][0]["function"],
            {"name": "calculate", "arguments": {"expression": "2+3"}},
        )
        self.assertEqual(rendered_messages[2]["role"], "tool")

    def test_rejects_malformed_native_tool_call(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            parse_qwen_tool_response(
                "<tool_call><function=calculate></tool_call>"
            )

    def test_rejects_empty_messages(self) -> None:
        provider = Qwen35LLM(
            model_name="test-model",
            model=FakeModel(),
            processor=FakeProcessor(),
        )

        with self.assertRaisesRegex(ValueError, "Messages cannot be empty"):
            provider.generate([])

    def test_streams_generated_text(self) -> None:
        model = FakeModel()
        processor = FakeProcessor()
        provider = Qwen35LLM(
            model_name="test-model",
            do_sample=False,
            model=model,
            processor=processor,
            streamer_factory=FakeTextIteratorStreamer,
        )

        parts = list(
            provider.stream_generate(
                [Message(role="user", content="你好")]
            )
        )

        self.assertEqual(parts, ["模型", "回答"])
        self.assertIs(
            model.received_arguments["streamer"].tokenizer,
            processor.tokenizer,
        )
        self.assertFalse(model.received_arguments["do_sample"])

    def test_rejects_empty_streamed_response(self) -> None:
        provider = Qwen35LLM(
            model_name="test-model",
            model=FakeModel(),
            processor=FakeProcessor(),
            streamer_factory=FakeTextIteratorStreamer,
        )

        original_parts = FakeTextIteratorStreamer.parts
        FakeTextIteratorStreamer.parts = []
        try:
            with self.assertRaisesRegex(RuntimeError, "empty response"):
                list(
                    provider.stream_generate(
                        [Message(role="user", content="你好")]
                    )
                )
        finally:
            FakeTextIteratorStreamer.parts = original_parts

    def test_surfaces_streaming_generation_error(self) -> None:
        provider = Qwen35LLM(
            model_name="test-model",
            model=FailingModel(),
            processor=FakeProcessor(),
            streamer_factory=FakeTextIteratorStreamer,
        )

        original_parts = FakeTextIteratorStreamer.parts
        FakeTextIteratorStreamer.parts = []
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "streaming generation failed",
            ) as caught:
                list(
                    provider.stream_generate(
                        [Message(role="user", content="你好")]
                    )
                )
        finally:
            FakeTextIteratorStreamer.parts = original_parts

        self.assertIsInstance(caught.exception.__cause__, ValueError)

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
