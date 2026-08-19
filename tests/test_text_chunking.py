import unittest

from voice_assistant.text_chunking import (
    split_reply_text,
    stream_reply_text,
)


class SplitReplyTextTest(unittest.TestCase):
    def test_prefers_natural_punctuation_before_hard_limit(self) -> None:
        chunks = split_reply_text(
            "语音合成比较慢，我们先播放第一段，然后继续生成。",
            max_chars=18,
        )

        self.assertEqual(
            chunks,
            ["语音合成比较慢，我们先播放第一段，", "然后继续生成。"],
        )

    def test_hard_splits_text_without_punctuation(self) -> None:
        self.assertEqual(
            split_reply_text("1234567890abcdefghij", max_chars=10),
            ["1234567890", "abcdefghij"],
        )

    def test_keeps_short_reply_as_one_chunk(self) -> None:
        self.assertEqual(
            split_reply_text("马上开始。", max_chars=18),
            ["马上开始。"],
        )

    def test_rejects_invalid_limit(self) -> None:
        with self.assertRaises(ValueError):
            split_reply_text("文本", max_chars=0)

    def test_streams_early_first_chunk_and_flushes_remainder(self) -> None:
        chunks = list(
            stream_reply_text(
                ["我是", "你的", "贴心", "中文", "语音", "助手。"],
                first_chunk_chars=6,
                max_chars=18,
            )
        )

        self.assertEqual(chunks, ["我是你的贴心", "中文语音助手。"])

    def test_streams_complete_short_sentence_at_punctuation(self) -> None:
        chunks = list(
            stream_reply_text(
                ["好的", "。", "马上", "开始"],
                first_chunk_chars=6,
                max_chars=18,
            )
        )

        self.assertEqual(chunks, ["好的。", "马上开始"])

    def test_keeps_punctuation_just_after_limit_with_first_chunk(self) -> None:
        chunks = list(
            stream_reply_text(
                ["我是你的助手", "。", "你好"],
                first_chunk_chars=6,
                max_chars=18,
            )
        )

        self.assertEqual(chunks, ["我是你的助手。", "你好"])

    def test_does_not_wait_for_distant_sentence_punctuation(self) -> None:
        chunks = list(
            stream_reply_text(
                ["一二三四五六七八九十。"],
                first_chunk_chars=4,
                max_chars=6,
            )
        )

        self.assertEqual(chunks, ["一二三四", "五六七八九十。"])

    def test_rejects_invalid_streaming_limits(self) -> None:
        with self.assertRaises(ValueError):
            list(
                stream_reply_text(
                    ["文本"],
                    first_chunk_chars=0,
                    max_chars=18,
                )
            )
        with self.assertRaises(ValueError):
            list(
                stream_reply_text(
                    ["文本"],
                    first_chunk_chars=19,
                    max_chars=18,
                )
            )


if __name__ == "__main__":
    unittest.main()
