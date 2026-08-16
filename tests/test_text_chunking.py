import unittest

from voice_assistant.text_chunking import split_reply_text


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


if __name__ == "__main__":
    unittest.main()
