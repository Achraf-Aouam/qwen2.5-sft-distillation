"""Smoke tests for the parquet → KD Dataset conversion."""

import math
import unittest

from KL_div_prep.build_kd_dataset import (
    _replace_none_ids,
    _replace_none_logprobs,
    row_to_kd_record,
)


class FakeStudentTokenizer:
    """Char-level tokenizer so `full_chat_text` round-trips via `ord`/`chr`."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _make_row(prompt: str, answer: str):
    full_text = prompt + answer
    full_ids = [ord(c) for c in full_text]
    start = len(prompt)
    gold_ids = full_ids[start:]
    n = len(gold_ids)
    # Teacher puts most mass on gold; one padded slot with None id/logprob.
    topk_ids = [[gold_ids[j], gold_ids[j] + 1, None] for j in range(n)]
    topk_lp = [[-0.01, -3.0, None] for _ in range(n)]
    return {
        "example_index": 0,
        "full_chat_text": full_text,
        "assistant_token_start": start,
        "assistant_token_count": n,
        "assistant_token_ids": gold_ids,
        "topk_token_ids": topk_ids,
        "topk_logprobs": topk_lp,
        "hard_label_in_topk": [True] * n,
    }


class HelperTests(unittest.TestCase):
    def test_replace_none_ids_marks_validity(self):
        cleaned, valid = _replace_none_ids([1, None, 3])
        self.assertEqual(cleaned, [1, 0, 3])
        self.assertEqual(valid, [True, False, True])

    def test_replace_none_logprobs_uses_large_negative(self):
        out = _replace_none_logprobs([-1.5, None, float("nan"), -0.1])
        self.assertAlmostEqual(out[0], -1.5)
        self.assertTrue(out[1] < -1e20 and math.isfinite(out[1]))
        self.assertTrue(out[2] < -1e20 and math.isfinite(out[2]))
        self.assertAlmostEqual(out[3], -0.1)


class RowToRecordTests(unittest.TestCase):
    def test_aligns_and_masks_padded_topk_slots(self):
        row = _make_row("SYS|USR|", "HELLO")
        record = row_to_kd_record(FakeStudentTokenizer(), row)

        # Tokenization of full_chat_text matches char codes.
        self.assertEqual(record["input_ids"], [ord(c) for c in "SYS|USR|HELLO"])
        self.assertEqual(record["assistant_start"], 8)
        self.assertEqual(record["assistant_len"], 5)
        self.assertEqual(record["gold_ids"], [ord(c) for c in "HELLO"])

        # topk_ids: 3rd slot was None → replaced with 0 and marked invalid.
        first_row = record["topk_ids"][0]
        self.assertEqual(first_row[0], ord("H"))
        self.assertEqual(first_row[2], 0)
        self.assertEqual(record["topk_valid_mask"][0], [True, True, False])
        # And its logprob is a large negative (not None).
        self.assertTrue(record["topk_logprobs"][0][2] < -1e20)

    def test_rejects_start_zero(self):
        row = _make_row("", "XY")
        with self.assertRaises(RuntimeError):
            row_to_kd_record(FakeStudentTokenizer(), row)

    def test_rejects_assistant_span_mismatch(self):
        row = _make_row("PROMPT ", "answer")
        # Break alignment by tampering with stored gold ids.
        row["assistant_token_ids"] = [0] * len(row["assistant_token_ids"])
        with self.assertRaises(RuntimeError):
            row_to_kd_record(FakeStudentTokenizer(), row)


if __name__ == "__main__":
    unittest.main()
