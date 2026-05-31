"""Shape + padding tests for ``kd_collator.KDCollator``."""

import math
import unittest

import torch

from kd_collator import KDCollator, NEG_INF_LOGPROB


def _make_feature(input_ids, assistant_start, gold_ids, topk_ids, topk_logprobs, topk_valid_mask):
    return {
        "input_ids": input_ids,
        "assistant_start": assistant_start,
        "assistant_len": len(gold_ids),
        "gold_ids": gold_ids,
        "topk_ids": topk_ids,
        "topk_logprobs": topk_logprobs,
        "topk_valid_mask": topk_valid_mask,
        "hard_label_in_topk": [True] * len(gold_ids),
        "example_index": 0,
    }


class KDCollatorShapeTests(unittest.TestCase):
    def test_pads_sequences_and_spans(self):
        pad = 99
        collator = KDCollator(pad_token_id=pad)

        # First sample: seq=8, span=3 (start=4, len=3); second sample: seq=6, span=2 (start=3, len=2).
        f1 = _make_feature(
            input_ids=[10, 11, 12, 13, 20, 21, 22, 30],
            assistant_start=4,
            gold_ids=[20, 21, 22],
            topk_ids=[[20, 100], [21, 200], [22, 300]],
            topk_logprobs=[[-0.1, -2.0], [-0.2, -2.1], [-0.3, -2.2]],
            topk_valid_mask=[[True, True]] * 3,
        )
        f2 = _make_feature(
            input_ids=[50, 51, 52, 60, 61, 99],
            assistant_start=3,
            gold_ids=[60, 61],
            topk_ids=[[60, 0], [61, 0]],
            topk_logprobs=[[-0.4, -5.0], [-0.5, -5.1]],
            topk_valid_mask=[[True, False], [True, False]],
        )

        batch = collator([f1, f2])

        self.assertEqual(batch["input_ids"].shape, (2, 8))
        self.assertEqual(batch["attention_mask"].shape, (2, 8))
        self.assertEqual(batch["labels"].shape, (2, 8))
        self.assertEqual(batch["gold_ids"].shape, (2, 3))
        self.assertEqual(batch["topk_ids"].shape, (2, 3, 2))
        self.assertEqual(batch["topk_logprobs"].shape, (2, 3, 2))
        self.assertEqual(batch["topk_valid"].shape, (2, 3, 2))
        self.assertEqual(batch["span_mask"].shape, (2, 3))

        # Right padding: shorter sequence padded at the right with pad_token_id.
        self.assertEqual(batch["input_ids"][1, 5].item(), f2["input_ids"][5])  # real token
        self.assertEqual(batch["input_ids"][1, 6].item(), pad)  # padded
        self.assertEqual(batch["attention_mask"][1].tolist(), [1, 1, 1, 1, 1, 1, 0, 0])

        # Labels: -100 everywhere except the assistant span.
        self.assertEqual(batch["labels"][0, 3].item(), -100)
        self.assertEqual(batch["labels"][0, 4].item(), 20)
        self.assertEqual(batch["labels"][0, 6].item(), 22)
        self.assertEqual(batch["labels"][0, 7].item(), -100)
        self.assertEqual(batch["labels"][1, 2].item(), -100)
        self.assertEqual(batch["labels"][1, 3].item(), 60)
        self.assertEqual(batch["labels"][1, 4].item(), 61)

        # Span mask: True where j < assistant_len[b].
        self.assertEqual(batch["span_mask"][0].tolist(), [True, True, True])
        self.assertEqual(batch["span_mask"][1].tolist(), [True, True, False])

        # Invalid teacher slots get -inf-ish logprobs.
        self.assertEqual(batch["topk_valid"][1, 0, 1].item(), False)
        self.assertTrue(batch["topk_logprobs"][1, 0, 1].item() <= NEG_INF_LOGPROB / 2)

        # Valid slot logprobs are preserved.
        self.assertAlmostEqual(batch["topk_logprobs"][0, 0, 0].item(), -0.1, places=5)

        # assistant_start / assistant_len tensors.
        self.assertEqual(batch["assistant_start"].tolist(), [4, 3])
        self.assertEqual(batch["assistant_len"].tolist(), [3, 2])

    def test_rejects_start_zero(self):
        collator = KDCollator(pad_token_id=0)
        bad = _make_feature(
            input_ids=[5, 6, 7],
            assistant_start=0,
            gold_ids=[5],
            topk_ids=[[5]],
            topk_logprobs=[[-0.1]],
            topk_valid_mask=[[True]],
        )
        with self.assertRaises(ValueError):
            collator([bad])


class KDCollatorSoftmaxSemanticsTests(unittest.TestCase):
    def test_invalid_slots_contribute_zero_mass(self):
        """softmax over the padded top-K row should ignore invalid slots entirely."""
        collator = KDCollator(pad_token_id=0)
        f = _make_feature(
            input_ids=[1, 2, 3, 4, 5],
            assistant_start=3,
            gold_ids=[4, 5],
            topk_ids=[[4, 0, 0], [5, 0, 0]],
            topk_logprobs=[[-0.01, -10.0, -10.0], [-0.02, -10.0, -10.0]],
            topk_valid_mask=[[True, False, False], [True, False, False]],
        )
        batch = collator([f])
        logp = batch["topk_logprobs"][0, 0]  # [K]
        probs = torch.softmax(logp, dim=-1)
        # Only slot 0 valid → its probability must be (essentially) 1.
        self.assertAlmostEqual(probs[0].item(), 1.0, places=5)
        self.assertAlmostEqual(probs[1].item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
