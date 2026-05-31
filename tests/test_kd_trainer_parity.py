"""Parity test: ``KDTrainer.compute_loss`` with alpha=0 equals plain next-token CE
on the assistant span. This pins down the slice/index math independently of
SFTTrainer internals by exercising ``compute_loss`` directly on a random toy model.
"""

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

from kd_collator import KDCollator
from kd_trainer import KDTrainer


class TinyLM(torch.nn.Module):
    """A linear over token embeddings — enough to produce [B, T, V] logits with a grad graph."""

    def __init__(self, vocab_size: int, hidden: int = 16):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, **_):
        hidden = self.embed(input_ids)
        logits = self.head(hidden)
        # Mimic the HF forward output namespace used by KDTrainer.
        return MagicMock(logits=logits)


def make_kd_compute_loss(alpha: float, temperature: float = 1.0):
    """Return a callable bound to a KDTrainer-like object without running __init__."""

    obj = KDTrainer.__new__(KDTrainer)
    obj.kd_alpha = alpha
    obj.kd_temperature = temperature
    obj._last_kd_metrics = {}
    return obj


class KDAlphaZeroParityTests(unittest.TestCase):
    def test_alpha_zero_matches_plain_ce_on_span(self):
        torch.manual_seed(42)
        vocab = 64
        k = 3

        # Build two samples with different span lengths / starts.
        pad = 0
        collator = KDCollator(pad_token_id=pad)

        f1 = {
            "input_ids": [5, 6, 7, 8, 9, 10, 11, 12],
            "assistant_start": 3,
            "assistant_len": 4,
            "gold_ids": [8, 9, 10, 11],
            "topk_ids": [[8, 1, 2]] * 4,
            "topk_logprobs": [[-0.1, -2.0, -3.0]] * 4,
            "topk_valid_mask": [[True, True, True]] * 4,
            "hard_label_in_topk": [True] * 4,
            "example_index": 0,
        }
        f2 = {
            "input_ids": [15, 16, 17, 18, 19],
            "assistant_start": 2,
            "assistant_len": 3,
            "gold_ids": [17, 18, 19],
            "topk_ids": [[17, 1, 2], [18, 1, 2], [19, 1, 2]],
            "topk_logprobs": [[-0.1, -2.0, -3.0]] * 3,
            "topk_valid_mask": [[True, True, True]] * 3,
            "hard_label_in_topk": [True] * 3,
            "example_index": 1,
        }

        batch = collator([f1, f2])
        model = TinyLM(vocab_size=vocab)

        kd = make_kd_compute_loss(alpha=0.0)
        loss_kd = kd.compute_loss(model, batch, return_outputs=False)

        # Reference: run the model and compute plain next-token CE restricted to labels != -100
        # (equivalent to SFT with train_on_responses_only masking).
        out = model(batch["input_ids"])
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        mask = shift_labels != -100
        ce = F.cross_entropy(
            shift_logits[mask].float(),
            shift_labels[mask].long(),
            reduction="mean",
        )

        # Both reductions are token-mean over the same set of positions.
        self.assertTrue(
            torch.allclose(loss_kd.detach().float(), ce.detach().float(), rtol=1e-5, atol=1e-6),
            msg=f"KDTrainer(alpha=0) loss {loss_kd.item()} != plain CE {ce.item()}",
        )

        # Sanity: KD metrics populated.
        self.assertIn("loss_hard", kd._last_kd_metrics)
        self.assertIn("loss_soft", kd._last_kd_metrics)
        self.assertAlmostEqual(kd._last_kd_metrics["loss_hard"], loss_kd.item(), places=5)

    def test_alpha_one_is_finite_and_depends_on_teacher(self):
        """alpha=1: loss is pure soft-CE. Must be finite and change when teacher distribution changes."""
        torch.manual_seed(7)
        vocab = 32
        pad = 0
        collator = KDCollator(pad_token_id=pad)

        base = {
            "input_ids": [1, 2, 3, 4, 5],
            "assistant_start": 2,
            "assistant_len": 3,
            "gold_ids": [3, 4, 5],
            "topk_ids": [[3, 10, 11], [4, 10, 11], [5, 10, 11]],
            "topk_logprobs": [[-0.01, -5.0, -5.0]] * 3,
            "topk_valid_mask": [[True, True, True]] * 3,
            "hard_label_in_topk": [True] * 3,
            "example_index": 0,
        }
        sharp = collator([base])

        # Same features but teacher puts most mass on an off-gold token.
        alt = {**base}
        alt["topk_logprobs"] = [[-5.0, -0.01, -5.0]] * 3
        flat = collator([alt])

        model = TinyLM(vocab_size=vocab)
        kd = make_kd_compute_loss(alpha=1.0, temperature=1.0)

        loss_sharp = kd.compute_loss(model, sharp, return_outputs=False)
        loss_flat = kd.compute_loss(model, flat, return_outputs=False)

        self.assertTrue(torch.isfinite(loss_sharp))
        self.assertTrue(torch.isfinite(loss_flat))
        self.assertNotAlmostEqual(loss_sharp.item(), loss_flat.item(), places=4)


if __name__ == "__main__":
    unittest.main()
