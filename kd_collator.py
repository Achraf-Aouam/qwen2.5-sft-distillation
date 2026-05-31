"""Batch collator for KD features produced by ``KL_div_prep/build_kd_dataset.py``.

Pads ``input_ids`` / ``labels`` to the per-batch max sequence length (right-side pad,
matching HF Trainer convention), and pads the per-token teacher top-K tensors to the
per-batch max assistant span length. Padded top-K slots are masked via ``topk_valid``
and their stored log-probabilities are set to a large negative number so ``softmax``
over them produces ~0 mass before renormalization.

Returned keys per batch (all tensors):
    input_ids        : LongTensor  [B, T]
    attention_mask   : LongTensor  [B, T]
    labels           : LongTensor  [B, T]      -100 outside the assistant span
    gold_ids         : LongTensor  [B, N]      padded with 0 where not valid
    topk_ids         : LongTensor  [B, N, K]   padded with 0 where not valid
    topk_logprobs    : FloatTensor [B, N, K]   padded with -1e30 where not valid
    topk_valid       : BoolTensor  [B, N, K]
    span_mask        : BoolTensor  [B, N]      True where j < assistant_len[b]
    assistant_start  : LongTensor  [B]
    assistant_len    : LongTensor  [B]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

NEG_INF_LOGPROB = -1.0e30


@dataclass
class KDCollator:
    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("KDCollator received an empty batch.")

        batch_size = len(features)
        seq_lens = [len(feat["input_ids"]) for feat in features]
        span_lens = [int(feat["assistant_len"]) for feat in features]
        top_k = len(features[0]["topk_ids"][0]) if span_lens[0] > 0 else 0
        for feat in features:
            if feat["assistant_len"] > 0 and len(feat["topk_ids"][0]) != top_k:
                raise ValueError("Inconsistent top-K width across batch.")

        max_seq_len = max(seq_lens)
        max_span_len = max(span_lens) if span_lens else 0

        input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        labels = torch.full((batch_size, max_seq_len), self.label_pad_token_id, dtype=torch.long)

        gold_ids = torch.zeros((batch_size, max_span_len), dtype=torch.long)
        topk_ids = torch.zeros((batch_size, max_span_len, top_k), dtype=torch.long)
        topk_logprobs = torch.full((batch_size, max_span_len, top_k), NEG_INF_LOGPROB, dtype=torch.float32)
        topk_valid = torch.zeros((batch_size, max_span_len, top_k), dtype=torch.bool)
        span_mask = torch.zeros((batch_size, max_span_len), dtype=torch.bool)

        assistant_start = torch.zeros(batch_size, dtype=torch.long)
        assistant_len = torch.zeros(batch_size, dtype=torch.long)

        for b, feat in enumerate(features):
            ids = feat["input_ids"]
            start = int(feat["assistant_start"])
            n = int(feat["assistant_len"])
            if start < 1:
                raise ValueError(
                    f"assistant_start must be >= 1 (needed for gather at start-1). Got {start}."
                )
            if start + n > len(ids):
                raise ValueError(
                    f"assistant_start+assistant_len exceeds input_ids length: "
                    f"{start}+{n} > {len(ids)}."
                )

            ids_tensor = torch.as_tensor(ids, dtype=torch.long)
            input_ids[b, : ids_tensor.shape[0]] = ids_tensor
            attention_mask[b, : ids_tensor.shape[0]] = 1

            gold = torch.as_tensor(feat["gold_ids"], dtype=torch.long)
            labels[b, start : start + n] = gold
            gold_ids[b, :n] = gold

            if n > 0:
                topk_ids[b, :n] = torch.as_tensor(feat["topk_ids"], dtype=torch.long)
                topk_logprobs[b, :n] = torch.as_tensor(feat["topk_logprobs"], dtype=torch.float32)
                topk_valid[b, :n] = torch.as_tensor(feat["topk_valid_mask"], dtype=torch.bool)
            span_mask[b, :n] = True

            assistant_start[b] = start
            assistant_len[b] = n

        # Where teacher candidate slots are invalid (padded), force their log-prob to -inf so they
        # contribute ~0 mass to the renormalized distribution.
        topk_logprobs = torch.where(topk_valid, topk_logprobs, torch.full_like(topk_logprobs, NEG_INF_LOGPROB))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "gold_ids": gold_ids,
            "topk_ids": topk_ids,
            "topk_logprobs": topk_logprobs,
            "topk_valid": topk_valid,
            "span_mask": span_mask,
            "assistant_start": assistant_start,
            "assistant_len": assistant_len,
        }
