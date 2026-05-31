"""Knowledge-distillation trainer.

``KDTrainer`` subclasses ``SFTTrainer`` and replaces ``compute_loss`` with

    loss = (1 - alpha) * hard_ce  +  alpha * T^2 * soft_ce_topk

where ``hard_ce`` is the standard next-token CE restricted to the assistant span
(equivalent to ``train_on_responses_only`` masking), and ``soft_ce_topk`` is the
cross-entropy between the student's token distribution (temperature-scaled) and
the teacher's top-K distribution renormalized over those K ids.

The batch is expected to come from ``kd_collator.KDCollator`` and contain:
    input_ids, attention_mask, labels, gold_ids, topk_ids, topk_logprobs,
    topk_valid, span_mask, assistant_start, assistant_len.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from trl import SFTTrainer


class KDTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        kd_alpha: float = 0.4,
        kd_temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kd_alpha = float(kd_alpha)
        self.kd_temperature = float(kd_temperature)
        if not (0.0 <= self.kd_alpha <= 1.0):
            raise ValueError(f"kd_alpha must be in [0, 1], got {self.kd_alpha}")
        if self.kd_temperature <= 0.0:
            raise ValueError(f"kd_temperature must be > 0, got {self.kd_temperature}")
        self._last_kd_metrics: Dict[str, float] = {}

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        gold_ids = inputs["gold_ids"]           # [B, N]
        topk_ids = inputs["topk_ids"]           # [B, N, K]
        topk_logprobs = inputs["topk_logprobs"] # [B, N, K]
        topk_valid = inputs["topk_valid"]       # [B, N, K]
        span_mask = inputs["span_mask"]         # [B, N]
        assistant_start = inputs["assistant_start"]  # [B]

        B, T = input_ids.shape
        N = topk_ids.shape[1]
        K = topk_ids.shape[2]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]
        V = logits.shape[-1]

        device = logits.device
        # Positions in the *sequence* whose logits predict the assistant span.
        # Student logits at position (start + j - 1) predict assistant token j.
        j_index = torch.arange(N, device=device).unsqueeze(0).expand(B, N)          # [B, N]
        pos = (assistant_start.to(device).unsqueeze(1) - 1) + j_index               # [B, N]
        # All valid positions are in [0, T-1] by construction (start>=1, start+N<=T).
        pos_clamped = pos.clamp(min=0, max=T - 1)
        gather_idx = pos_clamped.unsqueeze(-1).expand(B, N, V)
        student_slice = logits.gather(1, gather_idx)  # [B, N, V] in model dtype

        mask = span_mask.to(device)  # [B, N]
        denom = mask.sum().clamp(min=1).to(student_slice.dtype)

        # Teacher top-K / gold ids may reference vocab slots that don't exist in the
        # student (Qwen2.5-14B has V=152064 vs 0.5B V=151936). Mask those out of the
        # soft target and clamp gold ids into range — their span_mask positions stay
        # valid, but out-of-vocab slots contribute no loss.
        topk_ids_dev = topk_ids.to(device)
        topk_valid_dev = topk_valid.to(device)
        oob_topk = topk_ids_dev >= V
        if oob_topk.any():
            topk_valid_dev = topk_valid_dev & ~oob_topk
            topk_ids_dev = topk_ids_dev.masked_fill(oob_topk, 0)

        gold_ids_dev = gold_ids.to(device)
        oob_gold = gold_ids_dev >= V
        if oob_gold.any():
            # Drop these positions from the hard-CE mask; clamp ids so the gather is safe.
            mask = mask & ~oob_gold
            gold_ids_dev = gold_ids_dev.masked_fill(oob_gold, 0)

        # --- Hard CE ------------------------------------------------------
        student_slice_f32 = student_slice.float()
        ce_per_token = F.cross_entropy(
            student_slice_f32.reshape(B * N, V),
            gold_ids_dev.reshape(B * N),
            reduction="none",
        ).reshape(B, N)
        loss_hard = (ce_per_token * mask.float()).sum() / mask.float().sum().clamp(min=1)

        # --- Soft CE over teacher top-K ----------------------------------
        temperature = self.kd_temperature
        student_logp = F.log_softmax(student_slice_f32 / temperature, dim=-1)  # [B, N, V]
        # Gather at teacher top-K ids.
        student_logp_k = student_logp.gather(-1, topk_ids_dev)  # [B, N, K]

        # Teacher distribution: softmax of stored top-K logprobs / T, masked for invalid slots.
        teacher_logits_k = topk_logprobs.to(device) / temperature
        invalid = ~topk_valid_dev
        teacher_logits_k = teacher_logits_k.masked_fill(invalid, float("-inf"))
        teacher_p = torch.softmax(teacher_logits_k, dim=-1)  # [B, N, K]
        # If an entire row had no valid slots (shouldn't happen in a correct artifact), guard it.
        teacher_p = torch.nan_to_num(teacher_p, nan=0.0)

        soft_per_token = -(teacher_p * student_logp_k).sum(dim=-1)  # [B, N]
        loss_soft = (temperature ** 2) * (soft_per_token * mask.float()).sum() / mask.float().sum().clamp(min=1)

        alpha = self.kd_alpha
        loss = (1.0 - alpha) * loss_hard + alpha * loss_soft

        # --- Diagnostics --------------------------------------------------
        with torch.no_grad():
            hard_in_topk = (topk_ids_dev == gold_ids_dev.unsqueeze(-1)).any(dim=-1).float()
            frac_hard = (hard_in_topk * mask.float()).sum() / mask.float().sum().clamp(min=1)
            self._last_kd_metrics = {
                "loss_hard": loss_hard.detach().float().item(),
                "loss_soft": loss_soft.detach().float().item(),
                "kd_alpha": alpha,
                "kd_temperature": temperature,
                "frac_hard_in_topk": frac_hard.detach().float().item(),
                "kd_tokens_per_batch": float(mask.float().sum().item()),
            }

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:  # type: ignore[override]
        if self._last_kd_metrics and any(key.startswith("loss") for key in logs):
            logs = {**logs, **self._last_kd_metrics}
        return super().log(logs, *args, **kwargs)
