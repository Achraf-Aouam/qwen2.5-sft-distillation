import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import Trainer


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_raw_data(raw_data: list[dict[str, Any]], eval_num: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = []
    for sample_id, item in enumerate(raw_data):
        row = dict(item)
        row["_sample_id"] = sample_id
        indexed.append(row)

    rng = random.Random(seed)
    rng.shuffle(indexed)

    if eval_num <= 0:
        return indexed, []
    if eval_num >= len(indexed):
        raise ValueError(f"eval_num={eval_num} is too large for a dataset with only {len(indexed)} rows.")

    return indexed[:-eval_num], indexed[-eval_num:]


def build_prompt_messages(instruction: str, input_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]


def build_full_messages(instruction: str, input_text: str, output_text: str) -> list[dict[str, str]]:
    return build_prompt_messages(instruction, input_text) + [
        {"role": "assistant", "content": output_text or ""},
    ]


def format_token_for_display(token: str) -> str:
    display = token.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if display == "":
        return "<empty>"
    return display


@dataclass
class TeacherAPIConfig:
    base_url: str
    api_key: str
    model_name: str
    top_logprobs: int = 10
    max_tokens: int = 512
    token_limit_field: str = "max_tokens"
    temperature: float = 0.0
    timeout_s: float = 120.0
    max_retries: int = 3
    retry_sleep_s: float = 5.0
    sleep_between_requests_s: float = 0.0
    response_format_json: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)

    def completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"


def _extract_text_from_message(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        chunks: list[str] = []
        for item in message_content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"text", "output_text"} and isinstance(item.get("text"), str):
                chunks.append(item["text"])
            elif isinstance(item.get("content"), str):
                chunks.append(item["content"])
        return "".join(chunks)

    return str(message_content)


def _normalize_logprob_steps(choice: dict[str, Any]) -> list[dict[str, Any]]:
    raw_steps = choice.get("logprobs", {}).get("content") or []
    steps: list[dict[str, Any]] = []

    for position, step in enumerate(raw_steps):
        top_candidates = []
        for candidate in step.get("top_logprobs") or []:
            logprob = candidate.get("logprob")
            if logprob is None:
                continue
            top_candidates.append(
                {
                    "token": candidate.get("token", ""),
                    "logprob": float(logprob),
                    "prob": float(math.exp(logprob)),
                }
            )

        token_logprob = step.get("logprob")
        if token_logprob is None:
            continue

        steps.append(
            {
                "position": position,
                "token": step.get("token", ""),
                "logprob": float(token_logprob),
                "prob": float(math.exp(token_logprob)),
                "top_candidates": top_candidates,
            }
        )

    return steps


def request_teacher_completion(
    session: requests.Session,
    api_config: TeacherAPIConfig,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": api_config.model_name,
        "messages": messages,
        "temperature": api_config.temperature,
        "logprobs": True,
        "top_logprobs": api_config.top_logprobs,
    }
    payload[api_config.token_limit_field] = api_config.max_tokens
    payload.update(api_config.extra_body)

    if api_config.response_format_json:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_config.api_key}",
        "Content-Type": "application/json",
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, api_config.max_retries + 1):
        try:
            response = session.post(
                api_config.completions_url(),
                headers=headers,
                json=payload,
                timeout=api_config.timeout_s,
            )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            teacher_text = _extract_text_from_message(choice["message"]["content"])
            return {
                "teacher_output": teacher_text,
                "teacher_steps": _normalize_logprob_steps(choice),
                "raw_response": body,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == api_config.max_retries:
                break
            time.sleep(api_config.retry_sleep_s)

    raise RuntimeError(f"Teacher request failed after {api_config.max_retries} attempts: {last_error}") from last_error


def _read_cached_records(cache_path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not cache_path.exists():
        return records

    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records[int(row["_sample_id"])] = row
    return records


def load_cached_records(cache_path: Path) -> list[dict[str, Any]]:
    records = _read_cached_records(cache_path)
    return [records[key] for key in sorted(records)]


def prefetch_teacher_cache(
    samples: list[dict[str, Any]],
    cache_path: Path,
    api_config: TeacherAPIConfig,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    ensure_dir(cache_path.parent)
    cached = {} if overwrite else _read_cached_records(cache_path)

    if overwrite and cache_path.exists():
        cache_path.unlink()

    session = requests.Session()

    with cache_path.open("a", encoding="utf-8") as sink:
        for sample in samples:
            sample_id = int(sample["_sample_id"])
            if sample_id in cached:
                continue

            prompt_messages = build_prompt_messages(sample["instruction"], sample["input"])
            teacher_result = request_teacher_completion(session, api_config, prompt_messages)

            record = {
                "_sample_id": sample_id,
                "instruction": sample["instruction"],
                "input": sample["input"],
                "output": sample.get("output", ""),
                "_meta": sample.get("_meta"),
                "teacher_model": api_config.model_name,
                "teacher_output": teacher_result["teacher_output"],
                "teacher_steps": teacher_result["teacher_steps"],
            }

            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            cached[sample_id] = record

            if api_config.sleep_between_requests_s > 0:
                time.sleep(api_config.sleep_between_requests_s)

    return [cached[int(sample["_sample_id"])] for sample in samples]


def teacher_preview_rows(record: dict[str, Any], max_rows: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in record.get("teacher_steps", [])[:max_rows]:
        formatted_candidates = []
        for candidate in step.get("top_candidates", [])[:10]:
            formatted_candidates.append(
                f"{format_token_for_display(candidate['token'])} ({candidate['prob']:.4f})"
            )
        rows.append(
            {
                "position": step["position"],
                "teacher_token": format_token_for_display(step["token"]),
                "teacher_prob": round(step["prob"], 6),
                "top_10": " | ".join(formatted_candidates),
            }
        )
    return rows


def _as_token_ids(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return list(token_ids)


def _truncate(ids: list[int], max_seq_length: int) -> list[int]:
    if len(ids) <= max_seq_length:
        return ids
    return ids[:max_seq_length]


def _build_labels(full_ids: list[int], prompt_len: int) -> list[int]:
    effective_prompt_len = min(prompt_len, len(full_ids))
    return ([-100] * effective_prompt_len) + full_ids[effective_prompt_len:]


def _maybe_single_token_id(tokenizer: Any, token_text: str) -> Optional[int]:
    token_ids = tokenizer.encode(token_text, add_special_tokens=False)
    if len(token_ids) == 1:
        return int(token_ids[0])
    return None


def _candidate_distribution(
    tokenizer: Any,
    step: dict[str, Any],
    top_k: int,
) -> Optional[tuple[list[int], list[float]]]:
    merged: dict[int, float] = {}

    for candidate in step.get("top_candidates", []):
        token_id = _maybe_single_token_id(tokenizer, candidate.get("token", ""))
        if token_id is None:
            continue
        merged[token_id] = merged.get(token_id, 0.0) + float(candidate["prob"])

    teacher_token_id = _maybe_single_token_id(tokenizer, step.get("token", ""))
    if teacher_token_id is not None:
        merged[teacher_token_id] = merged.get(teacher_token_id, 0.0) + float(step.get("prob", 0.0))

    if not merged:
        return None

    sorted_items = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:top_k]
    total = sum(prob for _, prob in sorted_items)
    if total <= 0:
        return None

    ids = [token_id for token_id, _ in sorted_items]
    probs = [prob / total for _, prob in sorted_items]
    return ids, probs


def build_distillation_example(
    record: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
    top_k: int,
) -> dict[str, Any]:
    prompt_messages = build_prompt_messages(record["instruction"], record["input"])
    hard_messages = build_full_messages(record["instruction"], record["input"], record.get("output", ""))
    soft_messages = build_full_messages(record["instruction"], record["input"], record.get("teacher_output", ""))

    prompt_ids = _truncate(_as_token_ids(tokenizer, prompt_messages, add_generation_prompt=True), max_seq_length)
    hard_ids = _truncate(_as_token_ids(tokenizer, hard_messages, add_generation_prompt=False), max_seq_length)
    soft_ids = _truncate(_as_token_ids(tokenizer, soft_messages, add_generation_prompt=False), max_seq_length)

    hard_labels = _build_labels(hard_ids, len(prompt_ids))
    soft_target_length = max(len(soft_ids) - 1, 0)
    soft_target_ids = [[-1] * top_k for _ in range(soft_target_length)]
    soft_target_probs = [[0.0] * top_k for _ in range(soft_target_length)]
    soft_target_mask = [0] * soft_target_length

    assistant_soft_token_count = max(len(soft_ids) - len(prompt_ids), 0)
    aligned_steps = min(len(record.get("teacher_steps", [])), assistant_soft_token_count)

    for offset in range(aligned_steps):
        shift_index = len(prompt_ids) + offset - 1
        if shift_index < 0 or shift_index >= soft_target_length:
            continue

        mapped = _candidate_distribution(tokenizer, record["teacher_steps"][offset], top_k=top_k)
        if mapped is None:
            continue

        candidate_ids, candidate_probs = mapped
        soft_target_mask[shift_index] = 1
        soft_target_ids[shift_index][: len(candidate_ids)] = candidate_ids
        soft_target_probs[shift_index][: len(candidate_probs)] = candidate_probs

    return {
        "hard_input_ids": hard_ids,
        "hard_attention_mask": [1] * len(hard_ids),
        "hard_labels": hard_labels,
        "soft_input_ids": soft_ids,
        "soft_attention_mask": [1] * len(soft_ids),
        "soft_target_ids": soft_target_ids,
        "soft_target_probs": soft_target_probs,
        "soft_target_mask": soft_target_mask,
        "_sample_id": int(record["_sample_id"]),
    }


def build_hard_example(
    record: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    prompt_messages = build_prompt_messages(record["instruction"], record["input"])
    hard_messages = build_full_messages(record["instruction"], record["input"], record.get("output", ""))

    prompt_ids = _truncate(_as_token_ids(tokenizer, prompt_messages, add_generation_prompt=True), max_seq_length)
    hard_ids = _truncate(_as_token_ids(tokenizer, hard_messages, add_generation_prompt=False), max_seq_length)

    return {
        "hard_input_ids": hard_ids,
        "hard_attention_mask": [1] * len(hard_ids),
        "hard_labels": _build_labels(hard_ids, len(prompt_ids)),
        "_sample_id": int(record["_sample_id"]),
    }


def build_distillation_dataset(
    records: Iterable[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
    top_k: int,
) -> Dataset:
    rows = [build_distillation_example(record, tokenizer, max_seq_length, top_k) for record in records]
    return Dataset.from_list(rows)


def build_hard_dataset(
    records: Iterable[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> Dataset:
    rows = [build_hard_example(record, tokenizer, max_seq_length) for record in records]
    return Dataset.from_list(rows)


class DistillationDataCollator:
    def __init__(
        self,
        pad_token_id: int,
        top_k: int,
        label_pad_token_id: int = -100,
        soft_candidate_pad_id: int = -1,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.top_k = top_k
        self.label_pad_token_id = label_pad_token_id
        self.soft_candidate_pad_id = soft_candidate_pad_id

    @staticmethod
    def _pad_1d(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        padded = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def _pad_2d(self, sequences: list[list[list[int]]], pad_value: int) -> torch.Tensor:
        max_rows = max(len(seq) for seq in sequences)
        padded_batch = []
        for seq in sequences:
            padded_rows = [row + [pad_value] * (self.top_k - len(row)) for row in seq]
            padded_rows.extend([[pad_value] * self.top_k for _ in range(max_rows - len(seq))])
            padded_batch.append(padded_rows)
        return torch.tensor(padded_batch, dtype=torch.long)

    def _pad_2d_float(self, sequences: list[list[list[float]]], pad_value: float) -> torch.Tensor:
        max_rows = max(len(seq) for seq in sequences)
        padded_batch = []
        for seq in sequences:
            padded_rows = [row + [pad_value] * (self.top_k - len(row)) for row in seq]
            padded_rows.extend([[pad_value] * self.top_k for _ in range(max_rows - len(seq))])
            padded_batch.append(padded_rows)
        return torch.tensor(padded_batch, dtype=torch.float32)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {
            "hard_input_ids": self._pad_1d([feature["hard_input_ids"] for feature in features], self.pad_token_id),
            "hard_attention_mask": self._pad_1d([feature["hard_attention_mask"] for feature in features], 0),
            "hard_labels": self._pad_1d([feature["hard_labels"] for feature in features], self.label_pad_token_id),
        }

        if "soft_input_ids" not in features[0]:
            return batch

        batch["soft_input_ids"] = self._pad_1d(
            [feature["soft_input_ids"] for feature in features],
            self.pad_token_id,
        )
        batch["soft_attention_mask"] = self._pad_1d(
            [feature["soft_attention_mask"] for feature in features],
            0,
        )
        batch["soft_target_ids"] = self._pad_2d(
            [feature["soft_target_ids"] for feature in features],
            self.soft_candidate_pad_id,
        )
        batch["soft_target_probs"] = self._pad_2d_float(
            [feature["soft_target_probs"] for feature in features],
            0.0,
        )
        batch["soft_target_mask"] = self._pad_1d(
            [feature["soft_target_mask"] for feature in features],
            0,
        ).bool()
        return batch


class CachedTeacherDistillationTrainer(Trainer):
    def __init__(self, *args: Any, soft_label_weight: float = 0.3, distill_temperature: float = 1.0, **kwargs: Any) -> None:
        self.soft_label_weight = soft_label_weight
        self.distill_temperature = distill_temperature
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Any:
        hard_outputs = model(
            input_ids=inputs["hard_input_ids"],
            attention_mask=inputs["hard_attention_mask"],
            labels=inputs["hard_labels"],
        )
        hard_loss = hard_outputs.loss
        soft_loss = hard_loss.new_zeros(())

        if "soft_input_ids" in inputs:
            soft_outputs = model(
                input_ids=inputs["soft_input_ids"],
                attention_mask=inputs["soft_attention_mask"],
            )

            shift_logits = soft_outputs.logits[:, :-1, :]
            soft_target_mask = inputs["soft_target_mask"]

            if shift_logits.shape[1] != soft_target_mask.shape[1]:
                target_width = min(shift_logits.shape[1], soft_target_mask.shape[1])
                shift_logits = shift_logits[:, :target_width, :]
                soft_target_mask = soft_target_mask[:, :target_width]
                soft_target_ids = inputs["soft_target_ids"][:, :target_width, :]
                soft_target_probs = inputs["soft_target_probs"][:, :target_width, :]
            else:
                soft_target_ids = inputs["soft_target_ids"]
                soft_target_probs = inputs["soft_target_probs"]

            if soft_target_mask.any():
                selected_logits = shift_logits[soft_target_mask]
                selected_candidate_ids = soft_target_ids[soft_target_mask]
                selected_candidate_probs = soft_target_probs[soft_target_mask]

                valid_candidates = selected_candidate_ids >= 0
                if valid_candidates.any():
                    student_log_probs = F.log_softmax(selected_logits / self.distill_temperature, dim=-1)
                    gathered_log_probs = torch.gather(
                        student_log_probs,
                        1,
                        selected_candidate_ids.clamp_min(0),
                    )
                    gathered_log_probs = gathered_log_probs.masked_fill(~valid_candidates, 0.0)

                    candidate_probs = selected_candidate_probs.masked_fill(~valid_candidates, 0.0)
                    candidate_probs = candidate_probs / candidate_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    soft_loss = -(candidate_probs * gathered_log_probs).sum(dim=-1).mean()
                    soft_loss = soft_loss * (self.distill_temperature ** 2)

        total_loss = ((1.0 - self.soft_label_weight) * hard_loss) + (self.soft_label_weight * soft_loss)

        if return_outputs:
            return total_loss, hard_outputs
        return total_loss
