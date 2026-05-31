"""Inline generation evaluation, artifact writing, and trainer callback helpers."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from datasets import Dataset
from transformers import TrainerCallback

from accuracy_eval import evaluate_single_example


@dataclass
class CorpusSpec:
    name: str
    path: Path
    examples: List[Dict[str, Any]]
    eval_dataset: Dataset
    prompts: List[str]
    supervised_token_count: int


def load_json_examples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a list of examples.")
    return data


def format_training_text(tokenizer, example: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_generation_prompt(tokenizer, example: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_text_dataset(tokenizer, examples: List[Dict[str, Any]]) -> Dataset:
    rows = [{**example, "text": format_training_text(tokenizer, example)} for example in examples]
    return Dataset.from_list(rows)


def compute_supervised_token_count(tokenizer, examples: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for example in examples:
        token_ids = tokenizer(str(example.get("output", "")), add_special_tokens=False)["input_ids"]
        total += len(token_ids)
    return max(total, 1)


def load_corpus_specs(tokenizer, eval_files: Dict[str, Path]) -> Dict[str, CorpusSpec]:
    corpora: Dict[str, CorpusSpec] = {}
    for name, path in eval_files.items():
        examples = load_json_examples(path)
        corpora[name] = CorpusSpec(
            name=name,
            path=path,
            examples=examples,
            eval_dataset=build_text_dataset(tokenizer, examples),
            prompts=[build_generation_prompt(tokenizer, example) for example in examples],
            supervised_token_count=compute_supervised_token_count(tokenizer, examples),
        )
    return corpora


class LocalArtifactLogger:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.base_dir / "history.csv"
        self._history_rows = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        if not self.history_path.exists():
            return []
        with self.history_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    def logged_steps(self) -> set[int]:
        steps: set[int] = set()
        for row in self._history_rows:
            try:
                steps.add(int(row["step"]))
            except (KeyError, TypeError, ValueError):
                continue
        return steps

    def write_step(self, step: int, payload: Dict[str, Any], samples: List[Dict[str, Any]]) -> None:
        summary_path = self.base_dir / f"step_{step}_summary.json"
        samples_path = self.base_dir / f"step_{step}_samples.jsonl"

        summary = {
            "step": step,
            "payload": payload,
            "sample_count": len(samples),
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        with samples_path.open("w", encoding="utf-8") as handle:
            for record in samples:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        row = {"step": step, **payload}
        updated = False
        for index, existing in enumerate(self._history_rows):
            if str(existing.get("step")) == str(step):
                self._history_rows[index] = row
                updated = True
                break
        if not updated:
            self._history_rows.append(row)
        self._write_history()

    def _write_history(self) -> None:
        fieldnames = sorted({key for row in self._history_rows for key in row.keys()})
        with self.history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(self._history_rows, key=lambda item: int(item["step"])):
                writer.writerow(row)


class InlineGenerationEvaluator:
    def __init__(
        self,
        tokenizer,
        corpora: Dict[str, CorpusSpec],
        generation_batch_size: int,
        max_new_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.corpora = corpora
        self.generation_batch_size = generation_batch_size
        self.max_new_tokens = max_new_tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _enter_inference_mode(self, model) -> Dict[str, Any]:
        from unsloth import FastLanguageModel

        state = {
            "padding_side": self.tokenizer.padding_side,
            "use_cache": getattr(model.config, "use_cache", None),
            "was_training": model.training,
        }
        FastLanguageModel.for_inference(model)
        model.eval()
        model.config.use_cache = True
        self.tokenizer.padding_side = "left"
        return state

    def _restore_training_mode(self, model, state: Dict[str, Any]) -> None:
        from unsloth import FastLanguageModel

        self.tokenizer.padding_side = state["padding_side"]
        if state["use_cache"] is not None:
            model.config.use_cache = state["use_cache"]
        FastLanguageModel.for_training(model)
        if state["was_training"]:
            model.train()

    def _generate_for_prompts(self, model, prompts: List[str]) -> List[str]:
        outputs_text: List[str] = []
        for start in range(0, len(prompts), self.generation_batch_size):
            prompt_batch = prompts[start : start + self.generation_batch_size]
            encoded = self.tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)
            input_width = int(encoded["input_ids"].shape[1])

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for row in range(generated.shape[0]):
                token_slice = generated[row][input_width:]
                outputs_text.append(
                    self.tokenizer.decode(token_slice, skip_special_tokens=True).strip()
                )
        return outputs_text

    def evaluate(
        self,
        model,
        step: int,
        loss_payload: Optional[Dict[str, float]] = None,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        self._synchronize()
        total_start = time.perf_counter()
        state = self._enter_inference_mode(model)
        payload: Dict[str, Any] = {"step": step}
        if loss_payload:
            payload.update(loss_payload)

        total_examples = 0
        total_correct = 0
        total_fields = 0
        total_parse_errors = 0
        total_valid_json = 0
        sample_rows: List[Dict[str, Any]] = []

        try:
            for corpus_name, corpus in self.corpora.items():
                corpus_start = time.perf_counter()
                generations = self._generate_for_prompts(model, corpus.prompts)
                self._synchronize()
                corpus_time = time.perf_counter() - corpus_start

                corpus_correct = 0
                corpus_total = 0
                corpus_parse_errors = 0
                corpus_valid_json = 0

                for example, generation in zip(corpus.examples, generations):
                    accuracy, details = evaluate_single_example(
                        instruction=example["instruction"],
                        input_text=example["input"],
                        gt_output=example["output"],
                        llm_output=generation,
                    )
                    corpus_correct += details.get("correct", 0)
                    corpus_total += details.get("total", 0)
                    corpus_valid_json += 1 if details.get("valid_json") else 0
                    corpus_parse_errors += 0 if details.get("valid_json") else 1

                    meta = example.get("_meta", {})
                    sample_rows.append(
                        {
                            "step": step,
                            "corpus": corpus_name,
                            "sample_id": meta.get("sample_id"),
                            "document_name": meta.get("document_name"),
                            "source_file": meta.get("source_file"),
                            "gt_json": example.get("output"),
                            "raw_output": generation,
                            "parsed_json": details.get("parsed_output"),
                            "parse_result": details.get("parse_result"),
                            "valid_json": details.get("valid_json"),
                            "accuracy_pct": accuracy * 100,
                            "correct_fields": details.get("correct", 0),
                            "total_fields": details.get("total", 0),
                            "field_matches": details.get("field_matches", []),
                        }
                    )

                payload[f"gt_acc/{corpus_name}"] = (corpus_correct / corpus_total * 100) if corpus_total else 0.0
                payload[f"gt_parse_errors/{corpus_name}"] = corpus_parse_errors
                payload[f"gt_valid_json/{corpus_name}"] = corpus_valid_json
                payload[f"gt_matched_fields/{corpus_name}"] = corpus_correct
                payload[f"gt_total_fields/{corpus_name}"] = corpus_total
                payload[f"eval_examples/{corpus_name}"] = len(corpus.examples)
                payload[f"eval_time/generation_{corpus_name}_sec"] = corpus_time
                payload[f"throughput/{corpus_name}_examples_per_sec"] = (
                    len(corpus.examples) / corpus_time if corpus_time > 0 else 0.0
                )

                total_examples += len(corpus.examples)
                total_correct += corpus_correct
                total_fields += corpus_total
                total_parse_errors += corpus_parse_errors
                total_valid_json += corpus_valid_json
        finally:
            self._restore_training_mode(model, state)

        total_time = time.perf_counter() - total_start
        payload["gt_acc/overall_weighted"] = (total_correct / total_fields * 100) if total_fields else 0.0
        payload["gt_parse_errors/overall"] = total_parse_errors
        payload["gt_valid_json/overall"] = total_valid_json
        payload["gt_matched_fields/overall"] = total_correct
        payload["gt_total_fields/overall"] = total_fields
        payload["eval_examples/overall"] = total_examples
        payload["eval_time/generation_total_sec"] = total_time
        payload["throughput/overall_examples_per_sec"] = total_examples / total_time if total_time > 0 else 0.0
        return payload, sample_rows


class InlineEvalCallback(TrainerCallback):
    def __init__(
        self,
        corpora: Dict[str, CorpusSpec],
        generation_evaluator: InlineGenerationEvaluator,
        artifact_logger: LocalArtifactLogger,
        use_wandb: bool = False,
    ):
        self.corpora = corpora
        self.generation_evaluator = generation_evaluator
        self.artifact_logger = artifact_logger
        self.use_wandb = use_wandb
        self.pending_loss_payload: Dict[str, float] = {}
        self.pending_loss_step: Optional[int] = None
        self.logged_steps: set[int] = artifact_logger.logged_steps()

    def on_train_begin(self, args, state, control, **kwargs):
        if args.save_strategy != args.eval_strategy:
            raise ValueError("save_strategy and eval_strategy must match for inline eval alignment.")
        if args.save_steps != args.eval_steps:
            raise ValueError("save_steps must equal eval_steps for inline eval alignment.")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return control

        metrics = metrics or {}
        total_weighted_loss = 0.0
        total_weight = 0
        cleaned: Dict[str, float] = {}
        for corpus_name, corpus in self.corpora.items():
            raw_loss = metrics.get(f"eval_{corpus_name}_loss")
            if raw_loss is None:
                continue
            raw_loss = float(raw_loss)
            cleaned[f"loss/{corpus_name}"] = raw_loss
            total_weighted_loss += raw_loss * corpus.supervised_token_count
            total_weight += corpus.supervised_token_count

        if total_weight > 0:
            cleaned["loss/overall_weighted"] = total_weighted_loss / total_weight

        self.pending_loss_payload = cleaned
        self.pending_loss_step = int(state.global_step)
        return control

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control

        step = int(state.global_step)
        if step in self.logged_steps:
            return control

        loss_payload = self.pending_loss_payload if self.pending_loss_step == step else {}
        payload, samples = self.generation_evaluator.evaluate(
            model=kwargs["model"],
            step=step,
            loss_payload=loss_payload,
        )
        self.artifact_logger.write_step(step, payload, samples)
        self.logged_steps.add(step)

        if self.use_wandb:
            try:
                import wandb  # type: ignore

                if wandb.run is not None:
                    wandb.log(payload, step=step)
            except ImportError:
                pass
        return control
