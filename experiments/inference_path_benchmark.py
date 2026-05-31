"""
Tiny end-to-end benchmark for checkpoint evaluation paths.

This script:
1. Loads a small training subset from a JSON dataset (or falls back to eval JSON).
2. Fine-tunes a Qwen2.5 LoRA model with Unsloth for a few steps.
3. Benchmarks several inference paths after training:
   - raw generate() before FastLanguageModel.for_inference()
   - generate() with left vs right padding
   - sequential vs batched generation
   - cache on vs off
   - teacher-forced forward pass throughput
4. Logs timing + JSON extraction accuracy using the existing scorer.

The goal is to validate whether KV cache, padding side, batching, and Unsloth's
inference mode explain the slowdown / bad generations you saw earlier.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from accuracy_eval import calculate_dataset_accuracy, evaluate_single_example  # noqa: E402


DEFAULT_TRAIN_FALLBACK = REPO_ROOT / "eval" / "eval_invoice.json"
DEFAULT_EVAL_FILES = [
    REPO_ROOT / "eval" / "eval_invoice.json",
    REPO_ROOT / "eval" / "eval_vehicle.json",
    REPO_ROOT / "eval" / "eval_order.json",
]


@dataclass
class TrainConfig:
    model_name: str
    max_seq_length: int
    load_in_4bit: bool
    rank: int
    lora_alpha: int
    lora_dropout: float
    use_gradient_checkpointing: str
    train_size: int
    train_steps: int
    train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    seed: int


@dataclass
class BenchmarkResult:
    name: str
    kind: str
    fast_inference_enabled: bool
    padding_side: Optional[str]
    batch_size: int
    use_cache: Optional[bool]
    wall_time_sec: float
    examples_per_sec: float
    loss_value: Optional[float]
    accuracy_pct: Optional[float]
    parse_error_count: Optional[int]
    valid_json_count: Optional[int]
    note: str
    sample_generations: Optional[List[Dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Unsloth inference paths.")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Training JSON with instruction/input/output objects. Falls back to eval data.",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        nargs="+",
        default=[p for p in DEFAULT_EVAL_FILES if p.exists()],
        help="Evaluation JSON files.",
    )
    parser.add_argument("--train-size", type=int, default=24)
    parser.add_argument("--eval-size", type=int, default=18)
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--bench-batch-size",
        type=int,
        default=None,
        help="Single benchmark batch size. If omitted, --bench-batch-sizes is used.",
    )
    parser.add_argument(
        "--bench-batch-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64],
        help="Batch sizes to benchmark for cached left-padded generation.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "inference_path_benchmark_runs",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_examples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a list of examples.")
    return data


def choose_train_examples(
    explicit_train_path: Optional[Path],
    eval_paths: Sequence[Path],
    train_size: int,
) -> Tuple[List[Dict[str, Any]], str]:
    if explicit_train_path is not None:
        examples = load_json_examples(explicit_train_path)
        return examples[:train_size], str(explicit_train_path)

    fallback = next((p for p in [*eval_paths, DEFAULT_TRAIN_FALLBACK] if p.exists()), None)
    if fallback is None:
        raise FileNotFoundError("No train-data provided and no eval fallback JSON found.")
    examples = load_json_examples(fallback)
    return examples[:train_size], f"{fallback} (fallback)"


def choose_eval_examples(eval_paths: Sequence[Path], eval_size: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    all_examples: List[Dict[str, Any]] = []
    used_paths: List[str] = []
    for path in eval_paths:
        if not path.exists():
            continue
        examples = load_json_examples(path)
        all_examples.extend(examples)
        used_paths.append(str(path))
        if len(all_examples) >= eval_size:
            break
    if not all_examples:
        raise FileNotFoundError("No evaluation examples found.")
    return all_examples[:eval_size], used_paths


def format_train_example(tokenizer, example: Dict[str, Any]) -> Dict[str, str]:
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def build_generation_prompts(tokenizer, examples: Sequence[Dict[str, Any]]) -> List[str]:
    prompts = []
    for example in examples:
        messages = [
            {"role": "system", "content": example["instruction"]},
            {"role": "user", "content": example["input"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def decode_generation_outputs(
    tokenizer,
    outputs: torch.Tensor,
    input_width: int,
) -> List[str]:
    decoded: List[str] = []
    for row_idx in range(outputs.shape[0]):
        generated_tokens = outputs[row_idx][input_width:]
        decoded.append(
            tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()
        )
    return decoded


def score_outputs(
    examples: Sequence[Dict[str, Any]],
    generations: Sequence[str],
) -> Tuple[float, int, int]:
    results = []
    parse_error_count = 0
    valid_json_count = 0
    for example, generated_text in zip(examples, generations):
        accuracy, details = evaluate_single_example(
            instruction=example["instruction"],
            input_text=example["input"],
            gt_output=example["output"],
            llm_output=generated_text,
        )
        if "error" in details:
            parse_error_count += 1
        else:
            valid_json_count += 1
        results.append((accuracy, details))
    return calculate_dataset_accuracy(results), parse_error_count, valid_json_count


def truncate_text(text: str, max_chars: int = 220) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def collect_sample_generations(
    examples: Sequence[Dict[str, Any]],
    generations: Sequence[str],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for idx, (example, generation) in enumerate(zip(examples, generations)):
        meta = example.get("_meta", {}) if isinstance(example, dict) else {}
        samples.append(
            {
                "index": idx,
                "sample_id": meta.get("sample_id"),
                "source_file": meta.get("source_file"),
                "generated_snippet": truncate_text(generation),
                "expected_snippet": truncate_text(str(example.get("output", ""))),
            }
        )
        if len(samples) >= limit:
            break
    return samples


def generate_sequential(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    use_cache: bool,
) -> List[str]:
    outputs_text = []
    tokenizer.padding_side = "left"
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        generated_tokens = generated[0][prompt_len:]
        outputs_text.append(
            tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        )
    return outputs_text


def generate_batched(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    batch_size: int,
    padding_side: str,
    use_cache: bool,
) -> List[str]:
    tokenizer.padding_side = padding_side
    outputs_text: List[str] = []

    for start in range(0, len(prompts), batch_size):
        prompt_batch = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        input_width = int(encoded["input_ids"].shape[1])

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=tokenizer.pad_token_id,
            )

        outputs_text.extend(
            decode_generation_outputs(tokenizer, generated, input_width)
        )

    return outputs_text


def run_generation_benchmark(
    name: str,
    model,
    tokenizer,
    examples: Sequence[Dict[str, Any]],
    prompts: Sequence[str],
    max_new_tokens: int,
    batch_size: int,
    padding_side: Optional[str],
    use_cache: bool,
    note: str,
    fast_inference_enabled: bool,
) -> BenchmarkResult:
    model.eval()
    model.config.use_cache = use_cache

    synchronize_if_needed()
    start = time.perf_counter()
    try:
        if batch_size <= 1:
            generations = generate_sequential(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
            )
        else:
            generations = generate_batched(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                padding_side=padding_side or "left",
                use_cache=use_cache,
            )
        synchronize_if_needed()
        wall_time = time.perf_counter() - start

        accuracy_pct, parse_error_count, valid_json_count = score_outputs(examples, generations)
        sample_generations = collect_sample_generations(examples, generations)
        note_text = note
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        synchronize_if_needed()
        wall_time = time.perf_counter() - start
        accuracy_pct = None
        parse_error_count = None
        valid_json_count = None
        sample_generations = None
        note_text = f"{note} OOM at batch_size={batch_size}."

    return BenchmarkResult(
        name=name,
        kind="generation",
        fast_inference_enabled=fast_inference_enabled,
        padding_side=padding_side,
        batch_size=batch_size,
        use_cache=use_cache,
        wall_time_sec=wall_time,
        examples_per_sec=len(examples) / wall_time if wall_time > 0 else 0.0,
        loss_value=None,
        accuracy_pct=accuracy_pct,
        parse_error_count=parse_error_count,
        valid_json_count=valid_json_count,
        note=note_text,
        sample_generations=sample_generations,
    )


def run_eval_loss_pass(
    model,
    tokenizer,
    examples: Sequence[Dict[str, Any]],
    batch_size: int,
    max_seq_length: int,
    fast_inference_enabled: bool,
) -> BenchmarkResult:
    texts = [format_train_example(tokenizer, example)["text"] for example in examples]
    tokenizer.padding_side = "right"

    model.eval()
    model.config.use_cache = False
    synchronize_if_needed()
    start = time.perf_counter()
    total_examples = 0
    total_loss = 0.0
    total_label_count = 0

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx : start_idx + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(model.device)
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100

        with torch.inference_mode():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=labels,
            )
        valid_labels = int((labels != -100).sum().item())
        total_loss += float(outputs.loss.item()) * valid_labels
        total_label_count += valid_labels
        total_examples += len(batch_texts)

    synchronize_if_needed()
    wall_time = time.perf_counter() - start
    avg_loss = total_loss / total_label_count if total_label_count > 0 else None

    return BenchmarkResult(
        name="eval_loss_teacher_forced",
        kind="eval_loss",
        fast_inference_enabled=fast_inference_enabled,
        padding_side="right",
        batch_size=batch_size,
        use_cache=None,
        wall_time_sec=wall_time,
        examples_per_sec=total_examples / wall_time if wall_time > 0 else 0.0,
        loss_value=avg_loss,
        accuracy_pct=None,
        parse_error_count=None,
        valid_json_count=None,
        note="Teacher-forced eval loss pass over the eval set before generation benchmarks.",
        sample_generations=None,
    )


def print_result(result: BenchmarkResult) -> None:
    loss_str = "n/a" if result.loss_value is None else f"{result.loss_value:.4f}"
    accuracy_str = "n/a" if result.accuracy_pct is None else f"{result.accuracy_pct:.2f}%"
    parse_str = "n/a" if result.parse_error_count is None else str(result.parse_error_count)
    valid_str = "n/a" if result.valid_json_count is None else str(result.valid_json_count)
    print(
        f"{result.name:<34} "
        f"time={result.wall_time_sec:>7.2f}s "
        f"ex/s={result.examples_per_sec:>6.2f} "
        f"loss={loss_str:>7} "
        f"acc={accuracy_str:>8} "
        f"valid_json={valid_str:>4} "
        f"parse_errors={parse_str:>4}"
    )
    if result.sample_generations:
        for sample in result.sample_generations:
            sample_label = sample["sample_id"] if sample["sample_id"] is not None else sample["index"]
            print(f"  sample {sample_label} gen: {sample['generated_snippet']}")
            print(f"  sample {sample_label} exp: {sample['expected_snippet']}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    bench_batch_sizes = [args.bench_batch_size] if args.bench_batch_size else args.bench_batch_sizes
    bench_batch_sizes = sorted({size for size in bench_batch_sizes if size >= 1})
    if not bench_batch_sizes:
        raise ValueError("At least one positive benchmark batch size is required.")
    primary_bench_batch_size = bench_batch_sizes[0]

    output_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples, train_source = choose_train_examples(
        explicit_train_path=args.train_data,
        eval_paths=args.eval_data,
        train_size=args.train_size,
    )
    eval_examples, eval_sources = choose_eval_examples(
        eval_paths=args.eval_data,
        eval_size=args.eval_size,
    )

    print(f"Train source: {train_source}")
    print(f"Eval sources: {eval_sources}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Eval examples: {len(eval_examples)}")

    train_cfg = TrainConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing="unsloth",
        train_size=len(train_examples),
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    train_dataset = Dataset.from_list(train_examples)
    train_dataset = train_dataset.map(lambda ex: format_train_example(tokenizer, ex))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=1,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.train_steps,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            seed=args.seed,
            output_dir=str(output_dir / "trainer_outputs"),
            report_to="none",
            save_strategy="no",
            eval_strategy="no",
        ),
    )

    print("Training tiny subset...")
    train_start = time.perf_counter()
    trainer.train()
    synchronize_if_needed()
    train_wall_time = time.perf_counter() - train_start
    print(f"Tiny training completed in {train_wall_time:.2f}s")

    prompts = build_generation_prompts(tokenizer, eval_examples)
    results: List[BenchmarkResult] = []

    print("\nRunning teacher-forced eval-loss pass before generation...")
    eval_loss_result = run_eval_loss_pass(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        batch_size=primary_bench_batch_size,
        max_seq_length=args.max_seq_length,
        fast_inference_enabled=False,
    )
    results.append(eval_loss_result)
    print_result(eval_loss_result)

    print("\nBenchmarks before FastLanguageModel.for_inference():")
    raw_batch_result = run_generation_benchmark(
        name=f"raw_generate_batch_leftpad_cache_bs{primary_bench_batch_size}",
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=primary_bench_batch_size,
        padding_side="left",
        use_cache=True,
        note="Raw post-training generate() before Unsloth inference mode.",
        fast_inference_enabled=False,
    )
    results.append(raw_batch_result)
    print_result(raw_batch_result)

    print("\nSwitching model to FastLanguageModel.for_inference()...")
    FastLanguageModel.for_inference(model)

    post_modes = [
        dict(
            name="fast_generate_sequential_cache",
            batch_size=1,
            padding_side="left",
            use_cache=True,
            note="Sequential generate() with cache enabled.",
        ),
    ]
    for batch_size in bench_batch_sizes:
        post_modes.append(
            dict(
                name=f"fast_generate_batch_leftpad_cache_bs{batch_size}",
                batch_size=batch_size,
                padding_side="left",
                use_cache=True,
                note="Batched generate() with left padding and cache enabled.",
            )
        )
    post_modes.extend(
        [
            dict(
                name=f"fast_generate_batch_leftpad_nocache_bs{primary_bench_batch_size}",
                batch_size=primary_bench_batch_size,
                padding_side="left",
                use_cache=False,
                note="Batched generate() with cache disabled to expose recompute cost.",
            ),
            dict(
                name=f"fast_generate_batch_rightpad_cache_bs{primary_bench_batch_size}",
                batch_size=primary_bench_batch_size,
                padding_side="right",
                use_cache=True,
                note="Intentionally tests right padding on a decoder-only model.",
            ),
        ]
    )

    print("\nBenchmarks after FastLanguageModel.for_inference():")
    for mode in post_modes:
        result = run_generation_benchmark(
            name=mode["name"],
            model=model,
            tokenizer=tokenizer,
            examples=eval_examples,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=mode["batch_size"],
            padding_side=mode["padding_side"],
            use_cache=mode["use_cache"],
            note=mode["note"],
            fast_inference_enabled=True,
        )
        results.append(result)
        print_result(result)

    report = {
        "train_config": asdict(train_cfg),
        "train_source": train_source,
        "eval_sources": eval_sources,
        "bench_batch_sizes": bench_batch_sizes,
        "train_wall_time_sec": train_wall_time,
        "results": [asdict(result) for result in results],
    }

    report_path = output_dir / "benchmark_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
