"""Materialize teacher parquet shards into a tokenized, mmap-ready Arrow dataset.

Run this locally once; copy the output directory to the training VM. The dataset
produced here is consumed by ``train_kd.py`` via ``datasets.load_from_disk``.

Per-row fields:
    input_ids         : list[int32]            full_chat_text tokenized with the student tokenizer
    assistant_start   : int32                  index where the assistant span starts in input_ids
    assistant_len     : int32                  length of the assistant span
    gold_ids          : list[int32]  [N]       teacher-stored assistant token ids (hard labels)
    topk_ids          : list[list[int32]]  [N, K]
    topk_logprobs     : list[list[float32]] [N, K]  (masked slots = -inf)
    topk_valid_mask   : list[list[bool]]   [N, K]  (True where the slot is a real candidate)
    hard_label_in_topk: list[bool]    [N]
    example_index     : int64                  original example index (for provenance)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence as HFSequence, Value

DEFAULT_SOURCE_DIR = Path("KL_div_prep/artifacts/qwen2.5-14b-top10")
DEFAULT_OUTPUT_DIR = Path("KL_div_prep/artifacts/qwen2.5-14b-top10-kd")
DEFAULT_TRAIN_CONFIG = Path("train_config.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                        help="Directory with manifest.json + shards/*.parquet produced by prepare_soft_labels.py.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Destination for the Dataset.save_to_disk() output.")
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG,
                        help="Training config used to look up the student model name / tokenizer.")
    parser.add_argument("--student-model", type=str, default=None,
                        help="Override the student model name (otherwise read from train_config.toml).")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Optional limit for smoke tests.")
    parser.add_argument("--num-proc", type=int, default=1,
                        help="Dataset.save_to_disk num_shards (kept at 1 for mmap simplicity).")
    parser.add_argument("--preflight-samples", type=int, default=8,
                        help="How many rows to verify via student-retokenization before converting everything.")
    return parser.parse_args()


def read_student_model_name(config_path: Path) -> str:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    return str(config["model"]["model_name"]).strip()


def chat_template_fingerprint(tokenizer) -> str:
    template = getattr(tokenizer, "chat_template", None) or ""
    payload = template if isinstance(template, str) else json.dumps(template, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(source_dir: Path) -> Dict[str, Any]:
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json at {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def shard_paths(source_dir: Path, manifest: Dict[str, Any]) -> List[Path]:
    shards = sorted(manifest.get("shards", []), key=lambda item: int(item["shard_index"]))
    return [source_dir / shard["path"] for shard in shards]


def _replace_none_ids(ids_row: Sequence[Any], replacement: int = 0) -> Tuple[List[int], List[bool]]:
    out_ids: List[int] = []
    valid: List[bool] = []
    for token_id in ids_row:
        if token_id is None:
            out_ids.append(replacement)
            valid.append(False)
        else:
            out_ids.append(int(token_id))
            valid.append(True)
    return out_ids, valid


def _replace_none_logprobs(lp_row: Sequence[Any]) -> List[float]:
    out: List[float] = []
    neg_inf = -1.0e30  # avoid actual -inf in stored arrays; softmax treats this as ~0
    for value in lp_row:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            out.append(neg_inf)
        else:
            out.append(float(value))
    return out


def iter_rows(shards: Sequence[Path], max_rows: int = None) -> Iterator[Dict[str, Any]]:
    emitted = 0
    for path in shards:
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            yield row
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                return


def tokenize_full_chat(tokenizer, text: str) -> List[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def preflight_rows(
    tokenizer,
    rows: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> None:
    expected_fp = manifest.get("chat_template_fingerprint", "")
    actual_fp = chat_template_fingerprint(tokenizer)
    if expected_fp and actual_fp and expected_fp != actual_fp:
        raise RuntimeError(
            "Chat-template fingerprint mismatch between student tokenizer and teacher manifest. "
            f"student={actual_fp} manifest={expected_fp}"
        )

    for row in rows:
        assistant_start = int(row["assistant_token_start"])
        gold_ids = [int(token_id) for token_id in row["assistant_token_ids"]]
        assert len(gold_ids) == int(row["assistant_token_count"])

        student_ids = tokenize_full_chat(tokenizer, row["full_chat_text"])
        if len(student_ids) < assistant_start + len(gold_ids):
            raise RuntimeError(
                f"example_index={row['example_index']}: student tokenization is shorter than "
                f"assistant_start+len(gold_ids): {len(student_ids)} < {assistant_start + len(gold_ids)}"
            )

        span = student_ids[assistant_start : assistant_start + len(gold_ids)]
        if span != gold_ids:
            raise RuntimeError(
                f"example_index={row['example_index']}: student retokenization of assistant span "
                f"does not match teacher-stored assistant_token_ids. "
                f"First divergence: student={span[:10]}... vs gold={gold_ids[:10]}..."
            )


def row_to_kd_record(tokenizer, row: Dict[str, Any]) -> Dict[str, Any]:
    assistant_start = int(row["assistant_token_start"])
    assistant_len = int(row["assistant_token_count"])
    gold_ids = [int(token_id) for token_id in row["assistant_token_ids"]]
    assert len(gold_ids) == assistant_len
    if assistant_start < 1:
        raise RuntimeError(
            f"example_index={row['example_index']}: assistant_start must be >= 1 so that "
            f"logits[start-1] predicts gold_ids[0]. got {assistant_start}."
        )

    input_ids = tokenize_full_chat(tokenizer, row["full_chat_text"])
    span = input_ids[assistant_start : assistant_start + assistant_len]
    if span != gold_ids:
        raise RuntimeError(
            f"example_index={row['example_index']}: student retokenization mismatch on assistant span."
        )

    topk_ids_raw: List[Sequence[Any]] = row["topk_token_ids"]
    topk_lp_raw: List[Sequence[Any]] = row["topk_logprobs"]
    if len(topk_ids_raw) != assistant_len or len(topk_lp_raw) != assistant_len:
        raise RuntimeError(
            f"example_index={row['example_index']}: topk length does not match assistant_len."
        )

    topk_ids: List[List[int]] = []
    topk_valid_mask: List[List[bool]] = []
    topk_logprobs: List[List[float]] = []
    for ids_row, lp_row in zip(topk_ids_raw, topk_lp_raw):
        ids_clean, valid = _replace_none_ids(ids_row)
        topk_ids.append(ids_clean)
        topk_valid_mask.append(valid)
        topk_logprobs.append(_replace_none_logprobs(lp_row))

    return {
        "input_ids": input_ids,
        "assistant_start": assistant_start,
        "assistant_len": assistant_len,
        "gold_ids": gold_ids,
        "topk_ids": topk_ids,
        "topk_logprobs": topk_logprobs,
        "topk_valid_mask": topk_valid_mask,
        "hard_label_in_topk": [bool(x) for x in row["hard_label_in_topk"]],
        "example_index": int(row["example_index"]),
    }


def build_features() -> Features:
    return Features({
        "input_ids": HFSequence(Value("int32")),
        "assistant_start": Value("int32"),
        "assistant_len": Value("int32"),
        "gold_ids": HFSequence(Value("int32")),
        "topk_ids": HFSequence(HFSequence(Value("int32"))),
        "topk_logprobs": HFSequence(HFSequence(Value("float32"))),
        "topk_valid_mask": HFSequence(HFSequence(Value("bool"))),
        "hard_label_in_topk": HFSequence(Value("bool")),
        "example_index": Value("int64"),
    })


def build_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    args = parse_args()

    if args.student_model:
        student_model = args.student_model
    else:
        student_model = read_student_model_name(args.train_config)

    manifest = load_manifest(args.source_dir)
    shards = shard_paths(args.source_dir, manifest)
    if not shards:
        raise RuntimeError(f"No shards listed in manifest at {args.source_dir}")

    tokenizer = build_tokenizer(student_model)

    # Preflight on the first few rows of the first shard.
    first_shard = pq.read_table(shards[0]).to_pylist()[: max(1, args.preflight_samples)]
    preflight_rows(tokenizer, first_shard, manifest)
    print(f"[preflight] {len(first_shard)} rows OK with student={student_model}", flush=True)

    total_rows = sum(int(s["row_count"]) for s in manifest.get("shards", []))
    if args.max_rows is not None:
        total_rows = min(total_rows, args.max_rows)
    print(f"[build] converting {total_rows} rows from {len(shards)} shards", flush=True)

    def generator() -> Iterator[Dict[str, Any]]:
        seen = 0
        progress_every = max(256, total_rows // 40) if total_rows else 256
        for row in iter_rows(shards, max_rows=args.max_rows):
            yield row_to_kd_record(tokenizer, row)
            seen += 1
            if seen % progress_every == 0:
                print(f"[build]   {seen}/{total_rows}", flush=True)

    dataset = Dataset.from_generator(
        generator,
        features=build_features(),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))

    meta = {
        "source_manifest": str(args.source_dir / "manifest.json"),
        "student_model": student_model,
        "student_chat_template_fingerprint": chat_template_fingerprint(tokenizer),
        "teacher_chat_template_fingerprint": manifest.get("chat_template_fingerprint"),
        "teacher_model": manifest.get("teacher_model"),
        "top_k": manifest.get("top_k"),
        "num_rows": len(dataset),
    }
    with (args.output_dir / "kd_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[done] wrote {len(dataset)} rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
