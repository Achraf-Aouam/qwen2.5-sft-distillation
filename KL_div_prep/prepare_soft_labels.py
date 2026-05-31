"""Prepare assistant-token soft labels from a Qwen teacher via vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

DEFAULT_TRAIN_DATA = Path("data/data_04_12.json")
DEFAULT_TRAIN_CONFIG = Path("train_config.toml")
DEFAULT_TEACHER_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_OUTPUT_DIR = Path("KL_div_prep/artifacts/qwen2.5-14b-top10")
DEFAULT_TOP_K = 10
DEFAULT_SHARD_SIZE = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_RENDER_CHECK_SAMPLES = 3
DEFAULT_GPU_MEMORY_UTILIZATION = 0.80
DEFAULT_MAX_NUM_SEQS = 64
DEFAULT_MAX_NUM_BATCHED_TOKENS = 4096
DEFAULT_DTYPE = "bfloat16"


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


@dataclass
class TokenCandidate:
    token_id: Optional[int]
    token_text: str
    logprob: Optional[float]


@dataclass
class PromptLogprobResult:
    prompt_token_ids: Optional[List[int]]
    prompt_logprobs: List[Optional[List[TokenCandidate]]]


@dataclass
class RenderedExample:
    example_index: int
    example: Dict[str, Any]
    full_chat_text: str
    prompt_text: str
    full_token_ids: List[int]
    prompt_token_ids: List[int]
    assistant_token_start: int
    assistant_token_ids: List[int]


@dataclass
class ShardSpec:
    shard_index: int
    start: int
    end: int


class TeacherBackend(Protocol):
    def collect_prompt_logprobs(self, texts: Sequence[str], top_k: int) -> List[PromptLogprobResult]:
        """Return prompt token ids/logprobs for each rendered chat text."""


class ShardWriter(Protocol):
    suffix: str

    def write_records(self, path: Path, records: Sequence[Dict[str, Any]]) -> None:
        """Persist one shard of records."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare assistant-token soft labels with a Qwen teacher.")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA, help="Training JSON file.")
    parser.add_argument(
        "--teacher-model",
        default=DEFAULT_TEACHER_MODEL,
        help="Teacher model name for vLLM and tokenizer loading.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for parquet shards and manifest.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k prompt logprobs to keep.")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help="Rows per output shard.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Rendered examples per vLLM call.")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help="Tensor parallel size passed to vLLM.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="Maximum model length passed to vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory vLLM may use.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=DEFAULT_MAX_NUM_SEQS,
        help="vLLM max concurrent sequences.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
        help="Caps tokens per forward pass. Lower this if you see OOM during prompt logprobs.",
    )
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        help="vLLM model dtype (bfloat16 recommended on Blackwell/Ampere+).",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help="Config used to discover the current training tokenizer for render compatibility checks.",
    )
    parser.add_argument(
        "--render-check-samples",
        type=int,
        default=DEFAULT_RENDER_CHECK_SAMPLES,
        help="Number of examples used for startup render compatibility checks.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional limit for smoke runs/debugging.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip shards already recorded in the manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory before rebuilding.")
    return parser.parse_args()


def resolve_path(base_dir: Path, raw_path: Path) -> Path:
    if raw_path.is_absolute():
        return raw_path
    return (base_dir / raw_path).resolve()


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_reference_model_name(config_path: Path) -> str:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    return str(config["model"]["model_name"]).strip()


def chat_template_fingerprint(tokenizer) -> str:
    template = getattr(tokenizer, "chat_template", None) or ""
    payload = template if isinstance(template, str) else json.dumps(template, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def decode_single_token(tokenizer, token_id: Optional[int]) -> str:
    if token_id is None:
        return ""
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def tokenize_text(tokenizer, text: str) -> List[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def render_example(tokenizer, example_index: int, example: Dict[str, Any]) -> RenderedExample:
    full_chat_text = format_training_text(tokenizer, example)
    prompt_text = build_generation_prompt(tokenizer, example)
    full_token_ids = tokenize_text(tokenizer, full_chat_text)
    prompt_token_ids = tokenize_text(tokenizer, prompt_text)

    if full_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
        raise ValueError(
            f"Prompt token ids are not a prefix of the full chat text for example {example_index}."
        )

    assistant_token_ids = full_token_ids[len(prompt_token_ids) :]
    if not assistant_token_ids:
        raise ValueError(f"Example {example_index} has no assistant-supervised tokens.")

    return RenderedExample(
        example_index=example_index,
        example=example,
        full_chat_text=full_chat_text,
        prompt_text=prompt_text,
        full_token_ids=full_token_ids,
        prompt_token_ids=prompt_token_ids,
        assistant_token_start=len(prompt_token_ids),
        assistant_token_ids=assistant_token_ids,
    )


def validate_render_compatibility(
    reference_tokenizer,
    teacher_tokenizer,
    examples: Sequence[Dict[str, Any]],
    sample_count: int,
) -> None:
    for index, example in enumerate(examples[:sample_count]):
        reference_full = format_training_text(reference_tokenizer, example)
        teacher_full = format_training_text(teacher_tokenizer, example)
        if reference_full != teacher_full:
            raise RuntimeError(
                f"Full chat render mismatch between training tokenizer and teacher tokenizer on sample {index}."
            )

        reference_prompt = build_generation_prompt(reference_tokenizer, example)
        teacher_prompt = build_generation_prompt(teacher_tokenizer, example)
        if reference_prompt != teacher_prompt:
            raise RuntimeError(
                f"Prompt render mismatch between training tokenizer and teacher tokenizer on sample {index}."
            )


def plan_shards(total_examples: int, shard_size: int) -> List[ShardSpec]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    shards: List[ShardSpec] = []
    shard_index = 0
    for start in range(0, total_examples, shard_size):
        end = min(start + shard_size, total_examples)
        shards.append(ShardSpec(shard_index=shard_index, start=start, end=end))
        shard_index += 1
    return shards


def normalize_prompt_logprob_positions(
    prompt_token_ids: Sequence[int],
    prompt_logprobs: Sequence[Optional[List[TokenCandidate]]],
) -> List[Optional[List[TokenCandidate]]]:
    token_count = len(prompt_token_ids)
    logprob_count = len(prompt_logprobs)
    if logprob_count == token_count:
        return list(prompt_logprobs)
    if logprob_count == token_count - 1:
        return [None] + list(prompt_logprobs)
    raise ValueError(
        f"Unexpected prompt_logprobs length: got {logprob_count} entries for {token_count} prompt tokens."
    )


def pad_candidates(candidates: Sequence[TokenCandidate], top_k: int) -> List[TokenCandidate]:
    trimmed = list(candidates[:top_k])
    while len(trimmed) < top_k:
        trimmed.append(TokenCandidate(token_id=None, token_text="", logprob=None))
    return trimmed


def build_soft_label_record(
    rendered: RenderedExample,
    prompt_result: PromptLogprobResult,
    *,
    source_path: Path,
    source_dataset_sha256: str,
    teacher_model: str,
    top_k: int,
    tokenizer,
    template_fingerprint: str,
) -> Dict[str, Any]:
    prompt_token_ids = list(prompt_result.prompt_token_ids or rendered.full_token_ids)
    if prompt_token_ids != rendered.full_token_ids:
        raise ValueError(
            f"Teacher prompt token ids do not match local tokenizer ids for example {rendered.example_index}."
        )

    aligned_logprobs = normalize_prompt_logprob_positions(prompt_token_ids, prompt_result.prompt_logprobs)
    assistant_positions = range(rendered.assistant_token_start, len(prompt_token_ids))

    assistant_token_text = [decode_single_token(tokenizer, token_id) for token_id in rendered.assistant_token_ids]
    topk_token_ids: List[List[Optional[int]]] = []
    topk_token_text: List[List[str]] = []
    topk_logprobs: List[List[Optional[float]]] = []
    hard_label_in_topk: List[bool] = []

    for relative_index, absolute_index in enumerate(assistant_positions):
        position_candidates = aligned_logprobs[absolute_index]
        if position_candidates is None:
            raise ValueError(
                f"Missing prompt logprobs for assistant token position {absolute_index} "
                f"of example {rendered.example_index}."
            )

        padded = pad_candidates(position_candidates, top_k)
        hard_token_id = rendered.assistant_token_ids[relative_index]
        hard_label_in_topk.append(any(candidate.token_id == hard_token_id for candidate in padded))
        topk_token_ids.append([candidate.token_id for candidate in padded])
        topk_token_text.append([candidate.token_text for candidate in padded])
        topk_logprobs.append([candidate.logprob for candidate in padded])

    return {
        "example_index": rendered.example_index,
        "source_path": str(source_path),
        "teacher_model": teacher_model,
        "top_k": top_k,
        "instruction": rendered.example["instruction"],
        "input": rendered.example["input"],
        "output": rendered.example["output"],
        "full_chat_text": rendered.full_chat_text,
        "assistant_token_start": rendered.assistant_token_start,
        "assistant_token_count": len(rendered.assistant_token_ids),
        "assistant_token_ids": rendered.assistant_token_ids,
        "assistant_token_text": assistant_token_text,
        "topk_token_ids": topk_token_ids,
        "topk_token_text": topk_token_text,
        "topk_logprobs": topk_logprobs,
        "hard_label_in_topk": hard_label_in_topk,
        "chat_template_fingerprint": template_fingerprint,
        "source_dataset_sha256": source_dataset_sha256,
    }


class ParquetShardWriter:
    suffix = ".parquet"

    def write_records(self, path: Path, records: Sequence[Dict[str, Any]]) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(
                "pyarrow is required to write parquet shards. Install it before running this command."
            ) from exc

        table = pa.Table.from_pylist(list(records))
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)


class VLLMTeacherBackend:
    def __init__(
        self,
        *,
        teacher_model: str,
        tensor_parallel_size: int,
        max_model_len: int,
        tokenizer,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
        dtype: str = DEFAULT_DTYPE,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(
                "vLLM is required for teacher prompt logprobs. Install vllm before running this command."
            ) from exc

        self._SamplingParams = SamplingParams
        self._tokenizer = tokenizer
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self._llm = LLM(
            model=teacher_model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            dtype=dtype,
            trust_remote_code=True,
            enable_prefix_caching=True,
        )

    def collect_prompt_logprobs(self, texts: Sequence[str], top_k: int) -> List[PromptLogprobResult]:
        sampling_params = self._SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=1,
            prompt_logprobs=top_k,
        )
        outputs = self._llm.generate(list(texts), sampling_params, use_tqdm=True)
        return [self._convert_request_output(output, top_k=top_k) for output in outputs]

    def _convert_request_output(self, request_output: Any, top_k: int) -> PromptLogprobResult:
        prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
        raw_prompt_logprobs = list(getattr(request_output, "prompt_logprobs", []) or [])
        prompt_logprobs = [
            self._convert_position_candidates(position, top_k=top_k)
            for position in raw_prompt_logprobs
        ]
        return PromptLogprobResult(
            prompt_token_ids=list(prompt_token_ids) if prompt_token_ids is not None else None,
            prompt_logprobs=prompt_logprobs,
        )

    def _convert_position_candidates(
        self,
        raw_position: Any,
        *,
        top_k: int,
    ) -> Optional[List[TokenCandidate]]:
        if raw_position is None:
            return None

        if hasattr(raw_position, "items"):
            raw_items = list(raw_position.items())
        elif isinstance(raw_position, list):
            raw_items = list(enumerate(raw_position))
        else:
            raise TypeError(f"Unsupported prompt logprob container type: {type(raw_position)!r}")

        candidates: List[TokenCandidate] = []
        for raw_key, raw_value in raw_items:
            token_id = self._extract_token_id(raw_key, raw_value)
            logprob = self._extract_attr(raw_value, "logprob")
            token_text = self._extract_attr(raw_value, "decoded_token")
            if token_text is None:
                token_text = decode_single_token(self._tokenizer, token_id)
            candidates.append(TokenCandidate(token_id=token_id, token_text=token_text or "", logprob=logprob))

        candidates.sort(key=lambda item: float("-inf") if item.logprob is None else item.logprob, reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _extract_attr(raw_value: Any, key: str) -> Any:
        if raw_value is None:
            return None
        if isinstance(raw_value, dict):
            return raw_value.get(key)
        return getattr(raw_value, key, None)

    @staticmethod
    def _extract_token_id(raw_key: Any, raw_value: Any) -> Optional[int]:
        if isinstance(raw_key, int):
            return raw_key
        if isinstance(raw_key, str) and raw_key.lstrip("-").isdigit():
            return int(raw_key)
        if isinstance(raw_value, dict) and isinstance(raw_value.get("token_id"), int):
            return int(raw_value["token_id"])
        token_id = getattr(raw_value, "token_id", None)
        if isinstance(token_id, int):
            return token_id
        return None


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {"shards": []}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_manifest(
    *,
    source_path: Path,
    source_dataset_sha256: str,
    teacher_model: str,
    top_k: int,
    shard_size: int,
    batch_size: int,
    tensor_parallel_size: int,
    max_model_len: int,
    total_examples: int,
    template_fingerprint: str,
) -> Dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "source_dataset_sha256": source_dataset_sha256,
        "teacher_model": teacher_model,
        "top_k": top_k,
        "shard_size": shard_size,
        "batch_size": batch_size,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "total_examples": total_examples,
        "chat_template_fingerprint": template_fingerprint,
        "shards": [],
    }


def prepare_soft_label_dataset(
    *,
    examples: Sequence[Dict[str, Any]],
    source_path: Path,
    output_dir: Path,
    teacher_model: str,
    top_k: int,
    shard_size: int,
    batch_size: int,
    tokenizer,
    backend: TeacherBackend,
    writer: ShardWriter,
    overwrite: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if overwrite and resume:
        raise ValueError("Use either overwrite or resume, not both.")

    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    manifest_path = output_dir / "manifest.json"

    source_dataset_sha256 = compute_sha256(source_path)
    template_fingerprint = chat_template_fingerprint(tokenizer)
    manifest = build_manifest(
        source_path=source_path,
        source_dataset_sha256=source_dataset_sha256,
        teacher_model=teacher_model,
        top_k=top_k,
        shard_size=shard_size,
        batch_size=batch_size,
        tensor_parallel_size=getattr(backend, "tensor_parallel_size", None) or 0,
        max_model_len=getattr(backend, "max_model_len", None) or 0,
        total_examples=len(examples),
        template_fingerprint=template_fingerprint,
    )
    if resume:
        existing = load_manifest(manifest_path)
        manifest["shards"] = list(existing.get("shards", []))

    completed = {int(item["shard_index"]) for item in manifest["shards"]}
    shard_specs = plan_shards(len(examples), shard_size)

    for shard in shard_specs:
        shard_path = shards_dir / f"shard-{shard.shard_index:05d}{writer.suffix}"
        if resume and shard.shard_index in completed and shard_path.exists():
            continue

        print(
            f"[shard {shard.shard_index + 1}/{len(shard_specs)}] "
            f"examples {shard.start}-{shard.end}",
            flush=True,
        )
        records: List[Dict[str, Any]] = []
        for batch_start in range(shard.start, shard.end, batch_size):
            batch_end = min(batch_start + batch_size, shard.end)
            print(
                f"  batch examples {batch_start}-{batch_end} (size={batch_end - batch_start})",
                flush=True,
            )
            rendered_batch = [
                render_example(tokenizer, index, examples[index])
                for index in range(batch_start, batch_end)
            ]
            results = backend.collect_prompt_logprobs(
                [rendered.full_chat_text for rendered in rendered_batch],
                top_k=top_k,
            )
            if len(results) != len(rendered_batch):
                raise ValueError(
                    f"Teacher backend returned {len(results)} results for {len(rendered_batch)} prompts."
                )

            for rendered, prompt_result in zip(rendered_batch, results):
                records.append(
                    build_soft_label_record(
                        rendered,
                        prompt_result,
                        source_path=source_path,
                        source_dataset_sha256=source_dataset_sha256,
                        teacher_model=teacher_model,
                        top_k=top_k,
                        tokenizer=tokenizer,
                        template_fingerprint=template_fingerprint,
                    )
                )

        writer.write_records(shard_path, records)
        manifest["shards"] = [
            item for item in manifest["shards"] if int(item["shard_index"]) != shard.shard_index
        ]
        manifest["shards"].append(
            {
                "shard_index": shard.shard_index,
                "path": str(shard_path.relative_to(output_dir)),
                "example_start": shard.start,
                "example_end": shard.end,
                "row_count": len(records),
            }
        )
        manifest["shards"].sort(key=lambda item: int(item["shard_index"]))
        save_manifest(manifest_path, manifest)

    manifest["total_shards"] = len(manifest["shards"])
    save_manifest(manifest_path, manifest)
    return manifest


def build_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    train_data_path = resolve_path(root, args.train_data)
    output_dir = resolve_path(root, args.output_dir)
    train_config_path = resolve_path(root, args.train_config)

    examples = load_json_examples(train_data_path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    teacher_tokenizer = build_tokenizer(args.teacher_model)
    reference_model_name = read_reference_model_name(train_config_path)
    reference_tokenizer = build_tokenizer(reference_model_name)
    validate_render_compatibility(
        reference_tokenizer,
        teacher_tokenizer,
        examples,
        sample_count=args.render_check_samples,
    )

    backend = VLLMTeacherBackend(
        teacher_model=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        tokenizer=teacher_tokenizer,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        dtype=args.dtype,
    )

    manifest = prepare_soft_label_dataset(
        examples=examples,
        source_path=train_data_path,
        output_dir=output_dir,
        teacher_model=args.teacher_model,
        top_k=args.top_k,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        tokenizer=teacher_tokenizer,
        backend=backend,
        writer=ParquetShardWriter(),
        overwrite=args.overwrite,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "total_examples": len(examples),
                "total_shards": manifest.get("total_shards", len(manifest.get("shards", []))),
                "teacher_model": args.teacher_model,
                "top_k": args.top_k,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
