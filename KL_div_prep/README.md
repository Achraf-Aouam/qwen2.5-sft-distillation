# KL Soft-Label Prep

This folder prepares a teacher-labeled soft-target dataset from the same SFT training set used by `train.py`.

The current pipeline targets:

- source dataset: `data/data_04_12.json`
- teacher model: `Qwen/Qwen2.5-14B-Instruct`
- backend: `vLLM`
- label scope: assistant-supervised tokens only
- output format: sharded Parquet + `manifest.json`

## What It Stores

For each training example, the prep step writes one row containing:

- the original hard-label fields: `instruction`, `input`, `output`
- the rendered full chat text
- assistant-token span metadata
- hard-label token ids/text for supervised tokens
- top-10 teacher token ids/text/logprobs at each supervised position
- whether the hard label is present in the top-10
- teacher/model/template metadata

The raw OCR document content inside `input` is never modified.

## Prerequisites

`vllm` and `pyarrow` are installed by `scripts/reinstall_env.sh` alongside the training stack. If that was already run, skip install.

## Hardware Expectations

- Qwen2.5-14B in bf16 fits comfortably on a single 96GB GPU with room for high concurrency.
- For a safe first run, start with `--max-examples 32` to confirm the model loads and shards write, then launch the full dataset.
- Multi-GPU: increase `--tensor-parallel-size`.

## Usage

Default run (14B teacher, in-process vLLM, tuned for a single 96GB GPU):

```bash
python KL_div_prep/prepare_soft_labels.py
```

Recommended explicit run:

```bash
python KL_div_prep/prepare_soft_labels.py \
  --train-data data/data_04_12.json \
  --teacher-model Qwen/Qwen2.5-14B-Instruct \
  --output-dir KL_div_prep/artifacts/qwen2.5-14b-top10 \
  --top-k 10 \
  --batch-size 64 \
  --shard-size 256 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 \
  --dtype bfloat16
```

Resume an interrupted prep run (skips shards already recorded in `manifest.json`):

```bash
python KL_div_prep/prepare_soft_labels.py --resume
```

Rebuild from scratch:

```bash
python KL_div_prep/prepare_soft_labels.py --overwrite
```

Smoke run on a subset:

```bash
python KL_div_prep/prepare_soft_labels.py --max-examples 32 --overwrite
```

## Output Layout

The default artifact directory is:

```text
KL_div_prep/artifacts/qwen2.5-14b-top10/
├── manifest.json
└── shards/
    ├── shard-00000.parquet
    ├── shard-00001.parquet
    └── ...
```

`manifest.json` records:

- source dataset path + SHA256
- teacher model
- top-k
- batch/shard settings
- chat-template fingerprint
- completed shard metadata

## Parquet Schema

Each row corresponds to one original training example and contains:

- `example_index`
- `source_path`
- `teacher_model`
- `top_k`
- `instruction`
- `input`
- `output`
- `full_chat_text`
- `assistant_token_start`
- `assistant_token_count`
- `assistant_token_ids`
- `assistant_token_text`
- `topk_token_ids`
- `topk_token_text`
- `topk_logprobs`
- `hard_label_in_topk`
- `chat_template_fingerprint`
- `source_dataset_sha256`

Conventions:

- `assistant_token_ids`, `assistant_token_text`, and `hard_label_in_topk` all have length `N`
- `topk_token_ids`, `topk_token_text`, and `topk_logprobs` all have shape `N x 10`
- missing top-k slots are padded with null ids/logprobs and empty token text
- the hard-label token is stored separately even when it is absent from the teacher top-10

## Render Compatibility Check

At startup, the script loads:

- the teacher tokenizer
- the current training tokenizer from `train_config.toml`

It compares a small sample of rendered chat texts and prompt-only texts. If they diverge, the script fails fast instead of silently producing misaligned soft labels.

## Notes

- This pipeline prepares data only. It does not change `train.py` and does not consume the soft labels yet.
- The teacher tokenizer/chat template is the source of truth for the saved artifact.
- Assistant supervision is defined as the token suffix after the prompt-only chat template produced with `add_generation_prompt=True`.
