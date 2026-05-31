# Qwen2.5 SFT + Knowledge Distillation for Document KIE

Training stack for fine-tuning a small **Qwen2.5** student model to do
**key-information extraction** (KIE) from OCR'd documents — emitting structured
JSON. It covers two training recipes that share one config-driven, resumable
trainer with **inline GT-accuracy evaluation**:

1. **SFT** (`train.py`) — supervised fine-tuning with LoRA (Unsloth), plus a
   data-volume sweep.
2. **Knowledge distillation** (`train_kd.py`) — the same student trained against
   a frozen **Qwen2.5-14B** teacher's **top-K soft labels**, mixing a hard
   cross-entropy term with a temperature-scaled soft KL term over the assistant
   span.

> **Public release notice.** This is the public, sanitized version of a private
> project. It contains **no client data or proprietary entities**. The training
> and evaluation corpora (`data/`, `eval/`) are **tiny synthetic placeholders**
> with generic, invented entities (`ACME RETAIL`, `Sample Motors`, …) that match
> the real data's JSON schema so the code runs and the format is visible. The
> real OCR corpora, the teacher soft-label parquet shards
> (`KL_div_prep/artifacts/`), per-step generation samples, run outputs, and the
> trained weights are **not included**. The corpus names (`order`, `vehicle`,
> `invoice`) are generic placeholders.

## Documentation map

| Doc | What it covers |
|---|---|
| `README_KD.md` | KD quickstart, the experiment matrix, every config knob, and how to read the metrics. |
| `challenges.md` | The distillation loss math, the **vocabulary-mismatch bug** (teacher 152,064 vs student 151,936 vocab) and its fix, the silent-truncation bug, and why KD is so VRAM-hungry. |
| `notes/kl_vs_ce.md` | Why this project optimizes cross-entropy over the top-K teacher distribution (and why that is gradient-identical to KL). |
| `goal.md` | The original spec for the inline GT-accuracy evaluator. |
| `KL_div_prep/README.md` | How the teacher soft labels are gathered and what the parquet schema is. |
| `experiments/inference_path_benchmark_findings.md` | Why inline `generate()` beats per-checkpoint vLLM merge for eval. |

## Repository layout

```
train.py / train_config.toml            # SFT entrypoint + config
train_kd.py / train_config_kd.toml      # KD entrypoint + default config
configs/kd/*.toml                       # KD experiment matrix (alpha / temperature sweep)
kd_trainer.py / kd_collator.py          # KD mixed-loss trainer + padded collator
KL_div_prep/prepare_soft_labels.py      # offline: teacher -> top-K logprob parquet shards
KL_div_prep/build_kd_dataset.py         # parquet -> tokenized Arrow dataset (+ vocab-align checks)
inline_eval.py / accuracy_eval.py       # inline generation eval + GT-accuracy scoring
json_utils.py                           # tolerant JSON repair / key normalization for scoring
scripts/make_volume_subsets.py          # stratified nested subsets for the SFT volume sweep
scripts/plot_inline_eval_gt_acc_compare.py  # plot GT-accuracy across runs
experiments/                            # cached-teacher-distill notebooks + eval-path benchmark
tests/                                  # shape/parity/config unit tests
data/ , eval/                           # synthetic placeholders (see notice above)
```

## Quick start — SFT

1. Edit `train_config.toml` (`paths.train_data`, batch sizes, LoRA rank, W&B).
2. Run:

```bash
bash scripts/reinstall_env.sh
wandb login            # optional; leave [wandb].project empty to disable
python train.py --config train_config.toml
```

## Quick start — KD

```bash
# 1. (offline, needs the 14B teacher + your real data) gather top-K soft labels
python KL_div_prep/prepare_soft_labels.py
# 2. build the tokenized KD dataset (re-tokenizes with the student + vocab checks)
python KL_div_prep/build_kd_dataset.py
# 3. train a matrix variant
python train_kd.py --config configs/kd/kd_40_60.toml
```

See `README_KD.md` for the full matrix (`kd_baseline`, `kd_40_60`, `kd_40_60_T2`,
`kd_70_30`) and metric interpretation.

## Resuming after interruption

Designed for interruptible instances. Keep `paths.resume_from_checkpoint = "auto"`,
reuse the same `paths.output_dir`, and re-run the same command — training restarts
from the latest `checkpoint-*`, reuses the persisted run name, and resumes the
same W&B run id.

## Config notes

- `training.save_steps` must equal `training.eval_steps`.
- `eval.corpora` controls the per-corpus GT-accuracy runs.
- If `wandb.run_name` is empty, a name is generated automatically.
- For KD, `[training].packing` must stay `false` (packing breaks the
  `assistant_start` alignment the soft labels depend on).

## Inline-eval outputs (generated at runtime, not committed)

Each eval step writes `inline_eval/history.csv`, `inline_eval/step_<N>_summary.json`
(aggregate metrics), and `inline_eval/step_<N>_samples.jsonl` (per-sample
generations). `scripts/plot_inline_eval_gt_acc_compare.py` turns the `history.csv`
files from several runs into per-corpus GT-accuracy comparison plots. These output
directories are produced by your own runs and are git-ignored.
