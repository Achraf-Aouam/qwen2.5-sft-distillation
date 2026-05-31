# KD training — quickstart, matrix, and interpretation

Knowledge-distillation training on top of the SFT stack. The teacher is a frozen
Qwen2.5-14B-Instruct that was run once offline to produce per-token **top-K
logprobs over the assistant span** (stored under
`KL_div_prep/artifacts/qwen2.5-14b-top10/`). The student is the small Qwen2.5
model we actually ship.

The loss is a standard KD mix:

```
loss = (1 - alpha) * hard_ce   +   alpha * T^2 * soft_ce_topk
```

where `soft_ce_topk` is the cross-entropy between the student's distribution
(temperature-scaled) and the teacher's top-K distribution renormalized over the
K ids it stored. Both terms are averaged over assistant-span tokens only.

## TL;DR

One-time:

```bash
# 1. Build the tokenized KD dataset from the parquet shards (local, ~2–5 min).
python KL_div_prep/build_kd_dataset.py
```

Per experiment (one config → one run):

```bash
# 2. Pick a matrix variant and run.
python train_kd.py --config configs/kd/kd_40_60.toml
```

The run name auto-gets a random 3-char suffix (`kd_40_60-p22`, …) so re-running
the same config doesn't overwrite the previous wandb run.

## Files that matter

| file | role |
|---|---|
| `KL_div_prep/prepare_soft_labels.py` | offline: teacher → parquet shards. (Artifacts are **not** shipped in this public release — regenerate from your own data.) |
| `KL_div_prep/build_kd_dataset.py` | local: parquet → mmap-ready Arrow dataset (with tokenizer preflight). |
| `kd_collator.py` | right-padded batch collator; pads teacher top-K with masked log-probs. |
| `kd_trainer.py` | `KDTrainer(SFTTrainer)` with the mixed loss + wandb metrics. |
| `train_kd.py` | entrypoint. Every knob comes from the config; only `--config` and `--resume-from-checkpoint` are CLI flags. |
| `train_config_kd.toml` | default (alpha=0.4, T=1.0). Duplicate per experiment. |
| `configs/kd/*.toml` | matrix variants (`kd_baseline`, `kd_40_60`, `kd_40_60_T2`, `kd_70_30`). |
| `tests/test_kd_*.py` | shape + parity tests. Run with `python -m unittest tests.test_kd_collator tests.test_kd_trainer_parity tests.test_build_kd_dataset`. |

## Step 1 — build the KD dataset (local)

The parquet shards are teacher-tokenized and carry the character-level chat
text. `build_kd_dataset.py` re-tokenizes that text with the **student**
tokenizer, asserts the assistant span matches the teacher-stored ids, and saves
an Arrow dataset that the trainer mmaps at runtime.

```bash
python KL_div_prep/build_kd_dataset.py
# → KL_div_prep/artifacts/qwen2.5-14b-top10-kd/
```

It fails loudly if:

- the student chat-template fingerprint differs from the one recorded in the
  teacher manifest (indicates tokenizer drift);
- the re-tokenized assistant span disagrees with the teacher-stored
  `assistant_token_ids` on any row;
- any row has `assistant_start == 0` (the gather in `compute_loss` needs
  `logits[start-1]`, which doesn't exist).

Smoke-test it on a handful of rows first:

```bash
python KL_div_prep/build_kd_dataset.py --max-rows 64 --output-dir KL_div_prep/artifacts/_smoke
```

The resulting `qwen2.5-14b-top10-kd/` directory is what the trainer mmaps. It is
git-ignored (large, derived from your data); keep it locally or in a bucket /
git-lfs rather than committing it.

## Step 2 — VM setup

```bash
git clone https://github.com/Achraf-Aouam/qwen2.5-sft-distillation.git
cd qwen2.5-sft-distillation
bash scripts/reinstall_env.sh

# wandb login (not done by reinstall_env.sh). Either:
wandb login                    # interactive
# ...or non-interactive:
export WANDB_API_KEY=xxxxxxxx
```

`train_kd.py` will refuse to start if `[wandb].project` is set but the host has
neither `WANDB_API_KEY` in the environment nor an entry in `~/.netrc`.

## Step 3 — the experiment matrix

Always run from the project root so relative paths in the config resolve
correctly.

```bash
# Sanity control — alpha=0.0 must match a plain-SFT curve shape.
python train_kd.py --config configs/kd/kd_baseline.toml

# Target:
python train_kd.py --config configs/kd/kd_40_60.toml

# Ablations:
python train_kd.py --config configs/kd/kd_40_60_T2.toml     # softer teacher
python train_kd.py --config configs/kd/kd_70_30.toml        # more teacher weight
```

Each run writes checkpoints, `inline_eval/` JSONL, and a `run_state/` folder
under its own `outputs_kd/<run>/`. Ctrl+C is safe — re-running with the same
config resumes from the latest checkpoint (`resume_from_checkpoint = "auto"`)
and reuses the same wandb run id.

## Step 4 — reading the results

### In wandb

Per-step train metrics (logged every `logging_steps`):

| key | what it means |
|---|---|
| `loss` | the actual training loss = `(1-α)·loss_hard + α·T²·loss_soft`. Compare trajectories across α, not absolute values. |
| `loss_hard` | standard next-token CE on the assistant span. **This is the only number that's directly comparable to the plain SFT run.** |
| `loss_soft` | token-mean cross-entropy vs. the renormalized teacher top-K (pre-`T²` factor is included in the logged value). |
| `kd_alpha`, `kd_temperature` | constants, sanity-logged to confirm the config took effect. |
| `frac_hard_in_topk` | fraction of assistant tokens whose gold id is among the teacher's top-10. Expect ~0.9+ on well-teacher-labeled data; a drop means the teacher disagrees more than usual with the gold. |
| `kd_tokens_per_batch` | how many supervised tokens went into the loss. Useful if you change `max_seq_length` or batch size and want to keep the effective token count comparable. |

Per-save evaluation (fires at every `save_steps`):

| key | what it means |
|---|---|
| `gt_acc/{order,vehicle,invoice}` | % of JSON fields the student got right on each eval corpus. **This is the metric that matters.** |
| `gt_acc/overall_weighted` | field-weighted mean across corpora. Primary metric for picking a winner. |
| `gt_valid_json/{corpus}` | count of outputs that parse as JSON. A KD run with aggressive α should not regress this vs. baseline. |
| `gt_parse_errors/{corpus}` | inverse of the above. |
| `throughput/{corpus}_examples_per_sec` | eval-only; ignore unless investigating latency. |

### What to look for

- **Baseline vs. SFT**: `kd_baseline` (α=0) should track the existing
  `train.py` curves to within noise. If it doesn't, the KD pipeline is wrong
  (slice indexing, label masking, dataset drift). Investigate before trusting
  the other runs.
- **`kd_40_60` vs. baseline**: expect `gt_acc/overall_weighted` to improve or
  at least match the baseline, with lower variance across eval corpora. A large
  regression on `gt_valid_json` means the soft signal is pulling the student
  toward teacher-preferred-but-JSON-breaking tokens → drop α or raise T.
- **`kd_40_60_T2` vs. `kd_40_60`**: higher T flattens the teacher distribution,
  so more signal bleeds to non-top-1 candidates. Helps when the teacher is
  overconfident; hurts when the top-10 already captures the true mass.
  Compare `loss_soft` trajectories — T2 should have a higher absolute
  `loss_soft` but similar `loss_hard`.
- **`kd_70_30` vs. `kd_40_60`**: more weight on the teacher. If `gt_acc` goes
  up, the teacher is genuinely informative beyond the gold. If `gt_acc` goes
  down while `loss_soft` still trains fine, the teacher is *wrong* often enough
  that over-weighting it hurts.

### On-disk artifacts

Under `outputs_kd/<run>/`:

- `checkpoint-<step>/` — LoRA adapter weights + trainer state.
- `final_adapter/` — last checkpoint, written at the end.
- `inline_eval/history.csv` — flat table of per-save metrics. Easy to
  pandas-load offline. Each row = one `save_step`.
- `inline_eval/step_<N>_samples.jsonl` — every generation from that step, with
  ground truth, parsed output, and per-field match details. Use when wandb
  numbers look weird and you want to read the actual generations.
- `run_state/session.json` — run name + wandb run id. This is what makes
  resume idempotent.
- `run_state/resolved_config.json` — the exact config that was used, after CLI
  overrides. Always inspect this if a run's metrics are surprising.

## Main knobs (in priority order)

All knobs live in the TOML config. Edit, don't CLI-override.

| knob | where | what it does | when to change |
|---|---|---|---|
| `[kd].alpha` | `configs/kd/*.toml` | weight on the soft loss. 0 = pure SFT, 1 = pure KD. | **Main KD knob.** Sweep {0.0, 0.4, 0.7} as the matrix. |
| `[kd].temperature` | same | softmax temperature on *both* teacher and student in the soft term. `T²` multiplier keeps gradient scale roughly constant. | Raise (T=2) if the teacher is overconfident; leave at 1 otherwise. |
| `[training].per_device_train_batch_size` | same | batch size. | **Drop first** if you OOM — the KD forward materializes full `[B, T, V]` logits (unlike plain SFT's fused kernel). On a 96 GB GPU with Qwen2.5-0.5B (vocab ≈152k) and typical T≤2k, B=32 is safe. |
| `[training].gradient_accumulation_steps` | same | compensates for smaller batch size. | Bump when you drop batch size and want the same effective batch. |
| `[training].learning_rate` | same | LR. | Keep identical across matrix runs so KD and SFT curves are comparable. |
| `[training].num_train_epochs` | same | total pass count. | Smoke runs: set a small `max_steps` instead. Production: match the SFT baseline. |
| `[training].save_steps` / `eval_steps` | same | cadence of checkpoint + generation eval. Must be equal. | Lower for tight feedback during hyperparam search; raise for long production runs. |
| `[model].max_seq_length` | same | context length. | Lower aggressively (e.g. 2048) if OOM. Most of our supervised tokens sit well under 2k; the assistant span is small. |
| `[lora].rank` / `[lora].lora_alpha` | same | LoRA capacity. | Keep in lockstep with the SFT baseline for apples-to-apples comparison. |
| `[wandb].run_name` | same | stem of the run name. A random 3-char suffix is always appended. | Set to a descriptive stem per experiment; don't bother for one-offs. |

### Things you probably should NOT touch

- `[kd].dataset_dir` — point at whatever `build_kd_dataset.py` produced.
  Regenerate if the teacher or student tokenizer changes.
- `[training].packing` — must stay `false`. Packing breaks the
  `assistant_start` position and the soft-label alignment.
- `[eval.corpora]` — the callback's API depends on these keys. Add new eval
  sets as additional entries; don't remove.

## Common failure modes

| symptom | likely cause |
|---|---|
| `RuntimeError: Chat-template fingerprint mismatch …` at build time | student tokenizer on this machine drifted vs. the one used to build the parquet. Update `transformers`/`unsloth` or rebuild parquet. |
| `RuntimeError: student retokenization of assistant span does not match …` at build time | teacher and student tokenizers disagree on at least one token. Inspect the offending `example_index`; usually a byte-level edge case in the answer text. |
| `ValueError: assistant_start must be >= 1 …` in the collator | an example has no system/user prefix in its chat rendering. Shouldn't happen with the current Qwen template; investigate the source example. |
| CUDA OOM on the first training step | full `[B, T, V]` logits don't fit. Drop `per_device_train_batch_size` (and/or `max_seq_length`), bump `gradient_accumulation_steps` to compensate. |
| `loss_hard` tracks SFT but `loss_soft` explodes / stays flat across a run | teacher logprobs are miscalibrated or the K=10 window truncates too much mass. Raise T, or rebuild parquet with a larger `--top-k`. |
| Baseline run (`kd_baseline`, α=0) diverges from the plain-SFT curve | the KD pipeline is introducing an asymmetry. Check `loss_hard` equals the SFT loss shape; if not, inspect `kd_tokens_per_batch` vs. SFT's effective token count. |
| `wandb` error about "this host is not logged in" | run `wandb login` on the VM, or set `WANDB_API_KEY`. |
