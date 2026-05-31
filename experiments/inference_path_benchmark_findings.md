# Inference Path Benchmark Findings

Source reports (run ids; the raw JSON reports are not included in this public
release because they embed generated document text):
- `20260412_121922`
- `20260412_122930`
- `20260412_124212`

## Bottom Line

Use direct `model.generate(...)` with the in-memory Unsloth/Transformers model, not the current merge-to-vLLM-per-checkpoint workflow.

Recommended settings:
- Method: direct generation on the training model after switching to `FastLanguageModel.for_inference(model)`
- Padding: `left`
- Cache: `use_cache=True`
- Batch size for the real 101-doc eval set: start with `8`
- Acceptable alternative: `16`
- Avoid: `right` padding
- Avoid: `use_cache=False`

Why:
- On the final 101-doc benchmark, `bs=8` was the fastest good configuration at `32.68s` total generation time, `3.09 examples/s`.
- `bs=16` was slightly slower at `33.63s`.
- `bs>=32` got slower, not faster.
- `use_cache=False` was catastrophic: `271.32s`, about `8.3x` slower than cached `bs=8`.
- Right-padding materially harmed output quality and parseability.

## Direct Recommendation

For checkpoint evaluation, the best tested path is:

1. Train normally with Unsloth + LoRA.
2. At each checkpoint/eval event, pause training.
3. Run a normal eval-loss pass if you still want that metric.
4. Switch the model to inference mode with `FastLanguageModel.for_inference(model)`.
5. Run `model.generate(...)` directly on the eval prompts.
6. Score generations with the existing JSON/key-value metric.
7. Log results to W&B.
8. Resume training.

This removes:
- checkpoint merge to merged 16-bit weights
- separate model reload
- vLLM engine startup
- repeated shutdown/startup cycles
- server lifecycle overhead

## Final Run: 101-Doc Combined Eval

Report: run `20260412_124212` (raw JSON not included in this public release).

Config:
- Train source fallback: `eval/eval_invoice.json`
- Eval sources: `eval_invoice.json`, `eval_vehicle.json`, `eval_order.json`
- Eval size: `101`
- Tiny train steps: `8`
- Model: `unsloth/Qwen2.5-0.5B-Instruct`

Measured timings:

| Step | Config | Time (s) | Throughput |
|---|---|---:|---:|
| Tiny train | 8 steps, 32 train examples | 6.71 | n/a |
| Eval loss | teacher-forced, `bs=8` | 2.54 | 39.72 ex/s |
| Generation eval | sequential, cache on | 98.83 | 1.02 ex/s |
| Generation eval | left pad, cache on, `bs=8` | 32.68 | 3.09 ex/s |
| Generation eval | left pad, cache on, `bs=16` | 33.63 | 3.00 ex/s |
| Generation eval | left pad, cache on, `bs=32` | 35.22 | 2.87 ex/s |
| Generation eval | left pad, cache on, `bs=64` | 38.35 | 2.63 ex/s |
| Generation eval | left pad, cache on, `bs=96` | 44.53 | 2.27 ex/s |
| Generation eval | left pad, cache off, `bs=8` | 271.32 | 0.37 ex/s |
| Generation eval | right pad, cache on, `bs=8` | 32.78 | 3.08 ex/s |

Best practical configuration from the final run:
- `left padding`
- `use_cache=True`
- `batch_size=8`

Best total inline evaluation cost on 101 docs:
- Eval loss: `2.54s`
- Generation eval: `32.68s`
- Combined eval pause: about `35.23s`

## Quality / Parseability Findings

These benchmarks were not designed to optimize model quality. They were designed to validate the evaluation method. The tiny training run used only 32 fallback examples and 8 train steps, so accuracy numbers are only a sanity check, not a model-selection metric.

That said, several quality findings were still clear:

- Left-padding produced sane generations after the batched decode fix.
- Right-padding produced visibly corrupted outputs and more parse failures.
- Cached batched generation and cached sequential generation produced broadly similar kinds of outputs.
- `FastLanguageModel.for_inference(model)` did not materially change throughput in these runs.

Final run quality summary:

| Method | Accuracy (%) | Parse Errors | Valid JSON |
|---|---:|---:|---:|
| Sequential, cache on | 26.04 | 5 | 96 |
| Left pad, cache on, `bs=8` | 25.70 | 6 | 95 |
| Left pad, cache on, `bs=16` | 24.78 | 7 | 94 |
| Left pad, cache on, `bs=32` | 26.22 | 6 | 95 |
| Left pad, cache on, `bs=64` | 23.73 | 6 | 95 |
| Left pad, cache on, `bs=96` | 25.73 | 5 | 96 |
| Left pad, cache off, `bs=8` | 24.83 | 6 | 95 |
| Right pad, cache on, `bs=8` | 23.96 | 12 | 89 |

Interpretation:
- The small accuracy differences across sane left-padded batch sizes were not large enough to justify slower configs.
- `bs=8` is the best tradeoff.
- `bs=16` is acceptable if you want slightly fewer launch cycles.
- `bs>=32` offered no speed benefit on the real 101-doc workload.

## Why Huge Batches Did Not Help

The 96 GB GPU had plenty of headroom, but throughput still peaked around `bs=8`.

Likely reasons:
- The eval set is only 101 documents.
- Prompts are long and heterogeneous, so larger batches increase padding waste.
- Once batches get large enough, generation is dominated by long-sequence decode cost rather than launch overhead.
- For this workload, a “bigger batch” is not automatically better.

Practical conclusion:
- The GPU is comfortable, but the workload does not benefit from extreme batch sizes.
- The correct question is not “what is the largest batch that fits?” but “what batch gives the best wall-clock for 101 long prompts?”
- Based on these results, that answer is `8`, with `16` as a close second.

## Cache Finding

This was the clearest result in the entire experiment.

Final run:
- Cached left-pad `bs=8`: `32.68s`
- No-cache left-pad `bs=8`: `271.32s`

That means disabling KV cache made evaluation about `8.3x` slower.

Conclusion:
- The earlier pain during direct generation was very plausibly caused by running generation with cache disabled.
- In the real evaluator, `use_cache=True` should be treated as mandatory.

## Padding Finding

Right-padding was consistently harmful.

Symptoms in the reports:
- More parse errors
- More malformed or irrelevant text
- Worse accuracy

Final run:
- Left pad `bs=8`: `6` parse errors, `95` valid JSON
- Right pad `bs=8`: `12` parse errors, `89` valid JSON

Conclusion:
- For Qwen / decoder-only batched generation, use `tokenizer.padding_side = "left"`.
- Right-padding should not be used for the generation evaluator.

## `for_inference()` Finding

The benchmarks did not show a meaningful speedup from `FastLanguageModel.for_inference(model)` on this specific workload.

Examples:
- Run `20260412_122930`
- Raw batch left-pad cached `bs=4`: `16.586s`
- Fast inference left-pad cached `bs=4`: `16.573s`

Conclusion:
- `for_inference()` is still fine to use.
- It does not appear to be the main lever here.
- The main levers are cache and padding, then batch size.

## Three Runs Summary

### Run 1: `20260412_121922`

Purpose:
- Early validation run before the batched decode bug was fixed.

Useful findings that still stand:
- Cache matters a lot.
- Right-padding is bad.
- Sequential generation is much slower than batched generation.

Key numbers:
- Tiny train: `9.22s`
- Eval loss: `0.98s`
- Cached batch left-pad: `8.65s`
- No-cache batch left-pad: `54.98s`
- Right-pad batch cached: `9.06s`

Important caveat:
- This run used `eval_order` as train fallback and had the old batched decode issue, so it should not be used for final quality judgments.

### Run 2: `20260412_122930`

Purpose:
- Fixed batched decode
- Invoice-only fallback train + eval
- Batch sweep on a 32-doc eval

Key findings:
- `bs=8` and `bs=16` were best
- `bs>=32` stopped helping
- Cache off remained very slow

Key numbers:
- Tiny train: `6.72s`
- Eval loss: `0.82s`
- Cached batch left-pad `bs=8`: `13.52s`
- Cached batch left-pad `bs=16`: `13.56s`
- Cached batch left-pad `bs=32`: `15.02s`
- No-cache batch left-pad `bs=4`: `89.21s`

### Run 3: `20260412_124212`

Purpose:
- Final combined eval on all 101 docs
- Most realistic run

This is the run that should drive the implementation decision.

Best config:
- Left pad
- Cache on
- `bs=8`

Measured pause per evaluation event on 101 docs:
- Eval loss: `2.54s`
- Generation eval: `32.68s`
- Total: `35.23s`

## Inline Eval vs Separate vLLM Pipeline

Direct measurement available:
- Inline direct generation on the in-memory model: about `35.23s` total for loss + generation over 101 docs at the recommended settings.

Not directly measured in these reports:
- Full external vLLM workflow with checkpoint detection, LoRA merge, engine startup, shutdown, and next-checkpoint reload.

Inference from the benchmark and current code path:
- Inline direct generation is very likely faster end-to-end for this workload than the current vLLM watcher approach.
- The benchmark already shows that direct generation on the in-memory model only needs about `32.68s` for the actual generation work over 101 docs.
- The current watcher path adds extra steps that inline eval does not need:
  - save checkpoint to disk
  - load checkpoint into a second process
  - merge LoRA into base model
  - start vLLM engine
  - run inference
  - tear down engine
  - repeat for next checkpoint

Conclusion:
- If your main goal is faster checkpoint scoring and simpler orchestration, inline direct generation wins.
- If your main goal is minimizing cost by shutting down the expensive machine and running eval later on a cheaper GPU, a separate pipeline can still make sense financially, but it is unlikely to win on wall-clock speed.
- For day-to-day training iteration speed, the benchmark supports keeping eval inside the training machine and skipping the vLLM merge/start/stop loop.

## Will 101-Doc Eval Slow Training Significantly?

Measured inline pause at recommended settings:
- About `35.23s` total per evaluation event on 101 docs

Whether that is “significant” depends on checkpoint frequency:
- If checkpoints/evals happen every few seconds or every tiny number of steps, yes, that pause is significant.
- If checkpoints/evals happen every moderate chunk of training, a ~35 second pause is usually very reasonable compared with the complexity and overhead of the current external vLLM route.

Important context:
- The tiny benchmark training phase was only `8` steps, so its `6.71s` train time is not representative of a real checkpoint interval.
- The right comparison is not “35s vs 6.7s benchmark training,” because your real training between checkpoints will be much longer than that.

Practical recommendation:
- If you keep checkpoint eval inside training, do not run it too frequently.
- A reasonable starting policy is to evaluate every checkpoint save interval you already care about, but avoid extremely short intervals.

## Additional Findings

- `trainer_outputs` being empty in these benchmark runs is expected because the benchmark uses `save_strategy="no"`.
- The benchmark successfully validated the overall loop shape:
  - train
  - standard eval-loss pass
  - generation-based eval
- The model quality in these reports is not a verdict on the final training recipe. The tiny training subset and very small train step count were chosen for speed, not task performance.
- The sample generations show that the model is producing real JSON-like outputs under sane configs, so direct generation is viable as the basis for your custom metric.

## Recommended Implementation Settings

For the real evaluator:

- `FastLanguageModel.for_inference(model)`
- `model.eval()`
- `model.config.use_cache = True`
- `tokenizer.padding_side = "left"`
- `tokenizer.pad_token = tokenizer.eos_token` if needed
- `do_sample=False`
- `batch_size=8` to start
- Try `16` only if you want to retest on your exact production eval set and prompt lengths

Avoid:
- `padding_side="right"`
- `use_cache=False`
- very large batch sizes just because VRAM is available
- merge-to-vLLM-per-checkpoint unless you explicitly optimize for offloading eval to a different cheaper machine

## Final Recommendation

Move away from the current per-checkpoint vLLM merge/start/stop path and use direct cached left-padded generation from the in-memory training model.

Best tested config:
- Method: direct `generate()`
- Padding: left
- Cache: on
- Batch size: `8`

Measured inline eval cost on the full 101-doc benchmark:
- Eval loss: `2.54s`
- Generation eval: `32.68s`
- Total pause: `35.23s`

Based on the benchmark data, this is the best balance of speed, correctness, and implementation simplicity.

## Implementation Notes

### Interpreting the 35-Second Pause

On the full 101-doc benchmark, the best tested inline evaluation pause was about `35.23s` total:
- Eval loss: `2.54s`
- Generation scoring: `32.68s`

So if evaluation runs every 50 steps, the added wall time is approximately:

`35.23 seconds * number_of_eval_stops`

Example:
- 1000 training steps
- eval every 50 steps
- about 20 eval stops
- total added time about `704.6s`, around `11.7 minutes`

This is the direct cost of inline evaluation at the recommended settings, plus a bit of extra Python/W&B overhead.

### What `model.generate()` Returns

By default, `model.generate()` returns generated token ids, one row per prompt in the same batch order as the input prompts.

That means the implementation can keep each original example alongside its prompt and then zip results back in order:
- prompt/example list in
- generated token rows out
- decode per row
- pair each decoded generation with its original example

This is enough to keep track of:
- prompt
- expected JSON
- generated JSON
- parsed JSON
- sample id
- source corpus (`order`, `vehicle`, `invoice`)
- per-sample score

For richer metadata, Transformers also supports:
- `return_dict_in_generate=True`
- `output_scores=True`

Useful W&B logging structure:
- corpus-level metrics such as `acc_order`, `acc_vehicle`, `acc_invoice`
- total accuracy across all eval docs
- a `wandb.Table` with columns like:
  - `corpus`
  - `sample_id`
  - `source_file`
  - `expected_json`
  - `generated_json`
  - `parsed_ok`
  - `correct`
  - `total`
  - `accuracy`

### `FastLanguageModel.for_inference(model)` and Resuming Training

The benchmark and public Unsloth usage patterns support treating `FastLanguageModel.for_inference(model)` as an in-memory inference-mode switch, not as a LoRA merge-to-base operation like `save_pretrained_merged(...)`.

Practical interpretation:
- it does not appear to write merged weights to disk
- it does not appear to perform the same heavy merge step as the current vLLM watcher flow
- it is intended to make the current in-memory model faster for generation

Safe recommended pattern:

```python
FastLanguageModel.for_inference(model)
model.eval()
model.config.use_cache = True
# run generation + scoring

FastLanguageModel.for_training(model)
model.train()
model.config.use_cache = False
```

Recommendation:
- do not call `for_inference(model)` and then immediately continue training without switching back
- explicitly call `for_training(model)` before resuming the trainer loop

### Padding-Side Switching

Padding behavior should be phase-specific:
- For generation evaluation on a decoder-only model: use `left` padding
- For trainer-style teacher-forced loss evaluation: right padding is fine

So the callback/eval loop should likely:
- switch tokenizer to `left` padding for generation
- switch it back afterward if the training/eval stack expects right padding

This keeps the generation evaluator correct without interfering with the normal loss-eval path.
