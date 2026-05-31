# Cached Teacher Distillation Experiment

Files in this folder:

- `baseline_original.ipynb`: a straight copy of your current working notebook.
- `teacher_cached_distill.ipynb`: a new notebook that first caches teacher top-10 token distributions from an OpenAI-compatible API, then trains the 0.5B student offline against both hard and soft labels.
- `distill_utils.py`: helper code used by the new notebook.
- `build_notebooks.py`: small generator used to write the distilled notebook JSON.

Expected flow in the new notebook:

1. Point `DATA_PATH` to your JSON training file.
2. Set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and optionally `TEACHER_MODEL_NAME`.
3. Run the cache cell first. It writes `cache/teacher_top10_train.jsonl` and resumes cleanly if interrupted.
4. Inspect the cached token table for one example.
5. Start training. Training uses:
   - hard loss on the gold `output`
   - soft loss on the cached teacher top-10 distribution
   - a final mixed loss of `(1 - soft_weight) * hard_loss + soft_weight * soft_loss`

For the cleanest token-level soft labels, use a teacher from the same tokenizer family as the student, for example Qwen 2.5 72B with Qwen 2.5 0.5B.
