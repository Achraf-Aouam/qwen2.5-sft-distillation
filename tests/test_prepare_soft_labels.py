import json
import tempfile
import unittest
from pathlib import Path

from KL_div_prep.prepare_soft_labels import (
    PromptLogprobResult,
    ShardSpec,
    TokenCandidate,
    build_soft_label_record,
    chat_template_fingerprint,
    normalize_prompt_logprob_positions,
    plan_shards,
    prepare_soft_label_dataset,
    render_example,
)


class FakeTokenizer:
    chat_template = "fake-qwen-template"
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        chunks = [f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages]
        if add_generation_prompt:
            chunks.append("<|im_start|>assistant\n")
        text = "".join(chunks)
        if tokenize:
            return self(text, add_special_tokens=False)["input_ids"]
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeBackend:
    def collect_prompt_logprobs(self, texts, top_k):
        results = []
        for text in texts:
            token_ids = [ord(char) for char in text]
            position_candidates = []
            for token_id in token_ids:
                position_candidates.append(
                    [
                        TokenCandidate(token_id=token_id, token_text=chr(token_id), logprob=-0.01),
                        TokenCandidate(token_id=token_id + 1, token_text=chr(token_id + 1), logprob=-0.20),
                    ]
                )
            results.append(PromptLogprobResult(prompt_token_ids=token_ids, prompt_logprobs=position_candidates))
        return results


class FakeShardWriter:
    suffix = ".fake"

    def write_records(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(records), ensure_ascii=False), encoding="utf-8")


class PrepareSoftLabelsTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.example = {
            "instruction": "System instruction",
            "input": "User payload",
            "output": '{"date":"03/01/2025","recipient":"ACME RETAIL"}',
        }

    def test_render_example_extracts_assistant_suffix(self):
        rendered = render_example(self.tokenizer, 0, self.example)
        self.assertGreater(rendered.assistant_token_start, 0)
        self.assertEqual(rendered.full_token_ids[: rendered.assistant_token_start], rendered.prompt_token_ids)
        self.assertEqual(
            rendered.full_token_ids[rendered.assistant_token_start :],
            rendered.assistant_token_ids,
        )

    def test_normalize_prompt_logprob_positions_prepends_missing_first_entry(self):
        token_ids = [1, 2, 3]
        positions = [[TokenCandidate(2, "b", -0.1)], [TokenCandidate(3, "c", -0.1)]]
        normalized = normalize_prompt_logprob_positions(token_ids, positions)
        self.assertIsNone(normalized[0])
        self.assertEqual(len(normalized), 3)

    def test_build_soft_label_record_keeps_canonical_shapes(self):
        rendered = render_example(self.tokenizer, 0, self.example)
        prompt_logprobs = [
            [TokenCandidate(token_id=token_id, token_text=chr(token_id), logprob=-0.1)]
            for token_id in rendered.full_token_ids
        ]
        record = build_soft_label_record(
            rendered,
            PromptLogprobResult(prompt_token_ids=rendered.full_token_ids, prompt_logprobs=prompt_logprobs),
            source_path=Path("data/data_04_12.json"),
            source_dataset_sha256="abc123",
            teacher_model="Qwen/Qwen2.5-14B-Instruct",
            top_k=10,
            tokenizer=self.tokenizer,
            template_fingerprint=chat_template_fingerprint(self.tokenizer),
        )
        n_tokens = record["assistant_token_count"]
        self.assertEqual(len(record["assistant_token_ids"]), n_tokens)
        self.assertEqual(len(record["assistant_token_text"]), n_tokens)
        self.assertEqual(len(record["hard_label_in_topk"]), n_tokens)
        self.assertEqual(len(record["topk_token_ids"]), n_tokens)
        self.assertEqual(len(record["topk_token_text"]), n_tokens)
        self.assertEqual(len(record["topk_logprobs"]), n_tokens)
        self.assertTrue(all(len(row) == 10 for row in record["topk_token_ids"]))
        self.assertTrue(all(len(row) == 10 for row in record["topk_token_text"]))
        self.assertTrue(all(len(row) == 10 for row in record["topk_logprobs"]))

    def test_plan_shards_uses_fixed_ranges(self):
        self.assertEqual(
            plan_shards(total_examples=5, shard_size=2),
            [
                ShardSpec(shard_index=0, start=0, end=2),
                ShardSpec(shard_index=1, start=2, end=4),
                ShardSpec(shard_index=2, start=4, end=5),
            ],
        )

    def test_prepare_dataset_resume_skips_completed_shards(self):
        examples = [self.example, self.example, self.example]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_data = root / "train.json"
            train_data.write_text(json.dumps(examples), encoding="utf-8")
            output_dir = root / "artifacts"
            manifest_path = output_dir / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "shards": [
                            {
                                "shard_index": 0,
                                "path": "shards/shard-00000.fake",
                                "example_start": 0,
                                "example_end": 2,
                                "row_count": 2,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            first_shard = output_dir / "shards" / "shard-00000.fake"
            first_shard.parent.mkdir(parents=True, exist_ok=True)
            first_shard.write_text("already-done", encoding="utf-8")

            manifest = prepare_soft_label_dataset(
                examples=examples,
                source_path=train_data,
                output_dir=output_dir,
                teacher_model="Qwen/Qwen2.5-14B-Instruct",
                top_k=10,
                shard_size=2,
                batch_size=2,
                tokenizer=self.tokenizer,
                backend=FakeBackend(),
                writer=FakeShardWriter(),
                resume=True,
            )

            self.assertEqual(first_shard.read_text(encoding="utf-8"), "already-done")
            self.assertTrue((output_dir / "shards" / "shard-00001.fake").exists())
            self.assertEqual(manifest["total_shards"], 2)


if __name__ == "__main__":
    unittest.main()
