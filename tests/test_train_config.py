import tempfile
import unittest
from pathlib import Path

from train import (
    apply_overrides,
    build_default_run_name,
    effective_run_name,
    load_config,
    load_session_state,
    save_session_state,
)


class TrainConfigTests(unittest.TestCase):
    def test_load_and_resolve_config_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "train_config.toml"
            train_data = root / "train.json"
            eval_dir = root / "eval"
            eval_dir.mkdir()
            train_data.write_text("[]", encoding="utf-8")
            (eval_dir / "a.json").write_text("[]", encoding="utf-8")

            config_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'train_data = "train.json"',
                        'output_dir = "outputs"',
                        'local_log_dir = ""',
                        'resume_from_checkpoint = ""',
                        "",
                        "[model]",
                        'model_name = "unsloth/Qwen2.5-0.5B-Instruct"',
                        "max_seq_length = 4096",
                        "load_in_4bit = false",
                        "",
                        "[lora]",
                        "rank = 16",
                        "lora_alpha = 16",
                        "lora_dropout = 0.0",
                        'bias = "none"',
                        'use_gradient_checkpointing = "unsloth"',
                        "use_rslora = false",
                        'target_modules = ["q_proj"]',
                        "",
                        "[training]",
                        "per_device_train_batch_size = 4",
                        "per_device_eval_batch_size = 2",
                        "gradient_accumulation_steps = 1",
                        "learning_rate = 1e-5",
                        "weight_decay = 0.01",
                        "warmup_steps = 1",
                        "num_train_epochs = 1.0",
                        "max_steps = -1",
                        "logging_steps = 1",
                        "save_steps = 10",
                        "eval_steps = 10",
                        "save_total_limit = 1",
                        "dataset_num_proc = 1",
                        "packing = false",
                        "seed = 3407",
                        'optim = "adamw_8bit"',
                        'lr_scheduler_type = "linear"',
                        "",
                        "[eval]",
                        "gen_batch_size = 8",
                        "max_new_tokens = 128",
                        "",
                        "[eval.corpora]",
                        'demo = "eval/a.json"',
                        "",
                        "[wandb]",
                        'project = ""',
                        'run_name = ""',
                        "",
                        "[runtime]",
                        "allow_tf32 = true",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            resolved = apply_overrides(config, type("Args", (), {
                "train_data": None,
                "output_dir": None,
                "run_name": None,
                "wandb_project": None,
                "resume_from_checkpoint": None,
            })(), config_path.parent)

            self.assertEqual(resolved["paths"]["train_data"], train_data.resolve())
            self.assertEqual(resolved["eval"]["corpora"]["demo"], (eval_dir / "a.json").resolve())
            self.assertEqual(resolved["paths"]["local_log_dir"], (root / "outputs" / "inline_eval").resolve())

    def test_default_run_name_is_descriptive(self):
        config = {
            "model": {"model_name": "unsloth/Qwen2.5-0.5B-Instruct"},
            "training": {"per_device_train_batch_size": 6, "learning_rate": 1e-5},
        }
        run_name = build_default_run_name(config)
        self.assertIn("qwen2.5-0.5b-instruct", run_name)
        self.assertIn("-bs6-", run_name)
        self.assertIn("-lr1e-05-", run_name)

    def test_session_state_reuses_prior_run_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_session_state(output_dir, {"run_name": "saved-run", "wandb_run_id": "abc123"})
            session = load_session_state(output_dir)
            config = {
                "model": {"model_name": "unsloth/Qwen2.5-0.5B-Instruct"},
                "training": {"per_device_train_batch_size": 6, "learning_rate": 1e-5},
                "wandb": {"run_name": ""},
            }
            self.assertEqual(session["run_name"], "saved-run")
            self.assertEqual(effective_run_name(config, session), "saved-run")


if __name__ == "__main__":
    unittest.main()
