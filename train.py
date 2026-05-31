"""Single training entrypoint with config-driven inline generation evaluation."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG_PATH = Path("train_config.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen SFT with inline generation evaluation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to TOML config.")
    parser.add_argument("--train-data", type=Path, default=None, help="Optional override for paths.train_data.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for paths.output_dir.")
    parser.add_argument("--run-name", default=None, help="Optional override for wandb.run_name.")
    parser.add_argument("--wandb-project", default=None, help="Optional override for wandb.project.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Optional override for paths.resume_from_checkpoint. Use 'auto' to detect latest checkpoint.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    return config


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def sanitize_run_component(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9._-]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-") or "run"


def format_learning_rate(value: float) -> str:
    text = f"{value:.0e}" if value < 0.001 else f"{value:g}"
    return text.replace("+", "").replace(".", "p")


def short_model_name(model_name: str) -> str:
    return sanitize_run_component(model_name.split("/")[-1])


def build_default_run_name(config: Dict[str, Any]) -> str:
    model_name = config["model"]["model_name"]
    train_bs = config["training"]["per_device_train_batch_size"]
    learning_rate = float(config["training"]["learning_rate"])
    random_suffix = random.randint(1000, 9999)
    return (
        f"{short_model_name(model_name)}"
        f"-bs{train_bs}"
        f"-lr{format_learning_rate(learning_rate)}"
        f"-r{random_suffix}"
    )


def configured_run_name(config: Dict[str, Any]) -> str:
    run_name = str(config["wandb"].get("run_name", "")).strip()
    if run_name:
        return run_name
    return build_default_run_name(config)


def ensure_run_state_dir(output_dir: Path) -> Path:
    run_state_dir = output_dir / "run_state"
    run_state_dir.mkdir(parents=True, exist_ok=True)
    return run_state_dir


def load_session_state(output_dir: Path) -> Dict[str, Any]:
    session_path = ensure_run_state_dir(output_dir) / "session.json"
    if not session_path.exists():
        return {}
    with session_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_session_state(output_dir: Path, state: Dict[str, Any]) -> None:
    session_path = ensure_run_state_dir(output_dir) / "session.json"
    with session_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def persist_resolved_config(output_dir: Path, config: Dict[str, Any]) -> None:
    config_path = ensure_run_state_dir(output_dir) / "resolved_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(config), handle, indent=2, ensure_ascii=False)


def effective_run_name(config: Dict[str, Any], session_state: Dict[str, Any]) -> str:
    if str(session_state.get("run_name", "")).strip():
        return str(session_state["run_name"]).strip()
    return configured_run_name(config)


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace, config_dir: Path) -> Dict[str, Any]:
    updated = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in config.items()
    }
    for section in ["paths", "model", "lora", "training", "eval", "wandb", "runtime"]:
        updated.setdefault(section, {})

    if args.train_data is not None:
        updated["paths"]["train_data"] = str(args.train_data)
    if args.output_dir is not None:
        updated["paths"]["output_dir"] = str(args.output_dir)
    if args.run_name is not None:
        updated["wandb"]["run_name"] = args.run_name
    if args.wandb_project is not None:
        updated["wandb"]["project"] = args.wandb_project
    if args.resume_from_checkpoint is not None:
        updated["paths"]["resume_from_checkpoint"] = args.resume_from_checkpoint

    train_data_raw = str(updated["paths"].get("train_data", "")).strip()
    if not train_data_raw:
        raise ValueError(
            "paths.train_data is empty in the config. Set it in train_config.toml or pass --train-data."
        )

    updated["paths"]["train_data"] = resolve_path(config_dir, train_data_raw)
    updated["paths"]["output_dir"] = resolve_path(
        config_dir,
        str(updated["paths"].get("output_dir", "outputs")),
    )

    local_log_dir_raw = str(updated["paths"].get("local_log_dir", "")).strip()
    if local_log_dir_raw:
        updated["paths"]["local_log_dir"] = resolve_path(config_dir, local_log_dir_raw)
    else:
        updated["paths"]["local_log_dir"] = updated["paths"]["output_dir"] / "inline_eval"

    resume_raw = str(updated["paths"].get("resume_from_checkpoint", "")).strip()
    if resume_raw and resume_raw.lower() != "auto":
        updated["paths"]["resume_from_checkpoint"] = str(resolve_path(config_dir, resume_raw))
    else:
        updated["paths"]["resume_from_checkpoint"] = resume_raw

    eval_corpora = updated["eval"].get("corpora", {})
    if not isinstance(eval_corpora, dict) or not eval_corpora:
        raise ValueError("eval.corpora must be a non-empty table in the config.")
    updated["eval"]["corpora"] = {
        str(name): resolve_path(config_dir, str(path))
        for name, path in eval_corpora.items()
    }
    return updated


def detect_last_checkpoint(output_dir: Path) -> Optional[str]:
    if not output_dir.exists():
        return None
    checkpoints = [path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: int(item.name.split("-")[-1]))
    return str(checkpoints[-1])


def resolve_resume_from_checkpoint(config: Dict[str, Any]) -> Optional[str]:
    raw_value = str(config["paths"].get("resume_from_checkpoint", "")).strip()
    if not raw_value:
        return None
    if raw_value.lower() == "auto":
        return detect_last_checkpoint(config["paths"]["output_dir"])
    return raw_value


def require_wandb_if_needed(project_name: Optional[str]) -> bool:
    if not project_name:
        return False
    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed in this environment. Run scripts/reinstall_env.sh first."
        ) from exc
    os.environ.setdefault("WANDB_PROJECT", project_name)
    return True


def configure_runtime(runtime_cfg: Dict[str, Any]) -> None:
    import torch

    if torch.cuda.is_available() and runtime_cfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config_dir = config_path.parent
    config = apply_overrides(load_config(config_path), args, config_dir)
    session_state = load_session_state(config["paths"]["output_dir"])

    if int(config["training"]["save_steps"]) != int(config["training"]["eval_steps"]):
        raise ValueError("training.save_steps must equal training.eval_steps.")

    from inline_eval import (
        InlineEvalCallback,
        InlineGenerationEvaluator,
        LocalArtifactLogger,
        build_text_dataset,
        load_corpus_specs,
        load_json_examples,
    )
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer
    import torch

    configure_runtime(config["runtime"])

    use_wandb = require_wandb_if_needed(str(config["wandb"].get("project", "")).strip() or None)
    run_name = effective_run_name(config, session_state)

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    eval_cfg = config["eval"]
    path_cfg = config["paths"]

    session_state["run_name"] = run_name
    session_state["output_dir"] = str(path_cfg["output_dir"])
    session_state["train_data"] = str(path_cfg["train_data"])
    session_state["wandb_project"] = str(config["wandb"].get("project", "")).strip()
    save_session_state(path_cfg["output_dir"], session_state)
    persist_resolved_config(path_cfg["output_dir"], config)

    if use_wandb:
        import wandb

        wandb_init_kwargs = {
            "project": str(config["wandb"]["project"]).strip(),
            "name": run_name,
        }
        prior_run_id = str(session_state.get("wandb_run_id", "")).strip()
        if prior_run_id:
            wandb_init_kwargs["id"] = prior_run_id
            wandb_init_kwargs["resume"] = "allow"
        run = wandb.init(**wandb_init_kwargs)
        if run is not None:
            session_state["wandb_run_id"] = run.id
            save_session_state(path_cfg["output_dir"], session_state)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=int(model_cfg["max_seq_length"]),
        dtype=None,
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg["rank"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=str(lora_cfg.get("bias", "none")),
        use_gradient_checkpointing=str(lora_cfg.get("use_gradient_checkpointing", "unsloth")),
        random_state=int(train_cfg["seed"]),
        use_rslora=bool(lora_cfg.get("use_rslora", False)),
        loftq_config=None,
    )

    train_examples = load_json_examples(path_cfg["train_data"])
    train_dataset = build_text_dataset(tokenizer, train_examples)
    corpora = load_corpus_specs(tokenizer, eval_cfg["corpora"])
    eval_dataset = {name: spec.eval_dataset for name, spec in corpora.items()}

    artifact_logger = LocalArtifactLogger(path_cfg["local_log_dir"])
    generation_evaluator = InlineGenerationEvaluator(
        tokenizer=tokenizer,
        corpora=corpora,
        generation_batch_size=int(eval_cfg["gen_batch_size"]),
        max_new_tokens=int(eval_cfg["max_new_tokens"]),
    )

    training_args = SFTConfig(
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        warmup_steps=int(train_cfg["warmup_steps"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_steps=int(train_cfg["max_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(train_cfg["eval_steps"]),
        save_strategy="steps",
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        optim=str(train_cfg.get("optim", "adamw_8bit")),
        weight_decay=float(train_cfg["weight_decay"]),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "linear")),
        seed=int(train_cfg["seed"]),
        output_dir=str(path_cfg["output_dir"]),
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=int(model_cfg["max_seq_length"]),
        dataset_num_proc=int(train_cfg["dataset_num_proc"]),
        packing=bool(train_cfg["packing"]),
        args=training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.add_callback(
        InlineEvalCallback(
            corpora=corpora,
            generation_evaluator=generation_evaluator,
            artifact_logger=artifact_logger,
            use_wandb=use_wandb,
        )
    )

    resume_from_checkpoint = resolve_resume_from_checkpoint(config)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = path_cfg["output_dir"] / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
