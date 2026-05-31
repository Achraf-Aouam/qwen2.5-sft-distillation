"""KD training entrypoint — mirrors ``train.py`` but feeds ``KDTrainer``.

Data is read from a pre-materialized Arrow dataset produced by
``KL_div_prep/build_kd_dataset.py``. Generation eval on order/vehicle/invoice still fires
through ``InlineEvalCallback`` at every save step; token-level text eval is
disabled because the KD collator expects teacher top-K tensors that the eval
corpora don't carry.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG_PATH = Path("train_config_kd.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "KD training with top-K soft labels. Every knob lives in the TOML config — "
            "duplicate the config file (one per experiment) and pass --config."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to TOML config.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Optional override for paths.resume_from_checkpoint. "
            "Use 'auto' to pick up the latest checkpoint in output_dir."
        ),
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


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


def random_suffix(length: int = 3) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=length))


def build_default_run_name(config: Dict[str, Any]) -> str:
    model_name = config["model"]["model_name"]
    train_bs = config["training"]["per_device_train_batch_size"]
    learning_rate = float(config["training"]["learning_rate"])
    alpha = float(config["kd"]["alpha"])
    temperature = float(config["kd"].get("temperature", 1.0))
    return (
        f"kd-{short_model_name(model_name)}"
        f"-a{int(round(alpha * 100))}"
        f"-t{format_learning_rate(temperature)}"
        f"-bs{train_bs}"
        f"-lr{format_learning_rate(learning_rate)}"
        f"-{random_suffix()}"
    )


def configured_run_name(config: Dict[str, Any]) -> str:
    """Build a fresh run name. Always includes a random suffix so parallel/repeat
    runs that share a config don't collide in wandb or on disk."""

    run_name = str(config["wandb"].get("run_name", "")).strip()
    if run_name:
        return f"{run_name}-{random_suffix()}"
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


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace, base_dir: Path) -> Dict[str, Any]:
    updated = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in config.items()
    }
    for section in ["paths", "model", "lora", "training", "eval", "wandb", "runtime", "kd"]:
        updated.setdefault(section, {})

    if args.resume_from_checkpoint is not None:
        updated["paths"]["resume_from_checkpoint"] = args.resume_from_checkpoint

    kd_dataset_raw = str(updated["kd"].get("dataset_dir", "")).strip()
    if not kd_dataset_raw:
        raise ValueError("kd.dataset_dir is empty. Set it in train_config_kd.toml or pass --kd-dataset.")
    updated["kd"]["dataset_dir"] = resolve_path(base_dir, kd_dataset_raw)

    updated["paths"]["output_dir"] = resolve_path(
        base_dir,
        str(updated["paths"].get("output_dir", "outputs_kd")),
    )

    local_log_dir_raw = str(updated["paths"].get("local_log_dir", "")).strip()
    if local_log_dir_raw:
        updated["paths"]["local_log_dir"] = resolve_path(base_dir, local_log_dir_raw)
    else:
        updated["paths"]["local_log_dir"] = updated["paths"]["output_dir"] / "inline_eval"

    resume_raw = str(updated["paths"].get("resume_from_checkpoint", "")).strip()
    if resume_raw and resume_raw.lower() != "auto":
        updated["paths"]["resume_from_checkpoint"] = str(resolve_path(base_dir, resume_raw))
    else:
        updated["paths"]["resume_from_checkpoint"] = resume_raw

    eval_corpora = updated["eval"].get("corpora", {})
    if not isinstance(eval_corpora, dict) or not eval_corpora:
        raise ValueError("eval.corpora must be a non-empty table in the config.")
    updated["eval"]["corpora"] = {
        str(name): resolve_path(base_dir, str(path))
        for name, path in eval_corpora.items()
    }

    updated["kd"].setdefault("alpha", 0.4)
    updated["kd"].setdefault("temperature", 1.0)
    return updated


def detect_last_checkpoint(output_dir: Path) -> Optional[str]:
    if not output_dir.exists():
        return None
    checkpoints = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
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
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed in this environment.") from exc

    # Fail fast with a clear message if the host isn't logged in.
    if not os.environ.get("WANDB_API_KEY"):
        try:
            from netrc import netrc as _netrc

            hosts = _netrc().hosts
            logged_in = any("wandb" in host for host in hosts)
        except Exception:
            logged_in = False
        if not logged_in:
            raise RuntimeError(
                "wandb is configured but this host is not logged in. "
                "Run `wandb login` (or export WANDB_API_KEY=...) before starting training. "
                "If you want to skip wandb entirely, blank out [wandb].project in the config."
            )

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
    # Relative paths in the config (dataset_dir, output_dir, eval corpora, ...) are
    # resolved against the current working directory so `python train_kd.py --config
    # configs/kd/foo.toml` from the project root behaves the way a user expects.
    base_dir = Path.cwd()
    config = apply_overrides(load_config(config_path), args, base_dir)
    session_state = load_session_state(config["paths"]["output_dir"])

    from datasets import load_from_disk
    from inline_eval import (
        InlineEvalCallback,
        InlineGenerationEvaluator,
        LocalArtifactLogger,
        load_corpus_specs,
    )
    from unsloth import FastLanguageModel
    from trl import SFTConfig
    import torch

    from kd_collator import KDCollator
    from kd_trainer import KDTrainer

    configure_runtime(config["runtime"])

    use_wandb = require_wandb_if_needed(str(config["wandb"].get("project", "")).strip() or None)
    run_name = effective_run_name(config, session_state)

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    eval_cfg = config["eval"]
    path_cfg = config["paths"]
    kd_cfg = config["kd"]

    session_state["run_name"] = run_name
    session_state["output_dir"] = str(path_cfg["output_dir"])
    session_state["kd_dataset"] = str(kd_cfg["dataset_dir"])
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
    tokenizer.padding_side = "right"

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

    kd_dataset_dir = Path(kd_cfg["dataset_dir"])
    if not kd_dataset_dir.exists():
        raise FileNotFoundError(
            f"KD dataset not found at {kd_dataset_dir}. "
            f"Run KL_div_prep/build_kd_dataset.py first."
        )
    train_dataset = load_from_disk(str(kd_dataset_dir))

    # Unsloth silently truncates input_ids to max_seq_length but the KD metadata
    # (assistant_start / gold_ids / topk_ids) is still indexed against the original
    # sequence — drop any row that would exceed max_seq_length so the gather stays aligned.
    max_seq_length = int(model_cfg["max_seq_length"])
    n_before = len(train_dataset)
    train_dataset = train_dataset.filter(
        lambda ex: len(ex["input_ids"]) <= max_seq_length,
        num_proc=1,
    )
    n_after = len(train_dataset)
    if n_after < n_before:
        print(
            f"[KD] filtered {n_before - n_after}/{n_before} rows exceeding "
            f"max_seq_length={max_seq_length}."
        )

    # Sanity: the student's chat template fingerprint must still match the one recorded during build.
    meta_path = kd_dataset_dir / "kd_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            kd_meta = json.load(handle)
        import hashlib

        current_fp = hashlib.sha256(
            (getattr(tokenizer, "chat_template", None) or "").encode("utf-8")
        ).hexdigest()
        stored_fp = kd_meta.get("student_chat_template_fingerprint")
        if stored_fp and current_fp != stored_fp:
            raise RuntimeError(
                "Student chat template fingerprint has drifted since KD dataset build. "
                f"current={current_fp} stored={stored_fp}. Rebuild the KD dataset."
            )

    corpora = load_corpus_specs(tokenizer, eval_cfg["corpora"])
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
        eval_strategy="no",
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
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    collator = KDCollator(pad_token_id=int(tokenizer.pad_token_id))

    trainer = KDTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=collator,
        args=training_args,
        kd_alpha=float(kd_cfg["alpha"]),
        kd_temperature=float(kd_cfg["temperature"]),
    )

    class KDInlineEvalCallback(InlineEvalCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # KD run has no text eval_dataset, so skip the strategy-matching check.
            return control

    trainer.add_callback(
        KDInlineEvalCallback(
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
