import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


NOTEBOOK = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "qwen2.5-sft-distillation",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.9",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK["cells"] = [
    markdown_cell(
        """
        # Cached Teacher Distillation

        This notebook keeps your original SFT flow, but adds an offline teacher pass first.

        Workflow:
        1. Load and split the JSON dataset.
        2. Call a larger model through any OpenAI-compatible API and cache the top-10 token distributions.
        3. Inspect the cached teacher outputs in a readable table.
        4. Train the 0.5B student with a mixed hard-label and soft-label loss.

        Best fit: use another Qwen 2.5 teacher model so the returned token-level top-10 candidates line up cleanly with the student tokenizer.
        """
    ),
    code_cell(
        """
        # !pip install unsloth wandb requests pandas

        import os
        import sys
        from pathlib import Path

        import pandas as pd
        import torch
        import wandb
        from IPython.display import display
        from transformers import TrainingArguments
        from unsloth import FastLanguageModel

        MODULE_DIR = Path.cwd() / "experiments" / "cached_teacher_distill"
        if not (MODULE_DIR / "distill_utils.py").exists():
            MODULE_DIR = Path.cwd()
        if str(MODULE_DIR) not in sys.path:
            sys.path.insert(0, str(MODULE_DIR))

        from distill_utils import (
            CachedTeacherDistillationTrainer,
            DistillationDataCollator,
            TeacherAPIConfig,
            build_distillation_dataset,
            build_hard_dataset,
            ensure_dir,
            load_json,
            prefetch_teacher_cache,
            split_raw_data,
            teacher_preview_rows,
        )
        """
    ),
    code_cell(
        """
        EXPERIMENT_ROOT = MODULE_DIR
        CACHE_DIR = ensure_dir(EXPERIMENT_ROOT / "cache")
        OUTPUT_DIR = ensure_dir(EXPERIMENT_ROOT / "outputs")
        EXPORT_DIR = ensure_dir(EXPERIMENT_ROOT / "exports")

        REPO_ROOT = MODULE_DIR.parent.parent
        DATA_PATH = REPO_ROOT / "data.json"
        EVAL_NUM = 20
        RANDOM_SEED = 3407

        STUDENT_MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
        TEACHER_MODEL_NAME = os.getenv("TEACHER_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

        TEACHER_TOP_LOGPROBS = 10
        TEACHER_MAX_TOKENS = 512
        TEACHER_TOKEN_LIMIT_FIELD = os.getenv("TEACHER_TOKEN_LIMIT_FIELD", "max_tokens")
        TEACHER_TIMEOUT_S = 120.0
        TEACHER_FORCE_REFRESH = False
        TEACHER_SLEEP_BETWEEN_REQUESTS_S = 0.0
        TEACHER_RESPONSE_FORMAT_JSON = False

        MAX_SEQ_LENGTH = 6048
        RANK = 32
        LORA_ALPHA = 32
        LORA_DROPOUT = 0.0
        TRAIN_BATCH = 2
        EVAL_BATCH = 2
        ACCUMULATION_STEPS = 4
        NUM_EPOCHS = 2
        LEARNING_RATE = 1e-5
        WEIGHT_DECAY = 0.01
        SOFT_LABEL_WEIGHT = 0.3
        DISTILL_TEMPERATURE = 1.0
        TOP_K = 10

        wandb_project = "qwen-cached-teacher-distill"
        run_name = f"qwen-distill-soft{SOFT_LABEL_WEIGHT}"

        CACHE_PATH = CACHE_DIR / "teacher_top10_train.jsonl"
        """
    ),
    code_cell(
        """
        raw_data = load_json(DATA_PATH)
        train_data, val_data = split_raw_data(raw_data, eval_num=EVAL_NUM, seed=RANDOM_SEED)

        print(f"Total examples: {len(raw_data)}")
        print(f"Training examples: {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")
        """
    ),
    code_cell(
        """
        teacher_api_config = TeacherAPIConfig(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            model_name=TEACHER_MODEL_NAME,
            top_logprobs=TEACHER_TOP_LOGPROBS,
            max_tokens=TEACHER_MAX_TOKENS,
            token_limit_field=TEACHER_TOKEN_LIMIT_FIELD,
            temperature=0.0,
            timeout_s=TEACHER_TIMEOUT_S,
            response_format_json=TEACHER_RESPONSE_FORMAT_JSON,
            sleep_between_requests_s=TEACHER_SLEEP_BETWEEN_REQUESTS_S,
        )

        train_teacher_records = prefetch_teacher_cache(
            train_data,
            cache_path=CACHE_PATH,
            api_config=teacher_api_config,
            overwrite=TEACHER_FORCE_REFRESH,
        )

        print(f"Cached teacher rows: {len(train_teacher_records)}")
        print(f"Cache file: {CACHE_PATH}")
        """
    ),
    code_cell(
        """
        preview_record = train_teacher_records[0]
        preview_df = pd.DataFrame(teacher_preview_rows(preview_record, max_rows=25))
        display(preview_df)

        print("Teacher output preview:")
        print(preview_record["teacher_output"])
        """
    ),
    code_cell(
        """
        dtype = None
        load_in_4bit = False

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=STUDENT_MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=RANK,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=RANDOM_SEED,
            use_rslora=False,
            loftq_config=None,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        """
    ),
    code_cell(
        """
        train_dataset = build_distillation_dataset(
            train_teacher_records,
            tokenizer=tokenizer,
            max_seq_length=MAX_SEQ_LENGTH,
            top_k=TOP_K,
        )
        val_dataset = build_hard_dataset(
            val_data,
            tokenizer=tokenizer,
            max_seq_length=MAX_SEQ_LENGTH,
        )

        data_collator = DistillationDataCollator(
            pad_token_id=tokenizer.pad_token_id,
            top_k=TOP_K,
        )

        print(train_dataset)
        print(val_dataset)
        """
    ),
    code_cell(
        """
        wandb.login()
        wandb.init(project=wandb_project, name=run_name)

        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=TRAIN_BATCH,
            per_device_eval_batch_size=EVAL_BATCH,
            gradient_accumulation_steps=ACCUMULATION_STEPS,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=5,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=20,
            save_strategy="steps",
            save_steps=20,
            save_total_limit=2,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=RANDOM_SEED,
            report_to="wandb",
            run_name=run_name,
            remove_unused_columns=False,
        )

        trainer = CachedTeacherDistillationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            soft_label_weight=SOFT_LABEL_WEIGHT,
            distill_temperature=DISTILL_TEMPERATURE,
        )
        """
    ),
    code_cell(
        """
        trainer_stats = trainer.train()
        trainer_stats
        """
    ),
    code_cell(
        """
        sample = val_data[0]
        messages = [
            {"role": "system", "content": sample["instruction"]},
            {"role": "user", "content": sample["input"]},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True)
        decoded_output = tokenizer.batch_decode(outputs)
        print(decoded_output[0])
        """
    ),
    code_cell(
        """
        model.save_pretrained_merged(
            str(EXPORT_DIR / "model_final"),
            tokenizer,
            save_method="merged_16bit",
        )
        """
    ),
]


def main() -> None:
    notebook_path = ROOT / "teacher_cached_distill.ipynb"
    notebook_path.write_text(json.dumps(NOTEBOOK, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
