# this is the exact same notebook extracted as a py file

# %%
# !pip install unsloth
# # Also get the latest nightly Unsloth!
# !pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
# !pip install wandb

# %%
import os
import json
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import torch
from trl import SFTTrainer, SFTConfig
import wandb
import random

# %%
eval_num = 20
data_path = "data.json"

eval_paths = {
    "eval_order": "eval/eval_order.json",
    "eval_vehicle": "eval/eval_vehicle.json",
    "eval_invoice": "eval/eval_invoice.json",
}

# %%
# train_split_ratio = 0.95
max_seq_length = 6048  # Choose any! We auto support RoPE Scaling internally!
rank = 32
lora_alpha = 32
lora_dropout = 0.0
train_batch = 6
eval_batch = 4
accumulation_size = 4
weight_decay_val = 0.01
learning_rate_value = 1e-5

# %% [markdown]
# # data loading

# %%
# Load the data
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Ensure we have enough data
total_examples = len(raw_data)
print(f"Total examples: {total_examples}")
# Shuffle the data
random.shuffle(raw_data)
# Split into train and validation
train_data = raw_data[:-eval_num]
val_data = raw_data[-eval_num:]


print(f"Training examples: {len(train_data)}")


print(f"Validation examples: {len(val_data)}")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# %% [markdown]
# # model loading

# %%
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = False  # False as requested for full precision/bf16 loading

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-0.5B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r=rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# %% [markdown]
# # data formating


# %%
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        # Must handle possible None values if any, though dataset seems clean
        if output_text is None:
            output_text = ""

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {
        "text": texts,
    }


eval_datasets = {}

for eval_name, file_path in eval_paths.items():
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to HF Dataset
        dset = Dataset.from_list(data)

        # Apply the SAME formatting function immediately
        # We need to do this here because we need a dict of formatted datasets
        dset = dset.map(formatting_prompts_func, batched=True)

        eval_datasets[eval_name] = dset
        print(f"Loaded {eval_name}: {len(data)} examples")
    else:
        print(f"Warning: {file_path} not found. Skipping.")

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

eval_datasets["from_train"] = val_dataset

# %%
# Verify the formatting
print(train_dataset[0]["text"])

# %%
wandb.login()

# %%
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_datasets,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,  # Can make training 5x faster for short sequences.
    args=SFTConfig(
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        gradient_accumulation_steps=accumulation_size,
        warmup_steps=5,
        num_train_epochs=2,  # Set this for 1 full training run.
        # max_steps = 60, # Set to None for full training
        learning_rate=learning_rate_value,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        eval_strategy="steps",  # Evaluate during training
        eval_steps=20,  # Evaluate every 10 steps
        save_strategy="steps",
        save_steps=20,
        optim="adamw_8bit",
        weight_decay=weight_decay_val,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="wandb",  # Use WandB for logging
        run_name=f"qwenft-{str(rank)}-{str(learning_rate_value)}",
    ),
)

# %%
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# %%
if wandb.run is None:
    wandb.init(
        project="your_project_name",
        name=f"qwenft-{str(rank)}-{str(learning_rate_value)}",
    )

# --- ADD THIS BLOCK ---
# Save the Run ID and Project Name to a file so the external script can read it
with open("current_wandb_run.txt", "w") as f:
    f.write(f"{wandb.run.id},{wandb.run.project}")
print(f"W&B Run ID saved: {wandb.run.id}")

# %%
trainer_stats = trainer.train()

# %% [markdown]
# # inference

# %%
# Select a sample from validation set
sample = val_data[0]
instruction = sample["instruction"]
input_text = sample["input"]

messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": input_text},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True)
decoded_output = tokenizer.batch_decode(outputs)
print(decoded_output[0])

# %% [markdown]
# # save

# %%
model.save_pretrained_merged(
    "model_final",
    tokenizer,
    save_method="merged_16bit",
)
