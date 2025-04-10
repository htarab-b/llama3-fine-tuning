# Install dependencies (for local system)
# !pip install -U bitsandbytes transformers accelerate datasets trl sentence-transformers peft torch
# !pip install unsloth

from unsloth import FastLanguageModel, to_sharegpt, standardize_sharegpt, apply_chat_template, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Model config
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Load datasets
dataset = load_dataset("json", data_files="train.jsonl", split="train")
val_dataset = load_dataset("json", data_files="val.jsonl", split="train")

# Preprocess dataset
dataset = to_sharegpt(dataset, merged_prompt="{instruction}[[\nYour input is:\n{input}]]", output_column_name="output")
dataset = standardize_sharegpt(dataset)

chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

dataset = apply_chat_template(dataset, tokenizer=tokenizer, chat_template=chat_template)

# Training config
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

# Train
trainer_stats = trainer.train()

# Save model and tokenizer
model.save_pretrained("fine_tuned-llama3_model")
tokenizer.save_pretrained("fine_tuned-llama3_model")