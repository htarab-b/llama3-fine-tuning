# Install dependencies (for local system)
# !pip install -U bitsandbytes transformers accelerate datasets trl sentence-transformers peft torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
from trl import SFTTrainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files={"train": "train-1.jsonl"})

# Load tokenizer and model
model_name = "unsloth/llama-3-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["instruction"],  # Use only instruction (since input is empty)
        text_target=examples["response"], 
        padding="max_length",
        truncation=True,
        max_length=256,  # Adjusted for shorter responses
    )

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Prepare for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=4,  # Increased batch size for smaller model
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    bf16=torch.cuda.is_bf16_supported(),  # Enables bf16 if available
    report_to="none"
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_datasets["train"],
    args=training_args,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("llama3-finetuned")
tokenizer.save_pretrained("llama3-finetuned")