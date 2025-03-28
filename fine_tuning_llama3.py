from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
from trl import SFTTrainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files={"train": "train-1.jsonl"})

# Load model and tokenizer
model_name = "unsloth/llama-3-8b-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    inputs = [f"### Instruction:\n{prompt}\n\n### Response:\n" for prompt in examples["prompt"]]
    return tokenizer(
        inputs,
        text_target=examples["response"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load the 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# Enable training on 4-bit quantized model
model = prepare_model_for_kbit_training(model)

# Apply LoRA configuration
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
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_datasets["train"],
    args=training_args,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("llama3-finetuned")
tokenizer.save_pretrained("llama3-finetuned")