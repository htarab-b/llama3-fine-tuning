# !pip install -U transformers accelerate pytorch
# !pip install -U bitsandbytes
# !pip install unsloth

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load the fine-tuned model
model_path = "llama3-finetuned"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# Ensure model is on the correct device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_response(prompt):
    formatted_prompt = f"<|user|> {prompt} <|assistant|>"

    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    # Generate output
    output_ids = model.generate(
        **inputs, max_length=200, do_sample=True  # Remove temperature and top_p if not sampling
    )


    # Decode response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the assistant's reply
    response = response.split("<|assistant|>")[-1].strip()

    return response

# Example usage
prompt = "What is the fuel tank capacity of the Liebherr LTM 1130-5?"
response = generate_response(prompt)
print(response)