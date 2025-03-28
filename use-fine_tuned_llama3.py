from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load the fine-tuned model
model_path = "llama3-finetuned"

# Load with quantization
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# Ensure model is on the correct device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define response generation function
def generate_response(prompt):
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    # Tokenize input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # Generate output
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the assistant's reply
    response = response.replace(formatted_prompt, "").strip()

    return response

# Example usage
prompt = "What is the fuel tank capacity of this machine?"
response = generate_response(prompt)
print(response)