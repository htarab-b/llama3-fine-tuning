from unsloth import FastLanguageModel
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch

# Load fine-tuned model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "fine_tuned-llama3_model",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Prompt
messages = [
    {"role": "user", "content": "When is it necessary to ground a crane?"},
]

# Tokenize
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

# Generate
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids,
    streamer=text_streamer,
    max_new_tokens=128,
    pad_token_id=tokenizer.eos_token_id,
)