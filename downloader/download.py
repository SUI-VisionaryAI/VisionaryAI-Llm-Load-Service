from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
save_path = "./models/tinyllama"

# Download and save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_path)

# Download and save model
model = AutoModelForCausalLM.from_pretrained(model_id)
model.save_pretrained(save_path)
