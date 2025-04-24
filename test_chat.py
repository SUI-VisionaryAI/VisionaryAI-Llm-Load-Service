from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def chat(prompt, history=[]):
    model_path = "models/680a603777d8155cd4a7ec9b_680a613877d8155cd4a7ec9e"

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded to device: {next(model.parameters()).device}")

    # Prepare input prompt
    messages = history + [{"role": "user", "content": prompt}]
    prompt_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt_text += f"<|user|>{content}<|end|>"
        elif role == "assistant":
            prompt_text += f"<|assistant|>{content}<|end|>"

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    print(f"[INFO] Input tensor device: {input_ids.device}")

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return output_text.strip()


if __name__ == "__main__":
    user_input = "What is the capital of France?"
    reply = chat(user_input)
    print("Assistant:", reply)
