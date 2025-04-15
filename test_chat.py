from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def chat(prompt, history=[]):
    model_path = "models/tinyllama"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.eval()

    # Prepare the input
    messages = history + [{"role": "user", "content": prompt}]
    prompt_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt_text += f"<|user|>{content}<|end|>"
        elif role == "assistant":
            prompt_text += f"<|assistant|>{content}<|end|>"

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids

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
