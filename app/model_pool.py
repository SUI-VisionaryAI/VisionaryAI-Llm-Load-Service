from collections import deque
from threading import Lock
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MAX_SLOTS = 2

class ModelPool:
    def __init__(self):
        self.loaded_models = {}  # { model_id: { 'model':..., 'tokenizer':... } }
        self.status = {}         # { model_id: 'idle' | 'loading' | 'ready' | 'queued' }
        self.queue = deque()
        self.lock = Lock()

    def get_status(self, model_id):
        return self.status.get(model_id, "idle")

    def get_loaded_models(self):
        return list(self.loaded_models.keys())

    def unload_model(self, model_id):
        with self.lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                self.status[model_id] = "idle"
                return True
            return False

    def can_load(self):
        return len(self.loaded_models) < MAX_SLOTS

    def load_model(self, model_id, path):
        with self.lock:
            if model_id in self.loaded_models:
                return "ready"

            if not self.can_load():
                if model_id not in self.queue:
                    self.queue.append(model_id)
                    self.status[model_id] = "queued"
                return "queued"

            self.status[model_id] = "loading"

        # Load outside the lock to avoid blocking
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        with self.lock:
            self.loaded_models[model_id] = { "model": model, "tokenizer": tokenizer }
            self.status[model_id] = "ready"

        return "ready"

    def chat(self, model_id, prompt):
        if self.status.get(model_id) != "ready":
            return None

        model_entry = self.loaded_models[model_id]
        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]

        device = model.device

        prompt_text = f"<|user|>{prompt}<|end|>"

        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

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

model_pool = ModelPool()
