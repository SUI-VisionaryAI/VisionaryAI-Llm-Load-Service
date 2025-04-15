from collections import deque
from threading import Lock
from transformers import AutoTokenizer, AutoModelForCausalLM

MAX_SLOTS = 2

class ModelSlotManager:
    def __init__(self):
        self.slots = {}
        self.queue = deque()
        self.lock = Lock()

    def is_available(self):
        return len(self.slots) < MAX_SLOTS

    def is_loaded(self, model_id):
        return model_id in self.slots

    def load_model(self, model_id, file_path):
        with self.lock:
            if self.is_loaded(model_id):
                return self.slots[model_id]
            if not self.is_available():
                return None

            tokenizer = AutoTokenizer.from_pretrained(file_path)
            model = AutoModelForCausalLM.from_pretrained(file_path)
            model.eval()

            self.slots[model_id] = {"model": model, "tokenizer": tokenizer}
            return self.slots[model_id]

    def unload_model(self, model_id):
        with self.lock:
            if model_id in self.slots:
                del self.slots[model_id]

    def enqueue(self, user_id, model_id):
        self.queue.append({"user_id": user_id, "model_id": model_id})

    def next_in_queue(self):
        return self.queue.popleft() if self.queue else None

model_pool = ModelSlotManager()
