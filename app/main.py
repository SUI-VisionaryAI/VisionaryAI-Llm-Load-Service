# ---------- GPU SERVER ----------
# file: gpu_server.py

from fastapi import FastAPI, HTTPException, Request
import os
import requests
import shutil
from pathlib import Path
import threading
from dotenv import load_dotenv
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
load_dotenv()
app = FastAPI()

MODEL_CACHE_DIR = Path("models")
MODEL_CACHE_DIR.mkdir(exist_ok=True)

BE_API = os.getenv("BE_API_ENDPOINT", "http://localhost:8000/api")

# Track loaded models if you want to simulate memory load
loaded_models = {}  # e.g., key: f"{model_id}_{version_id}"

# ---------------------- Utilities ----------------------

def load_model_data(filename='data.json'):
    if not os.path.exists(filename):
        return {"current_models": []}
    with open(filename, 'r') as f:
        return json.load(f)

def save_model_data(data, filename='data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def check_model_exists(data, model_id, version_id):
    return any(
        m['model_id'] == model_id and m['version_id'] == version_id
        for m in data['current_models']
    )

def add_model(data, model_id, version_id):
    if not check_model_exists(data, model_id, version_id):
        print("[GPU] Adding model to tracking JSON")
        data['current_models'].append({
            "model_id": model_id,
            "version_id": version_id
        })

def remove_model(data, model_id, version_id):
    data['current_models'] = [
        m for m in data['current_models']
        if not (m['model_id'] == model_id and m['version_id'] == version_id)
    ]

def flatten_extracted_folder(folder_path: Path):
    items = list(folder_path.iterdir())
    if len(items) == 1 and items[0].is_dir():
        inner_folder = items[0]
        print(f"[GPU] Flattening inner folder: {inner_folder.name}")
        for item in inner_folder.iterdir():
            shutil.move(str(item), folder_path / item.name)
        inner_folder.rmdir()
        print(f"[GPU] Flattened model folder: {folder_path}")

def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Endpoints ----------------------

@app.get("/")
async def root():
    return {"message": "GPU Server is running"}

def load_model_thread(model_id: str, version_id: str):
    try:
        print(f"[GPU] Start loading model {model_id} version {version_id}")

        model_data = load_model_data()
        if check_model_exists(model_data, model_id, version_id):
            print(f"[GPU] Model {model_id} v{version_id} already loaded.")
            return

        extract_path = MODEL_CACHE_DIR / f"{model_id}_{version_id}"
        if extract_path.exists():
            print(f"[GPU] Model {model_id} v{version_id} already downloaded at {extract_path}")
        else:
            model_dir = MODEL_CACHE_DIR / model_id
            model_dir.mkdir(exist_ok=True)

            download_url = f"{BE_API}/models/{model_id}/download-file"
            print(f"[GPU] Downloading model {model_id} from {download_url}")

            response = requests.get(download_url, stream=True)
            if response.status_code != 200:
                print(f"[GPU] Failed to download model {model_id}")
                return

            model_path = MODEL_CACHE_DIR / f"{model_id}.zip"
            with open(model_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)
            print(f"[GPU] Model {model_id} downloaded to {model_path}")

            extract_path.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(model_path, extract_path)
            print(f"[GPU] Model {model_id} extracted to {extract_path}")

            # Flatten nested model folders if needed
            contents = list(extract_path.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                inner_dir = contents[0]
                for item in inner_dir.iterdir():
                    shutil.move(str(item), str(extract_path))
                inner_dir.rmdir()
                print(f"[GPU] Flattened nested model directory: {inner_dir.name}")

            os.remove(model_path)
            print(f"[GPU] Model {model_id} zip file removed")

        # Load model to memory
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GPU] Loading model on device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(extract_path)
        model = AutoModelForCausalLM.from_pretrained(
            extract_path, 
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        )
        model.to(device)
        model.eval()

        # Save to memory pool
        loaded_models[f"{model_id}_{version_id}"] = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device
        }

        add_model(model_data, model_id, version_id)
        save_model_data(model_data)
        print(f"[GPU] Model {model_id} v{version_id} marked as loaded in data.json")

    except Exception as e:
        print(f"[GPU] Error loading model {model_id}: {e}")


@app.post("/load-model")
async def load_model(req: Request):
    data = await req.json()
    model_id = data.get("modelId")
    version_id = data.get("versionId")
    print(f"[GPU] Request to load model {model_id} version {version_id}")
    if not model_id or not version_id:
        raise HTTPException(status_code=400, detail="modelId and versionId are required")

    model_data = load_model_data()
    if check_model_exists(model_data, model_id, version_id):
        print(f"[GPU] Model {model_id} v{version_id} already loaded.")
        return {"status": "already_loaded"}

    threading.Thread(target=load_model_thread, args=(model_id, version_id)).start()
    return {"status": "loading"}

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    model_id = data.get("model_id")
    version_id = data.get("version_id")
    prompt = data.get("prompt")
    history = data.get("history", [])

    key = f"{model_id}_{version_id}"
    if key not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not loaded")

    model_entry = loaded_models[key]
    model = model_entry["model"]
    tokenizer = model_entry["tokenizer"]
    device = model_entry["device"]

    print(f"[GPU] Running chat on model {key} using device: {device}")
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
    print(f"[GPU] Input tensor device: {input_ids.device}")

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
    return {"reply": output_text.strip()}
