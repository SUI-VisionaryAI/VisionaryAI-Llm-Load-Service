from fastapi import FastAPI, HTTPException
from app.db import models
from app.model_pool import model_pool
from bson.objectid import ObjectId

app = FastAPI()

@app.get("/models/loaded")
def get_loaded_models():
    return {"loaded": model_pool.get_loaded_models()}

@app.post("/models/load/{model_id}")
def load_model(model_id: str):
    doc = models.find_one({"_id": ObjectId(model_id)})
    if not doc or not doc.get("versions"):
        raise HTTPException(status_code=404, detail="Model not found")

    latest = doc["versions"][-1]
    path = latest["filePath"]

    result = model_pool.load_model(model_id, path)
    return {"status": result}

@app.post("/models/unload/{model_id}")
def unload_model(model_id: str):
    if model_pool.unload_model(model_id):
        return {"status": "unloaded"}
    raise HTTPException(status_code=404, detail="Model not loaded")

@app.get("/models/status/{model_id}")
def get_model_status(model_id: str):
    return {"status": model_pool.get_status(model_id)}

@app.post("/chat/{model_id}")
def chat(model_id: str, prompt: str):
    reply = model_pool.chat(model_id, prompt)
    if reply is None:
        raise HTTPException(status_code=400, detail="Model not ready")
    return {"response": reply}
