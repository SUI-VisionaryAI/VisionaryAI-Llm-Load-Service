from fastapi import FastAPI, HTTPException
from bson.objectid import ObjectId
from app.model_pool import model_pool
from app.db import models

app = FastAPI()

@app.get("/check-availability")
def check_slot():
    return {"available": model_pool.is_available()}

@app.post("/load/{model_id}")
def load_model(model_id: str):
    doc = models.find_one({"_id": ObjectId(model_id)})
    if not doc or not doc.get(\"versions\"):
        raise HTTPException(status_code=404, detail=\"Model not found\")

    version = doc[\"versions\"][-1]  # Latest
    file_path = version[\"filePath\"]

    result = model_pool.load_model(model_id, file_path)
    if result:
        return {\"status\": \"loaded\"}
    else:
        return {\"status\": \"queued\"}
