from fastapi import FastAPI, Response
from app.db import models
from bson.json_util import dumps

app = FastAPI()

@app.get("/models")
def list_models():
    model_docs = models.find({})
    return Response(dumps(model_docs), media_type="application/json")
