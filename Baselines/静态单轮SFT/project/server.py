from fastapi import FastAPI
from contextlib import asynccontextmanager
from models.model_manager import ModelManager
from pipeline import Pipeline
model_manager = ModelManager()
pipeline = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时自动加载
    global pipeline
    print(">>> Loading models on startup...")
    model_manager.load_all()
    pipeline = Pipeline(model_manager)
    print(">>> All models loaded ✔")
    yield
    # 关闭时的清理工作（如果需要）
    print(">>> Shutting down...")
app = FastAPI(title="Huawei IT SFT System", lifespan=lifespan)
# 移除 /load_models 接口，或者保留作为手动重新加载
@app.post("/load_models")
def load_models():
    global pipeline
    print(">>> Reloading models...")
    model_manager.load_all()
    pipeline = Pipeline(model_manager)
    print(">>> All modelsed ✔")
    return {"status": "models loaded"}
@app.post("/chat")
def chat(req: dict):
    global pipeline
    if pipeline is None:
        return {"error": "models not loaded. call /load_models first"}
    dialogue = req["dialogue"]
    result = pipeline.run(dialogue)
    return result
@app.get("/health")
def health():
    return {"status": "ok"}


## CUDA_VISIBLE_DEVICES=0,1,2,3 uvicorn server:app --host 0.0.0.0 --port 8338