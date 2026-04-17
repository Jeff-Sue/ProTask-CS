from fastapi import FastAPI
from pydantic import BaseModel

from models import QwenLoRAStrategyModel, QwenLoRAResponseModel
from orchestrator import SessionManager, TwoStageAgent, RagService, SubAgentService

app = FastAPI(title="Two-Stage Agent Demo")

session_manager = SessionManager()
strategy_model = QwenLoRAStrategyModel(
    model_name_or_path="你的Qwen3.5-4B底座路径",
    adapter_path="你的策略LoRA路径",
    use_fp16=True,
    max_new_tokens=16,
    do_sample=False,
)

response_model = QwenLoRAResponseModel(
    model_name_or_path="你的Qwen3.5-4B底座路径",
    adapter_path="你的回复LoRA路径",
    use_fp16=True,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
rag_service = RagService()
subagent_service = SubAgentService()

agent = TwoStageAgent(
    strategy_model=strategy_model,
    response_model=response_model,
    session_manager=session_manager,
    rag_service=rag_service,
    subagent_service=subagent_service,
)


class CreateSessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    user_input: str


@app.get("/")
def root():
    return {"message": "Two-Stage Agent backend is running."}


@app.post("/create_session", response_model=CreateSessionResponse)
def create_session():
    session_id = session_manager.create_session()
    return {"session_id": session_id}


@app.post("/chat")
def chat(req: ChatRequest):
    result = agent.chat(req.session_id, req.user_input)
    return result


@app.get("/history/{session_id}")
def get_history(session_id: str):
    return {
        "session_id": session_id,
        "history": session_manager.get_history(session_id)
    }