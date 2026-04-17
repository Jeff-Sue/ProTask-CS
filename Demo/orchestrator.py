import uuid
from typing import Dict, List, Optional


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, List[Dict]] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        return session_id

    def get_history(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, [])

    def append_turn(self, session_id: str, user_input: str, strategy: str, response: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({
            "role": "user",
            "content": user_input,
        })
        self.sessions[session_id].append({
            "role": "assistant",
            "strategy": strategy,
            "content": response,
        })


class RagService:
    def retrieve(self, user_input: str, history: List[Dict]) -> str:
        # 这里只是 mock，后面替换成你的真正检索
        return f"模拟检索结果：与“{user_input}”相关的知识库内容如下……"


class SubAgentService:
    def run(self, user_input: str, history: List[Dict]) -> str:
        # 这里只是 mock，后面替换成你的真正子agent执行
        return f"模拟子agent执行：已经为请求“{user_input}”生成执行计划。"


class TwoStageAgent:
    def __init__(
        self,
        strategy_model,
        response_model,
        session_manager: SessionManager,
        rag_service: Optional[RagService] = None,
        subagent_service: Optional[SubAgentService] = None,
    ):
        self.strategy_model = strategy_model
        self.response_model = response_model
        self.session_manager = session_manager
        self.rag_service = rag_service or RagService()
        self.subagent_service = subagent_service or SubAgentService()

    def _truncate_history(self, history: List[Dict], max_turns: int = 10) -> List[Dict]:
        # 简单截断：保留最后 max_turns*2 条消息
        return history[-max_turns * 2:]

    def chat(self, session_id: str, user_input: str) -> Dict:
        history = self.session_manager.get_history(session_id)
        history = self._truncate_history(history)

        # Step 1: 策略生成
        strategy = self.strategy_model.predict(history, user_input)

        # Step 2: 根据策略执行额外动作
        extra_context = {}

        if strategy == "调RAG":
            extra_context["rag_result"] = self.rag_service.retrieve(user_input, history)

        elif strategy == "调子agent":
            extra_context["agent_result"] = self.subagent_service.run(user_input, history)

        # Step 3: 生成回复
        response = self.response_model.generate(
            history=history,
            user_input=user_input,
            strategy=strategy,
            extra_context=extra_context,
        )

        # Step 4: 更新历史
        self.session_manager.append_turn(session_id, user_input, strategy, response)

        return {
            "session_id": session_id,
            "strategy": strategy,
            "response": response,
            "extra_context": extra_context,
            "history": self.session_manager.get_history(session_id),
        }