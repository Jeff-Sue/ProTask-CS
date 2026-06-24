class ModelManager:

    def __init__(self):
        self.models = {}

    def load_all(self):
        from models.sft_model import SFTModel

        print("Loading models...")

        self.models["query"] = SFTModel(
            base_path="/data/MODEL_ZOO/Qwen/Qwen3-8B",
            lora_path="/data/jiahui/data/LlamaFactory/saves/static_single/query_generator",
            device="cuda:0"
        )
        self.models["trigger"] = SFTModel(
            base_path="/data/MODEL_ZOO/Qwen/Qwen3-8B",
            lora_path="/data/jiahui/data/LlamaFactory/saves/static_single/rag_trigger",
            device="cuda:1"
        )
        self.models["policy"] = SFTModel(
            base_path="/data/MODEL_ZOO/Qwen/Qwen3-8B",
            lora_path="/data/jiahui/data/LlamaFactory/saves/static_single/policy_predictor",
            device="cuda:2"
        )
        self.models["response"] = SFTModel(
            base_path="/data/MODEL_ZOO/Qwen/Qwen3-8B",
            lora_path="/data/jiahui/data/LlamaFactory/saves/static_single/response_generator",
            device="cuda:3"
        )

        print("All models loaded ✔")

    def get(self, name):
        return self.models[name]