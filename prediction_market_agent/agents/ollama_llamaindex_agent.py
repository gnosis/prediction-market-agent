from llama_index.llms.ollama import Ollama

from prediction_market_agent.agents.llamaindex_agent import LlamaIndexAgent


class OllamaLlamaIndexAgent(LlamaIndexAgent):
    def __init__(self, ollama_model_name: str = "tinyllama") -> None:
        # Make sure ollama is running locally
        llm = Ollama(model=ollama_model_name, request_timeout=30.0)
        super().__init__(llm=llm)
