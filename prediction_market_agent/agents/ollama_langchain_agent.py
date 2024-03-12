from langchain_community.llms.ollama import Ollama

from prediction_market_agent.agents.langchain_agent import LangChainAgent
from prediction_market_agent.tools.ollama_utils import is_ollama_running


class OllamaLangChainAgent(LangChainAgent):
    def __init__(self) -> None:
        # Make sure Ollama is running locally
        if not is_ollama_running():
            raise EnvironmentError(
                "Ollama is not running, cannot instantiate Ollama agent"
            )
        llm = Ollama(
            model="mistral", base_url="http://localhost:11434"
        )  # Mistral since it supports function calling
        super().__init__(llm=llm)
