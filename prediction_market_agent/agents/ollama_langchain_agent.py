from langchain_community.llms.ollama import Ollama

from prediction_market_agent.agents.langchain_agent import LangChainAgent


class OllamaLangChainAgent(LangChainAgent):
    def __init__(self) -> None:
        # Make sure Ollama is running locally
        llm = Ollama(model='mistral', base_url='http://localhost:11434') # Mistral since it supports function calling
        super().__init__(llm=llm)
