from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_community.llms import OpenAI

from prediction_market_agent import utils
from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.data_models.market_data_models import AgentMarket


class LangChainAgent(AbstractAgent):
    def __init__(self) -> None:
        keys = utils.get_keys()
        llm = OpenAI(openai_api_key=keys.openai)
        # Can use pre-defined search tool
        # TODO: Tavily tool could give better results
        # https://docs.tavily.com/docs/tavily-api/langchain
        tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=keys.serp)
        self._agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )

    def answer_binary_market(self, market: AgentMarket) -> bool:
        objective = utils.get_market_prompt(market.question)
        result_str = self._agent.run(objective)
        return utils.parse_result_to_boolean(result_str)
