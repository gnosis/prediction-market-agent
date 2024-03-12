from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.llms import OpenAI
from prediction_market_agent_tooling.markets.agent_market import AgentMarket

from prediction_market_agent import utils
from prediction_market_agent.agents.abstract import AbstractAgent


class LangChainAgent(AbstractAgent):
    def __init__(self, llm=None) -> None:
        keys = utils.APIKeys()
        llm = OpenAI(openai_api_key=keys.openai_api_key) if not llm else llm
        # Can use pre-defined search tool
        # TODO: Tavily tool could give better results
        # https://docs.tavily.com/docs/tavily-api/langchain
        tools = load_tools(
            ["serpapi", "llm-math"], llm=llm, serpapi_api_key=keys.serp_api_key
        )
        self._agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )

    def answer_binary_market(self, market: AgentMarket) -> bool:
        objective = utils.get_market_prompt(market.question)
        result_str = self._agent.run(objective)
        return utils.parse_result_to_boolean(result_str)
