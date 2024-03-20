from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.tools import FunctionTool
from prediction_market_agent_tooling.markets.agent_market import AgentMarket

from prediction_market_agent import utils
from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.tools.web_scrape.basic_summary import web_scrape
from prediction_market_agent.tools.web_search.google import google_search


class LlamaIndexAgent(AbstractAgent):
    def __init__(self) -> None:
        google_search_tool = FunctionTool.from_defaults(fn=google_search)
        web_scraping_tool = FunctionTool.from_defaults(fn=web_scrape)
        llm = OpenAI(model="gpt-3.5-turbo-0613")
        self._agent = OpenAIAgent.from_tools(
            tools=[google_search_tool, web_scraping_tool],
            llm=llm,
            verbose=True,
            system_prompt="You are a researcher with tools to search and web scrape, in order to produce high quality, fact-based results for the research objective you've been given. Make sure you search for a variety of high quality sources, and that the results you produce are relevant to the objective you've been given. Do not try to scrape URLs that prevent scraping. Scrape from sufficient sources to produce a high quality result.",
        )

    def answer_binary_market(self, market: AgentMarket) -> bool:
        objective = utils.get_market_prompt(market.question)
        response = self._agent.chat(objective).response
        return utils.parse_result_to_boolean(response)
