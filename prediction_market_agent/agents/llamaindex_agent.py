from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.tools import FunctionTool

from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.tools.google_search import google_search
from prediction_market_agent.tools.web_scrape import web_scrape
from prediction_market_agent import utils


class LlamaIndexAgent(AbstractAgent):
    def __init__(self):
        google_search_tool = FunctionTool.from_defaults(fn=google_search)
        web_scraping_tool = FunctionTool.from_defaults(fn=web_scrape)
        llm = OpenAI(model="gpt-3.5-turbo-0613")
        self._agent = OpenAIAgent.from_tools(
            tools=[google_search_tool, web_scraping_tool],
            llm=llm,
            verbose=True,
            system_prompt="You are a researcher with tools to search and web scrape, in order to produce high quality, fact-based results for the research objective you've been given. Make sure you search for a variety of high quality sources, and that the results you produce are relevant to the objective you've been given. Do not try to scrape URLs that prevent scraping. Scrape from sufficient sources to produce a high quality result.",
        )

    def answer_boolean_market(self, market: str) -> bool:
        objective = utils.get_market_prompt(market)
        response = self._agent.chat(objective).response
        return utils.parse_result_to_boolean(response)
