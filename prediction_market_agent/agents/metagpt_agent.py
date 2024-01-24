import asyncio
import dotenv
import json
import os

from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.data_models.market_data_models import MarketProtocol


class MetaGPTAgent(AbstractAgent):
    def __init__(self, cheap=True):
        dotenv.load_dotenv()
        os.environ["OPENAI_API_MODEL"] = "gpt-4-1106-preview"
        os.environ["SERPAPI_API_KEY"] = os.getenv("SERP_API_KEY")
        try:
            from metagpt.roles import Searcher
            from metagpt.roles.researcher import Researcher
        except ImportError:
            raise RuntimeError("You need to install the `metagpt` package manually.")

        if cheap:
            self._agent = Searcher()
        else:
            # Gives better results but is very expensive (~$0.3 / run!!)
            self._agent = Researcher()

    def answer_binary_market(self, market: MarketProtocol) -> bool:
        async def main(objective: str):
            await self._agent.run(objective)

        objective = (
            f"Research and report on the following question:\n\n"
            f"{market.question}\n\n"
            f"Search and scrape the web for information that will help you give a high quality, nuanced answer to the question.\n\n"
            f"Return your answer in raw JSON format, with no special formatting such as newlines, as follows:\n\n"
            f'{{"prediction": <PREDICTION>, "reasoning": <REASONING>}}\n\n'
            f'where <PREDICTION> is a boolean string (either "True" or "False"), and <REASONING> is a free text field that contains a well though out justification for the predicition based on the summary of your findings.\n\n'
        )
        asyncio.run(main(objective=objective))
        result = json.loads(self._agent.rc.history[-1].content)
        assert "prediction" in result
        return result["prediction"]
