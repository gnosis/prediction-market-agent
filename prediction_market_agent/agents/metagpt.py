import asyncio
import dotenv
import json
import os

dotenv.load_dotenv()
os.environ["OPENAI_API_MODEL"] = "gpt-4-1106-preview"
os.environ["SERPAPI_API_KEY"] = os.getenv("SERP_API_KEY")
from metagpt.roles import Searcher

from prediction_market_agent.agents.abstract import AbstractAgent


class MetaGPTAgent(AbstractAgent):
    def __init__(self):
        self._agent = Searcher()

    def run(self, market: str) -> bool:
        async def main(objective: str):
            await self._agent.run(objective)

        objective = (
            f"Research and report on the following question:\n\n"
            f"{market}\n\n"
            f"Search and scrape the web for information that will help you give a high quality, nuanced answer to the question.\n\n"
            f"Return your answer in raw JSON format, with no special formatting such as newlines, as follows:\n\n"
            f'{{"prediction": <PREDICTION>, "reasoning": <REASONING>}}\n\n'
            f'where <PREDICTION> is a boolean string (either "True" or "False"), and <REASONING> is a free text field that contains a well though out justification for the predicition based on the summary of your findings.\n\n'
        )
        asyncio.run(main(objective=objective))
        result = json.loads(self._agent.rc.history[-1].content)
        assert "prediction" in result
        return result["prediction"]
