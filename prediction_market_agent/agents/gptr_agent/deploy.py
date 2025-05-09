import asyncio
import os

from gpt_researcher import GPTResearcher
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.tools.langfuse_ import observe
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

from prediction_market_agent.tools.openai_utils import get_openai_provider
from prediction_market_agent.tools.prediction_prophet.research import (
    prophet_make_prediction,
)
from prediction_market_agent.utils import APIKeys


class GPTRAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 1

    # TODO: Uncomment and configure after we get some historic bet data
    # def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
    #     return KellyBettingStrategy(
    #         max_bet_amount=get_maximum_possible_bet_amount(
    #             min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
    #         ),
    #         max_price_impact=0.7,
    #     )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        report = gptr_research_sync(market.question)
        prediction = prophet_make_prediction(
            market_question=market.question,
            additional_information=report,
            agent=Agent(
                OpenAIModel(
                    "gpt-4o",
                    provider=get_openai_provider(api_key=APIKeys().openai_api_key),
                ),
                model_settings=ModelSettings(temperature=0),
            ),
        )
        return prediction.outcome_prediction


@observe()
def gptr_research_sync(query: str) -> str:
    keys = APIKeys()
    # GPTR doesn't allow to pass keys via constructor.
    os.environ["OPENAI_API_KEY"] = keys.openai_api_key.get_secret_value()
    os.environ["TAVILY_API_KEY"] = keys.tavily_api_key.get_secret_value()
    # See https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/config/variables/default.py
    # for the default configuration that GPTR uses in the background.
    researcher = GPTResearcher(query=query, report_type="research_report")
    asyncio.run(researcher.conduct_research())
    report: str = asyncio.run(researcher.write_report())
    return report
