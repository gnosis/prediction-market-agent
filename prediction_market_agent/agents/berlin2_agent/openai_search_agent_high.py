from openai import OpenAI
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    MultiCategoricalMaxAccuracyBettingStrategy,
)
from prediction_market_agent_tooling.gtypes import USD, Probability
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic_ai.exceptions import UnexpectedModelBehavior

from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount


class Berlin2OpenaiSearchAgentHigh(DeployableTraderAgent):
    bet_on_n_markets_per_run = 2

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return MultiCategoricalMaxAccuracyBettingStrategy(
            max_position_amount=get_maximum_possible_bet_amount(
                min_=USD(1),
                max_=USD(5),
                trading_balance=market.get_trade_balance(self.api_keys),
            ),
        )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        client = OpenAI(api_key=self.api_keys.openai_api_key.get_secret_value())

        today = utcnow()

        # Ask search API for a probability estimate
        response = client.responses.create(
            model="gpt-4o",
            tools=[
                {
                    "type": "web_search_preview",
                    "search_context_size": "high",
                }
            ],
            input=[
                {
                    "role": "developer",
                    "content": f"""Today is {today}.

Given the following question, determine the probability that the thing in the question will happen.

Return ONLY the probability float number and confidence float number, separated by space, nothing else. So it will look like:

float float

NEVER give any other type of response.""",
                },
                {"role": "user", "content": f"{market.question}"},
            ],
        )

        probability_and_confidence = str(response.output_text)

        try:
            probability, confidence = map(float, probability_and_confidence.split())
        except Exception as e:
            raise UnexpectedModelBehavior(
                f"Could not parse {probability_and_confidence}"
            ) from e

        return ProbabilisticAnswer(
            confidence=confidence,
            p_yes=Probability(probability),
        )
