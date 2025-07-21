from openai import OpenAI
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    MultiCategoricalMaxAccuracyBettingStrategy,
)
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic_ai.exceptions import UnexpectedModelBehavior

from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount


class Berlin2OpenaiSearchAgentVariable(DeployableTraderAgent):
    bet_on_n_markets_per_run = 2

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return MultiCategoricalMaxAccuracyBettingStrategy(
            max_position_amount=get_maximum_possible_bet_amount(
                min_=1,
                max_=25,
                trading_balance=market.get_trade_balance(self.api_keys),
            ),
        )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        client = OpenAI(api_key=self.api_keys.openai_api_key.get_secret_value())
        today = utcnow()

        # Ask search API for a report
        search_response = client.responses.create(
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

You will be given a question in the following user message. Your task is to extract, compile, and organize every piece of relevant information that could help a reasoning model assess the likelihood of the described event occurring. The final outcome should include:

- A comprehensive set of evidence without omitting any pertinent details (unless they are obviously irrelevant). Do not provide links, explain the evidence in it's entirety.
- No conclusions or judgments unless the evidence overwhelmingly points to one.
- A clear presentation of all evidence, which will later be used to derive a probability and confidence level for the event.

Focus exclusively on presenting the evidence and avoid speculative analysis.""",
                },
                {"role": "user", "content": f"{market.question}"},
            ],
        )

        # Comprehensize report from search LLM
        context = search_response.output_text

        # Ask reasoning model for a probability estimate given context from report
        reasoning_response = client.responses.create(
            model="o3-mini",
            input=[
                {
                    "role": "developer",
                    "content": f"""Today is {today}.

Given the following question and information from the web, what's the probability that the thing in the question will happen?.

Return only the probability float number and confidence float number, separated by space, nothing else. So it looks like:

float float""",
                },
                {
                    "role": "user",
                    "content": f"""Question: {market.question}

Context: {context}""",
                },
            ],
            reasoning={"effort": "high"},
        )

        probability_and_confidence = reasoning_response.output_text

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
