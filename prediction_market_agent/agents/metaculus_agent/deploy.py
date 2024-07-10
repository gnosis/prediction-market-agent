import sys
from typing import Sequence

from prediction_market_agent_tooling.deploy.agent import Answer, DeployableAgent
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.metaculus.metaculus import (
    MetaculusAgentMarket,
)
from prediction_market_agent_tooling.tools.utils import check_not_none

from prediction_market_agent.agents.think_thoroughly_agent.think_thoroughly_agent import (
    CrewAIAgentSubquestions,
)

WARMUP_TOURNAMENT_ID = 3294
TOURNAMENT_ID = 3349


class DeployableMetaculusBotTournamentAgent(DeployableAgent):
    model: str = "gpt-4-turbo-2024-04-09"
    dummy_prediction: bool = False
    repeat_predictions: bool = False
    tournament_id: int = TOURNAMENT_ID

    def run(
        self,
        market_type: MarketType = MarketType.METACULUS,
    ) -> None:
        """
        Submit predictions to Metaculus markets using the CrewAIAgentSubquestions

        https://www.metaculus.com/notebooks/25525/announcing-the-ai-forecasting-benchmark-series--july-8-120k-in-prizes/
        """

        if market_type != MarketType.METACULUS:
            raise ValueError("Only Metaculus markets are supported for this agent")

        agent = CrewAIAgentSubquestions(model=self.model, memory=False)
        markets: Sequence[
            MetaculusAgentMarket
        ] = MetaculusAgentMarket.get_binary_markets(
            limit=sys.maxsize,
            tournament_id=self.tournament_id,
            filter_by=FilterBy.OPEN,
            sort_by=SortBy.NEWEST,
        )
        logger.info(f"Found {len(markets)} open markets to submit predictions for.")

        if not self.repeat_predictions:
            # Filter out markets that we have already answered
            markets = [market for market in markets if not market.have_predicted]
            logger.info(
                f"Found {len(markets)} unanswered markets to submit predictions for."
            )

        for market in markets[:1]:
            logger.info(f"Answering market {market.id}, question: {market.question}")
            if not self.dummy_prediction:
                # TODO incorporate 'Resolution criteria', 'Fine print', and
                # 'Background info' into the prompt given to the agent.
                answer = agent.answer_binary_market(
                    market.question, created_time=market.created_time
                )
            else:
                answer = Answer(
                    p_yes=Probability(0.5),
                    decision=True,
                    reasoning="Just a test.",
                    confidence=0.5,
                )

            if answer is None:
                logger.error("No answer was given. Skipping")
            else:
                market.submit_prediction(
                    p_yes=answer.p_yes,
                    reasoning=check_not_none(answer.reasoning),
                )
