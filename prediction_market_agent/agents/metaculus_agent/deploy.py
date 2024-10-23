import sys
from typing import Sequence

from prediction_market_agent_tooling.deploy.agent import DeployablePredictionAgent
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import (
    AgentMarket,
    FilterBy,
    SortBy,
)
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.metaculus.metaculus import (
    MetaculusAgentMarket,
)

from prediction_market_agent.agents.prophet_agent.deploy import (
    DeployablePredictionProphetGPTo1PreviewAgent,
)

WARMUP_TOURNAMENT_ID = 3294
TOURNAMENT_ID = 3349


class DeployableMetaculusBotTournamentAgent(DeployablePredictionAgent):
    bet_on_n_markets_per_run: int = (
        sys.maxsize
    )  # On Metaculus "betting" is free, we can just bet on everything available in one run.
    dummy_prediction: bool = False
    repeat_predictions: bool = False
    tournament_id: int = TOURNAMENT_ID
    supported_markets = [MarketType.METACULUS]

    def load(self) -> None:
        # Using this one because it had the lowest `p_yes mse` from the `match_bets_with_langfuse_traces.py` evaluation at the time of writing this.
        self.agent = DeployablePredictionProphetGPTo1PreviewAgent(
            enable_langfuse=self.enable_langfuse
        )

    def get_markets(self, market_type: MarketType) -> Sequence[AgentMarket]:  # type: ignore # TODO: Needs to be decided in https://github.com/gnosis/prediction-market-agent/pull/511#discussion_r1810034688 and then I'll implement it here.
        markets: Sequence[
            MetaculusAgentMarket
        ] = MetaculusAgentMarket.get_binary_markets(
            limit=self.bet_on_n_markets_per_run,
            tournament_id=self.tournament_id,
            filter_by=FilterBy.OPEN,
            sort_by=SortBy.NEWEST,
        )
        return markets

    def verify_market(self, market_type: MarketType, market: AgentMarket) -> bool:
        assert isinstance(
            market, MetaculusAgentMarket
        ), "Just making mypy happy. It's true thanks to the check in the `run` method via `supported_markets`."

        # Filter out the market if the agent isn't configured to re-bet.
        if not self.repeat_predictions and market.have_predicted:
            return False

        # Otherwise all markets on Metaculus are fine.
        return True

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        assert isinstance(
            market, MetaculusAgentMarket
        ), "Just making mypy happy. It's true thanks to the check in the `run` method via `supported_markets`."
        logger.info(f"Answering market {market.id}, question: {market.question}")
        answer: ProbabilisticAnswer | None
        if not self.dummy_prediction:
            full_question = f"""Question: {market.question}
Question's description: {market.description}
Question's fine print: {market.fine_print} 
Question's resolution criteria: {market.resolution_criteria}"""
            answer = self.agent.agent.predict(full_question).outcome_prediction
        else:
            answer = ProbabilisticAnswer(
                p_yes=Probability(0.5),
                reasoning="Just a test.",
                confidence=0.5,
            )
        return answer
